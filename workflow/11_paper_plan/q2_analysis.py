#!/usr/bin/env python3
"""Q2 (D-B115): base of offshore seismicity against isotherms (H_T) and velocity levels (H_C), with the rules
pre-registered in data/paper_plan/q2_depth_limit/RESUME.md (2026-11-27 13:27:22 PST, commit e6bab1ac).

Configurations (one run each, same code, same rules):
  main      v6.1, tiers A-B, sig_z <= 3 km, 0.5 deg bins (U2), D90, P mask        (decisions)
  U1        model columns                                                        (variant)
  D95       95th percentile                                                      (variant)
  sigz2     sig_z <= 2 km                                                        (C1)
  nosig     tiers A-B, no sig_z cut                                              (C1)
  abc       tiers A-C, no sig_z cut (plate-boundary set)                         (variant)
  v60       catalog v6.0 with its own segment table                              (C2)
  geom      nearest station within one focal depth (dist_min <= bsf)             (C3)
  cb        levels on the checkerboard-only mask                                 (C4)
  notomo    events not in any v0.4.2 tomoDD dataset                              (C5a)
Writes <Q2_OUT>/units_<cfg>.csv, groups_<cfg>.csv, pref_<cfg>.csv, answers_<cfg>.csv and <Q2_BIG>/draws_<cfg>.npz.

usage: q2_analysis.py [cfg ...]   (default: all)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402
import q2_common as Q  # noqa: E402

CONFIGS = {
    "main": dict(), "U1": dict(scale="col"), "D95": dict(q=95), "sigz2": dict(sigz=2.0),
    "nosig": dict(sigz=None), "abc": dict(tiers=("A", "B", "C"), sigz=None), "v60": dict(catalog="v6.0"),
    "geom": dict(subset="geom"), "cb": dict(mask="cb"), "notomo": dict(subset="notomo"),
    # post-registration control L (RESUME.md, rule 2026-11-27 13:52:46 PST): literature depth offsets, Lange depths
    "lit_med": dict(lit="med"), "lit_p90": dict(lit="p90"), "lit_gd": dict(lit="gd"),
    "lange": dict(subset="lange", lange="ref"), "lange_ours": dict(subset="lange", lange="ours"),
}
REGISTERED = ["main", "U1", "D95", "sigz2", "nosig", "abc", "v60", "geom", "cb", "notomo"]
PAIRED = Q.WD1 / "catalog_v6/v6.1_analysis/validation/depth_literature/paired_v6.1.csv"
LANGE_PAIRS = Q.WD1 / "tomo_v1/depth_literature/pairs/blanco_lange2026_abs_pairs.csv"
BL_SEGS = [(-131.0, -129.3, "West (WBD/EBD)"), (-129.3, -128.75, "Surveyor"),
           (-128.75, -128.45, "Cascadia Depression"), (-128.45, -127.1, "Blanco Ridge"),
           (-127.1, -126.0, "Gorda Depression")]


def lit_offsets(E, which):
    """Per-event depth offset (ours minus Lange, tiers A-B, absolute catalog) of control L1."""
    t = pd.read_csv(PAIRED)
    t = t[t.reference.str.contains("Lange") & t.reference.str.contains("absolute") & t.catalog.eq("v6.1")
          & t.tiers.eq("A-B")].set_index("group")
    col = "p90" if which == "p90" else "med"
    off = np.full(len(E), float(t.loc["all", col]))
    bl = E.seg.eq("BL").values
    for a, b, name in BL_SEGS:
        m = bl & (E.lon.values >= a) & (E.lon.values < b)
        off[m] = float(t.loc[name, col])
    if which == "gd":
        off[(E.lat.values < 42.5) & ~bl] = float(t.loc["Gorda Depression", "med"])
    return off
ISO_KEYS = [f"{m}_{int(T)}" for m in ("HSC", "GDH1") for T in Q.ISOTHERMS]
ISO_EXTRA = ["HSC_600_basement", "HSC_600_young", "HSC_600_old"]
# labelled variants from the D-B114 external-data job (v6.1 only; added after the registration, never primary):
#   EXT_HSC1300_600: its half-space 600 C isotherm (Tm 1300 C, kappa 1e-6, age at the epicentre floored at 0.1 Myr), bsl
#   HSC_600_warmtop: slab events, the registered incoming geotherm with the van Keken & Wilson (2023) D80 slab-surface
#                    temperature at the event's Slab2 depth as its top: T = Ts + (Tm - Ts) erf(z'/a)
#   CRUST1_moho: CRUST1.0 crustal thickness below the solid surface, converted to bsl (an independent Moho proxy)
EXT_VARIANTS = ["EXT_HSC1300_600", "HSC_600_warmtop", "CRUST1_moho"]
EXT_EVENTS = Q.WD1 / "external_geo/derived/events_external_geo_v61.parquet"
_CATALOG = ["v6.1"]
LEV_KEYS = ["V7.5", "V8.0"]
PRIMARY_T, PRIMARY_C = "HSC_600", "V7.5"
AGE_IN_DLAT = 0.2           # deg north of the triple junction for the incoming age of slab events south of it


# ------------------------------------------------------------------ static inputs (cached per process)
_CACHE = {}


def statics():
    if _CACHE:
        return _CACHE
    m = P.Model()
    age = Q.Grid(Q.AGE_GRID)
    sed = Q.Grid(Q.SED_GRID)
    pts, sub, tj = P.pb2002()
    zl = np.load(Q.WD1 / "paper_plan/v6.1/level_depth_members.npz")     # raw members, P mask (pb_structure.py)
    lev = {"P": {7.5: m.z8(7.5), 8.0: m.z8(8.0)}}
    mem = {"P": {7.5: zl["z75"].astype(float), 8.0: zl["z8"].astype(float)}}
    _CACHE.update(m=m, age=age, sed=sed, pts=pts, sub=sub, tj=tj, lev=lev, mem=mem, dip=Q.slab_dip_fn(),
                  slab=P.slab_depth_fn())
    return _CACHE


def cb_levels():
    """Levels on the checkerboard-only mask (C4): product and the 100 raw members."""
    S = statics()
    if "cb" in S["lev"]:
        return
    m = S["m"]
    names, mv = P.load_members()
    vp = mv["Vp"]
    S["lev"]["cb"] = {L: P.level_depth(m.Vp, m.cbP, m.rock, m.dep, L) for L in Q.LEVELS}
    S["mem"]["cb"] = {L: np.stack([P.level_depth(vp[i], m.cbP, m.rock, m.dep, L) for i in range(len(names))])
                      for L in Q.LEVELS}


def inflate(zmem, zprod):
    """Registered: deviation of each raw member level from the member median x 1.50, added to the product level."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(zmem, axis=0)
    return zprod[None] + Q.VINFL * (zmem - med[None])


# ------------------------------------------------------------------ predictors per event
def predictors(E: pd.DataFrame) -> pd.DataFrame:
    S = statics()
    m, age, sed = S["m"], S["age"], S["sed"]
    E = E.copy()
    lon, lat = E.lon.values, E.lat.values
    E["age"] = age(lon, lat)
    E["sed_km"] = sed(lon, lat) / 1000.0
    tl = P.trench_lon(lat, S["sub"])
    west = lon < tl
    sl = S["slab"](lon, lat)
    E["slab_km"] = sl
    E["dip"] = S["dip"](lon, lat)
    E["frame"] = np.where(west, "oceanic", np.where(np.isfinite(sl) & (E.depth.values >= sl - 5.0), "slab", "upper"))
    # incoming-plate age for slab events: grid age 10-30 km west of the trench at the event latitude. South of the
    # Gorda plate (latitudes below the triple junction) the latitude is set to the junction latitude + 0.2 deg: the
    # age grid smears the Mendocino age step over ~0.1 deg (12.8-13 Myr mixed Gorda/Pacific ages at +0.1 deg, 4.9 Myr
    # Gorda ages from +0.15 deg north; code test 2026-11-27 13:34 PST, RESUME.md).
    tjlat = S["tj"][1]
    la_in = np.round(np.maximum(lat, tjlat + AGE_IN_DLAT), 2)
    t_in = {}
    for la in np.unique(la_in[E.frame.values == "slab"]):
        x0 = P.trench_lon(np.array([la]), S["sub"])[0]
        d = np.array([10, 15, 20, 25, 30.0])
        lo = x0 - d / (P.KM * np.cos(np.radians(la)))
        t_in[la] = np.nanmedian(age(lo, np.full(len(d), la)))
    E["age_in"] = [t_in.get(a, np.nan) if f == "slab" else np.nan for a, f in zip(la_in, E.frame.values)]
    # transform ages on both sides (all oceanic events near a transform or in a transform segment)
    near = west & ((E.d_otf.values <= P.D_AXIS_KM) | E.seg.isin(Q.TRANSFORMS).values)
    E["age_young"], E["age_old"], E["d_trace"] = np.nan, np.nan, np.nan
    if near.any():
        y, o, _, _, dtr = Q.transform_side_ages(age, S["pts"]["OTF"], lon[near], lat[near])
        E.loc[near, "age_young"], E.loc[near, "age_old"], E.loc[near, "d_trace"] = y, o, dtr
    # thermal type per grouping (registered): segments -> transform for BL, MFZ, SOV; frames -> within 15 km of a trace
    seg_tr = E.seg.isin(Q.TRANSFORMS).values & west
    frm_tr = west & (E.d_otf.values <= P.D_AXIS_KM)
    E["ttype_seg"] = np.where(E.frame.values == "slab", "slab", np.where(seg_tr, "transform",
                              np.where(west, "oceanic", "none")))
    E["ttype_frame"] = np.where(E.frame.values == "slab", "slab", np.where(frm_tr, "transform",
                                np.where(west, "oceanic", "none")))
    for g in ("seg", "frame"):
        tt = E[f"ttype_{g}"].values
        for key in ISO_KEYS + ISO_EXTRA:
            E[f"{key}_{g}"] = iso_event(E, tt, key)
    ext_variants(E)
    # velocity levels at the event's column (bsl)
    iy, ix = m.column_index(lon, lat)
    E["iy"], E["ix"] = iy, ix
    for mk in S["lev"]:
        for L in Q.LEVELS:
            E[f"V{L}_{mk}"] = S["lev"][mk][L][iy, ix]
    E["bsf"] = E.depth + E.surface_km
    return E


def ext_variants(E):
    """Labelled D-B114 variants (v6.1 events only, joined by ID; NaN for another catalog)."""
    for g in ("seg", "frame"):
        for k in EXT_VARIANTS:
            E[f"{k}_{g}"] = np.nan
    E["Tslab_D80"] = np.nan
    if _CATALOG[0] != "v6.1" or not EXT_EVENTS.exists():
        return
    x = pd.read_parquet(EXT_EVENTS, columns=["ID", "iso600_hsc_bsl", "Tslab_D80", "crust1_thk_km"])
    x = x.set_index("ID")
    ids = E.ID.values
    iso = x.iso600_hsc_bsl.reindex(ids).values
    ts = x.Tslab_D80.reindex(ids).values
    moho = x.crust1_thk_km.reindex(ids).values - E.surface_km.values
    E["Tslab_D80"] = ts
    p = Q.THERMAL["HSC"]
    for g in ("seg", "frame"):
        tt = E[f"ttype_{g}"].values
        oc = np.isin(tt, ["oceanic", "transform"])
        E.loc[oc, f"EXT_HSC1300_600_{g}"] = iso[oc]
        sb = (tt == "slab") & np.isfinite(ts)
        a = Q.a_of(E.age_in.values[sb], p["kappa"])
        th = (600.0 - ts[sb]) / (p["Tm"] - ts[sb])
        z = np.where(th <= 0, 0.0, Q.isotherm_depth(np.clip(th, 1e-6, 0.999), a))
        E.loc[sb, f"HSC_600_warmtop_{g}"] = E.slab_km.values[sb] + z / np.cos(np.radians(np.nan_to_num(E.dip.values[sb])))
        E[f"CRUST1_moho_{g}"] = np.where(np.isin(tt, ["oceanic", "transform", "slab"]), moho, np.nan)


def z_below_top(model, T, age1, age2=None, kfac=1.0, Tm=None, afac=1.0):
    p = Q.THERMAL[model]
    Tm = p["Tm"] if Tm is None else Tm
    a1 = Q.a_of(np.asarray(age1, float) * afac, p["kappa"] * kfac)
    a2 = None if age2 is None else Q.a_of(np.asarray(age2, float) * afac, p["kappa"] * kfac)
    return Q.isotherm_depth(T / Tm, a1, a2, p["L"])


def iso_event(E, tt, key):
    """Isotherm depth bsl per event for predictor key (model_T[_variant]); NaN where the thermal type is none."""
    parts = key.split("_")
    model, T = parts[0], float(parts[1])
    var = parts[2] if len(parts) > 2 else ""
    out = np.full(len(E), np.nan)
    srf = E.surface_km.values
    oc = tt == "oceanic"
    out[oc] = z_below_top(model, T, E.age.values[oc]) - srf[oc]
    tr = tt == "transform"
    if tr.any():
        y, o = E.age_young.values[tr], E.age_old.values[tr]
        if var == "young":
            z = z_below_top(model, T, y)
        elif var == "old":
            z = z_below_top(model, T, o)
        else:
            z = z_below_top(model, T, y, o)
        out[tr] = z - srf[tr]
    if var == "basement":
        out[oc | tr] += np.nan_to_num(E.sed_km.values[oc | tr])
    sb = tt == "slab"
    if sb.any():
        z = z_below_top(model, T, E.age_in.values[sb]) / np.cos(np.radians(np.nan_to_num(E.dip.values[sb])))
        out[sb] = E.slab_km.values[sb] + z
    return out


# ------------------------------------------------------------------ units
def unit_table(E, grouping, scale):
    """Units: (segment or frame) x bin. Returns list of dicts with event index arrays."""
    if scale == "bin":
        bx, by = np.floor((E.lon.values + 132.0) / 0.5).astype(int), np.floor((E.lat.values - 38.0) / 0.5).astype(int)
        cen = lambda i, j: (-132.0 + 0.5 * i + 0.25, 38.0 + 0.5 * j + 0.25)      # noqa: E731
        blk = 1.0
    else:
        bx, by = E.ix.values, E.iy.values
        m = statics()["m"]
        cen = lambda i, j: (m.lon[i], m.lat[j])                                    # noqa: E731
        blk = 0.5
    key = E.seg.values if grouping == "seg" else E.frame.values
    df = pd.DataFrame({"k": key, "bx": bx, "by": by, "i": np.arange(len(E))})
    units = []
    for (k, i, j), g in df.groupby(["k", "bx", "by"]):
        if len(g) < 20:
            continue
        lo, la = cen(i, j)
        units.append(dict(grouping=grouping, key=k, bx=i, by=j, lon=lo, lat=la, idx=g.i.values,
                          block=f"{int(np.floor(lo / blk))}_{int(np.floor(la / blk))}"))
    return units


def unit_predictor(vals, minfrac=0.5):
    v = vals[np.isfinite(vals)]
    if len(vals) == 0 or len(v) < minfrac * len(vals):
        return np.nan
    return float(np.median(v))


# ------------------------------------------------------------------ draws
def run_config(cfg_name, E_all=None, meta=None, seed=None):
    cfg = dict(catalog="v6.1", tiers=("A", "B"), sigz=3.0, scale="bin", q=90, mask="P", subset=None, lit=None,
               lange=None)
    cfg.update(CONFIGS[cfg_name])
    # registered seed per configuration; `seed` only for the Monte Carlo precision study (q2_seeds.py)
    rng = np.random.default_rng(Q.SEED + list(CONFIGS).index(cfg_name) if seed is None else seed)
    if E_all is None:
        c, meta = Q.load_events(cfg["catalog"])
        E_all = Q.select(c, cfg["tiers"], cfg["sigz"])
    E = E_all
    if cfg["subset"] == "geom":
        E = E[E.dist_min <= (E.depth + E.surface_km)]
    elif cfg["subset"] == "notomo":
        E = E[~E.in_tomo]
    elif cfg["subset"] == "lange":
        lp = pd.read_csv(LANGE_PAIRS, usecols=["ID", "ref_depth", "ref_ez"])
        E = E[E.seg.eq("BL")].merge(lp, on="ID", how="inner")
    if cfg["mask"] == "cb":
        cb_levels()
    _CATALOG[0] = cfg["catalog"]
    E = predictors(E).reset_index(drop=True)
    if cfg["lit"]:                       # control L1: only the depths entering D90 move (segments, frames fixed)
        E["depth_offset"] = lit_offsets(E, cfg["lit"])
        E["depth"] = E.depth - E.depth_offset
    if cfg["lange"] == "ref":            # control L2: Lange's depths and errors for the same events
        E["depth"] = E.ref_depth.astype(float)
        E["sig_z"] = E.ref_ez.astype(float)
    S = statics()
    mk = cfg["mask"]
    zmem = {L: inflate(S["mem"][mk][L], S["lev"][mk][L]) for L in Q.LEVELS}
    Q.log(f"[{cfg_name}] events {len(E)}")
    units = []
    off = E.seg.isin(list(Q.OFFSHORE)).values
    for grouping in ("seg", "frame"):
        if grouping == "seg":
            Eg = E[off]
        else:
            Eg = E[E.frame.isin(["oceanic", "slab"])]
        for u in unit_table(Eg, grouping, cfg["scale"]):
            u["idx"] = Eg.index.values[u["idx"]]
            units.append(u)
    K = Q.NDRAW
    nU = len(units)
    D = np.full((nU, K), np.nan)
    Dsf = np.full((nU, K), np.nan)
    preds = {k: np.full((nU, K), np.nan) for k in ISO_KEYS + ISO_EXTRA + EXT_VARIANTS + LEV_KEYS}
    rows = []
    # systematic thermal draws
    kfac = rng.uniform(0.8, 1.2, K)
    Tm_d = {mod: rng.uniform(p["Tm_lo"], p["Tm_hi"], K) for mod, p in Q.THERMAL.items()}
    jm = rng.integers(0, zmem[7.5].shape[0], K)
    dep, sz, srf = E.depth.values, E.sig_z.values, E.surface_km.values
    for ui, u in enumerate(units):
        ix = u["idx"]
        n = len(ix)
        d0, s0, f0 = dep[ix], sz[ix], srf[ix]
        bi = rng.integers(0, n, (K, n))
        noise = rng.standard_normal((K, n))
        dd = d0[bi] + s0[bi] * noise
        D[ui] = np.percentile(dd, cfg["q"], axis=1)
        Dsf[ui] = np.percentile(dd + f0[bi], cfg["q"], axis=1)
        g = u["grouping"]
        e = E.iloc[ix]
        r = dict(grouping=g, key=u["key"], bx=u["bx"], by=u["by"], lon=u["lon"], lat=u["lat"], block=u["block"],
                 n=n, n_in_tomo=int(e.in_tomo.sum()),
                 D=float(np.percentile(d0, cfg["q"])), D_bsf=float(np.percentile(d0 + f0, cfg["q"])),
                 D_lo68=float(np.percentile(D[ui], 16)), D_hi68=float(np.percentile(D[ui], 84)),
                 D_lo95=float(np.percentile(D[ui], 2.5)), D_hi95=float(np.percentile(D[ui], 97.5)),
                 Dbsf_lo95=float(np.percentile(Dsf[ui], 2.5)), Dbsf_hi95=float(np.percentile(Dsf[ui], 97.5)),
                 D50=float(np.median(d0)), surface_km=float(np.median(f0)),
                 sig_z_med=float(np.median(s0)), dist_min_med=float(np.nanmedian(e.dist_min)),
                 age=float(np.nanmedian(e.age)) if e.age.notna().any() else np.nan,
                 sed_km=float(np.nanmedian(e.sed_km)) if e.sed_km.notna().any() else np.nan)
        tt = e[f"ttype_{g}"].value_counts()
        ttype = tt.index[0] if len(tt) else "none"
        r["ttype"] = ttype
        r["seg_major"] = e.seg.value_counts().index[0]
        r["age_young"] = float(np.nanmedian(e.age_young)) if e.age_young.notna().any() else np.nan
        r["age_old"] = float(np.nanmedian(e.age_old)) if e.age_old.notna().any() else np.nan
        r["age_in"] = float(np.nanmedian(e.age_in)) if e.age_in.notna().any() else np.nan
        r["slab_km"] = float(np.nanmedian(e.slab_km)) if e.slab_km.notna().any() else np.nan
        r["dip"] = float(np.nanmedian(e.dip)) if e.dip.notna().any() else np.nan
        uage = unit_age_inputs(r)
        afac = np.exp(0.15 * rng.standard_normal(K))
        for key in ISO_KEYS + ISO_EXTRA:
            zc = unit_predictor(e[f"{key}_{g}"].values)
            r[key] = zc
            if not np.isfinite(zc):
                continue
            parts = key.split("_")
            mod, T = parts[0], float(parts[1])
            var = parts[2] if len(parts) > 2 else ""
            nom = iso_unit(mod, T, uage, var, 1.0, Q.THERMAL[mod]["Tm"], 1.0)
            drw = iso_unit(mod, T, uage, var, kfac, Tm_d[mod], afac)
            preds[key][ui] = zc + (drw - nom)
        for key in EXT_VARIANTS:              # labelled variants: central value only (no predictor draws)
            zc = unit_predictor(e[f"{key}_{g}"].values)
            r[key] = zc
            preds[key][ui] = zc
        r["Tslab_D80"] = float(np.nanmedian(e.Tslab_D80)) if e.Tslab_D80.notna().any() else np.nan
        for L, lk in zip(Q.LEVELS, LEV_KEYS):
            zc = unit_predictor(e[f"V{L}_{mk}"].values)
            r[lk] = zc
            if not np.isfinite(zc):
                continue
            mvals = zmem[L][:, e.iy.values, e.ix.values]          # [member, event]
            per_mem = np.array([unit_predictor(mv) for mv in mvals])
            preds[lk][ui] = per_mem[jm]
            r[f"{lk}_sd"] = float(np.nanstd(per_mem))
        # temperature and velocity at the base (descriptive)
        r.update(base_temperatures(r, D[ui]))
        r.update(base_velocity(e, r["D"], D[ui], mk))
        rows.append(r)
    U = pd.DataFrame(rows)
    return cfg, E, U, D, Dsf, preds, meta


def unit_age_inputs(r):
    return dict(ttype=r["ttype"], age=r["age"], young=r["age_young"], old=r["age_old"], age_in=r["age_in"],
                dip=r["dip"])


def iso_unit(mod, T, u, var, kfac, Tm, afac):
    """Isotherm depth below the thermal top for the unit's median inputs (used for the draw deltas only)."""
    t = u["ttype"]
    if t == "oceanic":
        return z_below_top(mod, T, u["age"], None, kfac, Tm, afac)
    if t == "transform":
        if var == "young":
            return z_below_top(mod, T, u["young"], None, kfac, Tm, afac)
        if var == "old":
            return z_below_top(mod, T, u["old"], None, kfac, Tm, afac)
        return z_below_top(mod, T, u["young"], u["old"], kfac, Tm, afac)
    if t == "slab":
        return z_below_top(mod, T, u["age_in"], None, kfac, Tm, afac) / np.cos(np.radians(np.nan_to_num(u["dip"])))
    return np.full(np.broadcast(kfac, Tm, afac).shape, np.nan)


def base_temperatures(r, Ddraw):
    out = {}
    for mod in Q.THERMAL:
        def T_at(dbsl):
            t = r["ttype"]
            if t == "oceanic":
                return Q.temperature_at(dbsl + r["surface_km"], mod, r["age"])
            if t == "transform":
                return Q.temperature_at(dbsl + r["surface_km"], mod, r["age_young"], r["age_old"])
            if t == "slab":
                z = (dbsl - r["slab_km"]) * np.cos(np.radians(np.nan_to_num(r["dip"])))
                return np.where(z >= 0, Q.temperature_at(np.maximum(z, 0), mod, r["age_in"]), 0.0)
            return np.full(np.shape(dbsl), np.nan)
        out[f"Tbase_{mod}"] = float(T_at(np.array(r["D"])))
        if mod == "HSC" and r["ttype"] == "slab" and np.isfinite(r.get("Tslab_D80", np.nan)):
            ts = r["Tslab_D80"]
            z = max((r["D"] - r["slab_km"]) * np.cos(np.radians(np.nan_to_num(r["dip"]))), 0.0)
            out["Tbase_HSC_warmtop"] = float(ts + (Q.THERMAL["HSC"]["Tm"] - ts) * Q.F(z, Q.a_of(r["age_in"], 1e-6)))
        td = T_at(Ddraw)
        out[f"Tbase_{mod}_lo95"] = float(np.nanpercentile(td, 2.5)) if np.isfinite(td).any() else np.nan
        out[f"Tbase_{mod}_hi95"] = float(np.nanpercentile(td, 97.5)) if np.isfinite(td).any() else np.nan
    return out


def base_velocity(e, Dc, Ddraw, mk):
    """Vp of v0.4.2 at the unit's base depth in its events' columns (both bracketing nodes in the mask), median."""
    m = statics()["m"]
    mask = m.mP if mk == "P" else m.cbP
    cols = np.unique(np.c_[e.iy.values, e.ix.values], axis=0)

    def vp_at(z):
        k = np.searchsorted(m.dep, z) - 1
        if k < 0 or k >= len(m.dep) - 1:
            return np.nan
        w = (z - m.dep[k]) / (m.dep[k + 1] - m.dep[k])
        v = []
        for iy, ix in cols:
            if mask[k, iy, ix] and mask[k + 1, iy, ix]:
                v.append(m.Vp[k, iy, ix] * (1 - w) + m.Vp[k + 1, iy, ix] * w)
        return float(np.median(v)) if v else np.nan
    return dict(Vbase=vp_at(Dc), Vbase_lo95=vp_at(np.percentile(Ddraw, 2.5)),
                Vbase_hi95=vp_at(np.percentile(Ddraw, 97.5)))


# ------------------------------------------------------------------ group metrics and rules
def spearman_rows(x, y):
    """Spearman rho per column of x, y [U, K] on the rows finite in both."""
    K = x.shape[1]
    out = np.full(K, np.nan)
    for k in range(K):
        ok = np.isfinite(x[:, k]) & np.isfinite(y[:, k])
        if ok.sum() >= 4:
            a, b = rankdata(x[ok, k]), rankdata(y[ok, k])
            if a.std() > 0 and b.std() > 0:
                out[k] = np.corrcoef(a, b)[0, 1]
    return out


def cv_skill(Dd, Zd, blocks):
    """Leave-one-block-out: D = z + median(r_train) against D = median(D_train). Returns skill, rmse_L, rmse_null [K]."""
    r = Dd - Zd
    valid = np.isfinite(r)
    pred = np.full(Dd.shape, np.nan)
    null = np.full(Dd.shape, np.nan)
    ub = np.unique(blocks)
    if len(ub) < 2:
        nanK = np.full(Dd.shape[1], np.nan)
        return nanK, nanK, nanK
    for b in ub:
        te = blocks == b
        tr = ~te
        rt = np.where(valid[tr], r[tr], np.nan)
        dt = np.where(valid[tr], Dd[tr], np.nan)
        with np.errstate(all="ignore"):
            off = np.nanmedian(rt, axis=0)
            nul = np.nanmedian(dt, axis=0)
        pred[te] = Zd[te] + off[None]
        null[te] = nul[None]
    e1 = np.where(valid, Dd - pred, np.nan)
    e0 = np.where(valid, Dd - null, np.nan)
    with np.errstate(all="ignore"):
        r1 = np.sqrt(np.nanmean(e1 ** 2, axis=0))
        r0 = np.sqrt(np.nanmean(e0 ** 2, axis=0))
    return 1.0 - r1 / r0, r1, r0


def group_sets(U):
    """Registered groups -> boolean masks over units."""
    gs = {}
    seg = U.grouping.eq("seg")
    for g, segs in Q.GROUPS.items():
        gs[g] = (seg & U.key.isin(segs)).values
    for s in Q.OFFSHORE:
        gs[f"seg:{s}"] = (seg & U.key.eq(s)).values
    fr = U.grouping.eq("frame")
    gs["frame:oceanic"] = (fr & U.key.eq("oceanic")).values
    gs["frame:slab"] = (fr & U.key.eq("slab")).values
    gs["frame:oceanic+slab"] = fr.values
    return gs


def evaluate(U, D, preds):
    rows, prefs = [], []
    for g, msk in group_sets(U).items():
        if msk.sum() == 0:
            continue
        Ug = U[msk]
        Dg = D[msk]
        Dc = Ug.D.values
        blocks = Ug.block.values
        for L in ISO_KEYS + ISO_EXTRA + EXT_VARIANTS + LEV_KEYS:
            if L not in Ug:
                continue
            zc = Ug[L].values
            Zd = preds[L][msk]
            okc = np.isfinite(zc)
            r = dict(group=g, level=L, n_units=int(okc.sum()), n_units_all=int(len(Ug)), n_events=int(Ug.n[okc].sum()))
            if okc.sum() == 0:
                rows.append(r)
                continue
            rc = Dc[okc] - zc[okc]
            r["bias"] = float(np.median(rc))
            r["rms"] = float(np.sqrt(np.mean(rc ** 2)))
            res = np.where(np.isfinite(Zd), Dg - Zd, np.nan)
            with np.errstate(all="ignore"):
                b = np.nanmedian(res, axis=0)
                rms = np.sqrt(np.nanmean(res ** 2, axis=0))
            b = b[np.isfinite(b)]
            r["bias_lo95"], r["bias_hi95"] = np.percentile(b, [2.5, 97.5])
            r["rms_lo95"], r["rms_hi95"] = np.nanpercentile(rms, [2.5, 97.5])
            r["P_sit"] = float(np.mean(np.abs(b) <= Q.DELTA))
            r["P_far"] = float(np.mean(np.abs(b) > Q.DELTA))
            r["abs_cat"] = "sits" if r["P_sit"] >= 0.5 else ("falsified" if r["P_far"] >= 0.95 else "marginal")
            span = float(np.percentile(zc[okc], 90) - np.percentile(zc[okc], 10)) if okc.sum() >= 2 else 0.0
            r["span"] = span
            r["shape_testable"] = bool(span >= Q.DELTA and okc.sum() >= 5)
            r["rho"] = float(spearman_rows(Dc[okc, None], zc[okc, None])[0]) if okc.sum() >= 4 else np.nan
            rho_d = spearman_rows(Dg, Zd)
            r["rho_lo95"], r["rho_hi95"] = (np.nanpercentile(rho_d, [2.5, 97.5]) if np.isfinite(rho_d).any()
                                            else (np.nan, np.nan))
            sk, r1, r0 = cv_skill(Dg, Zd, blocks)
            r["n_blocks"] = int(len(np.unique(blocks[okc])))
            if np.isfinite(sk).any():
                r["skill_med"] = float(np.nanmedian(sk))
                r["skill_lo95"], r["skill_hi95"] = np.nanpercentile(sk, [2.5, 97.5])
                r["P_skill_pos"] = float(np.mean(sk[np.isfinite(sk)] > 0))
                r["cv_rmse"] = float(np.nanmedian(r1))
                r["cv_rmse_null"] = float(np.nanmedian(r0))
            if not r["shape_testable"]:
                r["shape_cat"] = "not testable"
            elif r.get("P_skill_pos", 0) >= 0.9:
                r["shape_cat"] = "follows"
            elif 1 - r.get("P_skill_pos", 1) >= 0.9:
                r["shape_cat"] = "does not follow"
            else:
                r["shape_cat"] = "undetermined"
            rows.append(r)
        # preference (units with both predictors in a draw)
        both_c = np.isfinite(Ug[PRIMARY_T].values) & np.isfinite(Ug[PRIMARY_C].values)
        pr = dict(group=g, n_units_both=int(both_c.sum()))
        if both_c.sum() >= 3:
            ok = np.isfinite(preds[PRIMARY_T][msk]) & np.isfinite(preds[PRIMARY_C][msk])
            for name, kt, kc in (("primary", [PRIMARY_T], [PRIMARY_C]), ("best_of_family", ISO_KEYS, LEV_KEYS)):
                okf = ok.copy()
                for L in kt + kc:
                    okf &= np.isfinite(preds[L][msk])
                with np.errstate(all="ignore"):
                    mt = np.nanmin(np.array([np.sqrt(np.nanmean(np.where(okf, Dg - preds[L][msk], np.nan) ** 2,
                                                                axis=0)) for L in kt]), axis=0)
                    mc = np.nanmin(np.array([np.sqrt(np.nanmean(np.where(okf, Dg - preds[L][msk], np.nan) ** 2,
                                                                axis=0)) for L in kc]), axis=0)
                v = np.isfinite(mt) & np.isfinite(mc)
                pT = float(np.mean(mt[v] < mc[v])) if v.any() else np.nan
                pr[f"{name}_P_T"] = pT
                pr[f"{name}_rms_T"] = float(np.nanmedian(mt))
                pr[f"{name}_rms_C"] = float(np.nanmedian(mc))
                pr[f"{name}_pref"] = ("H_T" if pT >= 0.9 else "H_C" if pT <= 0.1 else "not distinguished") \
                    if np.isfinite(pT) else "not testable"
            # cross-validated (shape) preference, primary pair
            okp = ok
            skT, rT, _ = cv_skill(Dg, np.where(okp, preds[PRIMARY_T][msk], np.nan), Ug.block.values)
            skC, rC, _ = cv_skill(Dg, np.where(okp, preds[PRIMARY_C][msk], np.nan), Ug.block.values)
            v = np.isfinite(rT) & np.isfinite(rC)
            pT = float(np.mean(rT[v] < rC[v])) if v.any() else np.nan
            pr["cv_P_T"] = pT
            pr["cv_pref"] = ("H_T" if pT >= 0.9 else "H_C" if pT <= 0.1 else "not distinguished") \
                if np.isfinite(pT) else "not testable"
        prefs.append(pr)
    return pd.DataFrame(rows), pd.DataFrame(prefs)


def answers(G, PR, U):
    """Registered rule 4-5 per group."""
    out = []
    for g, sub in G.groupby("group", sort=False):
        s = sub.set_index("level")
        get = lambda L, c: s.loc[L, c] if L in s.index and c in s.columns else np.nan   # noqa: E731
        iso = [L for L in ISO_KEYS if L in s.index and s.loc[L, "n_units"] > 0]
        lev = [L for L in LEV_KEYS if L in s.index and s.loc[L, "n_units"] > 0]
        nz = lambda v: int(v) if np.isfinite(v) else 0                                    # noqa: E731
        r = dict(group=g, n_units=int(s.n_units_all.max()), n_units_T=nz(get(PRIMARY_T, "n_units")),
                 n_units_C=nz(get(PRIMARY_C, "n_units")))
        T_sits = [L for L in iso if get(L, "abs_cat") == "sits"]
        C_sits = [L for L in lev if get(L, "abs_cat") == "sits"]
        T_follow = get(PRIMARY_T, "shape_cat") == "follows"
        C_follow = get(PRIMARY_C, "shape_cat") == "follows"
        T_fals = (len(iso) == 6 and all(get(L, "abs_cat") == "falsified" for L in iso)) or \
            get(PRIMARY_T, "shape_cat") == "does not follow"
        C_fals = (len(lev) == 2 and all(get(L, "abs_cat") == "falsified" for L in lev)) or \
            get(PRIMARY_C, "shape_cat") == "does not follow"
        pr = PR[PR.group == g]
        prefT = prefC = False
        if len(pr):
            p = pr.iloc[0]
            prefT = any(p.get(f"{n}_pref") == "H_T" for n in ("primary", "best_of_family", "cv"))
            prefC = any(p.get(f"{n}_pref") == "H_C" for n in ("primary", "best_of_family", "cv"))
        T_sup = bool(T_sits) or T_follow
        C_sup = bool(C_sits) or C_follow
        if not iso and not lev:
            ans = "not testable"
        else:
            tl = T_sup and (C_fals or prefT) and not prefC
            cl = C_sup and (T_fals or prefC) and not prefT
            if tl and cl:
                ans = "mixed"
            elif tl:
                ans = "temperature-limited"
            elif cl:
                ans = "composition-limited"
            elif T_sup and C_sup:
                ans = "both consistent"
            elif not T_sup and not C_sup:
                ans = "neither" if (iso and lev) else ("H_T not supported; H_C not testable" if iso else
                                                       "H_C not supported; H_T not testable")
            elif T_sup:
                ans = "H_T supported, not preferred" if lev else "H_T supported (H_C not testable)"
            else:
                ans = "H_C supported, not preferred" if iso else "H_C supported (H_T not testable)"
        # T* outside 350-800 C (reported with the falsification)
        ug = U[group_sets(U)[g]]
        Ts = {}
        for mod in Q.THERMAL:
            v = ug[f"Tbase_{mod}"].dropna()
            Ts[mod] = float(v.median()) if len(v) else np.nan
        r.update(T_sits=";".join(T_sits), C_sits=";".join(C_sits), T_follows=T_follow, C_follows=C_follow,
                 T_falsified=bool(T_fals), C_falsified=bool(C_fals), pref_T=prefT, pref_C=prefC, answer=ans,
                 Tbase_HSC_median=Ts["HSC"], Tbase_GDH1_median=Ts["GDH1"],
                 Vbase_median=float(ug.Vbase.median()) if ug.Vbase.notna().any() else np.nan)
        out.append(r)
    return pd.DataFrame(out)


def tstar_group(U, D):
    """Group T* interval: median over units of T at the base per D90 draw (thermal parameters nominal)."""
    rows = []
    for g, msk in group_sets(U).items():
        if msk.sum() == 0:
            continue
        for mod in Q.THERMAL:
            Ug = U[msk]
            vals = []
            for (_, r), dd in zip(Ug.iterrows(), D[msk]):
                rr = r.to_dict()
                t = base_temperatures(rr, dd)
                vals.append((t[f"Tbase_{mod}_lo95"], t[f"Tbase_{mod}"], t[f"Tbase_{mod}_hi95"]))
            v = np.array(vals, float)
            ok = np.isfinite(v[:, 1])
            if not ok.any():
                continue
            rows.append(dict(group=g, model=mod, n_units=int(ok.sum()), T_med=float(np.median(v[ok, 1])),
                             T_unit_lo95_med=float(np.median(v[ok, 0])), T_unit_hi95_med=float(np.median(v[ok, 2])),
                             T_min=float(v[ok, 1].min()), T_max=float(v[ok, 1].max()),
                             outside_350_800=bool(np.median(v[ok, 0]) > 800 or np.median(v[ok, 2]) < 350)))
    return pd.DataFrame(rows)


def main(names):
    Q.OUT.mkdir(parents=True, exist_ok=True)
    Q.BIG.mkdir(parents=True, exist_ok=True)
    cache = {}
    metas = {}
    for name in names:
        cfg = dict(catalog="v6.1", tiers=("A", "B"), sigz=3.0)
        cfg.update({k: v for k, v in CONFIGS[name].items() if k in cfg})
        key = (cfg["catalog"], cfg["tiers"], cfg["sigz"])
        if key not in cache:
            c, meta = Q.load_events(cfg["catalog"])
            cache[key] = (Q.select(c, cfg["tiers"], cfg["sigz"]), meta)
        E_all, meta = cache[key]
        cfgx, E, U, D, Dsf, preds, meta = run_config(name, E_all, meta)
        G, PR = evaluate(U, D, preds)
        A = answers(G, PR, U)
        TS = tstar_group(U, D)
        U.to_csv(Q.OUT / f"units_{name}.csv", index=False, float_format="%.4g")
        G.to_csv(Q.OUT / f"groups_{name}.csv", index=False, float_format="%.4g")
        PR.to_csv(Q.OUT / f"pref_{name}.csv", index=False, float_format="%.4g")
        A.to_csv(Q.OUT / f"answers_{name}.csv", index=False, float_format="%.4g")
        TS.to_csv(Q.OUT / f"tstar_{name}.csv", index=False, float_format="%.4g")
        np.savez_compressed(Q.BIG / f"draws_{name}.npz", D=D, Dsf=Dsf, **{k: v for k, v in preds.items()})
        if name == "main":
            E.to_parquet(Q.BIG / "events_q2_main.parquet")
        metas[name] = dict(meta, config={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfgx.items()},
                           n_events=int(len(E)), n_units=int(len(U)))
        Q.log(f"[{name}] units {len(U)}; answers:\n" + A[["group", "n_units", "answer"]].to_string())
    mp = Q.OUT / "run_meta.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    old.update(metas)
    mp.write_text(json.dumps(old, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or list(CONFIGS))
