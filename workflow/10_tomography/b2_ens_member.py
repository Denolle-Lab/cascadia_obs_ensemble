"""Track B Bayesian ensemble (D-B030, D-B032, P-M5, P-M6): prior ingredients, member draws, member inputs, one tomoDD
stage of a member, and the member merge. Configuration and layout: b2_ens_common.py, params tomodd.ensemble.

  python b2_ens_member.py prep                      -> ROOT/prior (once per configuration)
  python b2_ens_member.py draws KIND NAME           -> ROOT/KIND/NAME/draws.json, draws.npz, start_hypo.parquet
  python b2_ens_member.py stage KIND NAME STAGE     -> build inputs, run tomoDD (frozen binary), back-transform, trim
  python b2_ens_member.py merge KIND NAME           -> merged.npz (b2_merge rules + the D-B073 node rule)
KIND = synth | real; NAME = s000.. / r000.. (seed = index, synth offset 10000) or nf<i> (noise-floor clone of
synth member 0: every draw of s000 plus N(0, 1e-6 s) on every pick, seed i).

One member (draws once for the whole margin, shared by M0 and every tile, P-M5):
 1 starting model: mixture component (v4-NLL3D Vp/Vs 1.76 / 1.73 / 1.80 = Vp kept, Vp/Vs shifted; Delph et al. 2018
   blended where it covers, if its coverage passes prep), times (1 + f) with independent 3 % Gaussian random fields
   for Vp and Vs (30 km lateral, 8 km depth correlation, tapered to 0 from 50 to 60 km) on one Lambert lattice.
   M0 start = the deterministic M0 start (D-B005 box average) with these; tile start = D-B005 at the tile nodes with
   these, times (1 + this member's M0 change from its own start), Vp/Vs floor start_vpvs_min (D-B046 rule).
 2-3 station statics per station-phase: OBS S N(0, bootstrap sd of the near-event median + 0.05 s) around the
   D-B031 statics already in the data; land P 0.05, land S 0.08, OBS P 0.08 s.
 4 forward-error offsets per station-phase-distance bin: N(0, robust sd of tomoDD - P-1 reference - P-4 correction).
 5 pick noise per pick N(0, lambda_c sigma_pick), lambda_c per class (land/OBS x P/S) from its inverse-gamma
   posterior given the deterministic runs' final residuals (empirical Bayes).
 6 starting hypocentres per event: v4-NLL3D + N(0, h_max / 1.515) per horizontal axis and N(0, sig_z) in depth,
   truncated (exact inverse-CDF sampling, not clipping) at the merged solid surface of the drawn epicentre.
 7 damping: one factor per member, log-uniform [0.7, 1.5], times the configured damping of M0 and of every tile.
 8 D-B073 rule parameters (Vp/Vs U[1.95, 2.25], thickness U[2, 3] km), applied to the merged member's rock nodes.
Items 2-5 are summed into one time shift per pick, added to the absolute times and to both times of every catalog
differential time (statics cancel in dt; noise does not). Smoothing is fixed at the configured value.
"""
import argparse
import gzip
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
from b2_ens_common import CFG, PR, PRIOR, STAGES, T  # noqa: E402
from b2_fwd_common import mod_velocity, read_mod, write_mod  # noqa: E402
from b2_fwd_mods import ellipsoid_coords  # noqa: E402

CLASSES = ["land_P", "land_S", "obs_P", "obs_S"]


def pclass(is_obs, phase):
    return np.where(np.asarray(is_obs, bool), "obs_", "land_") + np.asarray(phase)


def region_cfg(stage):
    return T["regions"][C.stage_cfg(stage)["region"]]


def node_axes(stage):
    import b2_tomodd_inputs as TI
    R = region_cfg(stage)
    return TI.nodes(R["node_lon"]), TI.nodes(R["node_lat"]), np.array(R.get("node_z", T["node_z"]), float)


def forward_from(stage):
    """Inputs folder that holds the P-4 forward run of a stage (station_corr_from for tiles restarted from M0)."""
    F = region_cfg(stage).get("forward", {})
    return C.RUNS / F["station_corr_from"] / "inputs" if F.get("station_corr_from") else C.det_inputs(stage)


# ================================================================ prep
def prep_unions():
    pk, ev, codes = [], [], {}
    for s in STAGES:
        inp = C.det_inputs(s)
        p = pd.read_csv(inp / "picks_used.csv", usecols=["event_ID", "station", "code", "phase", "sigma_s", "is_obs",
                                                         "epi_dist_km"])
        p["stage"] = s
        pk.append(p)
        ev.append(pd.read_csv(inp / "events.csv", usecols=["ID", "origin_time", "lat", "lon", "depth_bsl_km", "h_max",
                                                           "sig_z"]))
        codes[s] = pd.read_csv(inp / "station_codes.csv")[["station", "code", "is_obs"]]
    pk = pd.concat(pk, ignore_index=True)
    chk = pk.groupby(["event_ID", "station", "phase"]).agg(s0=("sigma_s", "min"), s1=("sigma_s", "max"),
                                                           d0=("epi_dist_km", "min"), d1=("epi_dist_km", "max"))
    bad = int(((chk.s1 - chk.s0) > 1e-9).sum())
    if bad:
        raise SystemExit(f"{bad} picks with different sigma between regions")
    u = pk.drop_duplicates(["event_ID", "station", "phase"]).sort_values(["event_ID", "station", "phase"])
    u = u[["event_ID", "station", "phase", "sigma_s", "is_obs", "epi_dist_km"]].reset_index(drop=True)
    u["pidx"] = np.arange(len(u), dtype=np.int64)
    u["cls"] = pclass(u.is_obs, u.phase)
    b = PR["fwd_dist_bins_km"]
    u["bin"] = pd.cut(u.epi_dist_km, b, labels=False, right=True).astype(int)
    ev = pd.concat(ev).drop_duplicates("ID").sort_values("ID").reset_index(drop=True)
    st = u.groupby("station").is_obs.first().reset_index()
    sp = u[["station", "phase", "is_obs"]].drop_duplicates().sort_values(["station", "phase"]).reset_index(drop=True)
    sp["spidx"] = np.arange(len(sp))
    spb = u[["station", "phase", "is_obs", "bin"]].drop_duplicates().sort_values(["station", "phase", "bin"])
    spb = spb.reset_index(drop=True)
    spb["sbidx"] = np.arange(len(spb))
    u = u.merge(sp[["station", "phase", "spidx"]], on=["station", "phase"]).merge(
        spb[["station", "phase", "bin", "sbidx"]], on=["station", "phase", "bin"]).sort_values("pidx")
    u = u.reset_index(drop=True)
    for name, d in (("picks", u), ("events", ev), ("stations", st), ("station_phase", sp), ("station_phase_bin", spb)):
        d.to_parquet(PRIOR / f"{name}.parquet")
    return u, ev, codes, dict(picks=len(u), picks_all_regions=len(pk), events=len(ev), stations=len(st),
                              station_phases=len(sp), station_phase_bins=len(spb),
                              dist_check_max_km=float((chk.d1 - chk.d0).max()))


def prep_statics(sp):
    """OBS S static sd per station: bootstrap (500) sd of the near-event median of each region that has a static for
    it (max over regions) + obs_S_add; stations without a static: obs_S_nostatic. Other classes: fixed sd."""
    g = np.random.default_rng(20261125)
    boot = {}
    for s in STAGES:
        F = region_cfg(s)["forward"]
        f = C.REPO / F["statics_file"]
        if not f.exists():                          # margin_m0's statics file is not in git (built in ~/wt_tomo_res)
            f = Path("/home/mdenolle/wt_tomo_res") / F["statics_file"]
        shutil.copy(f, PRIOR / f"statics_{s}.csv")
        stf = pd.read_csv(f)
        ph = pd.read_parquet(C.RUNS / F["p1_reference"] / "obs_S_iter1_p1.parquet")
        ph["r"] = ph.obs_tt - ph.t_p1
        for r in stf.itertuples():
            if r.static_source == "none" or pd.isna(r.S_static_proposed):
                continue
            D = 60 if r.static_source == "near60" else 100
            x = ph[(ph.sta == r.station) & (ph.dist <= D)].r.values
            if len(x) < 2:
                continue
            b = float(np.median(x[g.integers(0, len(x), (500, len(x)))], axis=1).std())
            boot[r.station] = max(boot.get(r.station, 0.0), b)
    S = PR["static_sd"]
    cls = pclass(sp.is_obs, sp.phase)
    sd = np.array([S.get(c, 0.0) for c in cls], float)
    obsS = cls == "obs_S"
    sd[obsS] = [boot[s] + S["obs_S_add"] if s in boot else S["obs_S_nostatic"] for s in sp.station[obsS]]
    out = sp.assign(cls=cls, sd=sd, boot_sd=[boot.get(s, np.nan) if c == "obs_S" else np.nan
                                              for s, c in zip(sp.station, cls)])
    out.to_parquet(PRIOR / "static_sd.parquet")
    return out


def prep_fwd(u):
    rows = []
    for s in STAGES:
        inp = forward_from(s)
        f = pd.read_parquet(inp / "forward_start_model.parquet")
        c = pd.read_csv(inp / "station_corr.csv").set_index(["station", "phase"]).corr_s
        f["c"] = pd.MultiIndex.from_frame(f[["station", "phase"]]).map(c).values
        rows.append(f)
    f = pd.concat(rows).drop_duplicates(["event_ID", "station", "phase"])
    f = f.merge(u[["event_ID", "station", "phase", "cls", "bin"]], on=["event_ID", "station", "phase"])
    f["e"] = f.t_tomodd - f.t_p1 - f.c
    t = f.groupby(["cls", "bin"]).e.agg(n="size", median="median",
                                        sd=lambda x: 1.4826 * float(np.median(np.abs(x - np.median(x))))).reset_index()
    t.to_parquet(PRIOR / "fwd_sd.parquet")
    return t


def prep_lambda(u, codes):
    rows = []
    for s in STAGES:
        r = C.read_res_abs(C.det_run(s) / "Output_Files" / "tomoDD.res")
        r = r.merge(codes[s][["station", "code"]], on="code")
        rows.append(r.assign(stage=s))
    r = pd.concat(rows)
    d = r.merge(u[["event_ID", "station", "phase", "sigma_s", "cls"]], on=["event_ID", "station", "phase"])
    d["z"] = d.res / d.sigma_s
    out = []
    for c, g in d.groupby("cls"):
        z = g.z.values
        mad = 1.4826 * np.median(np.abs(z - np.median(z)))
        out.append(dict(cls=c, n=len(z), n_eff=int(g.event_ID.nunique()), lam_mad=float(mad),
                        lam_rms=float(np.sqrt(np.mean(z ** 2))), res_mad_s=float(1.4826 * np.median(np.abs(g.res - g.res.median())))))
    t = pd.DataFrame(out)
    t.to_parquet(PRIOR / "class_factors.parquet")
    return t


def prep_base_real(u, codes):
    for s in STAGES:
        inp = C.det_inputs(s)
        cm = dict(zip(codes[s].code, codes[s].station))
        key = u.set_index(["event_ID", "station", "phase"]).pidx
        a = C.parse_absolute(inp / "absolute.dat")
        a["station"] = a.code.map(cm)
        a["pidx"] = pd.MultiIndex.from_frame(a[["event_ID", "station", "phase"]]).map(key).values
        d = C.parse_dt(inp / "dt.ct")
        d["station"] = d.code.map(cm)
        d["pidx1"] = pd.MultiIndex.from_frame(d[["ev1", "station", "phase"]]).map(key).values
        d["pidx2"] = pd.MultiIndex.from_frame(d[["ev2", "station", "phase"]]).map(key).values
        for x, cols in ((a, ["pidx"]), (d, ["pidx1", "pidx2"])):
            if x[cols].isna().any().any():
                raise SystemExit(f"{s}: picks of absolute.dat / dt.ct not in picks_used.csv")
        a.to_parquet(PRIOR / f"abs_real_{s}.parquet")
        d.to_parquet(PRIOR / f"dt_real_{s}.parquet")


def db005_base(lon, lat, z80):
    """D-B005 nearest rock column at the nodes (b2_tomodd_inputs.sample_model box 1), NaN columns -> nearest valid."""
    import b2_tomodd_inputs as TI
    VP, VS = TI.sample_model(lon, lat, z80, 1)
    bad = np.isnan(VP[:, :, 0])
    if bad.any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        VP, VS = VP[idx[0], idx[1]], VS[idx[0], idx[1]]
    return VP, VS


def prep_bases():
    import b2_merge as BM
    out = {}
    for s in STAGES[1:]:
        lo, la, z = node_axes(s)
        VP, VS = db005_base(lo, la, z[z <= 80])
        np.savez(PRIOR / f"db005_{s}.npz", lon=lo, lat=la, z=z, vp=VP, vs=VS)
        # check: the deterministic tile start = this base x (1 + M0's change), Vp/Vs floor (b2_tomodd_inputs.write_mod)
        m = read_mod(C.det_inputs(s) / "MOD")
        vp, vs = m0_change_apply(VP, VS, lo, la, z[z <= 80], C.det_run("m0"))
        k = len(z[z <= 80])
        out[s] = dict(vp_maxdiff=float(np.abs(vp - m["vp"][1:-1, 1:-1, 1:1 + k]).max()),
                      vpvs_maxdiff=float(np.abs(vp / vs - (m["vp"] / m["vs"])[1:-1, 1:-1, 1:1 + k]).max()))
    lo, la, z = BM.grid_axes()
    VP, VS = db005_base(lo, la, z[z <= 80])
    np.savez(PRIOR / "db005_merge.npz", lon=lo, lat=la, z=z, vp=VP, vs=VS)
    vp0, vs0 = BM.reference(lo, la, z)
    vp1, vs1 = reference_member(None, C.det_run("m0"))
    out["merge_reference"] = dict(vp_maxdiff=float(np.abs(vp1 - vp0).max()), vs_maxdiff=float(np.abs(vs1 - vs0).max()))
    return out


def prep_delph():
    """Delph et al. (2018) Vs (data/tomography/Delph2018.nc): holes inside the footprint filled from the nearest valid
    cell, top levels copied up; weight W = clip((distance inside the footprint - 10 km) / taper, 0, 1), 0 below
    max_depth_km. Saved as grids for delph_at(); coverage = share of the deterministic runs' P DWS where W > 0."""
    import xarray as xr
    from scipy import ndimage
    f = C.REPO / "data/tomography/Delph2018.nc"
    if not f.exists():                               # untracked in git; the main checkout holds it (read only)
        f = Path("/home/mdenolle/cascadia_obs_ensemble/data/tomography/Delph2018.nc")
    if not f.exists():
        return dict(available=False)
    shutil.copy(f, PRIOR / "Delph2018.nc")
    d = xr.open_dataset(f)
    vs = d.vs.transpose("depth", "latitude", "longitude").values.astype(float)
    z, la, lo = d.depth.values.astype(float), d.latitude.values.astype(float), d.longitude.values.astype(float)
    foot = ~np.isnan(vs[int(np.searchsorted(z, 10.0))])
    for k in range(len(z)):
        bad = np.isnan(vs[k])
        if bad.all():
            continue
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        vs[k] = vs[k][idx[0], idx[1]]
    k0 = int(np.argmax([~np.isnan(vs[k]).all() for k in range(len(z))]))
    vs[:k0] = vs[k0]
    pad = np.pad(foot, 1, constant_values=False)
    dla, dlo = abs(la[1] - la[0]) * 111.19, abs(lo[1] - lo[0]) * 111.19 * np.cos(np.radians(np.mean(la)))
    din = ndimage.distance_transform_edt(pad, sampling=(dla, dlo))[1:-1, 1:-1]
    np.savez(PRIOR / "delph_grid.npz", vs=vs, z=z, lat=la, lon=lo, din=din)
    share_num = share_den = 0.0
    for s in STAGES[1:]:
        g = np.load(C.det_run(s) / "Output_Files" / "model_geo.npz")
        dp, _ = C.read_dws(C.det_run(s) / "Output_Files" / "tomoDD.vel", g["vp"].shape)
        dp = np.where(np.isfinite(dp), dp, 0.0)[1:-1, 1:-1, 1:-1]
        lo_, la_, z_ = g["lon"][1:-1], g["lat"][1:-1], g["z"][1:-1]
        LO, LA, Z = np.meshgrid(lo_, la_, z_, indexing="ij")
        W = delph_at(LO.ravel(), LA.ravel(), Z.ravel())[1].reshape(LO.shape)
        share_num += float(dp[W > 0].sum())
        share_den += float(dp[Z <= PR["delph"]["max_depth_km"]].sum())
    import hashlib
    cov = dict(available=True, file=str(f), md5=hashlib.md5(f.read_bytes()).hexdigest(),
               dws_share_inside=share_num / max(share_den, 1e-9),
               lat=[float(la.min()), float(la.max())], lon=[float(lo.min()), float(lo.max())])
    cov["use"] = cov["dws_share_inside"] >= PR["delph"]["min_share"]
    return cov


_DELPH = {}


def delph_at(lon, lat, z):
    """(Vs Delph, weight W) at points; W = 0 outside the footprint taper or below max_depth_km."""
    from scipy.interpolate import RegularGridInterpolator
    if "f" not in _DELPH:
        d = np.load(PRIOR / "delph_grid.npz")
        _DELPH["f"] = RegularGridInterpolator((d["z"], d["lat"], d["lon"]), d["vs"], bounds_error=False, fill_value=np.nan)
        _DELPH["d"] = RegularGridInterpolator((d["lat"], d["lon"]), d["din"], bounds_error=False, fill_value=0.0)
    D = PR["delph"]
    w = np.clip((_DELPH["d"](np.c_[lat, lon]) - 10.0) / D["taper_km"], 0, 1)
    v = _DELPH["f"](np.c_[np.maximum(z, 0.0), lat, lon])
    ok = np.isfinite(v) & (np.asarray(z) <= D["max_depth_km"])
    return np.where(ok, v, 1.0), np.where(ok, w, 0.0)


def brocher_vp(vs, vpvs_min=1.76):
    """Brocher (2005) eq. 9, floored at vpvs_min * Vs (as a1_build_model.py)."""
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs ** 2 + 0.2683 * vs ** 3 - 0.0251 * vs ** 4
    return np.maximum(vp, vpvs_min * vs)


def prep_geometry():
    import b2_merge as BM
    lon, lat, z = BM.grid_axes()
    tiles = [dict(region=c["region"], tag=c["tag"], run=C.det_run(s), mask_region=c["region"],
                  mask_damp=float(CFG["mask_damp"]), stage=s) for s, c in CFG["tiles"].items()]
    GEO = BM.geometry(tiles, lon, lat, z)
    with open(PRIOR / "merge_geometry.pkl", "wb") as f:
        pickle.dump(GEO, f)
    return dict(tiles=[t["region"] for t in tiles], gated=[g["tile"]["region"] for g in GEO["geo"] if g["gated"]])


def prep():
    PRIOR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    u, ev, codes, cnt = prep_unions()
    print("unions", cnt, flush=True)
    sp = pd.read_parquet(PRIOR / "station_phase.parquet")
    st = prep_statics(sp)
    print("OBS S static sd:", st[st.cls == "obs_S"].sd.describe().round(3).to_dict(), flush=True)
    fw = prep_fwd(u)
    lam = prep_lambda(u, codes)
    print(lam.round(3).to_string(index=False), flush=True)
    prep_base_real(u, codes)
    dl = prep_delph()
    print("Delph", dl, flush=True)
    geo = prep_geometry()
    (PRIOR / "prior_partial.json").write_text(json.dumps(dict(delph=dl), indent=1))
    chk = prep_bases()
    print("base checks", chk, flush=True)
    mix = PR["mixture"] if dl.get("use") else PR["mixture_no_delph"]
    info = dict(config=C.CONFIG, cfg=CFG, prior=PR, unions=cnt, delph=dl, mixture=mix, geometry=geo, base_checks=chk,
                class_factors=lam.set_index("cls")[["lam_mad", "n_eff", "n"]].to_dict("index"),
                fwd_sd=fw.round(4).to_dict("records"),
                static_sd_obs_S=st[st.cls == "obs_S"].sd.describe().round(4).to_dict(), wall_s=round(time.time() - t0))
    C.jdump(PRIOR / "prior.json", info)
    print("prior done", round(time.time() - t0), "s", flush=True)


# ================================================================ draws
def seed_of(kind, name):
    if name.startswith("nf"):
        return 10_000, int(name[2:])                  # clone of synth member s000
    return (10_000 if kind == "synth" else 0) + int(name[1:]), None


def trunc_normal(g, mu, sd, lo):
    """Exact draws from N(mu, sd) truncated to >= lo (scipy truncnorm, tail-safe): a prior bound, not a clip."""
    from scipy.stats import truncnorm
    sd = np.where(sd > 0, sd, 1e-6)
    return truncnorm.rvs((lo - mu) / sd, np.inf, loc=mu, scale=sd, random_state=g)


def draw_hypo(seed, ev):
    from b4_surface import surface_m
    g = C.rng(seed, "hypo")
    sh = ev.h_max.values / PR["h68_factor"]
    dx, dy = g.standard_normal(len(ev)) * sh, g.standard_normal(len(ev)) * sh
    lat = ev.lat.values + dy / 111.19
    lon = ev.lon.values + dx / (111.19 * np.cos(np.radians(ev.lat.values)))
    zsurf = -surface_m(lon, lat) / 1e3                   # solid surface, km below sea level
    mu, sd = ev.depth_bsl_km.values, ev.sig_z.values
    z = trunc_normal(g, mu, sd, zsurf)
    out = pd.DataFrame({"ID": ev.ID.values, "lat": lat, "lon": lon, "depth": z})
    return out, dict(n=len(ev), n_start_above_surface=int((mu < -surface_m(ev.lon.values, ev.lat.values) / 1e3).sum()),
                     dh_median_km=float(np.median(np.hypot(dx, dy))), dz_abs_median_km=float(np.median(np.abs(z - mu))),
                     n_drawn_above_surface=int((z < zsurf - 1e-9).sum()))


def draw_delta(seed, lam_scale=None, eps_seed=None):
    """Per-pick time shift (s) over the pick union: static + forward offset + pick noise (+ eps for clones)."""
    u = pd.read_parquet(PRIOR / "picks.parquet")
    st = pd.read_parquet(PRIOR / "static_sd.parquet")
    fw = pd.read_parquet(PRIOR / "fwd_sd.parquet").set_index(["cls", "bin"]).sd
    spb = pd.read_parquet(PRIOR / "station_phase_bin.parquet")
    lamt = pd.read_parquet(PRIOR / "class_factors.parquet").set_index("cls")
    stat = C.rng(seed, "static").standard_normal(len(st)) * st.sd.values
    fsd = pd.MultiIndex.from_arrays([pclass(spb.is_obs, spb.phase), spb.bin]).map(fw).values.astype(float)
    fsd = np.where(np.isfinite(fsd), fsd, np.nanmax(fw.values))
    fwd = C.rng(seed, "fwd").standard_normal(len(spb)) * fsd
    g = C.rng(seed, "lam")
    lam = {}
    for c in CLASSES:
        a = lamt.loc[c, "n_eff"] / 2.0
        lam[c] = float(np.sqrt(a * lamt.loc[c, "lam_mad"] ** 2 / g.gamma(a))) if lam_scale is None else float(lam_scale[c])
    lp = u.cls.map(lam).values
    noise = C.rng(seed, "noise").standard_normal(len(u)) * lp * u.sigma_s.values
    delta = stat[u.spidx.values] + fwd[u.sbidx.values] + noise
    if eps_seed is not None:
        delta = delta + C.rng(eps_seed, "eps").standard_normal(len(u)) * float(PR["eps_floor_s"])
    cls = u.cls.values
    info = dict(lambda_c=lam, static_sd_drawn={c: float(np.std(stat[(st.cls == c).values])) for c in CLASSES},
                fwd_sd_drawn=float(np.std(fwd)), noise_sd={c: float(np.std(noise[cls == c])) for c in CLASSES},
                delta_sd=float(np.std(delta)))
    return delta, stat, info


def draws(kind, name):
    md = C.member_dir(kind, name)
    md.mkdir(parents=True, exist_ok=True)
    if (md / "draws.json").exists():
        return json.loads((md / "draws.json").read_text())
    seed, eps = seed_of(kind, name)
    pri = json.loads((PRIOR / "prior.json").read_text())
    names = list(pri["mixture"])
    comp = names[int(C.rng(seed, "component").choice(len(names), p=[pri["mixture"][n] for n in names]))]
    fp, fs = C.random_field(C.rng(seed, "rf_p")), C.random_field(C.rng(seed, "rf_s"))
    delta, stat, dinfo = draw_delta(seed, eps_seed=eps)
    ev = pd.read_parquet(PRIOR / "events.parquet")
    hyp, hinfo = draw_hypo(seed, ev)
    g = C.rng(seed, "damp")
    a, b = PR["damp_factor"]
    dfac = float(np.exp(g.uniform(np.log(a), np.log(b))))
    gs = C.rng(seed, "sc")
    sc = dict(vpvs=float(gs.uniform(*PR["shallow_crust"]["vpvs"])),
              thickness_km=float(gs.uniform(*PR["shallow_crust"]["thickness_km"])))
    np.savez(md / "draws.npz", rf_p=fp, rf_s=fs, delta=delta, static=stat)
    hyp.to_parquet(md / "start_hypo.parquet")
    d = dict(kind=kind, member=name, seed=seed, eps_clone=eps, config=C.CONFIG, component=comp,
             rf_p_std_pct=float(100 * fp[:, :, :12].std()), rf_s_std_pct=float(100 * fs[:, :, :12].std()),
             damp_factor=dfac, damp={s: round(C.stage_cfg(s)["damp"] * dfac, 2) for s in STAGES},
             smooth={s: C.stage_cfg(s)["smooth"] for s in STAGES}, shallow_crust=sc, data=dinfo, hypo=hinfo)
    C.jdump(md / "draws.json", d)
    return d


# ================================================================ starting models
def comp_apply(vp, vs, comp, lon, lat, z):
    """Mixture component on start velocities [x, y, z] at geographic nodes (lon, lat, z 1D axes)."""
    vp, vs = vp.copy(), vs.copy()
    if comp.startswith("v4_"):
        shift = int(comp[3:]) / 100.0 - PR["vpvs_ref"]
        vs = vp / (vp / vs + shift)
    elif comp == "delph":
        LO, LA, Z = np.meshgrid(lon, lat, z, indexing="ij")
        vsd, W = (a.reshape(LO.shape) for a in delph_at(LO.ravel(), LA.ravel(), Z.ravel()))
        vp = W * brocher_vp(vsd) + (1 - W) * vp
        vs = W * vsd + (1 - W) * vs
    return vp, vs


def field_at(f, lon, lat, z):
    LO, LA, Z = np.meshgrid(lon, lat, z, indexing="ij")
    return C.sample_field(f, LO.ravel(), LA.ravel(), Z.ravel()).reshape(LO.shape)


def m0_change_apply(VP, VS, lon, lat, z80, m0_run):
    """x (1 + M0 run's change from its own start) at the nodes, then the start Vp/Vs floor (D-B046)."""
    g = np.load(Path(m0_run) / "Output_Files" / "model_geo.npz")
    s = read_mod(Path(m0_run) / "MOD")
    LO, LA, Z = np.meshgrid(lon, lat, z80, indexing="ij")
    args = (LO.ravel(), LA.ravel(), Z.ravel())
    fin = dict(lon=g["lon"], lat=g["lat"], z=g["z"], vp=g["vp"], vs=g["vs"])
    sta = dict(lon=g["lon"], lat=g["lat"], z=g["z"], vp=s["vp"], vs=s["vs"])
    fp, fs = (mod_velocity(fin, ph, *args, clamp_surface=False).reshape(LO.shape) for ph in "PS")
    sp, ss = (mod_velocity(sta, ph, *args, clamp_surface=False).reshape(LO.shape) for ph in "PS")
    VP, VS = VP * fp / sp, VS * fs / ss
    fl = float(T.get("start_vpvs_min", 0))
    return VP, np.where(VP / VS < fl, VP / fl, VS)


def start_values(d, lon, lat, z80, base_vp, base_vs, m0_run=None):
    """Member start on nodes z <= 80: component, x (1 + f), [x M0 change + floor]."""
    vp, vs = comp_apply(base_vp, base_vs, d["component"], lon, lat, z80) if d else (base_vp, base_vs)
    if d:
        dr = np.load(C.member_dir(d["kind"], d["member"]) / "draws.npz")
        vp = vp * (1 + field_at(dr["rf_p"], lon, lat, z80))
        vs = vs * (1 + field_at(dr["rf_s"], lon, lat, z80))
    if m0_run is not None:
        vp, vs = m0_change_apply(vp, vs, lon, lat, z80, m0_run)
    return vp, vs


def deep_and_pad(m, vp80, vs80, zi):
    """Interior: vp80/vs80 for z <= 80, deep_vp with the 80 km Vp/Vs below; boundary nodes as write_mod (lateral and
    top copies, bottom node Vp kept with the deepest interior Vp/Vs)."""
    r80 = (vp80 / vs80)[:, :, -1:]
    deep = [zz for zz in zi if zz > 80]
    vp = np.concatenate([vp80] + [np.full(vp80.shape[:2] + (1,), T["deep_vp"][int(zz)]) for zz in deep], axis=2)
    r = np.concatenate([vp80 / vs80] + [r80] * len(deep), axis=2)
    VP, R = m["vp"].copy(), (m["vp"] / m["vs"]).copy()
    VP[1:-1, 1:-1, 1:-1], R[1:-1, 1:-1, 1:-1] = vp, r
    for a in (VP, R):
        a[0], a[-1] = a[1], a[-2]
        a[:, 0], a[:, -1] = a[:, 1], a[:, -2]
        a[:, :, 0] = a[:, :, 1]
    R[:, :, -1] = R[:, :, -2]
    return {**m, "vp": VP, "vs": VP / R}


def m0_start(d):
    import b2_tomodd_inputs as TI
    m = read_mod(C.det_inputs("m0") / "MOD")
    R = region_cfg("m0")
    lon, lat, zi = TI.nodes(R["node_lon"]), TI.nodes(R["node_lat"]), np.array(R.get("node_z", T["node_z"]), float)
    k = int((zi <= 80).sum())
    vp80, vs80 = start_values(d, lon, lat, zi[:k], m["vp"][1:-1, 1:-1, 1:1 + k], m["vs"][1:-1, 1:-1, 1:1 + k])
    if d:                       # the D-B046 start floor for every member start: independent Vp and Vs fields (3 %
        fl = float(T.get("start_vpvs_min", 0))      # each) would otherwise give Vp/Vs down to ~1.5 at ~3 % of nodes
        vs80 = np.where(vp80 / vs80 < fl, vp80 / fl, vs80)
    return deep_and_pad(m, vp80, vs80, zi)


def tile_start(d, stage, m0_run):
    b = np.load(PRIOR / f"db005_{stage}.npz")
    m = read_mod(C.det_inputs(stage) / "MOD")
    k = int((b["z"] <= 80).sum())
    vp80, vs80 = start_values(d, b["lon"], b["lat"], b["z"][:k], b["vp"], b["vs"], m0_run)
    return deep_and_pad(m, vp80, vs80, b["z"])


def reference_member(d, m0_run):
    """V_ref on the merge grid: the tile-start rule (D-B005 nearest, component, x (1 + f), x M0 change, floor), deep
    nodes as b2_merge.reference. d = None: the deterministic reference."""
    b = np.load(PRIOR / "db005_merge.npz")
    k = int((b["z"] <= 80).sum())
    vp, vs = start_values(d, b["lon"], b["lat"], b["z"][:k], b["vp"], b["vs"], m0_run)
    deep = [zz for zz in b["z"] if zz > 80]
    r80 = (vp / vs)[:, :, -1:]
    VP = np.concatenate([vp] + [np.full(vp.shape[:2] + (1,), T["deep_vp"][int(zz)]) for zz in deep], axis=2)
    VS = np.concatenate([vs] + [np.full(vs.shape[:2] + (1,), T["deep_vp"][int(zz)]) / r80 for zz in deep], axis=2)
    return VP, VS


# ================================================================ one stage
def base_tables(kind, stage):
    src = PRIOR if kind == "real" else C.ROOT / "synth"
    return pd.read_parquet(src / f"abs_{kind}_{stage}.parquet"), pd.read_parquet(src / f"dt_{kind}_{stage}.parquet")


def build(kind, name, stage):
    d = json.loads((C.member_dir(kind, name) / "draws.json").read_text())
    md = C.member_dir(kind, name)
    sd = md / stage
    inp = sd / "inputs"
    if inp.exists():
        shutil.rmtree(inp)
    inp.mkdir(parents=True)
    det = C.det_inputs(stage)
    if stage == "m0":
        m = m0_start(d)
    else:
        m = tile_start(d, stage, md / "m0" / "run")
    write_mod(inp / "MOD", m)
    delta = np.load(md / "draws.npz")["delta"]
    a, dt = base_tables(kind, stage)
    a["t_m"] = a.t.values + delta[a.pidx.values]
    C.write_absolute(inp / "absolute.dat", a, "t_m")
    dt["t1_m"] = dt.t1.values + delta[dt.pidx1.values]
    dt["t2_m"] = dt.t2.values + delta[dt.pidx2.values]
    C.write_dt(inp / "dt.ct", dt, "t1_m", "t2_m")
    fw = json.loads((det / "forward.json").read_text())
    h = pd.read_parquet(md / "start_hypo.parquet").set_index("ID")
    la, lo = ellipsoid_coords(h.lat.values, h.lon.values, *fw["center"]) if fw.get("coords") == "ellipsoid" \
        else (h.lat.values, h.lon.values)
    hw = pd.DataFrame({"lat": la, "lon": lo, "depth": h.depth.values}, index=h.index)
    C.write_event_dat(inp / "event.dat", det / "event.dat", hw)
    (inp / "station.dat").symlink_to(det / "station.dat")
    for f in ("forward.json", "summary.json"):
        shutil.copy(det / f, inp / f)
    vpvs = (m["vp"] / m["vs"])[1:-1, 1:-1, 1:-1]
    return dict(start_vpvs_min=float(vpvs.min()), start_vpvs_median=float(np.median(vpvs)),
                start_vp_min=float(m["vp"].min()), start_vp_max=float(m["vp"].max()))


def write_inp(run, stage, damp, smooth):
    from b2_tomodd_run import INP
    summ = json.loads((run.parent / "inputs" / "summary.json").read_text())
    rows = [f"  {s[0]:<3d}  -9    -9    -9   -9  {s[1]:5.2f} {s[2]:5.2f} {s[3]:4g} {s[4]:4g} {s[5]:6g} {damp:6g}   1  "
            f"{T['inp']['thres']:g}" for s in T["inp"]["sets"]]
    w1, w2, w3 = (f"{v:g}" for v in smooth)
    (run / "tomoDD-SE.inp").write_text(INP.format(
        region=f"ensemble {C.CONFIG} {run.parent.parent.name} {stage}", tag="run", air=summ["air_dep_km"],
        nset=len(rows), stepl=T["inp"]["stepl_km"], clat=summ["center_lat"], clon=summ["center_lon"], w1=w1, w2=w2,
        w3=w3, sets="\n".join(rows)))


def backtransform(run):
    from b2_tomodd_backtransform import inverse, read_model_dat, table
    fw = json.loads((run.parent / "inputs" / "forward.json").read_text())
    o = run / "Output_Files"
    for nm in ("tomoDD.reloc",):
        table(o / nm, o / f"{nm}_geo", 1, 2, fw)
    if (o / "tomoDD.sta").exists():
        table(o / "tomoDD.sta", o / "tomoDD.sta_geo", 1, 2, fw, "{:.5f}")
    m = read_mod(run / "MOD")
    c = fw.get("center", [0.0, 0.0])
    lat = inverse(m["lat"], np.full_like(m["lat"], c[1]), fw)[0]
    lon = inverse(np.full_like(m["lon"], c[0]), m["lon"], fw)[1]
    nx, ny, nz = len(m["lon"]), len(m["lat"]), len(m["z"])
    out = dict(lon=lon, lat=lat, z=m["z"])
    for ph in ("Vp", "Vs"):
        out[ph.lower()] = read_model_dat(o / f"{ph}_model.dat", nx, ny, nz)
    np.savez_compressed(o / "model_geo.npz", **out)
    return out


def trim(run, codes):
    from b2_tomodd_summary import parse_log
    o = run / "Output_Files"
    g = np.load(o / "model_geo.npz")
    dp, ds = C.read_dws(o / "tomoDD.vel", g["vp"].shape)
    np.savez_compressed(run / "dws.npz", p=dp.astype("f4"), s=ds.astype("f4"))
    r = C.read_res_abs(o / "tomoDD.res")
    isobs = codes.set_index("code").is_obs
    r["cls"] = np.where(r.code.map(isobs).values.astype(bool), "obs_", "land_") + r.phase.values
    res = {c: dict(n=int(len(x)), rms=float(np.sqrt(np.mean(x.res ** 2))),
                   mad=float(1.4826 * np.median(np.abs(x.res - x.res.median()))), median=float(x.res.median()))
           for c, x in r.groupby("cls")}
    try:
        lg = parse_log(run / "tomoDD-SE.log")
    except Exception as ex:                          # noqa: BLE001
        lg = dict(error=str(ex))
    info = dict(resid_abs=res, log=lg, time=(run / "time.txt").read_text().strip()[-200:])
    C.jdump(run / "resid.json", info)
    keep = {"model_geo.npz", "tomoDD.reloc_geo", "tomoDD.sta_geo"}
    for f in o.iterdir():
        if f.name not in keep:
            f.unlink()
    shutil.rmtree(run / "Input_Files", ignore_errors=True)
    for f in ("ak135.15.SKS", "layer-16.dat", "grid.dat", "tomoDD-SE.res.ini"):
        (run / f).unlink(missing_ok=True)
    for f in ("tomoDD-SE.log", "stdout.txt"):
        if (run / f).exists():
            with open(run / f, "rb") as fi, gzip.open(run / f"{f}.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
            (run / f).unlink()
    inp = run.parent / "inputs"
    for f in ("absolute.dat", "dt.ct", "event.dat", "station.dat"):
        (inp / f).unlink(missing_ok=True)
    return info


def run_stage(kind, name, stage):
    md = C.member_dir(kind, name)
    sd = md / stage
    if (sd / "DONE").exists():
        return json.loads((sd / "DONE").read_text())
    if stage != "m0" and not (md / "m0" / "DONE").exists():
        raise SystemExit(f"{kind}/{name}/{stage}: M0 of this member not done")
    t0 = time.time()
    (sd / "FAILED").unlink(missing_ok=True)
    shutil.rmtree(sd / "run", ignore_errors=True)
    d = draws(kind, name)
    binfo = build(kind, name, stage)
    run = sd / "run"
    (run / "Output_Files").mkdir(parents=True)
    (run / "Input_Files").mkdir()
    for f in ("event.dat", "station.dat", "absolute.dat", "dt.ct"):
        (run / "Input_Files" / f).symlink_to(sd / "inputs" / f)
    shutil.copy(sd / "inputs" / "MOD", run / "MOD")
    for f in ("ak135.15.SKS", "layer-16.dat"):
        shutil.copy(Path(C.E["runtime_files"]) / f, run / f)
    damp = d["damp"][stage]
    write_inp(run, stage, damp, C.stage_cfg(stage)["smooth"])
    b = C.binary(stage)
    t1 = time.time()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "OMP_NUM_THREADS": "1"}   # system libgfortran 8.5
    with open(run / "stdout.txt", "w") as so, open(run / "time.txt", "w") as se:
        rc = subprocess.run(["nice", "-n", str(C.E["nice"]), "/usr/bin/time", "-f", "%e s %U user %M KB", b,
                             "tomoDD-SE.inp"], cwd=run, stdout=so, stderr=se, env=env).returncode
    t_run = time.time() - t1
    if rc or not (run / "Output_Files" / "tomoDD.reloc").exists() or not (run / "Output_Files" / "Vp_model.dat").exists():
        rec = dict(stage=stage, status="failed", rc=rc, time=(run / "time.txt").read_text()[-300:])
        C.jdump(sd / "FAILED", rec)
        raise SystemExit(f"{kind}/{name}/{stage}: tomoDD exit {rc}")
    backtransform(run)
    codes = pd.read_csv(C.det_inputs(stage) / "station_codes.csv")
    info = trim(run, codes)
    tt = (run / "time.txt").read_text().split()
    try:
        rss_kb = int(tt[tt.index("KB") - 1])
    except (ValueError, IndexError):
        rss_kb = None
    du = sum(p.stat().st_size for p in sd.rglob("*") if p.is_file() and not p.is_symlink())
    rec = dict(stage=stage, status="done", binary=b, damp=damp, smooth=C.stage_cfg(stage)["smooth"],
               wall_min=round((time.time() - t0) / 60, 2), tomodd_min=round(t_run / 60, 2), max_rss_gb=
               round(rss_kb / 1e6, 2) if rss_kb else None, kept_mb=round(du / 1e6, 2),
               rms_abs_final=info["log"].get("rms_abs_final"), cnd_last=info["log"].get("cnd_last"),
               events_final=sum(1 for _ in open(run / "Output_Files" / "tomoDD.reloc_geo")),
               events_start=sum(1 for _ in open(C.det_inputs(stage) / "event.dat")),
               aq_repeats=info["log"].get("aq_repeats"), passes=info["log"].get("passes"), build=binfo,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.jdump(sd / "DONE", rec)
    print(f"{kind}/{name}/{stage}: done {rec['wall_min']} min, {rec['max_rss_gb']} GB, {rec['kept_mb']} MB kept",
          flush=True)
    return rec


# ================================================================ merge
def merge(kind, name):
    import xarray as xr
    import b2_merge as BM
    md = C.member_dir(kind, name)
    if (md / "MERGED").exists():
        return json.loads((md / "MERGED").read_text())
    t0 = time.time()
    d = json.loads((md / "draws.json").read_text())
    for s in STAGES:
        if not (md / s / "DONE").exists():
            raise SystemExit(f"{kind}/{name}: stage {s} not done")
    with open(PRIOR / "merge_geometry.pkl", "rb") as f:
        GEO = pickle.load(f)
    for g in GEO["geo"]:
        g["tile"] = {**g["tile"], "run": md / g["tile"]["stage"] / "run"}
    VP0, VS0 = reference_member(d, md / "m0" / "run")
    B = BM.blend(GEO, VP0, VS0, quiet=True)
    shp = GEO["shape"]
    nd = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{CFG['node_model']}.nc")
    tr = lambda a: np.asarray(a).transpose(2, 1, 0)          # [z, y, x] -> [x, y, z]
    med = tr(nd.medium.values)
    off = nd.offshore.values.T.astype(bool)[:, :, None]
    surf = nd.surface_elevation.values.T.astype(float)[:, :, None]
    z = nd.depth.values[None, None, :]
    vp, vs = B["out"]["P"], B["out"]["S"]
    sc = d["shallow_crust"]
    rule = (med == 0) & off & (z > 0) & (z + surf <= sc["thickness_km"])
    vs_rule = np.where(med == 0, np.where(rule, np.minimum(vs, vp / sc["vpvs"]), vs), np.nan)
    vp_med = np.where(med == 0, vp, np.where(med == 1, 1.5, np.nan))
    f4 = lambda a: np.asarray(a, "float32")
    np.savez_compressed(md / "merged.npz", vp=f4(vp), vs=f4(vs), vp_start=f4(B["start"]["P"]), vs_start=f4(B["start"]["S"]),
                        vp_medium=f4(vp_med), vs_rule=f4(vs_rule), r_p=f4(B["Rg"]["P"]), r_s=f4(B["Rg"]["S"]),
                        dws_p=f4(B["dsum"]["p"].reshape(shp)), dws_s=f4(B["dsum"]["s"].reshape(shp)),
                        vs_reverted=B["vs_rev"].astype("int8"), rule=rule.astype("int8"))
    rec = dict(status="merged", wall_s=round(time.time() - t0, 1), fall=B["fall"],
               vs_reverted=int((B["vs_rev"] > 0).sum()), rule_nodes_lowered=int((rule & (vs_rule < vs)).sum()),
               shallow_crust=sc, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.jdump(md / "MERGED", rec)
    print(f"{kind}/{name}: merged in {rec['wall_s']} s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "draws", "stage", "merge"])
    ap.add_argument("kind", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("stage", nargs="?")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "draws":
        print(json.dumps(draws(a.kind, a.name), indent=1, default=str)[:3000])
    elif a.cmd == "stage":
        run_stage(a.kind, a.name, a.stage)
    else:
        merge(a.kind, a.name)


if __name__ == "__main__":
    main()
