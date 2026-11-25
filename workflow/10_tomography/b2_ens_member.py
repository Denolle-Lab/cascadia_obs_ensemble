"""Track B Bayesian ensemble (D-B030/D-B032), member inputs: priors (prep) and one member's tomoDD inputs (build).

  python b2_ens_member.py prep                       -> <runs>/mendocino_gorda_v2/ensemble_prior/
  python b2_ens_member.py build ENS M                -> <ens>/m<MMM>/inputs/  (seed = M)
  python b2_ens_member.py build ENS nf<i> --clone-of K
        noise-floor clone: every draw of member K, plus N(0, 1e-6 s) on every absolute time (seed i)

prep (once per region; numbers in ensemble_prior/prior.json, notes/ensemble_calibration.md):
  * pick-noise class factors lambda_c (land/OBS x P/S): robust scale 1.4826 MAD of (absolute residual / sigma_pick)
    of the baseline run d1000_s20 (final iteration); posterior lambda_c^2 ~ InvGamma(n_eff/2, n_eff/2 lambda_c^2),
    n_eff = events with a pick of that class (empirical Bayes, flat prior on log lambda)
  * OBS S static sigma: bootstrap (500) std of the D-B031 near-event median per station, + 0.05 s
  * forward-error sigma per station type x phase x distance bin: 1.4826 MAD of the v2 end-to-end residual
    (tomoDD - P-1 reference - P-4 correction), inputs/forward_start_model.parquet
  * rock-surface depth at each event epicentre (v4-NLL3D NLL model, first cell with Vp > 1.6): depth clip
  * Delph et al. (2018) component: Vs from data/tomography/Delph2018.nc (trilinear; depth km bsl, clamped to >= 0),
    Vp = Brocher (2005) floored at 1.76 Vs (a1_build_model.py), blended into the v4-NLL3D MOD with weight
    clip((d_inside - 10 km) / 30 km, 0, 1) (d_inside = distance inside the footprint), 0 below 80 km; used in the
    mixture only if >= 10% of the baseline DWS (P) lies where the weight is > 0 (coverage reported)
  * base data of the real ensemble: picks (t_abs = tt_abs of picks_used, t_dt = tt) and the dt.ct pair list

build: the design items, each from its own RNG stream (seed = member index, b2_ens_common.rng)
  1 starting model: component of the mixture (v4-NLL3D Vp/Vs 1.76 / 1.73 / 1.80 = Vp kept, Vp/Vs shifted by
    -0.03 / +0.04; Delph hybrid), times (1 + f) with independent 3% Gaussian random fields for Vp and Vs
  2-3 statics per station-phase; 4 forward-error offsets per station-phase-distance bin; 5 pick noise per pick
    N(0, lambda_c sigma_pick) with lambda_c drawn per member; all added to the absolute times, dt.ct rebuilt from the
    perturbed times on the fixed pair list (statics cancel in it)
  6 starting hypocentres: N(v4-NLL3D, sigma_h = h_max/1.515 per horizontal axis, sigma_z = sig_z), depth >= rock
    surface; origin times unchanged
  7 damping log-uniform [700, 1500], lateral smoothing uniform [15, 30], depth smoothing = half
Writes inputs/{event.dat, station.dat, absolute.dat, dt.ct, MOD, summary.json, forward.json, draws.json,
start_hypo.csv}. Base data: <ens>/base/picks.parquet (b2_ens_synth.py truth, or the real one from prep).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_ens_common import (BASELINE, DIST_BINS, INPUTS, PRIOR, PRIOR_CFG as C, REPO, T, brocher_vp,  # noqa: E402
                           ens_dir, random_field, read_dt_pairs, read_dws, read_res_abs, repad, rng, write_absolute,
                           write_dt, write_event_dat)
from b2_fwd_common import read_mod, write_mod  # noqa: E402
from b2_fwd_mods import ellipsoid_coords  # noqa: E402
from b2_tomodd_inputs import load_nll, PROJ  # noqa: E402
from manifest import write_manifest  # noqa: E402

CLASSES = ["land_P", "land_S", "obs_P", "obs_S"]


def pclass(is_obs, phase):
    return np.where(is_obs, "obs_", "land_") + np.asarray(phase)


# ================================================================ prep
def prep_lambda(pk):
    r = read_res_abs(BASELINE / "Output_Files" / "tomoDD.res")
    d = r.merge(pk[["event_ID", "code", "phase", "sigma_s", "is_obs"]], on=["event_ID", "code", "phase"])
    d["z"] = d.res / d.sigma_s
    d["cls"] = pclass(d.is_obs.values, d.phase.values)
    rows = []
    for c, g in d.groupby("cls"):
        z = g.z.values
        mad = 1.4826 * np.median(np.abs(z - np.median(z)))
        t = z[np.abs(z) < 5 * mad]
        rows.append(dict(cls=c, n=len(z), n_eff=int(g.event_ID.nunique()), lam_mad=mad,
                         lam_rms_trim5=float(np.sqrt(np.mean(t ** 2))), lam_rms=float(np.sqrt(np.mean(z ** 2))),
                         res_median_s=float(np.median(g.res)), res_mad_s=float(1.4826 * np.median(np.abs(g.res - np.median(g.res))))))
    return pd.DataFrame(rows)


def prep_statics():
    F = T["regions"]["mendocino_gorda_v2"]["forward"]
    st = pd.read_csv(REPO / F["statics_file"])
    ph = pd.read_parquet(Path(T["runs"]) / F["p1_reference"] / "obs_S_iter1_p1.parquet")
    ph["r"] = ph.obs_tt - ph.t_p1
    g = np.random.default_rng(20261122)
    out = []
    for s in st.itertuples():
        if s.static_source == "none" or pd.isna(s.S_static_proposed):
            out.append(dict(station=s.station, static=np.nan, boot_sd=np.nan, n=0, sd=C["static_sd"]["obs_S_nostatic"]))
            continue
        D = 60 if s.static_source == "near60" else 100
        x = ph[(ph.sta == s.station) & (ph.dist <= D)].r.values
        b = np.median(x[g.integers(0, len(x), (500, len(x)))], axis=1)
        out.append(dict(station=s.station, static=float(np.median(x)), static_file=s.S_static_proposed,
                        boot_sd=float(b.std()), n=len(x), sd=float(b.std()) + C["static_add_s"]))
    return pd.DataFrame(out)


def prep_fwd(pk):
    f = pd.read_parquet(INPUTS / "forward_start_model.parquet")
    c = pd.read_csv(INPUTS / "station_corr.csv").set_index(["station", "phase"]).corr_s
    f["c"] = pd.MultiIndex.from_frame(f[["station", "phase"]]).map(c).values
    f = f.merge(pk[["event_ID", "station", "phase", "is_obs", "epi_dist_km"]], on=["event_ID", "station", "phase"])
    f["e"] = f.t_tomodd - f.t_p1 - f.c
    f["bin"] = pd.cut(f.epi_dist_km, DIST_BINS, labels=False, right=True)
    f["cls"] = pclass(f.is_obs.values, f.phase.values)
    g = f.groupby(["cls", "bin"]).e
    t = pd.DataFrame({"n": g.size(), "median": g.median(),
                      "sd": g.apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))}).reset_index()
    return t


def prep_surface(ev):
    vp, (x0, y0, z0, dx) = load_nll("P")
    X, Y = PROJ(ev.lon.values, ev.lat.values)
    ix = np.clip(np.rint((X / 1e3 - x0) / dx).astype(int), 0, vp.shape[0] - 1)
    iy = np.clip(np.rint((Y / 1e3 - y0) / dx).astype(int), 0, vp.shape[1] - 1)
    col = vp[ix, iy, :] > 1.6
    return pd.DataFrame({"ID": ev.ID.values, "z_surf_km": z0 + dx * np.argmax(col, axis=1)})


def prep_delph(m0):
    import xarray as xr
    from scipy import ndimage
    from scipy.interpolate import RegularGridInterpolator
    d = xr.open_dataset(REPO / "data/tomography/Delph2018.nc")
    vs = d.vs.transpose("depth", "latitude", "longitude").values.astype(float)
    z, la, lo = d.depth.values.astype(float), d.latitude.values.astype(float), d.longitude.values.astype(float)
    foot = ~np.isnan(vs[np.searchsorted(z, 10.0)])
    for k in range(len(z)):                          # fill holes inside the footprint from the nearest valid cell
        bad = np.isnan(vs[k])
        if bad.all():
            vs[k] = np.nan
            continue
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        vs[k] = vs[k][idx[0], idx[1]]
    k0 = int(np.argmax([~np.isnan(vs[k]).all() for k in range(len(z))]))
    vs[:k0] = vs[k0]
    # distance (km) inside the footprint; outside the Delph box counts as outside
    pad = np.pad(foot, 1, constant_values=False)
    dlat_km, dlon_km = 0.2 * 111.19, 0.2 * 111.19 * np.cos(np.radians(41.3))
    din = ndimage.distance_transform_edt(pad, sampling=(dlat_km, dlon_km))[1:-1, 1:-1]
    fV = RegularGridInterpolator((z, la, lo), vs, bounds_error=False, fill_value=np.nan)
    fD = RegularGridInterpolator((la, lo), din, bounds_error=False, fill_value=0.0)
    s = np.load(INPUTS / "start_model.npz")
    LON, LAT = np.meshgrid(s["lon"], s["lat"], indexing="ij")
    w2 = np.clip((fD(np.c_[LAT.ravel(), LON.ravel()]).reshape(LON.shape) - 10.0) / C["delph_taper_km"], 0, 1)
    zi = s["z"]
    vp0, vs0 = m0["vp"][1:-1, 1:-1, 1:-1], m0["vs"][1:-1, 1:-1, 1:-1]
    vpd, vsd = vp0.copy(), vs0.copy()
    W = np.zeros(vp0.shape)
    for k, zk in enumerate(zi):
        if zk > 80:
            continue
        q = fV(np.c_[np.full(LON.size, max(zk, 0.0)), LAT.ravel(), LON.ravel()]).reshape(LON.shape)
        ok = np.isfinite(q) & (w2 > 0)
        W[:, :, k] = np.where(ok, w2, 0.0)
        qs = np.where(ok, q, 1.0)
        vpd[:, :, k] = W[:, :, k] * brocher_vp(qs) + (1 - W[:, :, k]) * vp0[:, :, k]
        vsd[:, :, k] = W[:, :, k] * qs + (1 - W[:, :, k]) * vs0[:, :, k]
    return vpd, vsd, W


def prep():
    PRIOR.mkdir(parents=True, exist_ok=True)
    pk = pd.read_csv(INPUTS / "picks_used.csv")
    ev = pd.read_csv(INPUTS / "events.csv")
    lam = prep_lambda(pk)
    lam.round(4).to_csv(PRIOR / "class_factors.csv", index=False)
    print(lam.round(3).to_string(index=False))
    sta = prep_statics()
    sta.round(4).to_csv(PRIOR / "static_sigma.csv", index=False)
    print("OBS S static sd:", sta.sd.describe().round(3).to_dict())
    fw = prep_fwd(pk)
    fw.round(4).to_csv(PRIOR / "fwd_sigma.csv", index=False)
    print(fw.round(3).to_string(index=False))
    surf = prep_surface(ev)
    surf.to_csv(PRIOR / "ev_surface.csv", index=False)
    m0 = read_mod(INPUTS / "MOD")
    vpd, vsd, W = prep_delph(m0)
    s = np.load(INPUTS / "start_model.npz")
    dp, _ = read_dws(BASELINE / "Output_Files" / "tomoDD.vel", m0["vp"].shape)
    dpi = dp[1:-1, 1:-1, 1:-1]
    dpi = np.where(np.isfinite(dpi), dpi, np.nanmax(dpi[np.isfinite(dpi)]))
    inside = W > 0
    z80 = (s["z"] <= 80)[None, None, :]
    resolved = dpi > T["inp"]["thres"]
    cov = dict(nodes_inside_frac=float(inside[:, :, s["z"] <= 80].mean()),
               dws_share_inside=float(dpi[inside].sum() / dpi[np.broadcast_to(z80, dpi.shape)].sum()),
               dws_nodes_inside_frac=float((inside & resolved).sum() / (resolved & z80).sum()),
               events_inside_frac=float(np.mean(_inside_events(ev, s, W))),
               full_weight_nodes_frac=float((W >= 1)[:, :, s["z"] <= 80].mean()))
    cov["dws_share_inside_by_depth"] = {f"{z:g}": round(float(dpi[:, :, k][inside[:, :, k]].sum() / max(dpi[:, :, k].sum(), 1e-9)), 3)
                                        for k, z in enumerate(s["z"]) if z <= 80}
    rel = (vsd / m0["vs"][1:-1, 1:-1, 1:-1] - 1)[inside]
    relp = (vpd / m0["vp"][1:-1, 1:-1, 1:-1] - 1)[inside]
    cov["dlnvs_inside_median_pct"] = float(100 * np.median(rel))
    cov["dlnvs_inside_p5_p95_pct"] = [float(100 * np.percentile(rel, 5)), float(100 * np.percentile(rel, 95))]
    cov["dlnvp_inside_median_pct"] = float(100 * np.median(relp))
    cov["dlnvp_inside_p5_p95_pct"] = [float(100 * np.percentile(relp, 5)), float(100 * np.percentile(relp, 95))]
    use_delph = cov["dws_share_inside"] >= C["delph_min_share"]
    np.savez_compressed(PRIOR / "delph_component.npz", vp=vpd, vs=vsd, w=W)
    # real-data base
    dtp = read_dt_pairs(INPUTS / "dt.ct")
    dtp.to_parquet(PRIOR / "dt_pairs.parquet")
    base = pk[["event_ID", "station", "code", "phase", "is_obs", "sigma_s", "w", "epi_dist_km", "tt_abs", "tt",
               "static_s"]].rename(columns={"tt_abs": "t_abs", "tt": "t_dt"})
    base.to_parquet(PRIOR / "picks_real.parquet")
    pri = dict(cfg=C, use_delph=bool(use_delph), mixture=C["mixture"] if use_delph else C["mixture_no_delph"],
               delph_coverage=cov, class_factors=lam.set_index("cls")[["lam_mad", "n_eff"]].to_dict("index"))
    (PRIOR / "prior.json").write_text(json.dumps(pri, indent=1, default=float))
    print(json.dumps(cov, indent=1))
    write_manifest("b2_ens_prep", PRIOR, inputs=[INPUTS / "picks_used.csv", INPUTS / "events.csv", INPUTS / "MOD",
                                                 BASELINE / "Output_Files" / "tomoDD.res", REPO / "data/tomography/Delph2018.nc"],
                   outputs=sorted(PRIOR.glob("*.csv")) + sorted(PRIOR.glob("*.json")) + [PRIOR / "delph_component.npz"],
                   params={"cfg": C}, counts={"use_delph": use_delph, **{k: v for k, v in cov.items() if not isinstance(v, dict)}})


def _inside_events(ev, s, W):
    i = np.clip(np.searchsorted(s["lon"], ev.lon.values), 0, len(s["lon"]) - 1)
    j = np.clip(np.searchsorted(s["lat"], ev.lat.values), 0, len(s["lat"]) - 1)
    k = np.clip(np.searchsorted(s["z"], ev.depth_bsl_km.values), 0, len(s["z"]) - 1)
    return W[i, j, k] > 0


# ================================================================ build
def draw_model(seed, pri):
    m0 = read_mod(INPUTS / "MOD")
    s = np.load(INPUTS / "start_model.npz")
    names = list(pri["mixture"])
    comp = names[int(rng(seed, "model").choice(len(names), p=[pri["mixture"][n] for n in names]))]
    m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m0.items()}
    if comp.startswith("v4_"):
        shift = int(comp[3:]) / 100.0 - C["vpvs_ref"]
        m["vs"] = m["vp"] / (m["vp"] / m["vs"] + shift)
    elif comp == "delph":
        dd = np.load(PRIOR / "delph_component.npz")
        m["vp"][1:-1, 1:-1, 1:-1], m["vs"][1:-1, 1:-1, 1:-1] = dd["vp"], dd["vs"]
    args = (s["lon"], s["lat"], s["z"], C["rf_std"], C["rf_len_lat_km"], C["rf_len_z_km"], C["rf_taper_km"])
    fp = random_field(rng(seed, "rf_p"), *args)
    fs = random_field(rng(seed, "rf_s"), *args)
    m["vp"][1:-1, 1:-1, 1:-1] *= 1 + fp
    m["vs"][1:-1, 1:-1, 1:-1] *= 1 + fs
    m = repad(m)
    info = dict(component=comp, rf_p_std_pct=float(100 * fp[:, :, s["z"] <= 50].std()),
                rf_s_std_pct=float(100 * fs[:, :, s["z"] <= 50].std()),
                start_vpvs_median=float(np.median((m["vp"] / m["vs"])[1:-1, 1:-1, 1:-1])))
    return m, info


def draw_data(seed, pk, lam_scale=None, eps_seed=None):
    """Total perturbation per pick: static + forward offset + pick noise (+ eps for noise-floor clones)."""
    sta = pd.read_csv(PRIOR / "static_sigma.csv").set_index("station").sd
    fw = pd.read_csv(PRIOR / "fwd_sigma.csv").set_index(["cls", "bin"]).sd
    lamt = pd.read_csv(PRIOR / "class_factors.csv").set_index("cls")
    cls = pclass(pk.is_obs.values, pk.phase.values)
    # items 2-3: one static per station-phase
    sp = pk[["station", "phase", "is_obs"]].drop_duplicates().sort_values(["station", "phase"]).reset_index(drop=True)
    sp_cls = pclass(sp.is_obs.values, sp.phase.values)
    sd = np.array([C["static_sd"][c] for c in np.where(sp_cls == "obs_S", "land_P", sp_cls)])   # placeholder for obs_S
    obsS = sp_cls == "obs_S"
    sd[obsS] = sp.station[obsS].map(sta).fillna(C["static_sd"]["obs_S_nostatic"]).values
    st_val = rng(seed, "static").standard_normal(len(sp)) * sd
    st_map = dict(zip(zip(sp.station, sp.phase), st_val))
    # item 4: one offset per station-phase-distance bin
    b = pd.cut(pk.epi_dist_km, DIST_BINS, labels=False, right=True).values
    spb = pd.DataFrame({"station": pk.station.values, "phase": pk.phase.values, "cls": cls, "bin": b}).drop_duplicates()
    spb = spb.sort_values(["station", "phase", "bin"]).reset_index(drop=True)
    fsd = pd.MultiIndex.from_frame(spb[["cls", "bin"]]).map(fw).fillna(fw.max()).values
    fv = rng(seed, "fwd").standard_normal(len(spb)) * fsd
    fw_map = dict(zip(zip(spb.station, spb.phase, spb.bin), fv))
    # item 5: class factors from the inverse-gamma posterior, then per-pick noise
    g = rng(seed, "lam")
    lam = {}
    for c in CLASSES:
        a = lamt.loc[c, "n_eff"] / 2.0
        l2 = a * lamt.loc[c, "lam_mad"] ** 2 / g.gamma(a)
        lam[c] = float(np.sqrt(l2)) if lam_scale is None else float(lam_scale[c])
    lam_pick = np.array([lam[c] for c in cls])
    noise = rng(seed, "noise").standard_normal(len(pk)) * lam_pick * pk.sigma_s.values
    stat = np.array([st_map[k] for k in zip(pk.station, pk.phase)])
    fwd = np.array([fw_map[k] for k in zip(pk.station, pk.phase, b)])
    delta = stat + fwd + noise
    if eps_seed is not None:
        delta = delta + np.random.default_rng([10_000, int(eps_seed)]).standard_normal(len(pk)) * C["eps_floor_s"]
    info = dict(lambda_c=lam, static_sd_by_class={c: float(np.std(st_val[sp_cls == c])) for c in CLASSES},
                fwd_sd_drawn=float(np.std(fv)), noise_sd_by_class={c: float(np.std(noise[cls == c])) for c in CLASSES},
                delta_sd=float(np.std(delta)),
                statics={f"{s}|{p}": round(float(v), 5) for (s, p), v in st_map.items()})
    return delta, info


def draw_hypo(seed, ev):
    surf = pd.read_csv(PRIOR / "ev_surface.csv").set_index("ID").z_surf_km.reindex(ev.ID).values
    g = rng(seed, "hypo")
    sh = ev.h_max.values / C["h68_factor"]
    dx, dy, dz = (g.standard_normal(len(ev)) * sh, g.standard_normal(len(ev)) * sh,
                  g.standard_normal(len(ev)) * ev.sig_z.values)
    lat = ev.lat.values + dy / 111.19
    lon = ev.lon.values + dx / (111.19 * np.cos(np.radians(ev.lat.values)))
    z = ev.depth_bsl_km.values + dz
    zmin = np.minimum(surf, ev.depth_bsl_km.values)
    nclip = int((z < zmin).sum())
    z = np.maximum(z, zmin)
    out = ev[["ID", "origin_time"]].copy()
    out["lat"], out["lon"], out["depth_bsl_km"] = lat, lon, z
    return out, dict(n_depth_clipped=nclip, dh_median_km=float(np.median(np.hypot(dx, dy))),
                     dz_abs_median_km=float(np.median(np.abs(z - ev.depth_bsl_km.values))))


def draw_reg(seed):
    g = rng(seed, "reg")
    a, b = C["damp_range"]
    damp = float(np.exp(g.uniform(np.log(a), np.log(b))))
    s = float(g.uniform(*C["smooth_range"]))
    return dict(damp=round(damp, 1), smooth=[round(s, 2), round(s, 2), round(s / 2, 2)])


def build(ens, mname, clone_of=None):
    E = ens_dir(ens)
    pri = json.loads((PRIOR / "prior.json").read_text())
    if mname.startswith("nf"):
        seed, eps = int(clone_of), int(mname[2:])
    else:
        seed, eps = int(mname[1:]), None
    out = E / mname / "inputs"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pk = pd.read_parquet(E / "base" / "picks.parquet").sort_values(["event_ID", "phase", "code"], kind="stable")
    pk = pk.reset_index(drop=True)
    ev = pd.read_csv(INPUTS / "events.csv")
    fw = json.loads((INPUTS / "forward.json").read_text())
    m, minfo = draw_model(seed, pri)
    write_mod(out / "MOD", m)
    delta, dinfo = draw_data(seed, pk, eps_seed=eps)
    pk["t_abs_m"] = pk.t_abs.values + delta
    pk["t_dt_m"] = pk.t_dt.values + delta
    order = pd.read_csv(INPUTS / "picks_used.csv", usecols=["event_ID", "code", "phase"])  # builder pick order
    pk_o = order.merge(pk, on=["event_ID", "code", "phase"], how="inner")
    write_absolute(out / "absolute.dat", pk_o, "t_abs_m")
    tmap = dict(zip(zip(pk.event_ID.values, pk.code.values, pk.phase.values), pk.t_dt_m.values))
    write_dt(out / "dt.ct", pd.read_parquet(PRIOR / "dt_pairs.parquet"), tmap)
    h, hinfo = draw_hypo(seed, ev)
    h.to_csv(out / "start_hypo.csv", index=False)
    hw = h.copy()
    hw["lat"], hw["lon"] = ellipsoid_coords(h.lat.values, h.lon.values, *fw["center"])
    write_event_dat(out / "event.dat", hw)
    (out / "station.dat").symlink_to(INPUTS / "station.dat")
    shutil.copy(INPUTS / "forward.json", out / "forward.json")
    shutil.copy(INPUTS / "summary.json", out / "summary.json")
    reg = draw_reg(seed)
    draws = dict(ensemble=ens, member=mname, seed=seed, eps_clone=eps, model=minfo, data=dinfo, hypo=hinfo, reg=reg,
                 base=str(E / "base" / "picks.parquet"))
    (out / "draws.json").write_text(json.dumps(draws, indent=1))
    print(f"{ens}/{mname}: seed {seed} eps {eps} | {minfo['component']} rf {minfo['rf_p_std_pct']:.2f}/"
          f"{minfo['rf_s_std_pct']:.2f}% | lambda {', '.join(f'{k} {v:.2f}' for k, v in dinfo['lambda_c'].items())} | "
          f"damp {reg['damp']} smooth {reg['smooth']} | hypo clip {hinfo['n_depth_clipped']}", flush=True)
    return draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "build"])
    ap.add_argument("ens", nargs="?")
    ap.add_argument("member", nargs="?")
    ap.add_argument("--clone-of", type=int)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    else:
        build(a.ens, a.member, a.clone_of)


if __name__ == "__main__":
    main()
