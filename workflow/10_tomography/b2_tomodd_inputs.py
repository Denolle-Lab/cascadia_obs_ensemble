"""Track B, step B2a: tomoDD-SE input files for one region from the B1b balanced dataset (D-B025..D-B027).

Writes <runs>/<region>/inputs/: event.dat, station.dat, absolute.dat, dt.ct, MOD, plus station_codes.csv,
events.csv (with tomoDD IDs = catalog IDs) and a summary. Parameters: params.yaml `tomodd`.

Data
* Events: B1b events (main and ridge rule) with epicentre inside the region's event box.
* Picks: B1b picks of those events at stations inside the box + station margin. Travel time = pick minus the
  v4-NLL3D origin time (B1 `travel_time_s`). OBS S picks: minus the D-B020 near-event static
  (S_static_proposed of obs_near_station_S_terms.csv); OBS P and land picks uncorrected.
* Weight per pick: w_bal (B1b) x (0.05/sigma)^2 (D-B008), scaled so the region's largest weight is 3
  (tomoDD drops weights outside [0, 3]); no clipping, relative weights unchanged. S is down-weighted through sigma (2x P), so WTCTS = WTCTP.
* Catalog differential times: each event paired with up to max_neighbours nearest events within
  max_sep_km (3D, v4-NLL3D hypocentres); a pair is kept with >= min_common common station-phase picks;
  weight = min of the two pick weights.
* Station labels: numeric codes S0000... (tomoDD keeps 7 characters); elevation in metres, + up.

Starting model (MOD, simul2000 format, lon/lat/depth nodes)
* Interior nodes from params (node_lon, node_lat, node_z), boundary nodes edge_deg outside and at edge_z.
* Vp and Vs from the v4-NLL3D 3D model (NLL SLOW_LEN grids, D-B005), rock nodes only (water Vp 1.5 and air
  0.3 km/s are excluded): mean over a model_box_avg_km horizontal box at the node depth (linear in depth
  between the 1 km NLL levels). Where the box has no rock at that depth (node above the seafloor or the
  ground), the velocity of the shallowest rock level below in the same box is used. Below 80 km: deep_vp
  with the 80 km Vp/Vs. Boundary nodes copy the nearest interior node.

Checks: compiled limits (MAXEVE, MAXSTA, MAXOBS = stations per event, MAXDATA) -> stop if exceeded.

Run: LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python \
        workflow/10_tomography/b2_tomodd_inputs.py mendocino_gorda
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402

P = load_params()
T = P["tomodd"]
B1B = Path(P["paths"]["base"]) / "b1b_balanced"
MODEL = Path(P["paths"]["start_model"])
PROJ = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")  # = NLL grid
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def nodes(spec):
    a, b, d = spec
    return np.round(np.arange(a, b + d / 2, d), 4)


# ---------------------------------------------------------------- data
def select(R):
    ev = pd.read_csv(B1B / "events.csv")
    pk = pd.read_csv(B1B / "picks.csv")
    (la0, la1), (lo0, lo1) = R["events_lat"], R["events_lon"]
    mla, mlo = R["station_margin_deg"]
    e = ev[ev.lat.between(la0, la1) & ev.lon.between(lo0, lo1)].copy()
    p = pk[pk.event_ID.isin(e.ID) & pk.sta_lat.between(la0 - mla, la1 + mla)
           & pk.sta_lon.between(lo0 - mlo, lo1 + mlo)].copy()
    log(f"events in box {len(e)}; picks {len(p)} (of {int(pk.event_ID.isin(e.ID).sum())} for these events; "
        f"the rest are at stations outside the box + margin)")
    return e, p


def obs_statics(p):
    st = pd.read_csv(REPO / T["obs_S_static"]).set_index("station").S_static_proposed
    m = p.is_obs & (p.phase == "S")
    s = p.loc[m, "station"].map(st)
    p["static_s"] = 0.0
    p.loc[m, "static_s"] = s.fillna(0.0).values
    p["tt"] = p.travel_time_s - p.static_s
    log(f"OBS S picks {int(m.sum())}: static applied to {int(s.notna().sum())} "
        f"(median {s.median():.3f} s, range {s.min():.2f} to {s.max():.2f}); none for {int(s.isna().sum())}")
    return p


def weights(p):
    W = T["weights"]
    q = p.w_bal * (0.05 / p.sigma_s) ** 2
    q = W["clip_max"] * q / q.max()                   # best pick -> clip_max; relative weights unchanged
    p["w_raw"] = q
    p["w"] = np.clip(q, 0.0, W["clip_max"])
    log(f"weights: median P {p.w[p.phase == 'P'].median():.3f}, S {p.w[p.phase == 'S'].median():.3f}; "
        f"clipped at {W['clip_max']}: {int((q > W['clip_max']).sum())} picks ({100 * (q > W['clip_max']).mean():.2f}%)")
    return p


def diff_times(e, p):
    D = T["dt"]
    lat0 = np.deg2rad(e.lat.mean())
    xyz = np.c_[e.lon.values * 111.19 * np.cos(lat0), e.lat.values * 111.19, e.depth_bsl_km.values]
    ids = e.ID.values
    dd, ii = cKDTree(xyz).query(xyz, k=D["max_neighbours"] + 1, distance_upper_bound=D["max_sep_km"])
    pairs = set()
    for a in range(len(ids)):
        for d_, b in zip(dd[a, 1:], ii[a, 1:]):
            if np.isfinite(d_):
                pairs.add((min(a, b), max(a, b)))
    key = {k: dict(zip(zip(v.code, v.phase), zip(v.tt, v.w))) for k, v in p.groupby("event_ID")}
    out, n_pairs = [], 0
    for a, b in sorted(pairs):
        A, B_ = key.get(ids[a], {}), key.get(ids[b], {})
        common = [k for k in A if k in B_]
        if len(common) < D["min_common"]:
            continue
        n_pairs += 1
        out.append(f"#  {ids[a]}  {ids[b]}")
        out += [f"{k[0]:<7s} {A[k][0]:9.4f} {B_[k][0]:9.4f} {min(A[k][1], B_[k][1]):6.3f} {k[1]}" for k in common]
    n_dt = len(out) - n_pairs
    log(f"differential times: {len(pairs)} candidate pairs, {n_pairs} kept (>= {D['min_common']} common), {n_dt} data")
    return out, n_pairs, n_dt


# ---------------------------------------------------------------- model
def load_nll(phase):
    h = (MODEL / f"cascadia3d.{phase}.mod.hdr").read_text().split()
    nx, ny, nz = map(int, h[:3])
    x0, y0, z0, dx = map(float, h[3:7])
    s = np.fromfile(MODEL / f"cascadia3d.{phase}.mod.buf", dtype="<f4").reshape(nx, ny, nz)
    return dx / s, (x0, y0, z0, dx)


def sample_model(xn, yn, zn_int):
    vp, g = load_nll("P")
    vs, _ = load_nll("S")
    x0, y0, z0, dx = g
    nx, ny, nz = vp.shape
    rock = vp > 1.6                                   # water 1.5, air 0.3 (a1_build_model.py)
    LON, LAT = np.meshgrid(xn, yn, indexing="ij")
    X, Y = PROJ(LON, LAT)
    ix = np.rint((X / 1e3 - x0) / dx).astype(int)
    iy = np.rint((Y / 1e3 - y0) / dx).astype(int)
    h = int(T["model_box_avg_km"] // 2)
    VP = np.full((len(xn), len(yn), len(zn_int)), np.nan)
    VS = np.full_like(VP, np.nan)
    zlev = z0 + dx * np.arange(nz)
    zq = np.clip(np.asarray(zn_int, float), zlev[0], zlev[-1])
    n_fill = 0
    for i in range(len(xn)):
        for j in range(len(yn)):
            a0, a1 = max(ix[i, j] - h, 0), min(ix[i, j] + h + 1, nx)
            b0, b1 = max(iy[i, j] - h, 0), min(iy[i, j] + h + 1, ny)
            if a0 >= a1 or b0 >= b1:
                continue
            r = rock[a0:a1, b0:b1, :]
            cnt = r.sum(axis=(0, 1))
            with np.errstate(invalid="ignore", divide="ignore"):
                mp = np.where(r, vp[a0:a1, b0:b1, :], 0).sum(axis=(0, 1)) / cnt
                ms = np.where(r, vs[a0:a1, b0:b1, :], 0).sum(axis=(0, 1)) / cnt
            ok = cnt > 0
            if not ok.any():
                continue
            first = np.argmax(ok)
            mp[:first], ms[:first] = mp[first], ms[first]      # above the rock surface: shallowest rock level
            n_fill += int((zlev[:first] >= zq.min()).sum() > 0)
            mp = pd.Series(mp).ffill().bfill().values          # isolated empty levels (none expected)
            ms = pd.Series(ms).ffill().bfill().values
            VP[i, j, :] = np.interp(zq, zlev, mp)
            VS[i, j, :] = np.interp(zq, zlev, ms)
    log(f"model sampled at {len(xn)} x {len(yn)} x {len(zn_int)} nodes; columns with rock fill above the surface "
        f"{n_fill}; nodes outside the NLL grid {int(np.isnan(VP).sum())}")
    return VP, VS


def write_mod(path, R):
    xi, yi = nodes(R["node_lon"]), nodes(R["node_lat"])
    zi = np.array(T["node_z"], float)
    z_model = zi[zi <= 80]
    VP, VS = sample_model(xi, yi, z_model)
    # fill any interior column outside the NLL grid with the nearest valid column
    bad = np.isnan(VP[:, :, 0])
    if bad.any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        VP, VS = VP[idx[0], idx[1]], VS[idx[0], idx[1]]
    ratio = VP / VS
    deep = [z for z in zi if z > 80]
    vp_full = np.concatenate([VP] + [np.full(VP.shape[:2] + (1,), T["deep_vp"][int(z)]) for z in deep], axis=2)
    r_full = np.concatenate([ratio] + [ratio[:, :, -1:]] * len(deep), axis=2)
    # boundary nodes
    E = T["edge_deg"]
    xn = np.r_[xi[0] - E, xi, xi[-1] + E]
    yn = np.r_[yi[0] - E, yi, yi[-1] + E]
    zt, zb = T["edge_z"]
    zn = np.r_[zt, zi, zb]
    vp_full = np.pad(vp_full, ((1, 1), (1, 1), (0, 0)), mode="edge")
    r_full = np.pad(r_full, ((1, 1), (1, 1), (0, 0)), mode="edge")
    vp_full = np.concatenate([vp_full[:, :, :1], vp_full, np.full(vp_full.shape[:2] + (1,), T["deep_vp"][int(zb)])], axis=2)
    r_full = np.concatenate([r_full[:, :, :1], r_full, r_full[:, :, -1:]], axis=2)
    lines = [f"0.01 {len(xn)} {len(yn)} {len(zn)}",
             " ".join(f"{v:.4f}" for v in xn), " ".join(f"{v:.4f}" for v in yn), " ".join(f"{v:.3f}" for v in zn)]
    for arr, fmt in ((vp_full, "{:.3f}"), (r_full, "{:.4f}")):
        for k in range(len(zn)):
            for j in range(len(yn)):
                lines.append(" ".join(fmt.format(v) for v in arr[:, j, k]))
    path.write_text("\n".join(lines) + "\n")
    npar = 2 * len(xi) * len(yi) * len(zi)
    log(f"MOD: {len(xn)} x {len(yn)} x {len(zn)} nodes (interior {len(xi)} x {len(yi)} x {len(zi)}, "
        f"{npar} Vp+Vs parameters); Vp {vp_full.min():.2f}-{vp_full.max():.2f} km/s, "
        f"Vp/Vs {r_full.min():.3f}-{r_full.max():.3f}")
    np.savez_compressed(path.with_name("start_model.npz"), lon=xi, lat=yi, z=zi, vp=vp_full[1:-1, 1:-1, 1:-1],
                        vpvs=r_full[1:-1, 1:-1, 1:-1])
    lim = P["tomodd"]
    return dict(nx=len(xn), ny=len(yn), nz=len(zn), npar=npar)


# ---------------------------------------------------------------- main
def main():
    region = sys.argv[1]
    R = T["regions"][region]
    out = Path(T["runs"]) / region / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    L = T["limits"]

    e, p = select(R)
    p = obs_statics(p)
    p = weights(p)
    st = p.groupby("station")[["network", "sta_lat", "sta_lon", "sta_elev_km", "is_obs"]].first().reset_index()
    st["code"] = [f"S{i:04d}" for i in range(len(st))]
    p["code"] = p.station.map(dict(zip(st.station, st.code)))
    e = e[e.ID.isin(p.event_ID)].copy()
    nsta_ev = p.groupby("event_ID").code.nunique()
    dt_lines, n_pairs, n_dt = diff_times(e, p)
    n_data = len(p) + n_dt
    checks = {"MAXEVE": (len(e), L["MAXEVE"]), "MAXSTA": (len(st), L["MAXSTA"]),
              "MAXOBS": (int(nsta_ev.max()), L["MAXOBS"]), "MAXDATA": (n_data, L["MAXDATA"])}
    for k, (v, lim) in checks.items():
        log(f"limit {k}: {v} of {lim}")
        if v > lim:
            raise SystemExit(f"{k} exceeded ({v} > {lim}): rebuild tomoDD with a larger {k} or shrink the region")

    with open(out / "station.dat", "w") as f:
        for r in st.itertuples():
            f.write(f"{r.code} {r.sta_lat:.5f} {r.sta_lon:.5f} {r.sta_elev_km * 1000.0:.1f}\n")
    with open(out / "event.dat", "w") as f:
        for r in e.itertuples():
            t = pd.Timestamp(r.origin_time)
            cs = int(round((t.second + t.microsecond / 1e6) * 100))
            f.write(f"{t.strftime('%Y%m%d')}  {t.hour:02d}{t.minute:02d}{cs:04d} {r.lat:10.5f} {r.lon:11.5f} "
                    f"{r.depth_bsl_km:8.3f} 1.0 0.0 0.0 0.0 {r.ID:10d} 0\n")
    with open(out / "absolute.dat", "w") as f:
        for eid, d in p.groupby("event_ID"):
            f.write(f"#  {eid}\n")
            for r in d.itertuples():
                f.write(f"{r.code:<7s} {r.tt:9.4f} {r.w:6.3f} {r.phase}\n")
    (out / "dt.ct").write_text("\n".join(dt_lines) + "\n")
    mod = write_mod(out / "MOD", R)
    st.to_csv(out / "station_codes.csv", index=False)
    e.to_csv(out / "events.csv", index=False)
    p[["event_ID", "station", "code", "phase", "travel_time_s", "static_s", "tt", "sigma_s", "w_bal", "w_raw", "w",
       "is_obs", "epi_dist_km"]].to_csv(out / "picks_used.csv", index=False)

    elev_max = float(st.sta_elev_km.max())
    summ = dict(region=region, events=len(e), stations=len(st), obs_stations=int(st.is_obs.sum()),
                picks=len(p), P=int((p.phase == "P").sum()), S=int((p.phase == "S").sum()),
                dt_pairs=n_pairs, dt_data=n_dt, max_sta_per_event=int(nsta_ev.max()),
                air_dep_km=round(-(elev_max + T["air_dep_margin_km"]), 3),
                center_lat=round(float(np.mean(R["events_lat"])), 3), center_lon=round(float(np.mean(R["events_lon"])), 3),
                **mod)
    pd.Series(summ).to_json(out / "summary.json", indent=1)
    (out / "build.log").write_text("\n".join(LOG) + "\n")
    log(summ)
    files = [out / f for f in ("event.dat", "station.dat", "absolute.dat", "dt.ct", "MOD", "station_codes.csv",
                               "events.csv", "picks_used.csv", "summary.json")]
    write_manifest(f"b2_tomodd_inputs_{region}", out,
                   inputs=[B1B / "events.csv", B1B / "picks.csv", REPO / T["obs_S_static"],
                           MODEL / "cascadia3d.P.mod.buf", MODEL / "cascadia3d.S.mod.buf"],
                   outputs=files, params={"tomodd": T, "region": region}, counts=summ)


if __name__ == "__main__":
    main()
