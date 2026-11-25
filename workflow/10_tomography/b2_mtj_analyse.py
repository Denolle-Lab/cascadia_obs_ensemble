"""MTJ shallow-anomaly diagnosis (notes/mtj_shallow.md): compare tomoDD runs on node changes and fit.

For each run tag (baseline d1000_s20 under mendocino_gorda_v2/, tests under mtj_tests/):
* node changes vs the run's own starting MOD on interior nodes: number with |dVp/Vp| > 25% (all, MTJ box, depth
  <= 5 km), number > 15%, max |dVp/Vp|, extreme Vp, Vp/Vs range, change at the reference nodes of the finding
* the share of the total change made in each weighting set (from tomoDD.vel)
* final residuals (tomoDD.res, weight > 0): RMS (ms) of absolute and differential P/S, overall and by station group
  (shelf OBS = OBS shallower than 1.5 km within 40-41 N; Cape land = land stations 40-41 N west of 123.6 W; deep OBS;
  other land), plus absolute RMS on the data common to all runs ("common") so removal tests are comparable
* event shifts vs the input event.dat (back-transformed) for the MTJ shallow events and all events
Writes data/tomo_v1/b2_mtj/runs_compare.csv, rms_groups.csv and prints them.

  python b2_mtj_analyse.py TAG [TAG ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_mtj_common import FIG, INPUTS, MTJ, grid, read_res, read_vel, run_dir, stations  # noqa: E402

REF_NODES = [(-124.125, 40.1, -1.0), (-125.125, 40.4, 5.0), (-124.125, 40.3, -1.0), (-124.5, 40.4, 1.0),
             (-124.75, 40.4, 3.0), (-125.375, 41.3, 9.0), (-126.75, 41.9, 3.0)]


def station_groups():
    st = stations()
    g = np.where(st.is_obs, np.where((st.sta_elev_km > -1.5) & st.sta_lat.between(40.0, 41.0), "shelf_obs",
                                     "deep_obs"),
                 np.where(st.sta_lat.between(40.0, 41.0) & (st.sta_lon < -123.6), "cape_land", "other_land"))
    return dict(zip(st.code, g))


def rms(x, w=None):
    if len(x) == 0:
        return np.nan
    if w is None:
        return float(np.sqrt(np.mean(x ** 2)))
    return float(np.sqrt(np.sum((w * x) ** 2) / np.sum(w ** 2)))


def model_stats(tag):
    g = grid(tag)
    V = read_vel(tag)
    its = sorted(V)
    vp = np.stack([g["vp0"]] + [V[i]["vp"] for i in its])
    vs = np.stack([g["vs0"]] + [V[i]["vs"] for i in its])
    inner = np.zeros(vp.shape[1:], bool)
    inner[1:-1, 1:-1, 1:-1] = True
    d = vp[-1] / vp[0] - 1
    LON, LAT, Z = np.meshgrid(g["lon"], g["lat"], g["z"], indexing="ij")
    mtj = inner & (LAT >= MTJ["lat"][0]) & (LAT <= MTJ["lat"][1]) & (LON >= MTJ["lon"][0]) & (LON <= MTJ["lon"][1]) \
        & (Z <= MTJ["zmax"])
    big = inner & (np.abs(d) > 0.25)
    ratio = (vp[-1] / vs[-1])[inner & (np.abs(vp[-1] - vp[0]) > 1e-3)]
    out = dict(n_gt25=int(big.sum()), n_gt25_mtj=int((big & mtj).sum()), n_gt25_z5=int((big & (Z <= 5)).sum()),
               n_gt15=int((inner & (np.abs(d) > 0.15)).sum()), max_abs_dlnvp=float(np.abs(d[inner]).max()),
               vp_min=float(vp[-1][inner].min()), vp_max=float(vp[-1][inner].max()),
               vpvs_p01=float(np.percentile(ratio, 1)) if len(ratio) else np.nan,
               vpvs_p99=float(np.percentile(ratio, 99)) if len(ratio) else np.nan,
               rms_dlnvp_pct=float(100 * np.sqrt(np.mean(d[inner] ** 2))),
               rms_dlnvp_mtj_pct=float(100 * np.sqrt(np.mean(d[mtj] ** 2))))
    # share of the change on the >25 % nodes made in each set of 3 iterations (sets of the baseline layout)
    step = np.abs(np.diff(vp, axis=0))[:, big]
    if step.size and len(its) == 12:
        tot = step.sum()
        for s in range(4):
            out[f"share_set{s + 1}"] = float(step[3 * s:3 * s + 3].sum() / tot)
    for lon, lat, z in REF_NODES:
        i, j, k = np.argmin(abs(g["lon"] - lon)), np.argmin(abs(g["lat"] - lat)), np.argmin(abs(g["z"] - z))
        out[f"vp@{lon},{lat},{z:g}"] = f"{vp[0][i, j, k]:.2f}->{vp[-1][i, j, k]:.2f}"
    return out, d, g


def fit_stats(tag, common_keys=None):
    r = read_res(tag)
    r = r[r.wt > 0]
    grp = station_groups()
    r["grp"] = r.code.map(grp)
    rows = []
    for (ab, ph), x in r.groupby(["absolute", "phase"]):
        kind = "abs" if ab else "dd"
        rows.append(dict(tag=tag, data=f"{kind}_{ph}", grp="all", n=len(x), rms_ms=rms(x.res_ms.values),
                         wrms_ms=rms(x.res_ms.values, x.wt.values)))
        for gname, y in x.groupby("grp"):
            rows.append(dict(tag=tag, data=f"{kind}_{ph}", grp=gname, n=len(y), rms_ms=rms(y.res_ms.values),
                             wrms_ms=rms(y.res_ms.values, y.wt.values)))
    return pd.DataFrame(rows), r


def main():
    tags = sys.argv[1:]
    rows, fits, res = [], [], {}
    for tag in tags:
        m, _, _ = model_stats(tag)
        f, r = fit_stats(tag)
        res[tag] = r
        fits.append(f)
        log = (run_dir(tag) / "tomoDD-SE.log").read_text(errors="replace")
        import re
        a = [float(x) for x in re.findall(r"absolute ct rms \[s\] =\s+([-\d.]+)", log)]
        w = [float(x) for x in re.findall(r"weighted ct rms \[s\][^=]*=\s+([-\d.]+)", log)]
        rows.append(dict(tag=tag, log_rms_final=a[-1] if a else np.nan, log_wrms_final=w[-1] if w else np.nan, **m))
    df = pd.DataFrame(rows)
    fit = pd.concat(fits)
    # absolute RMS on the data common to all runs (event, station, phase present with weight > 0 everywhere)
    keys = None
    for tag, r in res.items():
        k = set(map(tuple, r[r.absolute][["c1", "code", "phase"]].values))
        keys = k if keys is None else keys & k
    for i, (tag, r) in enumerate(res.items()):
        a = r[r.absolute]
        idx = pd.MultiIndex.from_frame(a[["c1", "code", "phase"]]).isin(list(keys))
        c = a[idx]
        for ph in "PS":
            x = c[c.phase == ph]
            df.loc[df.tag == tag, f"abs_{ph}_common_rms_ms"] = rms(x.res_ms.values)
            for gname in ("shelf_obs", "cape_land", "deep_obs", "other_land"):
                y = x[x.grp == gname]
                df.loc[df.tag == tag, f"abs_{ph}_common_{gname}"] = rms(y.res_ms.values)
        for kind, ab in (("abs", True), ("dd", False)):
            x = r[r.absolute == ab]
            df.loc[df.tag == tag, f"{kind}_rms_ms"] = rms(x.res_ms.values)
    FIG.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(FIG / "runs_compare.csv", index=False)
    fit.round(2).to_csv(FIG / "rms_groups.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 20):
        print(df.set_index("tag").T.to_string())
        print(fit.pivot_table(index=["data", "grp"], columns="tag", values="rms_ms").round(1).to_string())


if __name__ == "__main__":
    main()
