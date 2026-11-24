"""D-B048: compare the relocations of one case (B4 start, tomoDD-SE variants, GrowClust3D variants, truth if synthetic).

Metrics (each code on its own relocated set, and all on the events every compared run relocated = common set):
* centroid shift from the B4 start (east, north, down, km);
* spread and orientation: PCA of the relocated cloud (sigma1 >= sigma2 >= sigma3 km; strike and dip of the plane
  normal to the smallest axis); depth median, 10-90 % range;
* event-pair distances: all pairs of the common set; |d_A - d_B| between runs; for synthetic cases |d - d_true|;
* relative position differences between runs after removing the common-set centroid offset;
* uncertainties: GrowClust3D bootstrap (MAD and std, h and z), tomoDD LSQR errors, tomoDD bootstrap std (if run);
* synthetic: absolute and centroid-removed errors against the truth, plane recovered against the true plane;
* dt misfit: residuals of the case's differential times at each solution's hypocentres, computed in the B4 grids
  (GrowClust3D's forward model; tomoDD's own forward differs), per-pair median removed (origin-time term).
Outputs: data/tomo_v1/b5_test/<case>/ (metrics.json, summary tables .csv, figures .png) and a manifest.

  python b5_compare.py CASE
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from b5_common import B5, KM_DEG, REPO, Grids, case_dir, data_dir, eft_depth, enu, lambert_km, load_case, pca, \
    write_manifest

RUNS = [("tomodd", "hier"), ("tomodd", "dd"), ("growclust", "catalog"), ("growclust", "default")]
LABEL = {"tomodd/hier": "tomoDD abs+DD", "tomodd/dd": "tomoDD DD only", "growclust/catalog": "GrowClust3D catalog",
         "growclust/default": "GrowClust3D default", "b4": "B4 start", "truth": "truth"}
COLOR = {"tomodd/hier": "#1f5fa8", "tomodd/dd": "#6aa6e0", "growclust/catalog": "#c2410c",
         "growclust/default": "#f59e0b", "b4": "#9ca3af", "truth": "#111827"}


def boot_std(case, code, var, ids):
    """Per-event std (h, z) of bootstrap positions; tomoDD from its _bNNN runs, GrowClust from boot.npz."""
    lat0 = None
    if code == "growclust":
        f = case_dir(case) / "growclust" / var / "boot.npz"
        if not f.exists():
            return None
        b = np.load(f)
        idx = pd.Index(b["ID"]).get_indexer(ids)
        la, lo, z = b["lat"][idx], b["lon"][idx], b["depth"][idx]
    else:
        runs = sorted((case_dir(case) / "tomodd").glob(f"{var}_b[0-9][0-9][0-9]/reloc.csv"))
        if len(runs) < 5:
            return None
        R = [pd.read_csv(r).set_index("ID").reindex(ids) for r in runs]
        la = np.stack([r.lat.values for r in R], 1)
        lo = np.stack([r.lon.values for r in R], 1)
        z = np.stack([r.depth.values for r in R], 1)
    lat0 = np.nanmean(la)
    e = (lo - np.nanmean(lo, 1, keepdims=True)) * KM_DEG * np.cos(np.radians(lat0))
    n = (la - np.nanmean(la, 1, keepdims=True)) * KM_DEG
    sh = np.sqrt(np.nanmean(e ** 2 + n ** 2, 1))
    sz = np.nanstd(z, 1)
    nb = np.sum(np.isfinite(z), 1)
    return pd.DataFrame({"ID": ids, "boot_sh": sh, "boot_sz": sz, "boot_n": nb})


def dt_misfit(dt, P, ids):
    """rms and MAD (s) of the dt residuals at hypocentres P (lat, lon, depth by ID) for pairs within ids."""
    G = Grids()
    d = dt[dt.id1.isin(ids) & dt.id2.isin(ids)].copy()
    x, y = lambert_km(P.lon.values, P.lat.values)
    q = pd.DataFrame({"x": x, "y": y, "zf": eft_depth(P.depth.values)}, index=P.index)
    a, b = q.loc[d.id1], q.loc[d.id2]
    t1, t2 = np.zeros(len(d)), np.zeros(len(d))
    for (sta, ph), g in d.groupby(["station", "phase"]):
        i = d.index.get_indexer(g.index)
        t1[i] = G.time(ph, sta, a.x.values[i], a.y.values[i], a.zf.values[i])
        t2[i] = G.time(ph, sta, b.x.values[i], b.y.values[i], b.zf.values[i])
    d["r"] = (d.tt1 - d.tt2) - (t1 - t2)
    d["r"] = d.r - d.groupby(["id1", "id2"]).r.transform("median")
    out = {}
    for ph in ("P", "S"):
        r = d.r[d.phase == ph]
        out[f"rms_{ph}"] = float(np.sqrt(np.mean(r ** 2)))
        out[f"mad_{ph}"] = float(np.median(np.abs(r)))
    out["n_dt"] = len(d)
    return out


def pair_d(X):
    i, j = np.triu_indices(len(X), 1)
    return np.linalg.norm(X[i] - X[j], axis=1), i, j


def main():
    case = sys.argv[1]
    ev, pk, st, dt = load_case(case)
    summ = json.loads((data_dir(case) / "summary.json").read_text())
    synthetic = bool(summ.get("synthetic"))
    od = REPO / "data" / "tomo_v1" / "b5_test" / case
    od.mkdir(parents=True, exist_ok=True)
    lat0, lon0 = float(ev.lat.median()), float(ev.lon.median())
    pos = {"b4": ev.set_index("ID")[["lat", "lon", "depth"]]}
    rel, info = {}, {}
    for code, var in RUNS:
        f = case_dir(case) / code / var / "reloc.csv"
        if not f.exists():
            continue
        k = f"{code}/{var}"
        r = pd.read_csv(f).set_index("ID")
        rel[k] = r
        pos[k] = r.loc[r.relocated.astype(bool), ["lat", "lon", "depth"]]
        info[k] = json.loads((case_dir(case) / code / var / "run.json").read_text())
    if synthetic:
        pos["truth"] = pd.read_csv(data_dir(case) / "truth.csv").set_index("ID")[["lat", "lon", "depth"]]
    runs = [k for k in pos if k not in ("b4", "truth")]
    common = pos["b4"].index
    for k in runs:
        common = common.intersection(pos[k].index)
    X = {k: enu(v.lat, v.lon, v.depth, lat0, lon0) for k, v in pos.items()}
    Xc = {k: enu(v.loc[common].lat, v.loc[common].lon, v.loc[common].depth, lat0, lon0) for k, v in pos.items()}
    M = {"case": case, "synthetic": synthetic, "events": len(ev), "events_linked": summ["events_linked"],
         "dt_pairs": summ["dt_pairs"], "dt_data": summ["dt_data"], "common": len(common), "runs": {}}
    rows = []
    for k in ["b4"] + runs + (["truth"] if synthetic else []):
        for subset, XX in (("own", X[k]), ("common", Xc[k])):
            if len(XX) < 4:
                continue
            sg, ax, strike, dip = pca(XX)
            if k == "truth":
                dc = np.full(3, np.nan)
            elif subset == "common":
                dc = XX.mean(0) - Xc["b4"].mean(0)
            else:                                         # B4 start of the same events
                b = pos["b4"].loc[pos[k].index]
                dc = XX.mean(0) - enu(b.lat, b.lon, b.depth, lat0, lon0).mean(0)
            rows.append(dict(run=k, subset=subset, n=len(XX), sigma1=sg[0], sigma2=sg[1], sigma3=sg[2],
                             strike=strike, dip=dip, depth_med=np.median(XX[:, 2]),
                             depth_p10=np.percentile(XX[:, 2], 10), depth_p90=np.percentile(XX[:, 2], 90),
                             shift_e=dc[0], shift_n=dc[1], shift_z=dc[2]))
    shape = pd.DataFrame(rows)
    shape.round(3).to_csv(od / "shape.csv", index=False)
    # pair distances (common set)
    d = {k: pair_d(v)[0] for k, v in Xc.items()}
    prow = []
    keys = ["b4"] + runs + (["truth"] if synthetic else [])
    for a, b in combinations(keys, 2):
        dd = np.abs(d[a] - d[b])
        prow.append(dict(a=a, b=b, pairs=len(dd), absdiff_med=np.median(dd), absdiff_p90=np.percentile(dd, 90),
                         frac_lt_1km=np.mean(dd < 1.0), corr=np.corrcoef(d[a], d[b])[0, 1]))
    pairs = pd.DataFrame(prow)
    pairs.round(3).to_csv(od / "pair_distances.csv", index=False)
    # relative positions between runs (centroid offset removed), common set
    rrow = []
    for a, b in combinations(keys, 2):
        A, B = Xc[a] - Xc[a].mean(0), Xc[b] - Xc[b].mean(0)
        dh = np.hypot(*(A[:, :2] - B[:, :2]).T)
        dz = np.abs(A[:, 2] - B[:, 2])
        rrow.append(dict(a=a, b=b, n=len(A), dh_med=np.median(dh), dh_p90=np.percentile(dh, 90),
                         dz_med=np.median(dz), dz_p90=np.percentile(dz, 90),
                         offset_e=Xc[a].mean(0)[0] - Xc[b].mean(0)[0], offset_n=Xc[a].mean(0)[1] - Xc[b].mean(0)[1],
                         offset_z=Xc[a].mean(0)[2] - Xc[b].mean(0)[2]))
    relpos = pd.DataFrame(rrow)
    relpos.round(3).to_csv(od / "relative_positions.csv", index=False)
    # uncertainties
    urow = []
    for k in runs:
        code, var = k.split("/")
        r = rel[k].loc[pos[k].index]
        u = dict(run=k, n=len(r))
        if code == "growclust":
            u.update(gc_mad_h_med=r.boot_mad_h.median(), gc_mad_z_med=r.boot_mad_z.median())
        else:
            u.update(lsqr_eh_med=np.hypot(r.ex, r.ey).median(), lsqr_ez_med=r.ez.median())
        bs = boot_std(case, code, var, pos[k].index)
        if bs is not None:
            u.update(boot_sh_med=bs.boot_sh.median(), boot_sz_med=bs.boot_sz.median(),
                     boot_sh_p90=bs.boot_sh.quantile(0.9), boot_sz_p90=bs.boot_sz.quantile(0.9))
            bs.round(4).to_csv(od / f"boot_{code}_{var}.csv", index=False)
        urow.append(u)
    unc = pd.DataFrame(urow)
    unc.round(3).to_csv(od / "uncertainty.csv", index=False)
    mis = pd.DataFrame([dict(run=k, **dt_misfit(dt, pos[k], common)) for k in keys])
    mis.round(4).to_csv(od / "dt_misfit.csv", index=False)
    M["dt_misfit"] = mis.round(4).to_dict(orient="records")
    # synthetic: errors against the truth
    if synthetic:
        T = pos["truth"]
        erow = []
        for k in ["b4"] + runs:
            idx = pos[k].index.intersection(T.index)
            for subset, ii in (("own", idx), ("common", common)):
                A = enu(pos[k].loc[ii].lat, pos[k].loc[ii].lon, pos[k].loc[ii].depth, lat0, lon0)
                B = enu(T.loc[ii].lat, T.loc[ii].lon, T.loc[ii].depth, lat0, lon0)
                off = A.mean(0) - B.mean(0)
                Ar = A - A.mean(0)
                Br = B - B.mean(0)
                dpa, dpb = pair_d(A)[0], pair_d(B)[0]
                sg, ax, strike, dip = pca(A)
                # distance of the relocated events from the true plane (normal of the truth)
                nrm = np.array([np.cos(np.radians(B5["synth"]["strike_deg"])), -np.sin(np.radians(B5["synth"]["strike_deg"])), 0.0])
                off_plane = (A - B.mean(0)) @ nrm
                erow.append(dict(run=k, subset=subset, n=len(ii),
                                 abs_h_med=np.median(np.hypot(*(A[:, :2] - B[:, :2]).T)),
                                 abs_z_med=np.median(np.abs(A[:, 2] - B[:, 2])),
                                 rel_h_med=np.median(np.hypot(*(Ar[:, :2] - Br[:, :2]).T)),
                                 rel_h_p90=np.percentile(np.hypot(*(Ar[:, :2] - Br[:, :2]).T), 90),
                                 rel_z_med=np.median(np.abs(Ar[:, 2] - Br[:, 2])),
                                 rel_z_p90=np.percentile(np.abs(Ar[:, 2] - Br[:, 2]), 90),
                                 centroid_e=off[0], centroid_n=off[1], centroid_z=off[2],
                                 pair_err_med=np.median(np.abs(dpa - dpb)), pair_err_p90=np.percentile(np.abs(dpa - dpb), 90),
                                 pair_corr=np.corrcoef(dpa, dpb)[0, 1],
                                 plane_rms_km=np.sqrt(np.mean((off_plane - off_plane.mean()) ** 2)),
                                 sigma1=sg[0], sigma2=sg[1], sigma3=sg[2], strike=strike, dip=dip))
        err = pd.DataFrame(erow)
        err.round(3).to_csv(od / "synthetic_errors.csv", index=False)
        M["synthetic_errors"] = err.round(3).to_dict(orient="records")
    M["shape"] = shape.round(3).to_dict(orient="records")
    M["pairs"] = pairs.round(3).to_dict(orient="records")
    M["relative_positions"] = relpos.round(3).to_dict(orient="records")
    M["uncertainty"] = unc.round(3).to_dict(orient="records")
    M["runs"] = info
    (od / "metrics.json").write_text(json.dumps(M, indent=1, default=float))
    figs = make_figs(case, od, X, Xc, d, keys, runs, synthetic, common)
    write_manifest(f"b5_compare_{case}", od, inputs=[data_dir(case) / "dt.csv"] +
                   [case_dir(case) / k / "reloc.csv" for k in runs],
                   outputs=[od / f for f in ("metrics.json", "shape.csv", "pair_distances.csv", "relative_positions.csv",
                                             "uncertainty.csv")] + figs,
                   counts={"common": len(common), "runs": runs})
    print(shape.round(2).to_string())
    print(pairs.round(3).to_string())
    print(relpos.round(2).to_string())
    print(unc.round(3).to_string())
    print(mis.round(4).to_string())
    if synthetic:
        print(err.round(2).to_string())


def make_figs(case, od, X, Xc, d, keys, runs, synthetic, common):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strike = np.radians(B5["synth"]["strike_deg"])
    u = np.array([np.sin(strike), np.cos(strike)])            # along strike (east, north)
    v = np.array([np.cos(strike), -np.sin(strike)])           # across strike
    show = ["b4"] + runs + (["truth"] if synthetic else [])
    # 1. map and sections, each run on its own relocated set
    fig, ax = plt.subplots(3, len(show), figsize=(2.6 * len(show), 7.6), squeeze=False)
    allX = np.vstack([X[k] for k in show])
    lims = [(allX[:, 0], allX[:, 1]), (allX[:, :2] @ u, allX[:, 2]), (allX[:, :2] @ v, allX[:, 2])]
    lims = [((np.percentile(a, 0.5) - 1, np.percentile(a, 99.5) + 1), (np.percentile(b, 0.5) - 1, np.percentile(b, 99.5) + 1))
            for a, b in lims]
    for c, k in enumerate(show):
        XX = X[k]
        col = COLOR[k]
        a_ = XX[:, :2] @ u
        c_ = XX[:, :2] @ v
        ax[0, c].scatter(XX[:, 0], XX[:, 1], s=3, c=col, lw=0)
        ax[1, c].scatter(a_, XX[:, 2], s=3, c=col, lw=0)
        ax[2, c].scatter(c_, XX[:, 2], s=3, c=col, lw=0)
        ax[0, c].set_title(f"{LABEL[k]}\n(n={len(XX)})", fontsize=8)
        for r in range(3):
            ax[r, c].tick_params(labelsize=7)
            ax[r, c].set_xlim(*lims[r][0])
            ax[r, c].set_ylim(*(lims[r][1][::-1] if r else lims[r][1]))
            ax[r, c].set_aspect("equal", adjustable="box")
    ax[0, 0].set_ylabel("north (km)", fontsize=8)
    ax[1, 0].set_ylabel("depth (km bsl)\nalong strike 111°", fontsize=8)
    ax[2, 0].set_ylabel("depth (km bsl)\nacross strike", fontsize=8)
    for c in range(len(show)):
        ax[0, c].set_xlabel("east (km)", fontsize=7)
        ax[1, c].set_xlabel("along strike (km)", fontsize=7)
        ax[2, c].set_xlabel("across strike (km)", fontsize=7)
    fig.suptitle(f"{case}: start and relocations (own relocated sets)", fontsize=9)
    fig.tight_layout()
    f1 = od / "sections.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    # 2. depth histograms (common set) and pair distances against the reference
    ref = "truth" if synthetic else "tomodd/hier"
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = np.arange(-1, 61, 2)
    for k in show:
        ax[0].hist(Xc[k][:, 2], bins=bins, histtype="step", color=COLOR[k], label=LABEL[k], lw=1.3)
    ax[0].set_xlabel("depth (km bsl), common set", fontsize=8)
    ax[0].set_ylabel("events", fontsize=8)
    ax[0].legend(fontsize=7)
    for k in show:
        if k == ref or ref not in d:
            continue
        ax[1].scatter(d[ref], d[k], s=1, c=COLOR[k], alpha=0.4, lw=0, label=LABEL[k])
    if ref in d:
        m = max(d[ref].max(), 1)
        ax[1].plot([0, m], [0, m], color="0.3", lw=0.8)
        ax[1].set_xlabel(f"pair distance, {LABEL[ref]} (km)", fontsize=8)
        ax[1].set_ylabel("pair distance, other run (km)", fontsize=8)
        ax[1].legend(fontsize=7, markerscale=6)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.suptitle(f"{case}: depth distribution and event-pair distances ({len(common)} events in every run)", fontsize=9)
    fig.tight_layout()
    f2 = od / "depth_pairs.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    return [f1, f2]


if __name__ == "__main__":
    main()
