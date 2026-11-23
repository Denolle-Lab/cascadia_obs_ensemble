"""Track B Bayesian ensemble (D-B030): calibration of a synthetic ensemble against its known truth.

  python b2_ens_coverage.py ENS [--tag pilot] [--members m000-m019]

Nodes (interior, Vp, Vs, Vp/Vs): ensemble 68% and 95% intervals, two ways: empirical quantiles (16-84, 2.5-97.5%)
and Gaussian (mean +- 1 / 1.96 std; less noisy for small ensembles). Coverage = fraction of mask nodes whose truth lies
inside. Masks:
  * provisional: DWS (P for Vp, S for Vs, both for Vp/Vs) above the run threshold (params tomodd.inp.thres) in >= 90%
    of the members
  * dws100: same with DWS >= 100 (a stricter, illustrative cut)
  * chk100 / chk60: checkerboard masks of the parallel resolution tests (data/tomo_v1/b2_res_tests/
    resolution_mask.npz, b2_res_checker.py; D-B018 corr >= 0.7 and amp >= 0.5) at any cell size <= 100 km / <= 60 km
Hypocentres: per event and coordinate (east, north, depth, km from the truth = baseline relocation), same intervals,
for events relocated by >= 90% of the members; also by v4-NLL3D tier and depth class.
Noise floor: members nf0..nf4 and their parent (identical draws, 1e-6 s data perturbation): std per node and event.
Writes data/tomo_v1/b2_ensemble/<ENS>_<tag>_*.csv/.md/.png and <ens>/coverage_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_ens_common import BASE, FIGS, INPUTS, T, ens_dir, read_reloc  # noqa: E402

INNER = (slice(1, -1), slice(1, -1), slice(1, -1))


def load_members(E, names):
    mods, dws, rel, ok = [], [], [], []
    for n in names:
        md = E / n
        if not (md / "DONE").exists():
            continue
        g = np.load(md / "run" / "Output_Files" / "model_geo.npz")
        d = np.load(md / "dws.npz")
        mods.append((g["vp"][INNER], g["vs"][INNER]))
        dws.append((d["p"][INNER], d["s"][INNER]))
        rel.append(read_reloc(md / "run" / "Output_Files" / "tomoDD.reloc_geo"))
        ok.append(n)
    return ok, mods, dws, rel


def intervals(X, truth):
    """X [n_members, ...]; returns dict of boolean arrays (inside) and stats."""
    q = np.nanpercentile(X, [2.5, 16, 84, 97.5], axis=0)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1)
    return dict(e68=(truth >= q[1]) & (truth <= q[2]), e95=(truth >= q[0]) & (truth <= q[3]),
                g68=np.abs(truth - mu) <= sd, g95=np.abs(truth - mu) <= 1.96 * sd, mu=mu, sd=sd,
                z=(mu - truth) / np.where(sd > 0, sd, np.nan))


def cov_row(I, m, label):
    n = int(m.sum())
    if n == 0:
        return dict(group=label, n=0)
    return dict(group=label, n=n, e68=I["e68"][m].mean(), e95=I["e95"][m].mean(), g68=I["g68"][m].mean(),
                g95=I["g95"][m].mean(), rms_z=float(np.sqrt(np.nanmean(I["z"][m] ** 2))))


def checker_mask(shape):
    """Parallel checkerboard tests (b2_res_checker.py analyse): finest resolved cell (km, 0 = unresolved) per node.
    Returns {name: (mask_vp, mask_vs)}: chk100 = resolved at any tested cell size (<= 100 km), chk60 = <= 60 km."""
    f = FIGS.parent / "b2_res_tests" / "resolution_mask.npz"
    if not f.exists():
        return {}, None
    z = np.load(f)

    def fit(a):
        if a.shape == shape:
            return a
        if a.shape == tuple(s + 2 for s in shape):
            return a[INNER]
        raise ValueError(f"mask shape {a.shape} vs {shape}")
    fp, fs = fit(z["finest_p"]), fit(z["finest_s"])
    return {"chk100": (fp > 0, fs > 0), "chk60": ((fp > 0) & (fp <= 60), (fs > 0) & (fs <= 60))}, str(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ens")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--members", default="0-99")
    a = ap.parse_args()
    E = ens_dir(a.ens)
    lo, hi = (int(v) for v in a.members.split("-"))
    names = [f"m{i:03d}" for i in range(lo, hi + 1)]
    ok, mods, dws, rel = load_members(E, names)
    N = len(ok)
    print(f"{a.ens}: {N} members done of {len(names)}")
    tm = np.load(E / "truth" / "truth_model.npz")
    tvp, tvs = tm["vp"][INNER], tm["vs"][INNER]
    s = np.load(INPUTS / "start_model.npz")
    z = s["z"]
    VP = np.array([m[0] for m in mods])
    VS = np.array([m[1] for m in mods])
    DP = np.array([d[0] for d in dws])
    DS = np.array([d[1] for d in dws])
    thr = T["inp"]["thres"]
    masks = {"provisional": ((DP > thr).mean(0) >= 0.9, (DS > thr).mean(0) >= 0.9),
             "dws100": ((DP >= 100).mean(0) >= 0.9, (DS >= 100).mean(0) >= 0.9)}
    cm, cpath = checker_mask(tvp.shape)
    masks.update(cm)
    quant = {"Vp": (VP, tvp, 0), "Vs": (VS, tvs, 1), "VpVs": (VP / VS, tvp / tvs, 2)}
    rows = []
    summary = dict(ens=a.ens, tag=a.tag, members=N, checker_mask=cpath, nodes={}, hypo={})
    for mk, (mp, ms) in masks.items():
        for q, (X, tr, kind) in quant.items():
            mask = mp if kind == 0 else ms if kind == 1 else (mp & ms)
            I = intervals(X, tr)
            r = cov_row(I, mask, "all")
            summary["nodes"][f"{mk}_{q}"] = r
            rows.append(dict(mask=mk, q=q, depth="all", **{k: v for k, v in r.items() if k != "group"}))
            for k, zk in enumerate(z):
                mm = np.zeros_like(mask)
                mm[:, :, k] = mask[:, :, k]
                rr = cov_row(I, mm, f"{zk:g}")
                if rr["n"]:
                    rows.append(dict(mask=mk, q=q, depth=f"{zk:g}", **{k2: v for k2, v in rr.items() if k2 != "group"},
                                     sd_mean=float(np.mean(I["sd"][mm] / (tr[mm] if q != "VpVs" else 1.0)))))
    nodes = pd.DataFrame(rows)
    FIGS.mkdir(parents=True, exist_ok=True)
    pre = FIGS / f"{a.ens}_{a.tag}"
    nodes.round(4).to_csv(f"{pre}_nodes.csv", index=False)
    # spread: ensemble std (%) on provisional nodes by depth, and error of the mean
    sp = []
    for k, zk in enumerate(z):
        m = masks["provisional"][0][:, :, k]
        if m.sum() == 0:
            continue
        sp.append(dict(depth=zk, n_vp=int(m.sum()), sd_vp_pct=float(np.median(100 * VP[:, m].std(0, ddof=1) / tvp[:, :, k][m])),
                       sd_vs_pct=float(np.median(100 * VS[:, m].std(0, ddof=1) / tvs[:, :, k][m])),
                       sd_vpvs=float(np.median((VP / VS)[:, :, :, k][:, m].std(0, ddof=1))),
                       err_vp_pct=float(np.median(100 * np.abs(VP[:, m].mean(0) / tvp[:, :, k][m] - 1))),
                       true_rf_vp_pct=float(np.median(100 * np.abs(tm["fp"][:, :, k][m])))))
    spread = pd.DataFrame(sp)
    spread.round(4).to_csv(f"{pre}_spread.csv", index=False)
    # hypocentres
    th = pd.read_csv(E / "truth" / "truth_hypo.csv").set_index("ID")
    ids = th.index.values
    H = np.full((N, len(ids), 3), np.nan)
    for i, r in enumerate(rel):
        rr = r.reindex(ids)
        H[i, :, 0] = (rr.lon.values - th.lon.values) * 111.19 * np.cos(np.radians(th.lat.values))
        H[i, :, 1] = (rr.lat.values - th.lat.values) * 111.19
        H[i, :, 2] = rr.depth.values - th.depth.values
    present = np.isfinite(H[:, :, 0]).mean(0) >= 0.9
    ev = pd.read_csv(INPUTS / "events.csv").set_index("ID").reindex(ids)
    hrows = []
    for c, lab in enumerate(("east", "north", "depth")):
        I = intervals(H[:, :, c], np.zeros(len(ids)))
        for grp, m in [("all", present)] + [(f"tier {t}", present & (ev.tier.values == t)) for t in ("A", "B", "C")] + \
                [(f"z {a_}-{b_} km", present & (th.depth.values >= a_) & (th.depth.values < b_))
                 for a_, b_ in ((-5, 10), (10, 20), (20, 35), (35, 100))]:
            r = cov_row(I, m, grp)
            if r["n"]:
                r.update(coord=lab, sd_median_km=float(np.median(I["sd"][m])), err_median_km=float(np.median(np.abs(I["mu"][m]))))
                hrows.append(r)
    hyp = pd.DataFrame(hrows)
    hyp.round(4).to_csv(f"{pre}_hypo.csv", index=False)
    summary["hypo"] = hyp[hyp.group == "all"].round(4).to_dict("records")
    summary["events_present"] = int(present.sum())
    # noise floor
    nf = [n for n in ["nf0", "nf1", "nf2", "nf3", "nf4"] if (E / n / "DONE").exists()]
    if nf:
        par = json.loads((E / nf[0] / "inputs" / "draws.json").read_text())["seed"]
        okf, modf, _, relf = load_members(E, [f"m{par:03d}"] + nf)
        Vf = np.array([m[0] for m in modf])
        Sf = np.array([m[1] for m in modf])
        mprov = masks["provisional"][0]
        sdp = 100 * Vf.std(0, ddof=1) / Vf.mean(0)
        sds = 100 * Sf.std(0, ddof=1) / Sf.mean(0)
        Hf = np.full((len(relf), len(ids), 3), np.nan)
        for i, r in enumerate(relf):
            rr = r.reindex(ids)
            Hf[i, :, 0] = (rr.lon.values - th.lon.values) * 111.19 * np.cos(np.radians(th.lat.values))
            Hf[i, :, 1] = (rr.lat.values - th.lat.values) * 111.19
            Hf[i, :, 2] = rr.depth.values - th.depth.values
        hsd = np.nanstd(Hf, axis=0, ddof=1)
        ens_sd_vp = 100 * VP.std(0, ddof=1) / VP.mean(0)
        hsd_ens = np.nanstd(H, axis=0, ddof=1)
        summary["noise_floor"] = dict(
            runs=okf, vp_sd_pct_median=float(np.median(sdp[mprov])), vp_sd_pct_p95=float(np.percentile(sdp[mprov], 95)),
            vs_sd_pct_median=float(np.median(sds[masks["provisional"][1]])),
            vs_sd_pct_p95=float(np.percentile(sds[masks["provisional"][1]], 95)),
            ratio_to_ensemble_vp_median=float(np.median(sdp[mprov] / ens_sd_vp[mprov])),
            hypo_sd_km_median={c: float(np.nanmedian(hsd[:, k])) for k, c in enumerate(("east", "north", "depth"))},
            hypo_sd_km_p95={c: float(np.nanpercentile(hsd[:, k], 95)) for k, c in enumerate(("east", "north", "depth"))},
            hypo_ratio_to_ensemble_median={c: float(np.nanmedian(hsd[:, k] / hsd_ens[:, k])) for k, c in enumerate(("east", "north", "depth"))})
    # run statistics
    rl = [json.loads((E / n / "DONE").read_text()) for n in ok + nf]
    wall = np.array([r["wall_min"] for r in rl])
    rss = np.array([float(r["time"].split()[-2]) / 1e6 for r in rl])
    du = [sum(p.stat().st_size for p in (E / n).rglob("*") if p.is_file()) / 1e6 for n in ok + nf]
    summary["runs"] = dict(n=len(rl), wall_min_median=float(np.median(wall)), wall_min_max=float(wall.max()),
                           rss_gb_max=float(rss.max()), rss_gb_median=float(np.median(rss)), mb_per_member=float(np.median(du)),
                           rms_abs_final=[float(np.min([r["rms_abs_final"] for r in rl])), float(np.max([r["rms_abs_final"] for r in rl]))])
    summary["spread_by_depth"] = spread.round(3).to_dict("records")
    (E / f"coverage_{a.tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    (FIGS / f"{a.ens}_{a.tag}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(nodes[nodes.depth == "all"].round(3).to_string(index=False))
        print(nodes[(nodes["mask"] == "provisional")].round(3).to_string(index=False))
        print(spread.round(3).to_string(index=False))
        print(hyp.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("noise_floor", "runs", "events_present") if k in summary}, indent=1, default=float))
    figures(pre, nodes, spread, hyp, VP, tvp, masks, z, s, a)


def figures(pre, nodes, spread, hyp, VP, tvp, masks, z, s, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    for ax, q in zip(axs, ("Vp", "Vs", "VpVs")):
        d = nodes[(nodes["mask"] == "provisional") & (nodes.q == q) & (nodes.depth != "all")]
        zz = d.depth.astype(float)
        ax.axvspan(0.60, 0.76, color="0.85")
        ax.axvspan(0.90, 0.98, color="0.92")
        ax.plot(d.e68, zz, "o-", color="#1f5f99", label="68% (quantile)")
        ax.plot(d.e95, zz, "s-", color="#b3541e", label="95% (quantile)")
        ax.plot(d.g68, zz, "o--", color="#1f5f99", mfc="none", label="68% (Gaussian)")
        ax.plot(d.g95, zz, "s--", color="#b3541e", mfc="none", label="95% (Gaussian)")
        ax.set_xlim(0, 1)
        ax.set_title(f"{q}: truth inside interval", fontsize=9)
        ax.set_xlabel("fraction of resolved nodes")
    axs[0].invert_yaxis()
    axs[0].set_ylabel("node depth (km)")
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{a.ens} ({a.tag}): coverage by depth, provisional DWS mask; grey = target bands", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{pre}_coverage_depth.png", dpi=150)
    plt.close(fig)
    ks = [k for k, zk in enumerate(z) if zk in (7, 15, 25, 40)]
    fig, axs = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 6.2))
    sd = 100 * VP.std(0, ddof=1) / VP.mean(0)
    err = 100 * (VP.mean(0) / tvp - 1)
    for c, k in enumerate(ks):
        m = masks["provisional"][0][:, :, k]
        for r, (arr, lab, cmap, lim) in enumerate(((sd, "ensemble std Vp (%)", "viridis", (0, 3)),
                                                   (err, "mean - truth Vp (%)", "RdBu_r", (-4, 4)))):
            ax = axs[r, c]
            im = ax.pcolormesh(s["lon"], s["lat"], np.where(m, arr[:, :, k], np.nan).T, cmap=cmap, vmin=lim[0], vmax=lim[1],
                               shading="nearest")
            ax.set_title(f"{z[k]:g} km: {lab}", fontsize=8)
            ax.set_aspect(1 / np.cos(np.radians(41.3)))
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(f"{pre}_maps_vp.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
