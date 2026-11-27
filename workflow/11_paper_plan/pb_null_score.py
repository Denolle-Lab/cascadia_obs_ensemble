#!/usr/bin/env python3
"""Score the structure contrasts against the null-truth ensemble (D-B112), with the rule pre-registered in
data/paper_plan/plate_boundaries/RESUME.md (2026-11-27 10:13:48 PST, commit 6d45ad98) before the null truth was built.

For every contrast of pb_structure.py on the catalog of the PB_* variables (v6.1 for the paper; same node sets,
contrast_nodes.npz; depth-matched):
* D0_truth: the contrast in the null truth (ROOT/synth/truth/truth_model.npz of the null root);
* D0_rec,i: the contrast in null member i (merged vp, vs before the D-B073 node rule); n_i = D0_rec,i - D0_truth;
  N95 = 95th percentile of |n_i|; at least 15 members with a value, else not testable;
* rule: claimed iff the D-B111 rule holds (circularity_contrasts.csv: |D_real| >= 2 sigma_ens and >= S95) and
  |D_real| >= N95;
* reported: mean n_i and its sign against D_real; the variant with the 95th percentile of |D0_rec,i|; the paired
  difference calibration member minus null member with the same seed (s000-s019) and the truth difference.
Writes <PB_OUT>/null_truth_contrasts.csv and <PB_BIG>/figures/fig_null.png.
Environment: PB_NULL_ROOT (default /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null) besides the PB_* variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402
from pb_circularity import contrast  # noqa: E402

NULL_ROOT = Path(os.environ.get("PB_NULL_ROOT", "/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null"))
MIN_NULL = 15


def load_root(synth_dir, names):
    t = np.load(synth_dir / "truth" / "truth_model.npz")
    tr = {"Vp": np.transpose(t["vp"], (2, 1, 0)).astype(float), "Vs": np.transpose(t["vs"], (2, 1, 0)).astype(float)}
    tr["VpVs"] = tr["Vp"] / tr["Vs"]
    vp, vs = [], []
    for n in names:
        z = np.load(synth_dir / n / "merged.npz")
        vp.append(np.transpose(z["vp"], (2, 1, 0)))
        vs.append(np.transpose(z["vs"], (2, 1, 0)))
    vp, vs = np.stack(vp).astype(float), np.stack(vs).astype(float)
    return tr, {"Vp": vp, "Vs": vs, "VpVs": vp / vs}


def main():
    sd = NULL_ROOT / "synth"
    names = sorted(p.name for p in sd.iterdir() if p.name[0] == "s" and (p / "MERGED").exists())
    P.log(f"null members merged: {len(names)} ({NULL_ROOT})")
    nodes = np.load(P.BIG / "contrast_nodes.npz")
    layer = nodes["layer"]
    n = layer.size
    circ = pd.read_csv(P.OUT / "circularity_contrasts.csv")
    tr0, m0 = load_root(sd, names)
    trc, mc = load_root(P.SYNTH, names)                     # calibration members with the same seeds
    f = lambda d: {k: v.reshape(v.shape[0] if v.ndim == 4 else 1, -1) for k, v in d.items()}  # noqa: E731
    tr0f, m0f, trcf, mcf = f(tr0), f(m0), f(trc), f(mc)
    rows = []
    for _, r in circ.iterrows():
        key = f"{r.seg}|{r['var']}|{r.contrast}"
        i1, i2 = nodes[key + "|1"], nodes[key + "|2"]
        v = r["var"]
        dt0 = contrast(tr0f[v], layer, i1, i2, n)
        dr0 = contrast(m0f[v], layer, i1, i2, n)
        dtc = contrast(trcf[v], layer, i1, i2, n)
        drc = contrast(mcf[v], layer, i1, i2, n)
        row = dict(seg=r.seg, var=v, contrast=r.contrast, D_real=r.D_real, sigma_ens=r.sigma_ens, S95=r.get("S95"),
                   survives_DB111=bool(r.survives), n_null=0)
        d0t = float(dt0[0]) if dt0 is not None else np.nan
        row["D0_truth"] = d0t
        if dr0 is not None and np.isfinite(d0t):
            ok = np.isfinite(dr0)
            x = dr0[ok] - d0t
            row["n_null"] = int(ok.sum())
            if ok.sum() >= MIN_NULL:
                row.update(null_mean=float(x.mean()), null_sd=float(x.std(ddof=1)), null_p05=float(np.percentile(x, 5)),
                           null_p95=float(np.percentile(x, 95)), N95=float(np.percentile(np.abs(x), 95)),
                           N95_rec=float(np.percentile(np.abs(dr0[ok]), 95)),
                           null_bias_same_sign=bool(np.sign(x.mean()) == np.sign(r.D_real)))
                if drc is not None:
                    pair = drc - dr0
                    okp = np.isfinite(pair)
                    row.update(paired_cal_minus_null_mean=float(pair[okp].mean()),
                               paired_cal_minus_null_sd=float(pair[okp].std(ddof=1)),
                               truth_cal_minus_null=float(dtc[0] - d0t) if dtc is not None else np.nan)
        testable = row["n_null"] >= MIN_NULL and np.isfinite(row.get("N95", np.nan))
        row["testable_null"] = bool(testable)
        row["pass_null"] = bool(testable and abs(r.D_real) >= row["N95"])
        row["claimed"] = bool(row["survives_DB111"] and row["pass_null"])
        row["pass_null_variant_rec"] = bool(testable and abs(r.D_real) >= row.get("N95_rec", np.inf))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(P.OUT / "null_truth_contrasts.csv", index=False, float_format="%.4g")
    P.log(out[["seg", "var", "contrast", "D_real", "D0_truth", "null_mean", "N95", "S95", "survives_DB111",
               "pass_null", "claimed"]].to_string())
    P.log(f"members {len(names)}; testable {int(out.testable_null.sum())} of {len(out)}; D-B111 survivors "
          f"{int(out.survives_DB111.sum())}; pass null {int(out.pass_null.sum())}; claimed {int(out.claimed.sum())}")
    fig_null(out, len(names))


def fig_null(out, nmem):
    from pb_figures import TYPE_COL
    segs = [s for s in P.SEG_ORDER if s in set(out.seg)]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 5.6), sharey=True)
    off = {"near-far": 0.0, "across": 0.25, "along": -0.25}
    mk = {"near-far": "o", "across": "^", "along": "s"}
    for ax, v in zip(axs, ["Vp", "Vs", "VpVs"]):
        d = out[out["var"] == v]
        for i, sg in enumerate(segs):
            yy = len(segs) - 1 - i
            for cn in off:
                r = d[(d.seg == sg) & (d.contrast == cn)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = yy + off[cn]
                c = TYPE_COL[P.SEGMENTS[sg][1]]
                if r.testable_null:
                    ax.plot([-r.N95, r.N95], [y, y], color="0.72", lw=4, solid_capstyle="butt", zorder=1)
                ax.errorbar(r.D_real, y, xerr=2 * r.sigma_ens, fmt=mk[cn], ms=4.5, color=c, lw=0.9, capsize=1.5,
                            mfc=c if r.claimed else "white", zorder=3)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_title({"Vp": "Vp contrast (km/s)", "Vs": "Vs contrast (km/s)", "VpVs": "Vp/Vs contrast"}[v], fontsize=8)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axs[0].set_yticks(range(len(segs)))
    axs[0].set_yticklabels(segs[::-1])
    for cn in off:
        axs[2].plot([], [], mk[cn], color="0.3", mfc="white", label=cn)
    axs[2].plot([], [], color="0.72", lw=4, label="±N95 null truth")
    axs[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.suptitle(f"DRAFT: real contrasts ({P.CATLABEL} node sets) with ±2σ of the real ensemble and the 95th percentile of "
                 f"|recovered − truth| over {nmem} null-truth members (grey).\nFilled = claimed (D-B111 rule and "
                 "|Δ| ≥ N95, pre-registered 2026-11-27 10:13).", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    (P.BIG / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(P.BIG / "figures" / "fig_null.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
