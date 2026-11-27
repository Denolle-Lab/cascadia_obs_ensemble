#!/usr/bin/env python3
"""Circularity control of the structure contrasts (D-B111), with the rule pre-registered in
data/paper_plan/plate_boundaries/RESUME.md (2026-11-27 09:38 PST, commit 6b64abf1) before this script was run.

For every contrast of pb_structure.py (same node sets, contrast_nodes.npz; depth-matched):
* D_real, sigma_ens (real ensemble x factor: Vp 1.50, Vs 1.35, Vp/Vs 1.00) from structure_contrasts.csv;
* D_truth in the synthetic truth (v0.4.2 tomography x (1 + f), synth/truth/truth_model.npz) and D_rec,i in each
  synthetic member s000-s099 (merged vp, vs before the D-B073 node rule, as in the calibration);
  spurious s_i = D_rec,i - D_truth; S95 = 95th percentile of |s_i|;
* rule: survives iff |D_real| >= 2 sigma_ens and |D_real| >= S95 (fewer than 50 members with a value or no truth
  value: not testable, does not survive);
* reported: mean and sd of s_i, P05 / P95 of s_i, recovery ratio mean(D_rec) / D_truth, the noise-floor clones'
  spread (nf0-nf4 with s000), and the old rule (2 sigma_ens alone).

Null truth: the synthetic set holds one truth carrying the v0.4.2 contrasts; no member inverts a truth without them,
so a null-truth test is not possible with these members (stated in the output, not approximated).

Writes data/paper_plan/plate_boundaries/circularity_contrasts.csv and the figure fig_circularity.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402
from pb_structure import depth_matched  # noqa: E402

MIN_MEMBERS = 50


def contrast(vals, layer, i1, i2, n):
    g1 = np.zeros(n, bool)
    g2 = np.zeros(n, bool)
    g1[i1], g2[i2] = True, True
    d, _ = depth_matched(vals, layer, g1, g2)
    return d


def main():
    nodes = np.load(P.BIG / "contrast_nodes.npz")
    layer = nodes["layer"]
    n = layer.size
    ct = pd.read_csv(P.OUT / "structure_contrasts.csv")
    ct = ct[ct.value.notna()].copy()
    tr, names, sm = P.load_synth()
    P.log(f"synthetic members: {len(names)}")
    nf_names = ["s000"] + sorted(p.name for p in P.SYNTH.iterdir() if p.name.startswith("nf")
                                 and (p / "merged.npz").exists())
    nfv = {}
    for v in ("Vp", "Vs"):
        nfv[v] = []
    for nm in nf_names:
        z = np.load(P.SYNTH / nm / "merged.npz")
        nfv["Vp"].append(np.transpose(z["vp"], (2, 1, 0)))
        nfv["Vs"].append(np.transpose(z["vs"], (2, 1, 0)))
    nfv = {k: np.stack(v).astype(float) for k, v in nfv.items()}
    nfv["VpVs"] = nfv["Vp"] / nfv["Vs"]
    trf = {k: v.reshape(1, -1) for k, v in tr.items()}
    smf = {k: v.reshape(len(names), -1) for k, v in sm.items()}
    nff = {k: v.reshape(len(nf_names), -1) for k, v in nfv.items()}
    rows = []
    for _, r in ct.iterrows():
        key = f"{r.seg}|{r['var']}|{r.contrast}"
        i1, i2 = nodes[key + "|1"], nodes[key + "|2"]
        dt = contrast(trf[r["var"]], layer, i1, i2, n)
        dr = contrast(smf[r["var"]], layer, i1, i2, n)
        dn = contrast(nff[r["var"]], layer, i1, i2, n)
        row = dict(seg=r.seg, var=r["var"], contrast=r.contrast, n1=int(r.n1), n2=int(r.n2), D_real=r.value,
                   sigma_ens=r.sigma_members, ratio_ens=r.ratio)
        dtv = float(dt[0]) if dt is not None else np.nan
        drv = dr[np.isfinite(dr)] if dr is not None else np.array([])
        row["D_truth"] = dtv
        row["n_synth"] = int(len(drv))
        if np.isfinite(dtv) and len(drv) >= MIN_MEMBERS:
            s = drv - dtv
            a = np.abs(s)
            row.update(spur_mean=float(s.mean()), spur_sd=float(s.std(ddof=1)), spur_p05=float(np.percentile(s, 5)),
                       spur_p95=float(np.percentile(s, 95)), S95=float(np.percentile(a, 95)),
                       spur_rms=float(np.sqrt(np.mean(s ** 2))), D_rec_mean=float(drv.mean()),
                       recovery=float(drv.mean() / dtv) if abs(dtv) > 1e-6 else np.nan,
                       bias_same_sign=bool(np.sign(s.mean()) == np.sign(r.value)))
            row["testable"] = True
            row["pass_ens"] = bool(abs(r.value) >= 2 * r.sigma_members)
            row["pass_synth"] = bool(abs(r.value) >= row["S95"])
            row["survives"] = bool(row["pass_ens"] and row["pass_synth"])
            row["margin_S95"] = float(abs(r.value) / row["S95"]) if row["S95"] > 0 else np.nan
        else:
            row.update(testable=False, pass_ens=bool(abs(r.value) >= 2 * r.sigma_members), pass_synth=False,
                       survives=False)
        if dn is not None and np.isfinite(dn).sum() >= 2:
            row["noise_floor_sd"] = float(np.nanstd(dn, ddof=1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["old_rule_2sigma"] = out.ratio_ens >= 2
    out.to_csv(P.OUT / "circularity_contrasts.csv", index=False, float_format="%.4g")
    P.log(out[["seg", "var", "contrast", "D_real", "sigma_ens", "D_truth", "spur_mean", "S95", "recovery",
               "pass_ens", "pass_synth", "survives"]].to_string())
    P.log(f"testable {int(out.testable.sum())} of {len(out)}; old rule {int(out.old_rule_2sigma.sum())}; "
          f"pass synth {int(out.pass_synth.sum())}; survive {int(out.survives.sum())}")
    fig_circ(out)


def fig_circ(out):
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
                if r.testable:
                    ax.plot([-r.S95, r.S95], [y, y], color="0.75", lw=4, solid_capstyle="butt", zorder=1)
                ax.errorbar(r.D_real, y, xerr=2 * r.sigma_ens, fmt=mk[cn], ms=4.5, color=c, lw=0.9, capsize=1.5,
                            mfc=c if r.survives else "white", zorder=3)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_title({"Vp": "Vp contrast (km/s)", "Vs": "Vs contrast (km/s)", "VpVs": "Vp/Vs contrast"}[v], fontsize=8)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axs[0].set_yticks(range(len(segs)))
    axs[0].set_yticklabels(segs[::-1])
    for cn in off:
        axs[2].plot([], [], mk[cn], color="0.3", mfc="white", label=cn)
    axs[2].plot([], [], color="0.75", lw=4, label="±S95 spurious (synthetic)")
    axs[2].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("DRAFT: real contrasts (v0.4.2) with ±2σ of the real ensemble (Vp x1.50, Vs x1.35, Vp/Vs x1.00) and the "
                 "95th percentile of |recovered − truth| over 100 synthetic members (grey).\nFilled = survives the "
                 "pre-registered rule (|Δ| ≥ 2σ and |Δ| ≥ S95).", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    (P.BIG / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(P.BIG / "figures" / "fig_circularity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
