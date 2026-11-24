"""Report of the B5 weighting test (b5_wtest.py): one table per criterion, variants = S weight x re-weighting.

Criteria, each per cluster then summarised over clusters (median; "wins" = clusters better than production):
* CV: held-out dt residual rms per phase, folds pooled (sqrt of n-weighted mean square), divided by the production
  variant's (s0.25_prod) value in the same cluster; also divided by B4's held-out rms (B4 used every station's absolute
  picks, so this ratio favours B4).
* synthetic: centroid-removed relative position error after / before (horizontal, vertical).
* in-sample: dt residual rms per phase at the final positions / at B4.
* side effects: events above the surface (earth_relief 01m), air quakes, share relocated.

  python b5_wtest_report.py   ->  <out>/summary/report.md, variant_scores.csv, per_cluster.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402

P = load_params()["b5_wtest"]
OUT = Path(P["out"])
REF = "s0.25_prod"


def load():
    rows = []
    for f in OUT.glob("*/s*/*/result.json"):
        r = json.loads(f.read_text())
        r["variant"] = f.parts[-3]
        rows.append(r)
    R = pd.json_normalize(rows)
    base = {f.parts[-3]: json.loads(f.read_text()) for f in OUT.glob("*/base/result.json")}
    return R, base


def pooled(g, col):
    n, r = g[f"{col}.n"], g[f"{col}.rms"]
    return float(np.sqrt((n * r ** 2).sum() / n.sum())) if n.sum() else np.nan


def main():
    R, base = load()
    sel = pd.read_csv(OUT / "clusters.csv").set_index("name")
    per = []
    for (c, v), g in R.groupby(["cluster", "variant"]):
        row = dict(cluster=c, variant=v, region=sel.region.get(c))
        cv = g[g.kind.str.startswith("fold")]
        if len(cv) == P["folds"]:
            for ph in ("P", "S"):
                row[f"cv_{ph}"] = pooled(cv, f"heldout.{ph}")
                row[f"cvb4_{ph}"] = pooled(cv, f"heldout_b4.{ph}")
        fu = g[g.kind == "full"]
        if len(fu):
            fu = fu.iloc[0]
            for ph in ("P", "S"):
                row[f"in_{ph}"] = fu.get(f"insample.{ph}.rms") / base[c]["insample"][ph]["rms"]
            row["above"] = fu.get("above_surface", np.nan)
            row["airq"] = fu.get("airquakes", np.nan)
            row["reloc_share"] = fu.relocated / fu.events
            row["shift_h"], row["shift_z"] = fu.get("shift_h_med", np.nan), fu.get("shift_z_med", np.nan)
        sy = g[g.kind == "synth"]
        if len(sy) and isinstance(sy.iloc[0].get("err_final"), list):
            s0, s1 = sy.iloc[0]["err_start"], sy.iloc[0]["err_final"]
            row["syn_h"], row["syn_z"] = s1[0] / s0[0], s1[1] / s0[1]
            row["syn_h_km"], row["syn_z_km"] = s1
        per.append(row)
    D = pd.DataFrame(per)
    ref = D[D.variant == REF].set_index("cluster")
    for ph in ("P", "S"):
        D[f"cvrel_{ph}"] = D[f"cv_{ph}"] / D.cluster.map(ref[f"cv_{ph}"])
        D[f"cvb4rel_{ph}"] = D[f"cv_{ph}"] / D[f"cvb4_{ph}"]
    D["cvrel_mean"] = (D.cvrel_P + D.cvrel_S) / 2
    od = OUT / "summary"
    od.mkdir(exist_ok=True)
    D.to_csv(od / "per_cluster.csv", index=False, float_format="%.5g")
    agg = D.groupby("variant").agg(
        clusters=("cluster", "nunique"),
        cv_P=("cvrel_P", "median"), cv_S=("cvrel_S", "median"), cv_mean=("cvrel_mean", "median"),
        cv_wins_both=("cvrel_P", lambda s: int(((s < 1) & (D.loc[s.index, "cvrel_S"] < 1)).sum())),
        cvb4_P=("cvb4rel_P", "median"), cvb4_S=("cvb4rel_S", "median"),
        syn_h=("syn_h", "median"), syn_z=("syn_z", "median"),
        in_P=("in_P", "median"), in_S=("in_S", "median"),
        above=("above", "sum"), airq=("airq", "sum"), reloc=("reloc_share", "median"),
        shift_h=("shift_h", "median"), shift_z=("shift_z", "median"))
    sw = {v: v.split("_")[0][1:] for v in agg.index}
    agg.insert(0, "s_weight", [sw[v] for v in agg.index])
    agg.insert(1, "reweight", [v.split("_", 1)[1] for v in agg.index])
    agg = agg.sort_values("cv_mean")
    agg.to_csv(od / "variant_scores.csv", float_format="%.4g")
    cols = ["s_weight", "reweight", "clusters", "cv_P", "cv_S", "cv_mean", "cv_wins_both", "cvb4_P", "cvb4_S",
            "syn_h", "syn_z", "in_P", "in_S", "above", "airq", "reloc"]
    L = ["# B5 weighting test: variant scores", "",
         f"{D.cluster.nunique()} clusters; {len(R)} task results. Ratios are medians over clusters. cv_* = held-out dt rms "
         f"relative to production ({REF}) in the same cluster (< 1 better); cv_wins_both = clusters where both phases beat "
         "production; cvb4_* = held-out rms relative to B4 (biased towards B4, which used every station); syn_* = synthetic "
         "relative error after / before; in_* = in-sample rms / B4; above = events above the surface (sum); airq = air "
         "quakes (sum); reloc = share relocated (median).", "",
         "| " + " | ".join(["variant"] + cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for v, r in agg.iterrows():
        L.append("| " + " | ".join([v] + [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols]) + " |")
    # per region for the best three variants and production
    best = list(agg.index[:3]) + ([REF] if REF not in agg.index[:3] else [])
    reg = D[D.variant.isin(best)].groupby(["region", "variant"]).agg(cv_P=("cvrel_P", "median"), cv_S=("cvrel_S", "median"),
                                                                     syn_h=("syn_h", "median"), syn_z=("syn_z", "median"),
                                                                     above=("above", "sum")).reset_index()
    L += ["", "## By region (best three variants by cv_mean, and production)", "",
          "| region | variant | cv_P | cv_S | syn_h | syn_z | above |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r.region} | {r.variant} | {r.cv_P:.3f} | {r.cv_S:.3f} | {r.syn_h:.3f} | {r.syn_z:.3f} | {r.above:.0f} |"
          for r in reg.itertuples()]
    (od / "report.md").write_text("\n".join(L) + "\n")
    write_manifest("b5_wtest_report", od, outputs=[od / "report.md", od / "variant_scores.csv", od / "per_cluster.csv"],
                   params=P, counts={"clusters": int(D.cluster.nunique()), "results": len(R)})
    print("\n".join(L))


if __name__ == "__main__":
    main()
