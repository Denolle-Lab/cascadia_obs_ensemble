"""Paired comparison of B5 weighting variants (b5_wtest_report.py per_cluster.csv): per-cluster differences against a
reference variant, median and 90 % bootstrap interval over clusters (2,000 resamples), and the number of clusters
where the variant is better.  python b5_wtest_paired.py [REF]   -> <out>/summary/paired_<REF>.md"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params  # noqa: E402

OUT = Path(load_params()["b5_wtest"]["out"]) / "summary"
REF = sys.argv[1] if len(sys.argv) > 1 else "s1.0_prod"
D = pd.read_csv(OUT / "per_cluster.csv")
M = {m: D.pivot(index="cluster", columns="variant", values=m) for m in ("cvrel_P", "cvrel_S", "cvrel_mean", "syn_h", "syn_z", "in_P", "in_S")}
rng = np.random.default_rng(0)
L = [f"# Paired differences against {REF} (variant minus reference; negative = better)", "",
     "Median over 26 clusters [90 % bootstrap interval]; wins = clusters where the variant is better.", "",
     "| variant | cv_P | cv_S | cv_mean | syn_h | syn_z | in_P | in_S |", "|---|---|---|---|---|---|---|---|"]
for v in sorted(M["cvrel_mean"].columns):
    if v == REF:
        continue
    cells = []
    for m in ("cvrel_P", "cvrel_S", "cvrel_mean", "syn_h", "syn_z", "in_P", "in_S"):
        d = (M[m][v] - M[m][REF]).dropna().values
        bs = np.median(d[rng.integers(0, len(d), (2000, len(d)))], axis=1)
        lo, hi = np.percentile(bs, [5, 95])
        cells.append(f"{np.median(d):+.3f} [{lo:+.3f}, {hi:+.3f}] {int((d < 0).sum())}/{len(d)}")
    L.append(f"| {v} | " + " | ".join(cells) + " |")
(OUT / f"paired_{REF}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
