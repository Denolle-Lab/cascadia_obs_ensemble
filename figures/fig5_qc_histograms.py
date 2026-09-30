"""Fig. 5: quality-control histograms of the ensemble catalog (one 2x2 figure).

Reproduces the subsets and bins of fig5_histograms_cc.ipynb exactly, redrawn at
print size (18 cm wide) as step outlines instead of overlapping opaque fills.

Run from figures/:
    LD_LIBRARY_PATH=../.pixi/envs/default/lib ../.pixi/envs/default/bin/python fig5_qc_histograms.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from paper_style import use, FULL, CM, REF, INK, panel, save  # noqa: E402

use()

year, method, version = "all_regions", "reloc", "ver3"
tag = f"{method}_cog_{version}_cc_p_4_s_4_rms_2_5"
d = f"../data/datasets_{year}"

# --- same inputs and subsets as fig5_histograms_cc.ipynb -------------------------
df = pd.read_csv(f"{d}/origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv", index_col=0)
matched_morton = pd.read_csv(f"{d}/matched_events_with_morton_mycatalog_{tag}.csv")
matched_anss = pd.read_csv(f"{d}/matched_events_with_anss_mycatalog_{tag}.csv")
unmatched_morton = df[~df["orid"].isin(matched_morton["orid"].drop_duplicates())]
unmatched_anss = df[~df["orid"].isin(matched_anss["orid"].drop_duplicates())]

MORTON = "#009E73"
# (label, frame, color, linestyle, linewidth); all events drawn first, heavier
series = [
    ("All events", df, INK, "-", 1.0),
    ("Matched to ANSS", matched_anss, REF, "-", 1.0),
    ("Unmatched to ANSS", unmatched_anss, REF, (0, (3, 1.5)), 1.0),
    ("Matched to Morton 2023", matched_morton, MORTON, "-", 1.0),
    ("Unmatched to Morton 2023", unmatched_morton, MORTON, (0, (3, 1.5)), 1.0),
]

# (column, bins, x label, xlim) -- bins exactly as in the notebook
panels = [
    ("rms", np.linspace(df["rms"].min(), df["rms"].max(), 55), "RMS travel-time residual (s)", None),
    ("p_picks", np.linspace(df["p_picks"].min(), 60, 55), "P picks per event", (0, 60)),
    ("s_picks", np.linspace(df["s_picks"].min(), 60, 55), "S picks per event", (0, 60)),
    ("gap", np.linspace(df["gap"].min(), df["gap"].max(), 55), "Azimuthal gap (°)", None),
]

print("per-subset counts (rows plotted, as in the notebook):")
for lab, sub, *_ in series:
    print(f"  {lab:28s} {len(sub):6d} rows, {sub['orid'].nunique():6d} unique orid")

fig, axes = plt.subplots(2, 2, figsize=(FULL, 11 * CM))
for ax, letter, (col, bins, xlabel, xlim) in zip(axes.flat, "abcd", panels):
    for lab, sub, color, ls, lw in series:
        ax.hist(sub[col], bins=bins, histtype="step", color=color, linestyle=ls,
                linewidth=lw, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.set_xlim(xlim if xlim is not None else (bins[0], bins[-1]))
    ax.set_ylim(bottom=0)
    panel(ax, letter, x=-0.075, y=1.02)

handles = [Line2D([], [], color=c, linestyle=ls, linewidth=lw) for _, _, c, ls, lw in series]
fig.legend(handles, [s[0] for s in series], loc="upper center", ncol=5,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.8, w_pad=1.5)
save(fig, f"{d}/fig5_qc_histograms.png")
