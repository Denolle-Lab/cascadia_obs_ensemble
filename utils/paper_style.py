"""Shared matplotlib style for the manuscript figures (Seismica, two-column).

Every figure is drawn at its PRINTED size, so font sizes here are the sizes on the
page: Seismica's column is 8.6 cm and its text width 18.0 cm. Text uses Source Sans
Pro, the journal's sans face (vendored in utils/fonts, SIL OFL; see OFL.txt).

    import sys; sys.path.append("../../utils")
    from paper_style import use, FULL, COL, panel, CLASS_COLORS, save
    use()
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 3.0))
    panel(ax[0], "a"); ...; save(fig, out)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

CM = 1 / 2.54
COL = 8.6 * CM           # one column (in)
FULL = 18.0 * CM         # full text width, for figure* (in)

# Okabe-Ito, checked with the dataviz palette validator (all checks pass; the one
# green/pink pair at dE 7.6 deutan always has a second cue: volcano triangles/labels)
CLASS_COLORS = {
    "megathrust?": "#D55E00",
    "oceanic": "#009E73",
    "intraslab": "#0072B2",
    "volcanic": "#CC79A7",
    "crustal-fault": "#8C8C8C",      # neutral: the upper-plate background population
}
# two-series comparisons: this catalog vs a reference catalog / model
OURS, REF = "#D55E00", "#0072B2"
INK, MUTED = "#222222", "#666666"
# events whose preferred magnitude is a moment magnitude (ComCat tensor or calibrated)
MW_COLOR = "#6A3D9A"

_FONTS = Path(__file__).resolve().parent / "fonts"


def use():
    for f in _FONTS.glob("*.otf"):
        font_manager.fontManager.addfont(str(f))
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = "Source Sans Pro" if "Source Sans Pro" in have else "Nimbus Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlepad": 3,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.0, "lines.markersize": 3, "patch.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=[OURS, REF, "#009E73", "#CC79A7"]),
        "image.cmap": "cividis", "mathtext.fontset": "custom",
        "mathtext.rm": family, "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150, "pdf.fonttype": 42,
    })


def panel(ax, letter, x=-0.02, y=1.0):
    """Bold panel letter just outside the top-left corner of the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom")


def save(fig, out):
    fig.savefig(os.path.expanduser(out))
    print(f"wrote {out}")
