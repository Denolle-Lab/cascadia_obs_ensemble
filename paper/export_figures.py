#!/usr/bin/env python3
"""Export manuscript figures from the repo's figure notebooks into paper/figures/.

The figure notebooks in ``figures/`` read/write data under ``/wd1/hbito_data/...``
(the UW server), so they only run where that data exists. This script maps each
paper figure to the notebook that generates it and the PNG that notebook writes
(under ``--src-dir``), and copies it into ``paper/figures/`` with the exact
filename the manuscript ``\includegraphics`` expects. With ``--execute`` it runs
each notebook first (via jupyter nbconvert; needs the data + the analysis env).

Run on the machine that has the data:
    pixi run python paper/export_figures.py                 # collect existing PNGs
    pixi run python paper/export_figures.py --execute       # regen notebooks, then collect
    pixi run python paper/export_figures.py --src-dir /path # override output directory

fig2 (pipeline schematic) and fig3 (assembled picking-example panels) have no
single generated source and are maintained by hand -- they are not listed here.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Default directory the figure notebooks save into (override with --src-dir).
DEFAULT_SRC_DIR = "/wd1/hbito_data/data/datasets_all_regions"

# (paper/figures/<dest>, generating notebook, <src>.png the notebook writes in src_dir)
FIGURES = [
    ("fig1.png", "figures/fig1_map_stas.ipynb", "fig1.png"),
    ("fig4_cc_no_relief_tremor_contours.png", "figures/fig4_match_events.ipynb",
     "filtered_events_2010_2015_ver3_cc_nass_no_relief_tremor_contours.png"),
    ("fig5_histograms/hist_rms_reloc_cog_ver3_p_4_s_4_rms_2_5.png",
     "figures/fig5_histograms_cc.ipynb", "hist_rms_reloc_cog_ver3_p_4_s_4_rms_2_5.png"),
    ("fig5_histograms/hist_p_picks_reloc_cog_ver3_p_4_s_4_rms_2_5.png",
     "figures/fig5_histograms_cc.ipynb", "hist_p_picks_reloc_cog_ver3_p_4_s_4_rms_2_5.png"),
    ("fig5_histograms/hist_s_picks_reloc_cog_ver3_p_4_s_4_rms_2_5.png",
     "figures/fig5_histograms_cc.ipynb", "hist_s_picks_reloc_cog_ver3_p_4_s_4_rms_2_5.png"),
    ("fig5_histograms/hist_gaps_reloc_cog_ver3_p_4_s_4_rms_2_5.png",
     "figures/fig5_histograms_cc.ipynb", "hist_gaps_reloc_cog_ver3_p_4_s_4_rms_2_5.png"),
    # fig6 = "Earthquake Catalog in the Endeavor" -> the axial/endeavor subregion map
    ("fig6.png", "figures/fig6_subregions_cc.ipynb",
     "axial_endeavor_events_2010_2015_ver3_cc_nass.png"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default=DEFAULT_SRC_DIR,
                    help="directory the figure notebooks write PNGs into")
    ap.add_argument("--execute", action="store_true",
                    help="run each generating notebook first (needs data + env)")
    args = ap.parse_args()
    src_dir = Path(args.src_dir)
    figdir = HERE / "figures"

    if args.execute:
        for nb in sorted({nb for _, nb, _ in FIGURES}):
            print(f"executing {nb} ...")
            subprocess.run(["jupyter", "nbconvert", "--to", "notebook",
                            "--execute", "--inplace", str(ROOT / nb)], check=True)

    ok = miss = 0
    for dest, _nb, src in FIGURES:
        s, d = src_dir / src, figdir / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.exists():
            shutil.copy2(s, d); ok += 1
            print(f"  ok    {dest:58s} <- {s}")
        else:
            miss += 1
            print(f"  MISS  {dest:58s} (not found: {s})")
    print(f"\ncopied {ok}, missing {miss}.  "
          "(fig2 pipeline schematic + fig3 picking panels are maintained by hand.)")
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
