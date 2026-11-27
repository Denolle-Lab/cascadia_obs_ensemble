#!/usr/bin/env python3
"""Paper figures v2 (D-B114, PI 2026-11-27 13:13: "Make the paper figures more beautiful, even for the tomography").

One entry point, one style (paper_style.py), one configuration of input paths (paper_figures_v2.yaml):

    python workflow/08_figures/paper_figures_v2.py --config workflow/08_figures/paper_figures_v2.yaml all
    python workflow/08_figures/paper_figures_v2.py --config <yaml> fig4 fig5 review   # some figures, then the page

Figures (D-B107/D-B109 framing: seismicity style and fault-zone structure at the offshore plate boundaries; structure
only at plate-boundary scale, 40 km; no full-size model maps):
    fig1  plate boundaries and data: segments coloured by type, stations, plate motions, rate and network
    fig2  seismicity-style fingerprint per segment (dot matrix)
    fig3  fault geometry per boundary (fault surfaces v1 refit on the catalog)
    fig4  fault-zone tomography: fault-normal sections, Vp and Vp/Vs, ensemble 1 sigma as transparency
    fig5  structure against seismicity: seismicity bottom against the 7.5 km/s level; contrasts against sigma
    fig6  temporal character of the boundary types
    figS_mc, figS_catalog, figS_perspective   supplement: Mc maps, catalog comparison, Mendocino perspective view
    palette   the boundary-type palette under colour-vision deficiency (page asset)
    review    the review page (review.html in the output folder): figures, caption drafts, before/after

Outputs: <out>/<name>.pdf (vector) and .png (300 dpi), <out>/manifest.json (sizes, captions, inputs), review.html.
Run niced with at most 2 threads (run_figures_v2.sh does both). Nothing is written into paper_sciadv/figures/.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as S  # noqa: E402
from paper_v2 import data as D  # noqa: E402

FIGURES = {   # name: module in paper_v2 (each has make(cfg) -> list of manifest entries)
    "fig1": "f1_overview", "fig2": "f2_fingerprint", "fig3": "f3_faults", "fig4": "f4_sections",
    "fig5": "f5_structure_seismicity", "fig6": "f6_temporal", "figS_mc": "s_mc",
    "figS_catalog": "s_catalog_compare", "figS_perspective": "s_perspective",
}
ORDER = ["palette", *FIGURES, "review"]


def manifest_update(out, entries):
    f = Path(out) / "manifest.json"
    m = json.loads(f.read_text()) if f.exists() else {}
    for e in entries:
        e["built"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        m[e["name"]] = e
    f.write_text(json.dumps(m, indent=1, default=str))


def run_palette(cfg):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    S.palette_swatch_png(out / "palette_types_cvd.png")
    chk = S.check_palette()
    (out / "palette_check.json").write_text(json.dumps(chk, indent=1))
    return [dict(name="palette", kind="asset", files=["palette_types_cvd.png"], check=chk)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "paper_figures_v2.yaml"))
    ap.add_argument("figures", nargs="+", choices=ORDER + ["all"])
    a = ap.parse_args(argv)
    cfg = D.load_config(a.config)
    S.use()
    names = ORDER if "all" in a.figures else a.figures
    Path(cfg["out"]).mkdir(parents=True, exist_ok=True)
    for n in names:
        t0 = time.time()
        print(f"== {n} {datetime.now():%H:%M:%S}", flush=True)
        if n == "palette":
            ent = run_palette(cfg)
        elif n == "review":
            from paper_v2 import review
            review.make(cfg)
            continue
        else:
            mod = importlib.import_module(f"paper_v2.{FIGURES[n]}")
            ent = mod.make(cfg)
        manifest_update(cfg["out"], ent)
        print(f"   {n}: {time.time() - t0:.0f} s; " + ", ".join(f"{e['name']} {e.get('size_cm', '')}" for e in ent),
              flush=True)


if __name__ == "__main__":
    main()
