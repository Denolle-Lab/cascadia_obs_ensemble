"""Review page of the paper figures v2: <out>/review.html, self-contained in the output folder (images by relative
path). Per figure: the new figure, its size, a fact-only caption draft, and the figure it replaces (the current paper
figure from `old_figures`, or the plan mock-up from `plan_dir`/figures), copied into <out>/before/ for the comparison.
Also the style sheet (one line per choice), the boundary-type palette under colour-vision deficiency, and the rerun
recipe. Nothing is written into paper_sciadv/figures/.
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import paper_style as S
from paper_v2 import data as D

ORDER = [("Main figures", ["fig1_overview", "fig2_fingerprint", "fig3_fault_geometry", "fig4_fault_zone_sections",
                           "fig5_structure_seismicity", "fig6_temporal"]),
         ("Supplement", ["figS_mc_methods", "figS_mc_periods", "figS_catalog_compare", "figS_perspective_mendocino"])]
TITLES = {"fig1_overview": "Fig. 1 · Plate boundaries and data", "fig2_fingerprint": "Fig. 2 · Seismicity-style fingerprint",
          "fig3_fault_geometry": "Fig. 3 · Fault geometry per boundary",
          "fig4_fault_zone_sections": "Fig. 4 · Fault-zone tomography",
          "fig5_structure_seismicity": "Fig. 5 · Structure against seismicity",
          "fig6_temporal": "Fig. 6 · Temporal character", "figS_mc_methods": "Fig. S · Mc, three methods",
          "figS_mc_periods": "Fig. S · Mc by network period", "figS_catalog_compare": "Fig. S · Catalog comparison",
          "figS_perspective_mendocino": "Fig. S · Mendocino perspective view"}

STYLE = [
    ("Sizes", f"Science Advances widths {S.COL1} cm (one column) and {S.COL2} cm (two columns); every figure here is "
              f"{S.COL2} cm wide and at most {S.MAX_H:g} cm tall."),
    ("Type", "One sans-serif family (Helvetica metrics: Nimbus Sans in matplotlib, Helvetica in GMT), 6 pt ticks and "
             "notes, 7 pt labels and titles, 8 pt bold panel letters A, B, C at the top-left corner of each panel."),
    ("Lines", "Frames and ticks 0.5 pt; hairline grids (0.35 pt, solid, light grey) only where values are read; data "
              "lines 0.9 pt; ridges 1.1 pt, transforms and trench 0.8 pt with filled teeth on the upper plate."),
    ("Base map", "GMRT relief (GridServer 'med' tiles mosaicked once into the cache) as a light hillshade from the "
                 "north-west: grey-blue sea darkening with depth, warm light-grey land; coast 0.4 pt."),
    ("Plate boundaries", "PB2002 with the D-B113 bathymetric axes (southern Gorda Ridge, Gorda-Mendocino stub, JdF "
                         "Ridge 46.4-47.7 N) in place of the PB2002 ridges and oceanic transforms."),
    ("Boundary types", "One colour and one marker per type (ridge, transform, intraplate, triple junction, subduction, "
                       "crustal, arc; other in grey), checked under protanopia, deuteranopia and tritanopia (Machado "
                       "2009); the marker or the segment code always rides with the colour."),
    ("Colour maps", "Crameri SCM (D-B104): Vp batlow reversed (one range 5.6-8.2 km/s), Vp/Vs roma reversed centred on "
                    "1.75 (1.60-1.90), dV vik reversed; time lipari, Mc batlow, depth lajolla, surface dip acton."),
    ("Uncertainty", "Fig. 4: the ensemble 1σ of each binned value is its opacity (full colour at small σ, faint at "
                    "large σ, 2-D key); cells without resolved nodes hatched grey; the water column pale blue."),
    ("Events", "Symbol area from magnitude, one law for all figures (diameter 0.8 pt × 10^(0.2 M)); dense clouds as "
               "small translucent dots; size key where size carries magnitude."),
    ("Map furniture", "Scale bar and north arrow on maps; NNR-MORVEL56 relative plate motions as cased arrows."),
    ("No dual axes", "The old Fig. 1D and Fig. 6A each had two y axes; both are now two stacked panels on one time "
                     "axis."),
    ("Output", "Vector PDF (fonts embedded, dense layers rasterised at 300 dpi) and 300 dpi PNG per figure."),
]

CSS = """
:root { --ink:#1b1b1b; --ink2:#5c5c5c; --rule:#e4e4e4; --bg:#ffffff; --panel:#f7f7f5; --accent:#2B6CB0; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 20px 80px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 44px 0 10px; padding-top: 14px; border-top: 1px solid var(--rule); }
h3 { font-size: 16px; margin: 0 0 6px; }
.meta { color: var(--ink2); font-size: 13px; }
.fig { margin: 26px 0 38px; }
.fig img { width: 100%; height: auto; border: 1px solid var(--rule); background: #fff; }
.cap { background: var(--panel); border-radius: 6px; padding: 10px 14px; margin: 10px 0 0; font-size: 14px; }
.cap b { font-weight: 600; }
.ba { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 12px; }
.ba details summary { cursor: pointer; color: var(--accent); font-size: 14px; }
.ba img { max-width: 100%; border: 1px solid var(--rule); margin-top: 8px; }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0; }
td, th { border-bottom: 1px solid var(--rule); padding: 4px 10px 4px 0; text-align: left; vertical-align: top; }
th { color: var(--ink2); font-weight: 600; }
.sw { display:inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -1px; margin-right: 6px; }
code, pre { font: 12.5px/1.45 Menlo, Consolas, monospace; background: var(--panel); border-radius: 4px; }
pre { padding: 10px 12px; overflow-x: auto; }
code { padding: 1px 4px; }
.toc a { margin-right: 14px; color: var(--accent); text-decoration: none; font-size: 14px; }
.flag { color: #8a4b00; }
@media (max-width: 640px) { body { font-size: 14px; } main { padding: 18px 16px 60px; } }
"""


def _git(root, *a):
    try:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def make(cfg):
    out = Path(cfg["out"])
    man = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    before = out / "before"
    before.mkdir(exist_ok=True)
    old = D.ROOT / cfg.get("old_figures", "paper_sciadv/figures")
    plan_figs = Path(cfg["plan_dir"]) / "figures"
    root = D.ROOT
    head = _git(root, "rev-parse", "--short", "HEAD")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    built = datetime.now().astimezone().strftime("%a %d %b %Y %H:%M %Z")
    ct = D.contrasts(cfg)
    null_scored = bool(ct.null_scored.iloc[0])
    toc = []
    body = []
    for sec, names in ORDER:
        body.append(f"<h2 id='{sec.split()[0].lower()}'>{sec}</h2>")
        for n in names:
            e = man.get(n)
            if e is None:
                body.append(f"<div class='fig'><h3>{html.escape(TITLES.get(n, n))}</h3><p class='flag'>not built</p></div>")
                continue
            toc.append(f"<a href='#{n}'>{html.escape(TITLES.get(n, n).split(' · ')[0])}</a>")
            w, h = e.get("size_cm", ("?", "?"))
            pdf = f"{n}.pdf"
            parts = [f"<div class='fig' id='{n}'><h3>{html.escape(TITLES.get(n, n))}</h3>",
                     f"<div class='meta'>{w} × {h} cm · <a href='{pdf}'>PDF</a> · <a href='{n}.png'>PNG 300 dpi</a> · "
                     f"built {html.escape(str(e.get('built', '')))}</div>",
                     f"<a href='{n}.png'><img loading='lazy' src='{n}.png' alt='{html.escape(TITLES.get(n, n))}'></a>",
                     f"<div class='cap'><b>Caption draft.</b> {html.escape(e.get('caption', ''))}</div>"]
            olds = []
            for b in e.get("before", []) or []:
                src = old / b
                if src.exists():
                    shutil.copy2(src, before / b)
                    olds.append((f"before/{b}", f"current paper figure {b}"))
            for b in e.get("before_plan", []) or []:
                src = plan_figs / b
                if src.exists():
                    shutil.copy2(src, before / f"plan_{b}")
                    olds.append((f"before/plan_{b}", f"plan mock-up {b}"))
            if olds:
                parts.append("<div class='ba'>")
                for p, lab in olds:
                    parts.append(f"<details><summary>Before: {html.escape(lab)}</summary><img loading='lazy' src='{p}' "
                                 f"alt='{html.escape(lab)}'></details>")
                parts.append("</div>")
            elif n in ("fig4_fault_zone_sections", "fig5_structure_seismicity"):
                parts.append("<div class='meta'>New figure (no paper predecessor).</div>")
            parts.append("</div>")
            body.append("".join(parts))
    # palette
    chk = json.loads((out / "palette_check.json").read_text()) if (out / "palette_check.json").exists() else None
    pal = ["<h2 id='style'>Style</h2><table>"]
    for k, v in STYLE:
        pal.append(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>")
    pal.append("</table>")
    if chk:
        pal.append("<h3 style='margin-top:22px'>Boundary-type palette</h3><table><tr><th>type</th><th>colour</th>"
                   "<th>marker</th><th>OKLCH L</th><th>OKLCH C</th><th>contrast on white</th></tr>")
        for c in chk["colours"]:
            pal.append(f"<tr><td>{c['type']}</td><td><span class='sw' style='background:{c['hex']}'></span>{c['hex']}"
                       f"</td><td>{html.escape(S.TYPE_MK[c['type']])}</td><td>{c['L']}</td><td>{c['C']}</td>"
                       f"<td>{c['contrast']}</td></tr>")
        pal.append("</table><table><tr><th>pairs</th><th>worst CVD ΔE (protan/deutan)</th><th>worst normal ΔE</th>"
                   "<th>worst tritan ΔE</th><th>verdict</th></tr>")
        for k, v in chk["pairs"].items():
            pal.append(f"<tr><td>{html.escape(k)}</td><td>{v['cvd']} ({html.escape(v['cvd_pair'])})</td><td>"
                       f"{v['normal']} ({html.escape(v['normal_pair'])})</td><td>{v['tritan']} "
                       f"({html.escape(v['tritan_pair'])})</td><td>{html.escape(v['verdict'])}</td></tr>")
        pal.append("</table><p class='meta'>OKLab ΔE × 100; target ≥ 8 under dichromacy and ≥ 15 for normal vision "
                   "(dataviz method). Seven hues cannot all be pairwise distinct under dichromacy, so the marker and the "
                   "segment code carry identity wherever colours meet; the adjacent legend pairs and every pair of types "
                   "that touch on the map pass.</p><img src='palette_types_cvd.png' style='max-width:100%' "
                   "alt='palette under colour-vision deficiency'>")
    rerun = f"""<h2 id='rerun'>Rerun</h2>
<p>One command rebuilds every figure, the manifest and this page from the paths in the configuration:</p>
<pre>bash workflow/08_figures/run_figures_v2.sh workflow/08_figures/paper_figures_v2.yaml</pre>
<p>Inputs read: catalog {html.escape(cfg['label'])} event table with segments (<code>events_segments</code>), segment
tables (<code>seismicity_metrics</code>, <code>structure_feasibility</code>, <code>segment_axes</code>,
<code>surface_segments</code>), contrasts (<code>circularity_contrasts</code>, <code>null_truth_contrasts</code>),
<code>depth_vs_level*</code>, the D-B113 axes (<code>plate_axes</code>), clusters, fault surfaces v1, Mc maps, model
v0.4.2 and its 100 members. Segment codes, names and types come from the tables, so a new segment rule needs no code
change. After the null-truth score (D-B112) writes <code>null_truth_contrasts.csv</code> into
<code>{html.escape(str(cfg['tables']))}</code>, Figs. 4 and 5 switch on their own: filled symbols = claimed, the N95
bands appear, and the captions drop "pending". Null truth {'scored' if null_scored else 'not scored yet'} at this build.</p>
<p class='meta'>Configuration: {html.escape(cfg['config_path'])}</p>"""
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'><title>Paper figures v2</title><style>{CSS}</style>
</head><body><main>
<h1>Paper figures v2: review</h1>
<div class='meta'>D-B114 (PI 2026-11-27 13:13: "Make the paper figures more beautiful, even for the tomography"). Framing
D-B107/D-B109: seismicity style and fault-zone structure at the offshore plate boundaries; structure only at
plate-boundary scale (40 km); no full-size model maps. Catalog {html.escape(cfg['label'])}, model v0.4.2, segments
after D-B113. Built {built}; branch {html.escape(branch)} at {html.escape(head)}. Private working page; nothing here is in
paper_sciadv/figures/ yet (the PI picks first).</div>
<p class='toc'>{''.join(toc)} <a href='#style'>Style</a> <a href='#rerun'>Rerun</a></p>
{''.join(body)}
{''.join(pal)}
{rerun}
</main></body></html>"""
    (out / "review.html").write_text(page)
    print(f"wrote {out / 'review.html'}")
