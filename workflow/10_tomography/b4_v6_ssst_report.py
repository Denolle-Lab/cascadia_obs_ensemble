"""Track B, catalog v6.0 preparation (D-B092): HTML report of the NLL-SSST test (b4_v6_ssst_metrics.py outputs) and of
the v0.4.1 grid build, with the plan text of notes/v6_plan.md.

  python b4_v6_ssst_report.py [--grids g51[,g041]] -> /wd1/mdenolle_data/reloc_v6/report/v6_prep_report.html
"""
import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_relocation"))
from b4_v6_ssst import OUT, SETS, SCHEDULE, GRIDS, R6  # noqa: E402

REPORT = R6 / "report"
ARMS = ["v5.1", "static", "ssst_it0", "ssst_it1"] + [f"ssst_it{k}" for k in range(2, len(SCHEDULE) + 1)] + ["coherence", "static_coherence"]
LABEL = {"v5.1": "v5.1 catalog (tile box)", "static": "static terms (v5.1 file)", "ssst_it0": "no terms (it0)",
         "ssst_it1": "SSST it1 (static, 9999 km)", "coherence": "SSST + coherence", "static_coherence": "static + coherence"}
for k, c in enumerate(SCHEDULE[1:], start=2):
    LABEL[f"ssst_it{k}"] = f"SSST it{k} ({c:g} km)"
LABEL[f"ssst_it{len(SCHEDULE)}"] = f"SSST final it{len(SCHEDULE)} ({SCHEDULE[-1]:g} km)"


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def table(head, rows, cls=""):
    h = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def set_table(m, s):
    head = ["arm", "located", "tier A-B", "above surface", "floor-limited", "median depth km", "depth below seafloor km",
            "P rms s", "S rms s", "CC pair MAD ms (P / S)", "catalog-dt pair MAD ms", "repeater distance km (median)",
            "nearest neighbour km", "moved vs static: dh / dz km"]
    if s == "axial":
        head.insert(7, "Axial A-B within 2 km of seafloor %")
    rows = []
    for a in ARMS:
        if a not in m:
            continue
        r = m[a]
        t = r.get("tiers", {})
        ab = t.get("A", 0) + t.get("B", 0)
        vs = r.get("vs_static", {})
        row = [LABEL.get(a, a), fmt(r.get("located")), fmt(ab), fmt(r.get("above_surface")), fmt(r.get("floor_limited")),
               fmt(r.get("depth_t_median_km")), fmt(r.get("below_seafloor_median_km")),
               fmt(r.get("rms_P_s"), 3), fmt(r.get("rms_S_s"), 3),
               (f"{fmt(r.get('cc_mad_pair_ms'), 1)} ({fmt(r.get('cc_mad_pair_P_ms'), 1)} / "
                f"{fmt(r.get('cc_mad_pair_S_ms'), 1)})") if r.get("cc_mad_pair_ms") is not None else "",
               fmt(r.get("cat_mad_pair_ms"), 1), fmt(r.get("repeater_dist_median_km"), 3),
               fmt(r.get("nn_median_km"), 3),
               f"{fmt(vs.get('dh_median_km'), 3)} / {fmt(vs.get('dz_median_km'), 3)}" if vs else ""]
        if s == "axial":
            row.insert(7, fmt(r.get("axial_within_2.0_km"), 1))
        rows.append(row)
    return table(head, rows, "num")


def lit_table(m):
    rows = []
    for a in ARMS:
        L = (m.get(a) or {}).get("literature") or {}
        if not L:
            continue
        cells = [LABEL.get(a, a)]
        for t in ("A-D", "A-B"):
            q = L.get(t)
            cells.append(f"{q['med']:+.2f} [{q['p10']:+.1f}, {q['p90']:+.1f}] n={q['n']}" if q else "")
        rows.append(cells)
    return table(["arm", "tiers A-D: median [p10, p90]", "tiers A-B"], rows, "num") if rows else ""


def fig_set(s, g):
    """map and depth section of static, final SSST and coherence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from b4_v6_ssst_metrics import read_arm, located_frame
    import a2_parse_hyp as PH
    v4 = PH.read_v4()
    d = OUT / g / s
    arms = [("static", d / "static"), (f"ssst_it{len(SCHEDULE)}", d / "ssst" / f"it{len(SCHEDULE)}"),
            ("coherence", d / "coherence")]
    data = []
    for lab, rd in arms:
        if not (rd / "run.json").exists():
            continue
        ev, _ = read_arm(rd)
        if lab == "coherence":
            ev5, _ = read_arm(d / "ssst" / f"it{len(SCHEDULE)}")
            ev = pd.concat([ev, ev5[~ev5.ID.isin(ev.ID)]], ignore_index=True)
        data.append((lab, located_frame(ev, v4)))
    if not data:
        return ""
    allx = np.concatenate([e.x for _, e in data])
    ally = np.concatenate([e.y for _, e in data])
    xc, yc = np.median(allx), np.median(ally)
    lim = max(np.percentile(np.abs(allx - xc), 98), np.percentile(np.abs(ally - yc), 98)) + 1
    zlo = min(np.percentile(e.depth_t, 1) for _, e in data) - 0.5
    zhi = max(np.percentile(e.depth_t, 99) for _, e in data) + 0.5
    fig, ax = plt.subplots(2, len(data), figsize=(3.3 * len(data), 6.2), squeeze=False,
                           gridspec_kw=dict(height_ratios=[1, 0.8]))
    for j, (lab, e) in enumerate(data):
        ab = e.tier.isin(["A", "B"])
        for sel, col in ((~ab, "#9aa5b1"), (ab, "#1f5f99")):
            ax[0, j].scatter(e.x[sel] - xc, e.y[sel] - yc, s=2, c=col, lw=0)
            ax[1, j].scatter(e.x[sel] - xc, e.depth_t[sel], s=2, c=col, lw=0)
        ax[1, j].plot(np.sort(e.x - xc), -np.asarray(e.surface_km)[np.argsort(e.x - xc)], c="#b5651d", lw=0.8)
        ax[0, j].set(xlim=(-lim, lim), ylim=(-lim, lim), title=LABEL.get(lab, lab), aspect="equal")
        ax[1, j].set(xlim=(-lim, lim), ylim=(zhi, zlo), xlabel="east (km, Lambert)")
        ax[0, j].tick_params(labelsize=7)
        ax[1, j].tick_params(labelsize=7)
    ax[0, 0].set_ylabel("north (km)")
    ax[1, 0].set_ylabel("depth (km bsl)")
    fig.suptitle(f"{s}: blue tier A-B, grey C-D; brown line: solid surface at the epicentres", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return f'<img alt="{s} locations" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def md_html(text):
    """the small Markdown subset of the notes (headings, paragraphs, '-' lists, '|' tables, `code`, **bold**)."""
    import re
    def inline(t):
        t = html.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    out, para, lst, tab = [], [], [], []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if lst:
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in lst) + "</ul>")
            lst.clear()
        if tab:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in tab if not set(r) <= set("|-: ")]
            out.append("<div class='wrap'>" + table(rows[0], [[inline(c) for c in r] for r in rows[1:]]) + "</div>")
            tab.clear()
    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            n = len(line) - len(line.lstrip("#"))
            out.append(f"<h{n + 1}>{inline(line[n:].strip())}</h{n + 1}>")
        elif line.startswith("|"):
            if para or lst:
                flush()
            tab.append(line)
        elif re.match(r"^\s*[-*] ", line):
            if para or tab:
                flush()
            lst.append(re.sub(r"^\s*[-*] ", "", line))
        elif line.startswith("  ") and lst:
            lst[-1] += " " + line.strip()
        elif not line.strip():
            flush()
        else:
            if lst or tab:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


CSS = """
:root{--bg:#fbfbf9;--fg:#1d2329;--mut:#5b6670;--line:#d9dde1;--acc:#1f5f99;--hd:#eef1f4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15191d;--fg:#e4e8ec;--mut:#9aa6b1;--line:#333b43;--acc:#7fb2e5;--hd:#1f252b}}
:root[data-theme="dark"]{--bg:#15191d;--fg:#e4e8ec;--mut:#9aa6b1;--line:#333b43;--acc:#7fb2e5;--hd:#1f252b}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:0 16px}
main{max-width:1180px;margin:24px auto 80px}
h1{font-size:1.6em;margin:.2em 0}h2{font-size:1.25em;margin-top:2em;border-bottom:1px solid var(--line)}h3{font-size:1.05em}
p,li{max-width:900px}.mut{color:var(--mut)}code{font-size:.92em}
.wrap{overflow-x:auto}table{border-collapse:collapse;margin:.6em 0;font-size:.86em}
th,td{border:1px solid var(--line);padding:3px 7px;vertical-align:top}th{background:var(--hd);text-align:left}
table.num td{text-align:right}table.num td:first-child{text-align:left}
img{max-width:100%;height:auto;background:#fff}
.box{border-left:3px solid var(--acc);padding:4px 12px;background:var(--hd)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="g51")
    ap.add_argument("--intro", default=str(HERE / "notes" / "v6_prep_results.md"), help="summary (Markdown note)")
    ap.add_argument("--plan", default=str(HERE / "notes" / "v6_plan.md"), help="plan (Markdown note)")
    a = ap.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    parts = [f"<h1>Catalog v6.0 preparation: v0.4.1 grids and NLL-SSST test</h1>"
             f"<p class='mut'>Track B, D-B092 (PI 2026-11-25), implements D-B079. Generated by "
             f"workflow/10_tomography/b4_v6_ssst_report.py (branch reloc/v6-prep).</p>"]
    if Path(a.intro).exists():
        t = Path(a.intro).read_text()
        parts.append(md_html(t) if a.intro.endswith(".md") else t)
    for g in a.grids.split(","):
        parts.append(f"<h2>Cluster tests, grids {g}: {html.escape(GRIDS[g]['about'])}</h2>")
        for s in SETS:
            f = OUT / g / f"metrics_{s}.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            parts.append(f"<h3>{s}</h3><div class='wrap'>{set_table(m, s)}</div>")
            lt = lit_table(m)
            if lt:
                ref = next((v["literature"]["reference"] for v in m.values() if v.get("literature")), "")
                parts.append(f"<p>Depth minus published depth ({html.escape(ref)}), km, positive = ours deeper:</p>"
                             f"<div class='wrap'>{lt}</div>")
            for ca in ("coherence", "static_coherence"):
              coh = (m.get(ca) or {}).get("coherence_run")
              if coh:
                parts.append(f"<p class='mut'>{LABEL[ca]}: {coh.get('events')} of {coh.get('events_final_ssst')} "
                             f"events relocated (ellipsoid len3 &le; {coh.get('max_se3')} km), "
                             f"{coh.get('compact_with_compact_partner')} with at least one partner of coherence "
                             f"&ge; {coh.get('coh_min')} that passes the same filter; {coh.get('pairs_used')} pairs.</p>")
            ls = [(k, v.get("loc2ssst")) for k, v in m.items() if v.get("loc2ssst")]
            if ls:
                parts.append("<p class='mut'>Loc2ssst runs: " + "; ".join(
                    f"{k}: CharDist {v['char_dist_km']:g} km, {v['events_accepted']} events, {v['corrected_grids']} grids, "
                    f"{v['negative_nodes']} negative nodes (max {v.get('neg_max_dist_km', 0):g} km from the station), "
                    f"{v['wall_s']:.0f} s" for k, v in ls) + "</p>")
            try:
                parts.append(fig_set(s, g))
            except Exception as e:                       # a figure must not stop the report
                parts.append(f"<p class='mut'>figure failed: {html.escape(repr(e))}</p>")
    if Path(a.plan).exists():
        parts.append(md_html(Path(a.plan).read_text()))
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>Catalog v6.0 preparation</title><style>{CSS}</style>"
           f"</head><body><main>{''.join(parts)}</main></body></html>")
    out = REPORT / "v6_prep_report.html"
    out.write_text(doc)
    print(out, len(doc))


if __name__ == "__main__":
    main()
