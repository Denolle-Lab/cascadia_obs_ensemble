"""HTML report of the literature matches of the fault surfaces v0.

Reads data/faults/v0/fault_surfaces_v0_literature.csv, _summary.csv and the regional notes in
data/faults/v0/literature/*.md; writes data/faults/v0/fault_surfaces_v0_literature.html.
Needs the `markdown` package (run with `pixi exec --spec python=3.11 --spec markdown --spec pandas`).
"""
from html import escape
from pathlib import Path

import markdown
import pandas as pd

V0 = Path(__file__).resolve().parents[2] / "data" / "faults" / "v0"
REG = {"BL": "Blanco", "MT": "Mendocino triple junction", "GO": "Gorda", "PU": "Puget Lowland", "WA": "Offshore Washington",
       "ON": "Onshore and other", "OR": "Offshore Oregon", "EN": "Endeavour–Explorer"}
NOTES = [("blanco", "Blanco"), ("mendocino", "Mendocino triple junction"), ("gorda", "Gorda"),
         ("puget_wa", "Puget Lowland and offshore Washington"), ("onshore_or_en", "Onshore, offshore Oregon, Endeavour")]
CLASSES = ["known structure", "duplicate", "seismicity layer or volume", "no known counterpart", "likely artefact", "quarry blasts"]

FINDINGS = """
<li><b>Most surfaces are known structures.</b> 99 of 155 match a published fault, ridge segment, transform
strand or seismic zone; 26 of them with high or medium-high confidence. Examples: Blanco Ridge
(BL003, BL013, strike within 2°), the Cascadia Depression normal faults (BL005, BL008, BL021), the
Escanaba Trough and the Jackson and Central Gorda Ridge segments (GO002, GO008, GO003), the faults of
the 2010 Mw 6.5 and 2014 Mw 6.8 Gorda earthquakes (MT014, MT024; MT008, MT012), the 2005 Mw 7.2 and
Mw 6.6 fault zones (GO015, GO013), the Garberville swarm (MT009, the buried eastward Mendocino transform
of Shelly et al. 2025, 2026), the St. Helens seismic zone (ON001, ON003), the Twin Lakes segment of the
Mount Hood fault zone (ON015), the Portland Hills fault at depth (ON013), the Leech River fault zone
(PU011), and the fault of the 2014 Mw 6.4 Nootka sequence (EN003).</li>
<li><b>Map positions are good; depths are not.</b> On the Gorda Ridge, v4 epicentres sit on the
bathymetric axial valley, while ComCat puts the same events 10–20 km east. But nine Blanco surfaces
reach below 25 km, where published Blanco seismicity stops at 10–14 km, and off Oregon ComCat puts the
OR001 cluster at 26–33 km against 6 km in v4. The relocation (Track A) is needed before dips and depths
are interpreted.</li>
<li><b>Several fits join structures that the literature separates.</b> The long western Blanco surfaces
(BL001, BL002, BL006, BL009) cross right-stepping strands; the large Mendocino surfaces (MT001–MT007)
are the two layers of the double seismic zone in the Gorda slab, not single faults; the broad low-dip
Puget planes (PU001, PU002, PU005, PU013) are diffuse basement seismicity. 17 surfaces duplicate a
neighbour.</li>
<li><b>Candidate new structures (35 flagged).</b> The strongest:
a WNW right-lateral source in the Gorda plate near 41.85°N 126.1°W with repeated M4–5.7 events
(GO010, GO011, GO017); the 2012 Brookings swarm (GO020); a plane on the Cleft segment of the Juan de
Fuca Ridge (BL025); a north-dipping E–W structure east of 122.1°W that could continue the Tacoma fault
(PU015, PU018); an E–W vertical alignment north of the Mendocino transform (MT020); a NNW dextral strand
west of the Eaton Roughs fault (MT017); and a steep N-striking zone at Salmon Inlet, BC, active in 2015,
about 4 km from the 2025 Mww 4.8 epicentre (ON022).</li>
<li><b>To remove or fix before use.</b> Six surfaces are mostly quarry blasts (ON012, ON020, ON023,
ON029, ON037, ON040); four are likely artefacts (BL022, BL024, BL026, PU010). The 2010–2016 catalog
has no intraslab surface under Puget Sound and no surface on the Leech River trace itself, Klamath
Falls, Daisy Bank or Wecoma.</li>
"""


def table(df, cols, heads):
    h = "".join(f"<th>{escape(x)}</th>" for x in heads)
    body = ""
    for _, r in df.iterrows():
        body += "<tr>" + "".join(f"<td>{escape(str(r[c]))}</td>" for c in cols) + "</tr>"
    return f'<div class="tbl"><table><tr>{h}</tr>{body}</table></div>'


def main():
    lit = pd.read_csv(V0 / "fault_surfaces_v0_literature.csv")
    s = pd.read_csv(V0 / "fault_surfaces_v0_summary.csv")
    d = s[["surface_id", "n_events", "lon", "lat", "strike", "dip", "length_km"]].merge(lit, on="surface_id")
    d["region"] = d.surface_id.str[:2].map(REG)
    ct = pd.crosstab(d.region, d.lit_class).reindex(columns=CLASSES, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    ct = ct.reset_index()
    order = {"high": 0, "medium-high": 1, "medium": 2, "low-medium": 3, "low": 4, "none": 5}
    hi = d[(d.lit_class == "known structure") & d.lit_confidence.isin(["high", "medium-high"])].sort_values("surface_id")
    new = d[d.lit_new_candidate].sort_values("surface_id")
    bad = d[d.lit_class.isin(["quarry blasts", "likely artefact"])].sort_values("surface_id")
    allrows = d.assign(o=d.lit_confidence.map(order)).sort_values("surface_id")
    notes = ""
    for key, name in NOTES:
        md = (V0 / "literature" / f"matches_{key}.md").read_text()
        notes += f'<details><summary>{escape(name)}</summary><div class="md">{markdown.markdown(md, extensions=["tables"])}</div></details>'
    css = """
:root{--bg:#fbfaf7;--fg:#1d1c1a;--muted:#6b6862;--line:#e4e1d9;--card:#fff;--accent:#2f6fd6;--warn-bg:#fdf3e2;--warn:#7a4a0c;--link:#1f5f8b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--fg:#ebe9e3;--muted:#9c998f;--line:#2c2b28;--card:#1b1b19;--accent:#6aa3ff;--warn-bg:#2a2213;--warn:#e3bf7c;--link:#8cc3ea;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141413;--fg:#ebe9e3;--muted:#9c998f;--line:#2c2b28;--card:#1b1b19;--accent:#6aa3ff;--warn-bg:#2a2213;--warn:#e3bf7c;--link:#8cc3ea;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15.5px/1.6 "Source Serif 4",Georgia,serif}
.wrap{max-width:1000px;margin:0 auto;padding-block:40px 80px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1,h2,h3,table,summary,.sub{font-family:"IBM Plex Sans",system-ui,sans-serif}h1{font-size:28px;margin:0 0 6px}
.sub{color:var(--muted);font-size:14px;margin:0 0 24px}h2{font-size:21px;margin:42px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
a{color:var(--link)}.box{background:var(--warn-bg);color:var(--warn);border-radius:6px;padding:12px 16px;margin:18px 0;font-family:"IBM Plex Sans",sans-serif;font-size:14px}
.box ul,ul.f{padding-left:20px}ul.f li{margin-bottom:8px}.tbl{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
details{border:1px solid var(--line);border-radius:6px;margin:10px 0;background:var(--card)}summary{cursor:pointer;padding:10px 14px;font-weight:500}
.md{padding:0 16px 12px;font-size:14px}.md table{font-size:12.5px}
"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fault Surfaces v0 Literature</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>{css}</style></head><body><div class="wrap">
<h1>Fault surfaces v0: literature matches</h1>
<p class="sub">155 rectangular faults fitted to catalog v4 seismicity, each compared with published faults,
ridge and transform segments and seismic zones. Searches done 2026-11-22, one per region; every DOI was
checked on Crossref or DataCite unless marked in the notes. Strikes and map positions were compared;
v4 depths are biased shallow and its dips are weak.</p>
<div class="box"><b>Read first</b><ul>{FINDINGS}</ul></div>
<h2>Summary by region</h2>
{table(ct, ["region"] + CLASSES + ["total"], ["Region"] + CLASSES + ["Total"])}
<p>Confidence of the known-structure matches: {", ".join(f"{k} {v}" for k, v in d[d.lit_class == "known structure"].lit_confidence.value_counts().reindex(list(order), fill_value=0).items())}.</p>
<h2>Strong matches (high or medium-high)</h2>
{table(hi, ["surface_id", "lit_match", "lit_confidence", "strike", "dip", "length_km", "n_events"], ["Surface", "Match", "Conf.", "Strike", "Dip", "Length km", "Events"])}
<h2>Candidate new structures</h2>
<p>Surfaces that the searches could not tie to a published structure, or that suggest a structure
beyond the mapped one. Check each against the relocated catalog (Track A) before any claim.</p>
{table(new, ["surface_id", "lit_match", "lit_confidence", "lon", "lat", "strike", "dip", "n_events"], ["Surface", "What it is", "Conf.", "Lon", "Lat", "Strike", "Dip", "Events"])}
<h2>Quarry blasts and artefacts</h2>
{table(bad, ["surface_id", "lit_class", "lit_match"], ["Surface", "Class", "Note"])}
<h2>All surfaces</h2>
{table(allrows, ["surface_id", "lit_class", "lit_confidence", "lit_match", "strike", "dip", "length_km", "n_events"], ["Surface", "Class", "Conf.", "Match", "Strike", "Dip", "Length km", "Events"])}
<h2>Notes from the regional searches</h2>
<p>The full write-ups, with the evidence for each surface, the known structures of each region and the
reference lists with verification status. USGS ComCat moment tensors were used only to compare surfaces
with known earthquakes, never to build or steer the surfaces.</p>
{notes}
<h2>Limits</h2>
<ul class="f"><li>Matches rest on v4 map positions and strikes. Depth and dip will change with the relocation, and some
matches (especially the deep Blanco and Mendocino planes) may change with them.</li>
<li>Some sources were read as abstracts or preprints only (for example Lange et al. 2026 through its arXiv
version, Journeau et al. 2026, Chaytor et al. 2004, the Ren et al. EGU abstract); the notes say which.</li>
<li>The search briefs contained four wrong DOIs (Brocher 2001, Sherrod 2004, Pratt 1997, Ichinose 2006); the notes give the right ones.</li></ul>
</div></body></html>"""
    (V0 / "fault_surfaces_v0_literature.html").write_text(html)
    print("wrote", V0 / "fault_surfaces_v0_literature.html")


if __name__ == "__main__":
    main()
