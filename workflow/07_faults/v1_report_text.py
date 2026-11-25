"""Text of the fault surfaces v1 report (numbers computed from the output tables; called by v1_report.py)."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#5b5b60;--line:#d9d9de;--acc:#2c5d8f;--box:#f5f6f8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#e9e9ec;--mut:#a2a2a8;--line:#36363c;--acc:#8fb8e8;--box:#1e1e22}}
:root[data-theme="dark"]{--bg:#141416;--fg:#e9e9ec;--mut:#a2a2a8;--line:#36363c;--acc:#8fb8e8;--box:#1e1e22}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0 auto;max-width:1180px;padding:16px}
h1{font-size:1.55em;margin:.4em 0} h2{font-size:1.2em;margin-top:1.8em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{font-size:1.02em;margin-top:1.3em} p,li{max-width:62em} .mut{color:var(--mut)}
.box{background:var(--box);border:1px solid var(--line);border-radius:6px;padding:.6em 1em;margin:1em 0}
.tw{overflow-x:auto;margin:.6em 0 1.2em} table{border-collapse:collapse;font-size:12.5px}
th,td{border-bottom:1px solid var(--line);padding:3px 7px;text-align:left;vertical-align:top} th{position:sticky;top:0;background:var(--box)}
figure{margin:1em 0} img{max-width:100%;height:auto;background:#fff;border:1px solid var(--line)} figcaption{color:var(--mut);font-size:13px}
code{font-size:12.5px}
"""


def n(x):
    return f"{int(x):,}"


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


def text(x, FIGS, img, table, REG):
    narrative(x, table)
    s, pairs, st0, cmp1, rem = x["summ"], x["pairs"], x["st0"], x["cmp1"], x["rem"]
    sep, mlog, sel, reg0, reg1, asg, res = x["sep"], x["mlog"], x["sel"], x["reg0"], x["reg1"], x["asg"], x["res"]
    cat = x["cat"]
    nloc = int(cat.located.sum())
    nnone = int((~cat.located).sum())
    nfit = int(sel.loc[sel.rule == "used for the fit", "events"].iloc[0])
    ac = asg.assignment.value_counts()
    eq = asg.assignment != "not an earthquake"
    nmem = int(ac.get("fit member", 0))
    nass = int(ac.get("assigned", 0))
    namb = int(ac.get("ambiguous", 0))
    nbl = int(asg.assignment.str.startswith("non-earthquake timing").sum())
    s0 = pd.read_csv(__import__("v1_common").V0DIR / "fault_surfaces_v0_summary.csv")
    c1 = cmp1.change.value_counts()
    c0 = st0.change.value_counts()
    steep = int((s.dip >= 45).sum())
    m = pairs[(pairs.change == "matched") & pairs.v1_kept]
    rr = np.abs(res.resid_km) / res.sigma_n_km
    stab = int(s.stable.sum())
    blanco = s[s.region == "BL"]
    deep_bl = blanco[blanco.depth_km > 20]
    lowok = s[s.frac_depth_ok < 0.5]
    above = s[s.n_vertices_above_surface > 0]
    new = cmp1[cmp1.change == "new"].merge(s, left_on="v1_surface", right_on="surface_id", suffixes=("", "_s"))
    lost = st0[st0.change == "lost"]
    carried = cmp1[cmp1.lit_status.fillna("").str.startswith("carried") & ~cmp1.lit_status.fillna("").str.contains("re-check")]
    recheck = cmp1[cmp1.lit_status.fillna("").str.contains("re-check")]

    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Fault surfaces v1</title><style>{CSS}</style></head><body>"]
    h.append("<h1>Fault surfaces v1 from catalog v5.1</h1>")
    h.append(f"<p class='mut'>2026-11-25. Plane-based fault surfaces refitted on catalog v5.1 (NonLinLoc in the 3D model v0.3.1, "
             f"re-timed picks), compared surface by surface with v0 (catalog v4). Code: <code>workflow/07_faults/v1_*.py</code> "
             f"on branch analysis/v51-faults. Private working product.</p>")

    h.append("<h2>Summary</h2><ul>")
    h.append(f"<li><b>{len(s)} surfaces</b> ({steep} steep, dip &ge; 45&deg;) from {n(nfit)} well-constrained events "
             f"(v0: {len(s0)} surfaces from 76,281 events). {n(nmem)} events ({pct(nmem, nfit)} of the fit set) are fit members "
             f"of a surface (v0: 53,291, 70%).</li>")
    h.append(f"<li><b>Every located event</b> ({n(nloc)}; {n(nnone)} rows without a hypocentre left out) has an assignment: "
             f"{n(nmem)} fit members, {n(nass)} assigned to one surface within twice their combined uncertainty and within 10 km, "
             f"{n(namb)} compatible with two or more surfaces (ambiguous, not assigned), "
             f"{n(ac.get('none', 0))} on no surface, {n(ac.get('not an earthquake', 0))} non-earthquakes (event_flag) and "
             f"{n(nbl)} events of surfaces removed for their weekday-daytime timing.</li>")
    h.append(f"<li><b>Against v0</b> (by shared fit members): " + ", ".join(f"{k} {v}" for k, v in c1.items()) +
             f" among the v1 surfaces; of the {len(s0)} v0 surfaces: " + ", ".join(f"{k} {v}" for k, v in c0.items()) + ".</li>")
    h.append(f"<li><b>Literature:</b> {len(carried)} v1 surfaces carry a v0 match unchanged, {len(recheck)} carry one marked for "
             f"re-check (split, merged or moved), and {int(c1.get('new', 0))} are new and need a literature check "
             f"(<code>fault_surfaces_v1_literature_todo.csv</code>).</li>")
    h.append(f"<li><b>Uncertainty.</b> Median sigma_total of the surface position is {s.sigma_total_median_km.median():.2f} km "
             f"(10th-90th percentile {s.sigma_total_median_km.quantile(.1):.2f}-{s.sigma_total_median_km.quantile(.9):.2f} km). "
             f"Tier A catalog errors match the scatter of events about their surfaces; tier B and C errors are larger than it "
             f"(median |normal residual| / event error {np.median(rr):.2f} over all members, 0.67 if the errors described the scatter).</li>")
    h.append("</ul>")
    h.append("<div class='box'><b>Findings to read first</b>" + x.get("findings", "") + "</div>")

    h.append("<h2>Data and event selection</h2>")
    h.append("<p>Fit set (PI choice of 2026-11-25): tiers A-C located by NonLinLoc, without floor_limited, far_only and deep_pass "
             "events, without ComCat explosions, quarry blasts and other non-earthquakes and without the Texada quarry_blast_likely "
             "events. Axial Seamount stays out as in v0 (published fault maps there are better). Rows are removed by the first rule "
             "that applies:</p>")
    h.append(table(sel, names=["Rule", "Events"]))
    h.append("<p>Per-event error: sigma_h = sqrt((h_min&sup2; + h_max&sup2;)/2) / 1.515 (68 % semi-axes to one sigma, the "
             "convention of the tomography code; the catalog has no ellipse azimuth) and sigma_z = sig_z. Fit-set medians: "
             f"sigma_h {cat.sh[cat.fit].median():.2f} km, sigma_z {cat.sz[cat.fit].median():.2f} km; by tier "
             + ", ".join(f"{t} {cat.sh[cat.fit & (cat.tier == t)].median():.2f}/{cat.sz[cat.fit & (cat.tier == t)].median():.2f} km" for t in "ABC")
             + ". Depths are km below sea level.</p>")

    h.append("<h2>Method</h2><ol>")
    h.append("<li><b>Detection</b>: the v0 method and parameters (<code>fault_surfaces_planes.py</code>): sequential local RANSAC "
             "with growth, tolerance 1.5 &times; sqrt((1.5 km |n_h|)&sup2; + (3 km n_z)&sup2;), &ge; 30 events, planarity &le; 0.5, "
             "dip &ge; 30&deg;, k-planes reassignment, merging. One change: the dip rule is also applied after every reassignment "
             "pass (in v0, 24 of the 141 surfaces ended flatter than 30&deg; after reassignment).</li>")
    h.append("<li><b>Blanco strands</b> (rule of <code>split_strands.py</code>, now automatic): steep Blanco planes within 35&deg; "
             "of plate motion and mostly outside the Cascadia Depression box are re-searched as map-view strands in the three v0 pools "
             "(west, Blanco Ridge, Gorda Depression). Overlapping parallel strands stay apart only if Ashman's D &gt; 2 overall and "
             "in every year where both have &ge; 15 events; pairs whose depth offset would need a single fault dipping &ge; 60&deg; "
             "are merged (offset follows depth, the reason v0 merged its two 'deeper band' strands).</li>")
    h.append("<li><b>Rectangles</b> as v0 (2-98 % extent along strike and dip + 0.5 km, plane or gentle quadratic by BIC, 1 km mesh).</li>")
    h.append("<li><b>Uncertainty per surface</b>: bootstrap of the members (100 draws), location Monte Carlo moving every member by its "
             "own catalog error (100 draws; v0 used nominal 1.5/1.5/3 km), model choice, combined; strike/dip intervals from each; "
             "centroid spread; stability over the four detection reruns of v0 (Jaccard overlap).</li>")
    h.append("<li><b>Screening</b>: surfaces with &ge; 60 % of their events on weekdays 08-17 local time are removed (quarry blasts "
             "or other non-earthquake signals; catalog average 28 %); Blanco surfaces centred deeper than 25 km with fewer than 30 "
             "depth-ok events are removed as deep artefacts (v0 removed three such surfaces after its literature check); a final dip "
             "&lt; 30&deg; would also remove a surface (none left after the dip rule in detection).</li>")
    h.append("<li><b>Assignment of every located event</b>: distance d to the surface (normal offset inside the rectangle, plus the "
             "in-plane distance outside it); z = d / sqrt(s_e&sup2; + s_s&sup2; + W&sup2;) with s_e the event's error along that "
             "direction, s_s the surface's median sigma_total and W its zone width once the member errors are taken out. "
             "Compatible: z &le; 2 and d &le; 10 km. Assigned: exactly one compatible surface; ambiguous: two or more.</li>")
    h.append("<li><b>Comparison with v0</b>: v0 members (catalog v4 IDs, the same as v5.1 IDs) against v1 fit members. A v0 and a v1 "
             "surface are linked when they share &ge; 15 events and these are &ge; 50 % of the v0 surface's re-found events, or &ge; "
             "50 % of the v1 surface's events that were on a v0 surface, or &ge; 30 % of both. Connected groups: matched (1-1), split "
             "(1-n), merged (n-1), split and merged, new, lost. Removed v1 surfaces take part in the links, so a v0 surface whose "
             "events now form a removed surface is reported as lost with that reason.</li></ol>")

    h.append("<h2>Maps</h2>")
    h.append(img("map_v0_v1", "Rectangles of v0 (top, catalog v4) and v1 (bottom, catalog v5.1) over the located events of v5.1."))
    h.append(img("assignment_map", "Assignment of the located v5.1 events."))
    h.append(img("regions", "Share of events on a fault by region."))
    h.append(table(reg1, names=list(reg1.columns)))

    h.append("<h2>Comparison with v0</h2>")
    h.append(img("matched_changes", "Matched surfaces: change of strike, dip, depth and position."))
    if len(m):
        h.append(f"<p>Matched pairs (n = {len(m)}): median strike change {m.strike_change.median():.0f}&deg;, median |dip change| "
                 f"{m.dip_change.abs().median():.0f}&deg;, median centroid shift {m.centroid_shift_h_km.median():.1f} km horizontally and "
                 f"{m.centroid_shift_z_km.median():+.1f} km in depth (the members' own median v4 &rarr; v5.1 depth shift: "
                 f"{m.v0_members_median_shift_z_km.median():+.1f} km).</p>")
    h.append("<h3>Every v1 surface</h3>")
    cols = ["surface_id", "region", "n_events", "lon", "lat", "depth_km", "strike", "strike_68", "dip", "dip_68", "dip_depth_ok",
            "length_km", "width_km", "sigma_total_median_km", "frac_depth_ok", "stable", "change_from_v0", "v0_surfaces", "lit_status"]
    h.append(table(s[[c for c in cols if c in s]]))
    h.append("<h3>Every v0 surface: what became of it</h3>")
    h.append(table(st0[["v0_surface", "change", "v1_surfaces", "v1_removed", "v0_n_events", "located_in_v51", "in_v1_fit_set",
                        "fit_members_of_kept_v1", "top_v1_surfaces", "members_median_shift_z_km", "v0_lit_class", "note"]]))
    h.append("<h3>Lost v0 surfaces</h3>")
    h.append(table(lost[["v0_surface", "v0_n_events", "v0_dip", "v0_depth_km", "in_v1_fit_set", "fit_members_of_kept_v1",
                         "top_v1_surfaces", "v0_lit_class", "v0_lit_match", "note"]]))
    h.append("<h3>New v1 surfaces (literature check needed)</h3>")
    h.append(table(new[["v1_surface", "region", "n_events", "lon", "lat", "depth_km", "strike", "dip", "length_km",
                        "frac_depth_ok", "stable", "nearest_v0", "nearest_v0_km", "nearest_v0_normal_angle"]]))

    h.append("<h2>Blanco strands</h2>")
    h.append(x.get("blanco", ""))
    if len(mlog):
        h.append("<p>Strand pairs merged by the per-year and depth tests:</p>" + table(mlog))
    if len(sep):
        h.append("<p>Overlapping parallel strands kept apart:</p>" + table(sep))

    h.append("<h2>Screening</h2>")
    if len(rem):
        h.append(table(rem))
    h.append("<h2>Uncertainties</h2>")
    h.append(img("calibration", "Left: member residuals normal to their surface divided by the event's catalog error along the "
                 "normal. Middle: observed scatter / catalog error per surface. Right: median sigma_total against the number of events."))
    h.append(x.get("uncertainty", ""))
    h.append("<h2>Checks and limits</h2><ul>")
    h.append(f"<li>{len(above)} surfaces have mesh vertices above the ground or seafloor (GMT earth_relief 01m): "
             f"the rectangle is extrapolated 0.5 km and to the 2-98 % extent of the members; no hypocentre is above the surface "
             f"(catalog v5.1). Listed in the summary (n_vertices_above_surface).</li>")
    h.append(f"<li>{len(lowok)} surfaces have fewer than half of their events with depth_quality ok (Blanco outside the 2012-13 OBS "
             f"deployment, Gorda far from stations): their dips and depth extents rest on poorly constrained depths; "
             f"dip_depth_ok and events_depth_ok_2pct_km/98pct_km give the same from the ok events only.</li>")
    h.append(x.get("limits", ""))
    h.append("</ul>")
    h.append("<h2>Proposed decisions</h2><ol>" + x.get("decisions", "") + "</ol>")
    h.append("<h2>Files</h2>" + x.get("files", ""))
    h.append("</body></html>")
    return "\n".join(h)


def narrative(x, table):
    """Findings, Blanco notes, uncertainty notes, limits, proposed decisions and file list (HTML snippets)."""
    s, pairs, st0, cmp1, rem, mlog, sep = x["summ"], x["pairs"], x["st0"], x["cmp1"], x["rem"], x["mlog"], x["sep"]
    m = pairs[(pairs.change == "matched") & pairs.v1_kept]
    c0 = st0.change.value_counts()
    c1 = cmp1.change.value_counts()
    msh, br = x["msh"], x["br"]
    bk = x.get("brook")
    cal = x["cal"].set_index("tier")
    deepbl = s[(s.region == "BL") & (s.depth_km > 20)]
    en = s[(s.region == "EN")]
    rem_t = rem[rem.reason.str.startswith("non-earthquake")] if len(rem) else rem
    rem_msh = rem_t[(rem_t.lat.between(46.05, 46.3)) & (rem_t.depth_km > 30)] if len(rem_t) else rem_t
    n_dip_merge = int(mlog.reason.str.startswith("depth").sum()) if len(mlog) else 0
    n_year_merge = int(mlog.reason.str.startswith("per-year").sum()) if len(mlog) else 0
    f = []
    f.append(f"<li><b>Fewer events on a fault than v0, for a stated reason.</b> {int((s.region != '').sum())} surfaces hold "
             f"{int(s.n_events.sum()):,} fit members. The first v1 run, with the v0 code unchanged, found "
             f"{', '.join(str(x['nodip']).split(', ')[:2])} (74 %), but 26 of its planes ended flatter than 30&deg; after reassignment, as 24 of v0's 141 did. "
             f"With the dip rule applied in every reassignment pass, flat clouds no longer pass as faults; the share on a fault "
             f"drops most in Puget (v0 61 %, v1 34 %) and onshore, where v0 kept low-dip layers (PU001, PU002, ON002).</li>")
    if len(m):
        f.append(f"<li><b>Matched surfaces moved with their events.</b> {len(m)} v0 surfaces have one v1 counterpart. Their "
                 f"centroids moved a median {m.centroid_shift_h_km.median():.1f} km horizontally and {m.centroid_shift_z_km.median():+.1f} km "
                 f"in depth, close to the median v4 &rarr; v5.1 depth shift of their events ({m.v0_members_median_shift_z_km.median():+.1f} km). "
                 f"Strike changed by a median {m.strike_change.median():.0f}&deg; (normals {m.normal_angle.median():.0f}&deg; apart); "
                 f"{int((m.strike_change <= 10).sum())} kept their strike within 10&deg; and {int((m.normal_angle > 45).sum())} turned by more "
                 f"than 45&deg;. The other v0 surfaces: {int(c0.get('merged', 0))} merged, {int(c0.get('split', 0))} split, "
                 f"{int(c0.get('split and merged', 0))} in mixed groups, {int(c0.get('dispersed', 0))} dispersed over several v1 surfaces, "
                 f"{int(c0.get('lost', 0))} lost; {int(c1.get('new', 0))} v1 surfaces are new.</li>")
    f.append(f"<li><b>Blanco depths moved down; rectangles reach 30-50 km.</b> The events of the v0 Blanco Ridge strand (BL003a, "
             f"{br['n']:,} events) went from a median {br['v4']:.1f} km (v4) to {br['v51']:.1f} km (v5.1). Against the Lange et al. (2026) OBS "
             f"catalog, v5.1 tier A-C Blanco Ridge events are a median 2.9 km deeper with a 90th percentile of +13 km, Gorda Depression "
             f"+5.6 km (+16.6 km) (catalog paired_v5.1.md). This deep tail sets the bottom of the Blanco rectangles. "
             f"{len(deepbl)} Blanco surfaces are centred below 20 km ({', '.join(deepbl.surface_id)}), and one deep surface under the "
             f"Cascadia Depression with only 18 depth-ok events was removed as an artefact, as v0 did for three.</li>")
    cas = x["cas"]
    b4 = s.set_index("surface_id")
    b4 = b4.loc["v1_BL004"] if "v1_BL004" in b4.index else None
    f.append("<li><b>The Cascadia Depression N-S faults are not recovered as N-S planes.</b> The events of v0 BL005, BL008 and BL012 "
             "(N-S normal faults, high-confidence literature match) keep their N-S map elongation in v5.1 (azimuths "
             + ", ".join(f"{r.v51_azimuth}&deg;" for r in cas.itertuples()) + "). The v1 detection first fitted a plane striking "
             + (f"{b4.strike:.0f}&deg; and dipping {b4.dip:.0f}&deg; ({b4.n_events:,} events, {b4.length_km:.0f} km long)" if b4 is not None else "NW")
             + " through the dense cluster and on to the western Blanco Ridge (v1_BL004), then split the rest "
             "(" + "; ".join(f"{r.v0_surface}: {r.v1_surfaces}" for r in cas.itertuples()) + "). This comes from the detection order, "
             "not from the relocation. A fit restricted to the Cascadia Depression is needed before v1 is used there.</li>")
    if len(rem_msh):
        r_ = rem_msh.iloc[0]
        f.append(f"<li><b>Deep events under Mount St. Helens are probably a recurring non-earthquake signal.</b> Surface {r_.removed_id} "
                 f"({r_.n_events} events at {r_.depth_km:.0f} km under the volcano) was removed for its timing "
                 f"({r_.weekday_daytime_fraction:.0%} weekday daytime). In the Mount St. Helens box, {msh['deep']} of {msh['n']:,} located "
                 f"events are deeper than 30 km ({msh['zmin']:.0f}-{msh['zmax']:.0f} km; tiers " + ", ".join(f"{k} {v}" for k, v in sorted(msh['tiers'].items()))
                 + f"; median sig_z "
                 f"{msh['sigz']:.0f} km although stations are a few km away), and {msh['wed']} of them occur on Wednesdays between "
                 f"19:14 and 19:20 UTC (on Tuesdays at that time: {msh['tue']}). A weekly repeat at the same minute points to a test or "
                 f"calibration signal, not earthquakes. This is a catalog problem to pass on, not a fault.</li>")
    if bk:
        f.append(f"<li><b>The 2012 Brookings swarm is now at about {bk['z']:.0f} km.</b> {bk['sid']} ({bk['n']} events, {bk['y2012']} "
                 f"in 2012; v0 GO020, a candidate new structure at about 12 km) lies at {bk['zmin']:.0f}-{bk['zmax']:.0f} km in v5.1, "
                 f"against a median {bk['v4']:.0f} km in v4; depth_quality is ok for {bk['ok']:.0%} of its events, median sig_z "
                 f"{bk['sigz']:.0f} km, nearest station {bk['dmin']:.0f} km. Slab2 puts the slab top at about 15-20 km there. "
                 f"This depth needs a check against ComCat and the OBS data before the surface is interpreted.</li>")
    if len(en):
        f.append(f"<li><b>Endeavour-Explorer surfaces sit at {en.depth_km.min():.0f}-{en.depth_km.max():.0f} km</b> "
                 f"({len(en)} surfaces) on events whose depths the catalog marks as unconstrained (ridge regions). "
                 f"Their map traces can be used; their depths and dips cannot.</li>")
    f.append(f"<li><b>Catalog errors and scatter.</b> Tier A errors describe the scatter of events about their surface (median "
             f"|residual|/error {cal.loc['A', 'median_abs_resid_over_sigma']:.2f}, 0.67 expected). For tiers B and C the errors are "
             f"larger than the scatter ({cal.loc['B', 'median_abs_resid_over_sigma']:.2f}, {cal.loc['C', 'median_abs_resid_over_sigma']:.2f}). "
             f"The RMS residual is {cal.rms_resid_km.min():.1f}-{cal.rms_resid_km.max():.1f} km for every tier, close to what the "
             f"selection tolerance allows (about 2.3 km for vertical faults), so the zone widths are set by the method more than by the "
             f"locations.</li>")
    x["findings"] = "<ul>" + "".join(f) + "</ul>"

    rj = x.get("run", {})
    dips = mlog.reason.str.extract(r"dipping (\d+) deg")[0].dropna().astype(float) if len(mlog) else pd.Series([], dtype=float)
    b = [f"<p>{rj.get('blanco_strands', '?')} strands replace {rj.get('blanco_parents', '?')} Blanco planes. {n_dip_merge} "
         f"overlapping pairs were merged because their depth offset would need a single fault dipping "
         f"{dips.min():.0f}-{dips.max():.0f}&deg; (the Blanco faults are near vertical, so the offset is read as "
         f"depth-dependent location error, as for v0's 'deeper band' strands), and {n_year_merge} because the separation failed "
         f"in at least one year. Without the depth rule, v1 would keep more parallel strands; the table lists every merge.</p>"]
    br9 = s[s.surface_id.isin(["v1_BL009", "v1_BL015"])]
    if len(br9):
        b.append("<p>Two Blanco Ridge strands south of the main strand v1_BL005, " + ", ".join(
            f"{r.surface_id} ({r.n_events} events, {r.frac_depth_ok:.0%} depth-ok, centroid {r.depth_km:.0f} km)" for r in br9.itertuples())
            + ", pass the depth test because the test uses depth-ok events only, which sit at the same median depth as v1_BL005. "
            "Most of their events have unconstrained depths. They are where v0's merged 'deeper band' was, and need the same check "
            "before being read as separate faults.</p>")
    x["blanco"] = "".join(b)

    x["uncertainty"] = (
        f"<p>Per surface (1 sigma normal to the surface, median over vertices): sigma_total {s.sigma_total_median_km.median():.2f} km "
        f"(10th-90th percentile {s.sigma_total_median_km.quantile(.1):.2f}-{s.sigma_total_median_km.quantile(.9):.2f}), of which "
        f"location Monte Carlo {s.sigma_loc_median_km.median():.2f}, bootstrap {s.sigma_boot_median_km.median():.2f} and model choice "
        f"{s.sigma_method_median_km.median():.2f} km. The location term uses the catalog errors, which for tiers B and C are larger "
        f"than the scatter, so it is an upper bound. It treats errors as independent between events; errors shared by nearby events "
        f"(velocity model, station terms) move a whole surface and are not in it; the v0 &rarr; v1 shift of matched surfaces "
        f"(median {m.centroid_shift_h_km.median():.1f} km horizontally) shows their size between two location runs. "
        f"Strike and dip: bootstrap 68 % interval widths, median {_w(s.strike_68):.1f}&deg; and {_w(s.dip_68):.1f}&deg;; with the "
        f"location Monte Carlo {_w(s.strike_68_with_loc_mc):.1f}&deg; and {_w(s.dip_68_with_loc_mc):.1f}&deg;. Centroid: "
        f"{s.centroid_sigma_h_boot_km.median():.2f} km horizontally and {s.centroid_sigma_z_boot_km.median():.2f} km in depth (bootstrap). "
        f"{int(s.stable.sum())} of {len(s)} surfaces are stable (median Jaccard &ge; 0.5 over the four detection reruns; v0: 50 of 141). "
        f"Per tier:</p>" + table(x["cal"]))
    x["limits"] = (
        "<li>Rectangles are crude: one plane or gentle quadratic over the 2-98 % extent of the members. Curved or branching faults "
        "come out as several rectangles.</li>"
        "<li>No relative relocation: the scatter about the surfaces (RMS about 1.2 km) is still wider than double-difference "
        "catalogs show for the same faults.</li>"
        "<li>ComCat focal mechanisms were not used to steer or check the surfaces (PI rule).</li>"
        "<li>The mesh vertex elevations (surface_elev_km) use GMT earth_relief 01m, not the merged GMRT surface of the catalog.</li>")
    d = [
        "Dip rule in every reassignment pass (v1_detect.reassign_dip): a plane flatter than 30&deg; after reassignment is dropped "
        "and its events go back to the other planes. v0 checked the dip only at detection and merging; 24 of its 141 surfaces "
        "ended flatter. Effect on v5.1: 192 &rarr; 161 planes, 74 % &rarr; 67 % of the fit set on a plane.",
        "Event selection: 'deep-pass-only' read as deep_pass = True (83 tier A-C events); anss_other_event and anss_sonic_boom "
        "also excluded as non-earthquakes (1 event); Axial left out as in v0; the 2 loc_source comcat rows (tier U) are assigned "
        "with nominal errors of 1.5 km (horizontal) and 3 km (depth).",
        "Per-event error: sigma_h = sqrt((h_min&sup2; + h_max&sup2;)/2) / 1.515, isotropic (no ellipse azimuth in the catalog); "
        "sigma_z = sig_z.",
        "Blanco strands made automatic: parents are steep Blanco planes within 35&deg; of 111&deg; with under half of their events in "
        "the Cascadia Depression box (-128.82 to -128.60 E, 43.74 to 43.88 N); events in that box are kept out of the strand pools; "
        f"per-year Ashman D test ({n_year_merge} merges); pairs merged when their depth-ok median depths would need a single fault "
        f"dipping &ge; 60&deg; ({n_dip_merge} merges; v0 merged two such pairs by judgement).",
        "Screening: surfaces with &ge; 60 % of events on weekdays 08-17 local time removed (9; their 1,137 events are marked and never "
        "assigned); Blanco surfaces centred deeper than 25 km with &lt; 30 depth-ok events removed as deep artefacts (1).",
        "Assignment: compatible when z = d / sqrt(s_e&sup2; + s_s&sup2; + W&sup2;) &le; 2 and d &le; 10 km; assigned only when exactly "
        "one surface is compatible, otherwise 'ambiguous' with the two nearest listed. Without the 10 km cap and the uniqueness "
        "condition, events with errors of 10-15 km were 'compatible' with surfaces up to 70 km away.",
        "v0 comparison: link when &ge; 15 shared fit members that are &ge; 50 % of either side, or &ge; 30 % of both; literature carried "
        "unchanged only for matched surfaces with normals within 25&deg; and centroids within 10 km; all other carried matches "
        "marked 're-check'.",
        "Surface IDs: v1_&lt;region&gt;&lt;nnn&gt; (v0 region boxes, numbered by size); removed surfaces v1x_nn.",
        "Not done here, proposed as next steps: a fit restricted to the Cascadia Depression; a check of v1_BL009/v1_BL015 against the "
        "OBS-era depths; passing the Mount St. Helens weekly deep events and the Brookings swarm depth to the catalog owner.",
    ]
    x["decisions"] = "".join(f"<li>{v}</li>" for v in d)
    x["files"] = (
        "<p>In the worktree, <code>data/catalog_v5/v5.1_analysis/faults/</code>: "
        "<code>fault_surfaces_v1_summary.csv</code> (parameters, flags, carried literature), <code>_uncertainty.csv</code>, "
        "<code>_rectangles.geojson</code> (four corners per surface), <code>_members.csv</code> (fit members), "
        "<code>_assignment.csv.gz</code> (every located event), <code>_assignment_by_region.csv</code>, <code>_selection.csv</code>, "
        "<code>_removed.csv</code> and <code>_removed_members.csv</code>, <code>_strand_separation.csv</code>, "
        "<code>_strand_merges.csv</code>, <code>_traces.csv</code>, <code>_v0_pairs.csv</code> (every linked v0-v1 pair), "
        "<code>_v0_status.csv</code> (every v0 surface), <code>_v0_comparison.csv</code> (every v1 surface, with literature), "
        "<code>_literature_todo.csv</code> (new surfaces), the PNG figures and this report. "
        "On /wd1/mdenolle_data/catalog_v5/v5.1_analysis/faults/: <code>fault_surfaces_v1.geojson</code> (1 km triangle mesh, "
        "CRESCENT CFM layout), <code>_vertices.csv.gz</code> (per-vertex sigmas), <code>_ensemble.csv.gz</code> (100 draws per "
        "vertex), <code>_assignment.csv.gz</code>, work/ (detection runs, logs, member residuals) and RESUME.md.</p>")


def _w(col):
    v = []
    for x in col.dropna():
        a = [float(y) for y in str(x).split(";")]
        d = (a[1] - a[0]) % 360
        v.append(min(d, 360 - d) if d > 180 else d)
    return float(np.median(v)) if v else float("nan")
