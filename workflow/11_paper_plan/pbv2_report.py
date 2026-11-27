#!/usr/bin/env python3
"""Plate boundaries v2 (D-B116): HTML report section, offsets summary table, manuscript bullets.

  BIG/report_boundaries.html          method, bias tests, results, comparison with other models, limits
  OUT/offsets_summary.csv             offsets per section and segment (PB2002, bathymetric axis, verdict)
  OUT/segment_change_v1_v2.csv        what PB_AXES=v2 would change in the segment rule's assignment (v6.1, tier A-C)
  OUT/manuscript_bullets.md           bullets for the manuscript (bullets only, PI rule)
Usage: python -I pbv2_report.py ["time label"]   (time label from `date`)
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PB_CATALOG", "/wd1/mdenolle_data/catalog_v6/v6.1/cascadia_catalog_v6.1.csv")
os.environ.setdefault("PB_LABEL", "v6.1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pbv2_common as C  # noqa: E402
from pbv2_common import P, SECTIONS  # noqa: E402

E = html.escape


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def table(df, fmt=None, cls=""):
    fmt = fmt or {}
    h = [f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{E(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                s = "" if not np.isfinite(v) else (fmt.get(c, "{:.1f}").format(v))
            else:
                s = E(str(v)) if v is not None else ""
            cells.append(f"<td>{s}</td>")
        h.append("<tr>" + "".join(cells) + "</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


def segment_change():
    c = P.load_catalog()
    c = c[c.is_eq & c.abc]
    a = P.assign_segments(c, axes="gmrt").seg
    b = P.assign_segments(c, axes="v2").seg
    ct = pd.crosstab(a.rename("v1 (gmrt, D-B113)"), b.rename("v2 (D-B116)"))
    moved = ct.stack()
    moved = moved[(moved > 0) & (moved.index.get_level_values(0) != moved.index.get_level_values(1))]
    mv = moved.sort_values(ascending=False).rename("events").reset_index()
    mv.to_csv(C.OUT / "segment_change_v1_v2.csv", index=False)
    tot = pd.DataFrame({"v1": a.value_counts(), "v2": b.value_counts()}).fillna(0).astype(int)
    tot["change"] = tot.v2 - tot.v1
    return mv, tot[tot.change != 0].sort_values("change"), int((a != b).sum()), len(c)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
    O = C.OUT
    off = pd.read_csv(O / "offsets.csv")
    tests = pd.read_csv(O / "bias_tests.csv")
    ver = pd.read_csv(O / "verdict.csv")
    seg = pd.read_csv(O / "segments.csv")
    brk = pd.read_csv(O / "breaks.csv")
    cor = pd.read_csv(O / "corners.csv")
    bic = pd.read_csv(O / "bic.csv")
    zon = pd.read_csv(O / "zones.csv")
    mod = pd.read_csv(O / "model_comparison.csv")
    rg = pd.read_csv(O / "residual_geometry.csv")
    gj = json.loads((C.BIG / "proposed_boundaries_v1.geojson").read_text())
    feats = [f["properties"] for f in gj["features"]]
    mv, tot, nmov, ntot = segment_change()
    import pickle
    with open(C.BIG / "work" / "fit.pkl", "rb") as fh:
        FF = pickle.load(fh)
    en_n = FF["res"]["JRN"]["centroids"]["EN"]["n"]
    rj = FF["res"]["JRN"]
    ln0 = rj["lines"][0]
    dB = rj["bathy_dec"]
    mm = (dB[:, 0] > 20) & (dB[:, 0] < 230)
    jrn_dev = float(np.max(np.abs(dB[mm, 1] - (ln0["a"] + ln0["b"] * (dB[mm, 0] - ln0["sm"]))))) if mm.any() else np.nan
    # ---------------------------------------------------------------- offsets summary
    rows = []
    for r in off.itertuples():
        v = ver[ver.section == r.section]
        meaningful = r.n_bins >= 5
        rows.append(dict(section=r.section, name=r.name, span=r.span, data=r.mode, n_bins_S0=r.n_bins,
                         seis_minus_pb2002_km=r.O_pb2002 if meaningful else np.nan,
                         ci95_lo=r.O_pb2002_lo if meaningful else np.nan, ci95_hi=r.O_pb2002_hi if meaningful else np.nan,
                         seis_minus_bathy_km=r.O_bathy if meaningful else np.nan,
                         proposal_minus_pb2002_median_km=r.proposal_minus_pb2002_median,
                         proposal_minus_pb2002_p10_km=r.proposal_minus_pb2002_p10,
                         proposal_minus_pb2002_p90_km=r.proposal_minus_pb2002_p90,
                         sign="+ " + C.SIGN[SECTIONS[r.section].kind],
                         verdict=(v.verdict.iloc[0] if (len(v) and r.span == "section") else "")))
    summ = pd.DataFrame(rows)
    summ.to_csv(O / "offsets_summary.csv", index=False, float_format="%.2f")
    # ---------------------------------------------------------------- numbers used in the text
    g = lambda sec, span="section": off[(off.section == sec) & (off.span == span)].iloc[0]  # noqa: E731
    vd = ver.set_index("section")
    b_bl = brk[(brk.section == "BL") & (brk.origin == "BIC fit")].reset_index(drop=True)
    b_gr = brk[brk.section == "GR"].reset_index(drop=True)
    tj = cor[cor.name.str.contains("triple")].iloc[0]
    cn = cor.set_index("name")
    jrn = seg[seg.section == "JRN"].iloc[0]
    mfz_seg = seg[seg.section == "MFZ"]
    pk = pd.read_csv(O / "picks.csv")
    nS0 = {s: int(pk[(pk.section == s)].n_S0.sum()) for s in SECTIONS}
    # ---------------------------------------------------------------- bullets
    bl1, bl2 = b_bl.iloc[0], b_bl.iloc[1]
    bullets = [
        f"- Relocated seismicity (v6.1, tier A-B, horizontal error <= 3 km) places the active Mendocino transform "
        f"{g('MFZ').O_pb2002:.1f} km (95 %: {g('MFZ').O_pb2002_lo:.1f}-{g('MFZ').O_pb2002_hi:.1f}) north of the PB2002 "
        f"trace, at the base of the north-facing slope of the Mendocino Ridge (seismicity minus bathymetric pick "
        f"{g('MFZ').O_bathy:+.1f} km); PB2002, NNR-MORVEL56, Hasterok et al. (2022) and MAPRIDGES all put it 8-10 km "
        f"farther south, on the ridge.",
        f"- The Blanco transform seismicity lies {abs(g('BL').O_pb2002):.1f} km (95 %: {abs(g('BL').O_pb2002_hi):.1f}-"
        f"{abs(g('BL').O_pb2002_lo):.1f}) south of PB2002 on average and {abs(g('BL', 'segment 1').O_pb2002):.1f} km "
        f"along the Blanco Ridge segment (up to {abs(g('BL', 'segment 1').proposal_minus_pb2002_p10):.0f} km).",
        f"- Straight segments chosen by BIC (minimum 30 km, per-break penalty) resolve the Blanco transform into three "
        f"main segments with two step-overs, each stepping north toward the Juan de Fuca Ridge: {bl1.offset_km:.1f} km at the Cascadia Depression "
        f"({abs(bl1.lon):.2f} W; 68 %: {bl1['off_p16']:.1f}-{bl1['off_p84']:.1f} km) and {bl2.offset_km:.1f} km near the "
        f"Surveyor Depression ({abs(bl2.lon):.2f} W; 68 %: {bl2['off_p16']:.1f}-{bl2['off_p84']:.1f} km); BIC does not "
        f"support a third step-over.",
        f"- The southern Gorda Ridge seismicity sits in the axial valley (seismicity minus valley pick "
        f"{g('GR').O_bathy:+.1f} km), "
        f"{min(g('GR', 'segment 0').proposal_minus_pb2002_median, g('GR', 'segment 1').proposal_minus_pb2002_median):.0f}-"
        f"{max(g('GR', 'segment 0').proposal_minus_pb2002_median, g('GR', 'segment 1').proposal_minus_pb2002_median):.0f} "
        f"km east of PB2002 south of 41.7 N; going north the ridge axis steps east by "
        + ", ".join(f"{r.offset_km:.1f} km at {r.lat:.2f} N" for r in b_gr.itertuples()) + ".",
        f"- The offsets are not location bias: the same events in catalogs v4.1-v6.1 move the Blanco and Mendocino axes "
        f"by <= {max(vd.loc['BL', 'B_ver'], vd.loc['MFZ', 'B_ver']):.1f} km; common events in three OBS-based catalogs "
        f"(Lange et al. 2026; Zhou et al. 2024; Niksejel & Zhang 2026) differ across strike by <= "
        f"{vd.loc['BL', 'B_pair']:.1f} km at Blanco and {vd.loc['MFZ', 'B_pair']:.1f} km at Mendocino; their own "
        f"seismicity sits on the same side of PB2002.",
        f"- The Mendocino triple junction, where the straight Mendocino trace meets the PB2002 deformation front, lies "
        f"at {abs(tj.lon):.3f} W, {tj.lat:.3f} N ({tj.dist_to_pb2002_tj_km:.1f} km from PB2002; +-"
        f"{tj.sig_major_km:.1f} km from the transform alone).",
        f"- Between Axial Seamount and the Endeavour segment our catalog has too few well-located events to define an "
        f"axis; a straight Axial-Endeavour line (PI) lies up to {jrn_dev:.0f} km west of the CoAxial-Northern "
        f"Symmetric axial high (bathymetric picks), so it is flagged and the bathymetric axis is kept as the "
        f"alternative.",
        "- The Gorda plate interior and the Explorer plate edges are drawn as deformation zones (80 % epicentre "
        "density), not lines; Explorer, Sovanco and Nootka are not resolved by our catalog.",
    ]
    (O / "manuscript_bullets.md").write_text("# Plate boundaries v2 (D-B116): manuscript bullets\n\n" + "\n".join(bullets) + "\n")
    # ---------------------------------------------------------------- HTML
    css = """
:root{--bg:#ffffff;--fg:#1b1b1b;--mut:#5a5a5a;--line:#d9d9d9;--acc:#D55E00;--card:#f6f6f4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161616;--fg:#e8e8e8;--mut:#a8a8a8;--line:#3a3a3a;--card:#202020}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:1.6em;margin:.2em 0}h2{margin-top:1.8em;border-bottom:2px solid var(--acc);padding-bottom:.2em}h3{margin-top:1.3em}
.mut{color:var(--mut)}table{border-collapse:collapse;font-size:12.5px;margin:.6em 0;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}th{text-align:center;background:var(--card)}
td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto;border:1px solid var(--line);margin:.4em 0;background:#fff}
.card{background:var(--card);padding:10px 14px;border-left:4px solid var(--acc);margin:.8em 0}.flag{color:#b04a00}
code{font-size:12.5px}ul li{margin:.25em 0}
"""
    sec_rows = []
    for f in feats:
        if f["type"] in ("transform", "spreading ridge") and f.get("source") != "PB2002":
            sec_rows.append(dict(section=f["section_code"], segment=f["segment"], type=f["type"], data=f.get("data", ""),
                                 source=f["source"], length_km=f["length_km"], n_picks=f["n_picks"],
                                 unc68_km=f["uncertainty_km"], minus_pb2002_median_km=f["offset_from_pb2002_km"],
                                 max_abs_from_pb2002_km=f["max_abs_offset_from_pb2002_km"], flag=f["flag"]))
    segt = pd.DataFrame(sec_rows)
    brk_show = brk.copy()
    brk_show["s_68"] = [f"{a:.0f}-{b:.0f}" if np.isfinite(a) else "" for a, b in zip(brk_show.get("s_p16", np.nan),
                                                                                      brk_show.get("s_p84", np.nan))]
    brk_show["offset_68"] = [f"{a:.1f}-{b:.1f}" if np.isfinite(a) else "" for a, b in zip(brk_show.get("off_p16", np.nan),
                                                                                          brk_show.get("off_p84", np.nan))]
    brk_show["offset_95"] = [f"{a:.1f}-{b:.1f}" if np.isfinite(a) else "" for a, b in zip(brk_show.get("off_p2.5", np.nan),
                                                                                          brk_show.get("off_p97.5", np.nan))]
    brk_show = brk_show[["section", "break_k", "kind", "origin", "lon", "lat", "s_break", "s_68", "offset_km", "offset_68",
                         "offset_95"]]
    cor_show = cor[["name", "lon", "lat", "sig_major_km", "sig_minor_km", "az_major_deg"]].copy()
    if "dist_to_pb2002_tj_km" in cor:
        cor_show["dist_to_PB2002_TJ_km"] = cor.dist_to_pb2002_tj_km
    cor_show["note"] = cor.get("note", "").fillna("") if "note" in cor else ""
    if "flag" in cor:
        cor_show["note"] = (cor_show["note"].astype(str) + " " + cor.flag.fillna("").astype(str)).str.strip()
    vshow = ver[["section", "O_pb2002", "O_lo", "O_hi", "O_bathy", "B_pair", "B_ver", "B_geo", "B",
                 "independent_own_offsets", "verdict"]].copy()
    ind = tests[tests.test == "independent catalog"][["section", "label", "n_events_window", "n_pairs", "du_median_km",
                                                       "du_lo", "du_hi", "dh_median_km", "their_herr_median_km",
                                                       "own_n_bins", "own_O_pb2002", "own_minus_ours"]]
    ind = ind[(ind.n_events_window.fillna(0) > 0)]
    vt = tests[tests.test.isin(["version", "geometry", "CI year", "selection"])][["section", "test", "label", "D_km",
                                                                                  "n_bins", "O_pb2002", "n_events"]]
    vt = vt[vt.n_bins.fillna(0) > 0]
    bic_show = bic[["section", "data", "K", "n", "chi2", "bic", "chosen_bic", "drawn"]]
    gr_res = rg.pivot_table(index=["section", "var", "lo", "hi"], values=["n", "median_residual_km"]).reset_index()
    figs = lambda n, cap: f'<figure><img src="figures/{n}" alt="{E(cap)}"><figcaption class="mut">{E(cap)}</figcaption></figure>'  # noqa: E731
    commit = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(C.ROOT), "status", "--porcelain", "workflow/11_paper_plan"],
                           capture_output=True, text=True).stdout.strip()
    inputs = [(C.CAT, "catalog v6.1"), (C.TOPO, "D-B068 merged surface"),
              (C.FAULTS / "fault_surfaces_v1_traces.csv", "fault surfaces v1 (v6.1) traces"),
              (P.PB2002, "PB2002 steps"), (C.ANSS, "ComCat/ANSS 2010-16"),
              (C.LANGE / "blanco_lange2026_abs_pairs.csv", "Lange 2026 pairs (abs)"),
              (C.LANGE / "blanco_lange2026_rel_pairs.csv", "Lange 2026 pairs (rel)")]
    inp = pd.DataFrame([dict(input=n, path=str(p), md5=md5(p)) for p, n in inputs])
    prov = pd.read_csv(C.EXT / "PROVENANCE.csv")
    prov = prov[prov.dataset.astype(str).str.startswith("pbv2_")][["dataset", "doi", "licence", "file", "md5"]]
    bl_seg1 = g("BL", "segment 1")
    H = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Plate boundaries v2</title><style>{css}</style></head><body>",
         "<h1>Plate boundaries v2: a proposed redrawing of the offshore Cascadia plate boundaries</h1>",
         f"<p class=mut>D-B116 (PI, 2026-11-27 13:34: \"we should at least propose to re-draw plate boundaries\"). "
         f"Catalog v6.1. Built {E(label)}; code {E(commit)}{' (uncommitted changes)' if dirty else ''} on branch "
         f"analysis/plate-boundaries-v2 (workflow/11_paper_plan/pbv2_*.py). <b>Status: proposal for PI review, not "
         f"adopted.</b> The segment rule is unchanged; <code>PB_AXES=v2</code> switches the plate-boundary analysis to "
         f"these traces (default stays <code>gmrt</code> = axes v1, D-B113).</p>",
         "<div class=card><b>Summary</b><ul>" + "".join(f"<li>{E(b[2:])}</li>" for b in bullets) + "</ul></div>",
         "<p>Deliverables: <code>proposed_boundaries_v1.geojson</code> (this folder), <code>data/paper_plan/"
         "plate_boundaries_v2/</code> in git (offsets_summary.csv, segments.csv, breaks.csv, corners.csv, bias_tests.csv, "
         "verdict.csv, model_comparison.csv, plate_axes_v2.csv, RULES.md, manuscript_bullets.md).</p>",
         "<h2>1. Map</h2>",
         figs("map_proposal.png", "Proposal v2 (vermilion, straight segments; dashed vermilion = step-over) against "
              "PB2002 (blue dashed) over the shaded D-B068 surface (GMRT where available). Black dots: v6.1 tier A-B "
              "earthquakes with horizontal error <= 3 km (S0). Purple: deformation zones, corners (diamonds, 68 % "
              "ellipses) and the triple junction (star). Green dotted: bathymetric alternative where it disagrees."),
         "<h2>2. Method</h2><ul>",
         "<li><b>Events (S0)</b>: v6.1, NonLinLoc 3-D, earthquakes only, tier A-B, h_max <= 3 km, ell_len3 <= 10 km "
         f"({int((C.load_v61().S0).sum()):,} events). Variants: S1 strict (h_max <= 1.5 km, ell_len3 <= 6 km); "
         "S2 tier A-C. Events are never moved or clamped.</li>",
         "<li><b>Sections</b> (straight reference frames; u positive north for transforms, east for ridges): MFZ, BL, "
         "GR, JRS (Cleft-Axial), JRN (Axial-Endeavour). North of 48.4 N (West Valley, Sovanco, Explorer) PB2002 is kept.</li>",
         "<li><b>Seismicity axis</b>: in 5 km bins along strike, the peak of the across-strike epicentre density "
         "(Gaussian kernel 1.5 km) where >= 10 events; 300 day-block bootstrap resamples (whole UTC days, so swarms "
         "stay together) give the pick uncertainty and the 68 % band.</li>",
         "<li><b>Straight segments (parsimony, PI)</b>: K breaks, each segment >= 30 km and >= 4 picks, offsets "
         "between segments allowed (step-overs); BIC = chi2 + (3K+2) ln n + K ln n with pick sigma = bootstrap sigma "
         "and the median location error in quadrature (wiggles below the location uncertainty are not resolved); exact "
         "minimum by dynamic programming. Blanco drawn with K = 2 (PI); JdF Axial-Endeavour one line (PI). A break "
         "with offset < 2 sigma_loc is drawn as a bend.</li>",
         "<li><b>Bathymetric axis</b> (independent of the locations): D-B113 picks of pb_axes.py on the D-B068 "
         "surface (Gorda Ridge axial valley, Mendocino base of the north-facing slope, JdF axial high), decimated to "
         "one per 10 km; at Blanco the MAPRIDGES v2602 transform traces (GMRT-based, Sautter et al. 2026).</li>",
         "<li><b>Combination rule</b>: per segment the seismicity line and a straight bathymetric line over the same "
         "span are compared at the two ends and the middle; within 2 sqrt(sigma_seis^2 + sigma_bathy^2) everywhere: "
         "the bathymetric line is drawn (source \"both\"); otherwise the seismicity line is drawn, the bathymetric "
         "line kept as a flagged alternative. Sections with too few picks use the bathymetric fit (source "
         "\"bathymetry\").</li>",
         "<li><b>Corners and triple junction</b>: intersections of the end segments of adjacent sections; the triple "
         "junction is the Mendocino trace's intersection with the PB2002 deformation front. Uncertainties from the "
         "bootstrap (or the line-fit covariance) of the intersecting lines.</li>",
         "<li><b>Deformation zones</b>: 80 % highest-density region of epicentres (Gaussian kernel 5 km), events "
         "within 10 km of a proposed line left out.</li>",
         "<li><b>Pre-registration</b>: RULES.md was committed (d9a48a1d, 14:01 PST) before any fit. Post-hoc "
         "changes, each written down before the next run and flagged in the outputs: <span class=flag>A1</span> GR "
         "window widened (the axis was outside it); <span class=flag>A2</span> Blanco first 60 km without tier A-B "
         "picks filled from tier A-C picks; <span class=flag>A3</span> JdF Axial-Endeavour drawn through the Axial "
         "and Endeavour seismicity centroids (PI line; seismicity between them too sparse); <span class=flag>A4</span> "
         "Axial node = Axial seismicity centroid; <span class=flag>A5</span> ridge bathymetric picks within 5 km of, or "
         "beyond, an adjacent transform dropped (the first Gorda valley picks sampled the Mendocino trough and made "
         "an oblique 34 km hook).</li></ul>",
         "<h2>3. Location bias (step 1)</h2>",
         "<p>Offsets are measured on the seismicity axis (per-bin peaks) minus the PB2002 trace in the same bin; the "
         "section offset O is the median over bins with a 95 % day-block bootstrap interval. Bias terms: "
         "B_pair = largest across-strike shift (ours minus theirs, common events) against OBS-based catalogs; "
         "B_ver = spread of the axis over catalogs v4.1, v5.1, v6.0, v6.1 (same events); B_geo = largest change when "
         "only good station geometry (gap <= 180 deg, nearest station <= 15 km) or one Cascadia Initiative year is "
         "used. Verdict rule (pre-registered): not location bias if |O| >= 2 B and an independent catalog's own "
         "axis is offset the same way.</p>",
         table(vshow, {"O_pb2002": "{:+.1f}", "O_lo": "{:+.1f}", "O_hi": "{:+.1f}", "O_bathy": "{:+.2f}",
                       "B_pair": "{:.2f}", "B_ver": "{:.2f}", "B_geo": "{:.2f}", "B": "{:.2f}"}),
         "<ul>"
         f"<li><b>MFZ</b>: +{g('MFZ').O_pb2002:.1f} km north of PB2002; the same events in v4.1 (1-D models, "
         f"GraphDD) give {tests[(tests.section == 'MFZ') & (tests.label == 'v4.1')].O_pb2002.iloc[0]:+.1f} km. "
         f"Niksejel &amp; Zhang (2026, Cascadia Initiative OBS, deep-learning catalog) put their own axis at "
         f"{tests[(tests.section == 'MFZ') & (tests.label == 'Niksejel 2026')].own_O_pb2002.iloc[0]:+.1f} km; the "
         f"land-network catalogs of the 2016 and 2024 sequences (Liu et al. 2026) at +3.8 / +4.1 km. The axis agrees "
         f"with the bathymetric base-of-slope pick ({g('MFZ').O_bathy:+.1f} km).</li>"
         f"<li><b>Blanco</b>: {g('BL').O_pb2002:+.1f} km; Lange et al. 2026 (absolute, relative), Zhou et al. 2024 "
         f"(both from the 2012-13 Blanco OBS array) and Niksejel &amp; Zhang 2026 put their own axes at -4.3 to -5.4 "
         f"km; common-event shifts across strike are {tests[(tests.section == 'BL') & tests.label.str.startswith(('Lange', 'Zhou', 'Niksejel'))].du_median_km.abs().max():.1f} km or less.</li>"
         f"<li><b>Gorda Ridge</b>: +{g('GR').O_pb2002:.1f} km over the {int(g('GR').n_bins)} bins with tier A-B picks "
         f"(mostly the southern ridge); stable over versions (B_ver {vd.loc['GR', 'B_ver']:.1f} km) and station "
         f"geometry; no OBS catalog with enough events there, so by the rule the verdict is \"partly\". The seismicity "
         f"coincides with the axial-valley pick ({g('GR').O_bathy:+.2f} km), which is independent of the locations: "
         f"PB2002 is misplaced there, not the events.</li>"
         "<li><b>ComCat/ANSS</b> common events differ by 4-14 km across strike, consistent with their own offshore "
         "errors (median horizontalError 5-8 km); they are reported, not used in B.</li>"
         "<li><b>JdF (JRS, JRN)</b>: too few well-located events for an axis; not tested.</li></ul>",
         figs("bias.png", "Offset of the seismicity axis from PB2002 (black) and the bias terms; subsets and shifts are "
              "plotted as O + shift. Grey band: O +- B."),
         "<h3>Independent catalogs</h3>",
         table(ind, {"n_events_window": "{:.0f}", "n_pairs": "{:.0f}", "du_median_km": "{:+.2f}", "du_lo": "{:+.2f}",
                     "du_hi": "{:+.2f}", "dh_median_km": "{:.2f}", "their_herr_median_km": "{:.1f}",
                     "own_n_bins": "{:.0f}", "own_O_pb2002": "{:+.1f}", "own_minus_ours": "{:+.1f}"}),
         "<p class=mut>du = across-strike shift of common events, ours minus theirs (|dt| <= 3 s, <= 50 km; Lange by "
         "the pairs of b4_lit_paired.py); own_O_pb2002 = their seismicity axis minus PB2002 (same bins and rule).</p>",
         "<h3>Versions, station geometry, deployment years, strict selection</h3>",
         table(vt, {"D_km": "{:+.2f}", "n_bins": "{:.0f}", "O_pb2002": "{:+.1f}", "n_events": "{:.0f}"}),
         "<p class=mut>D = median over common bins of (subset axis minus all-event axis).</p>",
         "<h3>Residuals from the proposed line against station geometry</h3>",
         table(gr_res, {"lo": "{:.0f}", "hi": "{:.0f}", "n": "{:.0f}", "median_residual_km": "{:+.2f}"}),
         "<p class=mut>Events within 10 km of the line. Blanco events more than 20 km from the nearest station sit "
         "1.7-2.1 km south of the line (n = 400): a small geometry effect, much smaller than the offset from PB2002.</p>",
         "<h2>4. Results (steps 2-3)</h2>",
         "<h3>Segments</h3>",
         table(segt, {"length_km": "{:.0f}", "unc68_km": "{:.2f}", "minus_pb2002_median_km": "{:+.1f}",
                      "max_abs_from_pb2002_km": "{:.1f}"}),
         "<h3>Breaks: step-overs and bends</h3>",
         table(brk_show, {"lon": "{:.3f}", "lat": "{:.3f}", "s_break": "{:.0f}", "offset_km": "{:+.1f}"}),
         f"<p>Blanco: the BIC itself picks K = 2 (three segments), the PI's reading; K = 3 raises BIC by "
         f"{float(bic[(bic.section == 'BL') & (bic.K == 3)].bic.iloc[0] - bic[(bic.section == 'BL') & (bic.K == 2)].bic.iloc[0]):.1f}, "
         f"so the data do not support more step-overs at this resolution. The first step-over (Cascadia Depression) is "
         f"well located (s 68 %: {bl1['s_p16']:.0f}-{bl1['s_p84']:.0f} km); the second lies between the Surveyor "
         f"Depression and the East Blanco Depression (s 68 %: {bl2['s_p16']:.0f}-{bl2['s_p84']:.0f} km, "
         f"{abs(bl2.lon):.2f} W at the best fit). MAPRIDGES maps four step-overs; at the Gorda Depression (first 60 km) "
         f"tier A-B picks are missing and the tier A-C line joins the Blanco Ridge segment with a "
         f"{abs(brk[(brk.section == 'BL') & (brk.origin == 'A2 junction')].offset_km.iloc[0]):.1f} km bend. The "
         f"Blanco Ridge segment ({bl_seg1.n_bins:.0f} bins) runs {abs(seg[(seg.section == 'BL') & (seg.segment == 1)].dev_mid.iloc[0]):.1f} km "
         f"north of the MAPRIDGES trace at mid-segment (flagged; both drawn). The fault surfaces v1 (v6.1) give the "
         f"individual strands (GeoJSON features \"transform strand\").</p>",
         "<h3>Corners and the triple junction</h3>",
         table(cor_show, {"lon": "{:.4f}", "lat": "{:.4f}", "sig_major_km": "{:.1f}", "sig_minor_km": "{:.1f}",
                          "az_major_deg": "{:.0f}", "dist_to_PB2002_TJ_km": "{:.1f}"}),
         "<h3>BIC per section</h3>",
         table(bic_show, {"K": "{:.0f}", "n": "{:.0f}", "chi2": "{:.1f}", "bic": "{:.1f}"}),
         "<h3>Deformation zones</h3>",
         table(zon, {"level": "{:.3g}", "n_input": "{:.0f}", "area_km2": "{:.0f}", "n_events": "{:.0f}"}),
         "<h3>Offsets per section and segment</h3>",
         table(summ, {"n_bins_S0": "{:.0f}", "seis_minus_pb2002_km": "{:+.1f}", "ci95_lo": "{:+.1f}",
                      "ci95_hi": "{:+.1f}", "seis_minus_bathy_km": "{:+.2f}", "proposal_minus_pb2002_median_km": "{:+.1f}",
                      "proposal_minus_pb2002_p10_km": "{:+.1f}", "proposal_minus_pb2002_p90_km": "{:+.1f}"}),
         "<p class=mut>seis_minus_* only where >= 5 bins have tier A-B picks.</p>",
         "<h3>Along-strike view</h3>",
         figs("profiles.png", "Per section, in its frame: tier A-B epicentres (grey), bin picks with 68 % bootstrap "
              "(black), tier A-C picks (blue x), PB2002 (blue), MAPRIDGES (orange dashed), bathymetric picks (green "
              "squares), proposal with 68 % band (vermilion), bathymetric line where it disagrees (green dotted)."),
         "<h3>Section zooms</h3>", figs("legend_zoom.png", "Legend for the zooms"),
         figs("zoom_mendocino.png", "Mendocino transform and the triple junction"),
         figs("zoom_gorda_south.png", "Southern Gorda Ridge and the GR-MFZ corner"),
         figs("zoom_gorda_north.png", "Northern Gorda Ridge and the GR-Blanco corner"),
         figs("zoom_blanco.png", "Blanco transform: three segments, two step-overs (fit offsets with 68 % intervals)"),
         figs("zoom_jdf.png", "Juan de Fuca Ridge: Cleft-Vance (bathymetry), Axial node, Axial-Endeavour line (PI) "
              "against the CoAxial-Northern Symmetric bathymetric axis (green dotted, flagged)"),
         figs("zoom_explorer.png", "Explorer region: not resolved (PB2002 kept, low-confidence deformation zones)"),
         "<h2>5. Comparison with other boundary models</h2>",
         "<p>Median offset of each model from the proposal, per section (one value per 5 km bin, model point nearest "
         "the proposal, within 40 km). NNR-MORVEL56 (Argus et al. 2011) uses the Bird (2003) geometry here and matches "
         "PB2002 to 0.1 km; MORVEL (DeMets et al. 2010) itself has no boundary file. The GEM Global Active Faults "
         "database carries the same Bird (2003) segments offshore; USGS Qfaults and Watt et al. (2026) have no ridge or "
         "transform traces here (their \"Blanco transform fault zone\" lies east of the active trace). No digital "
         "high-resolution traces were found for the Blanco strands (Ren et al. 2023; Lange et al. 2026), Gorda, "
         "Endeavour, Sovanco or Nootka.</p>",
         table(mod[["section", "model", "n_bins", "median_offset_km", "median_abs_offset_km", "p90_abs_offset_km", "sign"]],
               {"n_bins": "{:.0f}", "median_offset_km": "{:+.1f}", "median_abs_offset_km": "{:.1f}",
                "p90_abs_offset_km": "{:.1f}"}),
         figs("models.png", "Proposal (vermilion) against PB2002, NNR-MORVEL56, Hasterok et al. (2022) and MAPRIDGES "
              "v2602 (blue dashed); grey: v6.1 tier A-B epicentres."),
         "<ul><li>Hasterok et al. (2022) and MAPRIDGES agree with the proposed Gorda Ridge to within 1 km (median); "
         "MAPRIDGES also with Blanco (1.8 km) and Cleft-Vance (0.8 km).</li>"
         "<li>All four models put the Mendocino transform 8-10 km south of the seismicity, on the ridge crest.</li>"
         "<li>Between Axial and Endeavour every model follows the CoAxial-Northern Symmetric ridge, 9-17 km east of "
         "the straight Axial-Endeavour line.</li></ul>",
         "<h2>6. What adopting v2 would change (PB_AXES=v2)</h2>",
         f"<p>Same segment rule; v6.1 tier A-C earthquakes. {nmov:,} of {ntot:,} events change segment against "
         f"axes v1 (D-B113). The triple junction moves {tj.dist_to_pb2002_tj_km:.1f} km, so the MTJ circle and the "
         "NAPA latitude limit move with it (the rule text is unchanged).</p>",
         "<ul><li><b>JR to JP</b>: the straight Axial-Endeavour line lies up to " + f"{jrn_dev:.0f}" + " km west of the "
         "Northern Symmetric ridge, so events there fall more than 15 km from the ridge trace. Drawing the bathymetric "
         "alternative instead would avoid this.</li>"
         "<li><b>GS and MTJ to NAPA</b>: the triple junction moves " + f"{tj.dist_to_pb2002_tj_km:.1f}" + " km, about "
         "4 km north (40.313 to 40.349 N), and NAPA is everything south of the triple-junction latitude. If the PI wants "
         "the new traces without this effect, delete the TJ row of plate_axes_v2.csv (pb_common then keeps the PB2002 "
         "triple junction).</li>"
         "<li><b>GR and GP to MFZ</b>: the Mendocino trace moves 8-9 km north, so more events near the GR-MFZ corner "
         "lie within 15 km of it (MFZ is matched before GR).</li></ul>",
         table(tot.reset_index().rename(columns={"index": "segment", "seg": "segment"}), {}),
         table(mv.head(15), {}),
         "<h2>7. Limits and unresolved sections</h2><ul>",
         f"<li><b>JdF Axial-Endeavour</b>: S0 has no picks between the two volcanic centres; the Endeavour anchor "
         f"rests on {en_n} tier A-C events with h_max <= 3 km (catalog region "
         "Endeavour). The PI's straight line disagrees with every bathymetric reference by up to " + f"{jrn_dev:.0f}" + " km (flagged, "
         "alternative in the GeoJSON). Adopting it would move JdF ridge events near the Northern Symmetric segment to "
         "the JdF plate group.</li>",
         "<li><b>Cleft-Vance (JRS)</b>: bathymetry only (two segments, 2.1 km step at 45.09 N).</li>",
         "<li><b>Gorda Ridge</b>: tier A-B picks cover 44 % of the ridge, below the 50 % rule; the segments come from "
         "the axial-valley picks, and the seismicity confirms them where present.</li>",
         f"<li><b>Mendocino</b>: both segments miss the agreement rule by < 0.1 km at one end (seismicity vs "
         f"base-of-slope pick, {mfz_seg.dev_mid.iloc[0]:.1f} / {mfz_seg.dev_mid.iloc[1]:.1f} km at mid-segment): drawn "
         "from the seismicity, bathymetric alternative kept.</li>",
         "<li><b>Triple junction</b>: the uncertainty includes only the Mendocino line; the deformation front is "
         "taken from PB2002 (the Watt et al. 2026 frontal thrust stops at 41.0 N).</li>",
         "<li><b>Explorer, Sovanco, Nootka</b>: 15 S0 events in the Explorer box; PB2002 kept north of 48.4 N; the "
         "deformation zones there use tier A-C and are low confidence.</li>",
         "<li>Uncertainties are statistical (day-block bootstrap) plus the median location error in the pick sigma; "
         "a common bias of all OBS-based catalogs (same stations) cannot be excluded by the paired tests, but the "
         "bathymetric axes are independent of every location and agree with the seismicity at Gorda and Mendocino.</li>",
         "<li>Post-hoc rule changes A1-A5 are listed in RULES.md; none changes a pre-registered threshold.</li></ul>",
         "<h2>8. Manuscript bullets</h2><ul>" + "".join(f"<li>{E(b[2:])}</li>" for b in bullets) + "</ul>",
         "<h2>9. Provenance</h2>", table(inp, {}),
         "<p>Downloads (public, md5 re-checked; rows in /wd1/mdenolle_data/external_geo/PROVENANCE.csv):</p>",
         table(prov, {}),
         "<p class=mut>Run: <code>workflow/11_paper_plan/run_pbv2.sh</code> (pbv2_fit, pbv2_bias, pbv2_figures, "
         "pbv2_report; pixi python with -I, nice 10, one process).</p>",
         "</body></html>"]
    out = C.BIG / "report_boundaries.html"
    doc = "\n".join(H).replace("<= ", "&le; ").replace(">= ", "&ge; ").replace(" < ", " &lt; ").replace(" > ", " &gt; ")
    out.write_text(doc)
    print(out)
    print("\n".join(bullets))


if __name__ == "__main__":
    main()
