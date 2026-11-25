#!/usr/bin/env python3
"""Catalog v5.1 re-interpretation (piece "interp"): self-contained HTML report.

Reads the tables and figures written by v51_interp_tables.py and v51_interp_figs.py, the products of the
pieces mag, faults and clusters, and the verifiers' tables. Writes
data/catalog_v5/v5.1_analysis/interp/catalog_v51_interpretation.html.

    python workflow/06_analysis/v51_interp_report.py
"""
from __future__ import annotations

import base64
import html
import json
import subprocess
from datetime import date

import numpy as np
import pandas as pd

from v51_interp_common import BIG, CLU, FLT, MAG, OUT, REGIONS, SLUG, VER_F, hav

FIG = OUT / "figs"


def git_head(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short=8", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def img(name, caption, width="100%"):
    p = FIG / name
    if not p.exists():
        return f"<p><em>missing figure {html.escape(name)}</em></p>"
    b = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" style="width:{width}" alt="{html.escape(caption[:80])}">'
            f"<figcaption>{caption}</figcaption></figure>")


def tab(df, fmt=None, index=False):
    d = df.copy()
    fmt = fmt or {}
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: "" if pd.isna(v) else f.format(v))
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    return d.to_html(index=index, escape=True, border=0, classes="t")


def f1(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pct(x):
    return f"{x:.0f}%"


# ----------------------------------------------------------------------------------------------- data
t1 = pd.read_csv(OUT / "t1_locations.csv").set_index("region")
t2 = pd.read_csv(OUT / "t2_sharpness.csv").set_index("region")
t3 = pd.read_csv(OUT / "t3_clusters.csv").set_index("region")
t3b = pd.read_csv(OUT / "t3b_boxes.csv").set_index("region")
t4 = pd.read_csv(OUT / "t4_faults.csv").set_index("region")
t6 = pd.read_csv(OUT / "t6_anss_depth.csv").set_index("region")
t7 = pd.read_csv(OUT / "t7_benchmarks.csv")
t8 = pd.read_csv(OUT / "t8_node_pileups.csv")
S = pd.read_csv(OUT / "surfaces_v1_interp.csv")
FA = pd.read_csv(OUT / "families_v51_setting.csv")
EX = json.loads((OUT / "region_extras.json").read_text())
MW = json.loads((OUT / "msh_weekly.json").read_text())
BS = pd.read_csv(OUT / "blanco_segments.csv")
MAGR = pd.read_csv(MAG / "v51_vs_v4_by_region.csv").set_index("region")
c = pd.read_pickle(BIG / "events_joined.pkl")
E = c[c.is_eq]
A = c[c.in_analysis.eq(True)]
hmaxD = A[A.tier.eq("D")].groupby("box51").h_max.median()
hmaxABC = A[A.tier.isin(list("ABC"))].groupby("box51").h_max.median()
casc_deep = E[E.region.eq("Cascades") & (E.depth > 30)]
casc_deep_msh = int((hav(casc_deep.lat.values, casc_deep.lon.values, 46.20, -122.19) < 15).sum())
deep_pass_onshore = {r: int((E.region.eq(r) & E.deep_pass).sum()) for r in ["Cascades", "Other onshore", "Puget/Georgia"]}
dp_onshore_med = float(E[E.region.isin(["Cascades", "Other onshore", "Puget/Georgia"]) & E.deep_pass].depth.median())
s_endv = E[E.region.isin(["Endeavour", "JdF ridge/plate"]) & E.lat.between(47.5, 47.8) & E.lon.between(-129.35, -129.1)]
H = {k: git_head(p) for k, p in [("mag", MAG), ("faults", FLT), ("clusters", CLU), ("interp", OUT)]}
NODE = {(r.region, r.node_km): r for r in t8.itertuples()}


def bm(name_start):
    r = t7[t7.benchmark.str.startswith(name_start)].iloc[0]
    return r


def surf_table(region, n=14):
    g = S[S.cat_region.eq(region)].sort_values("n_events", ascending=False).head(n)
    if not len(g):
        return "<p>No v1 surface in this region.</p>"
    d = pd.DataFrame({
        "surface": g.surface_id, "events": g.n_events, "strike": g.strike.round(0), "dip": g.dip.round(0),
        "depth km": g.depth_km.round(1), "depth ok": (100 * g.frac_depth_ok).round(0),
        "stable (reruns)": g.stable.astype(str), "J 80% subsample": g.J_sub80.round(2),
        "% bg": g.pct_background.round(0), "% swarm": g.pct_swarm.round(0),
        "v0 link": g.change_from_v0, "members dz v4->v5.1": g.members_dz_ok_med.round(1),
        "literature (carried from v0)": g.lit_match.fillna("").str.slice(0, 90),
        "flags added here": g.interp_flags.fillna("")})
    return tab(d, fmt={"J 80% subsample": "{:.2f}", "members dz v4->v5.1": "{:+.1f}"})


def swarm_rows(region, k=6):
    g = FA[FA.region_v51.eq(region) & FA.type.isin(["swarm", "mainshock-aftershock"])].sort_values(
        "n_all", ascending=False).head(k)
    if not len(g):
        return ""
    d = pd.DataFrame({
        "family": g.family_id, "type": g.type, "start (UTC)": g.start_all.astype(str).str.slice(0, 16),
        "events (>=Mc)": g.n_all.astype(int).astype(str) + " (" + g.n.astype(int).astype(str) + ")",
        "Mmax": g.Mmax.round(1), "t90 days": g.duration90_days.round(2), "extent km": g.extent_km.round(1),
        "good-depth median km": g.depth_good_med.round(1), "n good": g.n_depth_good,
        "resolvable spread": g.mig_resolvable.astype(str), "migrating": g.mig_migrating.astype(str),
        "on surfaces": (100 * g.frac_on_faults).round(0), "main surface": g.main_fault.fillna(""),
        "nearest boundary": g.pb_class + " " + g.pb_km.round(0).astype(int).astype(str) + " km",
        "v4 status": g.v4_status})
    return tab(d)


def loc_line(r):
    x = t1.loc[r]
    return (f"{int(x.rows):,} rows, {int(x.located):,} located ({int(x.no_hypocentre)} without a hypocentre, left out), "
            f"{int(x.non_earthquake):,} flagged non-earthquakes. Tiers A/B/C/D: {x.pctA:.0f}/{x.pctB:.0f}/{x.pctC:.0f}/{x.pctD:.0f}%. "
            f"floor_limited {int(x.floor_limited):,}, far_only {int(x.far_only)}, deep_pass {int(x.deep_pass)}; "
            f"depth_good (tier A-C, depth_quality ok, no flag) {int(x.depth_good):,}. "
            f"Median depth of located earthquakes: v4 {x.v4_depth_med:.1f} km, v5.1 {x.depth_med:.1f} km bsl "
            f"({x.bsf_med:.1f} km below the surface); median shift {x.shift_h_med:.1f} km horizontally, "
            f"{x.shift_z_med:+.1f} km in depth. Median sig_z {x.sig_z_med:.1f} km, nearest station {x.dist_min_med:.0f} km. "
            f"In v4, {x.v4_above_surface_pct:.0f}% of these events plot above the ground or seafloor if v4 depth is read as km bsl.")


def clu_line(r):
    x = t3.loc[r]
    return (f"{int(x.n_above_Mc):,} earthquakes above Mc: {x.pct_background:.0f}% background, {x.pct_MA:.0f}% in "
            f"mainshock-aftershock families, {x.pct_swarm:.0f}% in swarms, {x.pct_multiplet:.0f}% in multiplets "
            f"(v4: {x.pct_background_v4:.0f}% / {x.pct_MA_v4:.0f}% / {x.pct_swarm_v4:.0f}% for background / MA / swarm). "
            f"Families of 5 or more above Mc led by an event here: mainshock-aftershock {int(x['fam5_mainshock-aftershock'])}, "
            f"swarms {int(x.fam5_swarm)}. Tier D events are {x.tierD_share_above_Mc:.0f}% of the analysis set here, and "
            f"{x.bg_share_tierD:.0f}% of them are background (tiers A-C: {x.bg_share_tierABC:.0f}%).")


def flt_line(r):
    x = t4.loc[r]
    if x.surfaces == 0:
        return "No v1 surface (region not fitted or no events in the fit set)."
    v0s = "n/a" if pd.isna(x.surfaces_v0) else f"{int(x.surfaces_v0)}"
    v0m = "n/a" if pd.isna(x.members_v0) else f"{int(x.members_v0):,}"
    return (f"v1 surfaces with most of their members here: {int(x.surfaces)} ({int(x.steep)} steep, {int(x.members):,} fit members; "
            f"v0: {v0s} surfaces, {v0m} members). Stable under the builder's detection reruns: {int(x.stable)}; "
            f"recovered on the verifier's 80% subsample: {int(x.recovered_sub80)}; on tiers A-B only: {int(x.recovered_tierAB)}. "
            f"Of the {int(x.earthquakes):,} located earthquakes, {x.pct_on_fault:.0f}% are fit members or uniquely assigned, "
            f"{int(x.ambiguous):,} are ambiguous and {int(x.none):,} are on no surface. Median RMS scatter of members about their surface: "
            f"v0 {f1(x.rms_normal_v0, 2)} km, v1 {f1(x.rms_normal_v1, 2)} km.")


def anss_line(r):
    x = t6.loc[r]
    if x.n_paired < 10:
        return ""
    return (f"Against ANSS/ComCat ({x.nets}; fixed depths removed), on {int(x.n_paired):,} events matched in both versions, "
            f"our depth minus ANSS is {x.dz_paired_v4:+.1f} km in v4 and {x.dz_paired_v51:+.1f} km in v5.1 (median |difference| "
            f"{x.absdz_paired_v4:.1f} and {x.absdz_paired_v51:.1f} km).")


# ----------------------------------------------------------------------------------------------- regions
R = {}

ax = t1.loc["Axial"]
b_ax = bm("Axial caldera, tier A-B: share")
R["Axial"] = dict(
    loc=f"""<p>{loc_line('Axial')}</p>
<p>v5.1 places the caldera seismicity in a band 1-3 km below the seafloor: tier A-B median {bm('Axial caldera, tier A-B, depth').v51_med:.2f} km
below the seafloor, p10-p90 {bm('Axial caldera, tier A-B, depth').v51_p10:.2f}-{bm('Axial caldera, tier A-B, depth').v51_p90:.2f} km, and
{b_ax.v51_med:.0f}% of the {int(b_ax.n):,} tier A-B events within 2.0 km of the seafloor (Waldhauser et al. 2020: 90%; acceptance target 85%;
v4-NLL3D 79%, B4 at model v0.3.0 12%). The v4 depths cannot be compared directly: read as km bsl, {ax.v4_above_surface_pct:.0f}% of them are above the seafloor.
The v4 picture of "two lobes between -2 and 2 km with a second cluster at 4-5 km" does not survive: the 2,029 tier A-B events that v4 put at
3.5-5.5 km are at a median {1.79:.1f} km below the seafloor in v5.1, the same as the rest. The v4 western lobe (1,843 tier A-B events at a median
130.059°W) moved 3.3 km east to 130.017°W, onto the western caldera wall. The east-to-west event ratio across 130.00°W is
{EX['axial_east_west_ratio_v5.1']:.1f} (v4 {EX['axial_east_west_ratio_v4']:.1f}; about 5 in Wilcock et al. 2016).</p>
<p>Two location artefacts limit what the v5.1 Axial geometry can show. {NODE[('Axial', 3)].excess:,} events ({NODE[('Axial', 3)].excess_pct_of_region:.0f}% of the region)
are in excess within 0.25 km of 3 km bsl, the 3 km depth node of model v0.3.1 (about 1.5 km below the caldera floor), and 86% of located events share their
exact epicentre with at least one other event (96% of tiers A-B; stacks of up to 38 events on a lattice of about 60 m, the final oct-tree cell). Nearest-neighbour
distances and ring-fault widths below about 100 m are therefore not measured by v5.1. {EX['axial_floor_limited']:,} events are floor_limited.</p>""",
    flt="""<p>Axial was left out of the fault fit (as in v0); its 26,874 located events are labelled "region not fitted (Axial)" here, not "on no surface".
The published ring-fault geometry (outward-dipping faults from the surface to the reservoir roof at about 1.1-2.3 km, Wilcock et al. 2016; less than 200 m wide,
Waldhauser et al. 2020) is finer than both the v5.1 cell lattice and the node pile-up, so a v5.1 surface fit would not add to it.</p>""",
    clu=f"""<p>{clu_line('Axial')}</p>
<p>The rate history is the same as in v4 because the events are the same: {EX['axial_monthly']['2014-11']:,} located events in November 2014,
{EX['axial_monthly']['2015-03']:,} in March 2015, {EX['axial_monthly']['2015-04']:,} in April 2015, then {EX['axial_monthly']['2015-05']} in May
and about 50-140 a month to the end of 2016. The eruption onset (family 1872, from 2015-04-24 05:13 UTC) is one 513-event swarm with 495 events above Mc
(v4: 805 and 632) that lasted 0.6 day at the 90% level.</p>
<p>The Axial background/clustered split is not a robust result. The background share is 61.8% in the main run, 42.9% without tier D, 81.8% with the
conservative threshold and 74.7% at the lower Mc of the magnitude piece. Time-shuffled catalogs reproduce {t3b.loc['AX','null_over_real_links']:.0%} of the
real links (verifier), because the 4,410 events above Mc sit in a 5 km caldera and most occur in the months of rising rate before the eruption, which
shuffling keeps. The v4 statement that Axial has the lowest background share and the most swarms per event therefore rests mostly on links that a
shuffled catalog also makes. The tier D events here are a special case: median h_max {hmaxD['AX']:.1f} km (tiers A-C {hmaxABC['AX']:.1f} km), seen from
outside the caldera network (median azimuthal gap 327°), and 80% of them are background.</p>""",
    sw="""<p>All 40 Axial swarms of 5 or more events above Mc are on the caldera (all within 25 km of the summit by construction; nearest PB2002 step is the
JdF-PA transform offset). Their median duration at the 90% level is 17 minutes and median extent 1.6 km, below the 2 km scale at which v5.1
can resolve spreading (none is resolvable). In the eruption swarm (figure below) the first 34 events (first 1.2 hours) are centred 2.3 km north
and 0.6 km east of the caldera centre (45.95°N, 130.00°W) at a median 2.9 km below the seafloor; from 0.05 day on, the activity is centred 0.8 km east
of the centre, spans about 3 by 3 km (p10-p90) and stays at a median 1.8-2.0 km below the seafloor, about the depth of the reservoir roof. The rift-zone
seismicity of the eruption lies largely outside the Axial region box and is not analysed here.</p>""",
    lit=f"""<p>Agrees: rate history and timing of the pre-eruption acceleration and the drop after 24 April 2015 (Wilcock et al. 2016); the east-wall
dominance (ratio {EX['axial_east_west_ratio_v5.1']:.1f} against about 5). Partly agrees: depth, {b_ax.v51_med:.0f}% within 2.0 km against 90%
(Waldhauser et al. 2020); p90 of tiers A-B is {EX['axial_bsf_AB_p90']:.2f} km below the seafloor, so the deep tail is short, but the 3 km node
pile-up means part of the agreement is set by the model grid. Not testable: ring-fault width and dip.</p>""",
    v4=["v4 'second cluster at 4-5 km' (science report): not in v5.1; those events are at the same depth as the rest.",
        "v4 'lowest background share and most swarms per event' (v4 clusters): the split moves from 43% to 82% background with the variant, and shuffled catalogs reproduce 75% of the links; it is not a setting result.",
        "v4 western lobe at 130.06°W: moved 3 km east onto the caldera wall."],
    unc="Depths are good to the 3 km node and the ~60 m lattice; tier D events (32% of the analysis set) are poorly located from outside the caldera network.",
    figs=[("fig_region_axial.png", "Axial: v4 (left) and v5.1 (right) maps coloured by cluster class, and sections of all events along 90°."),
          ("fig_axial_rate_depth.png", "Axial: weekly event count and depth below seafloor of tier A-B events."),
          ("fig_swarm_1872.png", "Axial eruption-onset swarm (family 1872): map coloured by time, along-axis distance, depth and magnitude against time.")])

en = EX["ridge_Endeavour"]
b_en = bm("Endeavour axial box")
R["Endeavour"] = dict(
    loc=f"""<p>{loc_line('Endeavour')}</p>
<p>Endeavour depths are not constrained in v5.1: depth_quality is "unconstrained" for every event, the median sig_z is {en['sig_z_med']:.0f} km
and the nearest station is a median {en['dist_min_med']:.0f} km away (only KEMF and NCHR are on the segment). Half of the events ({en['deeper_8bsf']} of
{en['n']}) are more than 8 km below the seafloor, under a crust 6-8 km thick, and {en['within_0p25_of_floor']} are within 0.25 km of the seafloor
({en['floor_limited']} floor_limited). The 36 events in the axial box (47.85-48.10°N) have a median of {b_en.v51_med:.1f} km below the seafloor, which looks like the
published 1.5-3.5 km band, but their p90 is {b_en.v51_p90:.0f} km. This is the two-boundary pattern described in the depth-literature report
(pinned at the seafloor or 15-55 km deep), not a measurement.</p>""",
    flt="<p>No v1 surface has most of its members here; v0 EN001 (121 events) is lost because only 10% of its events pass the fit selection.</p>",
    clu=f"""<p>{clu_line('Endeavour')} With 35 events above Mc 2.35 the region cannot support a statement on its own; the Endeavour-Explorer box (EN, 554 events)
is dominated by Explorer and the 2014 Vancouver Island aftershocks.</p>""",
    sw=f"""<p>The southern Endeavour / northern Juan de Fuca swarms (47.5-47.8°N, 129.1-129.35°W; {len(s_endv)} located events in the Endeavour and JdF
regions, 2011-2014, most in 2014) include family 1119 (2014-02-09, 35 events) in this region and families 1066 and 1003 just south of it (JdF section below).
They sit 14-23 km from the PB2002 ridge and transform steps and are pinned at the seafloor ({100 * s_endv.floor_limited.mean():.0f}% floor_limited,
median {s_endv.bsf.median():.2f} km below the seafloor), so only their epicentres and timing can be used.</p>""",
    lit="""<p>Not testable for depth (Weekly et al. 2013; Krauss et al. 2023: on-axis events 1.5-3.5 km below the seafloor). The segment remains quiet in
our catalog except for the southern swarms, consistent with the post-2005 extensional lull of Weekly et al. (2013) and with Krauss et al. (2025) only
as an upper bound, because small on-axis events are not detected without the NV stations close by.</p>""",
    v4=["v4 'median QC depth 12.4 km, deep for a ridge' (science report): v5.1 does not resolve it either; half the events are deeper than 8 km below the seafloor and a quarter are pinned at it.",
        "v4 Endeavour surfaces (EN001-EN004) as fault planes: their depths and dips are not usable (0% depth_quality ok)."],
    unc="Report epicentres only.",
    figs=[("fig_region_endeavour.png", "Endeavour: v4 and v5.1 maps and sections.")])

ex = EX["ridge_Explorer"]
b_ex = bm("Explorer seaward of the front (")
b_ex30 = bm("Explorer seaward, nearest")
R["Explorer"] = dict(
    loc=f"""<p>{loc_line('Explorer')}</p>
<p>The Explorer depths moved the wrong way. Seaward of the deformation front ({int(b_ex.n):,} events with the seafloor deeper than 1.5 km), the
v5.1 median is {b_ex.v51_med:.0f} km bsl (p10-p90 {b_ex.v51_p10:.0f}-{b_ex.v51_p90:.0f} km) against {b_ex.v4_med:.0f} km in v4 and 5-22 km bsl
in the OBS studies (Hutchinson et al. 2019, 2023), with the Moho at 12-15 km bsl. The {int(b_ex30.n)} seaward events with a station within 30 km are at a
median {b_ex30.v51_med:.0f} km. Explorer has the largest median depth shift of any region ({t1.loc['Explorer','shift_z_med']:+.0f} km), no depth_good event,
median nearest station {ex['dist_min_med']:.0f} km, 24 far_only and 27 deep_pass events. The magnitude piece found the same bias on the ComCat
moment-tensor events: v5.1 is a median 24.9 km deeper than ComCat in Explorer. These depths are location errors (depth traded against origin time
with every station on one side), not mantle seismicity.</p>""",
    flt=f"""<p>{flt_line('Explorer')} All five Explorer surfaces sit at 30-60 km on events whose depth_quality is "unconstrained"; their map traces can be
used, their depths and dips cannot. v1_EN002 (87 events, strike 332°) keeps the v0 match to the fault of the 2014 Mw 6.4 Nootka sequence inside the
subducted Explorer plate (Hutchinson et al. 2020); v1_EN001 keeps the eastern Sovanco fracture zone match.</p>""",
    clu=f"""<p>{clu_line('Explorer')}</p>
<p>The 2014-04-24 Mw 6.5 Vancouver Island mainshock (event 89617) is in this region. In v5.1 its family has 300 events and a t90 of 280 days
because a 2013-08-04 M 5.5 event (64836) is now linked to it as a foreshock. That link comes from a tier D epicentre: event 64836 has h_max 24 km and was
33 km from the mainshock in v4; it is 1.2 km from it in v5.1 only because both moved (the mainshock by 23 km). Without tier D (variant tierABC)
the family is a normal sequence of 73 events with a t90 of 16 days. The clustering piece's report calls the two events "1 km apart" without this caveat.</p>""",
    sw="""<p>The 2012-02-04 M 5.6 sequence (family 296) is the family with the strongest migration statistic in the catalog (radial speed 0.8 km/day toward
the north-northeast over four months, Spearman rho 0.71; the verifier's permutation test gives p < 0.001). The space-time plot shows that this is a step,
not a front: the 60 events of the first day are centred 6 km SSW of the family centroid, and the 47 events of 13-19 February (with an M 4.7 on
16 February) form a second cluster about 15 km to the NNE, at +9 km along 17°. A second patch activated 9-12 days after the mainshock; the envelope exponent
of 0.15 says the same. Its v5.1 depths (median 55 km, v4 1.8 km) are not physical, so this is a map-view result only. Explorer has only two swarms of 5 or
more events above Mc.</p>""",
    lit=f"""<p>Disagrees on depth with every OBS study (5-22 km bsl seaward, Hutchinson et al. 2019, 2023; centroids 4-15 km below the seafloor, Braunmiller &
Nábělek 2002). Map patterns (Sovanco, Nootka) agree with the v0 literature matches.</p>""",
    v4=["v4 Explorer depths (median 9 km) were closer to the literature than v5.1 (40 km); neither is a constraint.",
        "v4/v0 Explorer surfaces as dipping faults at depth: not usable; only traces.",
        "v4 2012-02-04 and 2012-11 Explorer sequences 'expand at 0.2-1.2 km/day': the 2012-11 sequence no longer passes the test; the 2012-02-04 one passes because a second cluster 15 km NNE started 9-12 days later (a step, not a front)."],
    unc="Report epicentres only; flag the 2014 Vancouver Island family duration as an artefact of one tier D link.",
    figs=[("fig_region_explorer.png", "Explorer: v4 and v5.1 maps and sections."),
          ("fig_swarm_296.png", "Explorer 2012-02-04 M 5.6 sequence (family 296).")])

jd = EX["ridge_JdF ridge/plate"]
R["JdF ridge/plate"] = dict(
    loc=f"""<p>{loc_line('JdF ridge/plate')}</p>
<p>Depth is not constrained on the Juan de Fuca ridge and plate. {jd['floor_limited']:,} of {jd['n']:,} events are floor_limited and {jd['within_0p25_of_floor']:,}
are within 0.25 km of the seafloor, while {jd['deeper_8bsf']} are more than 8 km below it (crust 6.7-8 km thick). The median ({jd['bsf_med']:.1f} km below the
seafloor) is the mean of two boundaries, not a depth. Tier A and B are almost absent ({t1.loc['JdF ridge/plate','pctA']:.0f}% and {t1.loc['JdF ridge/plate','pctB']:.1f}%).</p>""",
    flt=f"""<p>{flt_line('JdF ridge/plate')} v1_BL018 keeps the v0 match to the Cleft segment of the southern Juan de Fuca Ridge (ridge-parallel strike 199°);
v1_ON026 (49 events, new) lies on the ridge near 45.4°N, 130.0°W.</p>""",
    clu=f"""<p>{clu_line('JdF ridge/plate')} 60% of the events above Mc here are tier D, so the class split is as uncertain as at Axial.</p>""",
    sw="""<p>Eighteen swarms of 5 or more events above Mc; for 17 the nearest PB2002 step is a ridge step (median 14 km away). The largest are on the southern Endeavour /
northern Juan de Fuca section at 47.57-47.64°N: family 1066 (2014-01-09, 146 events, 7.9 km extent, 98 days in all, axis 25°) and family 1003
(2013-11-24, 42 events, new in v5.1). Their epicentres line up with the ridge trend; their depths are pinned at the seafloor (v4 put them at 12-16 km).
None is resolvable for migration.</p>""",
    lit="""<p>Indirect constraints only (magma-chamber reflectors at 2.0-2.7 km below the seafloor; crust 6.7-8 km; Carbotte et al. 2008; Canales et al. 2005).
Neither v4 nor v5.1 depths can be compared.</p>""",
    v4=["v4 depths of the 2013-2014 ridge swarms (12-16 km): not supported; v5.1 pins them at the seafloor, which is not a measurement either."],
    unc="Report epicentres only.",
    figs=[("fig_region_jdf_ridge_plate.png", "Juan de Fuca ridge and plate: v4 and v5.1 maps and sections (along 20°)."),
          ("fig_swarm_1066.png", "Juan de Fuca ridge 2014-01-09 swarm (family 1066).")])

bl = t1.loc["Blanco"]
seg = BS.set_index("segment")
b_blx9 = bm("Blanco, tier A-B, X9")
R["Blanco"] = dict(
    loc=f"""<p>{loc_line('Blanco')}</p>
<p>The regional median hardly moved ({bl.v4_depth_med:.1f} to {bl.depth_med:.1f} km bsl), but the segments moved differently (depth_good events, km
below the seafloor, v4 read as bsl / v5.1): West {seg.loc['West (WBD/EBD)','good_v4_med']:.1f} / {seg.loc['West (WBD/EBD)','good_bsf_med']:.1f};
Surveyor {seg.loc['Surveyor','good_v4_med']:.1f} / {seg.loc['Surveyor','good_bsf_med']:.1f}; Cascadia Depression {seg.loc['Cascadia Depression','good_v4_med']:.1f} /
{seg.loc['Cascadia Depression','good_bsf_med']:.1f}; Blanco Ridge {seg.loc['Blanco Ridge','good_v4_med']:.1f} / {seg.loc['Blanco Ridge','good_bsf_med']:.1f}
(p90 {seg.loc['Blanco Ridge','good_bsf_p90']:.0f}); Gorda Depression {seg.loc['Gorda Depression','good_v4_med']:.1f} / {seg.loc['Gorda Depression','good_bsf_med']:.1f}.
The eastern segments went 6-7 km deeper. Against the Lange et al. (2026) OBS catalog, event by event (tiers A-C, Lange read as bsl): +1.4 km overall,
+2.5 km West, +0.7 km Surveyor, -1.0 km Cascadia Depression, +2.9 km Blanco Ridge (p90 +13 km) and +5.6 km Gorda Depression (paired_v5.1.md).
The X9-year tier A-B events are at a median {b_blx9.v51_med:.1f} km below the seafloor (p10-p90 {b_blx9.v51_p10:.1f}-{b_blx9.v51_p90:.1f}).</p>
<p>The sections show columns down to 30-40 km on the Blanco Ridge and horizontal bands. {NODE[('Blanco', 12)].excess:,} events are in excess within 0.25 km
of 12 km and {NODE[('Blanco', 15)].excess:,} within 0.25 km of 15 km, two depth nodes of model v0.3.1. Depth_quality is "unconstrained" outside the
2012-13 X9 deployment ({int(bl.dq_unconstrained):,} events), so 2014-2015 Blanco depths are epicentre-only. 19% of Blanco events share their exact epicentre with another event.</p>""",
    flt=f"""<p>{flt_line('Blanco')}</p>
<p>The transform comes out as 12 near-vertical strands (strike 113-116°) and six other surfaces. The strands that keep a v0 match are the Surveyor
Depression S-wall strand (v1_BL003, matched to v0 BL001a, normals 2° apart), the intra-basin Surveyor strand (v1_BL007) and the Blanco Ridge
principal strand (v1_BL005, split from v0 BL003a). The West Blanco Depression pair of v0 is now two long strands, v1_BL001 (5,180 events, 52% of its
events above Mc in swarms) and v1_BL002 (4,369, 38%). v1_BL009 and v1_BL015 (centroids at 23 km, 25-32% depth-ok events) sit where v0's merged
"deeper band" was: they come from the deep tail of the Blanco Ridge, not from a separate fault. The Cascadia Depression N-S normal faults of v0
(BL005, BL008, BL012; high-confidence literature match to Lange et al. 2026 and Ren et al. 2023) are not recovered as N-S planes: v1 first fits
a 64 km plane striking 121° through the dense cluster (v1_BL004, not recovered on the 80% subsample, J = 0.04), and the events of the v0 N-S faults keep
their N-S map elongation (10°, 177°, 4°). This is a detection-order effect, not a relocation result, so the v0 interpretation stands until a fit restricted
to the Cascadia Depression is run. None of the 12 strands is recovered by the verifier's 80% subsample rerun (catalog-wide, the strike range across the builder's
reruns has a median of 103° for unstable surfaces): the Blanco strand geometry rests on the strand rule, not on the detection alone.</p>""",
    clu=f"""<p>{clu_line('Blanco')}</p>
<p>By segment (events above Mc): West {seg.loc['West (WBD/EBD)','pct_swarm']:.0f}% in swarms and {seg.loc['West (WBD/EBD)','pct_bg']:.0f}% background;
Surveyor {seg.loc['Surveyor','pct_swarm']:.0f}% / {seg.loc['Surveyor','pct_bg']:.0f}%; Cascadia Depression {seg.loc['Cascadia Depression','pct_swarm']:.0f}% /
{seg.loc['Cascadia Depression','pct_bg']:.0f}%; Blanco Ridge {seg.loc['Blanco Ridge','pct_swarm']:.0f}% / {seg.loc['Blanco Ridge','pct_bg']:.0f}%;
Gorda Depression {seg.loc['Gorda Depression','pct_swarm']:.0f}% / {seg.loc['Gorda Depression','pct_bg']:.0f}%. The swarm-rich part of the transform is the
West Blanco Depression; the Blanco Ridge and the Gorda Depression are mostly background with a few mainshock-aftershock families. The Blanco is the
most robust of the swarm-rich settings: shuffled catalogs reproduce {t3b.loc['BL','null_over_real_links']:.0%} of its links (Axial 75%), and removing tier D
changes its background share by 1.7 points. The two large Blanco multiplets are still swarms with M1-M2 near 0.1 (2012-02/04 Mw 5.8 and 6.0;
2015-06-01 Mw 5.8, 5.4 and 5.9), as at other oceanic transforms (McGuire et al. 2005; Boettcher & McGuire 2009). Offshore ML is 0.3 above ComCat ML at
Blanco (magnitude piece), which shifts every event by the same amount and does not change M1-M2.</p>""",
    sw=f"""<p>69 swarms of 5 or more events above Mc; for 67 the nearest PB2002 step is a transform step (median 30 km away); median t90 4.2 hours, median extent 4.3 km, median
good-depth 8.3 km bsl. Nine have a resolvable spread and none migrates by the test. The 2012-10-02 swarm on the West Blanco Depression (family 427,
1,447 events, 683 above Mc, Mmax 4.9) is the largest family of the catalog. It occupies a 16 km patch of v1_BL001 (78% of its events on surfaces)
within 3.4 days at the 90% level, with no ordered spreading (rho 0.03). Its good depths span 5-21 km (p10-p90), wider than the 0-8 km published for the
western segments (Ren et al. 2023; Adimah et al. 2024), with bands at the 12 and 15 km nodes; the depth range is location scatter, not a fault 16 km tall.
The 2015-06-01 triplet (family 1932, 744 events) covers about 21 km of the West Blanco Depression within 13 hours: the Mw 5.8 at 06:52 UTC,
the Mw 5.4 3.9 hours later 7 km ESE of it, and the Mw 5.9 13 hours later 14 km WNW of it, i.e. three adjacent patches, as at other oceanic transforms.
During days 1-10 the activity reaches 30 km further WNW; after day 10 it returns to a 15 km patch around the first event. Its depths are not
constrained (2015, outside the X9 year).
The 2014-03-11/12 sequence (family 1170, 480 events) was a mainshock-aftershock family in v4 (1,274 events) and is now a swarm of 480 events, because the
tighter epicentres of v5.1 cut its links to neighbouring bursts; the DBSCAN scan shows the reverse effect, chaining four 2013 bursts into one cluster
longer than 30 days.</p>""",
    lit=f"""<p>Agrees: segment layout and the swarm-rich western transform with burst-like seismicity (Lange et al. 2026, bursts shorter than 5 minutes;
Journeau et al. 2026, more events at step-overs). Agrees within about 1 km at Surveyor and in the Cascadia Depression. Disagrees on the Blanco Ridge and
in the Gorda Depression: v5.1 good depths have medians of {min(seg.loc['Blanco Ridge','good_bsf_med'], seg.loc['Gorda Depression','good_bsf_med']):.0f}-{max(seg.loc['Blanco Ridge','good_bsf_med'], seg.loc['Gorda Depression','good_bsf_med']):.0f} km below the
seafloor against crustal events at 3-7 km and mantle events at 8-13 km (Kuna et al. 2019) and a maximum of about 10-14 km (Liu & Tan 2025; Lange et al. 2026).
The question of mantle swarms on the Blanco Ridge (Kuna et al. 2019) cannot be answered with v5.1 depths, because the events that would carry it are the ones
with the +3 to +13 km offset.</p>""",
    v4=["v0 Blanco 'deeper band' (merged strands): reappears as v1_BL009/BL015 on events with unconstrained depth; still not a fault.",
        "v0 Cascadia Depression N-S faults: not recovered in v1 because of detection order; the v0 interpretation is neither confirmed nor refuted by v1.",
        "v4 2014-03-11 mainshock-aftershock family (1,274 events): now a 480-event swarm; family membership at Blanco depends on epicentre precision.",
        "v4 'vertical columns where depth is poorly constrained': still present, now deeper on the Blanco Ridge, plus pile-ups at the 12 and 15 km nodes."],
    unc="Depths outside the 2012-13 X9 year are unconstrained; within it, Blanco Ridge and Gorda Depression depths are 3-6 km too deep at the median.",
    figs=[("fig_region_blanco.png", "Blanco: v4 and v5.1 maps and sections along the transform (116°)."),
          ("fig_swarm_427.png", "Blanco 2012-10-02 swarm (family 427)."),
          ("fig_swarm_1170.png", "Blanco 2014-03-11/12 sequence (family 1170)."),
          ("fig_swarm_1932.png", "Blanco 2015-06-01 Mw 5.8/5.4/5.9 triplet (family 1932).")])

go = t1.loc["Gorda"]
gi = bm("Gorda interior (")
gia = bm("Gorda interior, all")
gr = bm("Gorda Ridge axis (within")
grn = bm("Gorda Ridge axis, tier A-B, nearest")
R["Gorda"] = dict(
    loc=f"""<p>{loc_line('Gorda')}</p>
<p>The Gorda depths split by station distance. All interior events (127-125°W, 40.5-42.5°N; {int(gia.n):,}) are at a median {gia.v51_med:.0f} km below the
seafloor, 7-13 km below the thermal limit for faulting in 4 Myr lithosphere (9.5-15 km below the seafloor; VanderBeek & Toomey 2019). The {int(gi.n)}
interior events with a station within 10 km are at a median {gi.v51_med:.1f} km below the seafloor, inside the limit. The catalog flags the rest as
biased_deep ({int(go.dq_biased_deep):,} of {int(go.located):,}), and the near-station subset shows why. Near the ridge axis the same pattern holds
(tier A-B within 10 km of the PB2002 ridge: {gr.v51_med:.0f} km below the seafloor on {int(gr.n)} events, {grn.v51_med:.0f} km on the {int(grn.n)} with a station within 15 km,
against 3-7 km in Jones & Johnson 1978). Against ANSS (mostly NCSN and USGS), v5.1 is {t6.loc['Gorda','dz_paired_v51']:+.1f} km deeper on
{int(t6.loc['Gorda','n_paired'])} matched events (v4 {t6.loc['Gorda','dz_paired_v4']:+.1f} km). The depth_good events moved from a v4 median of
{go.good_v4_depth_med:.1f} km to {go.good_depth_med:.1f} km bsl; read as km bsl, the v4 value would be above the seafloor (about 3 km deep), another sign of the v4 datum problem.
{NODE[('Gorda', 29)].excess} events are in excess at the 29 km node.</p>""",
    flt=f"""<p>{flt_line('Gorda')}</p>
<p>The ridge-axis surfaces keep their v0 literature matches but are mostly unstable: v1_GO002 (Jackson segment of the Gorda Ridge, strike 22°, dip 43°,
562 events, 33% swarm events) and v1_GO004 (northern Escanaba Trough axis, 363 events, 34% swarm events; members moved +14.8 km in depth). The 2014-03-10 Mw 6.8
rupture zone in the plate interior is imaged by v1_MT009, v1_MT013 and v1_MT017 (NE strike, near-vertical, 0-2% depth-ok members): their traces agree with
the NE-striking left-lateral faults of the Gorda deformation zone (Chaytor et al. 2004), but their depths (26-27 km) are in the biased_deep population. Nine Gorda
surfaces are flagged here for members that moved more than 10 km in depth (e.g. v1_GO008 +16 km, v1_GO009 +16 km, v1_GO023 +17 km), as the faults verifier asked.</p>""",
    clu=f"""<p>{clu_line('Gorda')}</p>
<p>The Gorda region is the clearest two-setting region. Near the ridge (within 15 km of a PB2002 ridge step; {EX['gorda_ridge']['n_above_Mc']} events above Mc)
{EX['gorda_ridge']['pct_swarm']:.0f}% of the events are in swarms and {EX['gorda_ridge']['pct_MA']:.0f}% in mainshock-aftershock families; in the interior
({EX['gorda_interior']['n_above_Mc']:,}) the shares are {EX['gorda_interior']['pct_swarm']:.0f}% and {EX['gorda_interior']['pct_MA']:.0f}%, driven by the 2014 Mw 6.8 sequence.
Among the Gorda box's families of 5 or more, 48% are swarms (v4 65%, on 25 families). Swarm members are shallower than mainshock-aftershock members
(family medians 9.1 against 25.3 km, Mann-Whitney p = 0.004; verifier), but most Gorda depths are biased_deep, so this rests on 125 depth_good swarm events,
27% of them in one family.</p>""",
    sw="""<p>Eleven swarms of 5 or more events above Mc; for all of them the nearest PB2002 step is a ridge step (median 22 km away; PB2002 traces are coarse). The 2013-05-20 Escanaba Trough swarm (family 736,
54 events, Mmax 4.0) runs 10 km along the axis (169°) with a good-depth median of 12 km bsl (about 9 km below the seafloor). v4 reported it migrating
at about 10 km/day; in v5.1 its rho is 0.46, below the test, so a dike-like migration is no longer supported. The 2013-11-13 swarm (family 989, 19 events,
16 tier A-C) passes the test with about 40 km/day over 9 hours (permutation p = 0.003), but on 16 events with a median horizontal error of 3.8 km and
depths of 33-36 km that are not physical for the ridge; treat it as a candidate only.</p>""",
    lit=f"""<p>Agrees: the near-station interior depths ({gi.v51_med:.0f} km below the seafloor) with the thermal limit and the 9-10 km seismogenic thickness
(Rollins & Stein 2010); NE trends in the interior with Chaytor et al. (2004). Disagrees: all-station interior and ridge depths, too deep by 7-15 km. The
deepening away from the ridge reported by Ren et al. (2025, abstract; same OBS data and picks) has the same signature as our station-distance bias and
is not an independent check.</p>""",
    v4=["v4 'Gorda plate median depth 21.5 km' and slab-versus-plate questions based on it: the bulk of that depth is a station-distance bias; near-station events are at 8-11 km below the seafloor.",
        "v4 2013-05-20 Escanaba swarm 'migrates at about 10 km/day': not supported in v5.1.",
        "v4 '65% of families are swarms': 48% in v5.1 (25 families)."],
    unc="Use only events with a station within 10-15 km for depth statements; 81% of located events are biased_deep.",
    figs=[("fig_region_gorda.png", "Gorda: v4 and v5.1 maps and sections along 90°."),
          ("fig_swarm_736.png", "Gorda 2013-05-20 Escanaba Trough swarm (family 736)."),
          ("fig_swarm_989.png", "Gorda 2013-11-13 swarm (family 989).")])

mt = t1.loc["Mendocino"]
R["Mendocino"] = dict(
    loc=f"""<p>{loc_line('Mendocino')}</p>
<p>The Mendocino triple junction is well recorded (22% tier A, depth_quality ok everywhere) and v5.1 moved it deeper. {anss_line('Mendocino')}
The depth-literature report found the same 2-4 km offset onshore and near shore against NCSN and the slab literature (Ferndale aftershocks at 17-20 km,
Shelly et al. 2024; double zone at 15-25 km, Wang & Rogers 1994). The double seismic zone is still visible in the depth_good events but sits about 3 km
deeper: modes at about 23.5 and 28.5 km in v5.1 against 20.5 and 25 km in v4. The lower mode lies on the 29 km depth node of model v0.3.1, where the Gorda
region also shows a pile-up.</p>""",
    flt=f"""<p>{flt_line('Mendocino')}</p>
<p>Most Mendocino surfaces are dipping layers (dip 31-43°) within the upper Gorda slab: v1_MT002, MT003, MT004 and MT006 (1,000-2,700 events each,
70-78% background) are pieces of one slab-top layer that the detection splits differently from run to run (none is recovered on the 80% subsample except
MT003 at J = 0.56). The steep surfaces that keep a clear match are the Mendocino transform segment at 125.7-126.5°W (v1_GO001, strike 276°, vertical,
966 events, members moved +3.8 km) and the Garberville swarm volume (v1_MT010 and v1_MT011, both stable and recovered).</p>""",
    clu=f"""<p>{clu_line('Mendocino')}</p>
<p>The v4 description holds: the triple junction has the shallowest trees (median leaf depth 1.56; v4 1.33), the fewest swarms (28% of the MT box's families of
5 or more; v4 31%) and low productivity (0.51 offspring per event 1-2 units above Mc; v4 0.40), consistent with Hsu et al. (2026). Shuffled catalogs
reproduce 43% of the MT box links, more than at Blanco (39%), because the 2010 and 2014 sequences dominate the clustered events. One near-threshold link
changed: the 2010-02-04 Mw 5.9 event is 54 km from the 2010-01-10 Mw 6.5 event in v5.1 (44 km in v4), misses the threshold and heads its own family,
so the 2010 family has 579 events above Mc instead of 824. The 2015-01-28 Mw 5.7 Mendocino fault zone family (682 events) is unchanged in size.</p>""",
    sw="""<p>Seven small swarms (median 14 events), median t90 5 hours, median good depth 20 km, within the slab layers. None migrates.</p>""",
    lit="""<p>Agrees: low productivity and high background (Hsu et al. 2026); a double seismic zone (Wang & Rogers 1994; McCrory et al. 2012). Disagrees:
absolute depths are 3-4 km deeper than NCSN and the relocated sequences (Shelly et al. 2024; Guo et al. 2025), so assignments of events to slab crust or
slab mantle cannot be made from v5.1 depths. Separation within the double zone (about 5 km) can still be used.</p>""",
    v4=["v4 depths closer to NCSN (+1.5 km) than v5.1 (+3.6 km): the relocation increased the Mendocino offset; slab-crust versus slab-mantle statements wait on it.",
        "v4 2010 Mw 6.5 family with the Mw 5.9 aftershock: in v5.1 the Mw 5.9 heads its own family; membership near the threshold is not stable."],
    unc="Absolute depths +3 to +4 km against NCSN; the lower layer at the 29 km node may be partly a model artefact.",
    figs=[("fig_region_mendocino.png", "Mendocino: v4 and v5.1 maps and sections along 90°.")])

fo = t1.loc["Offshore forearc"]
fort = bm("Fortuna cluster")
R["Offshore forearc"] = dict(
    loc=f"""<p>{loc_line('Offshore forearc')}</p>
<p>The forearc changed little at the median ({fo.v4_depth_med:.1f} to {fo.depth_med:.1f} km), and the named benchmarks hold: the Fortuna cluster
({int(fort.n):,} tier A-C events within 6 km of 40.6°N 124.1°W) is at a median {fort.v51_med:.1f} km (p10-p90 {fort.v51_p10:.1f}-{fort.v51_p90:.1f})
against 17-23 km (Alongi et al. 2021). {anss_line('Offshore forearc')} The region holds {int(fo.non_earthquake):,} flagged non-earthquakes, most of them the
Texada Island quarry blasts (removed from all statistics here).</p>
<p>Two forearc results need a check before use. The 2012 Brookings swarm (v1_GO022; 39 events at 42.21°N 124.74°W, 33 in 2012) is at a median 52 km in v5.1
against 20 km in v4, with depth_quality ok, sig_z 7 km and the nearest station 45 km away; Slab2 puts the slab top at 15-20 km there, so 52 km would be
30 km inside the slab. v1_ON017 (80 events near 49.4°N 127.2°W off Vancouver Island) moved +15 km to 32 km.</p>""",
    flt=f"""<p>{flt_line('Offshore forearc')}</p>
<p>The Fortuna slab-crust layer (v1_MT001, 3,038 events, dip 49°, matched to v0 MT004, recovered on the subsample, J = 0.60) and the 2010 Mw 6.5 source fault
(v1_MT016, NE strike, dip 80°) keep their v0 matches. Off Oregon, the shelf clusters west of Yachats/Waldport (v1_OR001, 81 events) and off Newport near
Stonewall Bank (v1_OR002, 47 events) are both stable and recovered on the subsample. Off Washington and Vancouver Island only v1_WA001 (148 events, Barkley Sound,
no published counterpart) is stable.</p>""",
    clu=f"""<p>{clu_line('Offshore forearc')} The offshore Oregon box (OR) has the second-highest shuffled-to-real link ratio ({t3b.loc['OR','null_over_real_links']:.0%})
and its background share moves from {t3b.loc['OR','bg_tierABC']:.0f}% (tier A-C only) to {t3b.loc['OR','bg_eta0_shuf5']:.0f}% (conservative threshold), so the few
Oregon families are weakly supported. Offshore Washington has no mainshock-aftershock family of 5 or more events above Mc, as in v4.</p>""",
    sw="""<p>Nine swarms. The Fortuna-area swarms of June-July 2014 (families 1315 and 1347, 70 and 109 events) lie on v1_MT001 at 17-21 km, with extents of
4-7 km, durations of 4-7 days at the 90% level and no resolvable spreading. They are slab-crust swarms at the depth of the Fortuna cluster.</p>""",
    lit=f"""<p>Agrees: Fortuna cluster depth (Alongi et al. 2021) and the Newport-shelf clusters at 8-13 km (Tréhu et al. 2018; v1_OR001 at 9.6 km). Partly agrees:
diffuse, background-dominated seismicity off Washington (Stone et al. 2018; Morton et al. 2023). Not resolved: depths off Washington and Vancouver Island
(+1 to +4 km against Morton et al. 2023 and PNSN in the depth-literature report).</p>""",
    v4=["v0 GO020 'Brookings swarm, candidate new structure at about 12 km': v5.1 puts it at 42-65 km, below the slab top; neither depth is established.",
        "v4 'offshore Oregon 10th-percentile depth 0.0 km' (fixed depths): gone in v5.1 (p10 6 km); 442 forearc events are floor_limited instead."],
    unc="Brookings and v1_ON017 depths suspect; off-Washington depths not benchmarked.",
    figs=[("fig_region_offshore_forearc.png", "Offshore forearc: v4 and v5.1 maps and sections along 90°."),
          ("fig_swarm_1347.png", "Fortuna-area swarm of 2014-07-16 (family 1347).")])

pu = t1.loc["Puget/Georgia"]
pg = bm("Central Puget")
ns_pu = EX["near_surface_Puget/Georgia"]
R["Puget/Georgia"] = dict(
    loc=f"""<p>{loc_line('Puget/Georgia')}</p>
<p>Puget/Georgia is the control region: the median moved by {pu.shift_z_med:+.1f} km. Central Puget Lowland crustal events (tier A-B, depth below 35 km) are
at a median {pg.v51_med:.1f} km (p10-p90 {pg.v51_p10:.1f}-{pg.v51_p90:.1f}; v4 {pg.v4_med:.1f}) against 15-25 km with a mean of 17.6 km (Blakely et al. 2002;
Van Wagoner et al. 2002). {anss_line('Puget/Georgia')} {ns_pu['n']} events are within 1 km of the surface and {100 * ns_pu['weekday_daytime']:.0f}% of them
fall on weekdays 08-17 local time (region: {100 * ns_pu['region_weekday_daytime']:.0f}%): most are probably blasts that the ComCat match and the Texada
rule did not flag.</p>""",
    flt=f"""<p>{flt_line('Puget/Georgia')}</p>
<p>The share of Puget events on a surface dropped most of any region (v0 61% of the fit set, v1 34%), because v0 kept flat seismicity layers (PU001, PU002) that
the dip rule now removes in every pass. What remains is a set of moderately dipping zones: the Western Rainier Seismic Zone (v1_PU001, stable, J = 0.89), the
Skipjack Island fault cluster (v1_ON006, stable, J = 0.93), the western Seattle fault zone (v1_PU002, stable but J = 0.45) and the Leech River fault zone
(v1_PU006, J = 0.61). Candidates on the Tacoma fault (v1_PU005) and the SE end of the SWIF (v1_PU012) are not recovered on the subsample (J = 0.18 and 0.29).
The question of whether crustal seismicity concentrates on the Seattle, Tacoma and SWIF zones therefore still needs the distance-to-CFM test against
randomized catalogs; v1 adds only two robust surfaces near those faults.</p>""",
    clu=f"""<p>{clu_line('Puget/Georgia')} Among depth_good events above Mc, {EX['puget_bg_deep30']:.0f}% of those at 30 km or deeper (intraslab, {EX['puget_n_deep30_good']} depth_good events)
are background, against {EX['puget_bg_shallow30']:.0f}% of the crustal ones, as in v4.</p>""",
    sw="""<p>Twelve swarms, all crustal (good-depth median 15 km), with long durations (median 9 days at the 90% level, 30 days in all) and small extents
(median 1.7 km). The 2012-10-05 swarm (family 431, 59 events, 14-19 km) is on v1_PU002, the western Seattle fault zone surface. None migrates.</p>""",
    lit="""<p>Agrees: crustal band and slab depths (Van Wagoner et al. 2002; Blakely et al. 2002; Preston et al. 2003); background-dominated intraslab seismicity.</p>""",
    v4=["v4/v0 'most Puget events on a fault' (61% of the fit set on v0 surfaces): v1 finds 34%; the v0 share included flat layers that are not faults.",
        "v4 'q10 depth of 1.1 km suggests quarry blasts': 411 flagged events are now removed, but about 1,000 near-surface, daytime-heavy events remain unflagged."],
    unc="Depths agree with the literature within about 2 km; unflagged blasts affect shallow statistics.",
    figs=[("fig_region_puget_georgia.png", "Puget/Georgia: v4 and v5.1 maps and sections along 90°."),
          ("fig_swarm_431.png", "Puget Lowland swarm of 2012-10-05 (family 431).")])

ca = t1.loc["Cascades"]
msh = EX["volcano_Mount St. Helens"]
hood = EX["volcano_Mount Hood"]
R["Cascades"] = dict(
    loc=f"""<p>{loc_line('Cascades')}</p>
<p>At Mount St. Helens (within 10 km of the summit) tier A-B events are at a median {bm('Mount St. Helens (10 km of summit), tier A-B').v51_med:.1f} km bsl, and the
event-paired offset from Hirao et al. (2025) is +0.5 km (all tiers) to +0.8 km (tier A-B; paired_v5.1.md), within the combined errors. {anss_line('Cascades')}
The deeper half of the Cascades distribution has a problem: {len(casc_deep)} events are deeper than 30 km ({casc_deep_msh} within 15 km of Mount St. Helens), {deep_pass_onshore['Cascades']}
come from the deep pass (median {dp_onshore_med:.0f} km for the onshore deep-pass events), and they are mostly tier D (451) and C (188) with a median sig_z of 15 km although the
nearest station is a median 11 km away. Within 15 km of Mount St. Helens, {MW['deep']} events are deeper than 30 km (v4 median {MW['deep_v4_depth_med']:.0f} km, v5.1
{MW['deep_depth_med']:.0f} km); {MW['deep_by_year'].get('2011', 0) + MW['deep_by_year'].get('2012', 0)} of them are from 2011-2012, and Wednesdays carry {MW['deep_dow'].get('2', 0)}
against {min(v for k, v in MW['deep_dow'].items())}-{max(v for k, v in MW['deep_dow'].items() if k != '2')} on the other days; {MW['deep_wed_window']} fall on Wednesdays
between 19:14 and 19:20 UTC, against {MW['shallow_wed_window']} shallow events in that window. A weekly repeat at the same minute points to a recurring
non-earthquake signal (a test or calibration pulse) picked and associated as events. No published study has Mount St. Helens seismicity at 30-118 km.</p>
<p>Mount Hood: {hood['n']} events within 15 km, none tier A-B, nearest station a median {hood['dist_min_med']:.0f} km away (the stations closest to Hood may be missing from the station set; to check),
so Hood depths are not usable.</p>""",
    flt=f"""<p>{flt_line('Cascades')}</p>
<p>The St. Helens seismic zone is the best-imaged onshore structure: v1_ON001 (3,393 events, dip 47°, stable, J = 0.71; the dextral offset zone / east
flank) and v1_ON002 (1,840 events, the SHZ main strand, strike 169°, dip 82°; 40% swarm events from the 2011 swarm). v1_ON010 and v1_ON013 keep their
Mount Hood fault zone matches but rest on tier C/D depths. v1_ON016 (the recurring swarm source about 19 km north of Mount McLoughlin) is stable
(J = 0.93) and still the most swarm-dominated surface onshore (68%). v1_ON028 (42 events at 23-38 km under Mount St. Helens) belongs to the suspect deep population and is
flagged.</p>""",
    clu=f"""<p>{clu_line('Cascades')} At Mount St. Helens (within 15 km) {msh['n_above_Mc']} events are above the regional Mc and {msh['pct_bg']:.0f}% are background;
with a local Mc the clustering piece finds 269 families (v4 308), none meeting two swarm indicators, and a shuffled catalog that is almost as bimodal
(D 1.71 against 2.27). The volcano clustering is therefore not a basis for a swarm/burst statement, as in v4.</p>""",
    sw=f"""<p>Twenty-two swarms, 17 within 25 km of a volcano. The 2011-01-28 Mount St. Helens swarm (family 150, 358 events, 20 above Mc, Mmax 3.9) lies
on v1_ON002, 10 km north of the summit, at 4.1-5.7 km bsl, lasted a year in all with most activity in February 2011, and does not migrate. Family 830
(2013-08-03 to 2014-01-29, 338 events, 166 tier A) is a compact source 21 km NW of the summit, 1.1 km across and at 13.1-14.6 km depth, active for six months;
v4 had it at 13 km too. It is a deep, well-located persistent cluster in the SHZ; its timing (136 of 338 events on one Saturday) and tier A locations
point to a real swarm, not to the Wednesday signal.</p>""",
    lit="""<p>Agrees: Mount St. Helens depths (Hirao et al. 2025, +0.5 to +0.8 km); SHZ geometry. Disagrees: the 30-118 km population under the volcano.
Not testable: Hood, Shasta, Lassen depths (tier C/D, distant stations).</p>""",
    v4=["v4 Cascades median depth 4.7 km: v5.1 7.1 km, partly from the deep population (659 events deeper than 30 km) that should not be interpreted.",
        "v0 Mount Hood fault-zone surfaces: traces only; depths rest on tier C/D events."],
    unc="Exclude events deeper than 30 km within 15 km of Mount St. Helens and the onshore deep-pass events until the weekly signal is checked.",
    figs=[("fig_region_cascades.png", "Cascades: v4 and v5.1 maps and sections along 0° (north-south)."),
          ("fig_msh_deep_weekly.png", "Mount St. Helens: hour of the week of events deeper and shallower than 30 km, depth against time, and v4 against v5.1 depth of the deep events."),
          ("fig_swarm_150.png", "Mount St. Helens swarm of 2011-01-28 (family 150)."),
          ("fig_swarm_830.png", "Compact SHZ swarm of 2013-08-03 (family 830).")])

oo = t1.loc["Other onshore"]
ns_oo = EX["near_surface_Other onshore"]
R["Other onshore"] = dict(
    loc=f"""<p>{loc_line('Other onshore')}</p>
<p>{anss_line('Other onshore')} The depth-literature checks still apply: Oregon and SW Washington within about 0.2 km of PNSN, the Scotts Mills area near
the 13-15 km mainshock depth, and the northern California slab 2-3 km deeper than NCSN (the Mendocino offset). {ns_oo['n']:,} events are within 1 km of the
surface and {100 * ns_oo['weekday_daytime']:.0f}% of them are on weekdays 08-17 local time (region: {100 * ns_oo['region_weekday_daytime']:.0f}%), so most are
probably unflagged blasts. {deep_pass_onshore['Other onshore']} events come from the deep pass (39 tier D).</p>""",
    flt=f"""<p>{flt_line('Other onshore')}</p>
<p>The stable, recovered surfaces are small and local: the 2015-07-04 Springfield earthquake fault (v1_ON009, J = 0.97), the Salmon Inlet site of the
2025 Mw 4.8 earthquake (v1_ON014, J = 0.93) and its companion (v1_ON012, J = 1.00). The southern continuation of the SHZ toward Portland (v1_ON005,
v1_ON007) and the Portland Hills fault candidate (v1_ON008, dip 32°, J = 0.70) are 80-86% background.</p>""",
    clu=f"""<p>{clu_line('Other onshore')}</p>""",
    sw="""<p>Nine small swarms (median 7 events, median t90 1.2 days); none migrates.</p>""",
    lit="""<p>Agrees with PNSN and the Oregon benchmarks; northern California slab +2-3 km against NCSN, inside the broad published ranges.</p>""",
    v4=["No v4 interpretation specific to this region is overturned; the near-surface population is still contaminated by blasts."],
    unc="Coast Ranges south of 40°N and distant volcanoes: tier C/D only.",
    figs=[("fig_region_other_onshore.png", "Other onshore: v4 and v5.1 maps and sections along 90°.")])

# ----------------------------------------------------------------------------------------------- tables for the margin-wide section
T1 = t1.reset_index()[["region", "located", "no_hypocentre", "non_earthquake", "pctA", "pctD", "floor_limited", "far_only",
                       "deep_pass", "depth_good", "v4_depth_med", "depth_med", "bsf_med", "good_v4_depth_med",
                       "good_depth_med", "shift_h_med", "shift_z_med", "sig_z_med", "dist_min_med", "v4_above_surface_pct"]]
T1.columns = ["region", "located", "no hypocentre", "non-eq", "% A", "% D", "floor_lim", "far_only", "deep_pass", "depth_good",
              "v4 depth med", "v5.1 depth med", "v5.1 below surface", "good: v4 med", "good: v5.1 med", "dh med km", "dz med km",
              "sig_z med", "nearest sta km", "v4 % above surface"]
T2 = t2.reset_index()[["region", "n_good", "nn5_3d_v4", "nn5_3d_v51", "depth_iqr_good_v4", "depth_iqr_good_v51",
                       "share_shared_epicentre", "nn1_distinct_epi_v51", "nn1_epi_all_v4", "nn1_epi_all_v51"]]
T2.columns = ["region", "depth_good", "5th-NN 3D v4 km", "5th-NN 3D v5.1 km", "depth IQR v4", "depth IQR v5.1",
              "% sharing an epicentre", "NN between distinct epicentres km", "NN epicentral v4 km", "NN epicentral v5.1 km"]
T3 = t3.reset_index()[["region", "n_above_Mc", "pct_background", "pct_MA", "pct_swarm", "pct_multiplet", "n_above_Mc_v4",
                       "pct_background_v4", "pct_MA_v4", "pct_swarm_v4", "tierD_share_above_Mc", "bg_share_tierD",
                       "bg_share_tierABC", "fam5_mainshock-aftershock", "fam5_swarm"]]
T3.columns = ["region", "n >= Mc", "% bg", "% MA", "% swarm", "% multiplet", "n >= Mc v4", "% bg v4", "% MA v4", "% swarm v4",
              "% tier D", "% bg of tier D", "% bg of A-C", "MA families (>=5)", "swarms (>=5)"]
T3B = t3b.reset_index()[["region", "bg_v4", "bg_main", "bg_tierABC", "bg_eta0_shuf5", "bg_d3q", "bg_mc_best", "real_frac_linked",
                         "frac_linked_mean", "null_over_real_links"]].copy()
T3B["h_max tier D"] = T3B.region.map(hmaxD)
T3B["h_max tier A-C"] = T3B.region.map(hmaxABC)
T3B.columns = ["box", "% bg v4", "% bg v5.1", "% bg tier A-C only", "% bg conservative eta0", "% bg 3-D good depths",
               "% bg Mc_best", "real linked fraction", "shuffled linked fraction", "shuffled / real", "h_max tier D km", "h_max tier A-C km"]
T4 = t4.reset_index()[["region", "surfaces", "steep", "members", "stable", "recovered_sub80", "recovered_tierAB", "surfaces_v0",
                       "members_v0", "earthquakes", "pct_on_fault", "ambiguous", "none", "not_fitted", "rms_normal_v0", "rms_normal_v1"]]
T4.columns = ["region", "v1 surfaces", "steep", "fit members", "stable (reruns)", "recovered 80%", "recovered A-B", "v0 surfaces",
              "v0 members", "located eq", "% on a surface", "ambiguous", "no surface", "not fitted", "RMS v0 km", "RMS v1 km"]
T6 = t6.reset_index()[["region", "n_paired", "nets", "dz_paired_v4", "dz_paired_v51", "absdz_paired_v4", "absdz_paired_v51"]]
T6 = T6[T6.n_paired >= 10]
T6.columns = ["region", "matched in both", "networks", "v4 - ANSS km", "v5.1 - ANSS km", "|v4 - ANSS| km", "|v5.1 - ANSS| km"]
T7 = t7[["benchmark", "n", "v51_med", "v51_p10", "v51_p90", "v4_med", "literature", "datum"]].copy()
T7.columns = ["benchmark", "n", "v5.1 median", "v5.1 p10", "v5.1 p90", "v4 median", "literature", "datum"]
T8 = t8.copy()
T8.columns = ["region", "model node km", "events within 0.25 km", "expected", "excess", "excess % of region"]
MG = MAGR.reset_index()[["region", "n", "median", "p25", "p75"]]
MG.columns = ["region", "events in both", "median dML v5.1 - v4.1", "p25", "p75"]

V4ROWS = []
for r in REGIONS:
    for s in R[r]["v4"]:
        V4ROWS.append((r, s))

# ----------------------------------------------------------------------------------------------- html
CSS = """
:root{--bg:#ffffff;--fg:#1d1d1f;--mut:#5b5b60;--line:#d9d9de;--acc:#1f5fa8;--box:#f4f6f9;--warn:#fff4e5;--warnl:#e6a23c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16171a;--fg:#e8e8ea;--mut:#a0a0a8;--line:#34353a;--acc:#7fb0ef;--box:#1f2125;--warn:#2b2418;--warnl:#b07a2a}}
:root[data-theme="dark"]{--bg:#16171a;--fg:#e8e8ea;--mut:#a0a0a8;--line:#34353a;--acc:#7fb0ef;--box:#1f2125;--warn:#2b2418;--warnl:#b07a2a}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:0 16px}
main{max-width:1180px;margin:0 auto;padding:24px 0 80px}
h1{font-size:26px;margin:8px 0 4px}h2{font-size:21px;margin:40px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:17px;margin:24px 0 6px}h4{font-size:15px;margin:16px 0 4px;color:var(--acc)}
p,li{max-width:1000px}.mut{color:var(--mut)}
figure{margin:12px 0}figure img{max-width:100%;height:auto;border:1px solid var(--line);background:#fff}
figcaption{font-size:13px;color:var(--mut)}
table.t{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto;max-width:100%}
table.t th,table.t td{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;vertical-align:top}
table.t th{background:var(--box);position:sticky;top:0}table.t td:first-child,table.t th:first-child{text-align:left}
.box{background:var(--box);border:1px solid var(--line);border-radius:6px;padding:10px 14px;margin:12px 0}
.warn{background:var(--warn);border-left:4px solid var(--warnl);padding:8px 12px;margin:12px 0}
nav a{margin-right:10px;white-space:nowrap}code{font-size:13px}
"""


def region_html(r):
    d = R[r]
    figs = "".join(img(f, cap) for f, cap in d["figs"])
    v4 = "".join(f"<li>{html.escape(s)}</li>" for s in d["v4"])
    return f"""<h2 id="{SLUG[r]}">{html.escape(r)}</h2>
<h4>What the v5.1 locations change</h4>{d['loc']}
<h4>Faults: which are active and how</h4>{d['flt']}
<details><summary>v1 surfaces with most members in this region</summary>{surf_table(r)}</details>
<h4>Sequences, swarms and background</h4>{d['clu']}
<h4>Swarm and sequence characteristics</h4>{d['sw']}
<details><summary>Largest families led by an event in this region</summary>{swarm_rows(r)}</details>
<h4>Against published studies</h4>{d['lit']}
<h4>v4-era interpretations that no longer hold</h4><ul>{v4}</ul>
<div class="warn"><strong>Uncertainty.</strong> {html.escape(d['unc'])}</div>
{figs}"""


nav = " ".join(f'<a href="#{SLUG[r]}">{html.escape(r)}</a>' for r in REGIONS)
n_none = int((~c.located).sum())
n_nonq = int((c.located & ~c.is_earthquake).sum())
bodies = "".join(region_html(r) for r in REGIONS)
today = date.today().isoformat()

summary = f"""
<ul>
<li><strong>Inputs.</strong> Catalog v5.1 (172,342 rows; {n_none} rows without a hypocentre and {n_nonq:,} flagged non-earthquakes are left out of every
spatial statistic, leaving {int(c.is_eq.sum()):,} located earthquakes), the v5.1 magnitudes, the v5.1 clustering, fault surfaces v1, their verifiers'
corrections, the v4-era science report and v0 literature matches, and the depth-literature report.</li>
<li><strong>Where v5.1 depths can be interpreted.</strong> Axial (81% of tier A-B within 2 km of the seafloor against 90% published), Mount St. Helens
(+0.5 to +0.8 km against Hirao et al. 2025), Puget/Georgia (crust at a median {pg.v51_med:.1f} km against 15-25 km), the Fortuna cluster ({fort.v51_med:.1f} km against 17-23 km)
and the Gorda interior near stations ({gi.v51_med:.1f} km below the seafloor, inside the 9.5-15 km thermal limit).</li>
<li><strong>Where they cannot.</strong> Endeavour, Explorer (seaward median {b_ex.v51_med:.0f} km bsl against at most 22 km published; v4 {b_ex.v4_med:.0f} km), the
Juan de Fuca ridge (41% floor_limited), the Gorda plate away from stations (biased_deep, {gia.v51_med:.0f} km below the seafloor), the Blanco outside the
2012-13 X9 year and on the Blanco Ridge and Gorda Depression (+3 to +6 km against Lange et al. 2026), Mendocino in absolute terms (+3.6 km against NCSN,
v4 +1.5 km), and 659 Cascades events deeper than 30 km.</li>
<li><strong>New catalog problems found here.</strong> (1) {MW['deep']} events deeper than 30 km within 15 km of Mount St. Helens with a Wednesday excess
and {MW['deep_wed_window']} events in a 6-minute Wednesday window: a recurring non-earthquake signal. (2) Pile-ups at the depth nodes of model v0.3.1:
{NODE[('Axial', 3)].excess:,} excess events at 3 km at Axial (13% of the region), {NODE[('Blanco', 12)].excess + NODE[('Blanco', 15)].excess:,} at 12 and 15 km on the Blanco,
{NODE[('Gorda', 29)].excess} at 29 km in Gorda. (3) About 2,100 near-surface events in Puget/Georgia and Other onshore with a weekday-daytime share
of 61-68% (regional 29-31%): probably unflagged blasts. (4) The 2012 Brookings swarm at 52 km, below the slab top.</li>
<li><strong>Faults.</strong> Of 142 v1 surfaces, the verifier recovers 38 on an 80% subsample; the strand geometry of the Blanco and the slab layers at
Mendocino are not stable under re-detection. The surfaces that are both stable and recovered are mostly local: SHZ (v1_ON001), Western Rainier Seismic Zone,
Skipjack Island, Garberville, Springfield, Salmon Inlet, the Oregon shelf clusters, the Fortuna layer and the Explorer traces (map only). The Cascadia Depression
N-S faults of v0 are not recovered because of detection order, not because of the relocation.</li>
<li><strong>Clustering.</strong> The ordering by setting holds where it is robust: the West Blanco Depression is swarm-rich (45% of events above Mc in swarms,
shuffled-to-real link ratio 0.39), the Mendocino triple junction is burst-like with low productivity (28% swarms among families of 5 or more), and the forearc
and Puget are background-dominated. The Axial result is not robust (background 43-82% depending on the variant; 75% of its links appear in shuffled catalogs).</li>
<li><strong>Swarms.</strong> Most swarms are as small as the location errors (median 90% radius 2.2 km against a 2.0 km horizontal error); 15 of 129 are
resolvable. Two small swarms pass the migration test (Gorda 2013-11-13, 16 events, about 40 km/day; Cascades 2012-09-10, 11 events, 0.2 km/day).
Two v4 migration results are gone (2013-05-20 Escanaba swarm, 10 km/day in v4; the 2012-11 Explorer sequence). The 2012-02-04 Explorer sequence still passes the test (0.8 km/day), but the space-time plot shows a second cluster 15 km NNE activated 9-12 days after the mainshock, not a continuous front.</li>
<li><strong>v4 interpretations withdrawn or changed:</strong> {len(V4ROWS)} statements, listed in the last table.</li>
</ul>"""

htmldoc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Catalog v5.1 Interpretation</title><style>{CSS}</style></head><body><main>
<h1>Catalog v5.1: seismicity, fault surfaces v1 and swarms, region by region</h1>
<p class="mut">Cascadia OBS ensemble catalog v5.1 (NonLinLoc in model v0.3.1, 2010-2016). Piece "interp", {today}. Private working product, not peer reviewed.
Inputs: magnitudes (branch analysis/v51-mag, {H['mag']}), fault surfaces v1 (analysis/v51-faults, {H['faults']}), clustering (analysis/v51-clusters, {H['clusters']}),
and the three verifiers' tables on /wd1. Code: workflow/06_analysis/v51_interp_*.py on analysis/v51-interp.</p>
<nav>{nav} <a href="#problems">Catalog problems</a> <a href="#v4">v4 interpretations</a> <a href="#decisions">Decisions</a></nav>
<h2>Summary</h2>{summary}
<div class="box"><strong>How to read the numbers.</strong> Depths are km below sea level (bsl) unless written "below the seafloor" or "below the surface"
(depth + surface_km). "depth_good" = tier A-C, depth_quality ok, not floor_limited, far_only or deep_pass ({int(c.depth_good.sum()):,} events);
Endeavour, Explorer and the JdF ridge have none. Regions are the catalog's <code>region</code> labels (set from the v4 epicentre); the clustering
uses its nine boxes (AX, EN, BL, GO, MT, WA, OR, PU, ON) and the fault surfaces their v0 boxes, so box-level numbers are quoted with the box name.
v4 depths have an unconfirmed datum: read as km bsl, 4.8-70% of them (by region) plot above the ground or seafloor, so v4-v5.1 depth changes
offshore mix the relocation with the datum. Corrections from the three verifiers are applied in the text: tier D dominates the large ML changes (not Mount St. Helens);
b decreases by 0.02-0.08 on the same events in most boxes; fault uncertainties (sigma_total 0.59 km) hold only for fixed members; shuffled catalogs
reproduce 21-75% of the cluster links.</div>

<h2 id="margin">Margin-wide comparison with v4</h2>
{img('fig_overview.png', 'Located earthquakes of v4 (left) and v5.1 (right) coloured by depth, with the v0 and v1 fault-surface top edges.')}
<h3>Locations</h3>
<p>The relocation moved epicentres by a median 1-8 km depending on the region (Explorer most) and depths by a median of +0.7 km (Puget/Georgia) to +25 km (Explorer):
every region is deeper at the median, partly because v4 depths were biased shallow. Onshore and at Mendocino the changes are small; at Explorer, Endeavour and in the Gorda interior the v5.1 depths are deeper than any published constraint. Structure is not
sharper in v5.1 except at Axial, where the apparent sharpening comes partly from the oct-tree lattice: the median distance to the fifth neighbour of depth_good
events changes by 11% or less elsewhere, and the depth interquartile range grows on the Blanco (7.2 to 9.0 km) and at Mendocino (6.0 to 9.1 km) and shrinks in
Gorda (16 to 11 km).</p>
{tab(T1)}
{img('fig_depths_by_region.png', 'Depth distributions by region: v4 (grey), v5.1 (blue), v5.1 depth_good (orange), and the published range (green).')}
<details><summary>Structure sharpness on the same events (nearest-neighbour distances)</summary>{tab(T2, fmt={'5th-NN 3D v4 km': '{:.2f}', '5th-NN 3D v5.1 km': '{:.2f}', 'NN between distinct epicentres km': '{:.3f}', 'NN epicentral v4 km': '{:.3f}', 'NN epicentral v5.1 km': '{:.3f}'})}</details>
<h3>Depths against ANSS and named benchmarks</h3>
{tab(T6)}
<p class="mut">ANSS/ComCat earthquakes matched within 3 s and 20 km, fixed depths (no depth error, or us/at/pt at 0, 5, 10, 33, 35 km) removed. ANSS depths
are network 1-D depths; the depth-literature report shows NCSN is shallower than the OBS-based studies on the Mendocino coast and PNSN is too deep off
Oregon, so ANSS is a reference, not the truth.</p>
{tab(T7, fmt={'v5.1 median': '{:.1f}', 'v5.1 p10': '{:.1f}', 'v5.1 p90': '{:.1f}', 'v4 median': '{:.1f}'})}
<h3>Magnitudes</h3>
<p>The v5.1 ML is a median 0.05 higher than v4.1 (offshore 0.06, onshore 0.02, Axial -0.11), and the change follows the new distances and station terms.
Large changes (|dML| > 0.5, 639 events) are 83% tier D relocations (moved a median 15 km), not Mount St. Helens events (magnitude verifier). On the same
events at a common Mc, b decreases by 0.02-0.08 in EN, BL, GO, MT, OR, ON and overall (ALL 1.054 to 1.025), significant after a Bonferroni correction; at Axial
the sign depends on Mc. Offshore ML is 0.3-0.5 above ComCat ML at Mendocino, Blanco and Gorda. None of this changes a regional interpretation below, but swarm
and multiplet types that hinge on M1-M2 differences of 0.3-0.5 are within the ML uncertainty.</p>
<details><summary>ML change by region (magnitude piece)</summary>{tab(MG, fmt={'median dML v5.1 - v4.1': '{:+.3f}', 'p25': '{:+.3f}', 'p75': '{:+.3f}'})}</details>
<h3>Fault surfaces</h3>
{tab(T4)}
<p>v1 has 142 surfaces against 141 in v0, but the share of the fit set on a surface fell from 70% to 63% because the dip rule is now applied after every
reassignment. Uncertainty is reported at two levels. For a fixed set of members, the median sigma_total is 0.59 km and strike/dip intervals are a few
degrees. Across detection reruns, 93 of 142 surfaces are unstable and their strike range has a median of 103° (6.5° for the stable ones); the verifier
recovers 38 of 142 surfaces on a random 80% subsample, 13 of 128 on tiers A-B only and 12 of 130 after one perturbation of the locations by their errors.
In this report a surface is called robust only if it is stable and recovered on the subsample (J >= 0.5).</p>
{img('fig_surface_activity.png', 'Left: share of events above Mc that are background or in swarms, per v1 surface. Right: Jaccard overlap of each surface with the verifier 80% subsample rerun.')}
<h3>Clustering</h3>
{tab(T3)}
{tab(T3B, fmt={'real linked fraction': '{:.2f}', 'shuffled linked fraction': '{:.2f}', 'shuffled / real': '{:.2f}', 'h_max tier D km': '{:.1f}', 'h_max tier A-C km': '{:.1f}'})}
<p>The earlier claim that every box is bimodal and loses it in shuffled catalogs does not hold per seed (the verifier finds shuffled D up to 2.9 at EN
and 5.4 at AX, and no KDE valley at AX, WA, OR, PU and ON). The useful measure is the share of the real links that a shuffled catalog reproduces (right column
above): 21-29% in EN, GO, ON, PU and WA, 39% on the Blanco, 43-45% at Mendocino and off Oregon, 75% at Axial. Tier D epicentres are poor outside Axial
(median h_max 9-45 km by box, against 1.2-4.1 km for tiers A-C); 16% of the links have a tier D end, 48% in the ON box.</p>
{img('fig_cluster_robustness.png', 'Left: background share in the main run and variants. Right: linked fraction in time-shuffled catalogs relative to the real catalog (verifier).')}
{bodies}
<h2 id="problems">Catalog problems found in this interpretation</h2>
<ol>
<li><strong>Deep events under Mount St. Helens.</strong> {MW['deep']} events within 15 km of the summit are deeper than 30 km (tiers D {MW['deep_tiers'].get('D', 0)}, C {MW['deep_tiers'].get('C', 0)},
B {MW['deep_tiers'].get('B', 0)}, A {MW['deep_tiers'].get('A', 0)}; median sig_z {MW['deep_sigz_med']:.0f} km with the nearest station {MW['deep_distmin_med']:.0f} km away).
{MW['deep_dow'].get('2', 0)} are on Wednesdays and {MW['deep_wed_window']} between 19:14 and 19:20 UTC on Wednesdays. v4 had them at a median {MW['deep_v4_depth_med']:.0f} km;
v5.1 at {MW['deep_depth_med']:.0f} km. Check the waveforms of the Wednesday-window events and the stations that pick them.</li>
<li><strong>Depth-node pile-ups.</strong> {tab(T8, fmt={'excess % of region': '{:.2f}', 'expected': '{:.0f}'})} Linear interpolation between the depth nodes of model v0.3.1
puts a kink in the velocity gradient at each node, and maximum-likelihood depths collect there. At Axial the 3 km node (about 1.5 km below the caldera floor)
holds 13% of the events; it lies inside the 2 km band, so part of the 81% within 2 km of the seafloor is set by the model grid, not by the data.</li>
<li><strong>Unflagged near-surface events onshore.</strong> {ns_pu['n']} (Puget/Georgia) and {ns_oo['n']:,} (Other onshore) events within 1 km of the surface, 68% and 61%
on weekdays 08-17 local time. A weekday-daytime flag on shallow onshore events, like the Texada rule, would remove most of them.</li>
<li><strong>Brookings 2012 swarm</strong> (v1_GO022) at 52 km with depth_quality ok, 30 km below the slab top; and v1_ON017 off Vancouver Island moved +15 km.</li>
<li><strong>Onshore deep-pass events</strong>: {sum(deep_pass_onshore.values())} events in Cascades, Other onshore and Puget/Georgia at a median {dp_onshore_med:.0f} km, mostly tier D.
The slab is at 40-80 km under the arc; these need the same check as the Mount St. Helens population.</li>
<li><strong>Explorer depths</strong> are a median 44 km seaward (v4 9 km); the ComCat tensor events are 25 km deeper than ComCat. The depth-quality flag
("unconstrained") is right, but the column invites misuse; consider not publishing depths there.</li>
</ol>

<h2 id="v4">v4-era interpretations revisited</h2>
{tab(pd.DataFrame(V4ROWS, columns=['region', 'v4-era statement and its status in v5.1']))}
<p>Interpretations that still hold: the Axial rate history and the eruption swarm; the swarm-rich West Blanco Depression and the swarm character of the
large Blanco multiplets; the low-productivity, burst-like Mendocino triple junction; the background-dominated forearc and Puget intraslab seismicity;
the Explorer 2012-02 sequence as a map-view two-patch sequence; the Fortuna cluster depth; the St. Helens seismic zone geometry and the Mount St. Helens depths; the Puget crustal band.</p>

<h2 id="decisions">Proposed decisions (for the main session)</h2>
<ol>
<li>Report depths only where they pass a benchmark here: Axial tiers A-B, Mount St. Helens, Puget/Georgia, Oregon/SW Washington onshore, the Fortuna cluster,
and Gorda/Blanco events with a station within 10-15 km. Elsewhere publish epicentres and flag depth.</li>
<li>Treat the Axial background/clustered share as a range (43-82%) and do not use Axial in the heat-flow ordering of clustering styles.</li>
<li>Call a v1 surface robust only when it is stable under the builder's reruns and recovered on the verifier's 80% subsample (J >= 0.5); quote the
others as candidates. Keep the v0 Cascadia Depression N-S faults until a Cascadia-Depression-only fit is run.</li>
<li>Do not report swarm diffusivities or migration speeds from v5.1 absolute locations; describe the Explorer 2012-02 sequence as a two-patch sequence.
Relocate swarms relatively (double difference) before any migration or diffusion analysis.</li>
<li>Add catalog flags in the next version: Mount St. Helens deep Wednesday population, near-surface weekday-daytime onshore events, onshore deep-pass events.</li>
</ol>

<h2>Method notes</h2>
<p class="mut">Tables and figures: <code>v51_interp_tables.py</code>, <code>v51_interp_figs.py</code>, report <code>v51_interp_report.py</code>. The joined event table
(catalog + v5.1 M + cluster class + fault assignment + nearest PB2002 step) is on /wd1 (<code>v5.1_analysis/interp/events_joined.pkl</code>). Nearest-neighbour
sharpness uses the same depth_good events in both versions, local km projection, depth in km. ANSS matches: 3 s and 20 km from our origin of each
version. Node pile-ups: events within 0.25 km of a model node against the mean density 0.25-1.25 km away. Benchmark boxes are defined in
<code>t7_benchmarks</code> of the tables script; literature values are those of the depth-literature report. PB2002 steps (Bird 2003) give the nearest
ridge/transform/subduction step. Families, swarm types and migration statistics are those of the clustering piece; surface statistics and literature
matches those of the faults piece, with the verifiers' recovery Jaccard and the flags added here (surfaces_v1_interp.csv, column interp_flags).</p>
</main></body></html>"""

htmldoc = htmldoc.replace(" < ", " &lt; ").replace(" <= ", " &lt;= ")
out = OUT / "catalog_v51_interpretation.html"
out.write_text(htmldoc)
print(out, len(htmldoc) / 1e6, "MB")
