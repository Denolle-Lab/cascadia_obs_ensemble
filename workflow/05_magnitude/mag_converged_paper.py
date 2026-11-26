"""Paper numbers and figures with the converged Route A magnitudes (D-B091): stored against converged.

Stage `numbers` recomputes every paper number that depends on the Route A magnitudes, from the products of the
paper's own scripts (v4_mc_maps.py, v4_clusters_zbz.py, v4_clusters_zbz_report.py) run once with the stored and
once with the converged magnitudes, and writes data/magnitude_converged/paper_numbers.csv (one row per number:
where it is in the manuscript, the stored value, the converged value, how it is computed).

Stage `report` writes data/magnitude_converged/paper_switch_report.html: every remade figure next to the one on
main (downscaled, embedded), the pixels that changed, the numbers table and the manuscript bullets
(data/magnitude_converged/manuscript_bullets.md).

    python mag_converged_paper.py numbers --old <stored products> --new <converged products>
    python mag_converged_paper.py report  --old <stored products> --new <converged products> [--control <dir>]

A products folder holds data/catalog_v4/clusters_zbz, data/catalog_v4/mc_maps and paper_sciadv/figures, laid out as
in the repository (the stored ones: `git archive 5d6658bf` of those paths). Defaults: --old
<WD1>/magnitude_converged/paper/stored, --new the repository itself.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import mag_paths  # noqa: E402
import v51_paths as vp  # noqa: E402

OUTD = REPO / "data" / "magnitude_converged"
PAPER_DEFAULT = vp.WD1 / "magnitude_converged" / "paper"
ZBZ = Path("data/catalog_v4/clusters_zbz")
MCM = Path("data/catalog_v4/mc_maps")
FIG = Path("paper_sciadv/figures")
OFFSHORE_PB = ("EN", "GO", "BL")                     # offshore plate boundaries (main text, pooled Mc)
ONSHORE_FA = ("WA", "PU", "ON", "OR", "MT")          # onshore and forearc (main text, pooled Mc)
RIDGES = ("AX", "BL", "GO")                          # 'ridges and transforms' of the clustering paragraph
FOREARC = ("WA", "OR", "PU", "ON")
SENS = ("df1.3", "df2.0", "df_est", "d3", "mc+0.3", "mc_year")   # Text S6 '84-99% keep their label'
STRANDS = {"BL004a": "West Blanco", "BL015a": "West Blanco (RTI)", "BL006a": "Surveyor", "BL012": "Cascadia Depression"}
SIX = ("2010-01-10 Mw 6.5 Gorda", "2014-03-10 Mw 6.8 Gorda", "2011-09-09 Mw 6.4 Vancouver Island",
       "2014-04-24 Mw 6.5 Vancouver Island", "2012-11-08 Mw 6.1 Explorer plate", "2015-01-28 Mw 5.7 Mendocino fault zone")


# ------------------------------------------------------------------ products of one magnitude version
class Products:
    def __init__(self, root: Path, mag: Path):
        self.root, self.mag = Path(root), Path(mag)
        z, m = self.root / ZBZ, self.root / MCM
        self.z, self.m = z, m
        self.reg = pd.read_csv(z / "regions_zbz.csv").set_index("region")
        self.st = pd.read_csv(z / "setting_stats.csv").set_index("region")
        self.comp = pd.read_csv(z / "completeness_b_df.csv").set_index("region")
        self.sens = pd.read_csv(z / "sensitivity_zbz.csv")
        self.sh = pd.read_csv(z / "shuffle_test.csv")
        self.fam = pd.read_csv(z / "families_zbz.csv")
        for c in ("start", "end"):
            self.fam[c] = pd.to_datetime(self.fam[c], utc=True, format="ISO8601")
        self.val = pd.read_csv(z / "validation_sequences.csv")
        self.fc = pd.read_csv(z / "faults_v0_cluster_counts.csv").set_index("surface_id")
        self.R = pd.read_csv(m / "mc_regional.csv")
        self.N = pd.read_csv(m / "mc_nodes_by_region.csv")
        self.G = pd.read_csv(m / "mc_grid.csv")
        self.GQ = pd.read_csv(m / "mc_grid_qc.csv")
        self.GP = {p: pd.read_csv(m / f"mc_grid_{p}.csv") for p in ("2010", "2011-2012", "2013-2015", "2016")}
        self._ev = None

    @property
    def ev(self):
        if self._ev is None:
            e = pd.read_csv(self.z / "events_zbz.csv.gz", low_memory=False)
            e["time"] = pd.to_datetime(e.time, utc=True, format="ISO8601")
            self._ev = e
        return self._ev

    def mc(self, reg, per="all", col="mc_best", var="all"):
        x = self.R[(self.R.region == reg) & (self.R.period == per) & (self.R.variant == var)]
        return float(x[col].iat[0]) if len(x) else np.nan

    def node(self, reg, q, per="all"):
        x = self.N[(self.N.region == reg) & (self.N.period == per)]
        return float(x[q].iat[0]) if len(x) else np.nan


def f1(v):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.1f}"


def rng(vals, fmt="{:.1f}"):
    vals = [v for v in vals if np.isfinite(v)]
    return fmt.format(min(vals)) + "-" + fmt.format(max(vals))


# ------------------------------------------------------------------ the numbers
def numbers(o: Products, n: Products) -> pd.DataFrame:
    rows = []

    def add(section, item, fn, how, text=""):
        try:
            a = fn(o)
        except Exception as ex:                    # a missing input must show, not stop the table
            a = f"ERROR {type(ex).__name__}: {ex}"
        try:
            b = fn(n)
        except Exception as ex:
            b = f"ERROR {type(ex).__name__}: {ex}"
        rows.append(dict(section=section, item=item, text=text, stored=a, converged=b, how=how))

    # ---- magnitudes (Results first paragraph, Text S5, Methods)
    def mtab(p):
        m = pd.read_csv(p.mag, usecols=["event_id", "M", "M_type", "Mw_mt"])
        t = m.M_type.value_counts()
        return (f"{m.M.notna().sum():,} with M (ML {t.get('ML', 0):,}, Mw {t.get('Mw', 0)}, Mw_cal {t.get('Mw_cal', 0)}); "
                f"{int(m.Mw_mt.notna().sum()) - 3} matched tensors")
    add("Results 1; Text S5; Methods (tensors bullet)", "events with a magnitude; types; matched moment tensors", mtab,
        "count of the Route A file; matched tensors = rows with Mw_mt minus the 3 v4.1 additions (ComCat Mw)",
        "130,769 (Text S5: 130,767); 112 matched tensors")
    runs = pd.read_csv(OUTD / "phase19_runs.csv").set_index("run")
    rows.append(dict(section="Results 1; Text S5; Methods", item="ComCat ML anchors: number, scatter (std), offset",
                     text="3,416 common events, scatter 0.21",
                     stored="3,416; 0.21 (stored run, as in the text; 12-thread legacy rerun 0.207); offset +0.413",
                     converged=f"{int(runs.loc['v4.1 converged', 'n_anchors']):,}; "
                               f"{runs.loc['v4.1 converged', 'anchor_std']:.2f}; offset "
                               f"{runs.loc['v4.1 converged', 'anchor_offset']:+.3f}",
                     how="data/magnitude_converged/phase19_runs.csv (PR #45)"))

    # ---- Mc maps (Results 1, Fig. 2, Text S7, Table S6)
    add("Results 1 (Fig. 2)", "lowest node Mc: onshore (Mount St Helens) min; Axial min",
        lambda p: f"ON {f1(p.node('ON', 'min'))}, AX {f1(p.node('AX', 'min'))}", "mc_nodes_by_region.csv, period all",
        "0.0-0.3")
    add("Results 1 (Fig. 2)", "node medians Puget Lowland, offshore Washington",
        lambda p: f"PU {f1(p.node('PU', '50%'))}, WA {f1(p.node('WA', '50%'))}", "mc_nodes_by_region.csv", "1.0-1.1")

    def highest(p):
        g = p.G.dropna(subset=["mc_best"]).sort_values("mc_best", ascending=False)
        top = g.head(5)
        return (f"max {g.mc_best.max():.1f} at {', '.join(f'{r.lon:.1f}/{r.lat:.1f}' for r in top.head(3).itertuples())}; "
                f"BL max {f1(p.node('BL', 'max'))}, GO max {f1(p.node('GO', 'max'))}; nodes >= 2.9: "
                f"{int((g.mc_best >= 2.9 - 1e-9).sum())}")
    add("Results 1 (Fig. 2)", "highest node Mc (western Blanco, northern Gorda)", highest, "mc_grid.csv (Mc_best)",
        "2.9-3.2")
    add("Results 1 (Fig. 2); Table S6", "pooled Mc onshore and forearc (WA, PU, ON, OR, MT)",
        lambda p: rng([p.mc(r) for r in ONSHORE_FA]) + " (" + ", ".join(f"{r} {f1(p.mc(r))}" for r in ONSHORE_FA) + ")",
        "mc_regional.csv, variant all, period all, Mc_best", "1.2-1.7")
    add("Results 1 (Fig. 2); Table S6", "pooled Mc offshore plate boundaries (EN, GO, BL)",
        lambda p: rng([p.mc(r) for r in OFFSHORE_PB]) + " (" + ", ".join(f"{r} {f1(p.mc(r))}" for r in OFFSHORE_PB) + ")",
        "mc_regional.csv", "1.8-2.2")
    add("Results 1; Table S6", "pooled Mc all offshore (OFF) / whole catalog (ALL)",
        lambda p: f"OFF {f1(p.mc('OFF'))}, ALL {f1(p.mc('ALL'))}", "mc_regional.csv", "(not in the text)")

    def nodes_offshore(p, per):
        x = p.N[(p.N.period == per)].set_index("region")["count"]
        return ", ".join(f"{r} {int(x.get(r, 0) if pd.notna(x.get(r, 0)) else 0)}" for r in OFFSHORE_PB)
    add("Results 1 (Fig. 2b)", "offshore nodes with 150 events within 50 km (EN, GO, BL), by period",
        lambda p: f"2010: {nodes_offshore(p, '2010')}; 2016: {nodes_offshore(p, '2016')}; 2013-2015: "
                  f"{nodes_offshore(p, '2013-2015')}", "mc_nodes_by_region.csv count (node region as the clustering boxes)",
        "no offshore node in 2010 and 2016")
    add("Results 1", "rate of M >= 2.5 on the Blanco, 2010 and 2016, relative to 2013-2015",
        lambda p: f"{100 * p.mc('BL', '2010', 'rr25'):.0f}% (2010), {100 * p.mc('BL', '2016', 'rr25'):.0f}% (2016); "
                  f"M >= 3.0: {100 * p.mc('BL', '2010', 'rr30'):.0f}%, {100 * p.mc('BL', '2016', 'rr30'):.0f}%",
        "mc_regional.csv rr25 / rr30 (annual rate of the period over the 2013-2015 rate)", "5% (2010), 8% (2016)")
    add("Results 1", "Mc 2011-2012 -> 2013-2015 (pooled), EN, BL, GO",
        lambda p: "; ".join(f"{r} {f1(p.mc(r, '2011-2012'))} -> {f1(p.mc(r, '2013-2015'))}" for r in OFFSHORE_PB),
        "mc_regional.csv Mc_best by period", "improved by 0.2-0.4 (Blanco 2.6 to 2.2)")
    add("Results 1", "node medians 2011-2012 -> 2013-2015, EN, BL, GO",
        lambda p: "; ".join(f"{r} {f1(p.node(r, '50%', '2011-2012'))} -> {f1(p.node(r, '50%', '2013-2015'))}"
                            for r in OFFSHORE_PB), "mc_nodes_by_region.csv by period", "(README: EN 2.2 to 1.8, BL 2.3 to 2.1)")

    def period_change(p):
        a, b, c = (p.GP[k].set_index(["lon", "lat"]).mc_best for k in ("2010", "2013-2015", "2016"))
        d1 = (b - a).dropna()
        d2 = (c - b).dropna()
        return f"2010 -> 2013-2015 median {d1.median():+.1f} ({len(d1)} nodes); 2013-2015 -> 2016 {d2.median():+.1f} ({len(d2)})"
    add("Results 1", "Mc change between periods where both have a node (mostly onshore)", period_change,
        "median of the node difference", "median change 0.0 to 0.1")
    add("Results 1", "rate of M >= 2.5 near Mendocino (MT), 2016 relative to 2013-2015",
        lambda p: f"{100 * p.mc('MT', '2016', 'rr25'):.0f}%", "mc_regional.csv rr25", "10%")

    def conservative(p):
        g = p.G
        v = g.dropna(subset=["mc_maxc", "mc_best"])
        hi = (v.mc_maxc >= v.mc_best) & ((v.mc_maxc >= v.mc_mbs) | v.mc_mbs.isna())
        vm = g.dropna(subset=["mc_maxc", "mc_mbs"])
        return (f"MAXC highest at {100 * hi.mean():.0f}% of nodes; MBS - MAXC {(vm.mc_mbs - vm.mc_maxc).mean():+.2f}; "
                f"best - MAXC {(v.mc_best - v.mc_maxc).mean():+.2f}; GFT95 at {100 * g.mc_gft95.notna().mean():.0f}%, "
                f"GFT90 at {100 * g.mc_gft90.notna().mean():.0f}%, MBS at {100 * g.mc_mbs.notna().mean():.0f}% of nodes")
    add("Text S7", "which method is most conservative (nodes, whole period)", conservative,
        "mc_grid.csv; 'highest' = MAXC >= best and >= MBS where MBS exists; means over nodes that have both values",
        "84% of nodes; MBS 0.19 lower; best 0.14 lower")

    def qc(p):
        a = p.G.set_index(["lon", "lat"]).mc_best
        b = p.GQ.set_index(["lon", "lat"]).mc_best
        d = (b - a).dropna()
        return f"median {d.median():+.1f}, IQR {d.quantile(.25):+.1f} to {d.quantile(.75):+.1f} ({len(d):,} nodes)"
    add("Text S7", "QC-only minus all, Mc_best per node", qc, "mc_grid_qc.csv - mc_grid.csv",
        "median 0.0, IQR +-0.1")

    def zbz_vs_methods(p):
        out = []
        for r in p.comp.index:
            c = p.comp.Mc[r]
            vals = {k: p.mc(r, col=col) for k, col in (("MAXC", "mc_maxc"), ("MBS", "mc_mbs"), ("best", "mc_best"))}
            low = [f"{k} {v:.1f}" for k, v in vals.items() if np.isfinite(v) and c < v - 1e-6]
            if low:
                out.append(f"{r} {c:.2f} < {', '.join(low)}")
        return "; ".join(out) or "none"
    add("Text S7", "regions where the clustering Mc is below a map method (pooled region)", zbz_vs_methods,
        "completeness_b_df.csv Mc against mc_regional.csv", "only Blanco MBS (2.5 against 2.45)")
    add("Text S7", "Axial pooled: bootstrap sigma of Mc_best; MAXC; node Mc 10-90%",
        lambda p: f"sigma {p.mc('AX', col='sigma_boot'):.2f}; MAXC {f1(p.mc('AX', col='mc_maxc'))}; GFT/best "
                  f"{f1(p.mc('AX'))}; nodes {f1(p.node('AX', '10%'))}-{f1(p.node('AX', '90%'))}",
        "mc_regional.csv, mc_nodes_by_region.csv", "sigma 0.94; MAXC 0.7; nodes 0.4-1.0")

    # ---- clustering (Results 'How faults slip', Discussion, Text S6, Table S5, Fig. 5)
    add("Discussion Limits; Text S6", "events above the per-region Mc (analysis set)",
        lambda p: f"{int(p.reg.n_analysis.sum()):,}", "regions_zbz.csv n_analysis", "27,555")
    add("Text S6; Table S5", "clustering Mc range and per region",
        lambda p: rng(p.reg.Mc, "{:.2f}") + " (" + ", ".join(f"{r} {v:.2f}" for r, v in p.reg.Mc.items()) + ")",
        "regions_zbz.csv Mc (MAXC + 0.2, largest yearly value)", "0.75-2.45")
    add("Text S6", "measured epicentral correlation dimension D2 (df_est) range",
        lambda p: rng(p.comp.df_est), "completeness_b_df.csv", "0.8-1.4")
    add("Results (clustering)", "background share, ridges and transforms (AX, BL, GO)",
        lambda p: ", ".join(f"{r} {p.reg.pct_background[r]:.1f}%" for r in RIDGES), "regions_zbz.csv", "about half")
    add("Results (clustering); Discussion", "swarms among families of >= 5 events (AX, BL, GO)",
        lambda p: ", ".join(f"{r} {p.st.pct_swarm_of_ge5[r]:.1f}% of {int(p.st.n_fam_ge5[r])}" for r in RIDGES),
        "setting_stats.csv", "two thirds")
    add("Results (clustering)", "Mendocino triple junction: swarm share of families >= 5; events Mc+1..2 with no aftershock",
        lambda p: f"{p.st.pct_swarm_of_ge5['MT']:.1f}% of {int(p.st.n_fam_ge5['MT'])}; "
                  f"{p.st.pct_Mc1to2_no_offspring['MT']:.1f}% of {int(p.st.n_events_Mc1to2['MT'])}",
        "setting_stats.csv", "31%; 76%")
    add("Results (clustering)", "background share, forearc, Puget and onshore (WA, OR, PU, ON)",
        lambda p: rng([p.reg.pct_background[r] for r in FOREARC], "{:.0f}") + "% (" +
                  ", ".join(f"{r} {p.reg.pct_background[r]:.1f}" for r in FOREARC) + ")", "regions_zbz.csv", "68-84%")

    def bl_swarms(p):
        s = p.fam[(p.fam.region == "BL") & (p.fam.type == "swarm")]
        return (f"{len(s)} swarms; median duration (90%) {24 * s.duration90_days.median():.1f} h; "
                f"median extent {s.extent_km.median():.1f} km")
    add("Results (clustering)", "Blanco swarms: duration and extent", bl_swarms, "families_zbz.csv, region BL, type swarm",
        "median 4 hours, 4 km")
    add("Results (clustering); Discussion", "swarm share on the West Blanco, Surveyor and Cascadia Depression strands",
        lambda p: rng([p.fc.pct_swarm.get(k, np.nan) for k in STRANDS], "{:.0f}") + "% (" +
                  ", ".join(f"{k} {p.fc.pct_swarm.get(k, np.nan):.1f}" for k in STRANDS) + ")",
        "faults_v0_cluster_counts.csv pct_swarm (% of events above Mc on the strand); the four strands of the "
        "stored 33-45%", "33-45%")

    def bl_top(p):
        b = p.fc[(p.fc.region == "BL") & (p.fc.n_analysis >= 30)].sort_values("pct_swarm", ascending=False).head(5)
        return "; ".join(f"{i} {r.pct_swarm:.1f}" for i, r in b.iterrows())
    add("Results (clustering)", "check: the five Blanco strands with the highest swarm share (>= 30 events above Mc)",
        bl_top, "faults_v0_cluster_counts.csv", "")

    def six(p):
        v = p.val.set_index("label")
        out = []
        for lab in SIX:
            r = v.loc[lab]
            f = p.fam[p.fam.family_id == r.family_id]
            fb = float(f.frac_before_largest.iat[0]) if len(f) else np.nan
            out.append(f"{lab.split(' Mw')[0]} {r.type} ({r.role}, {100 * fb:.0f}% before)")
        return "; ".join(out)
    add("Results (clustering); Text S6", "six large earthquakes: family type, role of the large event, share of "
        "events before the largest", six, "validation_sequences.csv + families_zbz.csv", "all mainshock-aftershock, mainshock first")

    def erupt(p):
        r = p.val.set_index("label").loc["2015-04-24 Axial Seamount eruption onset"]
        return f"family {r.family_id}: {r.type}, {r.n} events above Mc ({r.n_all} with the events below Mc)"
    add("Results (clustering); Text S6", "2015 Axial eruption family", erupt, "validation_sequences.csv", "one 632-event swarm")

    def axial(p):
        ez = p.ev
        t0, t1 = pd.Timestamp("2014-10-01", tz="UTC"), pd.Timestamp("2015-07-01", tz="UTC")
        ax = ez[(ez.region == "AX") & ez.in_analysis & ez.time.between(t0, t1)]
        wk = ax.set_index("time").resample("7D").size()
        nov = wk[(wk.index >= "2014-11-01") & (wk.index < "2014-12-01")]
        mar = wk[(wk.index >= "2015-03-08") & (wk.index < "2015-03-29")]
        cal = ax.set_index("time").resample("W-SUN").size()
        post = wk[wk.index > wk.idxmax()].head(5).tolist()
        sw = p.fam[(p.fam.region == "AX") & (p.fam.type == "swarm")]
        mo = sw.groupby(sw.start.dt.strftime("%Y-%m")).size()
        return (f"7-day bins: Nov 2014 {nov.min()}-{nov.max()} (mean {nov.mean():.0f}), mid-Mar 2015 mean {mar.mean():.0f} "
                f"(max {wk[(wk.index >= '2015-03-01') & (wk.index < '2015-04-01')].max()}); eruption week "
                f"{cal.max():,} (calendar week Mon-Sun, week of {cal.idxmax():%d %b}) or {wk.max():,} (7-day bin of "
                f"Fig. 6A); next 7-day bins {post}; swarm families per month Nov 2014 {mo.get('2014-11', 0)}, "
                f"Mar 2015 {mo.get('2015-03', 0)}, Apr 2015 {mo.get('2015-04', 0)}")
    add("Results (short-lived), Fig. 6A", "Axial events above Mc per week; swarm families per month", axial,
        "events_zbz.csv.gz in_analysis, region AX; families_zbz.csv swarms by start month",
        "about 60 (Nov 2014) -> about 300 (mid-Mar 2015); 1 -> 16 (Mar) and 13 (Apr) swarms; eruption week 1,021; "
        "below 15 per week within four weeks; almost none above Mc 0.75")
    add("Results (short-lived)", "Axial clustering Mc", lambda p: f"{p.reg.Mc['AX']:.2f}", "regions_zbz.csv", "Mc 0.75")

    def wblanco(p):
        f = p.fam[(p.fam.region == "BL") & (p.fam.start.dt.strftime("%Y-%m") == "2015-06")].sort_values("n_all").iloc[-1]
        e = p.ev[p.ev.family_id == f.family_id].sort_values("time")
        x = (e.lon - e.lon.mean()) * 111.2 * np.cos(np.radians(e.lat.mean()))
        y = (e.lat - e.lat.mean()) * 111.2
        d = (x * np.sin(np.radians(111.0)) + y * np.cos(np.radians(111.0))).to_numpy()
        h = ((e.time - e.time.iloc[0]).dt.total_seconds() / 3600).to_numpy()
        span = lambda k: float(np.subtract(*np.percentile(d[k], [95, 5])))
        return (f"family {int(f.family_id)}: {int(f.n_all)} events ({int(f.n)} above Mc), {f.duration_days:.0f} days "
                f"({f.start:%d %b} to {pd.Timestamp(f.end):%d %b %Y}); along-strike extent (5-95%): 6 h {span(h <= 6):.0f} km, "
                f"1 day {span(h <= 24):.0f} km, whole family {span(h >= 0):.0f} km, after day 1 {span(h > 24):.0f} km; "
                f"Fig. 6B: {(h <= 288).sum()} events in 12 days, M {e.M.min():.1f}-{e.M.max():.2f}, "
                f"{int((e.M[h <= 288] < 1.5).sum())} of them below the colour-bar floor M 1.5")
    add("Results (short-lived), Fig. 6B", "West Blanco swarm of 1 June 2015", wblanco,
        "largest BL family starting in June 2015 (as Fig. 6B); distance along 111 deg as Fig. 6B, 5-95% range of all "
        "family events (reproduces the text's 23 and 41 km)", "23 km in 6 h, 41 km in 1 day, the length the family kept "
        "for the next 64 days (755 events)")

    def mendocino(p):
        cat = pd.read_csv(vp.main_data("catalog_v4", "Cascadia_relocated_catalog_v4.csv"), skipinitialspace=True)
        cat.columns = [c.strip() for c in cat.columns]
        cat["time"] = pd.to_datetime(cat.datetime.str.strip(), utc=True)
        m = pd.read_csv(p.mag, usecols=["event_id", "M"]).set_index("event_id").M
        cat["M"] = cat.ID.map(m)
        q = cat[(cat.Num_p_picks >= 4) & (cat.Num_s_picks >= 4) & cat.latitude.between(40.25, 40.62)
                & cat.longitude.between(-127.6, -124.9)
                & cat.time.between(pd.Timestamp("2012-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC"))]
        st = pd.read_csv(vp.STATIONS)                 # as v4_figures._obs_near: 7-day bins, OBS within 120 km
        for c in ("epoch_start", "epoch_end"):
            st[c] = pd.to_datetime(st[c], errors="coerce", utc=True).dt.tz_localize(None)
        st["epoch_end"] = st.epoch_end.fillna(pd.Timestamp("2030-01-01"))
        st = st[st.seafloor.astype(str).str.lower().eq("true")]
        st = st[np.hypot((st.lon + 126.2) * 111.2 * np.cos(np.radians(40.45)), (st.lat - 40.45) * 111.2) < 120]
        days = pd.date_range("2012-01-01", "2017-01-01", freq="7D")
        nobs = np.array([((st.epoch_start <= t) & (st.epoch_end >= t)).sum() for t in days])
        i = np.clip(np.searchsorted(days.values, q.time.dt.tz_localize(None).values, side="right") - 1, 0, len(days) - 1)
        on = nobs[i] > 0
        far = q.longitude.to_numpy() < -125.5
        off = q[~on & far]
        o16 = off[off.time.dt.year < 2016]
        return (f"QC events west of 125.5W: {int((on & far).sum()):,} in weeks with an OBS within 120 km, {len(off)} without "
                f"({len(o16)} in 2012-2015, M {o16.M.min():.1f}-{o16.M.max():.1f}; 2016: "
                f"{', '.join(f'M {v:.1f}' for v in off[off.time.dt.year == 2016].M)})")
    add("Results (short-lived), Fig. 6D", "western Mendocino transform: events away from the coast with and without OBS",
        mendocino, "QC events of Fig. 6D; weeks shaded as in Fig. 6D (an ocean-bottom station within 120 km of 40.45N "
        "126.2W)", "only events near the coast or above M 4 remain")

    sh_mean = lambda p: p.sh.groupby("region").mean(numeric_only=True)
    add("Text S6", "bimodality (Ashman D): real; shuffled (mean of 3 shuffles)",
        lambda p: f"{rng(p.reg.ashman_D)}; {rng(sh_mean(p).ashman_D)}", "regions_zbz.csv, shuffle_test.csv",
        "1.6-3.4 to 0.3-1.6")

    def shuf(p):
        s = sh_mean(p)
        o = s.drop(index="AX")
        return (f"linked below the real threshold: {rng(100 * o.frac_linked, '{:.0f}')}% (AX {100 * s.frac_linked['AX']:.0f}%); "
                f"below eta0 without the cap: {rng(100 * o.frac_below_eta0, '{:.0f}')}% (AX {100 * s.frac_below_eta0['AX']:.0f}%)")
    add("Text S6", "shuffled events that pass the real threshold (mean of 3 shuffles; regions except Axial)", shuf,
        "shuffle_test.csv frac_linked (threshold and distance cap) and frac_below_eta0", "3-17% (34% at Axial)")

    def sens(p):
        s = p.sens[p.sens.variant.isin(SENS)]
        if not len(s):
            return "no variants in sensitivity_zbz.csv"
        a = 100 * s.agree_bg_vs_clustered_with_main
        m = p.sens[p.sens.variant == "main"].set_index("region").pct_background
        c = p.sens[p.sens.variant == "eta0_shuf5"].set_index("region").pct_background
        dbg = (c - m).dropna()
        by = s.groupby("variant").agree_bg_vs_clustered_with_main.agg(["min", "max"]).mul(100).round(0)
        return (f"{a.min():.0f}-{a.max():.0f}% keep the label ("
                + ", ".join(f"{v} {r['min']:.0f}-{r['max']:.0f}" for v, r in by.iterrows())
                + f"); conservative threshold: background +{dbg.min():.0f} to +{dbg.max():.0f} points")
    add("Text S6", "sensitivity: label agreement under the variants; conservative threshold", sens,
        "sensitivity_zbz.csv (agree_bg_vs_clustered_with_main; eta0_shuf5 minus main pct_background)",
        "84-99%; +1 to +32 points")

    # ---- Table S4: largest Route A magnitude on two Gorda surfaces
    def s4(p):
        mem = pd.read_csv(REPO / "data/faults/v0/fault_surfaces_v0_members.csv")
        m = pd.read_csv(p.mag, usecols=["event_id", "otime", "M", "M_type"])
        x = mem.merge(m, on="event_id")
        out = []
        for sid in ("GO004", "GO011"):
            g = x[x.surface_id == sid].sort_values("M", ascending=False)
            r = g.iloc[0]
            out.append(f"{sid} {r.M_type} {r.M:.2f} (event {int(r.event_id)}, {str(r.otime)[:10]})")
        return "; ".join(out)
    add("Table S4 (fault_surfaces_v0_literature.csv)", "largest Route A M on GO004 and GO011", s4,
        "fault_surfaces_v0_members.csv joined with the Route A file",
        "GO004 'catalog maximum M5.6 on 2015-04-24'; GO011 'catalog maximum M4.1 on 2014-10-03' (catalog magnitude column)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
FIGS = [  # (label, old path, new path) relative to a products folder
    ("Fig. 1 (A: symbol size by M)", FIG / "fig1_overview.png", FIG / "fig1_overview.png"),
    ("Fig. 2a (= Fig. S3)", FIG / "fig2a_mc_three_methods.png", FIG / "fig2a_mc_three_methods.png"),
    ("Fig. 2b (= Fig. S5)", FIG / "fig2b_mc_periods.png", FIG / "fig2b_mc_periods.png"),
    ("Fig. 3 (B-D: symbol size by M)", FIG / "fig3_fault_surfaces.png", FIG / "fig3_fault_surfaces.png"),
    ("Fig. 5 (clustering)", FIG / "fig5_clustering.png", FIG / "fig5_clustering.png"),
    ("Fig. 6 (B: colour and size by M)", FIG / "fig6_short_lived.png", FIG / "fig6_short_lived.png"),
    ("Mendocino perspective view", FIG / "perspective_mendocino_fault.png", FIG / "perspective_mendocino_fault.png"),
    ("Fig. S3 (mc_maps fig1)", MCM / "fig1_mc_three_methods.png", MCM / "fig1_mc_three_methods.png"),
    ("Fig. S4 (mc_maps fig2)", MCM / "fig2_sigma_b.png", MCM / "fig2_sigma_b.png"),
    ("Fig. S5 (mc_maps fig3)", MCM / "fig3_mc_periods.png", MCM / "fig3_mc_periods.png"),
    ("Fig. S6 (mc_maps fig4)", MCM / "fig4_regional_fmd.png", MCM / "fig4_regional_fmd.png"),
    ("Fig. S7 (mc_maps fig5)", MCM / "fig5_qc_minus_all.png", MCM / "fig5_qc_minus_all.png"),
    ("Clustering report: fig_TR_density", ZBZ / "fig_TR_density.png", ZBZ / "fig_TR_density.png"),
    ("Clustering report: fig_sensitivity", ZBZ / "fig_sensitivity.png", ZBZ / "fig_sensitivity.png"),
]


def _img(path, width=1000, q=70):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    s = width / im.width
    im = im.resize((width, max(1, int(im.height * s))), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _diff(a, b, width=900):
    """Changed pixels (any channel > 8 of 255) in red over a faded copy of the new figure; share of changed pixels."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    A = np.asarray(Image.open(a).convert("RGB")).astype(np.int16)
    B = np.asarray(Image.open(b).convert("RGB")).astype(np.int16)
    if A.shape != B.shape:
        h, w = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
        A, B = A[:h, :w], B[:h, :w]
    d = np.abs(A - B).max(axis=2) > 8
    base = (255 - 0.25 * (255 - B.mean(axis=2)))[..., None].repeat(3, axis=2)
    base[d] = (213, 94, 0)
    im = Image.fromarray(base.astype(np.uint8))
    s = width / im.width
    im = im.resize((width, max(1, int(im.height * s))), Image.BOX)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=60, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), float(d.mean())


def _md(text):
    """Minimal Markdown (headings, nested bullets, bold, code) to HTML for the bullet file."""
    out, depth = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        m = re.match(r"^(#+)\s+(.*)", line)
        if m:
            out.append("</ul>" * depth)
            depth = 0
            out.append(f"<h{len(m.group(1)) + 1}>{_inline(m.group(2))}</h{len(m.group(1)) + 1}>")
            continue
        m = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if m:
            lvl = len(m.group(1)) // 2 + 1
            while depth < lvl:
                out.append("<ul>")
                depth += 1
            while depth > lvl:
                out.append("</ul>")
                depth -= 1
            out.append(f"<li>{_inline(m.group(2))}</li>")
        else:
            out.append("</ul>" * depth)
            depth = 0
            out.append(f"<p>{_inline(line)}</p>")
    out.append("</ul>" * depth)
    return "\n".join(out)


def _inline(s):
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


CSS = """
:root{--bg:#fff;--ink:#1d1d1f;--muted:#5f6368;--line:#d9dce1;--card:#f6f7f9;--accent:#b5520a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--ink:#e8e8e8;--muted:#a7abb1;
 --line:#34373c;--card:#212327;--accent:#f0a35e}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
 margin:0;padding:24px 16px 64px}
main{max-width:1280px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 8px;border-top:1px solid var(--line);padding-top:16px}
h3{font-size:16px;margin:20px 0 6px}
.muted{color:var(--muted)}code{font-size:13px;background:var(--card);padding:0 3px;border-radius:3px}
p,li,td,code{overflow-wrap:anywhere}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:8px 0 4px}
.pair figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}
.pair figcaption{font-size:13px;color:var(--muted);margin-bottom:4px}
.pair img{width:100%;height:auto;display:block;background:#fff}
details{margin:4px 0 12px}summary{cursor:pointer;color:var(--accent);font-size:14px}
details img{width:100%;max-width:1100px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
th{background:var(--card)}
.wrap{overflow-x:auto}
ul{margin:4px 0 4px 20px;padding:0}li{margin:2px 0}
@media (max-width:760px){.pair{grid-template-columns:1fr}}
"""


def report(old: Path, new: Path, control: Path | None, nums: pd.DataFrame, bullets: str, out: Path, notes: list[str]):
    h = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>Converged magnitudes in the paper</title>", f"<style>{CSS}</style></head><body><main>",
         "<h1>Converged magnitudes in the paper</h1>",
         "<p class='muted'>Decision D-B091 (PI, 2026-11-25): Mc, b and every paper figure or number that uses offshore ML "
         "are updated with the converged Route A magnitudes (exact phase19 solver, PR #45). Left: the figure on main "
         "(5d6658bf, stored magnitudes). Right: the same script with the converged magnitudes. Under each pair, the "
         "pixels that the magnitudes alone change, in orange.</p>"]
    h += [f"<p>{_inline(x)}</p>" for x in notes]
    h.append(_md(bullets))
    h.append("<h2>Figures, stored and converged</h2>")
    for lab, po, pn in FIGS:
        a, b = old / po, new / pn
        if not (a.exists() and b.exists()):
            h.append(f"<h3>{html.escape(lab)}</h3><p class='muted'>missing: {html.escape(str(a if not a.exists() else b))}</p>")
            continue
        c = control / pn if control is not None else None
        if c is not None and c.exists():          # orange = what the magnitudes alone change (control -> converged)
            uri, frac = _diff(c, b)
            _, fc = _diff(a, c)
            txt = (f"{100 * frac:.2f}% of pixels change from the control (stored magnitudes, today's scripts and "
                   f"inputs) to the converged figure (orange below). The control differs from the figure on main by "
                   f"{100 * fc:.2f}% of pixels" + (" (identical)." if fc == 0 else "."))
        else:
            uri, frac = _diff(a, b)
            txt = f"{100 * frac:.2f}% of pixels changed from main (orange below)."
        h.append(f"<h3>{html.escape(lab)}</h3><p class='muted'><code>{html.escape(str(pn))}</code>: {html.escape(txt)}</p>")
        h.append(f"<div class='pair'><figure><figcaption>stored (main)</figcaption><img src='{_img(a)}' alt='{html.escape(lab)}, "
                 f"stored'></figure><figure><figcaption>converged</figcaption><img src='{_img(b)}' alt='"
                 f"{html.escape(lab)}, converged'></figure></div>")
        h.append(f"<details><summary>changed pixels</summary><img src='{uri}' alt='changed pixels, {html.escape(lab)}'>"
                 "</details>")
    h.append("<h2>Numbers, stored and converged</h2>")
    h.append("<p class='muted'>Recomputed from the products of the paper's scripts (stored products: main 5d6658bf; "
             "converged: this branch). 'text' is what the manuscript says now.</p>")
    h.append("<div class='wrap'><table><tr><th>where</th><th>item</th><th>text</th><th>stored</th><th>converged</th>"
             "<th>how</th></tr>")
    for r in nums.itertuples():
        h.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (r.section, r.item, r.text, r.stored,
                                                                            r.converged, r.how)) + "</tr>")
    h.append("</table></div>")
    for name, f in (("Table S6 (mc_regional.csv), Mc_best by region, whole period", MCM / "mc_regional.csv"),):
        try:
            o = pd.read_csv(old / f)
            n = pd.read_csv(new / f)
            key = ["region", "period", "variant"]
            m = o.merge(n, on=key, suffixes=("_stored", "_conv"))
            m = m[(m.period == "all") & (m.variant == "all")]
            cols = [c for c in ("n", "mc_maxc", "mc_mbs", "mc_gft90", "mc_gft95", "mc_best", "b", "sigma_boot")
                    if f"{c}_stored" in m]
            h.append(f"<h3>{html.escape(name)}</h3><div class='wrap'><table><tr><th>region</th>" +
                     "".join(f"<th>{c} stored</th><th>{c} conv.</th>" for c in cols) + "</tr>")
            for r in m.itertuples():
                h.append(f"<tr><td>{r.region}</td>" + "".join(
                    f"<td>{getattr(r, c + '_stored'):.2f}</td><td>{getattr(r, c + '_conv'):.2f}</td>" for c in cols) + "</tr>")
            h.append("</table></div>")
        except Exception as ex:
            h.append(f"<p class='muted'>{html.escape(name)}: {html.escape(str(ex))}</p>")
    try:
        o = pd.read_csv(old / ZBZ / "regions_zbz.csv").set_index("region")
        n = pd.read_csv(new / ZBZ / "regions_zbz.csv").set_index("region")
        cols = ["n_analysis", "Mc", "b", "log_eta0", "ashman_D", "pct_background", "pct_MA", "pct_swarm", "pct_multiplet",
                "n_families", "n_swarm"]
        h.append("<h3>Table S5 (regions_zbz.csv)</h3><div class='wrap'><table><tr><th>region</th>" +
                 "".join(f"<th>{c} stored</th><th>{c} conv.</th>" for c in cols) + "</tr>")
        for r in n.index:
            h.append(f"<tr><td>{r}</td>" + "".join(f"<td>{o.loc[r, c]:g}</td><td>{n.loc[r, c]:g}</td>" for c in cols) + "</tr>")
        h.append("</table></div>")
    except Exception as ex:
        h.append(f"<p class='muted'>Table S5: {html.escape(str(ex))}</p>")
    h.append("</main></body></html>")
    out.write_text("\n".join(h))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["numbers", "report", "all"])
    ap.add_argument("--old", type=Path, default=PAPER_DEFAULT / "stored", help="products made with the stored magnitudes")
    ap.add_argument("--new", type=Path, default=REPO, help="products made with the converged magnitudes")
    ap.add_argument("--control", type=Path, default=None,
                    help="products remade with the stored magnitudes and today's scripts and inputs, same layout as --old "
                         "(separates the effect of the magnitudes from later script or catalog changes)")
    ap.add_argument("--note", action="append", default=[], help="paragraph added under the report title")
    a = ap.parse_args()
    nums_f = OUTD / "paper_numbers.csv"
    if a.stage in ("numbers", "all"):
        o = Products(a.old, mag_paths.v41("stored"))
        n = Products(a.new, mag_paths.v41("converged"))
        nums = numbers(o, n)
        nums.to_csv(nums_f, index=False)
        pd.set_option("display.width", 250, "display.max_colwidth", 400)
        for r in nums.itertuples():
            print(f"- {r.section} | {r.item}\n    text: {r.text}\n    stored: {r.stored}\n    converged: {r.converged}")
        print(f"wrote {nums_f}")
    if a.stage in ("report", "all"):
        nums = pd.read_csv(nums_f)
        bfile = OUTD / "manuscript_bullets.md"
        bullets = bfile.read_text() if bfile.exists() else "(manuscript_bullets.md not written yet)"
        report(a.old, a.new, a.control, nums, bullets, OUTD / "paper_switch_report.html", a.note)


if __name__ == "__main__":
    main()
