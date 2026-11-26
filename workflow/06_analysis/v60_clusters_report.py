"""Figures and self-contained HTML report for the catalog v6.0 clustering (v51_clusters_zbz.py --catalog v6.0,
v51_swarms.py --catalog v6.0), with the comparison against v5.1 (stored magnitudes), v5.1 locations with the
converged magnitudes, and v4.1 (converged magnitudes).

The text of the v5.1 report is specific to v5.1 against v4, so this is a separate report script; the figure
functions of v51_clusters_report.py are reused. Reads the v6.0 folder on /wd1 (OUT = BIG), the comparison runs
(v51_clusters_zbz.REFS) and the independent check in ../verify_clusters/, and writes fig_*.png and
report_v60_clusters.html (figures embedded) in the v6.0 folder. Run after the zbz steps, the swarm scan and the
verifier:
    python v60_clusters_report.py [--no-map]
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v51_clusters_report as R  # noqa: E402  (configures Z for v5.1 on import; reconfigured below)
import v51_clusters_zbz as Z  # noqa: E402
from v51_clusters_report import TYPE_COLORS, CSS, tab, img, tb, use, FULL, INK, MUTED  # noqa: E402
from v51_clusters_zbz import REGION_NAMES, REG_ORDER  # noqa: E402

TYPES = ["mainshock-aftershock", "swarm", "multiplet"]
VER = Path("/wd1/mdenolle_data/catalog_v6/v6.0_analysis/verify_clusters")
# the four runs of the magnitude x location decomposition: key -> label
RUNS = {"A": "v5.1 locations, stored M (v5.1 run)", "B": "v5.1 locations, converged M",
        "C": "v6.0 locations, stored M (variant storedM)", "D": "v6.0 locations, converged M (v6.0 run)"}


def esc(x) -> str:
    return html.escape(str(x))


def fnum(x, k=0):
    return f"{x:,.{k}f}"


# ---------------------------------------------------------------- loading
def read_events(path, cols=None):
    return pd.read_csv(path, usecols=(lambda c: c in cols) if cols else None, low_memory=False)


def load():
    O = Z.OUT
    ev = read_events(Z.BIG / Z.EVF)
    ev["time"] = pd.to_datetime(ev.time, format="ISO8601", utc=True)
    ev["excluded"] = ev.excluded.fillna("")
    fam = pd.read_csv(O / Z.FAMF, parse_dates=["start", "end"])
    d = dict(ev=ev, fam=fam)
    for k in ["regions_zbz", "thresholds_zbz", "completeness_b_df", "sensitivity_zbz", "shuffle_test",
              "validation_sequences", "volcano_families", "volcano_local_zbz", "setting_stats",
              "faults_v1_cluster_counts", "null_test_summary", "tierD_by_region", "mc_edge_check",
              "depth_swarm_vs_ma_families", "review_sets_in_clusters", "faults_v1_daytime_surfaces_events",
              "v51_families_status", "v60_families_status", "v51Mc_families_status", "v60_families_status_vs_v51Mc",
              "v4_families_status", "v60_families_status_vs_v4", "dbscan_clusters_v60", "dbscan_v51_status",
              "dbscan_v60_status", "dbscan_v4_status", "dbscan_v60_status_vs_v4", "v51_dissolved_members_fate"]:
        f = O / f"{k}.csv"
        d[k] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    d["run_params"] = json.loads((O / "run_params.json").read_text())
    d["input_commits"] = json.loads((O / "input_commits.json").read_text())
    # comparison runs
    for key, R_ in Z.REFS.items():
        dd = Path(R_["dir"])
        d[f"fam_{key}"] = pd.read_csv(dd / R_["families"])
        for k in ["regions_zbz", "thresholds_zbz", "completeness_b_df", "validation_sequences", "setting_stats"]:
            f = dd / f"{k}.csv"
            d[f"{k}_{key}"] = pd.read_csv(f) if f.exists() else pd.DataFrame()
        src = Path(R_.get("big", dd)) / R_["events"]
        d[f"ev_{key}"] = read_events(src if src.exists() else dd / R_["events"],
                                     ["event_id", "region", "in_analysis", "M", "family_id", "role", "linked",
                                      "log_eta", "lat", "lon", "depth", "excluded", "cls"])
    vd = Z.BIG / "variants"
    d["ev_storedM"] = pd.read_csv(vd / "events_storedM.csv.gz")
    d["fam_storedM"] = pd.read_csv(vd / "families_storedM.csv")
    d["comp_storedM"] = pd.read_csv(vd / "completeness_storedM.csv")
    d["sens_storedM"] = d["sensitivity_zbz"][d["sensitivity_zbz"].variant == "storedM"]
    cat = pd.read_csv(Z.CAT, usecols=["ID", "M", "magnitude_source", "shift_h_v51_km", "shift_z_v51_km", "region",
                                      "event_flag", "loc_source"], low_memory=False)
    d["cat"] = cat.rename(columns={"ID": "event_id"})
    d["verify_log"] = (VER / "v02_zbz_independent_v60.log").read_text() if (VER / "v02_zbz_independent_v60.log"
                                                                            ).exists() else ""
    for k in ["v02_family_types", "v03_spotcheck", "v03_spotcheck_notes"]:
        f = VER / f"{k}.csv"
        d[k] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return d


# ---------------------------------------------------------------- derived tables
def totals(reg: pd.DataFrame) -> dict:
    """Whole-catalog numbers from a per-region summary table (regions_zbz / variant summary)."""
    n = reg.n_analysis.sum()
    return dict(n_analysis=int(n), pct_background=float((reg.pct_background * reg.n_analysis).sum() / n),
                n_families=int(reg.n_families.sum()), n_MA=int(reg.n_MA.sum()), n_swarm=int(reg.n_swarm.sum()),
                n_multiplet=int(reg.n_multiplet.sum()), n_MA_ge5=int(reg.n_MA_ge5.sum()))


def decomposition(d) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabs = {"A": d["regions_zbz_v51"], "B": d["regions_zbz_v51Mc"], "C": d["sens_storedM"], "D": d["regions_zbz"]}
    rows, tot = [], []
    for k, t in tabs.items():
        t = t.set_index("region").reindex(REG_ORDER).reset_index()
        for _, r in t.iterrows():
            rows.append(dict(run=k, region=r.region, n_analysis=int(r.n_analysis), Mc=r.Mc, b=r.b,
                             log_eta0=r.log_eta0, ashman_D=r.ashman_D, pct_background=r.pct_background,
                             n_families=int(r.n_families), n_MA=int(r.n_MA), n_swarm=int(r.n_swarm),
                             n_multiplet=int(r.n_multiplet)))
        tot.append(dict(run=k, label=RUNS[k], **totals(t)))
    return pd.DataFrame(rows), pd.DataFrame(tot)


def members(ev):
    return ev[ev.family_id > 0][["event_id", "family_id"]]


def match(ev_ref, ev_new, ref, new):
    m4 = members(ev_ref).rename(columns={"family_id": "f4"})
    m5 = members(ev_new).rename(columns={"family_id": "f5"})
    return Z.match_families(m4, m5, None, None, ref=ref, new=new)


def match_matrix(d) -> tuple[pd.DataFrame, dict]:
    """Status counts of the family matching between the runs of the decomposition (both directions)."""
    E = {"A": d["ev_v51"], "B": d["ev_v51Mc"], "C": d["ev_storedM"], "D": d["ev"], "v4.1": d["ev_v4"]}
    pairs = [("A", "B", "magnitude change at v5.1 locations"), ("C", "D", "magnitude change at v6.0 locations"),
             ("A", "C", "location change with stored M"), ("B", "D", "location change with converged M"),
             ("A", "D", "v5.1 -> v6.0 (both changes)"), ("v4.1", "D", "v4.1 (converged M) -> v6.0")]
    rows, res = [], {}
    order_old = ["kept", "grown", "split", "merged", "changed", "dissolved"]
    order_new = ["kept", "grown", "split_part", "merged", "changed", "new"]
    for a, b, lab in pairs:
        _, r4, r5 = match(E[a], E[b], "r", "n")
        res[(a, b)] = (r4, r5)
        s4, s5 = r4.status.value_counts(), r5.status.value_counts()
        rows.append(dict(pair=f"{a} -> {b}", what=lab, n_old=len(r4), n_new=len(r5),
                         **{f"old_{k}": int(s4.get(k, 0)) for k in order_old},
                         **{f"new_{k}": int(s5.get(k, 0)) for k in order_new}))
    return pd.DataFrame(rows), res


def cause_of_v51_changes(d, res) -> pd.DataFrame:
    """For each v5.1 family that is not kept in v6.0: its status at v5.1 locations with the converged M (A->B) and with
    the v6.0 locations at stored M (A->C), to say whether the magnitude or the location change produced it."""
    s = d["v51_families_status"].copy()
    ab = res[("A", "B")][0].set_index("f4").status
    ac = res[("A", "C")][0].set_index("f4").status
    s["status_magnitude_only"] = s.f4.map(ab)
    s["status_location_only"] = s.f4.map(ac)
    ch = s[s.status != "kept"].copy()

    def cause(r):
        m, l_ = r.status_magnitude_only, r.status_location_only
        if m == r.status and l_ != r.status:
            return "magnitude"
        if l_ == r.status and m != r.status:
            return "location"
        if m == r.status and l_ == r.status:
            return "either alone"
        return "only both together"
    ch["cause"] = ch.apply(cause, axis=1)
    return ch


# ---------------------------------------------------------------- figures
def fig_eta(d, out):
    ev, t = d["ev"], d["thresholds_zbz"].set_index("region")
    t5 = d["thresholds_zbz_v51"].set_index("region")
    tb_ = d["thresholds_zbz_v51Mc"].set_index("region")
    e5 = d["ev_v51"]
    sh = np.load(Z.OUT / "shuffle_eta_seed0.npz")
    fig, axs = plt.subplots(3, 3, figsize=(FULL, FULL * 0.7), sharex=True)
    bins = np.arange(-11, 1.01, 0.2)
    for ax, r in zip(axs.flat, REG_ORDER):
        le = ev[ev.in_analysis & (ev.region == r)].log_eta.dropna().values
        ax.hist(le, bins=bins, density=True, color="#BBBBBB", label="v6.0")
        le5 = e5[e5.in_analysis.astype(bool) & (e5.region == r)].log_eta.dropna().values
        ax.hist(le5, bins=bins, density=True, histtype="step", color=INK, lw=0.6, label="v5.1 (stored M)")
        k = sh["region"] == r
        ax.hist(sh["log_eta"][k][np.isfinite(sh["log_eta"][k])], bins=bins, density=True, histtype="step",
                color="#0072B2", lw=0.8, label="v6.0 time-shuffled")
        ax.axvline(t.loc[r, "log_eta0"], color="#D55E00", lw=0.9, label="eta0 v6.0")
        ax.axvline(t5.loc[r, "log_eta0"], color=INK, lw=0.7, ls="--", label="eta0 v5.1 (stored M)")
        ax.axvline(tb_.loc[r, "log_eta0"], color="#009E73", lw=0.7, ls=":", label="eta0 v5.1 loc, converged M")
        ax.set_title(f"{r}  D={t.loc[r, 'ashman_D']:.2f} (v5.1 {t5.loc[r, 'ashman_D']:.2f})", fontsize=7)
    axs[0, 0].legend(loc="upper left", fontsize=4.8)
    for ax in axs[-1]:
        ax.set_xlabel("log$_{10}$ eta")
    fig.tight_layout()
    fig.savefig(out / "fig_eta_hist.png", dpi=170)
    plt.close(fig)


def fig_decomp(dec, out):
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.6))
    x = np.arange(len(REG_ORDER))
    w = 0.2
    cols = {"A": "#BBBBBB", "B": "#88AACC", "C": "#E8B37E", "D": "#D55E00"}
    for k, run in enumerate("ABCD"):
        t = dec[dec.run == run].set_index("region").reindex(REG_ORDER)
        lab = {"A": "v5.1 loc, stored M", "B": "v5.1 loc, converged M", "C": "v6.0 loc, stored M",
               "D": "v6.0 loc, converged M"}[run]
        axs[0].bar(x + (k - 1.5) * w, t.pct_background, w, color=cols[run], label=lab)
        axs[1].bar(x + (k - 1.5) * w, t.n_families, w, color=cols[run])
        axs[2].bar(x + (k - 1.5) * w, t.n_swarm, w, color=cols[run])
    for a, yl in zip(axs, ["% background (events above Mc)", "families", "swarm families"]):
        a.set_xticks(x, REG_ORDER, fontsize=6.5)
        a.set_ylabel(yl)
    axs[0].legend(fontsize=5.2, loc="upper left")
    for a, l_ in zip(axs, "abc"):
        a.set_title(l_, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_v51_compare.png", dpi=170)
    plt.close(fig)


def fig_mag(d, out):
    """Change of M (converged minus stored) per region box, with the Mc of both runs."""
    c = d["cat"]
    ev = d["ev"][["event_id", "region"]]
    s = d["ev_storedM"][["event_id", "M"]].rename(columns={"M": "M_stored"})
    j = ev.merge(c[["event_id", "M"]], on="event_id").merge(s, on="event_id")
    j = j[j.M.notna() & j.M_stored.notna()]
    comp, comp_s = d["completeness_b_df"].set_index("region"), d["comp_storedM"].set_index("region")
    fig, axs = plt.subplots(3, 3, figsize=(FULL, FULL * 0.62), sharex=True)
    for ax, r in zip(axs.flat, REG_ORDER):
        g = j[j.region == r]
        ax.hist(g.M - g.M_stored, bins=np.arange(-1.0, 0.61, 0.02), color="#888888")
        ax.set_title(f"{r}: median {np.median(g.M - g.M_stored):+.2f}; Mc {comp_s.Mc[r]:.2f} -> {comp.Mc[r]:.2f}",
                     fontsize=6.5)
        ax.set_yticks([])
    for ax in axs[-1]:
        ax.set_xlabel("M converged - M stored")
    fig.tight_layout()
    fig.savefig(out / "fig_mag_change.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- report
def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(Z.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "?"


def write_html(d, dec, dtot, mm, ch, out):
    ev, fam = d["ev"], d["fam"]
    reg, thr, comp = d["regions_zbz"], d["thresholds_zbz"], d["completeness_b_df"]
    a = ev[ev.in_analysis]
    T = dtot.set_index("run")
    nx = ev.excluded.value_counts()
    ftyp = fam.type.value_counts()
    s51, s60 = d["v51_families_status"], d["v60_families_status"]
    h = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
         "initial-scale=1'><title>v6.0 earthquake families</title>", f"<style>{R.CSS}</style></head><body>"]
    h.append("<h1>Earthquake families in catalog v6.0 (nearest-neighbour clustering)</h1>")
    h.append(f"<p class='muted'>Branch <code>analysis/v60-clusters</code> (commit {git_head()}), 2026-11-26. "
             "Private working product; not peer reviewed. Same method, parameters and code as the v5.1 clustering "
             "(PR #42; <code>v51_clusters_zbz.py</code>, now with <code>--catalog v6.0</code>), run on catalog v6.0 "
             "with its converged Route A magnitudes (D-B091).</p>")

    # ---- what changed
    A, B, C, D = (T.loc[k] for k in "ABCD")
    fs = s51.status.value_counts()
    f6 = s60.status.value_counts()
    big51 = s51[s51.n4_all >= 20]
    h.append("<h2>What changed against v5.1</h2><ul>")
    h.append(f"<li>Inputs. Locations: catalog v6.0 (NonLinLoc in model v0.4.1 with SSST station terms and "
             f"coherence relocation, D-B100) instead of v5.1. Magnitudes: the converged Route A values of D-B091 "
             f"carried in the v6.0 catalog ({fnum(d['run_params']['M_source'].get('v5.1_converged (D-B091, v5.1 hypocentres)', 0))} "
             f"from the v5.1 table, {fnum(d['run_params']['M_source'].get('v4.1_converged (D-B091, v4 hypocentres)', 0))} "
             f"from the v4.1 table) instead of the stored values used for v5.1. Median change of M (converged minus "
             f"stored) per region box: {mag_change_text(d)} (Fig. 1).</li>")
    n5 = d["ev_v51"].excluded.fillna("").value_counts()
    h.append(f"<li>Events used: {fnum(len(ev) - nx.get('no_hypocentre', 0) - nx.get('not_earthquake', 0))} located "
             f"earthquakes (v5.1: {fnum(n5.get('', 0))}); {fnum(nx.get('no_hypocentre', 0))} rows without a hypocentre "
             f"(v5.1: {fnum(n5.get('no_hypocentre', 0))}) and {fnum(nx.get('not_earthquake', 0))} flagged as not "
             f"earthquakes by the v6.0 <code>event_flag</code> (v5.1: {fnum(n5.get('not_earthquake', 0))}, from the "
             f"v5.1 magnitude table's is_earthquake). Above Mc (the analysis set): {fnum(D.n_analysis)} events "
             f"(v5.1: {fnum(A.n_analysis)}).</li>")
    h.append(f"<li>Families: {fnum(D.n_families)} in v6.0 against {fnum(A.n_families)} in v5.1 "
             f"({D.n_MA} / {D.n_swarm} / {D.n_multiplet} mainshock-aftershock / swarm / multiplet, against "
             f"{A.n_MA} / {A.n_swarm} / {A.n_multiplet}). Background (single events above Mc): "
             f"{D.pct_background:.1f} % against {A.pct_background:.1f} %.</li>")
    pa = dec[dec.run == "A"].set_index("region")
    pd_ = dec[dec.run == "D"].set_index("region")
    mc_txt = ", ".join(f"{r} {pa.Mc[r]:.2f} -> {pd_.Mc[r]:.2f}" for r in REG_ORDER)
    b_txt = ", ".join(f"{r} {pa.b[r]:.2f} -> {pd_.b[r]:.2f}" for r in REG_ORDER)
    n_txt = ", ".join(f"{r} {int(pa.n_analysis[r]):,} -> {int(pd_.n_analysis[r]):,}" for r in REG_ORDER)
    h.append(f"<li>Completeness and b, v5.1 -> v6.0 (same Mc rule: maximum curvature + 0.2, largest yearly value). "
             f"Mc: {mc_txt}. b: {b_txt}. Events above Mc: {n_txt}.</li>")
    h.append(f"<li>Of the {fnum(len(s51))} v5.1 families, {fnum(fs.get('kept', 0))} are kept in v6.0, "
             f"{fs.get('grown', 0)} grown, {fs.get('split', 0)} split, {fs.get('merged', 0)} merged into a larger "
             f"family, {fs.get('changed', 0)} changed and {fs.get('dissolved', 0)} dissolved (less than half of their "
             f"events in any v6.0 family). Of the {len(big51)} v5.1 families with 20 or more events, "
             f"{int((big51.status == 'kept').sum())} are kept, {int((big51.status == 'grown').sum())} grown, "
             f"{int((big51.status == 'split').sum())} split, {int((big51.status == 'merged').sum())} merged, "
             f"{int((big51.status == 'changed').sum())} changed and {int((big51.status == 'dissolved').sum())} "
             f"dissolved. {f6.get('new', 0)} v6.0 families have no v5.1 counterpart.</li>")
    cc = ch.cause.value_counts()
    h.append(f"<li>Cause of the {len(ch)} v5.1 families that are not kept: the same status already with the converged "
             f"magnitudes on the v5.1 locations and not with the v6.0 locations alone: {cc.get('magnitude', 0)} "
             f"(magnitude change); the reverse: {cc.get('location', 0)} (location change); either change alone gives "
             f"it: {cc.get('either alone', 0)}; only both together: {cc.get('only both together', 0)}. {mm_text(mm)}</li>")
    h.append(f"<li>Large families that changed: {big_changes_text(d)}</li>")
    h.append("<li>Unchanged: method, parameters (df 1.6, distance floor 0.5 km, link cap max(10 km, 3 L(M))), the "
             "nine region boxes, the classification rules, the 2-D epicentral distance of the main run.</li>")
    h.append("</ul>")

    # ---- caveats
    h.append("<h2>Caveats</h2><ul>")
    h.append("<li>Axial depths in v6.0 are about 0.22 km deeper than in v5.1 (median, SSST passes; D-B100, to be "
             "fixed in v6.1). The main run uses epicentral (2-D) distance, so families, parents and thresholds do not "
             "use depth. Depth enters the 3-D variants d3 and d3q, the family depth columns (depth_good_*) and the "
             "swarm-versus-mainshock depth test; read Axial depths there with this offset in mind.</li>")
    h.append("<li>The magnitudes were computed with the v5.1 hypocentres (or the v4 hypocentres for the 764 events not "
             "in the v5.1 table) and are not recomputed with the v6.0 hypocentres (catalog README).</li>")
    h.append("<li>Fault surfaces v1 were fitted on v5.1 locations. The join is by event ID with the faults piece's "
             "assignment (on_surface_strict), as for v5.1; it is not redone with the v6.0 locations (numbers in the "
             "fault section).</li>")
    h.append("<li>Variant mc_best uses Mc_best of the converged regional Mc map table "
             "(<code>/wd1/mdenolle_data/magnitude_converged/mc_maps_v41/mc_regional.csv</code>, rows variant all, "
             "period all; converged v4.1 magnitudes on v4 locations); there is no Mc map on v6.0 locations. The main "
             "run computes its Mc from the v6.0 magnitudes with the v4 rule.</li>")
    me = d["mc_edge_check"]
    mc2 = me.modal_counts_v4_float.str.split(";", expand=True).astype(int)
    gap = 100 * (mc2[0] - mc2[1]) / mc2[0]
    h.append("<li>Mc is the largest yearly maximum-curvature value. In the year that sets Mc, the two largest 0.1-unit "
             f"bins differ by {gap.min():.0f}-{gap.max():.0f} % (<code>mc_edge_check.csv</code>; WA: 2010, "
             f"{mc2[0][me.region.eq('WA')].iloc[0]} vs {mc2[1][me.region.eq('WA')].iloc[0]} events), so a small "
             f"change of the magnitudes can move Mc by a bin or two. WA shows the effect: its magnitudes changed by {mag_dm(d).groupby('region').dM.median()['WA']:+.2f} "
             f"(median) and its Mc by {pd_.Mc['WA'] - pa.Mc['WA']:+.2f} ({pa.Mc['WA']:.2f} -> {pd_.Mc['WA']:.2f}), "
             f"which changes its analysis set from {int(pa.n_analysis['WA']):,} to {int(pd_.n_analysis['WA']):,} "
             f"events.</li>")
    h.append("<li>No coherence relocation in the Mendocino region in v6.0 (D-B098): MT epicentres are the SSST "
             "locations. The waveform-review sets P1 and P2 are matched by event ID from the v5.1 review.</li>")
    h.append("<li>Migration and the sqrt(t) fits use tier A-C epicentres; the family classification does not use "
             "them. Treat fault and swarm-migration results as provisional until a cross-correlation "
             "double-difference catalog exists.</li></ul>")

    # ---- headline numbers
    h.append("<h2>Headline numbers (v6.0)</h2>")
    h.append(f"<p>{fnum(len(fam))} families: {ftyp.get('mainshock-aftershock', 0)} mainshock-aftershock "
             f"({D.n_MA_ge5} with 5 or more events above Mc), {ftyp.get('swarm', 0)} swarm, "
             f"{ftyp.get('multiplet', 0)} multiplet. {fnum(D.n_analysis)} events above Mc, {D.pct_background:.1f} % of "
             f"them background. With the events below Mc attached: {fnum(int((ev.family_id > 0).sum()))} events in "
             f"families, {fnum(int((ev.cls == 'linked_to_single').sum()))} linked to a single event above Mc.</p>")
    t = reg.merge(thr[["region", "log_eta0_shuf5", "log_eta0_kde"]], on="region")
    t["name"] = t.region.map(REGION_NAMES)
    h.append(tab(t, ["region", "name", "n_analysis", "Mc", "b", "df", "log_eta0", "log_eta0_shuf5", "ashman_D",
                     "pct_background", "pct_MA", "pct_swarm", "pct_multiplet", "n_families", "n_MA", "n_MA_ge5",
                     "n_swarm", "n_multiplet"],
                 rename={"n_analysis": "events &ge; Mc", "log_eta0": "log eta0", "log_eta0_shuf5": "log eta0 (5% shuffled)",
                         "ashman_D": "D", "pct_background": "% bg", "pct_MA": "% MA", "pct_swarm": "% swarm",
                         "pct_multiplet": "% mult", "n_families": "families", "n_MA": "MA", "n_MA_ge5": "MA n&ge;5",
                         "n_swarm": "swarms", "n_multiplet": "multiplets"}))
    h.append("<p class='muted'>Mc: v4 rule on the v6.0 magnitudes (maximum curvature of the 0.1 bins + 0.2, largest "
             "yearly value over years with 200 or more events); b: Aki maximum likelihood above Mc; df 1.6 in all "
             "regions; eta0: crossing of the two weighted components of a Gaussian mixture on log10 eta. D: Ashman "
             "separation of the mixture.</p>")
    if (out / "fig_family_map.png").exists():
        h.append(img(out / "fig_family_map.png", "family map"))
        h.append("<p class='muted'>Families with 5 or more events above Mc at their largest event (blue "
                 "mainshock-aftershock, orange swarm; symbol size grows with n); grey dots: background events above "
                 "Mc; dashed boxes: the nine regions.</p>")
    cmp_ = comp[["region", "n", "n_M", "Mc_all", "Mc_bstab", "Mc", "n_above", "b", "b_err", "df_est", "Mc_best",
                 "b_at_Mc_best"]]
    h.append("<h3>Completeness and b</h3>" + tab(cmp_, rename={"n_M": "with M", "Mc_bstab": "Mc b-stability",
                                                               "Mc_best": "Mc_best (mc_maps_v41)"}))

    # ---- decomposition
    h.append("<h2>Magnitude change against location change</h2>")
    h.append("<p>Four runs with the same code: v5.1 locations or v6.0 locations, each with the stored (pre-D-B091) or "
             "the converged magnitudes. A is the committed v5.1 run, D the v6.0 run. B was run here "
             "(<code>--catalog v5.1 --mag converged</code>, folder <code>ref_v51_Mconv/</code>); C is the v6.0 "
             "variant storedM (the v5.1 stored table, else the v4.1 stored table, on the v6.0 locations). Each run "
             "sets its own Mc, b and eta0.</p>")
    h.append(tab(dtot, ["run", "label", "n_analysis", "pct_background", "n_families", "n_MA", "n_MA_ge5", "n_swarm",
                        "n_multiplet"], rename={"n_analysis": "events &ge; Mc", "pct_background": "% bg",
                                                "n_families": "families", "n_MA": "MA", "n_MA_ge5": "MA n&ge;5",
                                                "n_swarm": "swarms", "n_multiplet": "multiplets"},
                 fmt=lambda x: f"{x:.1f}"))
    pv = dec.pivot(index="region", columns="run", values=["Mc", "b", "log_eta0", "pct_background", "n_families",
                                                         "n_swarm"]).reindex(REG_ORDER)
    pv.columns = [f"{v} {r}" for v, r in pv.columns]
    h.append(tab(pv.reset_index(), fmt=lambda x: f"{x:.2f}"))
    h.append(img(out / "fig_v51_compare.png", "decomposition by region"))
    h.append("<p>Family matching between the runs (shared event IDs, members including the attached events below "
             "Mc; a family is kept when at least half of its events are in one family of the other run and that "
             "family has at least half of its events from it; dissolved when less than half of its events are in any "
             "family of the other run). Columns old_*: status of the families of the first run; new_*: of the "
             "second run.</p>")
    h.append(tab(mm))
    h.append("<h3>v5.1 families not kept in v6.0, by cause</h3>")
    xt = pd.crosstab(ch.status, ch.cause, margins=True)
    h.append(tab(xt.reset_index()))
    cb = ch[ch.n4_all >= 20].sort_values("n4_all", ascending=False).copy()
    cb["start"] = cb.start.astype(str).str[:10]
    for c_ in ["f5_main", "shared_main"]:
        cb[c_] = cb[c_].astype("Int64")
    h.append("<p>v5.1 families with 20 or more events that are not kept (status_magnitude_only: status of the v5.1 "
             "family at v5.1 locations with the converged magnitudes; status_location_only: with the v6.0 "
             "locations and the stored magnitudes):</p>")
    h.append(tab(cb, ["f4", "region", "type", "n4_all", "Mmax", "start", "status", "f5_main", "type_v60_main",
                      "shared_main", "status_magnitude_only", "status_location_only", "cause"],
                 rename={"f4": "v5.1 family", "n4_all": "events", "f5_main": "main v6.0 family",
                         "type_v60_main": "its type", "shared_main": "shared"}))
    h.append("<h3>Fate of the members of dissolved v5.1 families</h3>")
    h.append(tab(d["v51_dissolved_members_fate"]))
    h.append(img(out / "fig_mag_change.png", "magnitude change"))
    h.append("<p class='muted'>Fig. 1. Converged minus stored magnitude of the v6.0 events with M in both, per region "
             "box, and Mc with the stored and with the converged magnitudes on the v6.0 locations.</p>")

    # ---- v4.1
    h.append("<h2>Against v4.1 (converged magnitudes)</h2>")
    r4 = d["regions_zbz_v4"].set_index("region").reindex(REG_ORDER)
    T4 = totals(r4.reset_index())
    h.append(f"<p>v4.1 clustering with the converged magnitudes (the paper run, "
             f"<code>data/catalog_v4/clusters_zbz/</code> on main): {fnum(T4['n_families'])} families "
             f"({T4['n_MA']} / {T4['n_swarm']} / {T4['n_multiplet']}), {fnum(T4['n_analysis'])} events above Mc, "
             f"{T4['pct_background']:.1f} % background.</p>")
    c4 = d["completeness_b_df_v4"].set_index("region").reindex(REG_ORDER)
    t4 = d["thresholds_zbz_v4"].set_index("region").reindex(REG_ORDER)
    rr = reg.set_index("region").reindex(REG_ORDER)
    tt = thr.set_index("region").reindex(REG_ORDER)
    q = pd.DataFrame(dict(region=REG_ORDER, Mc_v41=c4.Mc.values, Mc_v60=rr.Mc.values, b_v41=c4.b.values,
                          b_v60=rr.b.values, eta0_v41=t4.log_eta0.values, eta0_v60=tt.log_eta0.values,
                          D_v41=t4.ashman_D.values, D_v60=tt.ashman_D.values, bg_v41=r4.pct_background.values,
                          bg_v60=rr.pct_background.values, fam_v41=r4.n_families.values, fam_v60=rr.n_families.values,
                          sw_v41=r4.n_swarm.values, sw_v60=rr.n_swarm.values))
    h.append(tab(q))
    s4 = d["v4_families_status"].status.value_counts()
    s64 = d["v60_families_status_vs_v4"].status.value_counts()
    h.append(f"<p>v4.1 families in v6.0: {', '.join(f'{k} {v}' for k, v in s4.items())}. v6.0 families against v4.1: "
             f"{', '.join(f'{k} {v}' for k, v in s64.items())} (column v4_status of the family table).</p>")

    # ---- largest families
    h.append("<h2>Largest families</h2>")
    L = fam.sort_values("n_all", ascending=False).head(20).copy()
    f51 = d["fam_v51"].set_index("family_id")
    L["v51_n_all"] = L.v51_family_main.map(f51.n_all)
    L["v51_type"] = L.v51_family_main.map(f51.type)
    L["start"] = L.start.astype(str).str[:10]
    L["v51_family_main"] = L.v51_family_main.astype("Int64")
    L["v51_n_all"] = L.v51_n_all.astype("Int64")
    h.append(tab(L, ["family_id", "region", "region_v51", "type", "start", "n", "n_all", "Mmax", "duration90_days",
                     "extent_km", "v51_status", "v51_family_main", "v51_n_all", "v51_type", "v4_status"],
                 rename={"n": "n &ge; Mc", "n_all": "n all", "v51_family_main": "v5.1 family", "v51_n_all": "v5.1 n all",
                         "v51_type": "v5.1 type"}))

    # ---- validation
    h.append("<h2>Known sequences</h2>")
    v = d["validation_sequences"]
    v5 = d["validation_sequences_v51"].set_index("label")
    v = v.assign(v51_type=v.label.map(v5.type), v51_n=v.label.map(v5.n), v51_n_all=v.label.map(v5.n_all))
    h.append(tab(v, ["label", "expected", "region", "M", "role", "family_id", "type", "n", "n_all", "Mmax", "M2",
                     "duration90_days", "extent_km", "v51_type", "v51_n", "v51_n_all"]))
    h.append(img(out / "fig_examples.png", "example sequences"))

    # ---- threshold figures
    h.append("<h2>Rescaled time and distance, thresholds</h2>")
    h.append(img(out / "fig_TR_density.png", "T-R density"))
    h.append(img(out / "fig_eta_hist.png", "eta histograms"))
    st = d["shuffle_test"].groupby("region").mean(numeric_only=True).reindex(REG_ORDER).reset_index()
    st["n"] = st.n.astype(int)
    h.append("<p>Time-shuffled catalogs (3 seeds, times permuted within each region): share of events below the real "
             "eta0 and mixture separation D.</p>")
    h.append(tab(st, ["region", "n", "frac_below_eta0", "frac_linked", "ashman_D"]))
    ns = d["null_test_summary"]
    if len(ns):
        h.append("<h3>Null test of the links (real thresholds)</h3>")
        h.append(tab(ns, ["null", "region", "n_seeds", "frac_linked_mean", "real_frac_linked",
                          "link_ratio_null_over_real", "D_mean", "real_ashman_D", "n_swarm_mean", "real_n_swarm",
                          "swarm_ratio_null_over_real", "family_ratio_null_over_real"], fmt=lambda x: f"{x:.3f}"))

    # ---- sensitivity
    h.append("<h2>Sensitivity variants</h2>")
    s = d["sensitivity_zbz"]
    sv = s.groupby("variant", sort=False).apply(lambda g: pd.Series(totals(g)), include_groups=False).reset_index()
    ag = s.groupby("variant", sort=False).apply(
        lambda g: (g.agree_bg_vs_clustered_with_main * g.n_common_with_main).sum() / g.n_common_with_main.sum()
        if g.n_common_with_main.notna().any() else np.nan, include_groups=False).rename("agree_with_main")
    sv = sv.merge(ag.reset_index(), on="variant", how="left")
    h.append(tab(sv, fmt=lambda x: f"{x:.3f}" if abs(x) < 1 else f"{x:.1f}"))
    h.append("<p class='muted'>agree_with_main: share of the events in both analysis sets that have the same "
             "background / clustered label as in the main run. d3 uses every depth (the Axial depth offset of "
             "D-B100 enters it); d3q only pairs with two well-constrained depths.</p>")
    ax = s[s.region == "AX"][["variant", "n_analysis", "Mc", "log_eta0", "pct_background", "n_families", "n_swarm"]]
    h.append("<p>Axial in each variant:</p>" + tab(ax))
    h.append(img(out / "fig_sensitivity.png", "sensitivity"))

    # ---- topology, migration, depth
    h.append("<h2>Families: migration and depth</h2>")
    f10 = fam[fam.n_all >= 10]
    mig = fam[tb(fam.mig_migrating)]
    h.append(f"<p>{len(f10)} families with 10 or more events; {len(mig)} migrating by the permutation test "
             f"(radial Spearman rho >= 0.5, p_perm < 0.01, positive speed, 90 % radius at least twice the median "
             f"horizontal error).</p>")
    if len(mig):
        h.append(tab(mig.assign(start=mig.start.astype(str).str[:10]),
                     ["family_id", "region", "type", "start", "n", "n_all", "mig_n", "mig_rho", "mig_p_perm",
                      "mig_v_radial_km_day", "mig_r90_km", "h_err_med_km", "D90_m2s", "env_p", "mig_style",
                      "v51_status"]))
    h.append(img(out / "fig_migration.png", "migration"))
    dsw = d["depth_swarm_vs_ma_families"]
    if len(dsw):
        h.append("<p>Depth of swarm and mainshock-aftershock families (family medians of depth_good members, "
                 "Mann-Whitney):</p>" + tab(dsw, fmt=lambda x: f"{x:.3g}"))
    h.append(img(out / "fig_depth.png", "depth by class"))

    # ---- faults
    h.append("<h2>Fault surfaces v1</h2>")
    pf = d["faults_v1_cluster_counts"]
    c = d["cat"]
    asg_on = ev[ev.event_id.isin(pf_events(d))]
    sh = c.set_index("event_id").shift_h_v51_km.reindex(asg_on.event_id)
    h.append(f"<p>{len(pf)} surfaces hold v6.0 events joined by ID. The join uses the v5.1-location assignment: of "
             f"the {fnum(len(sh))} events on a surface (on_surface_strict), {fnum(int((sh > 2).sum()))} moved more "
             f"than 2 km horizontally from v5.1 to v6.0 (median {np.nanmedian(sh):.2f} km), so their v6.0 epicentre "
             f"may no longer be within the 2 km of the strict rule.</p>")
    h.append(tab(pf.sort_values("n_analysis", ascending=False).head(25),
                 ["surface_id", "region", "kind", "n_events_joined", "n_analysis", "pct_background", "pct_swarm",
                  "n_families", "n_swarm_families", "swarm_share_top1"]))
    h.append(img(out / "fig_faults.png", "fault surfaces"))

    # ---- DBSCAN
    h.append("<h2>Swarm scan (DBSCAN, all magnitudes)</h2>")
    db = d["dbscan_clusters_v60"]
    if len(db):
        s5, s6 = d["dbscan_v51_status"], d["dbscan_v60_status"]
        h.append(f"<p>{len(db)} space-time clusters ({int(tb(db.swarm_like).sum())} swarm-like); scan of v5.1: "
                 f"{len(s5)} clusters. v5.1 clusters in v6.0: {', '.join(f'{k} {v}' for k, v in s5.status.value_counts().items())}. "
                 f"v6.0 clusters against v5.1: {', '.join(f'{k} {v}' for k, v in s6.status.value_counts().items())}. "
                 f"The scan does not use magnitudes, so these changes come from the locations and the not-earthquake "
                 f"flags.</p>")
        g = db.groupby("region").agg(clusters=("n", "size"), events=("n", "sum"),
                                     swarm_like=("swarm_like", lambda x: int(tb(x).sum()))).reset_index()
        h.append(tab(g))

    # ---- verification
    h.append("<h2>Independent check</h2>")
    h.append("<p>The verifier script of v5.1 (<code>v02_zbz_independent.py</code>, written from the method "
             "description without the builder's code) was pointed at v6.0 and extended with a member-set comparison "
             "of every family and its own classification "
             f"(<code>{VER}/v02_zbz_independent_v60.py</code>). Its log:</p>")
    h.append(f"<pre style='font-size:11px;overflow-x:auto'>{esc(verify_excerpt(d['verify_log']))}</pre>")
    sc = d["v03_spotcheck"]
    if len(sc):
        h.append("<h3>Ten families checked by hand</h3><p>Member lists in time and space "
                 f"(<code>{VER}/v03_spotcheck_v60.py</code>): every link checked against eta0 and the distance cap, "
                 "every member's region and time, the distance from the largest event; maps and distance-time plots "
                 "looked at one by one.</p>")
        notes = d["v03_spotcheck_notes"]
        if len(notes):
            sc = sc.merge(notes, on="family_id", how="left")
        h.append(tab(sc))
        if (VER / "fig_spotcheck.png").exists():
            h.append(img(VER / "fig_spotcheck.png", "spot check"))

    # ---- atlas and files
    h.append("<h2>Files</h2><ul>")
    for fn, desc in [
        (f"{Z.BIG}/events_zbz_v60.csv.gz", "every catalog row (172,342): class, family, role, parent, eta, position; "
         "same columns as events_zbz_v51.csv.gz (region_v51 = catalog region label, identical in v5.1 and v6.0)"),
        (f"{Z.OUT}/families_zbz_v60.csv", "every family: v5.1 columns plus v51_status / v51Mc_status (matching with "
         "the v5.1 run and the v5.1-locations converged-M run); v4_status is against v4.1 with converged M"),
        ("regions_zbz.csv, thresholds_zbz.csv, completeness_b_df.csv, sensitivity_zbz.csv", "per-region tables"),
        ("v51_families_status.csv, v60_families_status.csv (+ _vs_v51Mc, _vs_v4)", "family matching"),
        ("ref_v51_Mconv/", "the v5.1-locations run with the converged magnitudes"),
        ("variants/", "the twelve sensitivity variants, including storedM"),
        (f"{VER}/", "independent check and spot checks")]:
        h.append(f"<li><code>{esc(fn)}</code>: {esc(desc)}</li>")
    h.append("</ul>")
    h.append("<h2>References</h2><ul>")
    for txt, doi in R.REFS:
        link = f" <a href='https://doi.org/{doi}'>doi:{doi}</a>" if doi else ""
        h.append(f"<li>{esc(txt)}{link}</li>")
    h.append("</ul></body></html>")
    (out / "report_v60_clusters.html").write_text("\n".join(h))


def big_changes_text(d, top: int = 25, nmin: int = 20) -> str:
    """The v6.0 families among the largest (n_all) that are not 'kept' against v5.1, with the v5.1 families of 20 or
    more events whose main v6.0 family they are."""
    fam = d["fam"].sort_values("n_all", ascending=False).head(top)
    s51 = d["v51_families_status"]
    f51 = d["fam_v51"].set_index("family_id")
    out = []
    for _, f in fam[fam.v51_status.ne("kept")].iterrows():
        parts = s51[(s51.f5_main == f.family_id) & (s51.n4_all >= nmin)].sort_values("n4_all", ascending=False)
        pt = "; ".join(f"F{int(r.f4)} ({f51.type[r.f4]}, {int(r.n4_all)} events, {r.status})" for _, r in parts.iterrows())
        out.append(f"v6.0 family F{int(f.family_id)} ({f.region}, {str(f.start)[:10]}, {f.type}, {int(f.n_all):,} events, "
                   f"{f.v51_status} against v5.1) holds the v5.1 families {pt or 'of fewer than 20 events'}")
    return ". ".join(out) + "." if out else "none among the largest families."


def mm_text(mm: pd.DataFrame) -> str:
    """One sentence on dissolved and split / merged families for each change, from the matching matrix."""
    m = mm.set_index("pair")
    g = lambda p, c: int(m.loc[p, c])  # noqa: E731
    return (f"Families dissolved by the magnitude change: {g('A -> B', 'old_dissolved')} (v5.1 locations) and "
            f"{g('C -> D', 'old_dissolved')} (v6.0 locations); by the location change: {g('A -> C', 'old_dissolved')} "
            f"(stored M) and {g('B -> D', 'old_dissolved')} (converged M). Split or merged: magnitude change "
            f"{g('A -> B', 'old_split') + g('A -> B', 'old_merged')} and {g('C -> D', 'old_split') + g('C -> D', 'old_merged')}, "
            f"location change {g('A -> C', 'old_split') + g('A -> C', 'old_merged')} and "
            f"{g('B -> D', 'old_split') + g('B -> D', 'old_merged')} (matching table below).")


def mag_dm(d) -> pd.DataFrame:
    """v6.0 events with M in the catalog and in the storedM variant: box region and the change of M."""
    ev = d["ev"][["event_id", "region"]]
    s = d["ev_storedM"][["event_id", "M"]].rename(columns={"M": "M_stored"})
    j = ev.merge(d["cat"][["event_id", "M"]], on="event_id").merge(s, on="event_id")
    j = j[j.M.notna() & j.M_stored.notna()]
    return j.assign(dM=j.M - j.M_stored)


def mag_change_text(d) -> str:
    j = mag_dm(d)
    med = j.groupby("region").dM.median().reindex(REG_ORDER)
    return ", ".join(f"{r} {v:+.2f}" for r, v in med.items())


def pf_events(d) -> set:
    """Event IDs on a fault surface (on_surface_strict) in the faults piece's assignment table."""
    asg = Z.git_csv(Z.FAULTS_REF, f"{Z.FAULTS}/fault_surfaces_v1_assignment.csv.gz",
                    usecols=["event_id", "on_surface_strict"])
    return set(asg.event_id[tb(asg.on_surface_strict)])


def verify_excerpt(text: str) -> str:
    keep = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(keep[-60:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-map", action="store_true")
    a = ap.parse_args()
    Z.configure("v6.0")
    R.OUT = Z.OUT
    out = Z.OUT
    use()
    d = load()
    df = Z.prepare(Z.load_all())
    dec, dtot = decomposition(d)
    mm, res = match_matrix(d)
    ch = cause_of_v51_changes(d, res)
    dec.to_csv(out / "decomposition_by_region.csv", index=False, float_format="%.3f")
    dtot.to_csv(out / "decomposition_totals.csv", index=False, float_format="%.3f")
    mm.to_csv(out / "family_match_matrix.csv", index=False)
    ch.to_csv(out / "v51_families_not_kept_cause.csv", index=False, float_format="%.3f")
    R.fig_tr(d, out)
    fig_eta(d, out)
    if not a.no_map:
        R.fig_map(d, out)
    fig_decomp(dec, out)
    fig_mag(d, out)
    R.fig_examples(d, out)
    R.fig_migration(d, df, out)
    R.fig_depth(d, out)
    R.fig_sens(d, out)
    R.fig_faults(d, out)
    write_html(d, dec, dtot, mm, ch, out)
    print("report written:", out / "report_v60_clusters.html")


if __name__ == "__main__":
    main()
