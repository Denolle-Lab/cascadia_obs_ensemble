"""Result-dependent sections of the Q1 report. Every number is read from the run's tables at build time (no number is
typed into the text); the wording was written after the results were seen and checked against the pre-registered
decision rules. The comparison with the D-B113 run reads its snapshot (runs/db113_axes_v1)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HOT = ["AX", "JR", "EN", "EXR", "GR", "BL", "SOV", "MSH"]
COLD = ["GP", "MFZ", "MTJ", "GS"]
NAME = {"AX": "Axial", "JR": "Juan de Fuca ridge", "EN": "Endeavour", "EXR": "Explorer ridge", "GR": "Gorda Ridge",
        "BL": "Blanco", "MFZ": "Mendocino fault zone", "SOV": "Sovanco", "JP": "Juan de Fuca plate",
        "GP": "Gorda plate", "EXP": "Explorer plate", "MTJ": "triple junction", "GS": "Gorda slab", "FA": "forearc",
        "SIF": "interface zone", "IS": "Juan de Fuca slab", "PU": "Puget", "MSH": "Mount St Helens",
        "CAS": "Cascade arc", "OC": "other onshore crust", "NAPA": "south of the triple junction"}
THE = {"JR", "GR", "MFZ", "JP", "GP", "EXP", "MTJ", "GS", "FA", "SIF", "IS", "CAS", "EXR"}
REGION = {"AX": "Axial", "MT": "Mendocino", "PU": "Puget", "EN": "Endeavour-Explorer", "BL": "Blanco", "GO": "Gorda",
          "WA": "offshore Washington", "OR": "offshore Oregon", "ON": "onshore and other"}
LABEL = {"swarm_share": "swarm share", "b": "b", "productivity": "productivity", "fam_mix": "family mix",
         "bg_frac": "background fraction", "age_myr": "plate age", "heatflow_mw": "heat flow",
         "rate_mmyr": "spreading rate", "vpvs_fam": "family Vp/Vs"}
RUNS = (["main", "v60", "mcu_count", "mcu_recluster", "mc_local", "ci_window"]
        + ["var_" + v for v in ["d3", "d3q", "df1.3", "df2.0", "df_est", "eta0_shuf5", "mc+0.3", "mc_best", "mc_year",
                                "noM_floor", "storedM", "tierABC"]])
THIN = ["mcu_count", "mcu_recluster", "mc_local", "ci_window", "cell04"]


def _p(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.4f}" if v >= 0.0001 else f"{v:.1e}"


def _p2(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) else (f"{v:.2g}" if v >= 0.001 else f"{v:.1e}")


def _r(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) else f"{v:+.{nd}f}"


def _f(v, nd=1):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) else f"{v:.{nd}f}"


def _row(T, test):
    return T.set_index("test").loc[test]


def _sec(T, unit, metric, cov, fam=None):
    k = T[(T.unit == unit) & (T.metric == metric) & (T.covariate == cov)]
    if fam:
        k = k[k.family == fam]
    return k.iloc[0] if len(k) else None


def _seg(m, nmin=100):
    """Segment rows of a metrics table with >= nmin events, indexed by segment code."""
    return m[(m.unit == "seg") & (m.n >= nmin)].set_index("unit_id")


def _range(x, codes):
    v = x.reindex(codes).dropna()
    if not len(v):
        return None
    return dict(lo=v.min(), hi=v.max(), lo_seg=v.idxmin(), hi_seg=v.idxmax(), med=v.median(), n=len(v),
                segs=list(v.index))


def _the(c):
    return ("the " if c in THE else "") + NAME.get(c, c)


def _names(codes, cap=False):
    codes = list(codes)
    n = [_the(c) for c in codes]
    t = n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]
    return t[:1].upper() + t[1:] if cap else t


def _names_n(vc):
    n = [f"{v} {NAME.get(k, k)}" for k, v in vc.items()]
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]


def _hot_cold(Q, root=None):
    rows = []
    root = root or (Q.BIG / "tables")
    for run in RUNS:
        f = root / f"metrics_{run}.csv"
        if not f.exists():
            continue
        x = _seg(pd.read_csv(f)).swarm_share
        hot, cold = x.reindex(HOT).dropna(), x.reindex(COLD).dropna()
        rows.append(dict(run=run, n_hot=len(hot), hot_median=hot.median(), hot_min=hot.min(), n_cold=len(cold),
                         cold_median=cold.median(), cold_max=cold.max(), separated=bool(hot.min() > cold.max())))
    return pd.DataFrame(rows)


def _within(t):
    wt = json.loads(t.within_type) if isinstance(t.within_type, str) else {}
    return "; ".join(f"{k} {('%+.2f' % v[0]) if v[0] is not None else 'constant'} ({v[1]} cells)" for k, v in wt.items())


# ------------------------------------------------------------------ verdict and changes
def TXT_VERDICT(d, h):
    dec = d["dec"]
    T = d["T"]
    t1, t2, t4a, t4b, t3 = (_row(T, k) for k in ("T1", "T2", "T4a", "T4b", "T3"))
    hc = dec["hot_cold"].get("mcu_recluster_n100", {})
    nr = dec.get("null_rule") or {}
    lines = [
        f"<b>Thermal control (T1, T2, T4): {h['esc'](dec['verdict_thermal'])}.</b> "
        f"Plate age: cell rho {_r(t1.rho)} (n = {int(t1.n)}, one-sided p {_p(t1.p)}, Holm {_p(t1.p_adj)}, block 95 % "
        f"interval {_r(t1.block_lo)} to {_r(t1.block_hi)}); within boundary types {_r(t4a.rho)} (Holm {_p(t4a.p_adj)}). "
        f"Heat flow: rho {_r(t2.rho)} (n = {int(t2.n)}, Holm {_p(t2.p_adj)}, block {_r(t2.block_lo)} to "
        f"{_r(t2.block_hi)}); within types {_r(t4b.rho)} (Holm {_p(t4b.p_adj)}).",
        f"Time-shuffle null rule: {nr.get('n_above', '-')} of 4 hot segments above the null 95th percentile "
        f"({'passed' if nr.get('passed') else 'failed'}). Uniform-Mc reclustering (Mc 2.15, segments with >= 100 "
        f"events): hot median {_f(hc.get('hot_median'))} %, cold median {_f(hc.get('cold_median'))} %.",
        f"<b>Fluid signal (T3, in-situ Vp/Vs): {h['esc'](dec['verdict_vpvs'])}.</b> "
        f"{int(t3.get('n_swarm', 0) or 0)} swarm and {int(t3.get('n_ma', 0) or 0)} mainshock-aftershock families "
        f"carry a valid Vp/Vs; the test needs 20 of each.",
    ]
    return '<div class="verdict">' + "<br>".join(lines) + "</div>"


def _seg_changes(d):
    """Tier A-C located events whose segment differs between the D-B113 run and this run."""
    p = d.get("prev")
    if not p or p.get("ev") is None:
        return None
    a = d["ev"][["ID", "seg"]].merge(p["ev"][["ID", "seg"]], on="ID", suffixes=("", "_v1"))
    a = a.merge(d["cat"][["ID", "tier"]], on="ID")
    ch = a[(a.seg != a.seg_v1) & a.tier.isin(["A", "B", "C"])]
    return len(ch), (ch.seg_v1 + " to " + ch.seg).value_counts()


def TXT_CHANGES(d, h):
    info = d["info"]["main"]
    out = [
        "Segment membership, first change (an input, not a test choice): the D-B113 ridge axes "
        "(plate_axes_v1.csv) replaced the PB2002 traces at 13:23-13:28 PST, around the registration. The first version "
        "of this report used them; that run is kept in runs/db113_axes_v1/.",
    ]
    sc = _seg_changes(d)
    moves = "" if sc is None else (f" {sc[0]:,} tier A-C events change segment against the D-B113 run (" + ", ".join(
        f"{k} {v:,}" for k, v in sc[1].head(6).items()) + ").")
    out.append(
        "Segment membership, second change: plate boundaries v2 (D-B117, PI 2026-11-27 19:09 PST; "
        "analysis/plate-boundaries 0904802e, plate_axes_v2_adopted.csv, md5 " + str(info.get("axes_file_md5", "-"))[:8]
        + "): straight Gorda Ridge, Mendocino and Blanco segments from the relocated seismicity and the bathymetry, the "
        "Juan de Fuca ridge on its bathymetric axis, the PB2002 triple junction kept. Every number here uses the v2 "
        "segments (segment join recomputed in q1_metrics.py; identical to the rebuilt events_segments_v61.parquet)."
        + moves + " Segments enter the cell tests only through T4 (cell type) and the hot/cold contrast; the whole chain "
        "was rerun, nulls included. The comparison with the D-B113 run is in the section Boundaries v2.")
    out += [
        "Covariate source: the D-B114 per-event table (events_external_geo_v61.parquet, workflow/12_external_geo, "
        "branch analysis/external-geo) appeared at 13:36 PST, after the registration. As registered, the decision uses "
        "it; the interim sampler (q1_covariates.py) is reported alongside. Column choices: age_myr; heat flow = "
        "hf_smooth (Gaussian-weighted median of GHFDB sites, sigma 15 km, at least 3 effective sites; none at Blanco); "
        "600 C isotherm = iso600_plate_bsf (GDH1 plate model, ages floored at 0.1 Myr); spreading rate = full_rate_mmyr "
        "(the Seton et al. 2020 rate at which the crust formed, not present-day plate motion; MORVEL was not "
        "available); roughness = rough20 (20.5 km window, plane removed) where the window holds no land; sediment = "
        "sed_thk_m. The slab-surface temperature Tslab_D80 was added to the exploratory family; the registration "
        "named a slab-model isotherm, which the table does not provide.",
        "The external-data job printed its own exploratory segment correlations (for example swarm share against plate "
        "age, rho -0.77, on the pre-D-B113 segments). They were seen after this registration, were not used, and "
        "nothing here was adapted to them.",
        "Vp/Vs: the registration named an orthogonal-regression variant without saying how outliers are handled; it is "
        "computed on the points within 5 MAD of the L1 line. The time-shuffle null stores the swarm members of each "
        "shuffled catalog, so the per-unit null shares follow the current segment join (implementation only).",
    ]
    return out


def TXT_V2(d, h):
    """Boundaries v2 note: key numbers of this run against the D-B113 run."""
    p = d.get("prev")
    if not p:
        return "<p>(no D-B113 snapshot found)</p>"
    S1, S0 = _seg(d["m"]["main"]), _seg(p["m"])
    T1_, T0_ = d["T"].set_index("test"), p["T"].set_index("test")
    D1_, D0_ = d["dec"], p["dec"]

    def rng(S, codes):
        r = _range(S.swarm_share, codes)
        return "-" if r is None else (f"{r['lo']:.1f}-{r['hi']:.1f} % ({NAME[r['lo_seg']]} to {NAME[r['hi_seg']]}; "
                                      f"{r['n']} segments)")

    def tst(T, k):
        r = T.loc[k]
        s = f"{_r(r.rho)} (n {int(r.n)}, Holm {_p(r.p_adj)}"
        if np.isfinite(r.get("block_lo", np.nan)):
            s += f", block {_r(r.block_lo)} to {_r(r.block_hi)}"
        return s + ")"

    def segage(T):
        r = _sec(T, "seg", "swarm_share", "age_myr", "secondary")
        return "-" if r is None else (f"{_r(r.rho)} (n {int(r.n)}, BH {_p(r.p_adj)}, "
                                      f"{'significant' if r.significant else 'not significant'})")

    def hcm(D, key):
        x = D["hot_cold"].get(key, {})
        return f"{_f(x.get('hot_median'))} % / {_f(x.get('cold_median'))} %"
    rows = [
        ("hot group swarm share", rng(S0, HOT), rng(S1, HOT)),
        ("cold group swarm share", rng(S0, COLD), rng(S1, COLD)),
        ("hot / cold median, main run", hcm(D0_, "main_n100"), hcm(D1_, "main_n100")),
        ("hot / cold median, reclustered at Mc 2.15", hcm(D0_, "mcu_recluster_n100"), hcm(D1_, "mcu_recluster_n100")),
        ("T1 plate age, cells", tst(T0_, "T1"), tst(T1_, "T1")),
        ("T4a plate age within types", tst(T0_, "T4a"), tst(T1_, "T4a")),
        ("T2 heat flow, cells", tst(T0_, "T2"), tst(T1_, "T2")),
        ("T4b heat flow within types", tst(T0_, "T4b"), tst(T1_, "T4b")),
        ("segment swarm share vs plate age (secondary, BH)", segage(p["T"]), segage(d["T"])),
        ("control runs with the predicted age sign",
         f"{D0_['d1_detail']['T1']['controls_sign_ok']} / {D0_['d1_detail']['T1']['controls_n']}",
         f"{D1_['d1_detail']['T1']['controls_sign_ok']} / {D1_['d1_detail']['T1']['controls_n']}"),
        ("time-shuffle null rule (hot segments above p95)", f"{D0_['null_rule']['n_above']} of 4",
         f"{D1_['null_rule']['n_above']} of 4"),
        ("verdict, thermal control", D0_["verdict_thermal"], D1_["verdict_thermal"]),
        ("verdict, Vp/Vs", D0_["verdict_vpvs"], D1_["verdict_vpvs"]),
    ]
    for c in ("GR", "MFZ", "GP", "JR", "JP"):
        a, b = S0.reindex([c]), S1.reindex([c])
        f0 = "-" if a.n.isna().all() else f"{_f(a.swarm_share.iloc[0])} % of {int(a.n.iloc[0]):,}"
        f1 = "-" if b.n.isna().all() else f"{_f(b.swarm_share.iloc[0])} % of {int(b.n.iloc[0]):,}"
        rows.append((f"{NAME[c]} swarm share (event-set events)", f0, f1))
    tab = h["table"](pd.DataFrame(rows, columns=["quantity", "D-B113 axes (first report)", "boundaries v2 (D-B117)"]),
                     left=("quantity", "D-B113 axes (first report)", "boundaries v2 (D-B117)"))
    same = (D0_["verdict_thermal"] == D1_["verdict_thermal"]) and (D0_["verdict_vpvs"] == D1_["verdict_vpvs"])
    c1, c0 = d["cells"].set_index("unit_id"), p["cells"].set_index("unit_id")
    kc = c1.index.intersection(c0.index)
    nchg = int((c1.seg_type_major[kc] != c0.seg_type_major[kc]).sum())
    prim_same = all(np.isclose(T1_.loc[k, "rho"], T0_.loc[k, "rho"], equal_nan=True) for k in ("T1", "T2", "T4a", "T4b"))
    txt = (f"<p>The verdicts {'do not change' if same else 'change'} with the v2 boundaries. The boundary moves "
           "reassign events between the Gorda Ridge, the Mendocino fault zone and the Gorda plate, and between the "
           "Juan de Fuca ridge and plate. The cell tests do not use segments except for the cell type in T4; "
           f"{nchg} of the {len(kc)} cells change type, so the primary statistics are "
           f"{'identical' if prim_same else 'changed'} (same events, cells, covariates and seeds). What moves is the "
           "segment table, the hot/cold contrast after reclustering, and the segment-level secondary tests.</p>")
    Bn, Bp = h["Q"].BIG / "nulls", h["Q"].BIG / "runs" / "db113_axes_v1" / "nulls"
    same_n = []
    for f in ("tshuf_swarm_ids.parquet", "mshuf_swarm_ids.parquet"):
        if (Bn / f).exists() and (Bp / f).exists():
            x = pd.read_parquet(Bn / f).sort_values(["seed", "ID"]).reset_index(drop=True)
            y = pd.read_parquet(Bp / f).sort_values(["seed", "ID"]).reset_index(drop=True)
            same_n.append(bool(x.equals(y)))
    if same_n:
        txt += (f"<p>The rerun recomputed the shuffled-catalog nulls from scratch; their swarm memberships "
                f"{'are identical to' if all(same_n) else 'differ from'} those of the first run (same seeds), as "
                "expected for outputs that do not depend on the segments.</p>")
    sa0 = _sec(p["T"], "seg", "swarm_share", "age_myr", "secondary")
    sa1 = _sec(d["T"], "seg", "swarm_share", "age_myr", "secondary")
    if sa0 is not None and sa1 is not None and bool(sa0.significant) != bool(sa1.significant):
        txt += (f"<p>One secondary result changes status: the segment-level correlation of swarm share with plate age "
                f"goes from {_r(sa0.rho)} (BH {_p(sa0.p_adj)}) to {_r(sa1.rho)} (BH {_p(sa1.p_adj)}) over "
                f"{int(sa1.n)} oceanic segments. A result that turns on where the boundaries are drawn is weak "
                "evidence; this is why the registration made the segment-free cells the primary unit.</p>")
    return txt + tab


def TXT_DATA(d, h):
    info = d["info"]["main"]
    V, S, cells = d["V"], d["S"], d["cells"]
    rows = [
        ("Catalog v6.1", "cascadia_catalog_v6.1.csv, md5 " + info.get("catalog_md5", "")[:8],
         f"{info['n_event_set']:,} events in the event set (tiers A-C, earthquakes, above the regional Mc)"),
        ("Clustering v6.1", "v51_clusters_zbz.py --catalog v6.1 (Zaliapin and Ben-Zion nearest-neighbour; nine regions)",
         "families typed swarm / mainshock-aftershock / multiplet; 12 sensitivity variants"),
        ("Mc maps v6.1", "magnitude_converged/mc_maps_v61 (non-earthquakes out)", "mc_best per 0.1 deg node (controls)"),
        ("Segments", "pb_common.assign_segments, plate boundaries v2 adopted (D-B117), recomputed in q1_metrics.py",
         f"{len(S)} segments with >= 100 events"),
        ("Cells", "0.2 x 0.2 deg from 132 W, 38 N", f"{len(cells)} cells with >= 50 events"),
        ("External covariates", "D-B114 events_external_geo_v61.parquet; PROVENANCE.csv",
         "Seton et al. 2020 age and full rate (doi 10.1029/2020GC009214); GHFDB R2024 (10.5880/fidgeo.2024.014); "
         "GlobSed v3 (10.1029/2018GC008115); GEBCO 2026 + GMRT"),
        ("CC differential times", "tomo_v1/cc_dt_all (v5.1 families, D-B093)",
         f"{len(V):,} v5.1 families; {int(V.carried.sum()):,} carry over to one v6.1 family (>= 50 % of members)"),
    ]
    return h["table"](pd.DataFrame(rows, columns=["input", "source", "use"]), left=("input", "source", "use"))


# ------------------------------------------------------------------ results
def TXT_RESULTS(d, h):
    Q, table, figure, FIG, esc = h["Q"], h["table"], h["figure"], h["FIG"], h["esc"]
    T, S, N, V, cells = d["T"], d["S"], d["N"], d["V"], d["cells"]
    t1, t2, t4a, t4b = (_row(T, k) for k in ("T1", "T2", "T4a", "T4b"))
    Sm = _seg(d["m"]["main"])
    nt = N[(N.null == "tshuf") & (N.unit == "seg")].set_index("unit_id")
    nm = N[(N.null == "mshuf") & (N.unit == "seg")].set_index("unit_id")
    s = S.copy()
    s["null_t95"] = s.unit_id.map(nt.null_p95)
    s["null_m_mean"] = s.unit_id.map(nm.null_mean)
    s["mig"] = [f"{int(a)}/{int(b)}" if b > 0 else "-" for a, b in zip(s.swarm10_migrating.fillna(0), s.n_swarm10.fillna(0))]
    s["group"] = ["hot" if u in HOT else "cold" if u in COLD else "" for u in s.unit_id]
    order = [x for x in Q.P.SEG_ORDER if x in set(s.unit_id)]
    s = s.set_index("unit_id").loc[order].reset_index()
    a = cells[cells.age_myr.notna()]
    hot, cold = _range(Sm.swarm_share, HOT), _range(Sm.swarm_share, COLD)
    other = Sm[~Sm.index.isin(HOT + COLD) & (Sm.seg_type_major != "other")].swarm_share
    b = Sm.b.dropna().sort_values()
    pr = Sm.productivity.dropna().sort_values()
    out = []
    A = out.append
    A("<h3>4.1 Segments</h3>")
    A(f"<p>Swarm share above Mc separates two groups of segments. The hot group ({_names(hot['segs'])}) puts "
      f"{hot['lo']:.1f}-{hot['hi']:.1f} % of its earthquakes in swarms; the cold group ({_names(cold['segs'])}) "
      f"{cold['lo']:.1f}-{cold['hi']:.1f} %. The lowest hot segment ({NAME[hot['lo_seg']]}, {hot['lo']:.1f} %) is "
      f"{hot['lo'] / cold['hi']:.1f} times the highest cold segment ({NAME[cold['hi_seg']]}, {cold['hi']:.1f} %). The "
      f"other segments range from {other.min():.1f} % ({NAME[other.idxmin()]}) to {other.max():.1f} % "
      f"({NAME[other.idxmax()]}). This contrast generated the hypothesis, so it is described here, not tested. b "
      f"ranges from {b.iloc[0]:.2f} ({NAME[b.index[0]]}) and {b.iloc[1]:.2f} ({NAME[b.index[1]]}) to {b.iloc[-2]:.2f} "
      f"({NAME[b.index[-2]]}) and {b.iloc[-1]:.2f} ({NAME[b.index[-1]]}); it does not split along the hot and cold "
      f"groups (Axial {Sm.b.get('AX', np.nan):.2f}, Gorda Ridge {Sm.b.get('GR', np.nan):.2f}). Productivity (direct "
      f"offspring of events 1-2 units above Mc) is highest at {_the(pr.index[-1])} ({pr.iloc[-1]:.2f}) and "
      f"{_the(pr.index[-2])} ({pr.iloc[-2]:.2f}) and lowest at {_the(pr.index[1])} ({pr.iloc[1]:.2f}) and "
      f"{_the(pr.index[0])} ({pr.iloc[0]:.2f}): in swarm settings the chain-like families give each event offspring, "
      "so this measure does not isolate aftershock productivity.</p>")
    A(figure(FIG / "fig_q1_segments.png",
             "Swarm share, b and productivity per segment (plate boundaries v2, >= 100 events). Bars: 95 % bootstrap "
             "intervals (families resampled as whole units for the swarm share; events for b; parent events for "
             "productivity). Black ticks: 95th percentile of the swarm share in 100 time-shuffled catalogs. Crosses: "
             "mean swarm share when magnitudes are shuffled within each clustering region and the families are "
             "retyped. Colours: batlowS by segment type with one marker per type."))
    cols = ["unit_id", "group", "seg_type_major", "n", "swarm_share", "swarm_share_lo", "swarm_share_hi", "bg_frac",
            "fam_mix", "b", "b_lo", "b_hi", "productivity", "swarm_dur_med_days", "mig", "age_myr", "heatflow_mw",
            "null_t95", "null_m_mean"]
    names = ["seg", "group", "type", "events", "swarm %", "lo", "hi", "background %", "swarm fam. share", "b", "lo", "hi",
             "productivity", "swarm duration (d)", "migrating / swarms n>=10", "age (Myr)", "heat flow", "time-null p95",
             "mag-null mean"]
    A(table(s, cols, names, nd={"swarm_share": 1, "swarm_share_lo": 1, "swarm_share_hi": 1, "bg_frac": 1,
                                "null_t95": 1, "null_m_mean": 1, "swarm_dur_med_days": 2, "age_myr": 2,
                                "heatflow_mw": 0}))
    A('<p class="muted">Swarm duration: median over swarm families with >= 5 events. Age and heat flow: median of the '
      "D-B114 values over the segment's events (empty when fewer than half carry a value).</p>")

    A("<h3>4.2 Spatial cells and the primary tests</h3>")
    Sc = S.set_index("unit_id")
    hf_empty = [c for c in ("BL", "AX") if c in Sc.index and not np.isfinite(Sc.heatflow_mw.get(c, np.nan))]
    ntj = int((a.seg_type_major == "triple junction").sum())
    A(f"<p>{len(cells)} cells of 0.2 degree hold at least 50 events. Plate age exists in {int(t1.n)} of them (oceanic "
      f"crust in the age grid: ridges, Blanco, the Mendocino fault zone, the Gorda plate and {ntj} triple-junction "
      f"cells), heat flow in {int(t2.n)} (mostly onshore and around the triple junction; the smoothed field has no "
      f"value at {_names(hf_empty) if hf_empty else 'no segment'}).</p>")
    A(figure(FIG / "fig_q1_map.png", "Cells with >= 50 events: swarm share, plate age and heat flow (D-B114 values, "
             "median over the cell's events). Open squares: no value. Crameri batlow and lajolla."))
    old = a[a.age_myr > 2]
    old_sw = old[old.swarm_share > 0]
    old_txt = ("none has any swarm events" if not len(old_sw) else
               f"{'one' if len(old_sw) == 1 else len(old_sw)} {'has' if len(old_sw) == 1 else 'have'} swarm events ("
               + "; ".join(f"a {NAME.get(r.seg_major, r.seg_major)} cell at {r.age_myr:.1f} Myr, swarm share "
                           f"{r.swarm_share:.0f} %" for _, r in old_sw.iterrows())
               + f") and the other {len(old) - len(old_sw)} "
               f"({_names(sorted(set(old.seg_major) - set(old_sw.seg_major)) or ['-'])}) have none")
    A(f"<p><b>T1, plate age.</b> rho = {_r(t1.rho)} over {int(t1.n)} cells, one-sided p = {_p(t1.p)}, Holm-adjusted "
      f"{_p(t1.p_adj)}: {'significant' if t1.significant else 'not significant'} with the "
      f"{'predicted' if t1.sign_ok else 'opposite'} sign. The 1 x 1 degree block bootstrap gives a 95 % interval of "
      f"{_r(t1.block_lo)} to {_r(t1.block_hi)}"
      f"{', which includes 0' if t1.block_lo < 0 < t1.block_hi else ''}. The {int(t1.n)} cells fall in "
      f"{a.block.nunique()} blocks, and the signal comes from one contrast. Of the {len(old)} cells older than 2 Myr, "
      f"{old_txt}. The cells younger than 2 Myr (ridges and Blanco) have a median swarm share of "
      f"{a[a.age_myr < 2].swarm_share.median():.1f} %.</p>")
    tr = a[a.seg_type_major == "transform"]
    trc = tr.seg_major.value_counts()
    bl = a[a.seg_major == "BL"]
    A(f"<p><b>T4a, plate age within boundary types.</b> Weighted within-type rho = {_r(t4a.rho)}, one-sided p = "
      f"{_p(t4a.p)} (Holm {_p(t4a.p_adj)}). Per type: {esc(_within(t4a))}. The {len(tr)} transform cells are "
      + _names_n(trc) + " cells. Looked at after the test (not "
      f"registered): the {len(bl)} Blanco cells alone ({bl.age_myr.min():.1f}-{bl.age_myr.max():.1f} Myr, swarm share "
      f"{bl.swarm_share.min():.0f}-{bl.swarm_share.max():.0f} %) give rho "
      f"{spearmanr(bl.age_myr, bl.swarm_share)[0]:+.2f}, "
      f"{'the opposite sign to the prediction' if spearmanr(bl.age_myr, bl.swarm_share)[0] > 0 else 'the predicted sign'}"
      f". The quasi-binomial model with type terms gives {float(t4a.glm_coef_per_sd):+.2f} +- {float(t4a.glm_se):.2f} "
      f"on the logit scale per standard deviation of age (dispersion {float(t4a.glm_dispersion):.0f}).</p>")
    ti2 = _row(d["Ti"], "T2") if d.get("Ti") is not None else None
    glm_ok = np.isfinite(t4b.glm_coef_per_sd) and abs(t4b.glm_coef_per_sd) < 50
    A(f"<p><b>T2 and T4b, heat flow.</b> rho = {_r(t2.rho)} over {int(t2.n)} cells, one-sided p = {_p(t2.p)}, Holm "
      f"{_p(t2.p_adj)}; block interval {_r(t2.block_lo)} to {_r(t2.block_hi)}. Within types {_r(t4b.rho)} (Holm "
      f"{_p(t4b.p_adj)}; {esc(_within(t4b))}); "
      + (f"the type-controlled model gives {float(t4b.glm_coef_per_sd):+.2f} +- {float(t4b.glm_se):.2f} per standard "
         "deviation. " if glm_ok else "the type-controlled model does not converge (types whose cells have no swarms "
         "separate the data), so it gives no effect size. ")
      + (f"With the interim heat-flow sampler (median of GHFDB sites within 25 km) T2 is {_r(ti2.rho)}: the heat-flow "
         "result depends on how sparse site values are spread to the events" if ti2 is not None else "")
      + (f" (with the interim values the heat-flow sign holds in "
         f"{d['dec_int']['d1_detail']['T2']['controls_sign_ok']} of 22 control runs, against "
         f"{d['dec']['d1_detail']['T2']['controls_sign_ok']} with the D-B114 field)" if d.get("dec_int") else "")
      + ".</p>")
    A(figure(FIG / "fig_q1_scatter.png", "Cell swarm share against plate age and heat flow (primary covariates) and "
             "against roughness and sediment thickness (exploratory). rho and p: Spearman and its permutation p "
             "(one-sided where a sign was predicted)."))
    prim = T[T.family == "primary"].copy()
    A(table(prim, ["test", "covariate", "rho", "n", "p", "p_adj", "block_lo", "block_hi", "significant", "sign_ok"],
            ["test", "covariate", "statistic", "n", "p (one-sided)", "Holm p", "block lo", "block hi",
             "significant (Holm)", "predicted sign"], nd={"rho": 3, "p": 4, "p_adj": 4}))
    A('<p class="muted">T3 statistic: difference of median Vp/Vs, swarm minus mainshock-aftershock families; empty '
      "when the test is not testable (it needs 20 families of each type).</p>")

    A("<h3>4.3 Secondary and exploratory tests</h3>")
    sec = T[T.family != "primary"].copy()
    A(table(sec, ["family", "unit", "metric", "covariate", "pred_sign", "rho", "n", "p", "p_adj", "significant"],
            ["family", "unit", "metric", "covariate", "predicted sign", "rho", "n", "p", "BH p", "significant (BH)"],
            nd={"rho": 3, "p": 4, "p_adj": 4}))
    s2 = T[T.family == "secondary"].dropna(subset=["p_adj"]).sort_values("p_adj")
    nsig = int(s2.significant.sum())
    close = s2.head(2)
    close_txt = "; ".join(f"{LABEL.get(r.metric, r.metric)} against {LABEL.get(r.covariate, r.covariate)} "
                          f"({int(r.n)} {'segments' if r.unit == 'seg' else 'cells'}, rho {_r(r.rho)}, BH "
                          f"{_p2(r.p_adj)}, {'predicted' if r.sign_ok else 'opposite'} sign"
                          f"{', significant' if r.significant else ''})" for _, r in close.iterrows())
    pa = _sec(T, "cell", "productivity", "age_myr")
    ex = {c: _sec(T, "cell", "swarm_share", c, "exploratory") for c in ("sed_m", "rough_m", "z600_km", "Tslab_C")}
    exs = _sec(T, "seg", "swarm_share", "Tslab_C", "exploratory")
    A(f"<p>{'None' if nsig == 0 else ('One' if nsig == 1 else nsig)} of the {len(s2)} secondary tests "
      f"{'passes' if nsig <= 1 else 'pass'} the Benjamini-Hochberg threshold. The two smallest BH values: "
      f"{close_txt}. Productivity against age has "
      f"{'the opposite sign to' if not pa.sign_ok else 'the sign of'} the prediction (rho {_r(pa.rho)}, "
      f"{int(pa.n)} cells). Among the exploratory covariates, the swarm share changes with sediment thickness (cells, "
      f"rho {_r(ex['sed_m'].rho)}) and roughness (rho {_r(ex['rough_m'].rho)}). Both follow the same geography as age "
      "(thin sediment and rough, young seafloor at the ridges and Blanco; thick sediment on the Gorda plate and near "
      "the margin), so they describe the same contrast and were not predicted. The 600 C isotherm depth has the ranks "
      f"of plate age (rho {_r(ex['z600_km'].rho)}; as registered it is not a separate test). The slab-surface "
      f"temperature gives rho {_r(exs.rho)} over {int(exs.n)} segments and {_r(ex['Tslab_C'].rho)} over cells: the "
      "hot-slab segments are the arc segments far from the trench, so this is the arc-forearc contrast again.</p>")

    A("<h3>4.4 In-situ Vp/Vs of earthquake families</h3>")
    vv = V[V.valid & V.carried]
    ax = V[(V.block == "axial") & V.vpvs.notna()]
    e = V[V.vpvs.notna()].copy()
    e["q"] = pd.qcut(e.snr, 5, labels=False)
    qm = e.groupby("q").vpvs.median()
    top = e[(e.q == 4) & e.valid]
    vb = V[V.valid].block.value_counts()
    ridge_valid = int(V[V.valid].seg_type.eq("ridge").sum())
    A(f"<p>Of the {len(V):,} v5.1 families, {int(V.carried.sum()):,} carry over to one v6.1 family (that family "
      f"holds >= 50 % of their members). With the registered rules (CC-accepted P and S at >= 3 common stations per "
      f"pair, pairs within 5 km, >= 5 events and >= 30 points per family), {int(V.vpvs.notna().sum())} families give "
      f"an estimate and {int(V.valid.sum())} are valid (95 % half-width &lt;= 0.10, value within 1.4-2.5): "
      f"{int((vv.type61 == 'swarm').sum())} swarm, {int((vv.type61 == 'mainshock-aftershock').sum())} "
      f"mainshock-aftershock and {int((vv.type61 == 'multiplet').sum())} multiplet families. T3 is therefore "
      f"{'not testable' if d['dec']['verdict_vpvs'] == 'not testable' else 'tested'}. "
      + ("None of the valid families is" if ridge_valid == 0 else f"{ridge_valid} of the valid families "
         + ("is" if ridge_valid == 1 else "are")) + f" on a ridge; the most are in the {vb.index[0]} CC block "
      f"({int(vb.iloc[0])}).</p>")
    A(f"<p>Two measurement limits explain the low yield. First, regression dilution: within a family the demeaned P "
      f"differential times spread over only about {1000 * e.dP_iqr_s.median():.0f} ms (median interquartile range), "
      f"close to the CC noise (median residual MAD {1000 * e.resid_mad_s.median():.0f} ms). The L1 slope then falls "
      f"toward zero as noise grows: the median estimate rises from {qm.iloc[0]:.2f} in the lowest fifth of "
      f"signal-to-noise to {qm.iloc[-1]:.2f} in the highest fifth, where the valid values span "
      f"{top.vpvs.min():.2f}-{top.vpvs.max():.2f}. Second, at Axial the S-P time is about 0.5 s, so the S window "
      f"(-0.3 to +1.0 s around S) contains P energy; the {len(ax)} Axial estimates have a median of "
      f"{ax.vpvs.median():.2f}, below any rock value, with a median half-width of {ax.halfw.median():.2f}. Catalog "
      f"picks (the registered fallback) are worse: their noise exceeds the spread, the median estimate is "
      f"{V.vpvs_cat.median():.2f} and {int(V.valid_cat.sum())} families are valid. The 11-25 feasibility count (621 "
      "families) counted catalog picks only and did not check precision.</p>")
    A(f"<p>For the record, not as a test: the valid swarm families have a median Vp/Vs of "
      f"{vv[vv.type61 == 'swarm'].vpvs.median():.2f} (n = {int((vv.type61 == 'swarm').sum())}) and the "
      f"mainshock-aftershock families {vv[vv.type61 == 'mainshock-aftershock'].vpvs.median():.2f} (n = "
      f"{int((vv.type61 == 'mainshock-aftershock').sum())}), "
      + ("the opposite order to the hypothesis" if vv[vv.type61 == 'swarm'].vpvs.median()
         < vv[vv.type61 == 'mainshock-aftershock'].vpvs.median() else "the order of the hypothesis")
      + ", on too few families to mean anything.</p>")
    A(figure(FIG / "fig_q1_vpvs.png", "Left: family Vp/Vs (L1 slope of demeaned S on P CC differential times) against "
             "signal-to-noise; grey band: the registered valid range; crosses: Axial, where S-P is about 0.5 s. Right: "
             "valid families by v6.1 type, with 95 % bootstrap intervals; markers and colours by segment type."))
    return "\n".join(out)


# ------------------------------------------------------------------ controls
def TXT_CONTROLS(d, h):
    Q, table, figure, FIG = h["Q"], h["table"], h["figure"], h["FIG"]
    C, N, dec = d["C"], d["N"], d["dec"]
    out = []
    A = out.append
    det = dec["d1_detail"]
    ca = C[C.covariate == "age_myr"].set_index("run")
    weak = ca.rho.sort_values(ascending=False)
    th = ca.reindex(THIN)
    hf_bad = C[(C.covariate == "heatflow_mw") & ~C.sign_ok].run.tolist()
    A(f"<p><b>Sign stability.</b> The cell rho of swarm share against plate age has the predicted sign in "
      f"{det['T1']['controls_sign_ok']} of {det['T1']['controls_n']} control runs; against heat flow in "
      f"{det['T2']['controls_sign_ok']} of {det['T2']['controls_n']}"
      + (f" (exceptions: {', '.join(hf_bad)})" if hf_bad else "") + ". The size is less stable than the sign: with "
      f"plate age, rho ranges from {weak.iloc[0]:+.2f} ({weak.index[0]}, {int(ca.n[weak.index[0]])} cells) and "
      f"{weak.iloc[1]:+.2f} ({weak.index[1]}) to {weak.iloc[-1]:+.2f} ({weak.index[-1]}), and the runs that thin the "
      f"cells ({', '.join(THIN)}) give {th.rho.min():+.2f} to {th.rho.max():+.2f} with one-sided p of "
      f"{th.p_one.min():.2g}-{th.p_one.max():.2g}.</p>")
    A(figure(FIG / "fig_q1_controls.png", "Spearman rho of cell swarm share with plate age and heat flow in the main "
             "run (black, with the 1 x 1 degree block-bootstrap 95 % interval) and in the 22 control runs (blue: "
             "predicted sign; orange: opposite sign)."))
    c = C.pivot(index="run", columns="covariate", values=["rho", "p_one", "n"])
    c.columns = [f"{a}_{b}" for a, b in c.columns]
    c = c.reindex(list(dict.fromkeys(C.run))).reset_index()
    for k in ("n_age_myr", "n_heatflow_mw"):
        c[k] = c[k].astype(int)
    A(table(c, ["run", "rho_age_myr", "p_one_age_myr", "n_age_myr", "rho_heatflow_mw", "p_one_heatflow_mw",
                "n_heatflow_mw"], ["control run", "rho age", "p", "cells", "rho heat flow", "p", "cells"],
            nd={"rho_age_myr": 3, "p_one_age_myr": 4, "rho_heatflow_mw": 3, "p_one_heatflow_mw": 4}))
    A('<p class="muted">mcu_count: only M >= 2.15 events counted, main families. mcu_recluster: clustering redone '
      "with Mc 2.15 in every region. mc_local: events at or above the Mc map value of their 0.1 degree node. var_*: "
      "the 12 v6.1 clustering variants. offshore: cells with most events offshore. partial: partial rank "
      "correlation controlling for the Mc map value and the distance to the nearest station. ci_window: 2011-10 to "
      "2015-09 only. cell04: 0.4 degree cells. N30 / N100: other minimum event counts. v60: catalog v6.0, its "
      "clustering and the same segment rule (interim covariates on the v6.0 positions).</p>")
    hc = _hot_cold(Q)
    mr = _seg(d["m"]["mcu_recluster"], 0)
    m0, m1 = _seg(d["m"]["v60"]), _seg(d["m"]["main"])
    k = m0.index.intersection(m1.index)
    rk = spearmanr(m0.swarm_share[k], m1.swarm_share[k])[0]
    hcr = dec["hot_cold"]["mcu_recluster_n100"]
    A(f"<p><b>Completeness, clustering parameters and catalog version (hot against cold segments).</b> The lowest hot "
      f"segment stays above the highest cold segment in {int(hc.separated.sum())} of {len(hc)} runs. With the "
      f"clustering redone at a uniform Mc of 2.15 (Axial keeps {int(mr.n.get('AX', 0))} tier A-C events above it), "
      f"the hot median is {hcr['hot_median']:.1f} % and the cold median {hcr['cold_median']:.1f} %, so the "
      f"falsification rule on completeness {'does not apply' if hcr['hot_above_cold'] else 'applies'}. The segment "
      f"swarm shares of v6.0 and v6.1 have a rank correlation of {rk:.2f} (same segment rule on both); Mount St "
      f"Helens moves from {m0.swarm_share.get('MSH', np.nan):.0f} % to {m1.swarm_share.get('MSH', np.nan):.0f} % "
      "(one family retyped, D-B111 family instability).</p>")
    A(table(hc, ["run", "n_hot", "hot_median", "hot_min", "n_cold", "cold_median", "cold_max", "separated"],
            ["run", "hot segments", "hot median %", "hot min %", "cold segments", "cold median %", "cold max %",
             "min hot > max cold"], nd={"hot_median": 1, "hot_min": 1, "cold_median": 1, "cold_max": 1}))
    nt = N[(N.null == "tshuf") & (N.unit == "seg")].copy()
    nm = N[(N.null == "mshuf") & (N.unit == "seg")].set_index("unit_id")
    nt["mshuf_mean"] = nt.unit_id.map(nm.null_mean)
    nt["mshuf_p05"] = nt.unit_id.map(nm.null_p05)
    nt["below_mshuf_p05"] = nt.unit_id.map(nm.below_p05)
    order = [x for x in Q.P.SEG_ORDER if x in set(nt.unit_id)]
    nt = nt.set_index("unit_id").loc[order]
    nr = dec.get("null_rule") or {}
    others = nt.drop(index=[x for x in ["AX"] if x in nt.index]).null_mean
    cold_txt = "; ".join(f"{NAME[c]} {nt.swarm_share[c]:.1f} % against {nt.mshuf_mean[c]:.0f} %"
                         for c in COLD if c in nt.index)
    rand = [c for c in HOT + ["CAS"] if c in nt.index and not nt.below_mshuf_p05[c]]
    mid = [c for c in HOT if c in nt.index and nt.below_mshuf_p05[c]]
    A(f"<p><b>Shuffled-catalog nulls.</b> Time shuffle (100 catalogs, event times permuted within each clustering "
      f"region, real thresholds): {_names([c for c in ['AX', 'BL', 'GR', 'MSH'] if nr.get('above', {}).get(c)])} lie "
      f"above the 95th percentile of their null swarm share, so the null rule "
      f"{'passes' if nr.get('passed') else 'fails'} ({nr.get('n_above', '-')} of 4). Dense seismicity alone makes "
      f"swarms at Axial (null mean {nt.null_mean['AX']:.1f} %, observed {nt.swarm_share['AX']:.1f} %); elsewhere the "
      f"null mean is {others.min():.1f}-{others.max():.1f} %. Magnitude shuffle (200 draws, family membership and "
      f"trees kept): in the cold segments the observed swarm share is below the 5th percentile of the "
      f"magnitude-shuffle null ({cold_txt}), so their families carry a dominant mainshock far more often than random "
      f"magnitudes would give. At {_names(rand)} the observed share is not below that range: their families look "
      f"like random draws in magnitude."
      + (f" {_names(mid, cap=True)} sit in between (below the 5th percentile, at "
         + " and ".join(f"{nt.swarm_share[c]:.0f} %" for c in mid) + ")." if mid else "") + "</p>")
    A(table(nt.reset_index(), ["unit_id", "n", "swarm_share", "null_mean", "null_p95", "above_p95", "mshuf_mean",
                               "mshuf_p05", "below_mshuf_p05"],
            ["seg", "events", "observed %", "time-null mean %", "time-null p95 %", "above p95",
             "mag-null mean %", "mag-null p05 %", "below mag-null p05"],
            nd={"swarm_share": 1, "null_mean": 1, "null_p95": 1, "mshuf_mean": 1, "mshuf_p05": 1}))
    pc, cw = ca.loc["partial"], ca.loc["ci_window"]
    regs = d["regions"].set_index("region").Mc
    A(f"<p><b>Network geometry and detection.</b> Every plate-age cell is offshore, so the offshore-only run equals the "
      f"main run for T1 (rho {ca.rho['offshore']:+.2f}). The partial correlation that removes the Mc map value and the "
      f"nearest-station distance gives rho {pc.rho:+.2f} (p {pc.p_one:.2g}). The Cascadia Initiative window (2011-10 "
      f"to 2015-09) gives {cw.rho:+.2f} on {int(cw.n)} cells. Axial has the lowest Mc (segment "
      f"{_seg(d['m']['main']).Mc_unit.get('AX', np.nan):.2f}, clustering region {regs.get('AX', np.nan):.2f}) "
      f"because of the cabled array; under the uniform Mc of 2.15 its swarm share is "
      f"{mr.swarm_share.get('AX', np.nan):.0f} %, so its high share is not a product of its low Mc.</p>")
    return "\n".join(out)


# ------------------------------------------------------------------ claims, bullets, limitations
def TXT_CLAIMS(d, h):
    dec, T = d["dec"], d["T"]
    t1, t2, t4a, t3 = (_row(T, k) for k in ("T1", "T2", "T4a", "T3"))
    Sm = _seg(d["m"]["main"])
    hot, cold = _range(Sm.swarm_share, HOT), _range(Sm.swarm_share, COLD)
    hcr = dec["hot_cold"]["mcu_recluster_n100"]
    hc = _hot_cold(h["Q"])
    a = d["cells"][d["cells"].age_myr.notna()]
    old = a[a.age_myr > 2]
    nsw_old = int((old.swarm_share > 0).sum())
    N = d["N"]
    nm = N[(N.null == "mshuf") & (N.unit == "seg")].set_index("unit_id")
    cold_below = [c for c in COLD if c in nm.index and nm.below_p05[c]]
    rand = [c for c in HOT if c in nm.index and not nm.below_p05[c]]
    det = dec["d1_detail"]["T1"]
    sa = _sec(T, "seg", "swarm_share", "age_myr", "secondary")
    sa0 = _sec(d["prev"]["T"], "seg", "swarm_share", "age_myr", "secondary") if d.get("prev") else None
    return (
        f'<div class="verdict"><b>Pre-registered outcome.</b> Thermal control: {dec["verdict_thermal"]} (T1 '
        f"{'is significant' if t1.significant else 'is not significant'} with the "
        f"{'predicted' if t1.sign_ok else 'opposite'} sign, the null rule "
        f"{'passes' if dec['null_rule']['passed'] else 'fails'}, the sign holds in {det['controls_sign_ok']} of "
        f"{det['controls_n']} control runs, "
        + ("but the block-bootstrap interval of T1 includes 0, so D1 fails" if not det["block_ci_excludes_0"]
           else "and the block-bootstrap interval of T1 excludes 0") + "; T4a gives "
        f"{_r(t4a.rho)} within boundary types). Falsification rules: "
        + ("none applies" if not any(v for v in dec["falsification"].values() if v) else
           ", ".join(k for k, v in dec["falsification"].items() if v) + " apply") + ". Fluid signal: "
        f"{dec['verdict_vpvs']}.</div>"
        "<h3>The paper can claim</h3><ul>"
        f"<li>Slip style splits by boundary setting: {hot['lo']:.0f}-{hot['hi']:.0f} % of earthquakes above Mc are in "
        f"swarms at {_names(hot['segs'])}, against {cold['lo']:.1f}-{cold['hi']:.1f} % at {_names(cold['segs'])}. The "
        f"split survives a uniform Mc of 2.15 (reclustered: hot median {hcr['hot_median']:.1f} %, cold median "
        f"{hcr['cold_median']:.1f} %), the 12 clustering variants and catalog v6.0 (lowest hot above highest cold in "
        f"{int(hc.separated.sum())} of {len(hc)} runs), and a time-shuffle null.</li>"
        f"<li>At {_names(cold_below)} the families carry a dominant mainshock far more often than random magnitude "
        f"ordering would give (swarm share below the 5th percentile of a magnitude-shuffle null), while at "
        f"{_names(rand)} the families cannot be told apart from random magnitude ordering.</li>"
        f"<li>Across {int(t1.n)} oceanic 0.2 degree cells, the swarm share falls with plate age (Spearman {_r(t1.rho)}, "
        f"Holm-adjusted p {_p(t1.p_adj)}); of the {len(old)} cells older than 2 Myr, "
        f"{'one' if nsw_old == 1 else nsw_old} {'has' if nsw_old == 1 else 'have'} swarms. Stated as a correlation "
        "that tracks the boundary settings, with its block interval.</li>"
        + (f"<li>Over the {int(sa.n)} oceanic segments the swarm share falls with plate age (Spearman {_r(sa.rho)}, "
           f"BH {_p(sa.p_adj)}, a secondary test); it is significant only with the v2 boundaries"
           + (f" (D-B113 segments: {_r(sa0.rho)}, BH {_p(sa0.p_adj)})" if sa0 is not None else "")
           + "; report it with that dependence.</li>" if sa is not None and sa.significant else "")
        + "</ul><h3>The paper cannot claim</h3><ul>"
        f"<li>That plate age or temperature sets the slip style beyond the boundary type: within types the age effect "
        f"is {_r(t4a.rho)} (weighted rho), and the cell correlation rests on {a.block.nunique()} one-degree blocks "
        f"(block interval {_r(t1.block_lo)} to {_r(t1.block_hi)}). Young lithosphere and the swarm-prone settings "
        "coincide here, so these data cannot separate the two.</li>"
        f"<li>Any heat-flow control: T2 is {'not ' if not t2.significant else ''}significant (rho {_r(t2.rho)}, Holm "
        f"{_p(t2.p_adj)}), measured heat flow is missing at Blanco and Axial, and the sign depends on how site values "
        "are spread.</li>"
        f"<li>A fluid (Vp/Vs) control on swarms: T3 is {dec['verdict_vpvs']} ({int(t3.get('n_swarm', 0) or 0)} swarm "
        f"and {int(t3.get('n_ma', 0) or 0)} mainshock-aftershock families with a valid Vp/Vs).</li>"
        "<li>That sediment thickness or roughness control the style: those correlations are exploratory and follow "
        "the same geography as age.</li>"
        "<li>Anything about individual families other than large sequences (D-B111).</li></ul>")


def TXT_BULLETS(d, h):
    dec, T, V = d["dec"], d["T"], d["V"]
    t1, t2 = _row(T, "T1"), _row(T, "T2")
    Sm = _seg(d["m"]["main"])
    hcr = dec["hot_cold"]["mcu_recluster_n100"]
    m0 = _seg(d["m"]["v60"])
    k = m0.index.intersection(Sm.index)
    rk = spearmanr(m0.swarm_share[k], Sm.swarm_share[k])[0]
    N = d["N"]
    nm = N[(N.null == "mshuf") & (N.unit == "seg")].set_index("unit_id")
    cold_below = [c for c in COLD if c in nm.index and nm.below_p05[c]]
    rand = [c for c in ["AX", "MSH"] if c in nm.index and not nm.below_p05[c]]
    nr = dec["null_rule"]
    det = dec["d1_detail"]["T1"]
    sa = _sec(T, "seg", "swarm_share", "age_myr", "secondary")
    sa0 = _sec(d["prev"]["T"], "seg", "swarm_share", "age_myr", "secondary") if d.get("prev") else None
    sh = lambda c: f"{Sm.swarm_share[c]:.1f} % {NAME[c]}"  # noqa: E731
    return [
        "[TO WRITE] Slip style by setting (v6.1, tier A-C earthquakes above the regional Mc, plate boundaries v2, "
        "D-B117): swarm share " + ", ".join(sh(c) for c in ["AX", "MSH", "BL", "GR", "JR"] if c in Sm.index) + "; "
        + ", ".join(sh(c) for c in COLD if c in Sm.index) + ".",
        f"[TO WRITE] Robustness: the split holds at a uniform Mc of 2.15 after reclustering (hot median "
        f"{hcr['hot_median']:.1f} %, cold median {hcr['cold_median']:.1f} %), in the 12 clustering variants, in catalog "
        f"v6.0 (segment rank correlation {rk:.2f}), and above a time-shuffle null (100 catalogs) at "
        f"{_names([c for c in ['AX', 'BL', 'GR', 'MSH'] if nr['above'].get(c)])}.",
        f"[TO WRITE] Mainshock dominance: at {_names(cold_below)} the swarm share is below the 5th percentile of a "
        f"magnitude-shuffle null (Gorda plate {Sm.swarm_share.get('GP', np.nan):.1f} % against "
        f"{nm.null_mean.get('GP', np.nan):.0f} %); at {_names(rand)} it equals the null within its range.",
        f"[TO WRITE] Plate age (Seton et al. 2020): across {int(t1.n)} oceanic 0.2 degree cells the swarm share falls "
        f"with age (Spearman {_r(t1.rho)}, Holm p {_p2(t1.p_adj)}, pre-registered), same sign in "
        f"{det['controls_sign_ok']} of {det['controls_n']} controls; block-bootstrap interval {_r(t1.block_lo)} to "
        f"{_r(t1.block_hi)} and no age effect within boundary types ({_r(_row(T, 'T4a').rho)}): age tracks the setting "
        "contrast, it does not explain it.",
        f"[TO WRITE] Heat flow (GHFDB R2024, smoothed): no significant relation (rho {_r(t2.rho)}, Holm p "
        f"{_p2(t2.p_adj)}); no measured heat flow on the Blanco transform.",
        f"[TO WRITE] In-situ Vp/Vs of families from CC differential times (Lin and Shearer 2007): "
        f"{int(V.valid.sum())} of {len(V):,} families give a valid value; too few to compare swarms with "
        "mainshock-aftershock families; limited by the small spread of differential times and by S-P of 0.5 s at "
        "Axial.",
        *([f"[TO WRITE] Segment level (secondary test, {int(sa.n)} oceanic segments): swarm share falls with plate age "
           f"(Spearman {_r(sa.rho)}, BH {_p2(sa.p_adj)}) with the v2 boundaries; with the D-B113 segments it was "
           f"{_r(sa0.rho)} (BH {_p2(sa0.p_adj)}), so the result depends on where the boundaries are drawn."]
          if sa is not None and sa0 is not None and sa.significant else []),
        "[TO WRITE] Methods: hypotheses, tests and decision rules registered before any covariate correlation "
        "(2026-11-27 13:26 PST); Holm over five primary tests; BH for secondary tests; segments from plate boundaries "
        "v2 (D-B117), an input fixed after the registration.",
        "[TO WRITE] Limitations: see the limitation bullets (family instability, Mc contrast, heat-flow coverage, "
        "Vp/Vs precision, no waveform double-difference catalog).",
    ]


def TXT_LIMITS(d, h):
    Sm, m0 = _seg(d["m"]["main"]), _seg(d["m"]["v60"])
    k = m0.index.intersection(Sm.index)
    rk = spearmanr(m0.swarm_share[k], Sm.swarm_share[k])[0]
    regs = d["regions"].set_index("region").Mc
    mu = _seg(d["m"]["mcu_recluster"])
    dropped = [c for c in Sm.index if c not in mu.index and c in HOT + COLD + ["MSH"]]
    a = d["cells"][d["cells"].age_myr.notna()]
    V = d["V"]
    e = V[V.vpvs.notna()]
    ms = d["ev"][d["ev"].event_set & (d["ev"].seg == "MSH")].merge(d["cat"][["ID", "depth"]], on="ID")
    sw20 = 100 * (ms[ms.depth < 20].cls == "swarm").mean()
    return [
        f"Families shift between catalog versions (D-B111): the shares here are over many families and change little "
        f"between v6.0 and v6.1 (segment rank correlation {rk:.2f}), but Mount St Helens moves from "
        f"{m0.swarm_share.get('MSH', np.nan):.0f} % to {Sm.swarm_share.get('MSH', np.nan):.0f} % because one family "
        "was retyped.",
        f"Mc differs between regions ({regs.min():.2f} at {REGION.get(regs.idxmin(), regs.idxmin())} to "
        f"{regs.max():.2f} at {REGION.get(regs.idxmax(), regs.idxmax())} in the "
        f"clustering regions; segment Mc {Sm.Mc_unit.get('AX', np.nan):.2f} at Axial and "
        f"{Sm.Mc_unit.get('MSH', np.nan):.2f} at Mount St Helens). The event set is above the regional Mc, and the "
        "uniform-Mc runs keep the contrast, but at Mc 2.15 "
        + (_names(dropped) if dropped else "no segment") + (" drops" if len(dropped) <= 1 else " drop")
        + " below 100 events.",
        "The swarm and mainshock-aftershock types come from topology and magnitude rules (Zaliapin and Ben-Zion) "
        "applied to families of >= 5 events; families of 2-4 events are multiplets or mainshock-aftershock pairs by the "
        "magnitude gap only. The magnitude-shuffle null shows that random magnitudes alone make many families swarms.",
        "Plate age exists only on oceanic crust in the age grid, so the cell tests cover ridges, Blanco, the Mendocino "
        "fault zone, the Gorda plate and a few triple-junction cells; the slab, the forearc and onshore settings enter "
        "only the segment contrast.",
        f"Spatial dependence: {len(a)} age cells in {a.block.nunique()} one-degree blocks. Neighbouring cells share "
        "families and structure, so the cell count overstates the independent samples; the block interval is the "
        "honest uncertainty.",
        "Heat flow: GHFDB sites are sparse offshore, absent on the Blanco transform, and ridge-crest values are "
        "lowered by hydrothermal circulation; the smoothed field and the 25 km median give different results.",
        f"Spreading rate is the rate at which the crust formed (Seton et al. 2020 grid), not present-day plate motion; "
        f"it varies little here ({a.rate_mmyr.min():.0f}-{a.rate_mmyr.max():.0f} mm/yr in the plate-age cells).",
        f"Vp/Vs: the CC differential times within families spread over about {1000 * e.dP_iqr_s.median():.0f} ms "
        f"against about {1000 * e.resid_mad_s.median():.0f} ms of noise, so the slope is biased low except in the "
        "best-sampled families; at Axial the S window contains P energy (S-P about 0.5 s). A useful family Vp/Vs needs "
        "a P-free S window and families with a wider spread of station geometry.",
        "No waveform double-difference catalog: swarm migration and duration are provisional.",
        f"Deep events under Mount St Helens are not interpreted (D-B111). The Mount St Helens event set holds "
        f"{len(ms)} events within 25 km of the summit at any depth; {int((ms.depth > 30).sum())} are deeper than "
        f"30 km, {'none' if not (ms[ms.depth > 30].cls == 'swarm').any() else int((ms[ms.depth > 30].cls == 'swarm').sum())} "
        f"of them in swarms. Without the "
        f"{int((ms.depth >= 20).sum())} events at 20 km or deeper the swarm share is {sw20:.1f} % "
        f"({Sm.swarm_share.get('MSH', np.nan):.1f} % with them).",
        "Segment boundaries changed twice after the registration (D-B113, then plate boundaries v2, D-B117); "
        "the comparison in the section Boundaries v2 shows what moved.",
    ]


def TXT_FILES(d, h):
    Q = h["Q"]
    return ("<h2>Files</h2><ul>"
            "<li>Code: <code>workflow/11_paper_plan/q1_*.py</code>, driver <code>run_q1.sh</code> (branch "
            "science/q1-slip-style). The segment join is recomputed inside q1_metrics.py; rerun the driver after any "
            "change of the segment rule.</li>"
            f"<li>Tables: <code>{Q.BIG}/tables/</code> (metrics_*.csv per run, tests_external.csv, "
            "tests_interim.csv, controls_*.csv, nulls_units.csv, decision.json); small copies in "
            "<code>data/paper_plan/q1_slip_style/</code>.</li>"
            f"<li>Nulls: <code>{Q.BIG}/nulls/</code>; Vp/Vs: <code>{Q.BIG}/vpvs/vpvs_families.csv</code>; figures: "
            f"<code>{Q.BIG}/figures/</code>.</li>"
            f"<li>The first run (D-B113 axes v1): <code>{Q.BIG}/runs/db113_axes_v1/</code> (tables, figures, nulls, "
            "report).</li>"
            "<li>Pre-registration: <code>data/paper_plan/q1_slip_style/RESUME.md</code> (commit 109915b8).</li></ul>")
