#!/usr/bin/env python3
"""Q2 report (D-B115): <Q2_BIG>/report_q2.html from the q2_* tables and figures. Numbers in the text come from the
tables; the interpretive sentences were written for the 2026-11-27 run (inputs and md5s listed in the report) and must
be re-read after any rerun. Also writes <Q2_OUT>/robustness.csv (registered control labels per group)."""
from __future__ import annotations

import base64
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q2_analysis as A  # noqa: E402
import q2_common as Q  # noqa: E402

FIG = Q.BIG / "figures"
CONTROLS = [("C1 sig_z <= 2 km", "sigz2"), ("C1 tier A-B, no sig_z cut", "nosig"), ("C2 catalog v6.0", "v60"),
            ("C3 nearest station within one focal depth", "geom"), ("C4 checkerboard-only mask", "cb"),
            ("C5a events not in the tomoDD data", "notomo")]
VARIANTS = [("U1 model columns", "U1"), ("D95", "D95"), ("tiers A-C, no sig_z cut", "abc")]
KEY_GROUPS = ["frame:oceanic", "frame:slab", "frame:oceanic+slab", "ridge", "ridge (no AX)", "transform", "intraplate",
              "triple junction", "slab", "Gorda slab (GS)", "seg:AX", "seg:GR", "seg:BL", "seg:MFZ", "seg:GP",
              "seg:MTJ", "seg:GS", "seg:SIF", "seg:IS"]
H_C_ANSWERS = {"composition-limited", "both consistent", "mixed", "H_C supported, not preferred",
               "H_C supported (H_T not testable)"}


def img(name, alt, width="100%"):
    p = FIG / name
    if not p.exists():
        return f"<p class='miss'>[figure {name} missing]</p>"
    b = base64.b64encode(p.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{b}' alt='{html.escape(alt)}' style='width:{width}'>"


def tbl(df, cls="t", fmt=None):
    fmt = fmt or {}
    h = "<table class='%s'><thead><tr>" % cls + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + \
        "</tr></thead><tbody>"
    for _, r in df.iterrows():
        h += "<tr>"
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                s = "" if not np.isfinite(v) else (fmt.get(c, "{:.2f}").format(v))
            else:
                s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            h += f"<td>{html.escape(s)}</td>"
        h += "</tr>"
    return h + "</tbody></table>"


def gname(g):
    return g.replace("seg:", "segment ").replace("frame:", "frame: ")


def load(name, cfg):
    p = Q.OUT / f"{name}_{cfg}.csv"
    return pd.read_csv(p) if p.exists() else None


def robustness():
    ans = {c: load("answers", c) for c in ["main"] + [c for _, c in CONTROLS] + [c for _, c in VARIANTS]}
    c3 = pd.read_csv(Q.OUT / "c3_geometry.csv")
    circ = pd.read_csv(Q.OUT / "circularity_levels.csv")
    rows = []
    for g in KEY_GROUPS:
        a = ans["main"][ans["main"].group.eq(g)]
        if not len(a):
            continue
        a = a.iloc[0]
        r = dict(group=gname(g), units=int(a.n_units), answer_main=a.answer, H_T_falsified=bool(a.T_falsified))
        changed, tf_changed = [], []
        for lab, c in CONTROLS:
            x = ans[c]
            x = x[x.group.eq(g)] if x is not None else None
            if x is None or not len(x):
                changed.append(f"{c}: absent")
                continue
            x = x.iloc[0]
            if x.answer != a.answer:
                changed.append(f"{c}: {x.answer}")
            if c != "cb" and bool(x.T_falsified) != bool(a.T_falsified):
                tf_changed.append(f"{c}: {bool(x.T_falsified)}")
        var = []
        for lab, c in VARIANTS:
            x = ans[c]
            x = x[x.group.eq(g)] if x is not None else None
            var.append(f"{c}: {x.iloc[0].answer if x is not None and len(x) else 'absent'}")
        flags = []
        for L in (A.PRIMARY_T, A.PRIMARY_C):
            s = c3[c3.group.eq(g) & c3.level.eq(L)]
            fl = s.iloc[0].get("flagged", False) if len(s) else False
            if len(s) and pd.notna(fl) and str(fl) == "True":
                flags.append(f"C3 residual vs station distance ({L}, rho {s.iloc[0].rho:.2f})")
        cc = circ[circ.group.eq(g) & circ.level.eq("V7.5") & circ.status.eq("ok")]
        c5 = ""
        if len(cc):
            q = cc.iloc[0]
            c5 = (f"C5b {'pass' if q.C5b_pass else 'fail'} (rho {q.rho_real:.2f}, corrected {q.rho_corrected:.2f}, "
                  f"90 % low {q.rho_corrected_lo90:.2f}); C5c {'pass' if q.get('C5c_pass', False) else 'fail'} "
                  f"(null p95 {q.get('rho_null_p95', np.nan):.2f}; {q.get('C5c_status', '')})")
        r["answer_changes_under_controls"] = "; ".join(changed) if changed else "none"
        r["H_T_falsification_changes"] = "; ".join(tf_changed) if tf_changed else "none"
        r["C3_flags"] = "; ".join(flags) if flags else "none"
        r["C5_on_7.5"] = c5 or "not testable (< 5 units with a level)"
        r["variants"] = "; ".join(var)
        hc = a.answer in H_C_ANSWERS
        circ_ok = (not hc) or (len(cc) and bool(cc.iloc[0].C5b_pass) and bool(cc.iloc[0].get("C5c_pass", False)))
        r["answer_robust"] = bool(not changed and not flags and circ_ok)
        r["H_T_falsification_robust"] = bool(a.T_falsified and not tf_changed)
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(Q.OUT / "robustness.csv", index=False)
    return R


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def main():
    U = pd.read_csv(Q.OUT / "units_main.csv")
    G = pd.read_csv(Q.OUT / "groups_main.csv")
    PR = pd.read_csv(Q.OUT / "pref_main.csv")
    AN = pd.read_csv(Q.OUT / "answers_main.csv")
    TS = pd.read_csv(Q.OUT / "tstar_main.csv")
    C = pd.read_csv(Q.OUT / "circularity_levels.csv")
    C3 = pd.read_csv(Q.OUT / "c3_geometry.csv")
    COV = pd.read_csv(Q.OUT / "covariates_units.csv")
    MC = pd.read_csv(Q.OUT / "mc_shift.csv")
    WC = pd.read_csv(Q.OUT / "within_cluster.csv")
    meta = json.loads((Q.OUT / "run_meta.json").read_text())
    R = robustness()
    E = pd.read_parquet(Q.BIG / "events_q2_main.parquet", columns=["ID", "seg", "frame", "in_tomo"])

    def gv(g, L, c):
        s = G[G.group.eq(g) & G.level.eq(L)]
        return float(s.iloc[0][c]) if len(s) and c in s and pd.notna(s.iloc[0][c]) else np.nan

    GL = {c: load("groups", c) for c in ("lit_med", "lit_p90", "lit_gd", "lange", "lange_ours")}
    AL = {c: load("answers", c) for c in ("lit_med", "lit_p90", "lit_gd", "lange", "lange_ours")}

    def gl(cfg, g, L, c="bias"):
        t = GL.get(cfg)
        if t is None:
            return np.nan
        x = t[t.group.eq(g) & t.level.eq(L)]
        return float(x.iloc[0][c]) if len(x) and pd.notna(x.iloc[0][c]) else np.nan

    def need(g):                       # depth bias that would bring the 800 C half-space isotherm within 3 km
        return gv(g, "HSC_800", "bias") - Q.DELTA

    def nu(g, L=None):
        if L is None:
            x = AN[AN.group.eq(g)]
            return int(x.iloc[0].n_units) if len(x) else 0
        v = gv(g, L, "n_units")
        return int(v) if np.isfinite(v) else 0

    def gc(cfg, g, L, c="bias"):
        t = load("groups", cfg)
        if t is None:
            return np.nan
        x = t[t.group.eq(g) & t.level.eq(L)]
        if not len(x) or pd.isna(x.iloc[0][c]):
            return np.nan
        v = x.iloc[0][c]
        return v if isinstance(v, str) else float(v)

    def segu(k):
        return U[U.grouping.eq("seg") & U.key.eq(k)]

    def rng_s(v, f="{:.0f}"):
        v = pd.Series(v).dropna()
        return (f.format(v.min()) + "-" + f.format(v.max())) if len(v) else "n/a"

    CTRL = {"sigz2": "sig_z &lt;= 2 km", "nosig": "no sig_z cut", "v60": "catalog v6.0",
            "geom": "station within one focal depth", "notomo": "events not in the tomography"}

    def ht_fail(g):
        out = []
        for c in CTRL:
            t = load("answers", c)
            if t is None or g not in set(t.group):
                continue
            if not bool(t.set_index("group").loc[g, "T_falsified"]):
                out.append(c)
        return out

    def ht_detail(g, c):
        return f"{gc(c, g, 'HSC_800'):+.1f} km, {gc(c, g, 'HSC_800', 'abs_cat')}"

    def ht_ctrl_text(g):
        f = ht_fail(g)
        if not f:
            return "holds under every registered control"
        return ("holds under the registered controls except " +
                "; ".join(f"{CTRL[c]} (800 C half-space isotherm {ht_detail(g, c)})" for c in f))

    def nf6(g):
        k = sum(G[G.group.eq(g) & G.level.eq(L)].abs_cat.iloc[0] == "falsified" for L in A.ISO_KEYS)
        return "all" if k == 6 else str(k)

    nf6_tr = nf6("transform")

    def load_csv(name):
        f = Q.OUT / name
        return pd.read_csv(f) if f.exists() else None

    def bias_s(g, L):
        return f"{gv(g, L, 'bias'):+.1f} km (95 % {gv(g, L, 'bias_lo95'):+.1f} to {gv(g, L, 'bias_hi95'):+.1f})"

    def ts(g, mod):
        s = TS[TS.group.eq(g) & TS.model.eq(mod)]
        return float(s.iloc[0].T_med) if len(s) else np.nan

    def circ(g, c):
        s = C[C.group.eq(g) & C.level.eq("V7.5") & C.status.eq("ok")]
        return s.iloc[0][c] if len(s) and c in s else np.nan

    off = E[E.seg.isin(list(Q.OFFSHORE))]
    n_null = int(C.n_null.max()) if "n_null" in C else 0
    _gp = C[C.group.eq("Gorda slab (GS)") & C.level.eq("V7.5") & C.status.eq("ok")]
    if len(_gp) and pd.notna(_gp.iloc[0].get("rho_null_p95", np.nan)):
        _d = float(_gp.iloc[0].rho_real - _gp.iloc[0].rho_null_p95)
        gs_c5c = (f"{'exceeds' if _d > 0 else 'does not exceed'} the null 95th percentile "
                  f"({_gp.iloc[0].rho_null_p95:.2f}) by {abs(_d):.2f}" if _d > 0 else
                  f"does not exceed the null 95th percentile ({_gp.iloc[0].rho_null_p95:.2f})")
    else:
        gs_c5c = "was not scored"
    if n_null >= 15:
        c5c_state = (f"The null-truth control C5c is scored on {n_null} of 20 merged members, which meets the "
                     f"registered minimum of 15. A group whose own valid member count is below 15 stays "
                     f"provisional (C5c_status column).")
    else:
        c5c_state = (f"The null-truth control C5c is provisional: {n_null} of 20 members had merged, below the "
                     f"registered minimum of 15.")
    six = [f"{gv('frame:oceanic', L, 'bias'):+.1f}" for L in A.ISO_KEYS]
    gs = U[U.grouping.eq("seg") & U.key.eq("GS")].copy()
    gs["below_top"] = gs.D - gs.slab_km
    gs_near = gs[gs.slab_km < 13]
    gs_far = gs[gs.slab_km >= 20]
    css = """
:root{--bg:#ffffff;--fg:#1d1d1f;--mut:#5f6368;--line:#d9d9de;--acc:#0b4f6c;--box:#f4f6f8;--warn:#7a4b00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#e8e8ea;--mut:#a5a8ad;
 --line:#34373c;--acc:#7fc4e0;--box:#1e2125;--warn:#f0b35a}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
 max-width:1060px;margin:0 auto;padding:16px}
h1{font-size:1.55em;margin:.2em 0}h2{font-size:1.2em;margin-top:1.8em;border-bottom:1px solid var(--line)}
h3{font-size:1.02em;margin-top:1.3em}.mut{color:var(--mut)}.box{background:var(--box);padding:10px 14px;
 border-radius:6px}.warn{color:var(--warn)}table.t{border-collapse:collapse;font-size:12.5px;margin:8px 0;
 display:block;overflow-x:auto}table.t th,table.t td{border:1px solid var(--line);padding:3px 6px;text-align:left;
 vertical-align:top}table.t th{background:var(--box)}img{max-width:100%;background:#fff;border-radius:4px}
code{font-size:.92em}li{margin:.25em 0}
"""
    H = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Q2 depth limit</title><style>{css}</style></head><body>"]
    H.append("<h1>Q2: what limits the depth of earthquakes in the offshore plate system?</h1>")
    H.append(f"<p class='mut'>Report for D-B115, catalog v6.1, model v0.4.2. Written 2026-11-27 (times from "
             f"<code>date</code>). Branch science/q2-depth-limit, code commit {git_head()}. Pre-registration "
             f"commit e6bab1ac, 2026-11-27 13:27:22 PST.</p>")
    # ---------------------------------------------------------------- summary
    # ---------------------------------------------------------------- Monte Carlo precision (seed study)
    MCP = load_csv("mc_precision.csv")
    MCA = load_csv("mc_precision_answers.csv")
    seed_var = []                       # groups whose registered answer changes across seeds
    if MCA is not None and len(MCA):
        for g, x in MCA.groupby("group"):
            if x.answer.nunique() > 1 and g in KEY_GROUPS:
                n = int(x.n_seeds.sum())
                seed_var.append((g, x.registered_answer.iloc[0],
                                 "; ".join(f"'{r.answer}' in {int(r.n_seeds)} of {n} seeds"
                                           for _, r in x.sort_values("n_seeds", ascending=False).iterrows())))
    straddle = MCP[MCP.straddles] if MCP is not None and len(MCP) else None
    H.append("<h2>Summary</h2><div class='box'><ul>")
    H.append(f"<li>In the young oceanic plates (Gorda plate, Blanco and Mendocino transforms, triple junction), the "
             f"base of seismicity (D90) lies {min(gv(g, 'HSC_600', 'bias') for g in ('seg:BL', 'seg:MFZ', 'intraplate', 'triple junction')):.0f} to "
             f"{max(gv(g, 'HSC_600', 'bias') for g in ('seg:BL', 'seg:MFZ', 'intraplate', 'triple junction')):.0f} km below the 600 C isotherm of conductive cooling models of "
             f"their plate age. In the segment-independent oceanic frame ({nu('frame:oceanic')} bins of 0.5 deg: every bin west of the "
             f"trench, ridges included), the bias against the six "
             f"isotherms (350, 600, 800 C; half-space and GDH1) is {', '.join(six)} km. The cooling models put the "
             f"base at about {ts('frame:oceanic', 'HSC'):.0f} C (half-space) or {ts('frame:oceanic', 'GDH1'):.0f} C "
             f"(GDH1). In the oceanic frame this rejection {ht_ctrl_text('frame:oceanic')}. In the Gorda plate "
             f"interior it {ht_ctrl_text('intraplate')}; in the MFZ it {ht_ctrl_text('seg:MFZ')}. It also holds "
             f"with the external-data isotherms, but not under every literature depth offset (next point).</li>")
    H.append(f"<li><b>Our depths run deep against local OBS catalogs, and the headline survives only in a narrower "
             f"form</b> (post-registration control L).<ul>"
             f"<li>Median paired offsets (Blanco segments; +3.17 km elsewhere): the oceanic frame still rejects all "
             f"six isotherms (bias {gl('lit_med', 'frame:oceanic', 'HSC_800'):+.1f} km for 800 C), and so do the "
             f"Gorda plate ({gl('lit_med', 'intraplate', 'HSC_800'):+.1f} km) and the MFZ "
             f"({gl('lit_med', 'seg:MFZ', 'HSC_800'):+.1f} km). The transforms as a group no longer do "
             f"(800 C {gl('lit_med', 'transform', 'HSC_800'):+.1f} km).</li>"
             f"<li>Blanco with Lange's own depths for the same events: the base sits at the 600-800 C isotherms "
             f"({gl('lange', 'seg:BL', 'HSC_600'):+.1f} and {gl('lange', 'seg:BL', 'HSC_800'):+.1f} km; our depths "
             f"for those events give {gl('lange_ours', 'seg:BL', 'HSC_600'):+.1f} and "
             f"{gl('lange_ours', 'seg:BL', 'HSC_800'):+.1f} km). At Blanco the deep base is a depth bias.</li>"
             f"<li>Worst cases: with the p90 offsets, or the Gorda Depression offset (+9.6 km, 23 events) moved to "
             f"the Mendocino-Gorda system, the 800 C isotherm fits the oceanic frame "
             f"({gl('lit_p90', 'frame:oceanic', 'HSC_800'):+.1f} and {gl('lit_gd', 'frame:oceanic', 'HSC_800'):+.1f} "
             f"km).</li>"
             f"<li>What survives: in the Gorda plate interior and the MFZ the base would reach the 800 C isotherm "
             f"only if our depths were {need('intraplate'):.0f}-{need('seg:MFZ'):.0f} km too deep. No paired OBS "
             f"catalog tests that there.</li></ul></li>")
    H.append(f"<li>In the same plates the base lies {gv('frame:oceanic', 'V7.5', 'bias'):.1f} km below the 7.5 km/s "
             f"level of v0.4.2 ({nu('frame:oceanic', 'V7.5')} bins). In the {nu('frame:oceanic', 'V8.0')} bins where the 8.0 km/s level is resolved, the base is "
             f"{-gv('frame:oceanic', 'V8.0', 'bias'):.1f} km above it (within 3 km in "
             f"{100 * gv('frame:oceanic', 'V8.0', 'P_sit'):.0f} % of draws). The base therefore lies between the two "
             f"levels. Deeper bases go with deeper "
             f"levels (cross-validated skill {gv('frame:oceanic', 'V7.5', 'skill_med'):.2f}). Three things limit "
             f"this result:"
             f"<ul><li>The relation is mostly the contrast between Blanco (shallow base, shallow level) and the "
             f"Mendocino-Gorda system (deep base, deep level).</li>"
             f"<li>The residuals grow with the distance to the nearest station (C3 flag).</li>"
             f"<li>No offshore 7.5 km/s crossing lies between checkerboard-resolved nodes (C4).</li></ul>"
             f"The velocity link is therefore reported, not claimed as robust. With the median literature depth "
             f"offsets the base sits at the 7.5 km/s level instead "
             f"({gl('lit_med', 'frame:oceanic', 'V7.5'):+.1f} km).</li>")
    H.append(f"<li>In the subducting Gorda slab, the base lies between two 600 C isotherms:"
             f"<ul><li>{gv('Gorda slab (GS)', 'HSC_600', 'bias'):+.1f} km from the coldest-slab bound (the incoming "
             f"plate's geotherm hung below Slab2);</li>"
             f"<li>{gv('Gorda slab (GS)', 'HSC_600_warmtop', 'bias'):+.1f} km from a labelled variant that uses the "
             f"slab-surface temperature of the van Keken &amp; Wilson (2023) models.</li></ul>"
             f"Both are consistent with a limit near 600-700 C. Neither the isotherm nor the 7.5 km/s level passes "
             f"every registered test of the down-dip trend. The Gorda-slab correlation with the 7.5 km/s level "
             f"(rho {circ('Gorda slab (GS)', 'rho_real'):.2f}) fails the block-bootstrap test (C5b; 90 % lower "
             f"bound {circ('Gorda slab (GS)', 'rho_real_lo90'):.2f} from {int(circ('Gorda slab (GS)', 'n_blocks'))} blocks). Against the null-truth members "
             f"(C5c, {n_null} of 20) it {gs_c5c}.</li>")
    H.append("<li>Ridges are not answered:"
             "<ul><li>Axial depths carry the SSST bias.</li>"
             "<li>The cooling models do not describe the axial magma system.</li>"
             f"<li>Gorda Ridge depths (D90 {rng_s(segu('GR').D_bsf)} km below the seafloor at near-zero age) are not trusted.</li>"
             "<li>No ridge bin has a resolved velocity level.</li></ul></li>")
    if seed_var:
        H.append("<li><b>Monte Carlo precision</b> (20 independent seeds at the registered 1000 draws; section 5a): "
                 + " ".join(f"The {gname(g)} answer is at a decision threshold within Monte Carlo precision: "
                            f"{txt}. The registered seed gives '{reg}'; this report gives both readings and does not "
                            f"let one seed choose." for g, reg, txt in seed_var) + "</li>")
    elif MCP is not None:
        H.append("<li><b>Monte Carlo precision</b> (20 independent seeds at the registered 1000 draws; section 5a): "
                 "no registered answer changes between seeds.</li>")
    H.append("<li>Answer to Q2 for the paper:<ul>"
             "<li>At Blanco, with independent OBS depths, the base is consistent with a 600-800 C limit.</li>"
             f"<li>In the Mendocino-Gorda system (Gorda plate interior, MFZ), the base lies "
             f"{gv('intraplate', 'HSC_800', 'bias'):.0f}-{gv('seg:MFZ', 'HSC_800', 'bias'):.0f} km below the 800 C "
             f"conductive isotherm. It is not a 350-800 C limit unless our depths there are "
             f"{need('intraplate'):.0f}-{need('seg:MFZ'):.0f} km too deep. No local catalog exists there to check, "
             f"and the nearest paired segment (Gorda Depression) has an offset of that size.</li>"
             "<li>A composition (velocity-level) control is not supported robustly.</li></ul></li>")
    H.append("</ul></div>")
    # ---------------------------------------------------------------- question
    H.append("<h2>1. Question</h2><p>What limits the depth of earthquakes across the offshore plate system: ridges, "
             "transforms, the Gorda and Juan de Fuca plate interiors, the Mendocino triple junction and the subducting "
             "slab? Temperature (an isotherm of a plate cooling model) or composition (a velocity level of the "
             "tomography model, used as a Moho proxy)?</p>")
    H.append("<p>Known before this work (plate-boundary analysis on v6.1):"
             "<ul><li>the D90 of Gorda-slab columns followed the 7.5 km/s level (rho 0.53, p 0.001, 67 columns, tiers "
             "A-C), weakly at Blanco (rho 0.12) and not at the MFZ;</li>"
             "<li>Gorda-plate events reach about 26 km (p90) in lithosphere younger than about 5 Myr;</li>"
             "<li>transforms are deeper than ridges.</li></ul>"
             "The external-data job also printed exploratory numbers, which were seen before this report was "
             "written: 95-96 % of the deep GP and MFZ events lie below the half-space 600 C isotherm. They are not "
             "tests of this report.</p>")
    # ---------------------------------------------------------------- pre-registration
    H.append("<h2>2. Pre-registered tests</h2>")
    H.append("<p>Written and committed at <b>2026-11-27 13:27:22 PST</b> (commit e6bab1ac, "
             "data/paper_plan/q2_depth_limit/RESUME.md), before any depth was compared with an isotherm or a level. "
             "Summary:</p><ul>"
             "<li><b>H_T</b>: the base follows an isotherm, 600 C (primary), 350 C, 800 C. Two cooling models:"
             "<ul><li>half-space: Tm 1350 C, kappa 1e-6 m2/s;</li>"
             "<li>GDH1 plate: 95 km, Tm 1450 C, kappa 8.04e-7 m2/s.</li></ul>"
             "Plate age from Seton et al. (2020). The thermal top is the seafloor.<ul>"
             "<li>Transforms: the mean of the two geotherms, with ages 20 km either side of the trace.</li>"
             "<li>Slab: the incoming-plate geotherm hung below Slab2, an upper bound on isotherm depth.</li></ul></li>"
             "<li><b>H_C</b>: the base follows the 7.5 km/s (primary) or 8.0 km/s level of v0.4.2, on the "
             "resolved mask (rock and (checkerboard or DWS &gt;= 100)). The level spread is the raw member levels "
             "about their median x1.50, applied once.</li>"
             "<li><b>Events</b>: tier A-B, depth_quality ok, sig_z &lt;= 3 km, not floor_limited, far_only, "
             "deep_pass or five_phase, earthquakes only.<ul>"
             "<li>Units: 0.5 deg bins with at least 20 events (decisions); model columns are a variant.</li>"
             "<li>D90 from 1000 draws: event bootstrap plus sig_z Monte Carlo.</li></ul></li>"
             "<li><b>Rules</b> (delta = 3 km):<ul>"
             "<li><i>Sits at the level</i> if |median residual| &lt;= 3 km in at least 50 % of draws; "
             "<i>falsified</i> if &gt; 3 km in at least 95 %.</li>"
             "<li><i>Follows</i> if the leave-one-1-deg-block-out cross-validated skill is &gt; 0 in at least "
             "90 % of draws. Testable only when the predictor spans at least 3 km over at least 5 units.</li>"
             "<li><i>Preferred</i> if P(RMS_T &lt; RMS_C) &gt;= 0.9 or &lt;= 0.1.</li>"
             "<li>H_T falsified if all six isotherms are falsified, or the primary isotherm does not follow; "
             "likewise for H_C with the two levels.</li></ul></li>"
             "<li><b>Controls</b>:<ul>"
             "<li>C1 depth quality: sig_z &lt;= 2 km, and no sig_z cut.</li>"
             "<li>C2 catalog v6.0.</li>"
             "<li>C3 station geometry: events with the nearest station within one focal depth, and the correlation "
             "of the residual with the station distance.</li>"
             "<li>C4 checkerboard-only mask.</li>"
             "<li>C5 circularity: (a) events not in the tomoDD data; (b) apparent following in the 100 synthetic "
             "members; (c) the D-B112 null-truth members.</li></ul>"
             "A result that changes category under a control is labelled and not claimed as robust.</li></ul>")
    H.append("<p><b>Changes after registration</b>, all logged in RESUME.md. None of them changes a rule.<ul>"
             "<li>(1) 13:30: the x1.50 spread applies to raw member levels, not to the stored, already inflated "
             "spread. This is a clarification.</li>"
             "<li>(2) 13:34, during a 50-draw code test whose answers were seen: slab events south of the "
             "triple junction had their incoming age sampled where the age grid mixes the Gorda and Pacific sides "
             "(12.8 Myr). It is now sampled 0.2 deg north of the junction (Gorda plate, 4.9 Myr).</li>"
             "<li>(3) Where one transform probe falls on land, the other side is used for both.</li>"
             "<li>(4) D-B113 axes (first run, 13:23 segment table md5 76542c2e): moved 1,046 GP events to GR and "
             "228 JP events to JR. This was an input, not a test choice.</li>"
             "<li>(5) Labelled variants from the D-B114 external-data job, added after the registration and "
             "never primary:<ul><li>its half-space isotherms (Tm 1300 C, epicentral age);</li>"
             "<li>a warm-top slab variant using the van Keken &amp; Wilson (2023) D80 slab-surface temperature;</li>"
             "<li>the CRUST1.0 Moho.</li></ul>"
             "The registered outputs were checked identical before and after adding them.</li>"
             "<li>(6) Reading of C5b: the block-bootstrap lower bound &gt; 0 is required in every case, with a "
             "correction of zero when not flagged. This is the stricter of the two readings of the registered "
             "sentence.</li>"
             "<li>(7) D-B117, plate boundaries v2 (PI 19:09 PST; analysis/plate-boundaries 0904802e): the segment "
             f"join now uses the v2 segment table (md5 {meta['main']['segments_md5'][:8]}, written "
             f"{meta['main']['segments_mtime']}). Recomputing the segments with PB_AXES=v2 reproduces it exactly. "
             "Against the D-B113 table it moves 727 tier A-C events (GR to MFZ 344, GP to MFZ 186, JP to JR 74, "
             "GP to GR 67, JR to JP 35, others below 11). The Mendocino and Blanco transform traces used for the "
             "two-sided ages are the v2 traces. Everything was rerun with the same rules (section 5c compares the "
             "runs). This is an input, not a test choice.</li>"
             "<li>(8) Control L (literature depth offsets), requested after the first report; rule written before "
             "it was computed (section 5b).</li></ul></p>")
    # ---------------------------------------------------------------- data
    H.append("<h2>3. Data</h2>")
    dat = pd.DataFrame([
        ("Catalog v6.1", str(Q.CATS["v6.1"]["csv"]), Q.md5(Q.CATS["v6.1"]["csv"])[:8]),
        ("Segment table v6.1 (plate boundaries v2, D-B117)", meta["main"]["segments"], meta["main"]["segments_md5"][:8]),
        ("Catalog v6.0 (C2)", str(Q.CATS["v6.0"]["csv"]), "see run_meta.json"),
        ("Model v0.4.2", str(A.P.MODEL), Q.md5(A.P.MODEL)[:8]),
        ("Ensemble v0.4.2+ens.100 (raw member levels)", "/wd1/mdenolle_data/paper_plan/v6.1/level_depth_members.npz",
         Q.md5(Q.WD1 / "paper_plan/v6.1/level_depth_members.npz")[:8]),
        ("Plate age, Seton et al. 2020 (1')", str(Q.AGE_GRID), "006070e0"),
        ("Sediment, GlobSed v3", str(Q.SED_GRID), "fetch_log.tsv"),
        ("Heat flow, GHFDB R2024", str(Q.HF_TABLE), "b16bd8d7"),
        ("Slab2 (Hayes et al. 2018)", str(A.P.SLAB), "-"),
        ("Slab-surface T (van Keken & Wilson 2023, D80), CRUST1.0, external isotherms", str(A.EXT_EVENTS),
         Q.md5(A.EXT_EVENTS)[:8] if A.EXT_EVENTS.exists() else "-"),
        ("tomoDD event lists (C5a)", ", ".join(Q.TOMODD_SETS), "inputs/events.csv"),
        ("Synthetic and null-truth members (C5b, C5c)", f"{Q.SYNTH}; {Q.NULL}", f"100; {n_null} merged"),
    ], columns=["input", "path", "md5 / note"])
    H.append(tbl(dat))
    H.append(f"<p>Depth-reliable events: {meta['main']['n_events']:,} in total, {len(off):,} in the offshore segments "
             f"({(off.in_tomo.mean() * 100):.0f} % of them were inverted in the tomography). "
             f"Frames: oceanic {int((E.frame == 'oceanic').sum()):,}, slab {int((E.frame == 'slab').sum()):,}, "
             f"upper plate (not tested) {int((E.frame == 'upper').sum()):,}. Units (0.5 deg, at least 20 events): "
             f"{int((U.grouping == 'seg').sum())} segment units and {int((U.grouping == 'frame').sum())} frame units. "
             f"Few units for GP ({int(((U.grouping == 'seg') & (U.key == 'GP')).sum())}), JP (0), ridges other "
             f"than Axial ({len(segu('GR')) + len(segu('JR')) + len(segu('EN')) + len(segu('EXR'))}, GR {len(segu('GR'))}): "
             f"the depth-reliable set keeps {int((off.seg == 'GP').sum())} GP events. The 'biased_deep' flag (Gorda, "
             f"nearest station &gt; 15 km) removes most of them.</p>")
    # ---------------------------------------------------------------- results
    H.append("<h2>4. Results</h2>")
    H.append("<h3>4.1 Where the base sits relative to each hypothesis</h3>")
    H.append(img("q2_map_residuals.png", "maps of residuals"))
    H.append("<p class='mut'>Frame units (segment-independent): D90 minus the 600 C half-space isotherm, minus the "
             "7.5 km/s level and minus the 8.0 km/s level. Red: seismicity deeper than the level. Hatched: slab "
             "frame. Dotted: no resolved level. Colour map: vik (Crameri), diverging, colour-vision safe.</p>")
    H.append(img("q2_age_depth.png", "depth against plate age"))
    H.append(f"<p class='mut'>Depth below the seafloor against plate age with the 350/600/800 C isotherms. "
             f"Grey: depth-reliable events. Symbols: unit D90 with the 95 % interval of the draws. The location "
             f"Monte Carlo shifts the D90 draws deeper by a median {MC['shift'].median():.2f} km (90th percentile "
             f"{MC['shift'].quantile(0.9):.2f} km) relative to the observed D90. That shift is the price of adding "
             f"noise to noisy depths, so the central values are the observed D90.</p>")
    H.append("<p>Fact by fact:<ul>"
             f"<li><b>Transforms</b> ({nu('transform')} units): the base is {bias_s('transform', 'HSC_600')} below the 600 C "
             f"half-space isotherm; {'all six isotherm variants are' if nf6_tr == 'all' else nf6_tr + ' of the six isotherm variants are'} "
             f"falsified. The temperature at the base is "
             f"{ts('transform', 'HSC'):.0f} C (half-space). Old-side-only variant: "
             f"{gv('transform', 'HSC_600_old', 'bias'):+.1f} km. Blanco alone: "
             f"{gv('seg:BL', 'HSC_600', 'bias'):+.1f} km. MFZ alone: {gv('seg:MFZ', 'HSC_600', 'bias'):+.1f} km.</li>"
             f"<li><b>Gorda plate interior</b> ({nu('intraplate')} units): {bias_s('intraplate', 'HSC_600')} below the 600 C "
             f"isotherm, at {ts('intraplate', 'HSC'):.0f} C.</li>"
             f"<li><b>Triple junction</b> ({nu('triple junction')} units): {bias_s('triple junction', 'HSC_600')}, at "
             f"{ts('triple junction', 'HSC'):.0f} C. Neither hypothesis fits: the 7.5 km/s level is "
             f"{gv('triple junction', 'V7.5', 'bias'):+.1f} km off and does not follow.</li>"
             f"<li><b>Oceanic frame</b> ({nu('frame:oceanic')} units, segment-independent): {bias_s('frame:oceanic', 'HSC_600')} for "
             f"600 C; {bias_s('frame:oceanic', 'HSC_800')} for 800 C; {bias_s('frame:oceanic', 'V7.5')} for "
             f"7.5 km/s ({nu('frame:oceanic', 'V7.5')} units); {bias_s('frame:oceanic', 'V8.0')} for 8.0 km/s ({nu('frame:oceanic', 'V8.0')} units).</li>"
             f"<li><b>Gorda slab</b> ({nu('Gorda slab (GS)')} units):<ul>"
             f"<li>coldest-slab 600 C: {bias_s('Gorda slab (GS)', 'HSC_600')};</li>"
             f"<li>GDH1 600 C: {bias_s('Gorda slab (GS)', 'GDH1_600')};</li>"
             f"<li>warm-top variant (labelled): {gv('Gorda slab (GS)', 'HSC_600_warmtop', 'bias'):+.1f} km;</li>"
             f"<li>7.5 km/s: {bias_s('Gorda slab (GS)', 'V7.5')} ({nu('Gorda slab (GS)', 'V7.5')} units).</li></ul>"
             f"Below the Slab2 top, the base reaches {gs_near.below_top.min():.0f}-{gs_near.below_top.max():.0f} km "
             f"in the bins where the slab is shallower than 13 km, near the deformation front, and "
             f"{gs_far.below_top.min():.0f}-{gs_far.below_top.max():.0f} km where the slab is deeper than 20 km. "
             f"This down-dip pattern is descriptive and was not registered.</li>"
             f"<li><b>Axial</b> ({nu('seg:AX')} units): the base is within 3 km of all six isotherms (bias "
             f"{gv('seg:AX', 'HSC_600', 'bias'):+.1f} km for 600 C). The isotherms there are 2-3 km below sea level "
             f"at ages of {rng_s(segu('AX').age, '{:.2f}')} Myr, so this is not a test of a thermal limit.</li>"
             f"<li><b>Gorda Ridge</b> ({nu('seg:GR')} units): D90 {rng_s(segu('GR').D_bsf)} km below the seafloor at near-zero age, "
             f"{gv('seg:GR', 'HSC_600', 'bias'):+.1f} km below 600 C. Not interpreted (see limitations).</li></ul></p>")
    H.append(img("q2_scatter.png", "D90 against predictors"))
    H.append(img("q2_forest.png", "bias per group"))
    H.append("<p class='mut'>Bias per group and predictor: the median over units of D90 minus the level, with the "
             "95 % interval of the draws. Grey band: the registered +-3 km tolerance. Right column: number of "
             "units.</p>")
    # decisions table
    show = AN[AN.group.isin(KEY_GROUPS)][["group", "n_units", "n_units_T", "n_units_C", "T_sits", "C_sits",
                                          "T_follows", "C_follows", "T_falsified", "C_falsified", "answer"]].copy()
    show["group"] = show.group.map(gname)
    H.append("<h3>4.2 Registered decisions (main configuration)</h3>")
    H.append(tbl(show.fillna("")))
    for g, reg, txt in seed_var:
        H.append(f"<p class='warn'>{gname(g)}: the registered seed gives '{reg}', but this answer sits at a decision "
                 f"threshold within Monte Carlo precision ({txt}; section 5a). Both readings stand.</p>")
    pr = PR[PR.group.isin(KEY_GROUPS)][["group", "n_units_both", "primary_P_T", "primary_rms_T", "primary_rms_C",
                                        "primary_pref", "best_of_family_P_T", "best_of_family_pref", "cv_P_T",
                                        "cv_pref"]].copy()
    pr["group"] = pr.group.map(gname)
    H.append("<p>Preference tests, on units that have both predictors. P_T is the share of draws with a smaller "
             "misfit for H_T. Primary pair: 600 C half-space against 7.5 km/s. The cv columns compare the "
             "cross-validated misfit.</p>")
    H.append(tbl(pr.dropna(subset=["primary_P_T"])))
    gsel = G[G.group.isin(["frame:oceanic", "frame:slab", "transform", "intraplate", "triple junction",
                           "Gorda slab (GS)", "seg:BL", "seg:MFZ", "seg:AX", "seg:GR"])
             & G.level.isin(A.ISO_KEYS + A.LEV_KEYS + A.EXT_VARIANTS + A.ISO_EXTRA)][
        ["group", "level", "n_units", "bias", "bias_lo95", "bias_hi95", "rms", "abs_cat", "span", "rho", "skill_med",
         "P_skill_pos", "shape_cat"]].copy()
    gsel["group"] = gsel.group.map(gname)
    H.append("<details><summary>All group x predictor metrics (registered predictors and labelled variants)"
             "</summary>" + tbl(gsel[gsel.n_units > 0]) + "</details>")
    us = U[U.grouping.eq("seg")].copy()
    us["type"] = us.key.map(Q.OFFSHORE)
    vbm = us.groupby("type").Vbase.median().dropna()
    vb = ", ".join(f"{k} {v:.2f} km/s" for k, v in vbm.items())
    H.append("<h3>4.3 Temperature and velocity at the base</h3>")
    H.append(img("q2_base_T_V.png", "temperature and Vp at the base"))
    H.append(f"<p>The cooling models put the base at {ts('transform', 'HSC'):.0f} C under the transforms, "
             f"{ts('intraplate', 'HSC'):.0f} C under the Gorda plate and {ts('triple junction', 'HSC'):.0f} C at the "
             f"triple junction (half-space; GDH1 is "
             f"{rng_s([ts(g, 'GDH1') - ts(g, 'HSC') for g in ('transform', 'intraplate', 'triple junction')])} C hotter). All are outside 350-800 C. In the Gorda slab "
             f"the coldest-slab bound gives {ts('Gorda slab (GS)', 'HSC'):.0f} C, a lower bound on the temperature; "
             f"the warm-top variant gives "
             f"{np.nanmedian(U[U.grouping.eq('seg') & U.key.eq('GS')].Tbase_HSC_warmtop):.0f} C. "
             f"The v0.4.2 Vp at the base (median per boundary type) is {vb}. That is the uppermost mantle, or the "
             f"crust-mantle transition, of the model.</p>")
    H.append("<h3>4.4 Covariates (reported, not in the decisions)</h3>")
    cov = COV[COV.grouping.eq("seg") & COV.q_n.gt(0)][["key", "lon", "lat", "D_bsf", "q_mWm2", "q_n",
                                                     "z600_linear_bsf", "sed_km"]]
    cvm = cov[~cov.key.isin(["GR", "AX"])]
    H.append(f"<p>Heat flow (GHFDB R2024) lies within 25 km of only {len(cov)} segment units. One is on the Gorda "
             f"Ridge axis ({cov[cov.key.eq('GR')].q_mWm2.max():.0f} mW/m2, a hydrothermal value, not used below). "
             f"The other {len(cvm)} are near the margin or the triple junction, where heat flow is "
             f"{cvm.q_mWm2.min():.0f}-{cvm.q_mWm2.max():.0f} mW/m2. That is far below the 200 mW/m2 or more that half-space "
             f"cooling predicts for 5 Myr crust. A linear conductive geotherm from it puts 600 C at "
             f"{cvm.z600_linear_bsf.min():.0f}-{cvm.z600_linear_bsf.max():.0f} km below the seafloor, close to or below the observed base. This is the direction a colder-than-conductive "
             f"lithosphere would give. Shallow hydrothermal circulation also lowers measured heat flow on young "
             f"crust, so the numbers do not separate the two. Sediment (GlobSed) is 0.0-1.0 km. Measuring the "
             f"isotherms from the basement moves them by less than 1 km "
             f"(oceanic-frame bias {gv('frame:oceanic', 'HSC_600_basement', 'bias'):+.1f} km).</p>")
    H.append(tbl(cov, fmt={"lon": "{:.2f}", "lat": "{:.2f}"}))
    H.append("<p>Labelled external-data variants (D-B114 job), main configuration:<ul>"
             f"<li>Half-space isotherms with Tm 1300 C and epicentral age: oceanic-frame bias "
             f"{gv('frame:oceanic', 'EXT_HSC1300_600', 'bias'):+.1f} km.</li>"
             f"<li>CRUST1.0 Moho, which does not come from the tomography: oceanic frame "
             f"{gv('frame:oceanic', 'CRUST1_moho', 'bias'):+.1f} km, Gorda slab "
             f"{gv('Gorda slab (GS)', 'CRUST1_moho', 'bias'):+.1f} km. CRUST1.0 (1 deg) does not resolve variation between bins (span "
             f"{gv('frame:oceanic', 'CRUST1_moho', 'span'):.1f} km).</li>"
             f"<li>Slab warm-top 600 C: slab frame {gv('frame:slab', 'HSC_600_warmtop', 'bias'):+.1f} km, Gorda "
             f"slab {gv('Gorda slab (GS)', 'HSC_600_warmtop', 'bias'):+.1f} km.</li></ul></p>")
    # ---------------------------------------------------------------- controls
    H.append("<h2>5. Controls</h2>")
    H.append(img("q2_controls.png", "answers under controls"))
    rr = R[["group", "units", "answer_main", "answer_changes_under_controls", "H_T_falsified",
            "H_T_falsification_changes", "C3_flags", "C5_on_7.5", "answer_robust", "H_T_falsification_robust"]]
    H.append(tbl(rr))
    H.append("<p>Reading the table:<ul>"
             f"<li><b>C4 (mask)</b>: with the checkerboard-only mask, no offshore unit has a 7.5 or 8.0 km/s "
             f"crossing between two resolved nodes. Every velocity level used here comes from the DWS &gt;= 100 part "
             f"of the mask, so every H_C result is mask-sensitive.</li>"
             f"<li><b>C3 (geometry)</b>: in the oceanic frame the residual grows with the distance to the nearest "
             f"station, for the isotherm (rho {C3[(C3.group == 'frame:oceanic') & (C3.level == 'HSC_600')].rho.iloc[0]:.2f}) "
             f"and for the 7.5 km/s level (rho {C3[(C3.group == 'frame:oceanic') & (C3.level == 'V7.5')].rho.iloc[0]:.2f}). "
             f"The Mendocino-Gorda bins are both the deepest and the farthest from stations. Restricting to events "
             f"with a station within one focal depth (geom) gives an isotherm bias of "
             f"{gc('geom', 'frame:oceanic', 'HSC_600'):+.1f} km (600 C) and a 7.5 km/s bias of "
             f"{gc('geom', 'frame:oceanic', 'V7.5'):+.1f} km.</li>"
             f"<li><b>C1, C2</b>: the H_T falsification in the oceanic frame {ht_ctrl_text('frame:oceanic')}. For "
             f"the MFZ it {ht_ctrl_text('seg:MFZ')}, and for the Gorda plate interior it "
             f"{ht_ctrl_text('intraplate')}.</li>"
             f"<li><b>C5a</b>: without the inverted events ({int(gc('notomo', 'frame:oceanic', 'HSC_600', 'n_units'))} oceanic units, "
             f"{int(gc('notomo', 'frame:oceanic', 'V7.5', 'n_units'))} with a level), the 7.5 km/s bias is "
             f"{gc('notomo', 'frame:oceanic', 'V7.5'):+.1f} km; shape: {gc('notomo', 'frame:oceanic', 'V7.5', 'shape_cat')}. The "
             f"base still follows the level.</li></ul></p>")
    H.append("<h3>Circularity of the velocity levels (C5b, C5c)</h3>")
    H.append(img("q2_circularity.png", "circularity"))
    cc = C[C.status.eq("ok")][["group", "level", "n_units", "rho_real", "rho_real_lo90", "rho_truth", "A_median",
                              "A_share_pos", "C5b_flag", "rho_corrected", "rho_corrected_lo90", "C5b_pass",
                              "rho_null_truth", "rho_null_median", "rho_null_p95", "n_null_valid", "C5c_pass",
                              "C5c_status"]].copy()
    cc["group"] = cc.group.map(gname)
    H.append(tbl(cc))
    H.append(f"<p>The synthetic members (truth = v0.4.2 perturbed) recover levels "
             f"{rng_s(-C[C.group.isin(['frame:oceanic', 'transform', 'seg:BL', 'seg:MFZ']) & C.status.eq('ok')].synth_rec_bias_km, '{:.1f}')} km shallower than the "
             f"truth in the oceanic units. In the oceanic frame they increase the D90-level correlation by "
             f"{circ('frame:oceanic', 'A_median'):.2f} ({100 * circ('frame:oceanic', 'A_share_pos'):.0f} % of members positive). The correction leaves rho "
             f"{circ('frame:oceanic', 'rho_corrected'):.2f}, with a 90 % lower bound of "
             f"{circ('frame:oceanic', 'rho_corrected_lo90'):.2f}.</p>"
             f"<p>The null-truth members ({n_null} of 20 merged at the time of this report) give rho "
             f"{circ('frame:oceanic', 'rho_null_median'):.2f} (median) to "
             f"{circ('frame:oceanic', 'rho_null_p95'):.2f} (95th percentile) in the oceanic frame, against the real "
             f"{circ('frame:oceanic', 'rho_real'):.2f}. The null truth flattens structure only within each segment's "
             f"zone, so the contrast between the Blanco and Mendocino-Gorda zones survives in it (null-truth rho "
             f"{circ('frame:oceanic', 'rho_null_truth'):.2f}). The real correlation is therefore hardly larger than "
             f"what the zone means plus the ray coverage produce.</p>"
             f"<p>For the Gorda slab the real rho {circ('Gorda slab (GS)', 'rho_real'):.2f} {gs_c5c}. Its 4 "
             f"one-degree blocks give a "
             f"block-bootstrap lower bound of {circ('Gorda slab (GS)', 'rho_real_lo90'):.2f}. The earlier column "
             f"result (rho 0.53, p 0.001) treated 67 columns as independent at a 40 km model resolution. With "
             f"0.5 deg bins and block resampling it is not significant. {c5c_state} "
             f"To rescore: <code>bash workflow/11_paper_plan/run_q2.sh circ</code>.</p>")
    H.append("<h3>Not registered: within-cluster correlations</h3>")
    H.append(tbl(WC))
    H.append("<p>Within the Blanco bins and within the Mendocino-Gorda bins, D90 correlates with each predictor at "
             f"rho {WC[~WC.cluster.str.startswith('oceanic')].rho.min():.2f}-"
             f"{WC[~WC.cluster.str.startswith('oceanic')].rho.max():.2f} ({rng_s(WC[~WC.cluster.str.startswith('oceanic')].n_units)} units). "
             f"For the 7.5 km/s level the within-cluster rho is "
             f"{' and '.join(f'{v:.2f}' for v in WC[~WC.cluster.str.startswith('oceanic') & WC.level.eq('V7.5')].rho)} "
             f"(Blanco, Mendocino-Gorda), against "
             f"{circ('frame:oceanic', 'rho_real'):.2f} for the whole oceanic frame: most of that correlation comes "
             f"from the difference between the two clusters.</p>")
    # ---------------------------------------------------------------- control L
    if MCP is not None:
        H.append("<h3>5a. Monte Carlo precision of the registered probabilities</h3>")
        H.append("<p>Coordinator check after the final report; no rule changed. The slab-frame answer changed between "
                 "the D-B113 and v2 runs only through the draw sequence, so the main configuration was rerun with "
                 f"{int(MCP.n_seeds.max())} independent seeds (SEED + 5000 + k) at the registered 1000 draws. The "
                 "table lists every registered probability that lies within 0.03 of its threshold in the registered "
                 "run: P_sit (sits if &gt;= 0.5), P_far (falsified if &gt;= 0.95), P_skill_pos (follows if &gt;= "
                 "0.9, does not follow if &lt;= 0.1) and preference P_T (&gt;= 0.9 or &lt;= 0.1). The last column is "
                 "the draw count at which the seed SD would be half the distance between the seed mean and the "
                 "threshold (SD scales as 1/sqrt(draws)). It is a precision note only; the registered count stays "
                 "1000.</p>")
        show = MCP.copy()
        show["group"] = show.group.map(gname)
        H.append(tbl(show, fmt={"threshold": "{:.2f}", "registered": "{:.3f}", "mean": "{:.3f}", "sd": "{:.3f}",
                                "min": "{:.3f}", "max": "{:.3f}", "share_at_or_above": "{:.2f}",
                                "draws_for_2sd": "{:,.0f}"}))
        if MCA is not None:
            va = MCA[MCA.group.isin([g for g, _, _ in seed_var])].copy()
            if len(va):
                va["group"] = va.group.map(gname)
                H.append("<p>Registered answers that change between seeds (count of seeds per answer):</p>")
                H.append(tbl(va))
        H.append("<p>" + (" ".join(f"{gname(g)}: {txt}." for g, _, txt in seed_var) if seed_var else
                          "No registered answer changes between seeds.") +
                 (" Probabilities whose seed range straddles the threshold: " +
                  "; ".join(f"{gname(r.group)} {r['item']} {r.prob} (mean {r['mean']:.3f}, SD {r.sd:.3f}, "
                            f"{r['min']:.3f}-{r['max']:.3f}, threshold {r.threshold:.2f})"
                            for _, r in straddle.iterrows()) + "." if straddle is not None and len(straddle) else
                  " No seed range straddles its threshold.") + "</p>")
        others = [] if straddle is None else [r for _, r in straddle.iterrows()
                                               if r.group not in [g for g, _, _ in seed_var]]
        H.append("<p>" + (f"The other {len(others)} straddling probabilities do not change any registered answer: each "
                          "sits inside a rule that another predictor or test already decides. " if others else "") +
                 " ".join(f"Resolving the {gname(r.group)} {r['item']} {r.prob} (seed mean {r['mean']:.3f}, threshold "
                          f"{r.threshold:.2f}) at 2 SD would take about {r.draws_for_2sd:,.0f} draws per run."
                          for _, r in (straddle.iterrows() if straddle is not None else [])
                          if r.group in [g for g, _, _ in seed_var]) +
                 " The registered draw count stays 1000.</p>")
    H.append("<h3>5b. Post-registration control L: are our depths too deep?</h3>")
    H.append("<p>Requested by the coordinator after the report, written in RESUME.md at 2026-11-27 13:52:46 PST "
             "(commit 0ca862fd) before it was computed. The decision rules are unchanged. Event-paired comparisons "
             "(b4_lit_paired.py, v6.1) put our depths deeper than local OBS catalogs. Against Lange et al. (2026) at "
             "Blanco the median is +1.01 km (tiers A-C) and +3.17 km (tiers A-B, p90 +9.87). By Blanco segment (tiers "
             "A-B) it is +5.31 West, -0.24 Surveyor, +1.63 Cascadia Depression, +3.84 Blanco Ridge and +9.58 Gorda "
             "Depression (23 events). Mount St Helens minus Hirao et al. (2025) is +0.85 km. No paired catalog "
             "exists for the Gorda plate, the MFZ, the triple junction or the Gorda slab.<ul>"
             "<li>L1 subtracts each event's regional offset before D90: Blanco segments by longitude, and the "
             "Blanco-wide value elsewhere. Variants:<ul>"
             "<li>lit_med: median offsets;</li>"
             "<li>lit_p90: p90 offsets, the worst case;</li>"
             "<li>lit_gd: lit_med, but with the Gorda Depression median for the Mendocino-Gorda system.</li></ul>"
             "Segments, frames and predictors are those of the unshifted events.</li>"
             "<li>L2 recomputes D90 from Lange's depths for our Blanco events paired with his absolute catalog "
             "(lange). The same events with our depths are lange_ours. Lange's datum is read as km below sea "
             "level, as in b4_lit_paired.py; if it were below the seafloor, his depths would be about 3 km deeper "
             "in our frame.</li></ul></p>")
    lrows = []
    for g in ["frame:oceanic", "transform", "seg:BL", "seg:MFZ", "intraplate", "triple junction", "Gorda slab (GS)",
              "frame:slab"]:
        r = dict(group=gname(g))
        for c in ("main", "lit_med", "lit_gd", "lit_p90", "lange_ours", "lange"):
            if c.startswith("lange") and g != "seg:BL":
                r[c] = ""
                continue
            if c == "main":
                b6, b8 = gv(g, "HSC_600", "bias"), gv(g, "HSC_800", "bias")
                a = AN[AN.group.eq(g)]
            else:
                b6, b8 = gl(c, g, "HSC_600"), gl(c, g, "HSC_800")
                t = AL.get(c)
                a = t[t.group.eq(g)] if t is not None else t
            if a is None or not len(a) or not np.isfinite(b6):
                r[c] = ""
                continue
            r[c] = (f"600 {b6:+.1f} / 800 {b8:+.1f}; H_T {'falsified' if bool(a.iloc[0].T_falsified) else 'not falsified'}"
                    f" ({int(a.iloc[0].n_units)} units)")
        r["shift that fits 800 C"] = f"{need(g):.1f} km" if np.isfinite(need(g)) else ""
        lrows.append(r)
    H.append(tbl(pd.DataFrame(lrows)))
    H.append("<p class='mut'>Each cell gives the bias of D90 against the 600 C and 800 C half-space isotherms (km; "
             "positive = base deeper) and the registered H_T verdict. The last column is the extra depth bias that "
             "would bring the 800 C isotherm within the 3 km tolerance: main bias minus 3 km.</p>")
    H.append(f"<p><b>Verdict.</b> The headline 'D90 {min(float(x) for x in six):.0f}-{max(float(x) for x in six):.0f} km "
             f"below all conductive isotherms' does not survive as stated.<ul>"
             f"<li><b>Velocity levels move too</b>: with the median offsets the oceanic-frame base sits at the "
             f"7.5 km/s level ({gl('lit_med', 'frame:oceanic', 'V7.5'):+.1f} km, was "
             f"{gv('frame:oceanic', 'V7.5', 'bias'):+.1f}). So 'about 5 km below 7.5 km/s' also depends on the depth "
             f"bias.</li>"
             f"<li><b>Median offsets</b>: the oceanic-frame rejection of 350-800 C holds; the bias to 800 C drops from "
             f"{gv('frame:oceanic', 'HSC_800', 'bias'):+.1f} to {gl('lit_med', 'frame:oceanic', 'HSC_800'):+.1f} km. "
             f"It also holds for the Gorda plate interior, the MFZ and the triple junction (the triple junction "
             f"through the shape rule). It fails for the transforms as a group.</li>"
             f"<li><b>Blanco (L2)</b>: Lange's depths put the base within 3 km of the 600 and 800 C mean-geotherm "
             f"isotherms. The excess depth at Blanco is the known depth bias, not a deep brittle layer.</li>"
             f"<li><b>Worst-case offsets</b> (p90 everywhere, or the Gorda Depression value in the Mendocino-Gorda "
             f"system): the 800 C isotherm fits the oceanic frame and the rejection fails.</li>"
             f"<li><b>What can be stated</b>: in the Gorda plate interior and the MFZ the seismicity base lies "
             f"{gv('intraplate', 'HSC_800', 'bias'):.0f}-{gv('seg:MFZ', 'HSC_800', 'bias'):.0f} km below the 800 C "
             f"isotherm. A temperature limit of at most 800 C would need our depths there to be "
             f"{need('intraplate'):.0f}-{need('seg:MFZ'):.0f} km too deep. That is "
             f"{need('intraplate') / 3.17:.0f}-{need('seg:MFZ') / 3.17:.0f} times the Blanco median offset "
             f"(3.2 km), and of the size of the one paired value nearby (Gorda Depression, 9.6 km).</li></ul></p>")
    ctrl_names = CTRL
    oc_ctrl_sentence = f"This {ht_ctrl_text('frame:oceanic')}; the external-data isotherms (Tm 1300 C) give the same."
    tr8 = [c for c in ctrl_names if gc(c, "transform", "HSC_800", "abs_cat") != "falsified"]
    tr6 = [c for c in ctrl_names if gc(c, "transform", "HSC_600", "abs_cat") != "falsified"]
    tr_ctrl_sentence = ("For the transforms alone, the 800 C isotherm is " +
                        ("rejected in every control" if not tr8 else "not rejected with " + " or with ".join(CTRL[c] for c in tr8))
                        + "; the 600 C isotherm is " + ("rejected in every control." if not tr6 else
                                                         "not rejected with " + " or with ".join(CTRL[c] for c in tr6) + "."))
    # ---------------------------------------------------------------- 5c: boundaries v2 against the D-B113 run
    OLD = Q.BIG / "run_DB113_axes_v1"
    if OLD.exists():
        oa = pd.read_csv(OLD / "answers_main.csv").set_index("group")
        og = pd.read_csv(OLD / "groups_main.csv")
        ol = {c: pd.read_csv(OLD / f"answers_{c}.csv").set_index("group") for c in ("lit_med", "lit_p90", "lange")
              if (OLD / f"answers_{c}.csv").exists()}
        oc = pd.read_csv(OLD / "circularity_levels.csv")

        def ogv(g, L, c="bias"):
            x = og[og.group.eq(g) & og.level.eq(L)]
            return float(x.iloc[0][c]) if len(x) and pd.notna(x.iloc[0][c]) else np.nan
        crow = []
        for g in ["frame:oceanic", "frame:slab", "transform", "seg:BL", "seg:MFZ", "intraplate", "triple junction",
                  "Gorda slab (GS)", "ridge (no AX)"]:
            na = AN.set_index("group")
            r = dict(group=gname(g),
                     units=f"{int(oa.loc[g, 'n_units']) if g in oa.index else 0} -> {nu(g)}",
                     answer=(f"{oa.loc[g, 'answer'] if g in oa.index else '-'} -> {na.loc[g, 'answer'] if g in na.index else '-'}"),
                     H_T_falsified=(f"{bool(oa.loc[g, 'T_falsified']) if g in oa.index else '-'} -> "
                                    f"{bool(na.loc[g, 'T_falsified']) if g in na.index else '-'}"),
                     bias_600=f"{ogv(g, 'HSC_600'):+.1f} -> {gv(g, 'HSC_600', 'bias'):+.1f}",
                     bias_800=f"{ogv(g, 'HSC_800'):+.1f} -> {gv(g, 'HSC_800', 'bias'):+.1f}",
                     bias_7p5=f"{ogv(g, 'V7.5'):+.1f} -> {gv(g, 'V7.5', 'bias'):+.1f}")
            for c in ("lit_med", "lit_p90"):
                t = AL.get(c)
                new = bool(t.set_index("group").loc[g, "T_falsified"]) if t is not None and g in set(t.group) else "-"
                old = bool(ol[c].loc[g, "T_falsified"]) if c in ol and g in ol[c].index else "-"
                r[f"H_T_falsified_{c}"] = f"{old} -> {new}"
            crow.append(r)
        v2cmp = pd.DataFrame(crow)
        v2cmp.to_csv(Q.OUT / "boundaries_v2_vs_DB113.csv", index=False)
        changed = v2cmp[[a.split(" -> ")[0] != a.split(" -> ")[1] for a in v2cmp.answer]]
        tf_changed = v2cmp[[a.split(" -> ")[0] != a.split(" -> ")[1] for a in v2cmp.H_T_falsified]]

        def ocirc(g):
            x = oc[oc.group.eq(g) & oc.level.eq("V7.5") & oc.status.eq("ok")]
            return x.iloc[0] if len(x) else None
        v2_note = dict(tab=v2cmp, changed=changed, tf_changed=tf_changed, ocirc=ocirc)
    else:
        v2_note = None
    if v2_note is not None:
        H.append("<h3>5c. Plate boundaries v2 (D-B117) against the D-B113 run</h3>")
        H.append("<p>Everything above uses the adopted plate boundaries v2. The first report used the D-B113 axes. "
                 "The rules, configurations and controls are the same. Each cell reads 'D-B113 run -> v2 run' (main "
                 "configuration; biases in km against the half-space isotherms and the 7.5 km/s level; H_T verdict "
                 "of the registered rule, also under the median and p90 literature offsets).</p>")
        H.append(tbl(v2_note["tab"]))
        ch = v2_note["changed"]
        why = []
        for _, r in ch.iterrows():
            g = [k for k in KEY_GROUPS if gname(k) == r.group]
            if not g:
                continue
            g = g[0]
            diffs = []
            for L in A.ISO_KEYS + A.LEV_KEYS:
                for col, lab in (("abs_cat", "fit"), ("shape_cat", "shape")):
                    xo = og[og.group.eq(g) & og.level.eq(L)]
                    xn = G[G.group.eq(g) & G.level.eq(L)]
                    if len(xo) and len(xn) and str(xo.iloc[0][col]) != str(xn.iloc[0][col]):
                        pc = "P_sit" if col == "abs_cat" else "P_skill_pos"
                        diffs.append(f"{L} {lab} {xo.iloc[0][col]} -> {xn.iloc[0][col]} "
                                     f"(P {xo.iloc[0][pc]:.3f} -> {xn.iloc[0][pc]:.3f})")
            why.append(f"{r.group}: " + ("; ".join(diffs) if diffs else "preference probabilities moved"))
        H.append("<p>Registered answers that changed: " + ("; ".join(f"{r.group}: {r.answer}" for _, r in ch.iterrows())
                 if len(ch) else "none") + ". Components that moved: " + (" | ".join(why) if why else "none") +
                 ". Where the unit data did not change (the slab frame does not use the segments), the shift comes "
                 "from the Monte Carlo draw sequence: one segment unit fewer precedes the frame units, so their "
                 "draws differ. A probability at the 0.9 threshold can then fall on either side. H_T verdicts that changed (main): " +
                 ("; ".join(f"{r.group}: {r.H_T_falsified}" for _, r in v2_note["tf_changed"].iterrows())
                  if len(v2_note["tf_changed"]) else "none") + ".</p>")
    # ---------------------------------------------------------------- claims
    H.append("<h2>6. What the paper can claim, and what it cannot</h2>")
    H.append("<h3>Can claim</h3><ul>"
             f"<li>In the Gorda plate interior and along the Mendocino fault zone (Gorda plate ages up to "
             f"~{np.nanmax(segu('GP').age):.0f} Myr; Pacific side of the MFZ {rng_s(segu('MFZ').age_old)} Myr), "
             "located earthquakes extend "
             f"{gv('intraplate', 'HSC_800', 'bias'):.0f}-{gv('seg:MFZ', 'HSC_800', 'bias'):.0f} km below the 800 C "
             "isotherm of half-space and GDH1 cooling models of their Seton et al. (2020) plate age. Transform ages "
             "are taken on both sides. A 350-800 C conductive limit there needs our depths to be "
             f"{need('intraplate'):.0f}-{need('seg:MFZ'):.0f} km too deep; the median depth offset measured at "
             "Blanco is 3.2 km.<ul>"
             f"<li>In the oceanic frame ({nu('frame:oceanic')} bins) no isotherm from 350 to 800 C fits the base within 3 km.</li>"
             f"<li>{oc_ctrl_sentence} {tr_ctrl_sentence}</li></ul>"
             "<li>With the median literature offsets the oceanic-frame rejection still holds (control L1).</li>"
             "</ul>"
             f"So, unless those depths are biased by {min(need('intraplate'), need('seg:MFZ')):.0f} km or more, the depth limit there is not a conductive-cooling "
             "isotherm of the standard brittle range.</li>"
             "<li>At Blanco, Lange's OBS depths put the base within 3 km of the 600-800 C isotherms of the "
             "two-sided transform geotherm (control L2).</li>"
             "<li>In the oceanic plates the base lies at or below the 7.5 km/s level of v0.4.2. With our depths "
             f"it is {gv('frame:oceanic', 'V7.5', 'bias'):.0f} km below it in the oceanic "
             f"frame ({gv('transform', 'V7.5', 'bias'):.1f} km at the transforms, "
             f"{gv('triple junction', 'V7.5', 'bias'):.1f} km at the triple junction), above the 8.0 km/s level "
             f"where it is resolved, and "
             f"{min(gv('frame:oceanic', 'CRUST1_moho', 'bias'), gv('Gorda slab (GS)', 'CRUST1_moho', 'bias')):.0f}-"
             f"{max(gv('frame:oceanic', 'CRUST1_moho', 'bias'), gv('Gorda slab (GS)', 'CRUST1_moho', 'bias')):.0f} km below the "
             "CRUST1.0 Moho (oceanic frame, Gorda slab). With the median literature offsets it sits at the 7.5 km/s "
             f"level ({gl('lit_med', 'frame:oceanic', 'V7.5'):+.1f} km).</li>"
             "<li>In the subducting Gorda slab the base lies between the 600 C isotherms of a coldest-slab bound "
             "and a warm-top variant. This is consistent with a thermal limit near 600-700 C, with the stated "
             "slab-model simplifications.</li></ul>")
    H.append("<h3>Cannot claim</h3><ul>" + "".join(
        f"<li>A single answer for the {gname(g)}: it sits at a registered threshold within Monte Carlo precision "
        f"({txt}). Both readings are stated.</li>" for g, _, txt in seed_var) +
             "<li>That earthquakes go deeper than conductive isotherms at Blanco, or across all transforms. With "
             "Lange's depths, or with the median offsets, the 800 C isotherm fits.</li>"
             "<li>That the rejection holds under any plausible depth bias. With the p90 offsets, or the Gorda "
             "Depression offset moved to the Mendocino-Gorda system, the 800 C isotherm fits the oceanic frame.</li>"
             "<li>That the base follows the 7.5 km/s level (composition control). The following comes from the "
             "Blanco versus Mendocino-Gorda contrast, and it fails three registered controls:<ul>"
             "<li>no level is checkerboard-resolved (C4);</li>"
             "<li>the residuals correlate with station distance (C3);</li>"
             f"<li>the Gorda-slab correlation fails C5b (C5c: it {gs_c5c}), and the oceanic-frame correlation passes "
             f"C5c by {circ('frame:oceanic', 'rho_real') - circ('frame:oceanic', 'rho_null_p95'):.2f} with "
             f"{n_null} members.</li></ul></li>"
             "<li>Which of composition or a colder-than-conductive lithosphere (deep hydrothermal cooling, "
             "compression of the Gorda plate) sets the limit in the oceanic plates. The measured heat flow is "
             "consistent with a colder plate, but too sparse to decide.</li>"
             "<li>Anything about ridge depth limits: Axial is biased, and the Gorda Ridge and other ridges are "
             "poorly constrained.</li>"
             "<li>A down-dip thermal control within the Gorda slab: no registered shape test passes there.</li>"
             "<li>The Juan de Fuca plate interior (no unit) and the Explorer region (no depth-reliable "
             "events).</li></ul>")
    # ---------------------------------------------------------------- manuscript bullets
    H.append("<h2>7. Manuscript bullets</h2><ul>")
    H.append(f"<li>Q2 test, pre-registered: depth-reliable v6.1 events (tier A-B, sig_z &lt;= 3 km); D90 per 0.5 deg "
             f"bin with bootstrap and location Monte Carlo. Compared with 350/600/800 C isotherms (half-space, GDH1; "
             f"Seton et al. 2020 ages; transforms: mean of both sides) and with the 7.5/8.0 km/s levels of v0.4.2 "
             f"(ensemble spread x1.5). Tolerance 3 km.</li>")
    H.append(f"<li>Oceanic plates (all 0.5 deg bins west of the trench: ridges, transforms, Gorda plate, triple "
             f"junction): D90 lies "
             f"{gv('frame:oceanic', 'HSC_600', 'bias'):.0f} km below the 600 C half-space isotherm (median of 26 "
             f"bins; 95 % {gv('frame:oceanic', 'HSC_600', 'bias_lo95'):.0f}-"
             f"{gv('frame:oceanic', 'HSC_600', 'bias_hi95'):.0f} km). The base is at about "
             f"{ts('frame:oceanic', 'HSC'):.0f} C in the cooling model. No 350-800 C isotherm fits within 3 km. "
             f"This {ht_ctrl_text('frame:oceanic')}; in the Gorda plate interior and the MFZ it "
             f"{'holds under every registered control' if not ht_fail('intraplate') and not ht_fail('seg:MFZ') else 'does not hold under every control'}.</li>")
    H.append(f"<li>Depth-bias check: removing the median offset against the Blanco OBS catalog (+3.2 km) keeps the "
             f"rejection in the Gorda plate interior and the MFZ: the base is "
             f"{gl('lit_med', 'intraplate', 'HSC_800'):.0f}-{gl('lit_med', 'seg:MFZ', 'HSC_800'):.0f} km below "
             f"800 C. At Blanco, the OBS catalog's own depths put the base at the 600-800 C isotherms of the "
             f"two-sided transform geotherm.</li>")
    H.append(f"<li>Same plates: D90 lies {gv('frame:oceanic', 'V7.5', 'bias'):.0f} km below the 7.5 km/s level "
             f"({nu('frame:oceanic', 'V7.5')} bins; {gl('lit_med', 'frame:oceanic', 'V7.5'):+.1f} km after the median literature offset) and "
             f"{-gv('frame:oceanic', 'V8.0', 'bias'):.0f} km above the 8.0 km/s level where that level is resolved "
             f"({nu('frame:oceanic', 'V8.0')} bins). Earthquakes reach at least the 7.5 km/s level of the model.</li>")
    H.append("<li>The apparent link between the seismicity base and the 7.5 km/s level is limited by three "
             "things: the Blanco versus Mendocino-Gorda contrast, levels resolved only by ray density (not "
             "checkerboard), and residuals that grow with station distance. It is not used as evidence for a "
             "compositional limit.</li>")
    H.append(f"<li>Gorda slab: D90 lies between the 600 C isotherms of a coldest-slab bound "
             f"({gv('Gorda slab (GS)', 'HSC_600', 'bias'):+.1f} km) and a warm-top variant using the slab-surface "
             f"temperature of van Keken &amp; Wilson (2023) ({gv('Gorda slab (GS)', 'HSC_600_warmtop', 'bias'):+.1f} "
             f"km). This is consistent with a 600-700 C limit.</li>")
    H.append("<li>Heat flow near the triple junction and the margin (34-82 mW/m2) is far below the cooling-model "
             "value for 5 Myr crust. That is consistent with a plate colder than conductive cooling predicts, but "
             "the measurements are too sparse to test.</li>")
    for g, _, txt in seed_var:
        H.append(f"<li>Monte Carlo precision: the {gname(g)} answer sits at a registered threshold ({txt}, 20 "
                 f"independent seeds of 1000 draws); both readings are reported.</li>")
    H.append("<li>Ridge depth limits are not interpreted: Axial depths carry the SSST bias, and other ridge "
             "depths are unconstrained.</li></ul>")
    # ---------------------------------------------------------------- limitations
    H.append("<h2>8. Limitations</h2><ul>"
             "<li>Axial depths carry the known SSST bias (D-B100/D-B103): 72.7 % of tier A-B Axial events are within "
             "2 km of the seafloor, against 90 % in Waldhauser et al. (2020). Cooling models do not describe the "
             "axial magma system.</li>"
             "<li>Ridge depths other than Axial are poorly constrained:<ul>"
             "<li>depth_quality is 'unconstrained' for the ridge regions;</li>"
             f"<li>the {int((off.seg == 'GR').sum())} depth-reliable Gorda Ridge events give D90 {rng_s(segu('GR').D_bsf)} km below the seafloor at near-zero "
             "age, which this report does not interpret.</li></ul></li>"
             "<li>Our depths are deeper than local OBS catalogs: +3.17 km median (tier A-B) at Blanco against Lange "
             "et al. (2026), +9.6 km at the Gorda Depression (23 events), +0.85 km at Mount St Helens. There is no "
             "paired reference for the Gorda plate, the MFZ, the triple junction or the Gorda slab. The "
             "Mendocino-Gorda result rests on depths that no local catalog has checked.</li>"
             "<li>Model resolution is about 40 km laterally. Depth nodes are 3-6 km apart at 15-35 km. No 7.5 or "
             "8.0 km/s crossing offshore lies between checkerboard-resolved nodes.</li>"
             "<li>The velocity levels come from an inversion of the same events. The circularity controls reduce "
             "but do not remove this dependence. " + c5c_state + "</li>"
             "<li>Cooling-model assumptions: conductive cooling from a seafloor at 0 C. There is no hydrothermal "
             "cooling, no shear or frictional heating on transforms, no sediment blanketing and no plate "
             "compression. Transforms use the mean of two one-sided geotherms. Ages come from a global grid that "
             "smooths the Mendocino age step over about 0.1 deg.</li>"
             "<li>The slab isotherms are bounds, not a slab thermal model:<ul>"
             "<li>the incoming geotherm is hung below Slab2 with a 0 C top (coldest slab);</li>"
             "<li>the warm-top variant uses the slab-surface temperature of the 2-D North and Central Cascadia "
             "models, not their interior temperatures.</li></ul></li>"
             f"<li>Small samples: {nu('intraplate')} Gorda-plate units, {nu('seg:MFZ')} MFZ units, "
             f"{nu('triple junction')} triple-junction units and {int(circ('Gorda slab (GS)', 'n_blocks'))} "
             f"Gorda-slab 1-degree blocks. "
             "The 0.5 deg units are still not independent at a 40 km resolution.</li>"
             "<li>The depth Monte Carlo adds noise to depths that already contain it, which shifts the D90 draws "
             "about 0.4 km deeper. The intervals are therefore slightly wide on the deep side.</li></ul>")
    H.append("<h2>References</h2><ul class='mut'>"
             "<li>Bird, P. (2003). An updated digital model of plate boundaries. G-cubed 4, 1027.</li>"
             "<li>Boettcher, M. S., &amp; Jordan, T. H. (2004). Earthquake scaling relations for mid-ocean ridge "
             "transform faults. JGR 109, B12302.</li>"
             "<li>Hayes, G. P., et al. (2018). Slab2, a comprehensive subduction zone geometry model. Science 362, "
             "58-61.</li>"
             "<li>Laske, G., et al. (2013). Update on CRUST1.0. EGU General Assembly abstract.</li>"
             "<li>Seton, M., et al. (2020). A global data set of present-day oceanic crustal age and seafloor "
             "spreading parameters. G-cubed 21, e2020GC009214.</li>"
             "<li>Stein, C. A., &amp; Stein, S. (1992). A model for the global variation in oceanic depth and heat "
             "flow with lithospheric age. Nature 359, 123-129.</li>"
             "<li>Straume, E. O., et al. (2019). GlobSed: updated total sediment thickness in the world's oceans. "
             "G-cubed 20, 1756-1772.</li>"
             "<li>Global Heat Flow Data Assessment Group et al. (2024). The Global Heat Flow Database, Release 2024. "
             "GFZ Data Services, doi:10.5880/fidgeo.2024.014.</li>"
             "<li>Turcotte, D. L., &amp; Schubert, G. (2002). Geodynamics, 2nd ed. Cambridge University Press.</li>"
             "<li>van Keken, P. E., &amp; Wilson, C. R. (2023). Progress in Earth and Planetary Science; models "
             "doi:10.5281/zenodo.14854311.</li>"
             "<li>Waldhauser, F., et al. (2020). Precision seismic monitoring and analysis at Axial Seamount using "
             "a real-time double-difference system. JGR 125.</li></ul>")
    H.append("<h2>Reproduce</h2><p><code>bash workflow/11_paper_plan/run_q2.sh</code> (all configurations, "
             "circularity, extras, figures, this report). Tables in data/paper_plan/q2_depth_limit/, large files "
             "and figures in /wd1/mdenolle_data/paper_plan/q2_depth_limit/.</p></body></html>")
    out = Q.BIG / "report_q2.html"
    out.write_text("\n".join(H))
    Q.log("report written", out)


if __name__ == "__main__":
    main()
