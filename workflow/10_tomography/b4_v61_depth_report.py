"""Track B, catalog v6.1 depth test (D-B100): HTML report (diagnosis, pre-registered rule, scores, decision).

  python b4_v61_depth_report.py [--implications notes.md]
      -> /wd1/mdenolle_data/reloc_v61/depth_test/report_depth_test.html (figures embedded; PNG copies in figs/)
"""
import argparse
import base64
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_v61_depth_figs as F  # noqa: E402

OUT = Path("/wd1/mdenolle_data/reloc_v61/depth_test")
SC = OUT / "score"
DIAG = OUT / "diag"
CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#d9d8d4;--card:#f3f2ef;--a:#2a78d6;--b:#eb6834;--good:#0b7a3e;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--card:#242422;--a:#3987e5;--b:#d95926;--good:#4cc27e;--bad:#ef7a73}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--card:#242422;--a:#3987e5;--b:#d95926;--good:#4cc27e;--bad:#ef7a73}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1150px;padding:16px}
h1{font-size:1.5em;margin:.3em 0}h2{font-size:1.2em;margin-top:1.6em;border-bottom:1px solid var(--line)}h3{font-size:1.02em;margin-top:1.2em}
p,li{max-width:85ch}.muted{color:var(--muted);font-size:.92em}
.tw{overflow-x:auto;margin:.5em 0 1em}table{border-collapse:collapse;font-size:.86em;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:600}
td:first-child,th:first-child{text-align:left}tr.hl td{background:var(--card)}td.out{color:var(--bad)}td.win{color:var(--good);font-weight:700}
.verdict{background:var(--card);border-left:4px solid var(--good);padding:.6em 1em;margin:1em 0}
.rule{background:var(--card);border-left:4px solid var(--a);padding:.4em 1em;margin:1em 0;font-size:.95em}
img{max-width:100%;height:auto;background:#fff;border-radius:4px;margin:.4em 0}
code{font-size:.9em}
"""


def esc(t):
    return html.escape(str(t))


def f(v, nd=2, sign=False):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return esc(v)
    if not np.isfinite(v):
        return ""
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def table(head, rows, cls=None):
    h = "<tr>" + "".join(f"<th>{esc(x)}</th>" for x in head) + "</tr>"
    body = []
    for i, r in enumerate(rows):
        c = cls[i] if cls else [""] * len(r)
        body.append("<tr>" + "".join(f"<td class='{cc}'>{x}</td>" for x, cc in zip(r, c)) + "</tr>")
    return f"<div class='tw'><table>{h}{''.join(body)}</table></div>"


def img(b, alt):
    return f"<img alt='{esc(alt)}' src='data:image/png;base64,{base64.b64encode(b).decode()}'>"


def md(text):
    """minimal markdown: paragraphs, '- ' lists, **bold**, `code`."""
    out, buf, lst = [], [], []

    def inline(t):
        t = esc(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", t)

    def flush():
        if buf:
            out.append("<p>" + inline(" ".join(buf)) + "</p>")
            buf.clear()
        if lst:
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in lst) + "</ul>")
            lst.clear()
    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
        elif s.lstrip().startswith("- "):
            if buf:
                flush()
            lst.append(s.lstrip()[2:])
        elif lst and s.startswith("  "):
            lst[-1] += " " + s.strip()
        else:
            if lst:
                flush()
            buf.append(s.strip())
    flush()
    return "\n".join(out)


def rule_text():
    t = (OUT / "RESUME.md").read_text()
    a = t.index("## Pre-registered decision rule")
    b = t.index("## Status log")
    return t[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--implications", default=None, help="markdown file with the summary/implication text")
    a = ap.parse_args()
    dec = json.loads((SC / "decision.json").read_text())
    Hd = pd.read_csv(SC / "heldout.csv")
    Dp = pd.read_csv(SC / "depth.csv")
    Sf = pd.read_csv(SC / "surface.csv")
    arms = dec["arms"]
    S = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         "content='width=device-width,initial-scale=1'><title>v6.1 station-term depth test</title>"
         f"<style>{CSS}</style></head><body>"]
    S.append("<h1>Catalog v6.1: station terms scored on relative fit and absolute depth</h1>")
    S.append(f"<p class='muted'>D-B100 follow-up. Rule written {esc(dec['rule_written'])}; scored "
             f"{esc(dec['scored'])}. Branch reloc/v61-depth-test; data /wd1/mdenolle_data/reloc_v61/depth_test/.</p>")
    if a.implications:
        S.append("<div class='verdict'>" + md(Path(a.implications).read_text()) + "</div>")
    # ---------------------------------------------------------------- diagnosis
    S.append("<h2>1. Why the SSST terms move depth (diagnosis on the v6.0 outputs)</h2>")
    dg = json.loads((DIAG / "t07_32" / "summary.json").read_text())
    bp = pd.read_csv(DIAG / "t07_32" / "by_pass.csv")
    ab = bp[bp.tiers == "A-B"]
    S.append("<p>Axial tile t07_32, the fixed set of the tile's Axial events with v6.0 tier A-B located in every pass "
             f"({int(ab.n.iloc[0]):,} events). Depth below the seafloor = true depth + merged solid surface.</p>")
    S.append(table(["pass", "within 1.6 km (%)", "within 2.0 km (%)", "median below seafloor (km)",
                    "floor-limited (%)", "median shift from it0 (km)", "median rms (s)"],
                   [[["it0 no terms", "it1 static (9999 km)", "it2 SSST 16 km", "it3 SSST 8 km",
                      "it4 SSST 4 km"][int(r.it)], f(r.within_1_6, 1), f(r.within_2_0, 1), f(r.median_bsf),
                     f(r.floor_limited, 1), f(r.dz_med, 3, True), f(r.rms_med, 3)] for r in ab.itertuples()]))
    S.append(img(F.fig_axial_passes(), "Axial diagnosis"))
    S.append(md((OUT / "diag_explanation.md").read_text()))
    reg = []
    for tile, lab in (("t07_32", "Axial"), ("t29_40", "Puget/Georgia"), ("t22_09", "Gorda"), ("t11_23", "Blanco")):
        p = DIAG / tile / "linearised.csv"
        if not p.exists():
            continue
        x = pd.read_csv(p)
        x = x[x.tier.isin(["A", "B", "C"])]
        reg.append([f"{lab} ({tile})", f"{len(x):,}"] + [f(x[f"dz_obs_it{k}"].median(), 2, True) for k in (1, 2, 3, 4)]
                   + [f(x[c].median(), 2, True) for c in ("dz_lin_it4", "dz_commonP_it4", "dz_commonS_it4",
                                                           "dz_rest_it4")]
                   + [f(x.SmP_common_it4.median() * 1e3, 0, True), f(x.corr_z_commonS.median(), 2),
                      f(x.sigz_ratio_commonS.median(), 1), f(100 * (x.dz_obs_it4.abs() > 1).mean(), 0)])
    S.append("<h3>The same decomposition in the busiest tile of other regions (tier A-C, medians over events)</h3>")
    S.append(table(["tile", "events", "dz it1", "it2", "it3", "it4 (km)", "linearised it4", "common-P part",
                    "common-S part", "station pattern", "S-P common term (ms)", "corr(z, common S)",
                    "sigma_z inflation", "|dz it4| > 1 km (%)"], reg))
    S.append(img(F.fig_tradeoff([("t07_32", "Axial t07_32"), ("t29_40", "Puget t29_40"), ("t22_09", "Gorda t22_09"),
                                 ("t11_23", "Blanco t11_23")]), "depth / common-S trade-off"))
    # ---------------------------------------------------------------- rule
    S.append("<h2>2. Pre-registered rule</h2>")
    S.append("<div class='rule'>" + md(rule_text().split("\n", 1)[1]) + "</div>")
    # ---------------------------------------------------------------- scores
    S.append("<h2>3. Scores</h2>")
    ss = OUT / "sets" / "sets_summary.csv"
    if ss.exists():
        t = pd.read_csv(ss)
        S.append("<p>Test clusters (each run: full = every station, fold0-2 = a third of the stations held out):</p>")
        S.append(table(["set", "region", "events", "picks", "stations", "accepted CC rows", "CC pairs", "time span"],
                       [[r.set, esc(r.region), f"{r.events:,}", f"{r.picks:,}", f"{r.stations}", f"{r.cc_rows:,}",
                         f"{r.cc_pairs:,}", f"{r.time0} to {r.time1}"] for r in t.itertuples()]))
    S.append("<h3>Relative fit: held-out CC differential times (3 station folds pooled)</h3>")
    S.append("<p class='muted'>MAD of the pair-demeaned held-out residual (ms) on the rows located in every arm; "
             "difference to the best arm of the cluster and phase with its 95 % paired-bootstrap interval "
             f"({dec['nboot']} resamples of event pairs). Red: out by rule (b) (lower end above 5 % of the best).</p>")
    for s in dict.fromkeys(Hd.set):
        h = Hd[Hd.set == s]
        rows, cls = [], []
        for arm in arms:
            r = {ph: h[(h.phase == ph) & (h.arm == arm)] for ph in ("P", "S")}
            if any(len(v) == 0 for v in r.values()):
                continue
            row, c = [arm], [""]
            for ph in ("P", "S"):
                x = r[ph].iloc[0]
                row += [f(x.mad_ms, 1), f"{f(x.diff_best_ms, 1, True)} [{f(x.lo_ms, 1, True)}, {f(x.hi_ms, 1, True)}]"]
                c += ["", "out" if x.out else ""]
            rows.append(row)
            cls.append(c)
        h0 = h.iloc[0]
        S.append(f"<p><b>{esc(s)}</b>: {int(h[h.phase == 'P'].rows.iloc[0]):,} P rows, "
                 f"{int(h[h.phase == 'S'].rows.iloc[0]):,} S rows</p>")
        S.append(table(["arm", "MAD P (ms)", "P - best", "MAD S (ms)", "S - best"], rows, cls))
    S.append(img(F.fig_scores(Hd, None, dec), "relative fit"))
    S.append("<h3>Absolute depth (full runs) against the published catalogs</h3>")
    S.append("<p class='muted'>Ours - published (km, + = ours deeper): median [p10, p90], v6.0 tier A-C paired events "
             "located in every arm. Axial: v6.0 tier A-B events, shares within 1.6 / 2.0 km of the seafloor and the "
             "quantile offset mean(q85 - 1.6, q90 - 2.0).</p>")
    for s in ("axial", "msh", "blancow", "gordad"):
        d = Dp[Dp.set == s].set_index("arm")
        if not len(d):
            continue
        S.append(f"<p><b>{esc(s)}</b> vs {esc(d.reference.iloc[0])}, n = {int(d.n.iloc[0]):,}</p>")
        if s == "axial":
            S.append(table(["arm", "within 1.6 km (%)", "within 2.0 km (%)", "q85 (km)", "q90 (km)", "median (km)",
                            "offset (km)"],
                           [[arm, f(d.loc[arm].within_1_6_pct, 1), f(d.loc[arm].within_2_0_pct, 1),
                             f(d.loc[arm].q85), f(d.loc[arm].q90), f(d.loc[arm].median_bsf),
                             f(d.loc[arm].offset_km, 3, True)] for arm in arms if arm in d.index]))
        else:
            hd = ["arm", "offset median (km)", "p10", "p90", "ours median (km)", "published median (km)"]
            if "rel_offset_km" in d and d.rel_offset_km.notna().any():
                hd += ["Lange relative: median [p10, p90]"]
            rows = []
            for arm in arms:
                if arm not in d.index:
                    continue
                x = d.loc[arm]
                row = [arm, f(x.offset_km, 3, True), f(x.p10_km, 2, True), f(x.p90_km, 2, True),
                       f(x.ours_median_km), f(x.ref_median_km)]
                if len(hd) > 6:
                    row.append(f"{f(x.rel_offset_km, 2, True)} [{f(x.rel_p10_km, 1, True)}, "
                               f"{f(x.rel_p90_km, 1, True)}] n={int(x.rel_n)}")
                rows.append(row)
            S.append(table(hd, rows))
    S.append(img(F.fig_depth(Dp, dec), "depth offsets"))
    S.append("<h3>Events above the solid surface; floor-limited share; coherence relocations (full runs)</h3>")
    sf = Sf[Sf.run == "full"].pivot(index="arm", columns="set", values="floor_limited_pct").reindex(arms)
    ab_ = Sf.groupby("arm").above_surface.sum().reindex(arms)
    rc = Sf[Sf.run == "full"].pivot(index="arm", columns="set", values="relocated_by_coherence").reindex(arms)
    S.append(table(["arm", "above surface (all runs)"] + [f"floor-limited % {s}" for s in sf.columns]
                   + [f"coherence-relocated {s}" for s in rc.columns],
                   [[arm, f"{int(ab_[arm]) if np.isfinite(ab_[arm]) else ''}"]
                    + [f(sf.loc[arm, s], 1) for s in sf.columns] + [f(rc.loc[arm, s], 0) for s in rc.columns]
                    for arm in arms]))
    # nf diagnostics
    nfr = []
    for s in ("axial", "msh", "blancow", "gordad"):
        for k in range(4):
            p = OUT / "runs" / s / "full" / "nf" / f"g{k}" / "nf.json"
            if p.exists():
                d = json.loads(p.read_text())
                nfr.append([s, f"g{k}", f"{d['events']:,}", f"{d['station_phases']}",
                            f(d['beta'][2], 3, True), f(d['beta'][3] * 1e3, 1, True),
                            f(d['dz_lin_mean_before'], 3, True), f(d['dz_lin_mean_after'], 3, True),
                            f"{d['negative_nodes']}"])
    if nfr:
        S.append("<h3>nf4: the removed common-shift part (full runs)</h3>")
        S.append("<p class='muted'>beta = the common shift (dx, dy, dz km, dt0 s) whose travel-time signature "
                 "a_i . beta was removed from the pass's increments (written as nf.json 'beta'; the key 'beta_names' of "
                 "the runs mislabels the units as s/km); linearised mean depth shift of each event from the pass's "
                 "increment, before and after the removal.</p>")
        S.append(table(["set", "pass", "events", "station-phases", "beta_z (km)", "beta_t0 (ms)",
                        "lin. dz before (km)", "after (km)", "negative nodes"], nfr))
    # ---------------------------------------------------------------- decision
    S.append("<h2>4. Decision by the rule</h2>")
    D = dec["D_km"]
    rows, cls = [], []
    for arm in arms:
        why = []
        if arm in dec["incomplete"]:
            why.append("incomplete")
        if arm in dec["out_above_surface"]:
            why.append("event above surface")
        if arm in dec["out_relative_fit"]:
            why.append("relative fit: " + "; ".join(dec["out_relative_fit"][arm]))
        st = "winner" if arm == dec["winner"] else ("tied" if arm in dec["tied"] else
                                                    ("eligible" if arm in dec["eligible"] else "out"))
        rows.append([arm, f(D.get(arm), 3), st, "<div style='white-space:normal;min-width:40ch;max-width:80ch'>"
                     + esc(" | ".join(why)) + "</div>"])
        cls.append(["", "", "win" if st == "winner" else ("out" if st == "out" else ""), ""])
    S.append(table(["arm", "D (km)", "status", "reason"], rows, cls))
    S.append(f"<p>Literature clusters in D: {', '.join(dec['literature_sets'])}. Best arm per cluster and phase: "
             + esc(", ".join(f"{k}: {v}" for k, v in dec["best_by_set_phase"].items())) + ".</p>")
    S.append(f"<p class='verdict'><b>Winner: {esc(dec['winner'])}</b>" +
             (f" (tied within 0.05 km: {esc(', '.join(dec['tied']))}; the simplest in the rule's order wins)"
              if len(dec["tied"]) > 1 else "") + "</p>")
    # ---------------------------------------------------------------- post-hoc
    ex = SC / "extra.json"
    if ex.exists():
        E = json.loads(ex.read_text())
        He = pd.read_csv(SC / "extra_heldout.csv")
        De = pd.read_csv(SC / "extra_depth.csv")
        S.append("<h2>5. Post-hoc arm (not part of the decision)</h2>")
        S.append(md((OUT / "report_posthoc_input.txt").read_text()) if (OUT / "report_posthoc_input.txt").exists()
                 else "")
        rows = []
        for r in He.itertuples():
            rows.append([r.arm, r.set, r.phase, f(r.mad_ms, 1), f"{r.best} {f(r.mad_best_ms, 1)}",
                         f"{f(r.diff_best_ms, 1, True)} [{f(r.lo_ms, 1, True)}, {f(r.hi_ms, 1, True)}]",
                         "out" if r.would_be_out else ""])
        S.append(table(["arm", "set", "phase", "MAD (ms)", "registered best (ms)", "arm - best", "gate (b)"], rows))
        rows = []
        for r in De.itertuples():
            extra_ = (f"within 1.6 / 2.0 km {f(r.within_1_6_pct, 1)} / {f(r.within_2_0_pct, 1)} %" if r.set == "axial"
                      else f"p10 {f(r.p10_km, 2, True)}, p90 {f(r.p90_km, 2, True)}")
            rows.append([r.arm, r.set, f"{int(r.n):,}", f(r.offset_km, 3, True), extra_])
        S.append(table(["arm", "set", "n", "offset (km)", ""], rows))
        S.append(f"<p>D (same four clusters): " + ", ".join(f"{k} {f(v, 3)} km" for k, v in E["D_km"].items())
                 + "; gate (b) would exclude: " + esc(json.dumps(E["would_be_out"])) + "</p>")
    # ---------------------------------------------------------------- reproducibility
    try:
        head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        head = "?"
    S.append("<h2>6. Reproducibility</h2>")
    S.append(md(f"""Code: branch `reloc/v61-depth-test` (commit `{head}`), workflow/10_tomography/:
`b4_v61_depth_diag.py` (diagnosis; `--tile t07_32`, `--tile t29_40 --region Puget/Georgia --near-km 25`,
`--tile t22_09 --region Gorda --near-km 40`, `--tile t11_23 --region Blanco --near-km 40`), `b4_v61_depth_sets.py`
(gordad set), `b4_v61_depth_test.py run SET RUN` (16 chains, /wd1/mdenolle_data/reloc_v61/depth_test/run_chains.sh),
`b4_v61_depth_test.py score --nboot 2000`, `b4_v61_depth_report.py`. Inputs: grids reloc_v6/time_v6_v0.4.1 and
wide_time_v6_v0.4.1 (manifest dbe2f8a1), picks Cascadia_picks_catalog_v4.1tc.csv (md5 1b21ab1c), CC sets of
D-B078 / termtest / cc_dt_all, literature pair tables tomo_v1/depth_literature/pairs/, catalog v6.0 tiers
(md5 ae77ad3c). NonLinLoc 7.1.07 (ut-beg-texnet 3eb0c1e5) unchanged."""))
    S.append("</body></html>")
    out = OUT / "report_depth_test.html"
    out.write_text("\n".join(S))
    print(out)


if __name__ == "__main__":
    main()
