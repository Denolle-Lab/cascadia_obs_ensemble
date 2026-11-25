"""Track B, v6.0 station-term test: HTML report (tables of b4_v6_termtest_score.py, a figure, the summary text).

  python b4_v6_termtest_report.py [--summary notes/v6_termtest_summary.md]
      -> /wd1/mdenolle_data/reloc_v6/termtest/report/termtest_report.html
"""
import argparse
import base64
import html
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_relocation"))
import b4_v6_termtest as T  # noqa: E402
import b4_v6_termtest_score as SC  # noqa: E402

REPORT = T.TT / "report"
SC_DIR = T.TT / "score"
LABEL = {"v5.1": "v5.1 catalog (reference; in-sample for held-out)", "static": "static terms alone",
         "ssst_it0": "SSST it0 (no terms)", "ssst_it1": "SSST it1 (set statics, 9999 km)", "ssst_it2": "SSST it2 (16 km)",
         "ssst_it3": "SSST it3 (8 km)", "ssst_it4": "SSST it4 (4 km)", "ssst_it5": "SSST to 2 km alone (it5)",
         "A": "A: static + coherence", "B": "B: SSST to 2 km + coherence"}
SETNAME = {"puget": "Puget Lowland (onshore)", "mtj": "Mendocino triple junction (NC land + OBS)",
           "blancow": "western Blanco (offshore OBS)"}
CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#d9d8d4;--card:#f3f2ef;--a:#2a78d6;--b:#eb6834;--good:#0b7a3e;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--card:#242422;--a:#3987e5;--b:#d95926;--good:#4cc27e;--bad:#ef7a73}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--card:#242422;--a:#3987e5;--b:#d95926;--good:#4cc27e;--bad:#ef7a73}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:1.5em;margin:.3em 0}h2{font-size:1.2em;margin-top:1.6em;border-bottom:1px solid var(--line)}h3{font-size:1.02em;margin-top:1.2em}
p,li{max-width:80ch}.muted{color:var(--muted);font-size:.92em}
.tw{overflow-x:auto;margin:.5em 0 1em}table{border-collapse:collapse;font-size:.88em;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:600}
td:first-child,th:first-child{text-align:left}tr.hl td{background:var(--card)}
.verdict{background:var(--card);border-left:4px solid var(--a);padding:.6em 1em;margin:1em 0}
.A{color:var(--a);font-weight:600}.B{color:var(--b);font-weight:600}
img{max-width:100%;height:auto;background:#fff;border-radius:4px}
code{font-size:.9em}
"""


def f(v, nd=1):
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if nd == 0 and isinstance(v, float) and np.isfinite(v):
        return f"{int(round(v)):,}"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    return "" if not np.isfinite(v) else f"{v:.{nd}f}"


def fci(t, nd=1):
    """(value, lo, hi) -> 'v [lo, hi]'."""
    if not t:
        return ""
    v, lo, hi = t
    return f"{f(v, nd)} <span class='muted'>[{f(lo, nd)}, {f(hi, nd)}]</span>"


def table(head, rows, hl=()):
    h = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    b = "".join(f"<tr{' class=hl' if i in hl else ''}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                for i, r in enumerate(rows))
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def md(text):
    """tiny markdown: paragraphs, '- ' lists, **bold**, `code`, '### ' headings."""
    out, ul, para = [], False, []

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", t)

    def flush():
        nonlocal para
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para = []
    for line in text.splitlines():
        if line.startswith("- "):
            flush()
            if not ul:
                out.append("<ul>")
                ul = True
            out.append("<li>" + inline(line[2:]) + "</li>")
            continue
        if ul and not line.startswith("  "):
            out.append("</ul>")
            ul = False
        if ul and line.startswith("  "):
            out[-1] = out[-1][:-5] + " " + inline(line.strip()) + "</li>"
            continue
        if line.startswith("### "):
            flush()
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    if ul:
        out.append("</ul>")
    return "\n".join(out)


def fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = ["static", "ssst_it5", "A", "B"]
    col = {"static": "#8a8985", "ssst_it5": "#8a8985", "A": "#2a78d6", "B": "#eb6834"}
    sets = list(res)
    fig, axs = plt.subplots(2, len(sets), figsize=(3.4 * len(sets), 4.6), squeeze=False)
    for j, s in enumerate(sets):
        for i, ph in enumerate(("P", "S")):
            ax = axs[i, j]
            c = res[s]["heldout"]["common"].get(ph)
            if not c:
                ax.axis("off")
                continue
            for k, a in enumerate(arms):
                v, lo, hi = c["mad"][a]
                ax.plot([lo, hi], [k, k], color=col[a], lw=2, solid_capstyle="round")
                ax.plot(v, k, "o", ms=8, color=col[a], mec="white", mew=1.5)
            ax.set_yticks(range(len(arms)))
            ax.set_yticklabels(["static", "SSST 2 km", "A", "B"] if j == 0 else [])
            ax.invert_yaxis()
            ax.set_title(f"{s} {ph} (n={c['n']:,})", fontsize=9)
            ax.grid(axis="x", color="#e5e4e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 1:
                ax.set_xlabel("held-out MAD (ms)", fontsize=8)
            ax.tick_params(labelsize=8)
    fig.suptitle("Held-out-station CC misfit, common rows, 95 % bootstrap intervals (lower is better)", fontsize=9)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


STOP_DIR = T.TT / "score_stop"
STOPS_KM = [16, 8, 4, 2]
SETNAME.update({"georgia": "Strait of Georgia (onshore)", "gorda": "Gorda plate interior (OBS + land)"})


def stop_fig(res, choice):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = list(res)
    fig, axs = plt.subplots(1, len(sets) + 1, figsize=(3.3 * (len(sets) + 1), 3.0), squeeze=False)
    panels = [(s, {f"coh{km}": res[s]["heldout"]["sum_PS"][f"coh{km}"] for km in STOPS_KM},
               {f"ssst{km}": res[s]["heldout"]["sum_PS"][f"ssst{km}"] for km in STOPS_KM}) for s in sets]
    panels.append(("mean of clusters (the rule)", {f"coh{km}": choice["score_coh"][str(km)] for km in STOPS_KM},
                   {f"ssst{km}": choice["score_no_coh"][str(km)] for km in STOPS_KM}))
    for ax, (title, coh, no) in zip(axs[0], panels):
        x = np.arange(len(STOPS_KM))
        for dat, key, col, lab, dx in ((no, "ssst", "#8a8985", "SSST alone", -0.08), (coh, "coh", "#eb6834",
                                                                                       "SSST + coherence", 0.08)):
            v = np.array([dat[f"{key}{km}"] for km in STOPS_KM])
            ax.plot(x + dx, v[:, 0], "-", color=col, lw=1.5)
            ax.vlines(x + dx, v[:, 1], v[:, 2], color=col, lw=2)
            ax.plot(x + dx, v[:, 0], "o", color=col, ms=7, mec="white", mew=1.5, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{km} km" for km in STOPS_KM], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
    axs[0, 0].set_ylabel("held-out MAD P + S (ms)", fontsize=8)
    axs[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle("SSST stopping distance: held-out misfit (lower is better), 95 % bootstrap intervals", fontsize=9)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def stop_section():
    if not (STOP_DIR / "choice.json").exists():
        return ""
    res = json.loads((STOP_DIR / "scores.json").read_text())
    ch = json.loads((STOP_DIR / "choice.json").read_text())
    sets = json.loads((T.SETDIR / "sets.json").read_text())
    arms = [f"ssst{km}" for km in STOPS_KM] + [f"coh{km}" for km in STOPS_KM]
    lab = {**{f"ssst{km}": f"SSST stopped at {km} km" for km in STOPS_KM},
           **{f"coh{km}": f"SSST stopped at {km} km + coherence" for km in STOPS_KM}}
    S = ["<h2>Follow-up: where to stop the SSST schedule (16, 8, 4 or 2 km)</h2>"]
    sp = HERE / "notes" / "v6_termtest_stop_summary.md"
    if sp.exists():
        S.append(md(sp.read_text()))
    S.append(f"<div class='verdict'><b>Choice by the rule written before the scores:</b> stop at "
             f"<b>{ch['chosen_km']} km</b>. Lowest score: {ch['best_km']} km; {html.escape(ch['reason'])}.</div>")
    rows = []
    for km in STOPS_KM:
        d = ch["diffs_coh"].get(f"{km}-{ch['best_km']}")
        rows.append([f"{km} km", fci(ch["score_coh"][str(km)]), fci(d) if d else "best",
                     fci(ch["score_no_coh"][str(km)])])
    S.append("<p class='muted'>Score S(d) = mean over the two clusters of held-out MAD P + MAD S (ms) with the "
             "coherence stage; 95 % paired bootstrap intervals (2,000 resamples of event pairs within each cluster). "
             "Rule: lowest S(d); the next coarser stop is taken if the interval of their difference includes 0.</p>")
    S.append(table(["stop", "S(d) with coherence [95 %]", "S(d) - S(best) [95 %]", "S(d) without coherence [95 %]"],
                   rows))
    S.append(f"<img alt='held-out misfit by stopping distance' src='data:image/png;base64,{stop_fig(res, ch)}'>")
    rows = []
    for s, i in sets.items():
        if i.get("group") != 2:
            continue
        rows.append([f"{s}: {SETNAME.get(s, s)}", html.escape(i["rule"]), f(i["events"]), f"{i['lat']:.3f}, {i['lon']:.3f}",
                     f(i["depth"], 1), f(i["radius_km"], 1), html.escape(", ".join(f"{k} {v}" for k, v in i["tiers"].items())),
                     f"{i['time_min'][:10]} to {i['time_max'][:10]}"])
    S.append(table(["set", "selection", "events", "median lat, lon", "median depth km", "radius km", "v5.1 tiers",
                    "dates"], rows))
    for s, r in res.items():
        h = r["heldout"]
        S.append(f"<h3>{s}: {SETNAME.get(s, s)}</h3>")
        rows = []
        for a_ in arms:
            rows.append([lab[a_], fci(h["common"]["P"][a_]), fci(h["common"]["S"][a_]), fci(h["sum_PS"][a_]),
                         f"{f(h['own'][a_].get('P', {}).get('mad_ms'))} / {f(h['own'][a_].get('S', {}).get('mad_ms'))}",
                         " / ".join(str(x) for x in h["located_per_fold"][a_])])
        rows.append(["v5.1 catalog (in-sample for held-out)", f(h["v5.1"].get("P", {}).get("mad_ms")),
                     f(h["v5.1"].get("S", {}).get("mad_ms")), "", "", ""])
        S.append(table(["arm", "held-out P MAD ms [95 %]", "held-out S MAD ms [95 %]", "P + S [95 %]",
                        "own rows P / S", "located per fold"], rows, hl=(4, 5, 6, 7)))
        S.append(f"<p class='muted'>Common rows: P {r['common_rows']['P']:,}, S {r['common_rows']['S']:,}, "
                 f"{r['common_pairs']:,} event pairs.</p>")
        rows = [[f"{km} km", fci(h["coh_minus_ssst"]["P"][str(km)]), fci(h["coh_minus_ssst"]["S"][str(km)])]
                for km in STOPS_KM]
        S.append("<p class='muted'>Does the coherence stage help? Held-out MAD with coherence minus without, same stop "
                 "(negative = coherence helps).</p>")
        S.append(table(["stop", "P: coh - SSST ms [95 %]", "S: coh - SSST ms [95 %]"], rows))
        c = r["ccfit"]
        rp = c.get("repeaters", {})
        rows = []
        for a_ in arms:
            x = c["arms"].get(a_, {})
            rows.append([lab[a_], fci(c["mad"].get(a_)), fci(c["rms"].get(a_)),
                         fci(rp["sep"].get(a_), 3) if rp.get("n") else "", f(x.get("located")), f(x.get("above_surface")),
                         f(x.get("floor_limited"))])
        rows.append(["v5.1 catalog", "", "", f(rp.get("v5.1_median_km"), 3), "", "", ""])
        S.append(table(["arm (full run)", "CC median per-pair residual ms [95 %]", "CC rms ms [95 %]",
                        f"repeater separation km [95 %] ({rp.get('n', 0)} pairs)", "located", "above surface",
                        "floor-limited"], rows, hl=(4, 5, 6, 7)))
        d = r["depth"]
        rows = []
        for t_ in ("A-D", "A-B"):
            x = d.get(t_, {})
            if not x.get("n"):
                continue
            for a_ in arms:
                rows.append([t_, lab[a_], f(x["n"]), fci(x["bias"].get(a_), 2), fci(x["absdev"].get(a_), 2)])
            rows.append([t_, "v5.1 catalog", f(x["n"]), f(x["all_bias"].get("v5.1"), 2), f(x["all_absdev"].get("v5.1"), 2)])
        S.append(f"<p class='muted'>Depths against {html.escape(d['reference'])} ({d['pairs']} paired events; "
                 f"tiers = v5.1 tier).</p>")
        S.append(table(["tiers", "arm", "events", "median dz km [95 %]", "median |dz| km [95 %]"], rows))
    return "\n".join(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(HERE / "notes" / "v6_termtest_summary.md"))
    a = ap.parse_args()
    res = json.loads((SC_DIR / "scores.json").read_text())
    V = pd.read_csv(SC_DIR / "verdict.csv")
    rec = json.loads((SC_DIR / "recommendation.json").read_text())
    sets = json.loads((T.SETDIR / "sets.json").read_text())
    S = []
    S.append("<h1>Station terms for the v6.0 relocation: test on three new clusters</h1>")
    S.append("<p class='muted'>Track B, 2026-11-25. Branch reloc/v6-termtest, code workflow/10_tomography/"
             "b4_v6_termtest*.py. Data /wd1/mdenolle_data/reloc_v6/termtest/. A = static station terms (v5.1) + "
             "NLL-coherence; B = NLL-SSST stopped at 2 km + NLL-coherence.</p>")
    sp = Path(a.summary)
    if sp.exists():
        S.append("<h2>Summary</h2>" + md(sp.read_text()))
    S.append(f"<div class='verdict'><b>Recommendation (decision rule written before the scores):</b> "
             f"{html.escape(rec['recommendation'])}. Held-out wins: A {rec['heldout_wins_A']}, B {rec['heldout_wins_B']} "
             f"of 3 clusters; events above the surface: A {rec['above_A']}, B {rec['above_B']}.</div>")
    # verdict table
    rows = []
    for r in V.itertuples():
        better = {"A": "<span class='A'>A</span>", "B": "<span class='B'>B</span>", "-": "no material difference"}.get(
            r.better, html.escape(str(r.better)))
        nd = 0 if r.criterion.startswith("3 above") else 2 if "km" in r.criterion else 1
        rows.append([html.escape(r.set), html.escape(r.criterion), f(r.A, nd), f(r.B, nd),
                     f(getattr(r, "diff"), nd) + (f" <span class='muted'>[{f(r.lo, nd)}, {f(r.hi, nd)}]</span>"
                                                  if np.isfinite(r.lo) else ""), better])
    S.append("<h2>Verdict per cluster and criterion</h2>")
    S.append("<p class='muted'>B - A with its 95 % paired bootstrap interval (2,000 resamples). Lower is better for "
             "every row. Material: interval excludes 0 and |B - A| at least 5 % of A (0.2 km for depths).</p>")
    S.append(table(["set", "criterion", "A", "B", "B - A [95 %]", "better"], rows))
    S.append(f"<img alt='held-out MAD per arm' src='data:image/png;base64,{fig(res)}'>")
    # clusters
    S.append("<h2>The three clusters</h2>")
    rows = []
    for s, i in sets.items():
        if i.get("group") == 2:
            continue
        cc = json.loads((T.CC_DIR / "qc" / f"summary_{s}.json").read_text()) if (T.CC_DIR / "qc" / f"summary_{s}.json").exists() else {}
        acc = sum(v.get("n", 0) for v in cc.get("above_by_phase", {}).values())
        rows.append([f"{s}: {SETNAME[s]}", html.escape(i["rule"]), f(i["events"]), f"{i['lat']:.3f}, {i['lon']:.3f}",
                     f(i["depth"], 1), f(i["radius_km"], 1), html.escape(", ".join(f"{k} {v}" for k, v in i["tiers"].items())),
                     f"{i['time_min'][:10]} to {i['time_max'][:10]}", f(acc)])
    S.append(table(["set", "selection", "events", "median lat, lon", "median depth km", "radius km", "v5.1 tiers",
                    "dates", "accepted CC rows"], rows))
    # criterion 1
    S.append("<h2>1. Held-out stations</h2>")
    S.append("<p class='muted'>-18's D-B078 scorer (b5_cmp_common.fold_score): 3 station folds, every arm relocated "
             "from scratch without the held-out stations' picks and CC rows; residual of each held-out CC dt minus "
             "the pair's median training residual; MAD and rms in ms pooled over folds. Common rows: both events "
             "located in every arm of the fold. Own rows: each arm's own located events (as -18's tables).</p>")
    for s in res:
        h = res[s]["heldout"]
        S.append(f"<h3>{s}: {SETNAME[s]}</h3>")
        rows = []
        order = ["static"] + [f"ssst_it{k}" for k in range(6)] + ["A", "B"]
        for arm in order:
            r = [LABEL[arm]]
            for ph in ("P", "S"):
                c = h["common"].get(ph)
                if not c:
                    r += ["", ""]
                    continue
                if arm in c["mad"]:
                    r += [fci(c["mad"][arm]), fci(c["rms"][arm])]
                else:
                    r += [f(c["all_mad"].get(arm)), f(c["all_rms"].get(arm))]
            o = h["own"].get(arm, {})
            r += [f"{f(o.get('P', {}).get('mad_ms'))} / {f(o.get('S', {}).get('mad_ms'))}",
                  " / ".join(str(x) for x in o.get("located_per_fold", []))]
            rows.append(r)
        r = ["B - A"]
        for ph in ("P", "S"):
            c = h["common"].get(ph)
            r += [fci(c["mad"]["B-A"]), fci(c["rms"]["B-A"])] if c else ["", ""]
        rows.append(r + ["", ""])
        v = h["v5.1"]
        rows.append([LABEL["v5.1"], f(v.get("P", {}).get("mad_ms")), f(v.get("P", {}).get("rms_ms")),
                     f(v.get("S", {}).get("mad_ms")), f(v.get("S", {}).get("rms_ms")), "", ""])
        n = " / ".join(f"{ph} {h['common'][ph]['n']:,} rows, {h['common'][ph]['pairs']:,} pairs"
                       for ph in ("P", "S") if ph in h["common"])
        S.append(table(["arm", "P MAD ms [95 %]", "P rms ms [95 %]", "S MAD ms [95 %]", "S rms ms [95 %]",
                        "own rows P / S MAD", "located per fold"], rows, hl=(7, 8)))
        S.append(f"<p class='muted'>Common rows: {n}.</p>")
        rows = []
        for ph in ("P", "S"):
            c = h["common"].get(ph, {})
            for lab, name in (("coh_both", "both events relocated by the coherence stage in A and in B"),
                              ("rest", "other rows")):
                x = c.get(lab, {})
                m = x.get("mad", {})
                rows.append([ph, name, f(x.get("n")), f(x.get("pairs"))] +
                            [fci(m.get(a)) if m else "" for a in ("static", "ssst_it5", "A", "B", "B-A")])
        S.append(table(["phase", "rows", "n", "pairs", "static", "SSST 2 km", "A", "B", "B - A"], rows))
    # criterion 2
    S.append("<h2>2. Fit of the CC differential times (full runs)</h2>")
    S.append("<p class='muted'>b4_v6_ssst_metrics.cc_score: per CC row, observed minus predicted differential time "
             "(with each arm's own corrections), minus the pair median; 'median per-pair residual' = median |.| "
             "(the v6-prep number). Repeaters: CC median >= 0.95 over >= 6 station-phases (qc/&lt;set&gt;_repeaters.csv), "
             "median 3-D separation of the pair.</p>")
    rows = []
    for s in res:
        c = res[s]["ccfit"]
        rp = c.get("repeaters", {})
        for arm in ["static", "ssst_it5", "A", "B", "B-A"]:
            a_ = c["arms"].get(arm, {})
            rows.append([s if arm == "static" else "", LABEL.get(arm, arm), fci(c["mad"].get(arm)),
                         fci(c["rms"].get(arm)), fci(rp["sep"].get(arm), 3) if rp.get("n") else "",
                         f(a_.get("located")), f(a_.get("above_surface")), f(a_.get("floor_limited"))])
        rows.append(["", "v5.1 repeater separation (km)", "", "", f(rp.get("v5.1_median_km"), 3), "", "", ""])
    S.append(table(["set", "arm", "median per-pair residual ms [95 %]", "rms ms [95 %]",
                    "repeater separation km [95 %]", "located", "above surface", "floor-limited"], rows))
    S.append("<p class='muted'>Rows / repeater pairs: " + "; ".join(
        f"{s} {res[s]['ccfit']['common_rows']:,} / {res[s]['ccfit'].get('repeaters', {}).get('n', 0)}" for s in res)
        + ".</p>")
    # criterion 3
    S.append("<h2>3. Depths against published catalogs (full runs)</h2>")
    S.append("<p class='muted'>Event pairs fixed per set (the same events for every arm); tiers = v5.1 tier. "
             "Depth = true depth km below sea level; reference depths as published (ANSS ComCat depths nominally "
             "below sea level; Lange et al. datum read as below sea level, as in the literature report).</p>")
    rows = []
    for s in res:
        d = res[s]["depth"]
        for lab in ("A-D", "A-B"):
            x = d.get(lab, {})
            if not x.get("n"):
                rows.append([s, html.escape(d["reference"]), lab, "0", "", "", "", "", ""])
                continue
            rows.append([s, html.escape(d["reference"]), lab, f(x["n"]),
                         fci(x["bias"]["A"], 2), fci(x["bias"]["B"], 2), fci(x["absdev"]["A"], 2),
                         fci(x["absdev"]["B"], 2), fci(x["absdev"]["B-A"], 2)])
            rows.append(["", "", "", "", f"static {f(x['all_bias'].get('static'), 2)}; SSST {f(x['all_bias'].get('ssst_it5'), 2)}; v5.1 {f(x['all_bias'].get('v5.1'), 2)}",
                         "", f"static {f(x['all_absdev'].get('static'), 2)}; SSST {f(x['all_absdev'].get('ssst_it5'), 2)}; v5.1 {f(x['all_absdev'].get('v5.1'), 2)}", "", ""])
    S.append(table(["set", "reference", "tiers", "events", "A median dz km", "B median dz km", "A median |dz| km",
                    "B median |dz| km", "B - A of median |dz|"], rows))
    S.append(stop_section())
    S.append("<h2>Method</h2>" + md((HERE / "notes" / "v6_termtest_method.md").read_text())
             if (HERE / "notes" / "v6_termtest_method.md").exists() else "")
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width,initial-scale=1'><title>v6 station-term test</title>"
            f"<style>{CSS}</style></head><body>" + "\n".join(S) + "</body></html>")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "termtest_report.html").write_text(page)
    print(REPORT / "termtest_report.html")


if __name__ == "__main__":
    main()
