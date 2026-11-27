"""Step 7: evidence page OUT/msh_analog_flags.html (self-contained: figures embedded as PNG data URIs).

  python maf_html.py --examples ID,... --real ID,...
"""
import argparse
import base64
import hashlib
import html
import json
import subprocess

import numpy as np
import pandas as pd

import maf_common as C

CSS = """
:root{--ink:#0b0b0b;--muted:#6b6a66;--grid:#d9d8d4;--bg:#fbfbfa;--card:#ffffff;--accent:#c03a2b;--blue:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#ecebe8;--muted:#a3a29e;--grid:#3a3936;
--bg:#161614;--card:#1f1f1c;--accent:#f07a6a;--blue:#6aa8f0}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.6em 0 .2em}h2{font-size:1.2rem;margin:1.6em 0 .4em;border-bottom:1px solid var(--grid)}
h3{font-size:1rem;margin:1.2em 0 .3em}.muted{color:var(--muted)}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.kpi{background:var(--card);border:1px solid var(--grid);
border-radius:8px;padding:8px 14px;min-width:140px}.kpi b{display:block;font-size:1.6rem}
img{max-width:100%;height:auto;background:#fff;border:1px solid var(--grid);border-radius:4px}
table{border-collapse:collapse;font-size:.82rem;margin:6px 0}th,td{border-bottom:1px solid var(--grid);padding:3px 7px;
text-align:right;white-space:nowrap}th{text-align:right;position:sticky;top:0;background:var(--card)}
td.l,th.l{text-align:left}.scroll{max-height:460px;overflow:auto;border:1px solid var(--grid);border-radius:6px;
max-width:100%}pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--grid);border-radius:6px;
padding:10px;font-size:.8rem}code{font-size:.85em}.cert{color:var(--accent);font-weight:600}
"""


def img(path):
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img alt="{html.escape(path.name)}" src="data:image/png;base64,{b}">'


def table(df, left=()):
    h = "".join(f'<th class="{"l" if c in left else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for r in df.itertuples(index=False):
        tds = []
        for c, v in zip(df.columns, r):
            if isinstance(v, float):
                v = "" if np.isnan(v) else f"{v:.2f}"
            tds.append(f'<td class="{"l" if c in left else ""}">{html.escape(str(v))}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="")
    ap.add_argument("--real", default="")
    ap.add_argument("--probable", default="")
    ap.add_argument("--notes", default="", help="HTML snippet file with the checks narrative")
    a = ap.parse_args()
    built = subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True).stdout.strip()
    summ = json.load(open(C.WORK / "summary.json"))
    cal = json.load(open(C.WORK / "calendar.json"))
    sig = json.load(open(C.WORK / "signal_stations.json"))
    rule = open(C.OUT / "RULE.md").read()
    rule_md5 = hashlib.md5(rule.encode()).hexdigest()
    ev = pd.read_csv(C.WORK / "events_scored.csv", low_memory=False)
    ev["t_origin"] = pd.to_datetime(ev.t_origin, utc=True, format="ISO8601")
    ev["local"] = ev.t_origin.map(lambda t: C.local(t).strftime("%Y-%m-%d %H:%M:%S %Z %a") if pd.notna(t) else "")
    scr = pd.read_csv(C.WORK / "screen.csv")
    scr = scr[scr.signal][["station", "dist_km", "n_data", "n_hit", "n_hit_neg", "cc_med", "amp_med"]]
    nflag = summ["flag_file"]
    cl = summ["classes"]
    yrs = list(range(2010, 2017))
    by = pd.DataFrame({"year": yrs,
                       "certain (flagged)": [summ["certain_by_year"].get(str(y), 0) for y in yrs],
                       "probable": [summ["probable_by_year"].get(str(y), 0) for y in yrs],
                       "mixed": [summ["mixed_by_year"].get(str(y), 0) for y in yrs]})
    cols = ["ID", "cls", "local", "lat", "lon", "depth", "tier", "M", "n_tot", "n_in", "n_nonS", "share", "n_wf",
            "n_match", "in_box", "comcat_id", "window"]
    cand = ev[ev.cls != "none"].sort_values(["cls", "t_origin"])[cols]
    box = ev[ev.in_box & ev.depth.notna()]
    dz = pd.DataFrame([dict(group=g, n=len(x), median=x.depth.median(), p16=x.depth.quantile(.16),
                            p84=x.depth.quantile(.84), deeper_30km=int((x.depth > 30).sum()))
                       for g, x in (("certain (flagged)", box[box.cls == "certain"]),
                                    ("probable", box[box.cls == "probable"]),
                                    ("mixed", box[box.cls == "mixed"]),
                                    ("unflagged box events", box[box.cls == "none"]))])
    deep = box[box.depth > 30]
    deep_share = dict(n_deep=len(deep), certain=int((deep.cls == "certain").sum()),
                      probable=int((deep.cls == "probable").sum()))
    notes = open(a.notes).read() if a.notes else ""
    ctrl = pd.DataFrame(summ["control"]["sets"])
    late = ", ".join(cal["late_local_weeks"])
    irr = pd.DataFrame(cal["irregular"])
    ex = [int(x) for x in a.examples.split(",") if x]
    re_ = [int(x) for x in a.real.split(",") if x]
    pr_ = [int(x) for x in a.probable.split(",") if x]
    fw = C.OUT / "msh_analog_waveform_only.csv"
    wo = pd.read_csv(fw) if fw.exists() else None
    H = []
    H.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MSH analog flags</title><style>{CSS}</style></head>
<body><main>
<h1>Mount St Helens analog test signal: event flags for catalog v6.1</h1>
<p class="muted">Built {html.escape(built)} by workflow/10_tomography/msh_flags/ (branch analysis/msh-analog-flags). Decision D-B108
(PI, 2026-11-26 16:09). Flag only: no event is deleted or moved. Private working page.</p>
<div class="kpis">
<div class="kpi"><b class="cert">{cl.get('certain', 0)}</b>certain: flagged <code>analog_test_signal</code></div>
<div class="kpi"><b>{cl.get('probable', 0)}</b>probable (timing only, not flagged)</div>
<div class="kpi"><b>{cl.get('mixed', 0)}</b>mixed (4+ picks without the signal, not flagged)</div>
<div class="kpi"><b>{nflag['rows']:,}</b>rows in event_flags_v61.csv</div>
</div>
<p>Flag file: <code>/wd1/mdenolle_data/reloc_v61/flags/event_flags_v61.csv</code>, columns ID,event_flag: {nflag['rows']:,} rows =
{nflag['carried_v60']:,} non-empty v6.0 flags carried over unchanged + {nflag['analog']} <code>analog_test_signal</code>
({nflag['analog_and_v60_flag']} of them also had a v6.0 flag, joined with ';'). Events not in the file have no flag.
Candidates with all numbers: <code>msh_analog_candidates.csv</code>; every event in scope: <code>work/events_scored.csv</code>.</p>
<p>In scope: {summ['n_scope']:,} events ({summ['n_box']:,} in the Mount St Helens box, {summ['n_with_S_picks']:,} with picks on a signal
channel); {summ['n_with_pick_in_W']:,} have a pick on the signal. Of the {deep_share['n_deep']} box events deeper than 30 km (v6.0),
{deep_share['certain']} are flagged certain and {deep_share['probable']} probable.</p>
{notes}
<h2>1. The rule (pre-registered)</h2>
<p>Written to <code>RULE.md</code> at 2026-11-26 16:27:27 PST (time from <code>date</code>) and pushed in commit 7ae9b1a9 before any
event was scored; md5 of the file now: <code>{rule_md5}</code> (registered: 61b3513e6de308ba40da3071fcb1d3b7).</p>
<details open><summary>RULE.md</summary><pre>{html.escape(rule)}</pre></details>
<h2>2. The signal</h2>
<p>A calibration-type pulse injected at once on the analog UW channels: a clipped negative box of 1.5 s, an exponential rise to a
plateau of 700-950 counts, the plateau end 27.5 s after the onset and an exponential return. Two pulses every Wednesday, about
20 minutes apart. Template (median of 16 traces, plateau normalised; the negative box is masked in the correlation):</p>
{img(C.FIG / 'template.png')}
<h3>Raw traces of one Wednesday (2012-05-02): every analog station near the volcano, none of the digital ones</h3>
{img(C.FIG / 'raw_allstations.png')}
<h2>3. When: weekday and hour</h2>
<p>Picks of HSR, EDM and STD.UW per weekday and local hour (left) and their share against all other stations' picks (right):</p>
{img(C.FIG / 'hist_week.png')}
<p>Wednesday 11:00-13:30 local in 1-minute bins, per year (orange: mean of the other weekdays at the same clock time):</p>
{img(C.FIG / 'hist_wed_fine.png')}
<h3>Pulse calendar from the continuous data</h3>
<p>Every day 2010-2016 at EDM.UW and every Wednesday at HSR, STD, SHW and ELK.UW, 11:00-13:30 local, scanned with the template.
Weekly pulses from <b>{cal['era_first']}</b> to <b>{cal['era_last']}</b> ({cal['n_weds']} Wednesdays; pulse 1 detected on
{cal['n_detected_weekly']['1']}, pulse 2 on {cal['n_detected_weekly']['2']}, the others predicted), drifting
{cal['drift_s_per_week']['1']:.1f} s per week. The generator keeps the <b>pre-2007 US daylight-saving rule</b>: in the weeks between
the 2007+ change date and the old one the pulses come one hour late in local time ({html.escape(late)}). In device time the
pulses stay at 11:54-12:00 and 12:14-12:20. {cal['n_irregular']} irregular common-mode pulses (other days or times) are windows too.</p>
{img(C.FIG / 'calendar.png')}
<details><summary>Irregular common-mode pulses</summary>{table(irr, left=('local', 'device', 'stations'))}</details>
<h2>4. Where: channels that carry the signal</h2>
<p>{html.escape(sig['rule'])}; 24 reference pulses, every station with picks within 300 km of the summit. {len(scr)} UW analog stations;
no digital station with data (CC, PB, TA, UO, ...) carries it (the XD and YH deployments came after the pulse era). Six carry an inverted copy of about 3 counts (cc_med &lt; 0).</p>
<div class="scroll">{table(scr, left=('station',))}</div>
<h2>5. Events</h2>
{table(by)}
{img(C.FIG / 'years.png')}
<h3>Depth: flagged events against the other box events</h3>
{table(dz, left=('group',))}
{img(C.FIG / 'depth.png')}
<h2>6. False-positive checks</h2>
<p>Waveform test on control events without picks on the signal (rule check b): an event passes WF only if 2 or more of its picks
sit on a pulse edge.</p>
{table(ctrl, left=('set',))}
<p>Picks tested in the windows: {summ['in_window_picks']['n']} ({summ['in_window_picks']['with_data']} with data,
{summ['in_window_picks']['matched']} matched); certain events by number of picks without the signal:
{html.escape(json.dumps(summ['certain_with_nonS']))}.</p>
""")
    H.append("<h2>7. Examples</h2>")
    for i in ex:
        H.append(f"<h3>Flagged event {i}</h3>" + img(C.FIG / f"ev_{i}.png"))
    for i in re_:
        H.append(f"<h3>Real event {i} in a pulse window (not flagged)</h3>" + img(C.FIG / f"ev_{i}.png"))
    if pr_:
        H.append("<h3>Probable events (timing only; not flagged by the rule)</h3><p>Their picks lie in the windows and "
                 "on signal channels only, but not on a pulse edge: a data gap after a pulse, a short box without a "
                 "plateau, glitches inside an irregular test sequence. By eye none is an earthquake.</p>")
        for i in pr_:
            H.append(img(C.FIG / f"ev_{i}.png"))
    if wo is not None:
        H.append(f"""<h2>7b. Post-hoc sweep: pulse edges outside the windows (not part of the rule; no flag)</h2>
<p>Events with no pick in the windows, 3 or more picks on signal channels and a P span of 1 s or less on them
({html.escape(str(len(wo)))} listed): up to 4 picks each (one per station) tested with the same edge test; listed when 2 or more
picks at 2 or more stations sit on a pulse edge. These are artificial pulses at times the weekly calendar and the noon scan do not
cover (irregular calibrations). File: <code>msh_analog_waveform_only.csv</code>. For the PI's review (v6.2); not in the v6.1 flags.</p>
<div class="scroll">{table(wo, left=('local', 'comcat_id'))}</div>""")
    H.append(f"""<h2>8. Candidate list</h2><p>certain, probable and mixed events (v6.0 hypocentres; local time PST/PDT).</p>
<div class="scroll">{table(cand, left=('cls', 'local', 'comcat_id', 'window'))}</div>
</main></body></html>""")
    out = C.OUT / "msh_analog_flags.html"
    out.write_text("\n".join(H))
    print(out, out.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
