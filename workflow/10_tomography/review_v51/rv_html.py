"""HTML review reports of the v5.1 location problems: OUT/<folder>/index.html (one page per problem: context, sortable
table, event views with lazy-loaded figures in fig/, review controls stored in localStorage, CSV download) and
OUT/index.html linking the four.

  python rv_html.py [P1 P2 P3 P4]
"""
import html
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import rv_common as C
import rv_event as E
import rv_facts as F
from rv_run import FOLDERS

TITLES = {"P1": "Deep events under Mount St Helens", "P2": "The 2012 Brookings swarm (v1_GO022)",
          "P3": "Events with ComCat moment tensors", "P4": "Deep-pass and far-only events"}
BUILT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
esc = html.escape


def ll(lat, lon, nd=4):
    return f"{abs(lat):.{nd}f}°{'N' if lat >= 0 else 'S'} {abs(lon):.{nd}f}°{'E' if lon >= 0 else 'W'}"


def fnum(x, nd=1, sign=False):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6b6a66;--line:#e2e1dd;--accent:#2a78d6;
--chip:#eef3fb;--ok:#1b7f4b;--warn:#a15c00;--bad:#b3261e;--h1:#2a78d6;--h2:#eb6834;--h3:#1baf7a;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222221;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#9c9b94;--line:#3a3a37;--accent:#3987e5;--chip:#26303d;--ok:#4cc38a;--warn:#e0a030;--bad:#ef6b62;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222221;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9c9b94;--line:#3a3a37;
--accent:#3987e5;--chip:#26303d;--ok:#4cc38a;--warn:#e0a030;--bad:#ef6b62;color-scheme:dark}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif} main{max-width:1500px;margin:0 auto;padding:0 16px 80px}
h1{font-size:22px;margin:18px 0 4px} h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:16px;margin:0 0 6px} a{color:var(--accent)} .sub{color:var(--ink2);margin:0 0 12px}
.bar{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--line);padding:8px 16px;
display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:13px}
.bar input[type=text]{padding:4px 6px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink);min-width:160px}
.bar button,.btn{padding:5px 10px;border:1px solid var(--line);border-radius:4px;background:var(--chip);color:var(--ink);cursor:pointer;font:inherit}
.bar .warn{color:var(--bad)} .prog{font-weight:600}
.note{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);padding:8px 12px;margin:10px 0;border-radius:4px}
.ev{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px;margin:18px 0}
.grp{font-weight:400;color:var(--ink2);font-size:13px;margin-left:8px} .chip{display:inline-block;padding:1px 7px;border-radius:10px;
font-size:12px;background:var(--chip);color:var(--ink2);margin-left:6px} .chip.done{background:var(--ok);color:#fff}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:2px 18px;font-size:13px;margin:6px 0 10px}
.kv div{display:flex;gap:6px;border-bottom:1px dotted var(--line);padding:2px 0} .kv b{color:var(--ink2);font-weight:500;min-width:105px}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0} th,td{border-bottom:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}
th{background:var(--chip);position:sticky;top:44px;cursor:pointer;font-weight:600} td.l,th.l{text-align:left}
.tablewrap{overflow-x:auto;max-height:560px;overflow-y:auto;border:1px solid var(--line);border-radius:4px}
.tablewrap th{top:0}
ul.facts{margin:6px 0 10px 18px;padding:0} ul.facts li{margin:2px 0}
figure{margin:10px 0} figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
figcaption{font-size:12px;color:var(--ink2)}
details{margin:8px 0} summary{cursor:pointer;color:var(--accent)}
.review{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start;background:var(--chip);padding:10px 12px;border-radius:6px;margin-top:10px}
.review fieldset{border:1px solid var(--line);border-radius:4px;padding:4px 10px} .review legend{font-size:12px;color:var(--ink2)}
.review label{margin-right:10px;white-space:nowrap} .review textarea{min-width:320px;flex:1;min-height:52px;border:1px solid var(--line);
border-radius:4px;background:var(--bg);color:var(--ink);font:inherit;padding:4px}
.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
@media (max-width:700px){.review textarea{min-width:100%}}
"""

JS = r"""
(function(){
const REPORT=document.body.dataset.report, PROBLEM=document.body.dataset.problem;
const KEY='v51review:'+REPORT, RKEY='v51review:reviewer';
let ok=true, store={};
try{store=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){ok=false}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(store))}catch(e){ok=false;warn()}}
function warn(){const w=document.getElementById('storewarn');if(w)w.textContent='Answers cannot be stored in this browser (localStorage blocked): download the CSV before closing the page.'}
if(!ok)warn();
const rv=document.getElementById('reviewer');
try{rv.value=localStorage.getItem(RKEY)||''}catch(e){}
rv.addEventListener('input',()=>{try{localStorage.setItem(RKEY,rv.value)}catch(e){}});
const arts=[...document.querySelectorAll('article.ev')];
function val(a,name){const x=a.querySelector('input[name="'+name+'-'+a.dataset.id+'"]:checked');return x?x.value:''}
function status(id){const s=store[id];return s&&(s.eq||s.dep)?((s.eq&&s.dep)?'done':'part'):''}
function paint(id){const a=document.getElementById('ev-'+id);const st=status(id);
 if(a){const c=a.querySelector('.chip.st');c.textContent=st==='done'?'reviewed':(st==='part'?'partly reviewed':'not reviewed');c.classList.toggle('done',st==='done')}
 document.querySelectorAll('td.rs[data-id="'+id+'"]').forEach(td=>{td.textContent=st==='done'?'yes':(st==='part'?'part':'');td.dataset.v=st});}
function progress(){const n=arts.filter(a=>status(a.dataset.id)==='done').length;document.getElementById('prog').textContent=n+' of '+arts.length+' reviewed'}
arts.forEach(a=>{const id=a.dataset.id, s=store[id]||{};
 ['eq','dep'].forEach(k=>{if(s[k]){const x=a.querySelector('input[name="'+k+'-'+id+'"][value="'+s[k]+'"]');if(x)x.checked=true}});
 const ta=a.querySelector('textarea');ta.value=s.comment||'';
 const upd=()=>{store[id]={eq:val(a,'eq'),dep:val(a,'dep'),comment:ta.value,ts:new Date().toISOString()};persist();paint(id);progress()};
 a.querySelectorAll('input[type=radio]').forEach(x=>x.addEventListener('change',upd));ta.addEventListener('input',upd);paint(id)});
progress();
function q(v){v=(v==null?'':String(v));return /[",\n\r]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
document.getElementById('dl').addEventListener('click',()=>{
 const now=new Date().toISOString(), who=rv.value.trim();
 const head=['report','problem','event_id','view_group','origin_time_utc','v51_depth_km','reviewer','earthquake','v51_depth','comment','answered_at_utc','exported_at_utc'];
 const rows=[head.join(',')];
 arts.forEach(a=>{const s=store[a.dataset.id]||{};rows.push([REPORT,PROBLEM,a.dataset.id,a.dataset.group,a.dataset.time,a.dataset.depth,who,s.eq||'',s.dep||'',s.comment||'',s.ts||'',now].map(q).join(','))});
 const blob=new Blob([rows.join('\r\n')+'\r\n'],{type:'text/csv'});const u=URL.createObjectURL(blob);
 const l=document.createElement('a');l.href=u;l.download='v51_review_'+REPORT+'_'+(who||'reviewer').replace(/[^A-Za-z0-9_-]+/g,'_')+'_'+now.slice(0,10)+'.csv';
 document.body.appendChild(l);l.click();setTimeout(()=>{URL.revokeObjectURL(u);l.remove()},1500)});
document.getElementById('next').addEventListener('click',()=>{const a=arts.find(a=>status(a.dataset.id)!=='done');if(a)a.scrollIntoView({behavior:'smooth'})});
document.querySelectorAll('table.sortable').forEach(t=>{t.querySelectorAll('th').forEach((th,i)=>{th.addEventListener('click',()=>{
 const tb=t.tBodies[0];const asc=th.dataset.s!=='a';t.querySelectorAll('th').forEach(h=>delete h.dataset.s);th.dataset.s=asc?'a':'d';
 const rs=[...tb.rows];const key=r=>{const c=r.cells[i];const v=c.dataset.v!==undefined?c.dataset.v:c.textContent;const n=parseFloat(v);return isNaN(n)?v.toLowerCase():n};
 rs.sort((x,y)=>{const a=key(x),b=key(y);if(typeof a==='number'&&typeof b==='number')return asc?a-b:b-a;return asc?String(a).localeCompare(String(b)):String(b).localeCompare(String(a))});
 rs.forEach(r=>tb.appendChild(r))})})});
})();
"""


def page(title, report, problem, body, bar=True):
    b = ""
    if bar:
        b = ("<div class='bar'><span><b>Reviewer</b> <input id='reviewer' type='text' placeholder='your name'></span>"
             "<span id='prog' class='prog'></span><button id='next' type='button'>next not reviewed</button>"
             "<button id='dl' type='button'>download answers (CSV)</button><a href='../index.html'>all reports</a>"
             "<span id='storewarn' class='warn'></span></div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{esc(title)}</title><style>{CSS}</style></head><body data-report='{report}' "
            f"data-problem='{problem}'>{b}<main>{body}</main>" + (f"<script>{JS}</script>" if bar else "") + "</body></html>")


def table(df, cols, link_col=None, cls="sortable", left=()):
    h = "".join(f"<th class='{'l' if c in left else ''}'>{esc(lab)}</th>" for c, lab, _ in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c, lab, fmt in cols:
            v = r.get(c, "")
            if c == "_status":
                tds.append(f"<td class='rs' data-id='{int(r.ID)}'></td>")
                continue
            s = fmt(v) if fmt else ("" if (isinstance(v, float) and not np.isfinite(v)) else str(v))
            dv = ""
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                dv = f" data-v='{float(v)}'"
            if c == link_col and r.get("has_view", True):
                s = f"<a href='#ev-{int(r.ID)}'>{esc(s)}</a>"
            else:
                s = esc(s)
            tds.append(f"<td class='{'l' if c in left else ''}'{dv}>{s}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='tablewrap'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def legend_html():
    return ("<div class='note'><b>How to read an event view.</b> Three hypocentres are compared in the <i>same</i> v5.1 "
            "travel-time grids and station terms, so differences come from the location only: "
            "<span class='sw' style='background:var(--h1)'></span><b>v5.1</b> (NonLinLoc maximum-likelihood point, run "
            "b4_v51_iter1 or its deep pass b4_v51_deep; solid lines, star), "
            "<span class='sw' style='background:var(--h2)'></span><b>v4</b> (catalog v4 hypocentre; dashed, square), "
            "<span class='sw' style='background:var(--h3)'></span><b>ComCat</b> (matched ComCat hypocentre when one "
            "exists; dotted, diamond). Observed picks are black (the D-B072 re-timed picks NonLinLoc used). "
            "Predicted arrival = origin time + grid travel time + station term. Each hypothesis' origin time is refitted "
            "in the v5.1 grids (its own origin time + weighted mean of observed minus predicted, weights 1/(pick error&sup2; + "
            "0.2&sup2;)), so the ticks and the 'OT refit' residuals compare locations, not origin times; residuals with "
            "the catalog origin times are given too. The travel time of each arrival comes from the grid NonLinLoc used "
            "for it in v5.1 (fine 1-km grid, or wide 2-km grid); the grid reader reproduces NonLinLoc's own predicted "
            "times at the v5.1 point to 0.001 s or better for every event (value in the header). Depth: km below sea "
            "level, true depth (Earth flattening undone)."
            "<br><b>Figures.</b> (1) maps, E-W depth section with the NonLinLoc scatter samples, the D-B068 solid surface "
            "and Slab2 (Hayes et al. 2018) where it exists, pick times against distance; (2) record section on the v5.1 "
            "origin-time axis, vertical left, horizontals right, stations sorted by epicentral distance, short coloured "
            "ticks = predicted P (above the trace) and S (below), black = observed pick; stations without a pick "
            "(grey label) are the nearest ones with data; (3) zoom on P (vertical) and S (horizontal, or vertical when "
            "the station has none) at the nearest picked stations. Filters: zero-phase 2-corner Butterworth band-pass "
            "applied forward and backward (acausal: ringing can appear up to about 1/f<sub>low</sub> s before a sharp "
            "onset); band chosen per event by distance and magnitude (2-15 Hz if the median displayed distance is "
            "below 40 km, 1-10 Hz below 150 km, 1-6 Hz beyond, 1-5 Hz for M &ge; 4.5; the low corner stays at 1 Hz or "
            "above because the ocean microseism dominates below it) and stated on each figure. "
            "Each trace is scaled to its own peak in the window (both horizontals of a station share one scale): "
            "amplitudes cannot be compared between rows."
            "<br><b>Waveforms.</b> pnwstore short reads of the records around the window, on the MiniSEED record-header "
            "time base (the D-B072 definition), never a merged day; the picker's band rule (HH, else BH, else EH). NC "
            "and BK are not in pnwstore: they are read from NCEDC FDSN (the project's route for these networks, "
            "utils/data_client.py) on NCEDC's times. No instrument response is removed.</div>")


def review_html(eid):
    q1 = "".join(f"<label><input type='radio' name='eq-{eid}' value='{v}'> {lab}</label>"
                 for v, lab in (("earthquake", "earthquake"), ("not_earthquake", "not an earthquake"), ("unsure", "unsure")))
    q2 = "".join(f"<label><input type='radio' name='dep-{eid}' value='{v}'> {lab}</label>"
                 for v, lab in (("plausible", "v5.1 depth plausible"), ("implausible", "implausible"),
                                ("cannot_tell", "cannot tell")))
    return (f"<div class='review'><fieldset><legend>Is this an earthquake?</legend>{q1}</fieldset>"
            f"<fieldset><legend>Is the v5.1 location plausible?</legend>{q2}</fieldset>"
            f"<textarea placeholder='comment (optional)' aria-label='comment for event {eid}'></textarea></div>")


def flags_of(r):
    f = [k for k in ("floor_limited", "far_only", "deep_pass", "five_phase") if bool(r[k])]
    return ", ".join(f) if f else "none"


def event_html(eid, problem, group, extra):
    J = json.loads((C.WORK / "events" / f"{eid}.json").read_text())
    r = C.catalog().loc[eid]
    folder = C.OUT / FOLDERS[problem]
    if "error" in J:
        return (f"<article class='ev' id='ev-{eid}' data-id='{eid}' data-group='{esc(group)}' data-time='{r.time}' "
                f"data-depth='{r.depth}'><h3>Event {eid}<span class='grp'>{esc(group)}</span><span class='chip st'></span></h3>"
                f"<p>{esc(J['error'])}</p>{review_html(eid)}</article>")
    P = pd.DataFrame(J["phases"])
    hy = J["hyp"]
    nP, nS = int((P.phase == "P").sum()), int((P.phase == "S").sum())
    ot = pd.Timestamp(hy["v5.1"]["OT"])
    m = C.mags()
    kv = [("origin UTC", f"{ot:%Y-%m-%d %H:%M:%S.%f}"[:-4]), ("local", C.local_time(ot)),
          ("v5.1 hypocentre", f"{ll(hy['v5.1']['lat'], hy['v5.1']['lon'])} {hy['v5.1']['depth']:.2f} km"),
          ("expectation depth", f"{r.exp_depth:.2f} km"), ("tier", r.tier), ("sig_z", f"{r.sig_z:.2f} km"),
          ("h_min / h_max", f"{r.h_min:.2f} / {r.h_max:.2f} km"), ("P / S picks", f"{nP} / {nS} (NonLinLoc)"),
          ("azimuthal gap", f"{r.gap:.0f}°"), ("nearest station", f"{r.dist_min:.1f} km"),
          ("dist/depth", f"{r.dist_over_depth:.2f}"), ("rms (NonLinLoc)", f"{r.rms:.3f} s"),
          ("flags", flags_of(r)), ("depth_quality", str(r.depth_quality)),
          ("event_flag", str(r.event_flag) if isinstance(r.event_flag, str) else "none"), ("region", r.region),
          ("magnitude", f"{r.M:.2f} {r.M_type}" if np.isfinite(r.M) else "none"),
          ("surface at epicentre", f"{r.surface_km:+.2f} km"),
          ("v4 hypocentre", f"{ll(r.v4_lat, r.v4_lon)} {r.v4_depth:.2f} km, {str(r.v4_time)[:22]}"),
          ("NonLinLoc workdir", f"{'b4_v51_deep' if r.deep_pass else 'b4_v51_iter1'}/{r.tile}"),
          ("grid check", f"max |grid - NLLoc TTpred| {J.get('tt_check_max_s')} s")]
    if "ComCat" in hy:
        ci = hy["ComCat"]["info"]
        kv.append(("ComCat", f"{ci['id']} {ll(hy['ComCat']['lat'], hy['ComCat']['lon'])} "
                             f"{hy['ComCat']['depth']:.2f} km, {ci['mag']} {ci['magtype']}, {str(ci['t'])[:22]}; {ci['src']}"))
    else:
        kv.append(("ComCat", "no matched ComCat event"))
    kv += extra
    kvh = "".join(f"<div><b>{esc(k)}</b><span>{esc(str(v))}</span></div>" for k, v in kv)
    # hypothesis table
    hrows = []
    for h in E.HYPS:
        if h not in hy:
            continue
        d = hy[h]
        dz = d["depth"] - hy["v5.1"]["depth"] if h != "v5.1" else 0.0
        dh = float(C.hav(hy["v5.1"]["lat"], hy["v5.1"]["lon"], d["lat"], d["lon"])) if h != "v5.1" else 0.0
        dt = (pd.Timestamp(d["OT"]) - ot).total_seconds() if d["OT"] else np.nan
        hrows.append(f"<tr><td class='l'><span class='sw' style='background:var(--h{E.HYPS.index(h) + 1})'></span>{h}</td>"
                     f"<td>{d['lat']:.4f}</td><td>{d['lon']:.4f}</td><td>{d['depth']:.2f}</td><td>{fnum(dz, 1, True)}</td>"
                     f"<td>{fnum(dh, 1)}</td><td>{fnum(dt, 2, True)}</td><td>{fnum(d['ot_refit_shift'], 2, True)}</td>"
                     f"<td>{fnum(d['rms_ot'], 3)}</td><td>{fnum(d['rms_fit'], 3)}</td><td>{fnum(d['rms_fit_P'], 3)}</td>"
                     f"<td>{fnum(d['rms_fit_S'], 3)}</td><td>{d['n']}</td><td>{d['wide_used']}</td><td class='l'>{esc(d['src'])}</td></tr>")
    htab = ("<div class='tablewrap' style='max-height:none'><table><thead><tr><th class='l'>hypothesis</th><th>lat</th>"
            "<th>lon</th><th>depth km</th><th>depth - v5.1</th><th>epicentre to v5.1 km</th><th>OT - v5.1 OT s</th>"
            "<th>OT refit shift s</th><th>rms (own OT) s</th><th>rms (OT refit) s</th><th>rms P</th><th>rms S</th><th>picks</th>"
            "<th>wide-grid arrivals</th><th class='l'>source</th></tr></thead><tbody>" + "".join(hrows) + "</tbody></table></div>")
    facts = "".join(f"<li>{esc(x)}</li>" for x in F.facts(J, r, problem))
    figs = []
    for name, cap in (("geom", "Maps, E-W depth section and pick times against distance."),
                      ("rec", f"Record section, band-pass {J['band'][0]:g}-{J['band'][1]:g} Hz (zero-phase), each row scaled to its own peak."),
                      ("zoom", "P and S windows at the nearest picked stations (OT refit per hypothesis)."),
                      ("raw", "Vertical raw counts (minus the median, no filter): same rows and window as the record section.")):
        f = folder / "fig" / f"{eid}_{name}.webp"
        if name == "raw" and problem != "P1":
            continue
        if f.exists():
            figs.append(f"<figure><a href='fig/{f.name}' target='_blank'><img loading='lazy' src='fig/{f.name}' "
                        f"alt='event {eid} {name}'></a><figcaption>{esc(cap)}</figcaption></figure>")
        else:
            figs.append(f"<p class='note'>figure {name} missing</p>")
    # residual table
    rr = []
    ot_s = ot
    for _, p in P.iterrows():
        tob = (pd.Timestamp(p.tobs) - ot_s).total_seconds()
        rr.append("<tr>" + "".join([f"<td class='l'>{esc(p.sta)}</td>", f"<td class='l'>{p.phase}</td>",
                                    f"<td data-v='{p.dist}'>{p.dist:.1f}</td>", f"<td>{p.err:.2f}</td>",
                                    f"<td>{p.weight:.2f}</td>", f"<td>{tob:.2f}</td>",
                                    f"<td>{'fine' if p.root == 0 else ('wide' if p.root == 1 else '-')}</td>",
                                    f"<td>{p.tcorr:+.2f}</td>", f"<td>{p.res:+.2f}</td>"]
                                   + [f"<td data-v='{p[f'resfit_{h}']}'>{fnum(p[f'resfit_{h}'], 2, True)}</td>" for h in E.HYPS if h in hy]
                                   + [f"<td>{fnum(p[f'res_{h}'], 2, True)}</td>" for h in E.HYPS if h in hy and h != "v5.1"]) + "</tr>")
    hh = "".join(f"<th>{h} (OT refit)</th>" for h in E.HYPS if h in hy) + "".join(f"<th>{h} (own OT)</th>" for h in E.HYPS if h in hy and h != "v5.1")
    rtab = (f"<details><summary>Residuals per pick (observed minus predicted, s; {len(P)} picks)</summary>"
            "<div class='tablewrap'><table class='sortable'><thead><tr><th class='l'>station</th><th class='l'>phase</th>"
            "<th>dist km</th><th>pick error s</th><th>NLLoc weight</th><th>obs after v5.1 OT s</th><th>grid</th>"
            f"<th>station term s</th><th>v5.1 (NLLoc Res)</th>{hh}</tr></thead><tbody>{''.join(rr)}</tbody></table></div>"
            "<p class='sub'>rms rows: " + "; ".join(f"{h}: own OT {fnum(hy[h]['rms_ot'], 3)} s, OT refit "
                                                   f"{fnum(hy[h]['rms_fit'], 3)} s" for h in E.HYPS if h in hy)
            + "</p></details>")
    notes = "".join(f"<p class='note'>{esc(n)}</p>" for n in J.get("notes", []))
    return (f"<article class='ev' id='ev-{eid}' data-id='{eid}' data-group='{esc(group)}' data-time='{r.time}' "
            f"data-depth='{r.depth:.2f}'><h3>Event {eid}<span class='grp'>{esc(group)}</span><span class='chip st'></span></h3>"
            f"<div class='kv'>{kvh}</div>{htab}<ul class='facts'>{facts}</ul>{notes}{''.join(figs)}{rtab}{review_html(eid)}</article>")


def ctx_fig(folder, name, cap):
    f = C.OUT / FOLDERS[folder] / "ctx" / name
    if not f.exists():
        return f"<p class='note'>context figure {esc(name)} not built</p>"
    return (f"<figure><a href='ctx/{name}' target='_blank'><img loading='lazy' src='ctx/{name}' alt='{esc(cap)}'></a>"
            f"<figcaption>{esc(cap)}</figcaption></figure>")


def summary():
    return json.loads((C.WORK / "sets" / "summary.json").read_text())


def ctxstats():
    f = C.WORK / "context_stats.json"
    return json.loads(f.read_text()) if f.exists() else {}


# ------------------------------------------------------------------------------------------------ per problem
def build(problem, only=None, out_name="index.html"):
    S = summary()
    V = pd.read_csv(C.WORK / "sets" / f"{problem}_views.csv")
    folder = C.OUT / FOLDERS[problem]
    folder.mkdir(parents=True, exist_ok=True)
    body = [f"<h1>{problem}. {TITLES[problem]}</h1><p class='sub'>Catalog v5.1 location review, built {BUILT}. "
            f"Requested by the PI on 2026-11-25 (07:08 PST): waveform moveouts and the information needed to judge, for each "
            f"event, whether it is an earthquake and whether its v5.1 location is plausible. This page documents; it "
            f"changes nothing in the catalog. Answers are kept in this browser (localStorage) and leave it only "
            f"through the CSV download.</p>"]
    extra_of = {}
    if problem == "P1":
        s, cs = S["P1"], ctxstats().get("P1", {})
        T = pd.read_csv(C.WORK / "sets" / "P1_table.csv")
        T["dow"] = pd.to_datetime(T.time, format="ISO8601").dt.dayofweek
        body.append("<h2 id='context'>The problem in numbers</h2><div class='note'><ul>"
                    f"<li>Box 46.05-46.30°N, 122.35-122.10°W (the faults piece's box): {s['box_located']:,} located "
                    f"events; {s['deep']} deeper than 30 km ({s['deep_depth_range'][0]:.0f}-{s['deep_depth_range'][1]:.0f} km), "
                    f"tiers " + ", ".join(f"{k} {v}" for k, v in sorted(s['deep_tiers'].items())) +
                    f"; median sig_z {s['deep_sigz_med']:.1f} km; median nearest station {s['deep_distmin_med']:.1f} km.</li>"
                    f"<li>{s['wed_deep']} of the deep events fall on Wednesdays 19:14-19:20 UTC, 0 on Tuesdays at that time "
                    f"({s['wed_all']} box events of any depth in the Wednesday window). Day of week of the deep events "
                    f"(Mon..Sun, UTC): {', '.join(str(cs.get('deep_dow', {}).get(str(k), cs.get('deep_dow', {}).get(k, 0))) for k in range(7))}.</li>"
                    f"<li>Tier C labels among the deep events: {int((T.tier == 'C').sum())}, of which "
                    f"{int(((T.tier == 'C') & (T.sig_z < 14) & (T.dist_min < 4)).sum())} have sig_z below 14 km with a station "
                    f"within 4 km.</li>"
                    "<li>The faults piece removed surface v1x_06 (39 events at about 60 km under the volcano; 64 % weekday "
                    "daytime). Surface v1_ON028 holds 42 events at 23-38 km under the volcano.</li></ul></div>")
        body.append(ctx_fig("P1", "week_hist.webp", "Time of week and hour of day of the box events by v5.1 depth class."))
        if cs.get("pops"):
            pp = pd.DataFrame(cs["pops"])
            body.append("<h2>Populations compared</h2><p class='sub'>Picks from the v4.1t pick file. P-pick span: time between "
                        "the first and the last P pick of an event, for events whose farthest P pick is more than 8 km from "
                        "the v5.1 epicentre. Weekday daytime: Monday-Friday 08-17 local time.</p>")
            body.append(table(pp.assign(ID=0, has_view=False),
                              [("population", "population", None), ("n", "events", None), ("depth_med", "median depth km", None),
                               ("wed_window", "Wed 19:14-19:20 UTC", None), ("wed_19h", "Wed 19 UTC hour", None),
                               ("weekday_daytime_pct", "weekday daytime %", None), ("median_P", "median P picks", None),
                               ("median_S", "median S picks", None), ("pct_no_S", "% without S pick", None),
                               ("P_span_med", "median P-pick span s", None), ("pct_P_span_lt_0p5", "% P span < 0.5 s", None),
                               ("n_span", "events with span", None), ("pct_P_at_UW", "% of P picks at UW stations", None)],
                              left=("population",)))
        # Wednesday table
        W = T[T.wed_window].copy() if "wed_window" in T else T.iloc[:0]
        Vw = V[V.view_group.str.startswith("Wednesday")].copy()
        picks = pd.read_csv(C.PICKS, skipinitialspace=True, usecols=["event_ID", "station", "phase"])
        picks = picks[picks.event_ID.isin(Vw.ID)]
        Vw["P_sta"] = Vw.ID.map(picks[picks.phase == "P"].groupby("event_ID").station.apply(lambda s: " ".join(sorted(x.split('.')[0] for x in s))))
        Vw["S_sta"] = Vw.ID.map(picks[picks.phase == "S"].groupby("event_ID").station.apply(lambda s: " ".join(sorted(x.split('.')[0] for x in s))))
        Vw["has_view"] = True
        body.append("<h2>Wednesday 19:14-19:20 UTC events: which stations pick them</h2>")
        body.append(table(Vw, [("ID", "ID", lambda v: str(int(v))), ("time", "origin UTC", lambda v: str(v)[:21]),
                               ("depth", "depth km", lambda v: fnum(v, 1)), ("v4_depth", "v4 depth", lambda v: fnum(v, 1)),
                               ("tier", "tier", None), ("sig_z", "sig_z", lambda v: fnum(v, 1)), ("M", "M", lambda v: fnum(v, 2)),
                               ("n_P", "P", lambda v: fnum(v, 0)), ("n_S", "S", lambda v: fnum(v, 0)),
                               ("P_span_s", "P-pick span s", lambda v: fnum(v, 2)),
                               ("P_dmin_km", "P stations from km", lambda v: fnum(v, 1)), ("P_dmax_km", "to km", lambda v: fnum(v, 1)),
                               ("P_sta", "P picks at", None), ("S_sta", "S picks at", None), ("_status", "reviewed", None)],
                          link_col="ID", left=("P_sta", "S_sta")))
        cnt = picks.groupby(["station", "phase"]).size().unstack(fill_value=0).sort_values("P", ascending=False).reset_index()
        body.append("<p class='sub'>Picks per station over these events: " + ", ".join(
            f"{r.station} {int(r.get('P', 0))} P / {int(r.get('S', 0))} S" for _, r in cnt.iterrows()) + ".</p>")
        # continuous data
        body.append("<h2>Continuous data on Wednesdays 19:05-19:30 UTC against the Tuesday before</h2>")
        scan = C.WORK / "p1_cont" / "weekly_scan_summary.json"
        body.append(ctx_fig("P1", "raw_examples.webp", "Raw counts around six Wednesday-window events at three analog UW stations and the digital STD.CC (co-located with STD.UW)."))
        body.append(ctx_fig("P1", "pulses.webp", "All Wednesday-window events, 2-15 Hz, aligned on each event's first P pick: analog UW (left four) and digital CC stations."))
        if scan.exists():
            ss = json.loads(scan.read_text())
            body.append("<div class='note'>Weekly scan of every Tuesday and Wednesday 2010-2016 at HSR.UW, STD.UW (analog) and "
                        "STD.CC (digital), 19:05-19:30 UTC: days with a quiet stretch of 10 s or more (1-s windows of the "
                        "2-15 Hz trace with a standard deviation below 10 % of the window's median), and how many of those "
                        "start at 19:14-19:21 UTC. " +
                        "; ".join(f"<b>{k}</b>: Wednesdays {v['Wed']['quiet_ge10']} of {v['Wed']['days']} "
                                  f"({v['Wed']['quiet_ge10_in_14_21']} at 19:14-19:21; first {v['Wed']['first']}, last "
                                  f"{v['Wed']['last']}), Tuesdays {v['Tue']['quiet_ge10']} of {v['Tue']['days']}"
                                  for k, v in ss.items() if not k.startswith("_")) + ". "
                        + ("".join(f"At {k} the stretches starting 19:14-21 UTC ({v['n']} Wednesdays of {v['wednesdays_with_data_in_span']} "
                                   f"with data, {v['first']} to {v['last']}) last a median {v['quiet_median_s']:.0f} s and start "
                                   f"{v['drift_s_per_week']:+.1f} s earlier each week on average (from 19:{v['start_first']:04.1f} to "
                                   f"19:{v['start_last']:04.1f} UTC; rms about the line {v['fit_rms_s']:.0f} s). "
                                   for k, v in ss.get("_drift", {}).items() if k == "STD.UW"))
                        + "HSR.UW also has quiet stretches on Tuesdays and Wednesdays in late 2015 to 2016 at other clock "
                        "times (a different pattern).</div>")
        body.append(ctx_fig("P1", "weekly_scan.webp", "Weekly scan: longest quiet stretch per day and its clock time, Wednesdays vs Tuesdays."))
        for st in ("HSR_UW", "STD_UW", "STD_CC", "SEP_CC"):
            body.append(ctx_fig("P1", f"cont_{st}.webp", f"{st.replace('_', '.')}: 19:05-19:30 UTC, Wednesdays with a catalogued Wednesday-window event (black) and the Tuesdays before (grey)."))
        ck = C.WORK / "p1_clock" / "clock_summary.json"
        if ck.exists():
            k = json.loads(ck.read_text())
            st_, po = k["stations"], k["populations"]
            body.append("<h2>Station clock errors at MH07.XD and MG07.XD (2015-11-21 to 2016-05-02)</h2>")
            body.append("<div class='note'>The CC differential-time QC (tomo/cc-dt, cc_dt/README.md) found record-header clock "
                        "errors at MH07.XD (about 0.45 s) and MG07.XD (about 0.28 s) in co-located event pairs of 2015-11-21 "
                        "to 2016-05-02; the re-timing cannot correct them, so the catalog picks carry them. In the v5.1 "
                        "NonLinLoc residuals of the Mount St Helens box events (daily median minus the daily median of the "
                        "other XD stations, days with 2 or more picks): " +
                        "; ".join(f"<b>{n}</b> offset by more than 0.25 s on {st_[n]['days_abs_offset_gt_0p25']} of "
                                  f"{st_[n]['days_with_2plus_picks_in_period']} days ({st_[n]['offset_day_range'][0]} to "
                                  f"{st_[n]['offset_day_range'][1]}), median {st_[n]['offset_median_on_those_days']:+.2f} s on those days"
                                  for n in ("MH07.XD", "MG07.XD") if st_.get(n, {}).get("offset_day_range")) +
                        ". P and S are offset together. Use by the populations of this report in that period: " +
                        "; ".join(f"{lab}: {v['n_in_period']} of {v['n']} events in the period, {v['n_using_MH07_or_MG07']} with "
                                  f"picks at MH07.XD or MG07.XD" for lab, v in po.items()) +
                        ". Event views with such picks carry a line on it.</div>")
            body.append(ctx_fig("P1", "xd_clock.webp", "NonLinLoc residuals at MH07.XD and MG07.XD against the other XD stations, Mount St Helens box events 2015-09 to 2016-08."))
        # all 331 table
        T["has_view"] = T.ID.isin(V.ID)
        body.append(f"<h2>All {len(T)} events deeper than 30 km (sortable; linked rows have an event view)</h2>")
        body.append(table(T, [("ID", "ID", lambda v: str(int(v))), ("time", "origin UTC", lambda v: str(v)[:19]),
                              ("dow", "weekday", lambda v: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(v)] if np.isfinite(v) else ""),
                              ("depth", "depth km", lambda v: fnum(v, 1)), ("v4_depth", "v4 depth", lambda v: fnum(v, 1)),
                              ("tier", "tier", None), ("sig_z", "sig_z", lambda v: fnum(v, 1)),
                              ("dist_min", "nearest sta km", lambda v: fnum(v, 1)), ("nphs", "phases", lambda v: fnum(v, 0)),
                              ("n_P", "P", lambda v: fnum(v, 0)), ("n_S", "S", lambda v: fnum(v, 0)),
                              ("P_span_s", "P-pick span s", lambda v: fnum(v, 2)), ("P_dmax_km", "P stations to km", lambda v: fnum(v, 0)),
                              ("M", "M", lambda v: fnum(v, 2)), ("wed_window", "Wed window", lambda v: "yes" if v is True or v == "True" else ""),
                              ("deep_pass", "deep pass", lambda v: "yes" if v is True or v == "True" else ""),
                              ("_status", "reviewed", None)], link_col="ID"))
    elif problem == "P2":
        s, cs = S["P2"], ctxstats().get("P2", {})
        body.append("<h2 id='context'>The problem in numbers</h2><div class='note'><ul>"
                    f"<li>{s['n']} members of fault surface v1_GO022 (v0 surface GO020 at about 12 km); years "
                    + ", ".join(f"{k}: {v}" for k, v in s['years'].items()) +
                    f". v5.1 depths {s['depth_range'][0]:.1f}-{s['depth_range'][1]:.1f} km (median {s['depth_med']:.1f}); "
                    f"v4 median {s['v4_depth_med']:.1f} km. depth_quality ok for all; median sig_z {s['sigz_med']:.1f} km; "
                    f"median nearest station {s['distmin_med']:.1f} km.</li>"
                    f"<li>Slab2 slab top at the swarm: {float(C.slab2_depth(np.array([-124.75]), np.array([42.21]))[0]):.1f} km "
                    "(Hayes et al. 2018, cas_slab2_dep 02.24.18, at 42.21°N 124.75°W; the faults piece quoted about 15-20 km).</li>"
                    "</ul></div>")
        body.append(ctx_fig("P2", "depths_map.webp", "v4 vs v5.1 depths, members and stations, depth against time."))
        body.append("<h2>What constrains depth here: S-P at the nearest stations</h2>")
        if cs.get("sp"):
            body.append("<div class='tablewrap' style='max-height:none'><table><thead><tr><th class='l'>station</th><th>dist km</th>"
                        "<th>members with P and S</th><th>observed S-P median s</th><th>IQR s</th><th>predicted at v5.1 median depth</th>"
                        "<th>predicted at v4 median depth</th><th>depth(s) where predicted = observed median (km)</th>"
                        "<th>share of pairs with an NLLoc weight &lt; 0.1</th></tr></thead><tbody>" +
                        "".join(f"<tr><td class='l'>{x['sta']}</td><td>{x['dist']}</td><td>{x['n']}</td><td>{x['sp_obs_med']}</td>"
                                f"<td>{x['sp_obs_iqr'][0]}-{x['sp_obs_iqr'][1]}</td><td>{x['sp_pred_at_v51_median_depth']}</td>"
                                f"<td>{x['sp_pred_at_v4_median_depth']}</td><td>{', '.join(str(z) for z in x['depth_where_pred_equals_obs']) or 'none (observed outside the predicted range)'}</td>"
                                f"<td>{x['share_weight_lt_0p1']}</td></tr>" for x in cs["sp"]) + "</tbody></table></div>")
        body.append(ctx_fig("P2", "sp_depth.webp", "Predicted S-P against depth below the median v5.1 epicentre (v5.1 grids) and the observed S-P of the members."))
        V["has_view"] = True
        body.append(f"<h2>All {len(V)} members (sortable)</h2>")
        body.append(table(V, [("ID", "ID", lambda v: str(int(v))), ("time", "origin UTC", lambda v: str(v)[:19]),
                              ("depth", "depth km", lambda v: fnum(v, 1)), ("v4_depth", "v4 depth", lambda v: fnum(v, 1)),
                              ("shift_z_km", "v5.1 - v4", lambda v: fnum(v, 1, True)), ("tier", "tier", None),
                              ("sig_z", "sig_z", lambda v: fnum(v, 1)), ("dist_min", "nearest sta km", lambda v: fnum(v, 1)),
                              ("gap", "gap", lambda v: fnum(v, 0)), ("n_P", "P", lambda v: fnum(v, 0)), ("n_S", "S", lambda v: fnum(v, 0)),
                              ("M", "M", lambda v: fnum(v, 2)), ("_status", "reviewed", None)], link_col="ID"))
        V = V.sort_values("M", ascending=False)
    elif problem == "P3":
        s = S["P3"]
        body.append("<h2 id='context'>The problem in numbers</h2><div class='note'><ul>"
                    f"<li>{s['n']} located events with a ComCat moment tensor: {s['n_v51_match']} matched in v5.1 "
                    f"(comcat_mt_matched_v51.csv) and {s['n_carried']} whose v4.1 tensor is carried (8 large movers and "
                    f"the v4.1 addition 900001; 900002-900003 have no v5.1 hypocentre and are not shown).</li>"
                    f"<li>v5.1 matches: v5.1 minus ComCat depth median {s['dz_med_v51match']:+.1f} km, origin time "
                    f"{s['dt_med_v51match']:+.2f} s; by region: " + ", ".join(f"{k} {v:+.1f}" for k, v in s['dz_med_by_region'].items()) + " km.</li>"
                    "<li>Large movers (horizontal, depth, 3-D distance km; origin time s): " +
                    "; ".join(f"{m['ID']}: {m['dh_cc']}, {m['dz_cc']:+}, {m['d3_cc']}; {m['dt_cc']:+}" for m in s['movers']) + ".</li>"
                    "<li>ComCat depths of 10 km are often fixed (default) depths; the ComCat hypocentre comes from the "
                    "mag piece's ANSS cache (anss_2010-16.csv).</li></ul></div>")
        body.append(ctx_fig("P3", "comcat_diff.webp", "v5.1 minus ComCat: depth by region, epicentre vs depth, origin time."))
        V["has_view"] = True
        body.append(f"<h2>All {len(V)} tensor events (sortable; page order: large movers first)</h2>")
        body.append(table(V, [("ID", "ID", lambda v: str(int(v))), ("time", "origin UTC", lambda v: str(v)[:19]),
                              ("region", "region", None), ("tier", "tier", None), ("cc_mag", "ComCat M", lambda v: fnum(v, 1)),
                              ("depth", "v5.1 depth", lambda v: fnum(v, 1)), ("cc_depth", "ComCat depth", lambda v: fnum(v, 1)),
                              ("dz_cc", "dz", lambda v: fnum(v, 1, True)), ("dh_cc", "epicentre km", lambda v: fnum(v, 1)),
                              ("dt_cc", "dt s", lambda v: fnum(v, 2, True)), ("v4_depth", "v4 depth", lambda v: fnum(v, 1)),
                              ("sig_z", "sig_z", lambda v: fnum(v, 1)), ("dist_min", "nearest sta km", lambda v: fnum(v, 0)),
                              ("view_group", "group", None), ("_status", "reviewed", None)], link_col="ID", left=("region", "view_group")))
    elif problem == "P4":
        s, cs = S, ctxstats().get("P4", {})
        body.append("<h2 id='context'>The problem in numbers</h2><div class='note'><ul>"
                    f"<li>deep_pass: {s['P4_deep_pass']['n']} events, median depth {s['P4_deep_pass']['depth_med']:.1f} km "
                    f"(v4 {s['P4_deep_pass']['v4_depth_med']:.1f} km).</li>"
                    f"<li>far_only: {s['P4_far_only']['n']} events, median depth {s['P4_far_only']['depth_med']:.1f} km "
                    f"(v4 {s['P4_far_only']['v4_depth_med']:.1f} km).</li>"
                    "<li>Blanco Ridge against Lange et al. (2026), tiers A-C: v5.1 minus Lange depth median +2.85 km, p10 "
                    "-3.0, p90 +13.4 km, n = 1,963 (catalog v5.1 paired_v5.1.md; Lange datum read as km below sea level).</li>"
                    "<li>Event views: about 40 of each flag, stratified by region (at least 2 per region where available) "
                    "and spread over the depth range of each region; events shown in P1-P3 are not repeated.</li></ul></div>")
        body.append(ctx_fig("P4", "depth_hist.webp", "v4 vs v5.1 depth histograms by region."))
        T = pd.read_csv(C.WORK / "sets" / "P4_table.csv")
        T["has_view"] = T.ID.isin(V.ID)
        body.append(f"<h2>All {len(T)} deep_pass and far_only events (sortable; linked rows have an event view)</h2>")
        body.append(table(T, [("ID", "ID", lambda v: str(int(v))), ("flag", "flag", None), ("region", "region", None),
                              ("time", "origin UTC", lambda v: str(v)[:19]), ("depth", "depth km", lambda v: fnum(v, 1)),
                              ("v4_depth", "v4 depth", lambda v: fnum(v, 1)), ("tier", "tier", None),
                              ("sig_z", "sig_z", lambda v: fnum(v, 1)), ("dist_min", "nearest sta km", lambda v: fnum(v, 0)),
                              ("gap", "gap", lambda v: fnum(v, 0)), ("nphs", "phases", lambda v: fnum(v, 0)),
                              ("M", "M", lambda v: fnum(v, 2)), ("_status", "reviewed", None)], link_col="ID",
                          left=("flag", "region")))
        V = V.sort_values(["flag", "region", "depth"])
    if only:
        V = V[V.ID.isin(only)]
    body.append("<h2 id='views'>Event views</h2>")
    body.append(legend_html())
    body.append(f"<p class='sub'>{len(V)} event views: " + ", ".join(f"{k} {v}" for k, v in V.view_group.value_counts().items()) + ".</p>")
    for _, v in V.iterrows():
        eid = int(v.ID)
        extra = [("view group", v.view_group)]
        if problem == "P1":
            extra.append(("Wednesday 19:14-19:20 UTC", "yes" if bool(v.wed_window) else "no"))
        if problem == "P2":
            extra.append(("fault surface", "v1_GO022"))
        if problem == "P3":
            rM = C.mags().loc[eid] if eid in C.mags().index else None
            if rM is not None:
                extra.append(("tensor", f"Mw_mt {rM.Mw_mt}, mt_match {rM.mt_match}, comcat_id {rM.comcat_id}"))
        if problem == "P4":
            extra.append(("flag", v.flag))
        try:
            body.append(event_html(eid, problem, v.view_group, extra))
        except Exception as e:
            body.append(f"<article class='ev' id='ev-{eid}' data-id='{eid}' data-group='{esc(v.view_group)}' data-time='' "
                        f"data-depth=''><h3>Event {eid}<span class='chip st'></span></h3><p class='note'>view failed: "
                        f"{esc(type(e).__name__)}: {esc(str(e))}</p>{review_html(eid)}</article>")
    out = folder / out_name
    out.write_text(page(f"v5.1 review {problem}", FOLDERS[problem], problem, "".join(body)))
    return out, len(V)


def top(counts):
    cards = []
    S = summary()
    lines = {"P1": f"{S['P1']['deep']} of {S['P1']['box_located']:,} box events deeper than 30 km; {S['P1']['wed_deep']} on Wednesdays 19:14-19:20 UTC.",
             "P2": f"{S['P2']['n']} swarm members at {S['P2']['depth_range'][0]:.0f}-{S['P2']['depth_range'][1]:.0f} km in v5.1 (v4 median {S['P2']['v4_depth_med']:.0f} km).",
             "P3": f"{S['P3']['n']} tensor events; v5.1 minus ComCat depth median {S['P3']['dz_med_v51match']:+.1f} km; 8 large movers first.",
             "P4": f"deep_pass {S['P4_deep_pass']['n']} (median {S['P4_deep_pass']['depth_med']:.0f} km), far_only {S['P4_far_only']['n']} (median {S['P4_far_only']['depth_med']:.0f} km)."}
    for p in ("P1", "P2", "P3", "P4"):
        cards.append(f"<div class='card'><h3><a href='{FOLDERS[p]}/index.html'>{p}. {TITLES[p]}</a></h3>"
                     f"<p>{esc(lines[p])}</p><p class='sub'>{counts.get(p, '?')} event views</p></div>")
    body = (f"<h1>Catalog v5.1 location review</h1><p class='sub'>Built {BUILT}. Four reports with waveform moveouts for "
            "human review (PI request, 2026-11-25 07:08 PST). Each folder works on its own (zip and share); answers are "
            "stored per report in the browser and downloaded as CSV from each report.</p><div class='cards'>"
            + "".join(cards) + "</div>" + legend_html())
    (C.OUT / "index.html").write_text(page("v5.1 location review", "top", "", body, bar=False))


if __name__ == "__main__":
    probs = sys.argv[1:] or ["P1", "P2", "P3", "P4"]
    counts = {}
    for p in probs:
        o, n = build(p)
        counts[p] = n
        print(p, o, n, flush=True)
    allc = {}
    for p in ("P1", "P2", "P3", "P4"):
        allc[p] = len(pd.read_csv(C.WORK / "sets" / f"{p}_views.csv"))
    top(allc)
