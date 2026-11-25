"""Stage 6 of the all-family CC differential times (D-B093): pair coherence table, loader check, HTML summary.

* pair_coherence.parquet: one row per event pair of the scope (family and knn pairs, also those without a common
  station-phase): ID1, ID2, pair, pair_source, family_id, same_family, block, sep_km, n_stph (common
  station-phases = CC rows), n_ok (status ok), n_meas (status ok and peak not at the lag edge), n_accepted
  (threshold rows written to dt_<block>.cc), coh (median cc_max over the n_meas rows: the pair coherence of
  b4_v6_ssst.coherences), cc_med_accepted.
* loader check: b4_v6_ssst.coherences (worktree /home/mdenolle/wt_v6, branch reloc/v6-prep, imported read only in
  a subprocess with CC_DIR pointed here) on every cc_<block>.parquet; its (ID1, ID2, coh, n) must equal
  pair_coherence rows with n_meas >= 3 -> qc/loader_check.json.
* summary.html: counts, QC per block, clock screen, figures (embedded).

  python b5_ccall_final.py [--skip-loader]
"""
import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402

V6 = Path("/home/mdenolle/wt_v6/workflow/10_tomography")
PY = "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"

LOADER = r"""
import sys, json
from pathlib import Path
sys.path.insert(0, "{v6}")
import pandas as pd
import b4_v6_ssst as V
V.CC_DIR = Path("{out}")
pc = pd.read_parquet("{out}/pair_coherence.parquet", columns=["ID1", "ID2", "block", "coh", "n_meas"])
res = {{}}
for b in {blocks}:
    g = V.coherences(b)
    ref = pc[(pc.block == b) & (pc.n_meas >= V.COH_NMIN)]
    m = g.merge(ref, on=["ID1", "ID2"], how="outer", indicator=True)
    both = m[m._merge == "both"]
    res[b] = dict(loader_pairs=int(len(g)), table_pairs=int(len(ref)), matched=int(len(both)),
                  only_loader=int((m._merge == "left_only").sum()), only_table=int((m._merge == "right_only").sum()),
                  coh_max_abs_diff=float((both.coh_x - both.coh_y).abs().max()) if len(both) else 0.0,
                  n_equal=bool((both.n == both.n_meas).all()), coh_min=V.COH_MIN, coh_nmin=V.COH_NMIN,
                  pairs_coh_ge_min=int((g.coh >= V.COH_MIN).sum()))
print(json.dumps(res))
"""


def pair_coherence():
    pr = pd.read_parquet(A.TASKS / "pairs.parquet", columns=["pair", "ID1", "ID2", "pair_source", "family_id",
                                                             "same_family", "block", "sep_km", "n_common"])
    st = pd.concat([pd.read_parquet(f) for f in sorted((A.WORK / "pairstats").glob("*.parquet"))], ignore_index=True)
    pc = pr.rename(columns={"n_common": "n_stph"}).merge(st, on="pair", how="left")
    for c in ("n_rows", "n_ok", "n_meas", "n_accepted"):
        pc[c] = pc[c].fillna(0).astype(np.int32)
    assert (pc.n_rows == pc.n_stph).all(), "CC rows per pair != common station-phases"
    pc = pc.drop(columns="n_rows")
    pc = pc[["ID1", "ID2", "pair", "pair_source", "family_id", "same_family", "block", "sep_km", "n_stph", "n_ok",
             "n_meas", "n_accepted", "coh", "cc_med_accepted"]]
    pc.to_parquet(A.OUT / "pair_coherence.parquet", index=False)
    return pc


def loader_check(blocks):
    code = LOADER.format(v6=V6, out=A.OUT, blocks=repr(blocks))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       env={"LD_LIBRARY_PATH": "/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib",
                            "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}, cwd="/tmp")
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    res = json.loads(r.stdout.strip().splitlines()[-1])
    ok = all(v["only_loader"] == 0 and v["only_table"] == 0 and v["coh_max_abs_diff"] < 1e-12 and v["n_equal"]
             for v in res.values())
    res = dict(ok=ok, loader=str(V6 / "b4_v6_ssst.py") + ":coherences", git=subprocess.run(
        ["git", "-C", str(V6), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(), blocks=res)
    (A.QC / "loader_check.json").write_text(json.dumps(res, indent=1))
    return res


def img(p):
    return base64.b64encode(Path(p).read_bytes()).decode() if Path(p).exists() else ""


def fmt(x, d=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:,.{d}f}" if isinstance(x, (int, float, np.integer, np.floating)) else str(x)


def html(pc, loader, counts):
    rows = []
    for b in A.BLOCKS:
        f = A.QC / f"summary_{b}.json"
        if not f.exists():
            continue
        S = json.loads(f.read_text())
        c = S["closure"]
        rows.append(dict(block=b, rows=S["rows"], acc=S["status"].get("ok", 0), above=S["dt_file"]["lines"],
                         frac=S["accepted_frac_of_measured"], dtpairs=S["dt_file"]["pairs"],
                         famp=S["pairs_family"]["pairs"], knnp=S["pairs_knn"]["pairs"],
                         cP=c["P"]["mad_ms"], cS=c["S"]["mad_ms"], fP=c["P"]["gt20_frac"], fS=c["S"]["gt20_frac"],
                         q=S["closure_sampling"]["sample_frac"], rep=S["repeaters"]["n"],
                         rstd=S["repeaters"]["dt_cc_std_median_ms"], cstd=S["repeaters"]["dt_cat_std_median_ms"],
                         clk=S["clock_epochs"]["rows"], clka=S["clock_epochs"]["accepted"]))
    T = pd.DataFrame(rows)
    clk = json.loads((A.QC / "clock_screen_cc.json").read_text()) if (A.QC / "clock_screen_cc.json").exists() else {}
    cand = pd.read_csv(A.QC / "clock_candidates_cc.csv") if (A.QC / "clock_candidates_cc.csv").exists() else pd.DataFrame()
    css = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e2dfd8;--acc:#2f6f8f;--card:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--line:#34332f;--acc:#7fb8d4;--card:#1f1f1d}}
:root[data-theme="dark"]{--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--line:#34332f;--acc:#7fb8d4;--card:#1f1f1d}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.15rem;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
p,li{color:var(--fg)}.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}th:first-child,td:first-child{text-align:left}
.wrap{overflow-x:auto}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}.tile b{display:block;font-size:1.3rem}
img{max-width:100%;height:auto;background:#fff;border-radius:6px}code{font-size:12px}
"""
    h = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>CC dt all families</title><style>{css}</style></head><body><main>"]
    h.append("<h1>Waveform CC differential times for every v5.1 family</h1>")
    h.append(f"<p class='mut'>D-B093, catalog v6.0 NLL-coherence stage. Built {time.strftime('%Y-%m-%d %H:%M')}. "
             f"Code: branch tomo/cc-dt-all (b5_ccall_*.py; measurement b5_cc_measure of tomo/cc-dt, unchanged). "
             f"Output: <code>{A.OUT}</code>.</p>")
    tot_rows = int(T.rows.sum()) if len(T) else 0
    tot_acc = int(T.above.sum()) if len(T) else 0
    h.append("<div class='tiles'>")
    for lab, v in (("events", counts["events"]), ("pairs", counts["pairs"]), ("pair-station-phase rows", tot_rows),
                   ("accepted rows (dt.cc lines)", tot_acc), ("pairs with >= 1 accepted", int((pc.n_accepted > 0).sum())),
                   ("pairs with coherence >= 0.5, n >= 3", int(((pc.coh >= 0.5) & (pc.n_meas >= 3)).sum()))):
        h.append(f"<div class='tile'><span class='mut'>{lab}</span><b>{v:,}</b></div>")
    h.append("</div>")
    h.append("<h2>Scope</h2><p>Every event of a v5.1 family (26,852) and every located earthquake within 3 km of a "
             f"family event ({counts['events_neighbour']:,}): {counts['events']:,} events, {counts['picks']:,} picks "
             f"(v4.1tc). Pairs: all pairs inside each family ({counts['pairs_family']:,}) and each event's 20 nearest "
             f"neighbours within 3 km with at least 4 common station-phases ({counts['pairs_knn']:,} more).</p>")
    h.append("<h2>Per block</h2><div class='wrap'><table><tr><th>block</th><th>family pairs</th><th>knn pairs</th>"
             "<th>rows</th><th>measured</th><th>accepted</th><th>accepted / measured</th><th>dt.cc pairs</th>"
             "<th>closure MAD P / S (ms)</th><th>triangles &gt; 20 ms P / S</th><th>triangle sample</th>"
             "<th>repeaters</th><th>repeater spread CC / catalog (ms)</th><th>clock-epoch rows (accepted)</th></tr>")
    for r in T.itertuples():
        h.append(f"<tr><td>{r.block}</td><td>{fmt(r.famp)}</td><td>{fmt(r.knnp)}</td><td>{fmt(r.rows)}</td>"
                 f"<td>{fmt(r.acc)}</td><td>{fmt(r.above)}</td><td>{r.frac:.0%}</td><td>{fmt(r.dtpairs)}</td>"
                 f"<td>{fmt(r.cP, 1)} / {fmt(r.cS, 1)}</td><td>{fmt(100 * r.fP, 1)} % / {fmt(100 * r.fS, 1)} %</td>"
                 f"<td>{r.q:.2g}</td><td>{fmt(r.rep)}</td><td>{fmt(r.rstd, 1)} / {fmt(r.cstd, 1)}</td>"
                 f"<td>{fmt(r.clk)} ({fmt(r.clka)})</td></tr>")
    h.append("</table></div>")
    h.append("<h2>Loader check</h2>")
    h.append(f"<p>b4_v6_ssst.coherences (reloc/v6-prep {loader.get('git', '')[:8]}) read every cc_&lt;block&gt;.parquet: "
             f"{'all pairs and coherences equal to pair_coherence.parquet' if loader.get('ok') else 'MISMATCH, see qc/loader_check.json'}.</p>")
    if clk:
        h.append("<h2>Station clock screen on the CC rows</h2>")
        hd, cr = clk.get("dt_cc_hdr", {}), clk.get("dt_cc", {})
        h.append(f"<p>Co-located pairs: {fmt(hd.get('colocated_pairs'))} ({fmt(hd.get('colocated_rows'))} accepted rows); "
                 f"{fmt(hd.get('stations_screened'))} stations screened by month. Header time base: "
                 f"{fmt(hd.get('candidate_months'))} candidate station-months at {', '.join(hd.get('candidate_stations', []))}. "
                 f"After the D-B090 correction: {fmt(cr.get('candidate_months'))} at {', '.join(cr.get('candidate_stations', []))}. "
                 "Nothing is applied.</p>")
        if len(cand):
            c = cand[cand.base == "dt_cc"]
            h.append("<div class='wrap'><table><tr><th>station</th><th>month</th><th>term (s)</th><th>P (s)</th><th>S (s)</th>"
                     "<th>rows</th><th>clock-like</th><th>pick-residual candidate</th></tr>")
            for r in c.itertuples():
                h.append(f"<tr><td>{r.station}</td><td>{r.month}</td><td>{r.o:+.3f}</td><td>{fmt(r.o_P, 3)}</td>"
                         f"<td>{fmt(r.o_S, 3)}</td><td>{r.n_inf}</td><td>{r.clock_like}</td>"
                         f"<td>{getattr(r, 'pick_timing_candidate_start', '') if isinstance(getattr(r, 'pick_timing_candidate_start', ''), str) else ''}</td></tr>")
            h.append("</table></div>")
    h.append("<h2>QC figures</h2>")
    for b in T.block:
        s = img(A.QC / f"{b}_qc.png")
        if s:
            h.append(f"<h3>{b}</h3><img alt='{b} QC' src='data:image/png;base64,{s}'>")
    h.append("</main></body></html>")
    (A.OUT / "summary.html").write_text("\n".join(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loader", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    pc = pair_coherence()
    blocks = [b for b in A.BLOCKS if (A.OUT / f"cc_{b}.parquet").exists()]
    loader = {} if a.skip_loader else loader_check(blocks)
    print(json.dumps(loader, indent=1))
    counts = json.loads((A.TASKS / "counts.json").read_text())
    html(pc, loader, counts)
    A.write_state("final", done=True, pair_coherence_rows=len(pc), loader_ok=loader.get("ok"),
                  wall_s=round(time.time() - t0))
    A.log(f"final: pair_coherence {len(pc)} rows, loader ok {loader.get('ok')}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
