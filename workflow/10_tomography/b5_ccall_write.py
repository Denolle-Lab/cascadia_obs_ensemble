"""Stage 4 of the all-family CC differential times (D-B093): final tables, dt.cc files and QC per block.

Input: work/raw/<block>/*.parquet (b5_ccall_measure.py), tasks/{events,picks,pairs}.parquet.
Output in <OUT> (default /wd1/mdenolle_data/tomo_v1/cc_dt_all):
* cc_<block>.parquet: one row per pair, station and phase (every row: below the threshold and failed rows too),
  the columns of the D-B078 cc_<set>.parquet (README of cc_dt) plus block, family_id, same_family, dt_cc_hdr,
  t_hdr1/2, tri_sample_frac. `set` holds the block name, pair_source is family or knn.
* dt_<block>.cc: accepted rows, hypoDD format (# ID1 ID2 0.0 / STA DT CC PHA), DT = dt_cc on the v4.1tc time base
  with the v5.1 origin times (OTC 0). Rows across a clock sub-epoch are kept (dt corrected with the D-B090 model).
* work/pairstats/<block>.parquet: per pair n_rows, n_meas, coh (median cc_max, status ok and not at the lag edge,
  as b4_v6_ssst.coherences), n_ok, n_accepted, cc_med_accepted (pair_coherence.parquet in b5_ccall_final.py).
* qc/summary_<block>.json, qc/<block>_by_station.csv, _drift_slopes.csv, _repeaters.csv,
  _repeater_station_outliers.csv (corrected dt) and _repeater_station_outliers_hdr.csv (header base), _qc.png.

Definitions (as b5_cc_write): above = status ok, peak not at the lag edge, cc_max >= 0.7, cc_max - cc_2nd >= 0.1,
same sampling rate. Clock (D-B090 model, tools/pick_timing clock_model.json): clock_c = c(t) = -clock_correction_s
of the pick, clock_dt_s = c1 - c2, clock_epoch = 20 ms sub-epoch id (0 outside the drift), straddles_clock_epoch =
epoch ids differ; dt_cc = dt_cc_hdr - clock_dt_s, dt_cat = v4.1tc catalog dt (= header-base dt - clock_dt_s).
Closure: triangles (A < B < C, same station-phase, all three legs accepted), e = dt_AB + dt_BC - dt_AC. When a
block has more than CAP two-leg paths, the middle events B are sampled at random (fraction tri_sample_frac, the
same for the whole block); n_tri / n_tri_fail20 then count the sampled closed triangles.

  python b5_ccall_write.py --blocks jdf,mendocino | --all
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402
import b5_cc_write as W  # noqa: E402
from b5_cc_study import mad  # noqa: E402

CAP = 30_000_000          # two-leg paths per block for the closure test
CHUNK_ROWS = 2_000_000

STR, F64, F32, I64, I32, BOOL = pa.string(), pa.float64(), pa.float32(), pa.int64(), pa.int32(), pa.bool_()
SCHEMA = pa.schema([
    ("set", STR), ("block", STR), ("pair", I64), ("pair_source", STR), ("family_id", I32), ("same_family", BOOL),
    ("ID1", I64), ("ID2", I64), ("sep_km", F64), ("station", STR), ("net", STR), ("obs_class", STR), ("phase", STR),
    ("status", STR), ("above", BOOL), ("cc_max", F64), ("cc_2nd", F64), ("margin", F64), ("edge", BOOL),
    ("lag_s", F64), ("dt_cat", F64), ("dt_cc", F64), ("dt_diff", F64), ("dt_cc_hdr", F64), ("comps", STR),
    ("channel1", STR), ("channel2", STR), ("chan_mismatch", BOOL), ("rate_mismatch", BOOL), ("rate1", F64),
    ("rate2", F64), ("rate", F64), ("filter", STR), ("band_lo", F64), ("band_hi", F64), ("window", STR),
    ("win_pre", F64), ("win_post", F64), ("maxlag_s", F64), ("pick_time1", STR), ("pick_time2", STR),
    ("t_pick1", F64), ("t_pick2", F64), ("t_hdr1", F64), ("t_hdr2", F64), ("T0_1", F64), ("T0_2", F64),
    ("timing1", STR), ("timing2", STR), ("retime_flag1", STR), ("retime_flag2", STR), ("retimed_any", BOOL),
    ("unretimable_any", BOOL), ("snr1", F64), ("snr2", F64), ("n_tri", I32), ("n_tri_fail20", I32),
    ("tri_sample_frac", F64), ("pick_row1", I64), ("pick_row2", I64), ("pick_prob1", F64), ("pick_prob2", F64),
    ("clock_epoch1", I32), ("clock_epoch2", I32), ("clock_c1", F64), ("clock_c2", F64), ("clock_dt_s", F64),
    ("straddles_clock_epoch", BOOL)])

T = {}


def load_tasks():
    if T:
        return T
    T["ev"] = pd.read_parquet(A.TASKS / "events.parquet")
    T["pk"] = pd.read_parquet(A.TASKS / "picks.parquet")
    T["pr"] = pd.read_parquet(A.TASKS / "pairs.parquet", columns=["pair", "ID1", "ID2", "i1", "i2", "pair_source",
                                                                  "family_id", "same_family", "block", "sep_km",
                                                                  "n_common"])
    pk = T["pk"]
    M = A.clock_model()
    edges = A.clock_epoch_edges(M)
    ep = np.zeros(len(pk), np.int32)
    for st, e in edges.items():
        m = (pk.station == st).values
        ep[m] = A.clock_epoch_id(pk.t_pick.values[m], e)
    T["pk_epoch"] = ep
    T["pk_c"] = -pk.clock_correction_s.values
    T["pick_time"] = pk.pick_datetime.str.strip().values
    return T


def part_files(block):
    return sorted(glob.glob(str(A.WORK / "raw" / block / "*.parquet")))


def closure_block(i1, i2, spc, dt, ok_idx, cap=CAP, seed=0):
    """Closure on accepted rows. i1 < i2 event indices, spc station-phase code, dt corrected CC dt, ok_idx row
    positions. Returns (n_tri, n_fail per row of ok_idx order, triangles DataFrame (spc, e), sample fraction)."""
    E = pd.DataFrame({"a": i1, "b": i2, "s": spc, "dt": dt, "r": np.arange(len(i1))})
    indeg = E.groupby(["s", "b"]).size().rename("nin")
    outdeg = E.groupby(["s", "a"]).size().rename("nout")
    deg = pd.concat([indeg.rename_axis(["s", "n"]), outdeg.rename_axis(["s", "n"])], axis=1).fillna(0)
    deg["paths"] = deg.nin * deg.nout
    total = float(deg.paths.sum())
    q = min(1.0, cap / total) if total > 0 else 1.0
    mids = deg[deg.paths > 0]
    if q < 1:
        rng = np.random.default_rng(seed)
        mids = mids[rng.random(len(mids)) < q]
    mids = mids.reset_index()[["s", "n", "paths"]]
    n_tri = np.zeros(len(i1), np.int32)
    n_fail = np.zeros(len(i1), np.int32)
    tris = []
    # batches of middle nodes with <= 5M paths each
    mids = mids.sort_values(["s", "n"])
    cum = mids.paths.cumsum().values
    bidx = (cum // 5_000_000).astype(int)
    AC = E.set_index(["s", "a", "b"])[["dt", "r"]]
    for _, mb in mids.groupby(bidx):
        sb = pd.MultiIndex.from_frame(mb[["s", "n"]])
        ab = E[pd.MultiIndex.from_arrays([E.s, E.b]).isin(sb)]
        bc = E[pd.MultiIndex.from_arrays([E.s, E.a]).isin(sb)]
        t = ab.merge(bc, left_on=["s", "b"], right_on=["s", "a"], suffixes=("_ab", "_bc"))
        if not len(t):
            continue
        t = t.rename(columns={"a_ab": "A", "b_bc": "Cc"})
        t = t.join(AC, on=["s", "A", "Cc"], how="inner")
        e = t.dt_ab.values + t.dt_bc.values - t.dt.values
        f = np.abs(e) > 0.02
        for col in ("r_ab", "r_bc", "r"):
            n_tri += np.bincount(t[col].values, minlength=len(i1)).astype(np.int32)
            n_fail += np.bincount(t[col].values, weights=f, minlength=len(i1)).astype(np.int32)
        tris.append(pd.DataFrame({"s": t.s.values, "e": e}))
    tri = pd.concat(tris, ignore_index=True) if tris else pd.DataFrame({"s": [], "e": []})
    return n_tri, n_fail, tri, q, total


def build_chunk(block, raw, P, cfg, n_tri, n_fail, q):
    """Full output columns for a chunk of raw rows (one or more part files)."""
    tk = load_tasks()
    pk, pr = tk["pk"], tk["pr"]
    p1, p2 = raw.p1.values, raw.p2.values
    prs = pr.iloc[raw.pair.values]
    d = {}
    d["set"] = np.full(len(raw), block, object)
    d["block"] = d["set"]
    d["pair"] = raw.pair.values
    d["pair_source"] = prs.pair_source.values
    d["family_id"] = prs.family_id.values.astype(np.int32)
    d["same_family"] = prs.same_family.values
    d["ID1"], d["ID2"] = prs.ID1.values, prs.ID2.values
    d["sep_km"] = prs.sep_km.values.astype(np.float64)
    d["station"] = pk.station.values[p1]
    d["net"] = pk.net.values[p1]
    d["obs_class"] = np.where(pd.Series(d["net"]).isin(A.OBS_NETS).values, "seafloor", "land")
    d["phase"] = pk.phase.values[p1]
    st = raw.status.astype(str).values
    d["status"] = st
    edge = raw.edge.fillna(False).astype(bool).values if "edge" in raw else np.zeros(len(raw), bool)
    rm = raw.rate_mismatch.astype("boolean").fillna(False).astype(bool).values
    cm = raw.chan_mismatch.astype("boolean").fillna(False).astype(bool).values
    cc = raw.cc_max.values.astype(float)
    cc2 = raw.cc_2nd.values.astype(float)
    margin = cc - cc2
    th = P
    above = (st == "ok") & ~edge & (cc >= th["cc_min"]) & (margin >= th["margin_min"]) & ~rm
    d["above"] = above
    d["cc_max"], d["cc_2nd"], d["margin"], d["edge"] = cc, cc2, margin, edge
    c1, c2 = tk["pk_c"][p1], tk["pk_c"][p2]
    cdt = c1 - c2
    t1, t2 = pk.t_pick.values[p1], pk.t_pick.values[p2]
    T01, T02 = pk.T0.values[p1], pk.T0.values[p2]
    dt_cat = (t1 - T01) - (t2 - T02)
    dt_hdr = raw.dt_cc_hdr.values.astype(float)
    dt_cc = dt_hdr - cdt
    d["lag_s"] = raw.lag_s.values.astype(float)
    d["dt_cat"], d["dt_cc"], d["dt_diff"], d["dt_cc_hdr"] = dt_cat, dt_cc, dt_cc - dt_cat, dt_hdr
    for c in ("comps", "channel1", "channel2"):
        v = raw[c].values if c in raw else np.full(len(raw), None, object)
        d[c] = np.where(pd.isna(v), None, v)
    d["chan_mismatch"], d["rate_mismatch"] = cm, rm
    for c in ("rate1", "rate2", "rate", "band_lo", "band_hi", "win_pre", "win_post", "maxlag_s", "snr1", "snr2"):
        d[c] = raw[c].values.astype(float) if c in raw else np.full(len(raw), np.nan)
    blo = pd.Series(d["band_lo"]).round(2).astype(str)
    bhi = pd.Series(d["band_hi"]).round(2).astype(str)
    d["filter"] = np.where(st == "ok", "Butterworth band-pass 4 poles zero-phase " + blo + "-" + bhi + " Hz", "")
    wp = pd.Series(d["win_pre"])
    d["window"] = np.where(wp.notna(), "[-" + wp.round(3).astype(str) + ", +" + pd.Series(d["win_post"]).round(3).astype(str)
                           + "] s around pick", "")
    d["pick_time1"] = tk["pick_time"][p1]
    d["pick_time2"] = tk["pick_time"][p2]
    d["t_pick1"], d["t_pick2"] = t1, t2
    d["t_hdr1"], d["t_hdr2"] = pk.t_hdr.values[p1], pk.t_hdr.values[p2]
    d["T0_1"], d["T0_2"] = T01, T02
    tm1, tm2 = pk.timing.values[p1], pk.timing.values[p2]
    d["timing1"], d["timing2"] = tm1, tm2
    fl = pk.flag.values
    d["retime_flag1"] = np.where(pd.isna(fl[p1]), None, fl[p1])
    d["retime_flag2"] = np.where(pd.isna(fl[p2]), None, fl[p2])
    d["retimed_any"] = (tm1 == "retimed") | (tm2 == "retimed")
    d["unretimable_any"] = (tm1 == "unretimable") | (tm2 == "unretimable")
    d["n_tri"], d["n_tri_fail20"] = n_tri, n_fail
    d["tri_sample_frac"] = np.full(len(raw), q)
    d["pick_row1"], d["pick_row2"] = pk.pick_row.values[p1], pk.pick_row.values[p2]
    d["pick_prob1"], d["pick_prob2"] = pk.pick_prob.values[p1], pk.pick_prob.values[p2]
    e1, e2 = tk["pk_epoch"][p1], tk["pk_epoch"][p2]
    d["clock_epoch1"], d["clock_epoch2"] = e1, e2
    d["clock_c1"], d["clock_c2"], d["clock_dt_s"] = c1, c2, cdt
    d["straddles_clock_epoch"] = e1 != e2
    arrays = []
    for f in SCHEMA:
        v = d[f.name]
        arrays.append(pa.array(v, type=f.type, from_pandas=True))
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


def write_dt_lines(tb, fh):
    """Accepted rows of a chunk (pairs are whole inside a part file) -> hypoDD lines; returns (pairs, lines)."""
    a = tb.filter(tb["above"]).select(["pair", "ID1", "ID2", "station", "dt_cc", "cc_max", "phase"]).to_pandas()
    if not len(a):
        return 0, 0
    a = a.sort_values(["pair", "phase", "station"])
    n = 0
    out = []
    for (pr, i1, i2), g in a.groupby(["pair", "ID1", "ID2"], sort=False):
        out.append(f"# {i1} {i2} 0.0\n")
        out.extend(f"{s} {x:.5f} {c:.4f} {p}\n" for s, x, c, p in zip(g.station.values, g.dt_cc.values,
                                                                     g.cc_max.values, g.phase.values))
        n += 1
    fh.write("".join(out))
    return n, len(a)


def run_block(block):
    t0 = time.time()
    logf = A.OUT / "logs" / "write.log"
    files = part_files(block)
    if not files:
        A.log(f"{block}: no raw parts", logf)
        return None
    tk = load_tasks()
    pk, pr = tk["pk"], tk["pr"]
    cfg, P = A.load_params(block)
    # ---------------------------------------------------------------- pass 1: numeric columns of the whole block
    cols = ["pair", "p1", "p2", "status", "edge", "cc_max", "cc_2nd", "rate_mismatch", "dt_cc_hdr", "lag_s",
            "snr1", "snr2", "comps", "channel1", "rate"]
    parts = []
    lens = []
    for f in files:
        x = pd.read_parquet(f, columns=None)
        for c in cols:
            if c not in x:
                x[c] = np.nan
        x = x[cols]
        lens.append(len(x))
        parts.append(x)
    R = pd.concat(parts, ignore_index=True)
    del parts
    R["status"] = R.status.astype(str).astype("category")
    for c in ("comps", "channel1"):
        R[c] = R[c].astype("category")
    R["edge"] = R.edge.astype("boolean").fillna(False).astype(bool)
    R["rate_mismatch"] = R.rate_mismatch.astype("boolean").fillna(False).astype(bool)
    R["margin"] = R.cc_max - R.cc_2nd
    R["above"] = ((R.status == "ok") & ~R.edge & (R.cc_max >= P["cc_min"]) & (R.margin >= P["margin_min"])
                  & ~R.rate_mismatch)
    p1, p2 = R.p1.values, R.p2.values
    cdt = tk["pk_c"][p1] - tk["pk_c"][p2]
    R["dt_cc"] = R.dt_cc_hdr - cdt
    R["dt_cat"] = (pk.t_pick.values[p1] - pk.T0.values[p1]) - (pk.t_pick.values[p2] - pk.T0.values[p2])
    R["dt_diff"] = R.dt_cc - R.dt_cat
    R["clock_dt_s"] = cdt
    R["straddles_clock_epoch"] = tk["pk_epoch"][p1] != tk["pk_epoch"][p2]
    R["spc"] = pk.spc.values[p1]
    R["station"] = pk.station.values[p1]
    R["phase"] = pk.phase.values[p1]
    R["timing1"] = pk.timing.values[p1]
    R["timing2"] = pk.timing.values[p2]
    R["t_pick1"] = pk.t_pick.values[p1]
    R["t_pick2"] = pk.t_pick.values[p2]
    prs = pr.iloc[R.pair.values]
    R["ID1"], R["ID2"] = prs.ID1.values, prs.ID2.values
    R["i1"], R["i2"] = prs.i1.values, prs.i2.values
    R["sep_km"] = prs.sep_km.values
    R["pair_source"] = prs.pair_source.values
    R["net"] = pk.net.values[p1]
    R["obs_class"] = np.where(R.net.isin(A.OBS_NETS), "seafloor", "land")
    # closure on accepted rows
    ai = np.where(R.above.values)[0]
    n_tri_a, n_fail_a, tri, q, total_paths = closure_block(R.i1.values[ai], R.i2.values[ai], R.spc.values[ai],
                                                           R.dt_cc.values[ai], ai)
    n_tri = np.zeros(len(R), np.int32)
    n_fail = np.zeros(len(R), np.int32)
    n_tri[ai], n_fail[ai] = n_tri_a, n_fail_a
    tri["phase"] = np.where(tri.s.values % 2 == 1, "S", "P")
    tri["station"] = np.array(tk_stations())[(tri.s.values // 2).astype(int)] if len(tri) else []
    A.log(f"{block}: pass 1 {len(R)} rows, {len(ai)} accepted, closure {len(tri)} triangles (paths {total_paths:.3g}, "
          f"sample {q:.3g}) {time.time() - t0:.0f} s", logf)
    # pair stats (coherence as b4_v6_ssst.coherences: status ok, not at the lag edge)
    meas = (R.status == "ok") & ~R.edge
    g = R.groupby("pair")
    ps = pd.DataFrame({"n_rows": g.size(), "n_ok": (R.status == "ok").groupby(R.pair).sum(),
                       "n_meas": meas.groupby(R.pair).sum(), "n_accepted": R.above.groupby(R.pair).sum()})
    ps["coh"] = R[meas].groupby("pair").cc_max.median()
    ps["cc_med_accepted"] = R[R.above].groupby("pair").cc_max.median()
    ps = ps.reset_index()
    (A.WORK / "pairstats").mkdir(parents=True, exist_ok=True)
    ps.to_parquet(A.WORK / "pairstats" / f"{block}.parquet", index=False)
    # ---------------------------------------------------------------- pass 2: full tables, written in chunks
    out_pq = A.OUT / f"cc_{block}.parquet"
    out_dt = A.OUT / f"dt_{block}.cc"
    tmp_pq, tmp_dt = out_pq.with_suffix(".tmp"), out_dt.with_suffix(".tmp")
    writer = pq.ParquetWriter(tmp_pq, SCHEMA, compression="zstd")
    n_pairs_dt = n_lines = 0
    off = 0
    with open(tmp_dt, "w") as fh:
        k = 0
        while k < len(files):
            grp, n = [], 0
            while k < len(files) and (n == 0 or n + lens[k] <= CHUNK_ROWS):
                grp.append(files[k])
                n += lens[k]
                k += 1
            raw = pd.concat([pd.read_parquet(f) for f in grp], ignore_index=True)
            tb = build_chunk(block, raw, P, cfg, n_tri[off:off + len(raw)], n_fail[off:off + len(raw)], q)
            assert (tb["pair"].to_numpy() == R.pair.values[off:off + len(raw)]).all()
            off += len(raw)
            writer.write_table(tb, row_group_size=500_000)
            a, b = write_dt_lines(tb, fh)
            n_pairs_dt += a
            n_lines += b
    writer.close()
    os.replace(tmp_pq, out_pq)
    os.replace(tmp_dt, out_dt)
    A.log(f"{block}: wrote {out_pq.name} ({off} rows) and {out_dt.name} ({n_pairs_dt} pairs, {n_lines} lines) "
          f"{time.time() - t0:.0f} s", logf)
    S = qc_block(block, R, tri, q, total_paths, P, cfg, dict(pairs=n_pairs_dt, lines=n_lines))
    A.write_state("write", **{block: dict(rows=int(len(R)), accepted=int(R.above.sum()), dt_pairs=n_pairs_dt,
                                          dt_lines=n_lines, s=round(time.time() - t0))})
    A.log(f"{block}: QC done {time.time() - t0:.0f} s", logf)
    return S


def tk_stations():
    if "stations" not in T:
        T["stations"] = json.loads((A.TASKS / "stations.json").read_text())
    return T["stations"]


def qc_block(block, R, tri, q, total_paths, P, cfg, files):
    out = A.QC
    out.mkdir(parents=True, exist_ok=True)
    ok = R[R.status == "ok"]
    a = R[R.above]
    S = dict(block=block, rows=int(len(R)), status=R.status.value_counts().to_dict(), threshold=P, cfg=cfg,
             dt_file=files)
    S["accepted_frac_of_measured"] = float(len(a) / max(len(ok), 1))
    S["accepted_frac_of_rows"] = float(len(a) / max(len(R), 1))
    pr = load_tasks()["pr"]
    pb = pr[pr.block == block]
    for src in ("family", "knn"):
        dd = R[R.pair_source == src]
        na = dd[dd.above].groupby("pair").size()
        npairs = int((pb.pair_source == src).sum())
        S[f"pairs_{src}"] = dict(pairs=npairs, pairs_with_rows=int(dd.pair.nunique()), obs=int(len(dd)),
                                 obs_ok=int((dd.status == "ok").sum()), obs_above=int(dd.above.sum()),
                                 ge1=int((na >= 1).sum()), ge4=int((na >= 4).sum()), ge8=int((na >= 8).sum()),
                                 cc_median_ok=float(dd[dd.status == "ok"].cc_max.median()),
                                 cc_median_above=float(dd[dd.above].cc_max.median()))
    S["above_by_phase"] = {ph: W.qstats(a[a.phase == ph]) for ph in ("P", "S")}
    S["above_by_class"] = {c: W.qstats(a[a.obs_class == c]) for c in sorted(a.obs_class.unique())}
    S["ok_by_class"] = {c: dict(n=int((ok.obs_class == c).sum()),
                                frac_above=float(a.obs_class.eq(c).sum() / max((ok.obs_class == c).sum(), 1)))
                        for c in sorted(ok.obs_class.unique())}
    S["timing"] = [dict(timing=cls, obs=int(((R.timing1 == cls) | (R.timing2 == cls)).sum()),
                        above=int(((a.timing1 == cls) | (a.timing2 == cls)).sum()))
                   for cls in ("retimed", "checked", "not_checked", "unretimable")]
    S["closure"] = {ph: dict(n_tri=int((tri.phase == ph).sum()), mad_ms=mad(tri[tri.phase == ph].e) * 1e3,
                             gt10_frac=float((tri[tri.phase == ph].e.abs() > 0.01).mean()) if (tri.phase == ph).any() else np.nan,
                             gt20_frac=float((tri[tri.phase == ph].e.abs() > 0.02).mean()) if (tri.phase == ph).any() else np.nan)
                    for ph in ("P", "S")}
    S["closure_sampling"] = dict(two_leg_paths=total_paths, sample_frac=q, cap=CAP)
    # per station-phase
    rows = []
    trs = tri.groupby(["station", "phase"]).e.agg(n_tri="size", tri_mad_ms=lambda e: mad(e) * 1e3,
                                                  tri_gt20=lambda e: float((e.abs() > 0.02).mean())) if len(tri) else None
    for (sta, ph), g in ok.groupby(["station", "phase"]):
        ga = g[g.above]
        r = dict(station=sta, phase=ph, obs_class=g.obs_class.iloc[0],
                 comps=",".join(sorted(g.comps.dropna().astype(str).unique())),
                 channel=",".join(sorted(g.channel1.dropna().astype(str).unique()))[:40], rate=g.rate.median(),
                 n_ok=len(g), n_above=len(ga), frac_above=len(ga) / len(g), cc_median=g.cc_max.median(),
                 diff_median_ms=ga.dt_diff.median() * 1e3 if len(ga) else np.nan,
                 diff_mad_ms=mad(ga.dt_diff) * 1e3 if len(ga) else np.nan,
                 snr_median=float(np.nanmedian(np.r_[g.snr1, g.snr2])))
        if trs is not None and (sta, ph) in trs.index:
            r.update(trs.loc[(sta, ph)].to_dict())
        rows.append(r)
    st = pd.DataFrame(rows).sort_values(["phase", "n_above"], ascending=[True, False])
    st.to_csv(out / f"{block}_by_station.csv", index=False, float_format="%.4g")
    S["station_outliers_closure"] = (st[(st.get("n_tri", pd.Series(0, index=st.index)).fillna(0) >= 30) &
                                        (st.get("tri_gt20", pd.Series(0, index=st.index)).fillna(0) > 0.10)]
                                     [["station", "phase", "n_above", "n_tri", "tri_gt20"]].round(3).to_dict("records")
                                     if "n_tri" in st else [])
    dr = W.drift_slopes(a)
    dr.to_csv(out / f"{block}_drift_slopes.csv", index=False, float_format="%.4g")
    S["drift_slopes_gt_0.03"] = dr[(dr.n >= 30) & ((dr.lo95 > 0.03) | (dr.hi95 < -0.03))].round(4).to_dict("records")
    for lab, col, suf in (("clock_check", "dt_cc", ""), ("clock_check_hdr", "dt_cc_hdr", "_hdr")):
        npc, co, rps = W.clock_outliers(R.assign(dt_cc=R[col]))
        co.to_csv(out / f"{block}_repeater_station_outliers{suf}.csv",
                                                          index=False, float_format="%.5g")
        S[lab] = dict(colocated_pairs=int(npc), outlier_rows=int(len(co)),
                      by_station={k: dict(rows=int(len(g)), of_rows=int(rps.get(k, 0)),
                                          dev_median_s=float(g.dev.median()), phases="".join(sorted(set(g.phase))),
                                          days=sorted(set(g.day1) | set(g.day2))[:20])
                                  for k, g in co.groupby("station") if len(g) >= 2})
    rp = W.repeaters(R)
    rp.to_csv(out / f"{block}_repeaters.csv", index=False, float_format="%.5g")
    S["repeaters"] = dict(n=len(rp), dt_cc_std_median_ms=float(rp.dt_cc_std.median() * 1e3) if len(rp) else np.nan,
                          dt_cat_std_median_ms=float(rp.dt_cat_std.median() * 1e3) if len(rp) else np.nan,
                          sep_km_median=float(rp.sep_km.median()) if len(rp) else np.nan)
    sc = R[R.straddles_clock_epoch]
    S["clock_epochs"] = dict(rows=int(len(sc)), accepted=int(sc.above.sum()),
                             by_station={k: int(v) for k, v in sc.groupby("station").size().items()},
                             abs_clock_dt_gt_0p02=int((R.clock_dt_s.abs() > 0.02).sum()))
    # model check on the accepted rows across epochs: station dt minus the other stations' median, before / after
    if len(sc[sc.above]):
        cons = a[~a.station.isin(list(A.clock_model()))].groupby("pair").dt_cc.agg(["median", "size"])
        x = sc[sc.above].join(cons, on="pair")
        x = x[x["size"] >= 5]
        S["clock_epochs"]["model_check"] = {st_: dict(n=int(len(g)),
                                                      misfit_hdr_median_abs_s=float((g.dt_cc_hdr - g["median"]).abs().median()),
                                                      misfit_corr_median_abs_s=float((g.dt_cc - g["median"]).abs().median()),
                                                      gt50ms_hdr=float(((g.dt_cc_hdr - g["median"]).abs() > 0.05).mean()),
                                                      gt50ms_corr=float(((g.dt_cc - g["median"]).abs() > 0.05).mean()))
                                            for st_, g in x.groupby("station")}
    S["lag_abs_q"] = {ph: [float(v) for v in np.quantile(a[a.phase == ph].lag_s.abs(), [0.5, 0.9, 0.99])]
                      if (a.phase == ph).any() else [] for ph in ("P", "S")}
    S["edge_frac_ok"] = float(ok.edge.mean()) if len(ok) else np.nan
    (out / f"summary_{block}.json").write_text(json.dumps(S, indent=1, default=float))
    # figure
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    bins = np.linspace(0, 1, 51)
    for ph, c in (("P", "C0"), ("S", "C3")):
        ax[0, 0].hist(ok[ok.phase == ph].cc_max, bins, histtype="step", color=c, label=f"{ph} measured")
        ax[0, 0].hist(a[a.phase == ph].cc_max, bins, histtype="stepfilled", alpha=0.3, color=c, label=f"{ph} accepted")
    ax[0, 0].set_xlabel("cc_max")
    ax[0, 0].legend(fontsize=7)
    m = ok.dt_diff.abs() < 1
    if m.any():
        ax[0, 1].hist2d(ok[m].cc_max, ok[m].dt_diff * 1e3, bins=[40, 80], cmap="Greys", cmin=1)
    ax[0, 1].axvline(P["cc_min"], color="r", lw=0.8)
    ax[0, 1].set_xlabel("cc_max")
    ax[0, 1].set_ylabel("dt_cc - dt_cat (ms)")
    ax[0, 1].set_ylim(-500, 500)
    for ph, c in (("P", "C0"), ("S", "C3")):
        e = tri[tri.phase == ph].e * 1e3
        ax[1, 0].hist(np.clip(e, -30, 30), np.linspace(-30, 30, 61), histtype="step", color=c,
                      label=f"{ph} n={len(e)} MAD={mad(e):.2f} ms")
    ax[1, 0].set_xlabel(f"triangle closure (ms), accepted rows (sample {q:.2g})")
    ax[1, 0].legend(fontsize=7)
    for ph, c in (("P", "C0"), ("S", "C3")):
        x = a[a.phase == ph].dt_diff * 1e3
        ax[1, 1].hist(np.clip(x, -400, 400), np.linspace(-400, 400, 81), histtype="step", color=c,
                      label=f"{ph} med {np.median(x) if len(x) else np.nan:.1f} MAD {mad(x):.1f} ms")
    ax[1, 1].set_xlabel("dt_cc - dt_cat (ms), accepted rows")
    ax[1, 1].legend(fontsize=7)
    fig.suptitle(f"{block}: {cfg['band']} Hz, cc >= {P['cc_min']}, margin >= {P['margin_min']}; "
                 f"{len(R):,} rows, {len(a):,} accepted", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{block}_qc.png", dpi=110)
    plt.close(fig)
    # accepted rows for the clock screen (b5_ccall_clock.py)
    (A.WORK / "accepted").mkdir(parents=True, exist_ok=True)
    a[["pair", "ID1", "ID2", "sep_km", "station", "phase", "cc_max", "dt_cc", "dt_cc_hdr", "dt_cat", "t_pick1",
       "t_pick2", "clock_dt_s", "straddles_clock_epoch"]].to_parquet(A.WORK / "accepted" / f"{block}.parquet",
                                                                      index=False)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    blocks = A.BLOCKS if a.all else a.blocks.split(",")
    st = A.read_state("write")
    for b in blocks:
        if not a.redo and b in st and (A.OUT / f"cc_{b}.parquet").exists() and (A.QC / f"summary_{b}.json").exists():
            A.log(f"{b}: done before, skipped")
            continue
        run_block(b)
    st = A.read_state("write")
    A.write_state("write", done=all(b in st for b in blocks if part_files(b)))


if __name__ == "__main__":
    main()
