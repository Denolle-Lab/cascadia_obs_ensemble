"""v0.4.0 (PI 2026-11-25): the B1 / B1b tomography datasets of v0.3.1 with ONLY the pick times re-timed (D-B072, D-B075).

How pick times enter B1 (b1_select_events.py): picks of the selected events are read from catalog v4.0
(params paths.picks_v4 = Cascadia_picks_catalog_v4.csv, md5 17b7d0bc...), matched to the iter1 NLLoc phases on
(event, station, phase, pick time to 0.1 ms), and written with pick_time (v4.0 pick_datetime, microseconds) and
travel_time_s = pick_time - v4-NLL3D origin time. B1b (b1b_balance.py) keeps a subset of those rows and adds weights.
b2_tomodd_inputs.py uses travel_time_s (absolute.dat and dt.ct); nothing else in the tomography reads the pick time.

Pick ID. The pick files have no ID column; a pick is identified by (event_ID, station, phase, catalog pick time), which
is unique in catalog v4.0 and in v4.1 (no duplicate keys). Re-timed file (D-B075): Cascadia_picks_catalog_v4.1t.csv
(md5 762f18e8...), whose corrected rows carry the new pick_datetime; corrections.csv(.gz) gives old and new time per
corrected row (pick_row = 0-based data row of v4.1). The original time of every v4.1 row is rebuilt (corrections'
old_pick_datetime where applied, else pick_datetime) and joined to v4.0 on the key: 2,503,184 rows in both; v4.1 adds
443 rows (events 900001-3 and 172114), v4.0 has 5 rows (event 172113) that v4.1 dropped. Each tomography pick (B1
kept and rejected, B1b kept and dropped) is then joined to exactly one v4.0 row and one v4.1 row (checked; the script
stops if any pick has no match or more than one).

Outputs (same files as the source datasets, under new names, params retime.datasets):
  <base>/b1_dataset_ct, <base>/b1b_balanced_ct: every file of b1_dataset_c / b1b_balanced_c copied unchanged, except
  picks.csv, picks_rejected.csv (B1) and picks.csv, picks_dropped.csv (B1b): same rows in the same order, pick_time and
  travel_time_s re-timed, plus columns pick_time_v40 (the v0.3.1 time), timing_corrected, timing_correction_s
  (new - old, s, exact to 1 us), retime_epoch_class (listed / found / empty), pick_row_v40, pick_row_v41.
  The NLLoc diagnostic columns (iter1_res_s, iter0_res_s, iter1_station_term_s, iter1_weight) are left as they were:
  they describe the iter0/iter1 NLLoc runs, which used the v4.0 times.
  travel_time_s = round(new pick time - origin time, 4) with the origin time of events.csv; the script first checks that
  this formula reproduces the v0.3.1 travel_time_s of every row from the v4.0 time.
Also writes <out>/dataset/retime_join.csv (one row per B1 pick: key, rows, correction) and join_summary.json.

  python b1t_retime_dataset.py
"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402

P = load_params()
RT = P["retime"]
BASE = Path(P["paths"]["base"])
OUT = Path(RT["out"]) / "dataset"
KEY = ["event_ID", "station", "phase"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ns(s):
    """ISO pick times (with or without Z) -> int64 ns."""
    return pd.to_datetime(pd.Series(s).astype(str).str.rstrip("Z"), format="ISO8601").astype("int64").values


def catalog_rows():
    """One row per (event_ID, station, phase, original time in ns) with row40, row41, correction."""
    for k in ("picks_v40", "picks_v41t"):
        m = md5(RT[k])
        if m != RT[f"{k}_md5"]:
            raise SystemExit(f"{RT[k]}: md5 {m}, expected {RT[k + '_md5']}")
    v40 = pd.read_csv(RT["picks_v40"], skipinitialspace=True, usecols=["event_ID", "pick_datetime", "station", "phase"])
    v41 = pd.read_csv(RT["picks_v41t"], skipinitialspace=True,
                      usecols=["event_ID", "pick_datetime", "station", "phase", "timing_corrected", "timing_correction_s"])
    cor = pd.read_csv(RT["corrections"], usecols=["pick_row", "event_ID", "station", "phase", "old_pick_datetime",
                                                  "new_pick_datetime", "timing_correction_s", "applied", "epoch_class"])
    cor = cor[cor.applied.astype(bool)].set_index("pick_row")
    v41["row41"] = np.arange(len(v41))
    v40["row40"] = np.arange(len(v40))
    tc = v41.timing_corrected.astype(str).str.strip() == "True"
    if tc.sum() != len(cor) or not np.array_equal(np.flatnonzero(tc.values), np.sort(cor.index.values)):
        raise SystemExit("timing_corrected rows of v4.1t differ from the applied rows of corrections.csv")
    c = cor.loc[v41.row41[tc].values]
    for col in ("event_ID", "station", "phase"):
        if not np.array_equal(c[col].values, v41.loc[tc, col].values):
            raise SystemExit(f"corrections.csv {col} differs from v4.1t at the corrected rows")
    if not np.array_equal(c.new_pick_datetime.values, v41.loc[tc, "pick_datetime"].values):
        raise SystemExit("corrections.csv new_pick_datetime differs from v4.1t pick_datetime")
    t_new = ns(v41.pick_datetime)
    t_old = t_new.copy()
    t_old[tc.values] = ns(c.old_pick_datetime.values)
    v41["t_old"], v41["t_new"] = t_old, t_new
    v41["corr_ns"] = t_new - t_old
    v41["epoch_class"] = ""
    v41.loc[tc, "epoch_class"] = c.epoch_class.values
    dcorr = np.abs(v41.corr_ns / 1e9 - v41.timing_correction_s.astype(float)).max()
    if dcorr > 1e-6:
        raise SystemExit(f"new - old time differs from timing_correction_s by up to {dcorr} s")
    v40["t_old"] = ns(v40.pick_datetime)
    k = KEY + ["t_old"]
    for name, d in (("v4.0", v40), ("v4.1t", v41)):
        n = int(d.duplicated(k).sum())
        if n:
            raise SystemExit(f"{name}: {n} duplicate pick keys (event, station, phase, time): pick ID not unique")
    # outer join on the keys only (for the counts): an outer join of the value columns would turn the int64 ns times
    # into float64 and lose the microseconds
    j = v40[k].merge(v41[k], on=k, how="outer", indicator=True)
    summ = {"v40_rows": len(v40), "v41_rows": len(v41), "both": int((j._merge == "both").sum()),
            "v41_only": int((j._merge == "right_only").sum()), "v40_only": int((j._merge == "left_only").sum()),
            "v41_only_events": sorted(map(int, j.loc[j._merge == "right_only", "event_ID"].unique())),
            "v40_only_events": sorted(map(int, j.loc[j._merge == "left_only", "event_ID"].unique())),
            "v41_corrected": int(tc.sum())}
    print(json.dumps(summ), flush=True)
    rows = v40[k + ["row40"]].merge(v41[k + ["row41", "t_new", "corr_ns", "epoch_class"]], on=k, how="inner")
    for c in ("row40", "row41", "t_new", "corr_ns", "t_old"):
        assert rows[c].dtype == np.int64, (c, rows[c].dtype)
    assert len(rows) == summ["both"]
    return rows, summ


def retime(df, rows, origin, name):
    """Join one pick table to the catalog rows (one-to-one) and re-time it."""
    d = df.copy()
    d["t_old"] = ns(d.pick_time)
    m = d.reset_index().merge(rows, on=KEY + ["t_old"], how="left", indicator=True)
    if len(m) != len(d):
        raise SystemExit(f"{name}: join is not one-to-one ({len(m)} rows from {len(d)} picks)")
    miss = int((m._merge != "both").sum())
    if miss:
        raise SystemExit(f"{name}: {miss} picks match no v4.0/v4.1 row")
    m = m.set_index("index").loc[d.index]
    has = d.event_ID.isin(origin.index).values      # rejected picks of events dropped after QC: not in events.csv
    ot = pd.to_datetime(d.event_ID.map(origin)).values.astype("datetime64[ns]").astype("int64")
    tt_old = np.round((d.t_old.values - ot) / 1e9, 4)
    bad = has & (np.abs(tt_old - d.travel_time_s.values) > 5.1e-5)
    if bad.any():
        raise SystemExit(f"{name}: travel_time_s not reproduced from v4.0 time - origin time at {int(bad.sum())} rows")
    assert m.t_new.dtype == np.int64 and m.corr_ns.dtype == np.int64, "times must stay int64 ns (microseconds exact)"
    t_new = m.t_new.values
    out = df.copy()
    out["pick_time_v40"] = out.pick_time
    out["pick_time"] = pd.to_datetime(t_new).strftime("%Y-%m-%dT%H:%M:%S.%f")
    # events in events.csv: new time - origin time (reproduces v0.3.1 exactly from the v4.0 time, checked above);
    # rejected picks of dropped events (no origin in events.csv): v0.3.1 travel time + correction
    out["travel_time_s"] = np.where(has, np.round((t_new - ot) / 1e9, 4),
                                    np.round(d.travel_time_s.values + (t_new - d.t_old.values) / 1e9, 4))
    same = (t_new == d.t_old.values)
    if not np.array_equal(out.travel_time_s.values[same], d.travel_time_s.values[same]):
        raise SystemExit(f"{name}: travel time changed at a pick whose time did not change")
    print(f"{name}: travel time from the origin for {int(has.sum())} rows, exact v0.3.1 reproduction at "
          f"{int((has & (tt_old == d.travel_time_s.values)).sum())}; from v0.3.1 tt + correction for {int((~has).sum())}",
          flush=True)
    out["timing_corrected"] = m.corr_ns.values != 0
    out["timing_correction_s"] = np.round(m.corr_ns.values / 1e9, 6)
    out["retime_epoch_class"] = m.epoch_class.fillna("").values
    out["pick_row_v40"] = m.row40.astype(int).values
    out["pick_row_v41"] = m.row41.astype(int).values
    # rows re-timed by the D-B075 file with a zero correction cannot exist (corrected rows have corr != 0)
    s = dict(picks=len(out), corrected=int(out.timing_corrected.sum()),
             abs_corr_median_s=float(out.timing_correction_s[out.timing_corrected].abs().median())
             if out.timing_corrected.any() else 0.0,
             abs_corr_max_s=float(out.timing_correction_s.abs().max()))
    print(f"{name}: {s}", flush=True)
    return out, s


def write_csv(df, path):
    df.to_csv(path, index=False, float_format=None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summ = catalog_rows()
    srcs = {RT["datasets"]["b1"][0]: RT["datasets"]["b1"][1], RT["datasets"]["b1b"][0]: RT["datasets"]["b1b"][1]}
    stats, joins = {}, []
    outs = []
    for src, dst in srcs.items():
        S, D = BASE / src, BASE / dst
        if D.exists():
            raise SystemExit(f"{D} exists: outputs are never overwritten (remove it by hand if it is a failed build)")
        D.mkdir(parents=True)
        ev = pd.read_csv(S / "events.csv")
        origin = ev.set_index("ID").origin_time
        for f in sorted(S.iterdir()):
            if f.name in ("picks.csv", "picks_rejected.csv", "picks_dropped.csv"):
                df = pd.read_csv(f)
                o, s = retime(df, rows, origin, f"{src}/{f.name}")
                o["travel_time_s"] = o.travel_time_s.round(4)
                write_csv(o, D / f.name)
                stats[f"{src}/{f.name}"] = s
                joins.append(o.assign(file=f"{src}/{f.name}")[["file", *KEY, "pick_time_v40", "pick_time",
                                                                "timing_correction_s", "retime_epoch_class",
                                                                "pick_row_v40", "pick_row_v41"]])
                outs.append(D / f.name)
            elif f.is_file() and not f.name.endswith(".manifest.json") and not f.name.endswith(".out"):
                shutil.copy2(f, D / f.name)
                outs.append(D / f.name)
    # B1b rows are a subset of B1 rows: same correction for the same pick
    b1 = pd.concat([j for j in joins if j.file.iloc[0].startswith(RT["datasets"]["b1"][0])])
    b1b = pd.concat([j for j in joins if j.file.iloc[0].startswith(RT["datasets"]["b1b"][0])])
    chk = b1b.merge(b1, on=KEY + ["pick_time_v40"], suffixes=("", "_b1"))
    if len(chk) != len(b1b) or not np.allclose(chk.timing_correction_s, chk.timing_correction_s_b1, atol=0, rtol=0):
        raise SystemExit("B1b picks do not map onto B1 picks with the same correction")
    jf = OUT / "retime_join.csv.gz"
    pd.concat(joins).to_csv(jf, index=False, compression="gzip")
    summ["files"] = stats
    (OUT / "join_summary.json").write_text(json.dumps(summ, indent=1))
    write_manifest("b1t_retime_dataset", OUT, inputs=[RT["picks_v40"], RT["picks_v41t"], RT["corrections"]] +
                   [BASE / s / f for s in srcs for f in ("events.csv", "picks.csv")],
                   outputs=outs + [jf, OUT / "join_summary.json"], params={"retime": RT}, counts=summ)


if __name__ == "__main__":
    main()
