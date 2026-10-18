"""Package QuakeScope PickBlue OBS picks in the ELEP pick-table layout, for the Zenodo draft.

Source: s3://quakescope-picks-2026/obs/ (public), synced to $QS (picks/, avail/, stations.parquet).
Stations: the Cascadia OBS stations that both ELEP ver5 and PickBlue picked (taken from the
comparison catalog made by compare_elep_pickblue.py), 2010-2016.

Writes to $OUT:
  pickblue_picks_partN.csv.gz   ELEP columns (each part has its own header) + amp_vel, rid
  pickblue_station_days.csv     PickBlue run status per station-day (loaded / no_data / ...)
  pickblue_summary.txt          picks by network and year
  MD5SUMS
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

QS = Path("/wd1/mdenolle_data/quakescope_compare")
OUT = Path("/wd1/mdenolle_data/picks_share/pickblue")
THRESH = 0.2  # PickBlue threshold for P and S (QuakeScope page)
NPARTS = 8    # ~78 MB parts: Zenodo returned 502 on 156 MB PUTs (2026-10-16)
T0, T1 = np.datetime64("2010-01-01", "ms"), np.datetime64("2017-01-01", "ms")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(table, sink):
    """Unquoted CSV like the ELEP tables (no field contains a comma). pyarrow always quotes
    the header, so the header line is written by hand."""
    sink.write((",".join(table.column_names) + "\n").encode())
    pcsv.write_csv(table, sink, write_options=pcsv.WriteOptions(include_header=False, quoting_style="none"))


def iso(col):
    """timestamp[ms] -> '2010-01-01T01:09:14.490000Z', the ELEP time format."""
    return pc.binary_join_element_wise(
        pc.strftime(pc.cast(col, pa.timestamp("us")), format="%Y-%m-%dT%H:%M:%S"), "Z", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    common = pc.unique(pq.read_table(QS / "comparative_catalog_obs_2010_2016.parquet",
                                     columns=["tid"])["tid"])
    print(f"stations: {len(common)}", flush=True)

    t = ds.dataset(QS / "picks", format="parquet", partitioning="hive").to_table(
        columns=["tid", "cha", "pha", "start", "peak", "end", "conf", "amp_vel", "rid"],
        filter=(ds.field("peak") >= pa.scalar(T0)) & (ds.field("peak") < pa.scalar(T1))
        & ds.field("tid").isin(common))
    n0 = t.num_rows
    # de-duplicate on (tid, cha, pha, peak), as the QuakeScope page instructs
    key = np.stack([pc.dictionary_encode(t[c]).combine_chunks().indices.to_numpy(zero_copy_only=False).astype(np.int64)
                    for c in ("tid", "cha", "pha")]
                   + [t["peak"].to_numpy().astype("datetime64[ms]").astype(np.int64)], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    t = t.take(pa.array(np.sort(first)))
    print(f"picks: {n0:,} -> {t.num_rows:,} after de-dup", flush=True)

    parts = pc.split_pattern(t["tid"], ".")
    net = pc.list_element(parts, 0)
    sta = pc.list_element(parts, 1)
    loc = pc.list_element(parts, 2)
    day = pc.cast(pc.floor_temporal(t["peak"], unit="day"), pa.timestamp("ms"))
    t = pa.table({"net": net, "sta": sta, "loc": loc, "cha": t["cha"], "pha": t["pha"], "day": day,
                  "start": t["start"], "peak": t["peak"], "end": t["end"], "conf": t["conf"],
                  "amp_vel": t["amp_vel"], "rid": t["rid"], "tid": t["tid"]})
    t = t.sort_by([("net", "ascending"), ("sta", "ascending"), ("peak", "ascending")])
    n = t.num_rows
    idx = pa.array(np.arange(n, dtype=np.int64))
    # same 15 columns as the ELEP tables (pandas index columns included), then amp_vel and rid
    out = pa.table({
        "": idx, "Unnamed: 0": idx,
        "network": t["net"], "station": t["sta"], "location": t["loc"],
        "band_inst": t["cha"], "label": t["pha"],
        "trace_starttime": iso(t["day"]),
        "trigger_onset": iso(t["start"]), "pick_time": iso(t["peak"]), "trigger_offset": iso(t["end"]),
        "max_prob": t["conf"], "thresh_prob": pa.array(np.full(n, THRESH, np.float32)),
        "pick_id": idx,
        "station_id": pc.binary_join_element_wise(t["net"], t["sta"], t["loc"], "."),
        "amp_vel": t["amp_vel"], "rid": t["rid"],
    })
    bounds = np.linspace(0, n, NPARTS + 1).astype(int)
    for i in range(NPARTS):
        f = OUT / f"pickblue_picks_part{i + 1}.csv.gz"
        with pa.CompressedOutputStream(str(f), "gzip") as fo:
            write_csv(out.slice(bounds[i], bounds[i + 1] - bounds[i]), fo)
        print(f"{f.name}: rows {bounds[i]:,}-{bounds[i + 1] - 1:,}, {f.stat().st_size / 1e6:.0f} MB", flush=True)

    # run status per station-day
    av = ds.dataset(QS / "avail", format="parquet", partitioning="hive").to_table(
        columns=["tid", "date", "status", "npks"])
    d = pc.cast(av["date"], pa.timestamp("ms"))
    av = av.filter(pc.and_(pc.is_in(av["tid"], value_set=common),
                           pc.and_(pc.greater_equal(d, pa.scalar(T0)), pc.less(d, pa.scalar(T1)))))
    av = av.sort_by([("tid", "ascending"), ("date", "ascending")])
    with pa.OSFile(str(OUT / "pickblue_station_days.csv"), "wb") as fo:
        write_csv(av, fo)

    # summary by network and year
    yr = pc.year(t["peak"]).to_numpy()
    netn = t["net"].to_numpy(zero_copy_only=False)
    phn = t["pha"].to_numpy(zero_copy_only=False)
    lines = [f"PickBlue picks: {n:,} ({int((phn == 'P').sum()):,} P, {int((phn == 'S').sum()):,} S) "
             f"at {len(common)} stations, 2010-2016", "",
             "picks by network and year:",
             "network " + "".join(f"{y:>10}" for y in range(2010, 2017)) + f"{'All':>11}"]
    for nt in sorted(set(netn)):
        m = netn == nt
        lines.append(f"{nt:8s}" + "".join(f"{int((m & (yr == y)).sum()):>10,}" for y in range(2010, 2017))
                     + f"{int(m.sum()):>11,}")
    st = av["status"].to_numpy(zero_copy_only=False)
    lines += ["", "station-days by PickBlue status:"]
    lines += [f"  {s:12s} {int((st == s).sum()):>8,}" for s in sorted(set(st))]
    (OUT / "pickblue_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    files = sorted(p for p in OUT.iterdir() if p.name != "MD5SUMS" and p.is_file())
    (OUT / "MD5SUMS").write_text("".join(f"{md5(p)}  {p.name}\n" for p in files))


if __name__ == "__main__":
    sys.exit(main())
