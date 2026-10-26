"""Package the ver5 Route A amplitudes (amps_v5, 33 GB) for the Zenodo draft, next to the picks.

Source: /wd1/mdenolle_data/amps_v5/amps/amps_{net}.{sta}.{year}.parquet (one row per ver5
pick, 69 windows x 2 bands, float32), made by workflow/05_magnitude/route_a_day_amplitudes.py.

Each amplitude is stored as round(1000 log10 a), as an int16 increment over the previous
window that has a value. The values are running maxima, so most increments are 0 and the
files are ~15 times smaller (about 2.3 GB in all). utils/read_amps_v5.py decodes them; it is
uploaded with the files. Every part is decoded and checked against its source before it is
kept (log10 error <= 0.0005, same missing values, same pick_ids).

Writes to $OUT:
  ver5_amps_{NET}_partNN.parquet   station-year files packed per network, <= ~150 MB each
  ver5_amps_status.csv             picks by network and status (where amplitudes are missing)
  ver5_amps_files.csv              which station-years are in which part (to replace a part later)
  MD5SUMS
"""
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_amps_v5 import SCALE, decode  # noqa: E402

SRC = Path(os.environ.get("AMPS_SRC", "/wd1/mdenolle_data/amps_v5/amps"))
OUT = Path(os.environ.get("AMPS_SHARE_OUT", "/wd1/mdenolle_data/picks_share/amps_v5"))
TARGET = 150e6 / 0.067   # source bytes per part: ~150 MB after encoding (ratio measured on 12 files)
N_PICKS = int(os.environ.get("AMPS_N_PICKS", 60_918_847))
PREFIX = os.environ.get("AMPS_PREFIX", "ver5")   # ver6: the v6 addition alone
TOL = 0.5 / SCALE + 1e-6


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def windows(cols, pre):
    return [float(c[len(pre):-1].replace("p", ".")) for c in cols if c.startswith(pre)]


def encode(t):
    wins = windows(t.column_names, "wa_mm_")
    assert wins == windows(t.column_names, "disp_um_")
    out = {}
    for c in t.column_names:
        if not (c.startswith("wa_mm_") or c.startswith("disp_um_")):
            out[c] = t[c]
    for apre, qpre in (("wa_mm_", "wa_dq_"), ("disp_um_", "disp_dq_")):
        A = np.column_stack([t[f"{apre}{w:05.2f}s".replace(".", "p")].to_numpy(zero_copy_only=False)
                             for w in wins]).astype(np.float64)
        assert not (A < 0).any(), "negative amplitude"
        # exact zeros (flat data): running maxima, so they fill the first windows only;
        # stored as a count of leading zero windows, the windows themselves as missing
        zero = A == 0
        nz = zero.cumprod(axis=1).sum(axis=1)
        assert (zero.sum(axis=1) == nz).all(), "zero amplitude after a positive one"
        out[f"{qpre[:-3]}nzero"] = pa.array(nz.astype(np.int8))
        miss = np.isnan(A) | zero
        L = np.where(miss, 0, np.round(np.log10(np.where(miss, 1, A)) * SCALE))
        prev = np.zeros(len(A))
        D = np.zeros_like(L)
        for i in range(L.shape[1]):          # increment over the last window with a value
            D[:, i] = np.where(miss[:, i], 0, L[:, i] - prev)
            prev = np.where(miss[:, i], prev, L[:, i])
        assert np.abs(D).max(initial=0) < 32767
        for i, w in enumerate(wins):
            out[f"{qpre}{w:05.2f}s".replace(".", "p")] = pa.array(D[:, i].astype(np.int16), mask=miss[:, i])
    md = {"amps_windows_s": json.dumps(wins),
          "amps_encoding": "int16 increments of round(1000*log10(amp)) over the previous window with a "
                           "value; decode: cumsum along windows, /1000, 10**x (utils/read_amps_v5.py)",
          "amps_units": "wa: mm (Wood-Anderson), disp: um (0.05-2 Hz displacement)"}
    return pa.table(out).replace_schema_metadata(md)


def check(src, enc):
    dec = decode(enc)
    assert len(dec) == src.num_rows and (dec.pick_id.values == src["pick_id"].to_numpy()).all()
    worst = 0.0
    for c in src.column_names:
        if c.startswith("wa_mm_") or c.startswith("disp_um_"):
            a = src[c].to_numpy(zero_copy_only=False).astype(np.float64)
            b = dec[c].to_numpy().astype(np.float64)
            assert (np.isnan(a) == np.isnan(b)).all(), c
            assert ((a == 0) == (b == 0)).all(), c
            m = a > 0
            if m.any():
                worst = max(worst, float(np.abs(np.log10(b[m]) - np.log10(a[m])).max()))
    assert worst <= TOL, worst
    return worst


def build(job):
    name, files = job
    out = OUT / name
    tabs = [pq.read_table(f) for f in files]
    src = pa.concat_tables(tabs, promote_options="permissive")
    src = src.take(pa.array(np.argsort(src["pick_id"].to_numpy(), kind="stable")))
    enc = encode(src)
    worst = check(src, enc)
    pq.write_table(enc, out, compression="zstd", compression_level=9, row_group_size=500_000)
    worst = max(worst, check(src, pq.read_table(out)))   # check the file as written
    st = pd.Series(src["status"].to_pandas().astype(str)).value_counts()
    return name, src.num_rows, os.path.getsize(out), worst, st.to_dict(), md5(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(SRC / "amps_*.parquet")))
    by_net = defaultdict(list)
    for f in files:
        by_net[Path(f).name.split("_", 1)[1].split(".")[0]].append(f)
    jobs, rows = [], []
    for net, fs in sorted(by_net.items()):
        parts, cur, size = [], [], 0
        for f in fs:
            if cur and size + os.path.getsize(f) > TARGET:
                parts.append(cur)
                cur, size = [], 0
            cur.append(f)
            size += os.path.getsize(f)
        parts.append(cur)
        for k, p in enumerate(parts, 1):
            name = f"{PREFIX}_amps_{net}_part{k:02d}.parquet"
            jobs.append((name, p))
            rows += [(name, Path(f).name[5:-8]) for f in p]
    print(f"{len(files)} station-year files -> {len(jobs)} parts", flush=True)
    pd.DataFrame(rows, columns=["file", "station_year"]).to_csv(OUT / f"{PREFIX}_amps_files.csv", index=False)

    n, st, sums = 0, [], []
    with Pool(6) as pool:
        for name, nr, sz, worst, counts, h in pool.imap_unordered(build, jobs):
            n += nr
            st += [(name.split("_")[2], k, v) for k, v in counts.items()]
            sums.append(f"{h}  {name}")
            print(f"{name}: {nr:,} picks, {sz / 1e6:.0f} MB, max log10 error {worst:.5f}", flush=True)
    assert n == N_PICKS, n
    st = pd.DataFrame(st, columns=["network", "status", "n"]).groupby(["network", "status"]).n.sum().unstack(fill_value=0)
    st["total"] = st.sum(axis=1)
    st["ok_frac"] = (st["ok"] / st["total"]).round(3)
    st.to_csv(OUT / f"{PREFIX}_amps_status.csv")
    with open(OUT / "MD5SUMS", "w") as f:
        f.write("\n".join(sorted(sums, key=lambda s: s.split()[1])) + "\n")
    print(f"all {n:,} picks; status table and MD5SUMS written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
