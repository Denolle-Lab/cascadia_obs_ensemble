"""Read the shared ver5 Route A amplitude files (ver5_amps_*.parquet) back into amplitudes.

The files store, for each pick and each window W, round(1000 * log10(amplitude)) as an
increment over the previous window that has a value (the first window is stored as is).
Because the amplitudes are running maxima, most increments are 0, which is what makes the
files ~15 times smaller than float32. Decoding: cumulative sum along the windows, / 1000,
10**x. Precision: 0.0005 in log10 (0.12%). Missing values stay missing. Amplitudes that
are exactly 0 (flat data, 19 picks) are kept as the count of leading zero windows (wa_nzero,
disp_nzero) and restored as 0.

    import pandas as pd
    from read_amps_v5 import read_amps
    amps = read_amps("ver5_amps_7D_part01.parquet")           # all windows
    amps = read_amps("ver5_amps_7D_part01.parquet", windows=[2.0, 10.0])
    picks = pd.read_csv("all_picks_all_regions_2010_2016_ver5.csv")
    picks = picks.merge(amps, on="pick_id", how="left")

Needs numpy, pandas, pyarrow.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCALE = 1000.0
QUANT = {"wa": ("wa_dq_", "wa_mm_"), "disp": ("disp_dq_", "disp_um_")}


def windows_of(path):
    """Window lengths (s) stored in a file, from its schema metadata."""
    md = pq.read_schema(path).metadata
    return json.loads(md[b"amps_windows_s"])


def decode(table, windows=None):
    """pyarrow Table of a ver5_amps file -> pandas DataFrame with float32 wa_mm_*, disp_um_*."""
    all_w = json.loads(table.schema.metadata[b"amps_windows_s"])
    keep = all_w if windows is None else [w for w in all_w if any(abs(w - x) < 1e-9 for x in windows)]
    out = {c: table[c].to_pandas() for c in table.column_names
           if not any(c.startswith(q) or c == f"{q[:-3]}nzero" for q, _ in QUANT.values())}
    for qpre, apre in QUANT.values():
        cols = [f"{qpre}{w:05.2f}s".replace(".", "p") for w in all_w]
        D = np.column_stack([table[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in cols])
        miss = np.isnan(D)
        L = np.cumsum(np.where(miss, 0.0, D), axis=1)
        A = np.where(miss, np.nan, 10.0 ** (L / SCALE)).astype(np.float32)
        nz = table[f"{qpre[:-3]}nzero"].to_numpy(zero_copy_only=False)
        for k in np.unique(nz[nz > 0]):       # flat data: amplitude exactly 0 in the first k windows
            A[np.ix_(nz == k, np.arange(k))] = 0.0
        for i, w in enumerate(all_w):
            if w in keep:
                out[f"{apre}{w:05.2f}s".replace(".", "p")] = A[:, i]
    return pd.DataFrame(out)


def read_amps(path, windows=None, columns=None, filters=None):
    """Read one ver5_amps file (or a list of them) and decode it."""
    if isinstance(path, (list, tuple)):
        return pd.concat([read_amps(p, windows, columns, filters) for p in path], ignore_index=True)
    return decode(pq.read_table(path, columns=columns, filters=filters), windows)
