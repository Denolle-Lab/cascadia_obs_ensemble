"""Band check for the land blocks outside Mount St Helens (cascades, puget, onshore): 3-15 Hz (used) vs 2-12 Hz.

The D-B078 study chose 2-12 Hz for the Mount St Helens set (land stations) and 3-15 Hz for the seafloor sets; the
other land blocks were not tested. This re-measures the family pairs of a block (all, or a random sample of
--max-pairs) with both bands, same windows, lags and threshold, and scores them as b5_cc_study: accepted rows
per phase and the share of closed triangles off by more than 20 ms. Nothing in the outputs changes.

  python b5_ccall_bandcheck.py --blocks cascades,puget,onshore [--max-pairs 20000] [--workers 24]
Output: qc/band_check.csv
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402
import b5_ccall_measure as MM  # noqa: E402
import b5_cc_measure as M  # noqa: E402
from b5_cc_study import closure, mad  # noqa: E402

BANDS = ([3.0, 15.0], [2.0, 12.0])


def run_pairs(args):
    pairs, block = args
    evs = np.unique(np.r_[MM.G["pr_i1"][pairs], MM.G["pr_i2"][pairs]])
    store = MM.Store()
    st = MM.G["starts"]
    for e in evs:
        k = np.arange(st[e], st[e + 1])
        store.load(int(MM.G["ev_id"][e]), zip(MM.G["pk_sta"][k], MM.G["pk_ph"][k], MM.G["pk_t"][k]))
    pid, p1, p2 = MM.obs_rows(pairs)
    out = []
    base = MM.G["cfg"][block]
    for band in BANDS:
        cfg = {**base, "band": band}
        for p, a, b in zip(pid, p1, p2):
            e1, e2 = MM.G["pr_i1"][p], MM.G["pr_i2"][p]
            row = type("R", (), {})()
            row.pair, row.station, row.phase = int(p), MM.G["pk_sta"][a], MM.G["pk_ph"][a]
            row.ID1, row.ID2 = int(MM.G["ev_id"][e1]), int(MM.G["ev_id"][e2])
            row.sp_est1, row.sp_est2 = MM.G["pk_sp"][a], MM.G["pk_sp"][b]
            row.t_pick1, row.t_pick2, row.T01, row.T02 = MM.G["pk_t"][a], MM.G["pk_t"][b], MM.G["pk_T0"][a], MM.G["pk_T0"][b]
            row.dt_cat = (row.t_pick1 - row.T01) - (row.t_pick2 - row.T02)
            try:
                r = M.measure(row, store, cfg)
            except Exception as e:
                r = dict(pair=int(p), status=f"error:{type(e).__name__}")
            r.update(band=f"{band[0]:g}-{band[1]:g}", ID1=row.ID1, ID2=row.ID2, station=row.station, phase=row.phase)
            out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="cascades,puget,onshore")
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    MM.load_tables()
    pr = pd.read_parquet(A.TASKS / "pairs.parquet", columns=["pair", "block", "pair_source", "n_common", "i1", "tile_a",
                                                             "tile_b"])
    _, th = A.load_params("cascades")
    res = []
    for blk in a.blocks.split(","):
        x = pr[(pr.block == blk) & (pr.pair_source == "family") & (pr.n_common > 0)]
        if len(x) > a.max_pairs:
            x = x.sample(a.max_pairs, random_state=0)
        x = x.sort_values(["tile_a", "tile_b", "i1"])
        chunks = [(c.pair.values, blk) for _, c in x.groupby(["tile_a", "tile_b"])]
        with ProcessPoolExecutor(a.workers) as ex:
            d = pd.concat(list(ex.map(run_pairs, chunks)), ignore_index=True)
        d["edge"] = d.edge.astype("boolean").fillna(False).astype(bool)
        d["rate_mismatch"] = d.rate_mismatch.astype("boolean").fillna(False).astype(bool)
        d["above"] = ((d.status == "ok") & ~d.edge & (d.cc_max >= th["cc_min"]) & (d.cc_max - d.cc_2nd >= th["margin_min"])
                      & ~d.rate_mismatch)
        for band, g in d.groupby("band"):
            acc = g[g.above]
            cl = closure(acc)
            for ph in ("P", "S"):
                c = cl[cl.phase == ph]
                ga = acc[acc.phase == ph]
                res.append(dict(block=blk, band=band, phase=ph, pairs=int(x.pair.nunique()),
                                measured=int(((g.status == "ok") & (g.phase == ph)).sum()), accepted=len(ga),
                                diff_mad_ms=mad(ga.lag_s) * 1e3, n_tri=len(c), tri_mad_ms=mad(c.e) * 1e3,
                                tri_gt20=float((c.e.abs() > 0.02).mean()) if len(c) else np.nan))
        print(pd.DataFrame(res).to_string(), flush=True)
    out = pd.DataFrame(res)
    A.QC.mkdir(parents=True, exist_ok=True)
    out.to_csv(A.QC / "band_check.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
