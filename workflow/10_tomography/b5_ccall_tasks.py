"""Stage 1 of the all-family CC differential times (D-B093): scope, picks, pairs, tiles.

Scope (brief of 2026-11-25):
* every event of a v5.1 family (events_zbz_v51: family_id >= 0; excluded events, i.e. non-earthquakes and rows
  without a hypocentre, are never in a family);
* every located earthquake (not excluded, with a v5.1 hypocentre) within 3 km (3-D) of a family event.
Pairs:
* family: all pairs of events of the same family (pair_source = family);
* knn: for each scoped event its 20 nearest neighbours within 3 km (3-D, v5.1 hypocentres) among the scoped
  events with >= 4 common station-phases, when not already a family pair (as the D-B078 added pairs).
Picks: Cascadia_picks_catalog_v4.1tc.csv (md5 checked), one pick per event-station-phase (highest probability),
timing class from the D-B072 corrections, clock correction of D-B090; t_hdr = header-time pick (window placement).
Tiles: events sorted along a 3-D Morton curve (1 km cells) and cut into groups of G events; a pair belongs to the
tile (group(ID1), group(ID2)); a measure job is a tile (or a slice of one, <= JOB_ROWS rows), so a worker loads
at most 2 G events.

  python b5_ccall_tasks.py [--test-families 1,2,3] [--group 250]
  -> <OUT>/tasks/{events,picks,pairs,jobs}.parquet, stations.json, counts.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_ccall_common as A  # noqa: E402

JOB_ROWS = 60000


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def timing_status(flag, applied):
    """D-B072 classes, as b5_cc_common.timing_status (vectorized)."""
    flag = pd.Series(flag)
    out = np.where(flag.isna(), "not_checked",
                   np.where(flag.isin(["off_grid", "rate_not_100", "error"]), "unretimable",
                            np.where(pd.Series(applied).fillna(False).astype(bool), "retimed", "checked")))
    return out


def morton3(ix, iy, iz, bits=21):
    def spread(v):
        v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
        return v
    return spread(ix) | (spread(iy) << np.uint64(1)) | (spread(iz) << np.uint64(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-families", default="", help="comma list of family ids: build a test scope only")
    ap.add_argument("--group", type=int, default=250)
    a = ap.parse_args()
    t0 = time.time()
    A.TASKS.mkdir(parents=True, exist_ok=True)
    m = md5(A.PICKS)
    assert m == A.PICKS_MD5, f"pick file md5 {m} != {A.PICKS_MD5}"

    # ------------------------------------------------------------------------------------------------ events
    cat = pd.read_csv(A.CATALOG, usecols=["ID", "time", "lat", "lon", "depth", "region", "tier", "event_flag",
                                          "loc_source", "tile"])
    cl = pd.read_csv(A.CLUSTERS, usecols=["event_id", "family_id", "family_type", "cls", "excluded"])
    ev = cat.merge(cl.rename(columns={"event_id": "ID"}), on="ID", how="left", validate="1:1")
    fams = pd.read_csv(A.FAMILIES, usecols=["family_id", "region_v51", "n_all", "type"])
    located_eq = ev.excluded.isna() & ev.lat.notna() & ev.depth.notna()
    fam = (ev.family_id >= 0).values
    assert (located_eq.values | ~fam).all(), "a family event is excluded or has no hypocentre"
    assert fam.sum() == fams.n_all.sum() == 26852 and fams.family_id.nunique() == 2097, (fam.sum(), fams.n_all.sum())
    if a.test_families:
        tf = [int(x) for x in a.test_families.split(",")]
        fam = fam & ev.family_id.isin(tf).values
    X = A.xyz(ev.lat.fillna(0), ev.lon.fillna(0), ev.depth.fillna(0))
    cand = located_eq.values & ~fam
    d, _ = cKDTree(X[fam]).query(X[cand], k=1, distance_upper_bound=A.NEIGH_R_KM)
    nb = np.zeros(len(ev), bool)
    nb[np.where(cand)[0][np.isfinite(d)]] = True
    ev["scope"] = np.where(fam, "family", np.where(nb, "neighbour", ""))
    ev["nn_family_km"] = np.nan
    ev.loc[nb, "nn_family_km"] = d[np.isfinite(d)]
    ev = ev[ev.scope != ""].copy()
    ev["T0"] = pd.to_datetime(ev.time, utc=True).astype("int64") / 1e9
    ev["block"] = ev.region.map(A.REGION_BLOCK)
    assert ev.block.notna().all(), ev[ev.block.isna()].region.unique()
    msh = A.epi_km(ev.lat.values, ev.lon.values, A.MSH_LAT, A.MSH_LON) <= A.MSH_R_KM
    ev.loc[msh, "block"] = "msh"
    ev["family_id"] = ev.family_id.fillna(-1).astype(int)
    ev = ev.sort_values("ID").reset_index(drop=True)
    print(f"scope: {len(ev)} events ({int(fam.sum())} family, {int(nb.sum())} neighbours) {time.time() - t0:.0f} s",
          flush=True)

    # ------------------------------------------------------------------------------------------------ picks
    pk = pd.read_csv(A.PICKS, skipinitialspace=True, low_memory=False,
                     usecols=["event_ID", "pick_datetime", "station", "phase", "pick_prob", "timing_corrected",
                              "timing_correction_s", "clock_correction_s", "clock_correction_sd_s", "clock_epoch",
                              "clock_flag"])
    pk["pick_row"] = np.arange(len(pk))
    pk = pk[pk.event_ID.isin(ev.ID)].rename(columns={"event_ID": "ID"})
    co = pd.read_csv(A.CORR, usecols=["pick_row", "flag", "applied"])
    pk = pk.merge(co, on="pick_row", how="left")
    pk["timing"] = timing_status(pk.flag.values, pk.applied.values)
    pk["t_pick"] = pd.to_datetime(pk.pick_datetime.str.strip(), utc=True).astype("int64") / 1e9
    pk["clock_correction_s"] = pk.clock_correction_s.fillna(0.0)
    pk["t_hdr"] = pk.t_pick - pk.clock_correction_s          # header-time pick (window placement)
    pk = pk.merge(ev[["ID", "T0"]], on="ID")
    pk["tt"] = pk.t_pick - pk.T0
    pk["net"] = pk.station.str.split(".").str[1]
    nd = len(pk)
    pk = pk.sort_values("pick_prob", ascending=False, kind="stable").drop_duplicates(["ID", "station", "phase"])
    ndup = nd - len(pk)
    spv = pk.pivot_table(index=["ID", "station"], columns="phase", values="tt", aggfunc="first")
    spv["sp"] = spv.get("S") - spv.get("P")
    spv["sp_est"] = np.where(spv.sp.notna(), spv.sp, np.where(spv.get("P").notna(), 0.73 * spv.get("P"),
                                                               spv.get("S") * 0.73 / 1.73))
    pk = pk.merge(spv[["sp", "sp_est"]].reset_index(), on=["ID", "station"], how="left")
    # clock model check: the pick file's correction is -c(t) of the confirmed model at the header-time pick
    M = A.clock_model()
    for st in M:
        x = pk[pk.station == st]
        c = A.clock_c(st, x.t_hdr.values, M)
        dev = np.abs(c + x.clock_correction_s.values)
        print(f"clock model check {st}: {int((x.clock_correction_s != 0).sum())} corrected picks in scope, "
              f"max |c(t_hdr) + clock_correction| = {dev.max() if len(dev) else 0:.2e} s", flush=True)
    stations = sorted(pk.station.unique())
    sidx = {s: k for k, s in enumerate(stations)}
    pk["sta_i"] = pk.station.map(sidx).astype(np.int32)
    pk["ph_i"] = (pk.phase == "S").astype(np.int8)
    pk["spc"] = (pk.sta_i * 2 + pk.ph_i).astype(np.int32)
    epos = {i: k for k, i in enumerate(ev.ID)}
    pk["ev_i"] = pk.ID.map(epos).astype(np.int32)
    pk = pk.sort_values(["ev_i", "spc"]).reset_index(drop=True)
    pk["pk_i"] = np.arange(len(pk), dtype=np.int32)
    print(f"picks: {len(pk)} ({ndup} duplicates dropped), {len(stations)} stations, "
          f"NC/BK {int(pk.net.isin(A.NCEDC_NETS).sum())} {time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------------------------------------------ pairs
    nev = len(ev)
    Mx = sp.csr_matrix((np.ones(len(pk), np.float32), (pk.ev_i.values, pk.spc.values)),
                       shape=(nev, 2 * len(stations)))
    fam_pairs = []
    evf = ev[ev.family_id >= 0]
    for fid, g in evf.groupby("family_id"):
        idx = np.array([epos[i] for i in g.ID])
        Mf = Mx[idx]
        Cf = (Mf @ Mf.T).toarray()
        iu, ju = np.triu_indices(len(idx), 1)
        fam_pairs.append(pd.DataFrame({"i1": idx[iu], "i2": idx[ju], "n_common": Cf[iu, ju].astype(np.int32),
                                       "family_id": fid}))
    fp = pd.concat(fam_pairs, ignore_index=True)
    fp["pair_source"] = "family"
    print(f"family pairs: {len(fp)} ({int((fp.n_common > 0).sum())} with >= 1 common station-phase, "
          f"rows {int(fp.n_common.sum())}) {time.time() - t0:.0f} s", flush=True)

    Xs = A.xyz(ev.lat, ev.lon, ev.depth)
    kq = min(nev, 4 * A.ADD_K + 1)
    dd, jj = cKDTree(Xs).query(Xs, k=kq, distance_upper_bound=A.ADD_R_KM)
    ii = np.repeat(np.arange(nev), kq).reshape(nev, kq)
    ok = np.isfinite(dd) & (jj != ii) & (jj < nev)
    I, J, D = ii[ok], jj[ok], dd[ok]
    ncom = np.zeros(len(I), np.int32)
    for k in range(0, len(I), 1_000_000):
        s = slice(k, k + 1_000_000)
        ncom[s] = np.asarray(Mx[I[s]].multiply(Mx[J[s]]).sum(1)).ravel()
    cdf = pd.DataFrame({"i": I, "j": J, "d": D, "n": ncom})
    cdf = cdf[cdf.n >= A.ADD_NMIN].sort_values(["i", "d"], kind="stable")
    cdf = cdf[cdf.groupby("i").cumcount() < A.ADD_K]
    kp = pd.DataFrame({"i1": np.minimum(cdf.i, cdf.j), "i2": np.maximum(cdf.i, cdf.j), "n_common": cdf.n})
    kp = kp.drop_duplicates(["i1", "i2"])
    fk = set(zip(fp.i1.values.tolist(), fp.i2.values.tolist()))
    infam = np.array([(x, y) in fk for x, y in zip(kp.i1.values.tolist(), kp.i2.values.tolist())])
    n_knn_in_family = int(infam.sum())
    kp = kp[~infam].copy()
    kp["family_id"] = -1
    kp["pair_source"] = "knn"
    del fk
    pairs = pd.concat([fp, kp], ignore_index=True)
    pairs["ID1"] = ev.ID.values[pairs.i1]
    pairs["ID2"] = ev.ID.values[pairs.i2]
    pairs["sep_km"] = np.linalg.norm(Xs[pairs.i1.values] - Xs[pairs.i2.values], axis=1).astype(np.float32)
    # block of a pair: the family's block (most common block of its events), else the block of ID1
    fblock = evf.groupby("family_id").block.agg(lambda s: s.value_counts().index[0])
    pairs["block"] = np.where(pairs.family_id >= 0, pairs.family_id.map(fblock), ev.block.values[pairs.i1])
    pairs["same_family"] = (ev.family_id.values[pairs.i1] >= 0) & (ev.family_id.values[pairs.i1] == ev.family_id.values[pairs.i2])
    print(f"knn pairs: {len(kp)} new ({n_knn_in_family} already family pairs); total {len(pairs)} "
          f"{time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------------------------------------------ tiles / jobs
    q = np.floor((Xs - Xs.min(0)) / 1.0).astype(np.int64)
    order = np.argsort(morton3(q[:, 0], q[:, 1], q[:, 2]), kind="stable")
    rank = np.empty(nev, np.int64)
    rank[order] = np.arange(nev)
    grp = (rank // a.group).astype(np.int32)
    ev["group"] = grp
    ev["ev_i"] = np.arange(nev)
    pr = pairs[pairs.n_common > 0]
    g1, g2 = grp[pr.i1.values], grp[pr.i2.values]
    pairs["tile_a"] = -1
    pairs["tile_b"] = -1
    pairs.loc[pr.index, "tile_a"] = np.minimum(g1, g2)
    pairs.loc[pr.index, "tile_b"] = np.maximum(g1, g2)
    pairs = pairs.sort_values(["tile_a", "tile_b", "i1", "i2"]).reset_index(drop=True)
    pairs["pair"] = np.arange(len(pairs), dtype=np.int64)
    pm = pairs[pairs.n_common > 0]
    cum = pm.groupby(["tile_a", "tile_b"]).n_common.cumsum()
    sub = ((cum - 1) // JOB_ROWS).astype(int)
    jobs = (pm.assign(sub=sub.values).groupby(["tile_a", "tile_b", "sub"])
            .agg(pair_lo=("pair", "min"), pair_hi=("pair", "max"), rows=("n_common", "sum"), n_pairs=("pair", "size"))
            .reset_index())
    jobs["job"] = [f"t{x:04d}_{y:04d}_{z:02d}" for x, y, z in zip(jobs.tile_a, jobs.tile_b, jobs["sub"])]
    jobs = jobs.sort_values("rows", ascending=False).reset_index(drop=True)
    print(f"jobs: {len(jobs)}, rows {int(jobs.rows.sum())}, largest {int(jobs.rows.max())}, groups {grp.max() + 1} "
          f"{time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------------------------------------------ save
    ev_cols = ["ID", "time", "T0", "lat", "lon", "depth", "region", "block", "tier", "tile", "family_id", "family_type",
               "cls", "scope", "nn_family_km", "group", "ev_i"]
    ev[ev_cols].to_parquet(A.TASKS / "events.parquet", index=False)
    pk_cols = ["pk_i", "ev_i", "ID", "station", "net", "phase", "sta_i", "ph_i", "spc", "pick_datetime", "t_pick",
               "t_hdr", "T0", "tt", "pick_prob", "pick_row", "timing", "flag", "applied", "timing_corrected",
               "timing_correction_s", "clock_correction_s", "clock_correction_sd_s", "clock_epoch", "clock_flag",
               "sp", "sp_est"]
    pk[pk_cols].to_parquet(A.TASKS / "picks.parquet", index=False)
    pairs[["pair", "ID1", "ID2", "i1", "i2", "pair_source", "family_id", "same_family", "block", "sep_km",
           "n_common", "tile_a", "tile_b"]].to_parquet(A.TASKS / "pairs.parquet", index=False)
    jobs.to_parquet(A.TASKS / "jobs.parquet", index=False)
    (A.TASKS / "stations.json").write_text(json.dumps(stations))
    rows_by_block = pairs.groupby(["block", "pair_source"]).agg(pairs=("pair", "size"),
                                                                 pairs_ge1=("n_common", lambda s: int((s > 0).sum())),
                                                                 rows=("n_common", "sum"))
    counts = dict(
        events=len(ev), events_family=int((ev.scope == "family").sum()),
        events_neighbour=int((ev.scope == "neighbour").sum()),
        events_by_block={b: dict(family=int(((ev.block == b) & (ev.scope == "family")).sum()),
                                 neighbour=int(((ev.block == b) & (ev.scope == "neighbour")).sum()))
                         for b in A.BLOCKS},
        picks=len(pk), duplicate_picks_dropped=ndup, picks_ncedc=int(pk.net.isin(A.NCEDC_NETS).sum()),
        picks_clock_corrected=int((pk.clock_correction_s != 0).sum()),
        timing=pk.timing.value_counts().to_dict(),
        pairs=len(pairs), pairs_family=int((pairs.pair_source == "family").sum()),
        pairs_knn=int((pairs.pair_source == "knn").sum()), knn_already_family=n_knn_in_family,
        pairs_with_common=int((pairs.n_common > 0).sum()), rows=int(pairs.n_common.sum()),
        by_block={f"{b}:{s}": {k: int(v) for k, v in r.items()} for (b, s), r in rows_by_block.iterrows()},
        jobs=len(jobs), groups=int(grp.max() + 1), group_size=a.group, job_rows=JOB_ROWS,
        test_families=a.test_families, picks_md5=m)
    (A.TASKS / "counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps({k: v for k, v in counts.items() if k != "by_block"}, indent=1))
    print(rows_by_block.to_string())
    A.write_state("tasks", done=True, **{k: counts[k] for k in ("events", "picks", "pairs", "rows", "jobs")})
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
