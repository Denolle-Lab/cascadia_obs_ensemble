"""Track B, step B5 (D-B053; D-B060: catalog differential times now, waveform cross-correlation times later):
double-difference relocation of the whole catalog with tomoDD-SE in relocation-only mode (JOINT = 0, velocity
frozen), variant `dd` of the D-B048 test (differential times only, WTCD = 0), LSQR damping 100, frozen D-B054
binaries. Parameters: params.yaml block `b5`. Notes: notes/b5_relocation.md.

  python b5_relocate.py STAGE --version 0.3.1 [--run NAME] [--b4-run R1] [--iter0 R0] [--time T]
                              [--nproc N] [--only CID[,CID]] [--nboot K]
  STAGE: prep | split | run | collect (b5_relocate.sh runs them in order, detached, with a state file)

Run name and model version are parameters: run <b5.out>/<run>/ (default run b5_v<version>), B4 run b4_iter1_v<version>,
station terms of b4_iter0_v<version>, time grids reloc_v5/time_b4_v<version> (b5.defaults). The first stage writes
<run>/config.json; later stages read it and stop if the command line disagrees.

prep (rules 1-3 of the B5 brief, as b5_prep.py of the D-B048 test, for the whole catalog)
* Start: the B4 NLLoc solutions at full precision (runs/<R1>/locations.csv, status LOCATED, one row per ID), EFT
  undone (z = Re (1 - exp(-z'/Re)), as b4_catalog.py); IDs, tiers and regions from the B4 catalog
  (catalog_b4_<R1>.csv), checked against it to its 5th digit. Tier U (v4 hypocentre kept) is not relocated.
  b5.exclude_ids (172113, mis-associated in v4.0, removed in v4.1) are left out everywhere, catalog included.
* Picks: catalog v4.0 picks (the B4 input) of those events, travel time = pick - B4 origin time, kept where the
  event lies >= grid_margin_km inside the station's B4 time grid (x, y, z'), at stations with a grid. Weight
  (0.05/sigma)^2, D-B008 sigma of the pick_prob class (b5_common). A repeated event-station-phase keeps the pick
  with the highest pick_prob.
* Differential times (tomodd.dt rule): each event paired with its max_neighbours nearest events within max_sep_km
  (3D distance at the B4 hypocentres, Earth-centred coordinates on the sphere of radius Re - depth; the tiles used
  a flat projection about the mean latitude, which distorts east-west distances by up to 8 % over the whole
  margin), pair kept with >= min_common common station-phases, one dt per common station-phase, weight = the smaller
  of the two pick weights.
* Components: connected components of the kept-pair graph (scipy csgraph). Events in no kept pair are "unlinked".

split (rule 3): a cluster (one tomoDD run) is a component if it fits a frozen binary (b5.tomodd.classes, tried in
order: MAXEVE events, MAXDATA = dt + absolute rows, MAXSTA stations, MAXOBS distinct stations of one event;
RaySPDR.inc maxnx/maxny/maxnz MOD nodes). A component that fits none is cut by recursive bisection: the core
events are split at the median of their projection on the principal horizontal axis; each chunk = core + halo,
halo = the events within halo_hops pair-graph hops of the core (so every pair of a core event, and with 2 hops every
pair of its partners, is in the chunk); a chunk that still fits no binary is bisected again. Every event is core in
exactly one chunk, the chunk where it is most interior (graph distance to the chunk edge >= halo_hops + 1, against
<= halo_hops in any chunk where it is halo), and its B5 location comes from that chunk only. Pairs lost (no chunk
holds both events) are counted (0 by construction for halo_hops >= 1), as are the pairs a halo event loses at the
chunk edge, and the overlap duplicates give a direct measure of the split effect (core vs halo solution).

run (rule 4): per cluster, tomoDD-SE relocation-only in model v<version> (netCDF nodes over the cluster's events and
stations + node_margin_deg, edge nodes as the tiles, D-B028 P-2 ellipsoid rescaling about the frame centre),
Air_dep = -(highest station elevation + air_dep_margin_km), frame centre = b5_common.frame_centre (no event within
frame_clear_km, D-B053). Absolute times (zero weight in dd) = pick - B4 origin - B4 station term (iter0 run). Peak
RSS and wall time are recorded; a new tomoDD starts only while MemAvailable >= min_free_gb + the largest RSS seen.
Bootstrap (--nboot K, default b5.nboot = 0): K extra runs per cluster with the dt rows resampled with replacement
(seed 48000 + k, as the D-B048 test).

collect (rules 5-7): per cluster, the centroid of the relocated core events is put back on their B4 centroid
(D-B053; the shift removed, and the mean origin-time shift, are recorded), origin time = B4 time + tomoDD's shift
(10 ms steps in tomoDD.reloc) - the cluster mean shift. Events never relocated keep the B4 (loc_source nll3d) or v4
(tier U, loc_source v4) position; relocated events get loc_source b5_tomodd. surface_km / above_surface as
b4_catalog.py (D-B055: counted and reported, never corrected). Catalog '%.8g'; cluster summary; compare.md;
dt residual rms in the B4 time grids before (B4) and after (B5), per-pair median removed (as b5_compare.py).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b2_fwd_common import RE, eft_depth, lambert_km, read_nll_hdr  # noqa: E402
from b5_common import frame_centre, pca, pick_class, pick_sigma, unflatten, weight  # noqa: E402

P = load_params()
B = P["b5"]
T = P["tomodd"]
KM_DEG = 111.19
STAGES = ("prep", "split", "run", "collect")


def log(msg):
    print(f"[b5 {time.strftime('%F %T')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ configuration
def config(a):
    v = a.version
    D = {k: s.format(v=v) for k, s in B["defaults"].items()}
    C = dict(version=v, run=a.run or D["run"], b4_run=a.b4_run or D["b4_run"], iter0=a.iter0 or D["b4_iter0"],
             time_grids=a.time or D["time_grids"])
    rv = Path(B["reloc_v5"])
    C.update(dir=str(Path(B["out"]) / C["run"]), catalog_b4=str(rv / "catalog" / f"catalog_b4_{C['b4_run']}.csv"),
             locations=str(rv / "runs" / C["b4_run"] / "locations.csv"),
             station_terms=str(rv / "runs" / C["iter0"] / "station_terms.in"),
             grids=str(rv / C["time_grids"]),
             model=str(Path(T["models"]["store"]) / T["models"]["name"] / f"v{v}.nc"),
             summary=str(REPO / B["summary"] / C["run"]))
    for k in ("dir", "summary"):     # never write into the D-B048 test, B4 runs, tiles or the round-2 scratch
        assert not any(C[k].startswith(f) for f in B["forbidden_out"]), f"{k} {C[k]} is a forbidden output path"
    f = Path(C["dir"]) / "config.json"
    if f.exists():
        old = json.loads(f.read_text())
        diff = {k: (old.get(k), C[k]) for k in C if old.get(k) != C[k]}
        assert not diff, f"{f} disagrees with the command line: {diff}"
    else:
        Path(C["dir"]).mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(C, indent=1))
    C = {k: (Path(x) if k in ("dir", "summary") else x) for k, x in C.items()}
    return C


def mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


class Grids:
    """Trilinear lookup in the B4 time grids of one model version (b5_common.Grids, root as a parameter)."""

    def __init__(self, root):
        self.root, self.c = Path(root), {}

    def get(self, phase, station):
        k = (phase, station)
        if k not in self.c:
            r = self.root / f"{B['grid_prefix']}.{phase}.{station}.time"
            g = read_nll_hdr(r)
            a = np.memmap(f"{r}.buf", dtype="<f4", mode="r", shape=(g["nx"], g["ny"], g["nz"]))
            self.c[k] = (g, a)
        return self.c[k]

    def inside(self, phase, station, x, y, zf, margin):
        g, _ = self.get(phase, station)
        x1, y1 = g["x0"] + g["dx"] * (g["nx"] - 1), g["y0"] + g["dx"] * (g["ny"] - 1)
        z1 = g["z0"] + g["dx"] * (g["nz"] - 1)
        return ((x >= g["x0"] + margin) & (x <= x1 - margin) & (y >= g["y0"] + margin) & (y <= y1 - margin)
                & (zf >= g["z0"]) & (zf <= z1 - margin))

    def time(self, phase, station, x, y, zf):
        g, a = self.get(phase, station)
        fx, fy, fz = (x - g["x0"]) / g["dx"], (y - g["y0"]) / g["dx"], (zf - g["z0"]) / g["dx"]
        ix = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2)
        iz = np.clip(np.floor(fz).astype(int), 0, g["nz"] - 2)
        wx, wy, wz = fx - ix, fy - iy, fz - iz
        o = np.lexsort((iz, iy, ix))                # read the memmap in file order
        ix, iy, iz, wx, wy, wz = ix[o], iy[o], iz[o], wx[o], wy[o], wz[o]
        out = np.zeros(len(x))
        for dx_, ax in ((0, 1 - wx), (1, wx)):
            for dy_, ay in ((0, 1 - wy), (1, wy)):
                for dz_, az in ((0, 1 - wz), (1, wz)):
                    out += ax * ay * az * a[ix + dx_, iy + dy_, iz + dz_]
        res = np.empty(len(x))
        res[o] = out
        return res


def grid_stations(C):
    man = json.loads((Path(C["grids"]) / f"b4_time_grids_{C['time_grids']}.manifest.json").read_text())
    src = [i["path"] for i in man["inputs"] if i["path"].endswith("stations_GTSRCE.in")][0]
    rows = [l.split() for l in Path(src).read_text().splitlines() if l.startswith("GTSRCE")]
    return pd.DataFrame({"station": [r[1] for r in rows], "lat": [float(r[3]) for r in rows],
                         "lon": [float(r[4]) for r in rows], "zsrc_km": [float(r[5]) for r in rows]}), src


def ecef(lat, lon, depth):
    r = RE - np.asarray(depth, float)
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)]


# ------------------------------------------------------------------------------------------------ prep
def stage_prep(C):
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    d = C["dir"] / "prep"
    d.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(C["catalog_b4"], usecols=["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region"])
    loc = pd.read_csv(C["locations"], usecols=["status", "ID", "time", "lat", "lon", "depth"])
    loc = loc[loc.status == "LOCATED"].drop_duplicates("ID").set_index("ID")
    nll = cat.loc_source == B["loc_source"]
    assert set(loc.index) == set(cat.ID[nll]), "locations.csv and the B4 catalog disagree on the located events"
    excl = [int(i) for i in B["exclude_ids"]]
    c = cat[nll & ~cat.ID.isin(excl)].set_index("ID")
    l = loc.loc[c.index].copy()
    l["depth"] = unflatten(l.depth)
    for col in ("lat", "lon", "depth"):            # full precision against the catalog (rounded to 5 digits or not)
        new, old = l[col].to_numpy(float), c[col].to_numpy(float)
        assert np.all(np.abs(new - old) <= 1e-4 * np.abs(old) + 2e-3), (col, np.nanmax(np.abs(new - old)))
    assert (l.time.values == c.time.values).all(), "origin times differ between locations.csv and the catalog"
    t_ns = pd.to_datetime(l.time, format="ISO8601").values.astype("datetime64[ns]").astype(np.int64)
    x, y = lambert_km(l.lon.values, l.lat.values)
    ev = pd.DataFrame({"ID": c.index.values.astype(np.int64), "time": l.time.values, "t_ns": t_ns,
                       "lat": l.lat.values, "lon": l.lon.values, "depth": l.depth.values, "x_km": x, "y_km": y,
                       "zf_km": eft_depth(l.depth.values), "tier": c.tier.values, "region": c.region.values})
    ev = ev.sort_values("ID").reset_index(drop=True)
    log(f"start: {len(ev)} B4 events located in 3D ({int(nll.sum())} in the catalog, excluded {excl}); "
        f"tier U {int((cat.loc_source != B['loc_source']).sum())} keep their v4 hypocentre")

    # picks
    pk = pd.read_csv(B["picks"], skipinitialspace=True)
    pk.columns = [s.strip() for s in pk.columns]
    pk = pk[["event_ID", "pick_datetime", "station", "phase", "pick_prob"]]
    n_v4 = len(pk)
    pk = pk[pk.event_ID.isin(ev.ID)]
    n_ev = len(pk)
    gs, gtsrce = grid_stations(C)
    pk = pk[pk.station.isin(gs.station)]
    n_nogrid = n_ev - len(pk)
    ei = pd.Index(ev.ID).get_indexer(pk.event_ID)
    pk = pk.assign(ev=ei.astype(np.int32))
    tp = pd.to_datetime(pk.pick_datetime.str.rstrip("Z"), format="ISO8601").values.astype("datetime64[ns]")
    pk["tt"] = (tp.astype(np.int64) - ev.t_ns.values[ei]) / 1e9
    pk = pk.drop(columns="pick_datetime")
    G = Grids(C["grids"])
    ok = np.zeros(len(pk), bool)
    pos = pd.RangeIndex(len(pk))
    pk = pk.reset_index(drop=True)
    for (sta, ph), g in pk.groupby(["station", "phase"]):
        e = g.ev.values
        ok[g.index.values] = G.inside(ph, sta, ev.x_km.values[e], ev.y_km.values[e], ev.zf_km.values[e],
                                      B["grid_margin_km"])
    n_out = int((~ok).sum())
    pk = pk[ok]
    pk = pk.sort_values(["ev", "station", "phase", "pick_prob"], ascending=[True, True, True, False])
    dup = pk.duplicated(["ev", "station", "phase"])
    n_dup = int(dup.sum())
    pk = pk[~dup].copy()
    pk["cls"] = pick_class(pk.pick_prob.values).values.astype(np.int8)
    pk["sigma"] = pick_sigma(pk.cls.values, pk.phase.values).astype(np.float32)
    pk["w"] = weight(pk.sigma).astype(np.float32)
    st = gs[gs.station.isin(pk.station)].sort_values("station").reset_index(drop=True)
    st["code"] = [f"S{i:04d}" for i in range(len(st))]
    pk["sta"] = pd.Index(st.station).get_indexer(pk.station).astype(np.int16)
    pk["sp"] = (2 * pk.sta + (pk.phase == "S")).astype(np.int16)         # station-phase index
    pk = pk[["ev", "sta", "sp", "phase", "tt", "cls", "sigma", "w", "pick_prob"]].reset_index(drop=True)
    del pos
    log(f"picks: v4.0 {n_v4}, of these events {n_ev}, at stations without a grid {n_nogrid}, outside the grid "
        f"margin {n_out}, repeated {n_dup}; kept {len(pk)} at {len(st)} stations")

    # differential times
    D = B["dt"]
    xyz = ecef(ev.lat.values, ev.lon.values, ev.depth.values)
    K = D["max_neighbours"]
    dd, ii = cKDTree(xyz).query(xyz, k=K + 2, distance_upper_bound=D["max_sep_km"])
    n = len(ev)
    a = np.repeat(np.arange(n), K + 2)
    b = ii.ravel()
    fin = np.isfinite(dd.ravel()) & (b != a)
    a, b = a[fin], b[fin]
    # the K nearest others (the query can list an event at the same position before the event itself)
    rank = pd.Series(np.ones(len(a), int)).groupby(a).cumsum().values
    keep = rank <= K
    a, b = a[keep], b[keep]
    pa, pb = np.minimum(a, b), np.maximum(a, b)
    pairs = np.unique(np.c_[pa, pb], axis=0)
    n_cand = len(pairs)
    nsp = 2 * len(st)
    M = sparse.csr_matrix((np.ones(len(pk), np.int8), (pk.ev.values, pk.sp.values)), shape=(n, nsp))
    key = pk.ev.values.astype(np.int64) * nsp + pk.sp.values          # pk sorted by ev; keys sorted per event
    order = np.argsort(key, kind="stable")
    skey = key[order]
    rows, keptp = [], []
    blk = 200_000
    for s in range(0, n_cand, blk):
        p = pairs[s:s + blk]
        prod = M[p[:, 0]].multiply(M[p[:, 1]]).tocsr()
        ncom = np.diff(prod.indptr)
        kp = ncom >= D["min_common"]
        prod = prod[kp]
        pr = np.repeat(np.arange(prod.shape[0]), np.diff(prod.indptr))
        q = p[kp]
        e1, e2, sp = q[pr, 0], q[pr, 1], prod.indices
        i1 = order[np.searchsorted(skey, e1.astype(np.int64) * nsp + sp)]
        i2 = order[np.searchsorted(skey, e2.astype(np.int64) * nsp + sp)]
        rows.append(pd.DataFrame({"e1": e1.astype(np.int32), "e2": e2.astype(np.int32), "sp": sp.astype(np.int16),
                                  "pk1": i1.astype(np.int32), "pk2": i2.astype(np.int32)}))
        keptp.append(np.c_[q, ncom[kp]])
    dt = pd.concat(rows, ignore_index=True)
    assert (pk.ev.values[dt.pk1] == dt.e1.values).all() and (pk.sp.values[dt.pk2] == dt.sp.values).all()
    dt["w"] = np.minimum(pk.w.values[dt.pk1], pk.w.values[dt.pk2]).astype(np.float32)
    kp = np.concatenate(keptp)
    pr = pd.DataFrame({"e1": kp[:, 0].astype(np.int32), "e2": kp[:, 1].astype(np.int32),
                       "n_dt": kp[:, 2].astype(np.int32)})
    pr["sep_km"] = np.linalg.norm(xyz[pr.e1] - xyz[pr.e2], axis=1).astype(np.float32)
    A = sparse.coo_matrix((np.ones(len(pr)), (pr.e1, pr.e2)), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    deg = np.bincount(np.r_[pr.e1, pr.e2], minlength=n)
    linked = deg > 0
    # components numbered by size (0 = largest); unlinked events -> -1
    lab_l = lab[linked]
    sizes = np.bincount(lab_l)
    rank_c = np.argsort(np.argsort(-sizes, kind="stable"))
    comp = np.full(n, -1, np.int32)
    comp[linked] = rank_c[lab_l]
    ev["comp"] = comp
    ev["n_pairs"] = deg.astype(np.int32)
    ev["n_picks"] = np.bincount(pk.ev, minlength=n).astype(np.int32)
    ev["n_sta"] = np.diff(sparse.csr_matrix((np.ones(len(pk)), (pk.ev, pk.sta)), shape=(n, len(st))).indptr) \
        .astype(np.int32)
    ev.to_parquet(d / "events.parquet", index=False)
    pk.to_parquet(d / "picks.parquet", index=False)
    dt.to_parquet(d / "dt.parquet", index=False)
    pr.to_parquet(d / "pairs.parquet", index=False)
    st.to_csv(d / "stations.csv", index=False, float_format="%.6f")
    cs = np.bincount(comp[comp >= 0])
    counts = dict(events=len(ev), tier_U=int((cat.loc_source != B["loc_source"]).sum()), excluded=excl,
                  picks_v4=n_v4, picks_events=n_ev, picks_no_grid=n_nogrid, picks_outside_grid=n_out,
                  picks_repeated=n_dup, picks=len(pk), P=int((pk.phase == "P").sum()), S=int((pk.phase == "S").sum()),
                  stations=len(st), pair_candidates=int(n_cand), pairs=len(pr), dt=len(dt),
                  dt_P=int((dt.sp % 2 == 0).sum()), dt_S=int((dt.sp % 2 == 1).sum()),
                  events_linked=int(linked.sum()), events_unlinked=int((~linked).sum()), components=int(len(cs)),
                  component_sizes={"2": int((cs == 2).sum()), "3-9": int(((cs >= 3) & (cs < 10)).sum()),
                                   "10-99": int(((cs >= 10) & (cs < 100)).sum()),
                                   "100-999": int(((cs >= 100) & (cs < 1000)).sum()),
                                   "1000-9999": int(((cs >= 1000) & (cs < 10000)).sum()), ">=10000": int((cs >= 10000).sum())},
                  largest_components=[int(v) for v in np.sort(cs)[::-1][:10]],
                  max_sta_per_event=int(ev.n_sta.max()), max_picks_per_event=int(ev.n_picks.max()))
    (d / "prep.json").write_text(json.dumps(counts, indent=1))
    write_manifest("b5_prep", d, inputs=[C["catalog_b4"], C["locations"], B["picks"], gtsrce],
                   outputs=[d / f for f in ("events.parquet", "picks.parquet", "dt.parquet", "pairs.parquet",
                                            "stations.csv", "prep.json")],
                   params={"b5": B, "config": {k: str(v) for k, v in C.items()}}, counts=counts)
    log(f"dt: {n_cand} candidate pairs, {len(pr)} kept, {len(dt)} dt; {counts['events_linked']} events linked in "
        f"{len(cs)} components (largest {counts['largest_components'][:5]}); {counts['events_unlinked']} unlinked")


# ------------------------------------------------------------------------------------------------ split
class Limits:
    """Frozen-binary limits and the MOD node count of an event set (positions into the events table)."""

    def __init__(self, ev, pk, st, pr, model_nodes):
        from scipy import sparse
        n = len(ev)
        self.ev, self.st = ev, st
        self.ES = sparse.csr_matrix((np.ones(len(pk), np.int8), (pk.ev.values, pk.sta.values)), shape=(n, len(st)))
        self.npk = ev.n_picks.values
        self.nst = ev.n_sta.values
        self.pr = pr
        self.lon_nodes, self.lat_nodes, self.nz = model_nodes
        self.cls = B["tomodd"]["classes"]
        self.mx = B["tomodd"]["mod_max"]

    def stats(self, idx, pairs_idx):
        """idx: event positions; pairs_idx: positions (into pr) of candidate pairs (the component's pairs)."""
        inset = np.zeros(len(self.ev), bool)
        inset[idx] = True
        p = self.pr.iloc[pairs_idx] if pairs_idx is not None else self.pr
        both = inset[p.e1.values] & inset[p.e2.values]
        n_dt = int(p.n_dt.values[both].sum())
        sta = np.unique(self.ES[idx].indices)
        m = B["tomodd"]["node_margin_deg"]
        lo = np.r_[self.ev.lon.values[idx], self.st.lon.values[sta]]
        la = np.r_[self.ev.lat.values[idx], self.st.lat.values[sta]]
        nx = int(((self.lon_nodes >= lo.min() - m) & (self.lon_nodes <= lo.max() + m)).sum()) + 2
        ny = int(((self.lat_nodes >= la.min() - m) & (self.lat_nodes <= la.max() + m)).sum()) + 2
        s = dict(n_events=int(len(idx)), n_sta=int(len(sta)), n_dt=n_dt, n_pairs=int(both.sum()),
                 n_abs=int(self.npk[idx].sum()), max_obs=int(self.nst[idx].max()), mod_nx=nx, mod_ny=ny,
                 mod_nz=self.nz + 2)
        s["n_data"] = s["n_dt"] + s["n_abs"]
        s["cls"] = self.fit(s)
        return s

    def fit(self, s):
        if s["mod_nx"] > self.mx["nx"] or s["mod_ny"] > self.mx["ny"] or s["mod_nz"] > self.mx["nz"]:
            return None
        for name, L in self.cls.items():
            if (s["n_events"] <= L["maxeve"] and s["n_data"] <= L["maxdata"] and s["n_sta"] <= L["maxsta"]
                    and s["max_obs"] <= L["maxobs"]):
                return name
        return None


def halo(adj, core, hops):
    """Events within `hops` pair-graph hops of the core (core excluded), with their hop number."""
    seen = np.zeros(adj.shape[0], np.int8)
    seen[core] = 1
    front, out, hop = core, [], []
    for h in range(1, hops + 1):
        nb = np.unique(adj[front].indices)
        nb = nb[seen[nb] == 0]
        seen[nb] = 1
        out.append(nb)
        hop.append(np.full(len(nb), h, np.int8))
        front = nb
        if len(nb) == 0:
            break
    return (np.concatenate(out) if out else np.array([], int)), (np.concatenate(hop) if hop else np.array([], np.int8))


def bisect(ev, core):
    """Split core events at the median of their projection on the principal horizontal axis."""
    lat0 = float(np.mean(ev.lat.values[core]))
    e = (ev.lon.values[core] - ev.lon.values[core].mean()) * KM_DEG * np.cos(np.radians(lat0))
    nn = (ev.lat.values[core] - lat0) * KM_DEG
    X = np.c_[e, nn]
    w, v = np.linalg.eigh(np.cov(X.T))
    proj = X @ v[:, np.argmax(w)]
    o = np.argsort(proj, kind="stable")
    h = len(core) // 2
    return core[o[:h]], core[o[h:]]


def stage_split(C):
    import xarray as xr
    from scipy import sparse
    d = C["dir"] / "prep"
    ev = pd.read_parquet(d / "events.parquet")
    pk = pd.read_parquet(d / "picks.parquet", columns=["ev", "sta"])
    pr = pd.read_parquet(d / "pairs.parquet")
    st = pd.read_csv(d / "stations.csv")
    ds = xr.open_dataset(C["model"])
    L = Limits(ev, pk, st, pr, (ds.longitude.values, ds.latitude.values, len(ds.depth)))
    n = len(ev)
    adj = sparse.csr_matrix((np.ones(2 * len(pr), np.int8), (np.r_[pr.e1, pr.e2], np.r_[pr.e2, pr.e1])), shape=(n, n))
    hops = int(B["split"]["halo_hops"])
    comp = ev.comp.values
    pcomp = comp[pr.e1.values]
    porder = np.argsort(pcomp, kind="stable")
    pptr = np.searchsorted(pcomp[porder], np.arange(comp.max() + 2))
    eorder = np.argsort(comp, kind="stable")
    eptr = np.searchsorted(comp[eorder], np.arange(-1, comp.max() + 2))     # comp -1 first
    clusters, members, covered_lost = [], [], dict(pairs_split_components=0, pairs_lost=0, halo_edge_cuts=0)
    for k in range(comp.max() + 1):
        idx = eorder[eptr[k + 1]:eptr[k + 2]]
        pidx = porder[pptr[k]:pptr[k + 1]]
        s = L.stats(idx, pidx)
        if s["cls"] is not None:
            clusters.append(dict(cid=f"c{k:06d}", comp=k, part=-1, n_core=len(idx), n_halo=0, **s))
            members.append(pd.DataFrame({"cid": f"c{k:06d}", "ev": idx, "core": True, "hop": np.int8(0)}))
            continue
        log(f"component {k}: {s} fits no binary, bisecting (halo {hops} hops)")
        stack, leaves = [idx], []
        while stack:
            core = stack.pop()
            hal, hh = halo(adj, core, hops)
            sc = L.stats(np.r_[core, hal], pidx)
            if sc["cls"] is not None:
                leaves.append((core, hal, hh, sc))
                continue
            if len(core) < B["split"]["min_core"]:
                raise SystemExit(f"component {k}: core of {len(core)} events still fits no binary: {sc}")
            stack += list(bisect(ev, core))
        cov = np.zeros(len(pidx), bool)
        p1, p2 = pr.e1.values[pidx], pr.e2.values[pidx]
        for j, (core, hal, hh, sc) in enumerate(sorted(leaves, key=lambda t: -len(t[0]))):
            cid = f"c{k:06d}_p{j:03d}"
            inset = np.zeros(n, bool)
            inset[np.r_[core, hal]] = True
            cov |= inset[p1] & inset[p2]
            ishalo = np.zeros(n, bool)
            ishalo[hal] = True
            covered_lost["halo_edge_cuts"] += int(((ishalo[p1] & ~inset[p2]) | (ishalo[p2] & ~inset[p1])).sum())
            clusters.append(dict(cid=cid, comp=k, part=j, n_core=len(core), n_halo=len(hal), **sc))
            members.append(pd.DataFrame({"cid": cid, "ev": np.r_[core, hal], "core": np.r_[np.ones(len(core), bool),
                                                                                         np.zeros(len(hal), bool)],
                                         "hop": np.r_[np.zeros(len(core), np.int8), hh]}))
        covered_lost["pairs_split_components"] += len(pidx)
        covered_lost["pairs_lost"] += int((~cov).sum())
        log(f"component {k}: {len(leaves)} chunks, cores {sorted(len(t[0]) for t in leaves)}, "
            f"halos {sorted(len(t[1]) for t in leaves)}, pairs lost {int((~cov).sum())} of {len(pidx)}")
    cl = pd.DataFrame(clusters)
    mem = pd.concat(members, ignore_index=True)
    mem["ID"] = ev.ID.values[mem.ev]
    assert mem[mem.core].ev.is_unique and set(mem[mem.core].ev) == set(np.where(comp >= 0)[0])
    g = ev.groupby("comp").region.agg(lambda r: r.value_counts().index[0])
    cl["region"] = cl.comp.map(g)
    cen = mem[mem.core].assign(lat=ev.lat.values[mem[mem.core].ev], lon=ev.lon.values[mem[mem.core].ev],
                               depth=ev.depth.values[mem[mem.core].ev]).groupby("cid")[["lat", "lon", "depth"]].mean()
    cl = cl.join(cen, on="cid")
    out = C["dir"] / "split"
    out.mkdir(exist_ok=True)
    cl.to_csv(out / "clusters.csv", index=False, float_format="%.6f")
    mem.to_parquet(out / "members.parquet", index=False)
    counts = dict(clusters=len(cl), components=int(comp.max() + 1), split_components=int((cl.part >= 0).groupby(
        cl.comp).any().sum()), chunks=int((cl.part >= 0).sum()), halo_hops=hops,
        classes={k: int(v) for k, v in cl.cls.value_counts().items()}, **covered_lost,
        halo_events=int(cl.n_halo.sum()), max_events=int(cl.n_events.max()), max_sta=int(cl.n_sta.max()),
        max_data=int(cl.n_data.max()), max_mod_nx=int(cl.mod_nx.max()), max_mod_ny=int(cl.mod_ny.max()))
    (out / "split.json").write_text(json.dumps(counts, indent=1))
    write_manifest("b5_split", out, inputs=[d / "events.parquet", d / "pairs.parquet", d / "picks.parquet", C["model"]],
                   outputs=[out / "clusters.csv", out / "members.parquet", out / "split.json"],
                   params={"split": B["split"], "classes": B["tomodd"]["classes"], "mod_max": B["tomodd"]["mod_max"]},
                   counts=counts)
    log(f"split: {counts}")


# ------------------------------------------------------------------------------------------------ run
INP = """* tomoDD-SE.inp written by b5_relocate.py ({run} / {cid}{tag})
*--- input file selection
* cross correlation diff times:

*
*catalog diff times:
Input_Files/dt.ct
*
* event file:
Input_Files/event.dat
*
* station file:
Input_Files/station.dat
*
*--- output file selection
* original locations:
Output_Files/tomoDD.loc
* relocations:
Output_Files/tomoDD.reloc
* station information:
Output_Files/tomoDD.sta
* residual information:
Output_Files/tomoDD.res
* source paramater information:
Output_Files/tomoDD.src
* Velocity file
Output_Files/tomoDD.vel
* Vp Model
Output_Files/Vp_model.dat
* Vs Model
Output_Files/Vs_model.dat
* Absolute file
Input_Files/absolute.dat
*
*--- data type selection:
* IDAT   IPHA   DIST
   2      3     8000
*
*--- event clustering:
* OBSCC  OBSCT   Air_dep
    0      0     {air:.3f}
*
*--- solution control:
*  ISTART  ISOLV  NSET
     2       2     {nset}
* iuses iuseq invdel  stepl
    2     0     0       {stepl}
* wlat   wlon   rota
  {clat}  {clon}   0
* weight1 weight2  weight3  CC_format
    {w1}    {w2}    {w3}       1
*
*--- data weighting and re-weighting:
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT WTCD DAMP JOINT THRES
{sets}
*--- event selection:
* CID
    0
* ID
"""

_W = {}      # per-process cache: model, shared tables (inherited by fork)


def load_shared(C):
    d = C["dir"] / "prep"
    ev = pd.read_parquet(d / "events.parquet")
    pk = pd.read_parquet(d / "picks.parquet")
    dt = pd.read_parquet(d / "dt.parquet", columns=["e1", "e2", "sp", "pk1", "pk2", "w"])
    st = pd.read_csv(d / "stations.csv")
    mem = pd.read_parquet(C["dir"] / "split" / "members.parquet")
    cl = pd.read_csv(C["dir"] / "split" / "clusters.csv")
    comp = ev.comp.values
    dcomp = comp[dt.e1.values]
    o = np.argsort(dcomp, kind="stable")
    dt = dt.iloc[o].reset_index(drop=True)
    dptr = np.searchsorted(dcomp[o], np.arange(comp.max() + 2))
    pptr = np.searchsorted(pk.ev.values, np.arange(len(ev) + 1))
    assert (np.diff(pk.ev.values) >= 0).all()
    terms = {}
    for line in Path(C["station_terms"]).read_text().splitlines():
        if line.startswith("LOCDELAY"):
            r = line.split()
            terms[(r[1], r[2])] = float(r[4])
    st_term = np.array([[terms.get((s, ph), np.nan) for ph in ("P", "S")] for s in st.station])
    _W.update(ev=ev, pk_ev=pk.ev.values, pk_sta=pk.sta.values, pk_sp=pk.sp.values, pk_tt=pk.tt.values,
              pk_w=pk.w.values, dt=dict(e1=dt.e1.values, e2=dt.e2.values, sp=dt.sp.values, pk1=dt.pk1.values,
                                         pk2=dt.pk2.values, w=dt.w.values), dptr=dptr, pptr=pptr, st=st,
              st_term=st_term, mem=mem.groupby("cid"), cl=cl.set_index("cid"), C=C)


def model():
    if "ds" not in _W:
        import xarray as xr
        _W["ds"] = xr.open_dataset(_W["C"]["model"]).load()
    return _W["ds"]


def write_mod(path, lat, lon, center):
    """MOD (b5_tomodd.write_mod of the D-B048 test, model version as parameter): netCDF nodes over the given points
    + node_margin_deg, edge nodes tomodd.edge_deg / edge_z, Vp/Vs, D-B028 P-2 rescaling about the centre."""
    from b2_fwd_mods import ellipsoid_coords
    ds = model()
    m = B["tomodd"]["node_margin_deg"]
    s = ds.sel(longitude=slice(lon.min() - m, lon.max() + m), latitude=slice(lat.min() - m, lat.max() + m))
    xi, yi, zi = s.longitude.values, s.latitude.values, s.depth.values
    vp = s.Vp.values.transpose(2, 1, 0).astype(float)
    r = (s.Vp / s.Vs).values.transpose(2, 1, 0).astype(float)
    E = T["edge_deg"]
    xn, yn = np.r_[xi[0] - E, xi, xi[-1] + E], np.r_[yi[0] - E, yi, yi[-1] + E]
    zt, zb = T["edge_z"]
    zn = np.r_[zt, zi, zb]
    vp = np.pad(vp, ((1, 1), (1, 1), (0, 0)), mode="edge")
    r = np.pad(r, ((1, 1), (1, 1), (0, 0)), mode="edge")
    vp = np.concatenate([vp[:, :, :1], vp, np.full(vp.shape[:2] + (1,), T["deep_vp"][int(zb)])], axis=2)
    r = np.concatenate([r[:, :, :1], r, r[:, :, -1:]], axis=2)
    mx = B["tomodd"]["mod_max"]
    assert len(xn) <= mx["nx"] and len(yn) <= mx["ny"] and len(zn) <= mx["nz"], (len(xn), len(yn), len(zn))
    assert np.isfinite(vp).all() and np.isfinite(r).all()
    yw, xw = ellipsoid_coords(yn, xn, *center)
    lines = [f"0.01 {len(xw)} {len(yw)} {len(zn)}", " ".join(f"{v:.4f}" for v in xw),
             " ".join(f"{v:.4f}" for v in yw), " ".join(f"{v:.3f}" for v in zn)]
    for arr, fmt in ((vp, "{:.4f}"), (r, "{:.5f}")):
        for k in range(len(zn)):
            for j in range(len(yn)):
                lines.append(" ".join(fmt.format(v) for v in arr[:, j, k]))
    Path(path).write_text("\n".join(lines) + "\n")
    return dict(nx=len(xn), ny=len(yn), nz=len(zn), lon=[float(xi[0]), float(xi[-1])], lat=[float(yi[0]), float(yi[-1])])


def cluster_data(cid):
    """Event positions, pick rows and dt rows of one cluster (shared tables)."""
    g = _W["mem"].get_group(cid)
    e = g.ev.values
    comp = int(_W["cl"].loc[cid, "comp"])
    D = _W["dt"]
    s0, s1 = _W["dptr"][comp], _W["dptr"][comp + 1]
    inset = np.zeros(len(_W["ev"]), bool)
    inset[e] = True
    rows = np.arange(s0, s1)[inset[D["e1"][s0:s1]] & inset[D["e2"][s0:s1]]]
    pp = np.concatenate([np.arange(_W["pptr"][i], _W["pptr"][i + 1]) for i in np.sort(e)])
    return g, np.sort(e), pp, rows


def write_inputs(rdir, cid, boot=0):
    from b2_fwd_mods import ellipsoid_coords
    ev, st = _W["ev"], _W["st"]
    g, e, pp, rows = cluster_data(cid)
    inp = rdir / "Input_Files"
    inp.mkdir(parents=True, exist_ok=True)
    (rdir / "Output_Files").mkdir(exist_ok=True)
    center = frame_centre(ev.lat.values[e], ev.lon.values[e], B["frame_clear_km"])
    dmin = float(np.min(np.hypot((ev.lat.values[e] - center[0]) * KM_DEG,
                                 (ev.lon.values[e] - center[1]) * KM_DEG * np.cos(np.radians(center[0])))))
    assert dmin >= B["frame_clear_km"], f"{cid}: an event lies {dmin * 1e3:.1f} m from the frame centre"
    sta = np.unique(_W["pk_sta"][pp])
    code = st.code.values
    la, lo = ellipsoid_coords(st.lat.values[sta], st.lon.values[sta], *center)
    with open(inp / "station.dat", "w") as f:
        for c_, a, b, z in zip(code[sta], la, lo, st.zsrc_km.values[sta]):
            f.write(f"{c_} {a:.6f} {b:.6f} {-z * 1000.0:.1f}\n")
    la, lo = ellipsoid_coords(ev.lat.values[e], ev.lon.values[e], *center)
    t_ns = ev.t_ns.values[e]
    tcs = (t_ns // 10_000_000) * 10_000_000          # written origin time: floor to 10 ms
    ts = pd.to_datetime(tcs)
    with open(inp / "event.dat", "w") as f:
        for i, (t, a, b) in enumerate(zip(ts, la, lo)):
            f.write(f"{t.strftime('%Y%m%d')}  {t.hour:02d}{t.minute:02d}{t.second * 100 + t.microsecond // 10000:04d} "
                    f"{a:10.6f} {b:11.6f} {ev.depth.values[e[i]]:8.3f} 1.0 0.0 0.0 0.0 {ev.ID.values[e[i]]:10d} 0\n")
    # absolute times (zero weight in variant dd): pick - B4 origin - B4 station term (missing term -> 0)
    ph = _W["pk_sp"][pp] % 2
    term = _W["st_term"][_W["pk_sta"][pp], ph]
    n_noterm = int(np.isnan(term).sum())
    tt = _W["pk_tt"][pp] - np.nan_to_num(term)
    ids = ev.ID.values[_W["pk_ev"][pp]]
    lines, last = [], None
    phs = np.where(ph == 1, "S", "P")
    for i in range(len(pp)):
        if ids[i] != last:
            lines.append(f"#  {ids[i]}")
            last = ids[i]
        lines.append(f"{code[_W['pk_sta'][pp[i]]]:<7s} {tt[i]:9.4f} {_W['pk_w'][pp[i]]:6.3f} {phs[i]}")
    (inp / "absolute.dat").write_text("\n".join(lines) + "\n")
    D = _W["dt"]
    if boot:
        rng = np.random.default_rng(48000 + boot)
        rows = rows[np.sort(rng.integers(0, len(rows), len(rows)))]
    e1, e2, sp = D["e1"][rows], D["e2"][rows], D["sp"][rows]
    t1, t2, w = _W["pk_tt"][D["pk1"][rows]], _W["pk_tt"][D["pk2"][rows]], D["w"][rows]
    o = np.lexsort((sp, e2, e1))
    lines, last = [], None
    id1, id2 = ev.ID.values[e1], ev.ID.values[e2]
    for i in o:
        k = (id1[i], id2[i])
        if k != last:
            lines.append(f"#  {k[0]}  {k[1]}")
            last = k
        lines.append(f"{code[sp[i] // 2]:<7s} {t1[i]:9.4f} {t2[i]:9.4f} {w[i]:6.3f} {'S' if sp[i] % 2 else 'P'}")
    (inp / "dt.ct").write_text("\n".join(lines) + "\n")
    mod = write_mod(rdir / "MOD", np.r_[ev.lat.values[e], st.lat.values[sta]],
                    np.r_[ev.lon.values[e], st.lon.values[sta]], center)
    air = round(float(st.zsrc_km.values[sta].min()) - B["tomodd"]["air_dep_margin_km"], 3)
    return dict(center=list(center), min_dist_centre_km=dmin, air_dep_km=air, mod=mod, n_events=len(e),
                n_sta=len(sta), n_abs=len(pp), n_dt=len(rows), picks_missing_term=n_noterm), tcs


def parse_stdout(text):
    """Iteration table of tomoDD-SE (IT EV CT RMSCT RMSST DX DY DZ DT OS AQ CND)."""
    it = []
    for line in text.splitlines():
        t = line.split()
        if len(t) in (13, 14):          # an air-quake iteration (no update) prints no second iteration number
            try:
                v = [float(x) for x in t]
            except ValueError:
                continue
            if len(t) == 13:
                v.insert(1, np.nan)
            it.append(dict(zip(["it", "jit", "ev_pct", "ct_pct", "rmsct_ms", "rmsct_chg", "rmsst_ms", "dx_m", "dy_m",
                                "dz_m", "dt_ms", "os_m", "aq", "cnd"], v)))
    return it


def run_tomodd(rdir, cid, info, tag="", damp=None):
    cls = _W["cl"].loc[cid, "cls"]
    binf = Path(B["tomodd"]["bin_dir"]) / f"tomoDD-SE_{cls}"
    rt = Path(B["tomodd"]["runtime"])
    for f in ("ak135.15.SKS", "layer-16.dat"):
        if not (rdir / f).exists():
            shutil.copy(rt / f, rdir / f)
    sets = B["tomodd"]["sets"]
    damp = B["tomodd"]["damp"] if damp is None else damp
    rows = [f"  {s[0]:<3d}  -9    -9    -9   -9  {s[1]:5.2f} {s[2]:5.2f} {s[3]:4g} {s[4]:4g} {s[5]:6g} {damp:6g}"
            f"   0  {T['inp']['thres']:g}" for s in sets]
    w1, w2, w3 = (f"{v:g}" for v in T["inp"]["smooth"])
    (rdir / "tomoDD-SE.inp").write_text(INP.format(
        run=_W["C"]["run"], cid=cid, tag=tag, air=info["air_dep_km"], nset=len(rows), stepl=T["inp"]["stepl_km"],
        clat=info["center"][0], clon=info["center"][1], w1=w1, w2=w2, w3=w3, sets="\n".join(rows)))
    # memory guard: wait while free memory is short (another session, NLLoc)
    need = B["min_free_gb"] + _W.get("rss_gb", 1.0)
    waited = 0
    while mem_available_gb() < need:
        time.sleep(15)
        waited += 15
    t0 = time.time()
    with open(rdir / "stdout.txt", "w") as so:
        rc = subprocess.run(["/usr/bin/time", "-f", "%M %e %U", "-o", "time.txt", str(binf), "tomoDD-SE.inp"],
                            cwd=rdir, stdout=so, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    try:
        rss_kb, _, cpu = (float(x) for x in (rdir / "time.txt").read_text().split()[-3:])
    except Exception:
        rss_kb, cpu = np.nan, np.nan
    it = parse_stdout((rdir / "stdout.txt").read_text(errors="replace"))
    lg = (rdir / "tomoDD-SE.log").read_text(errors="replace") if (rdir / "tomoDD-SE.log").exists() else ""
    aq_ids = sorted({int(l.split("-")[-1]) for l in lg.splitlines() if "negative depth" in l})
    n_aq = len(aq_ids)
    skipped = "skipping this cluster" in lg
    return dict(cls=cls, bin=str(binf), exit=rc, wall_s=round(wall, 1), cpu_s=cpu, rss_gb=round(rss_kb / 1024 ** 2, 3),
                mem_wait_s=waited, iterations=len(it), airquakes=n_aq, airquake_ids=aq_ids, skipped=skipped,
                cnd_first=it[0]["cnd"] if it else None, cnd_last=it[-1]["cnd"] if it else None,
                rmsct_first_ms=it[0]["rmsct_ms"] if it else None, rmsct_last_ms=it[-1]["rmsct_ms"] if it else None,
                ev_pct_last=it[-1]["ev_pct"] if it else None, ct_pct_last=it[-1]["ct_pct"] if it else None,
                os_last_m=it[-1]["os_m"] if it else None), it


def read_reloc(path, center, e_ids):
    from b2_fwd_mods import radii
    from b2_tomodd_backtransform import inverse
    cols = ["ID", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez", "yr", "mo", "dy", "hr", "mi", "sc", "mag",
            "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
    r = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    M, N = radii(center[0])
    fw = {"coords": "ellipsoid", "center": list(center), "lat_scale_M_over_R": M / RE, "lon_scale_N_over_R": N / RE}
    r["lat"], r["lon"] = inverse(r.lat.values, r.lon.values, fw)
    for c in ("ex", "ey", "ez"):
        r[c] = r[c] / 1000.0
    sec = np.floor(r.sc.values).astype(np.int64)
    ns = np.rint((r.sc.values - sec) * 1000).astype(np.int64) * 1_000_000
    t = pd.to_datetime(dict(year=r.yr, month=r.mo, day=r.dy, hour=r.hr, minute=r.mi)).values.astype(
        "datetime64[ns]").astype(np.int64) + sec * 1_000_000_000 + ns
    r["t_ns"] = t
    assert r.ID.isin(e_ids).all()
    return r[["ID", "lat", "lon", "depth", "ex", "ey", "ez", "nctp", "ncts", "rct", "t_ns"]]


def do_cluster(task):
    """One tomoDD run. Returns its run.json record; status ok / skipped (tomoDD: < 2 events left) / failed (tomoDD
    exit != 0 or no output; recorded, not retried) / error (exception here; no run.json, retried by a rerun) / busy."""
    cid, boot = task
    C = _W["C"]
    rdir = run_dir(C, cid, boot)
    if (rdir / "run.json").exists():
        return json.loads((rdir / "run.json").read_text())
    lockf = C["dir"] / "locks" / (cid + (f"_b{boot:03d}" if boot else "") + ".lock")
    if not acquire(lockf):                        # another driver runs it now: leave it alone
        return dict(cid=cid, boot=boot, status="busy")
    if alive_pid(rdir / "running.pid"):           # (a driver of the first version: no lock file)
        lockf.unlink(missing_ok=True)
        return dict(cid=cid, boot=boot, status="busy")
    try:
        if (rdir / "run.json").exists():          # finished by another driver meanwhile
            return json.loads((rdir / "run.json").read_text())
        return _do_cluster(cid, boot, rdir)
    except Exception as e:                        # noqa: BLE001 (reported, retried by a rerun of the stage)
        return dict(cid=cid, boot=boot, status="error", error=f"{type(e).__name__}: {e}")
    finally:
        lockf.unlink(missing_ok=True)


def _do_cluster(cid, boot, rdir):
    from manifest import _md5
    out = rdir / "reloc.csv"
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    pidf = rdir / "running.pid"
    pidf.write_text(str(os.getpid()))
    t0 = time.time()
    info, tcs = write_inputs(rdir, cid, boot)
    t_in = time.time() - t0
    res, it = run_tomodd(rdir, cid, info, "" if boot == 0 else f" boot {boot}")
    _W["rss_gb"] = max(_W.get("rss_gb", 1.0), res["rss_gb"] if np.isfinite(res["rss_gb"]) else 0)
    rel = rdir / "Output_Files" / "tomoDD.reloc"
    ev = _W["ev"]
    g = _W["mem"].get_group(cid).sort_values("ev")
    if res["exit"] == 0 and rel.exists() and rel.stat().st_size > 0:
        r = read_reloc(rel, info["center"], ev.ID.values[g.ev.values])
        tw = pd.Series(tcs, index=ev.ID.values[np.sort(g.ev.values)])
        r["dt_s"] = (r.t_ns.values - tw.loc[r.ID].values) / 1e9       # tomoDD origin-time shift (10 ms steps)
        r = r.drop(columns="t_ns")
        r.to_csv(out, index=False, float_format="%.8g")
        res["relocated"] = len(r)
        res["status"] = "ok"
    else:
        res["relocated"] = 0
        res["status"] = "skipped" if res["skipped"] else "failed"
    res.update(cid=cid, boot=boot, inputs_s=round(t_in, 1), **info)
    (rdir / "iterations.json").write_text(json.dumps(it))
    # per-run record with checksums (local; the stage writes one manifest for all runs)
    (rdir / "files.json").write_text(json.dumps(
        {"inputs": [_md5(f) for f in (rdir / "MOD", rdir / "Input_Files" / "dt.ct", rdir / "Input_Files" / "event.dat",
                                      rdir / "Input_Files" / "absolute.dat", rdir / "Input_Files" / "station.dat",
                                      res["bin"])],
         "outputs": [_md5(f) for f in (out, rel, rdir / "tomoDD-SE.inp", rdir / "stdout.txt")]}, indent=1))
    # large and not used: initial residuals; velocity outputs (relocation only: the input model unchanged)
    for f in ("tomoDD-SE.res.ini", "Output_Files/tomoDD.vel", "Output_Files/Vp_model.dat", "Output_Files/Vs_model.dat"):
        if (rdir / f).exists():
            (rdir / f).unlink()
    (rdir / "run.json").write_text(json.dumps(res, indent=1, default=float))
    pidf.unlink()
    return res


def alive_pid(f):
    try:
        os.kill(int(Path(f).read_text()), 0)
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def acquire(lockf):
    """Atomic per-run lock (O_EXCL) holding the owner's PID; a lock whose owner is gone is taken over."""
    lockf.parent.mkdir(exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(lockf, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if alive_pid(lockf):
                return False
            try:
                os.unlink(lockf)
            except FileNotFoundError:
                pass
    return False


def run_dir(C, cid, boot=0):
    """clusters/<cid> (main run) or boot/<cid>/bNNN (bootstrap member k)."""
    return C["dir"] / "clusters" / cid if boot == 0 else C["dir"] / "boot" / cid / f"b{boot:03d}"


def stage_run(C, nproc, only=None, nboot=0, reverse=False):
    import multiprocessing as mp
    load_shared(C)
    cl = _W["cl"]
    cids = list(cl.sort_values("n_data", ascending=reverse).index)     # largest first (reverse: a helper driver)
    if only:
        cids = [c for c in cids if c in only]
    tasks = [(c, 0) for c in cids] + [(c, k) for c in cids for k in range(1, nboot + 1)]
    todo = [t for t in tasks if not (run_dir(C, *t) / "run.json").exists()]
    log(f"run: {len(tasks)} tomoDD runs ({len(cids)} clusters, nboot {nboot}), {len(todo)} to do, {nproc} processes, "
        f"free memory {mem_available_gb():.0f} GB")
    t0 = time.time()
    n_ok = n_bad = n_skip = n_fail = 0
    recs = []
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, maxtasksperchild=200) as pool:
        for i, r in enumerate(pool.imap_unordered(do_cluster, todo, chunksize=1)):
            recs.append({k: r.get(k) for k in ("cid", "boot", "status", "cls", "wall_s", "rss_gb", "n_events",
                                                "relocated", "airquakes", "cnd_first", "cnd_last")})
            if r.get("status") == "ok":
                n_ok += 1
            elif r.get("status") == "skipped":     # tomoDD: fewer than 2 events left (air quakes); events keep B4
                n_skip += 1
                log(f"{r.get('cid')} boot {r.get('boot')}: skipped by tomoDD ({r.get('airquakes')} air quakes)")
            elif r.get("status") == "failed":      # tomoDD exit != 0: recorded (events keep B4), not retried
                n_fail += 1
                log(f"{r.get('cid')} boot {r.get('boot')}: tomoDD FAILED (exit {r.get('exit')}), see its stdout.txt")
            else:
                n_bad += 1
                log(f"{r.get('cid')} boot {r.get('boot')}: {r.get('status')} {r.get('error', '')}")
            if (i + 1) % 200 == 0 or r.get("n_events", 0) >= 1000:
                log(f"{i + 1}/{len(todo)} done ({n_bad} not ok), last {r.get('cid')} {r.get('n_events')} events "
                    f"{r.get('wall_s')} s rss {r.get('rss_gb')} GB; elapsed {(time.time() - t0) / 60:.1f} min, "
                    f"free {mem_available_gb():.0f} GB")
    wall = (time.time() - t0) / 60
    log(f"run done: {n_ok} ok, {n_skip} skipped by tomoDD, {n_fail} tomoDD failures, {n_bad} errors or busy, "
        f"{wall:.1f} min")
    if not only:
        (C["dir"] / "run_stage.json").write_text(json.dumps(recs, default=float))
        write_manifest("b5_run", C["dir"], inputs=[C["dir"] / "prep" / f for f in ("events.parquet", "picks.parquet",
                                                                                   "dt.parquet")]
                       + [C["dir"] / "split" / "members.parquet", C["model"], C["station_terms"]]
                       + [Path(B["tomodd"]["bin_dir"]) / f"tomoDD-SE_{c}" for c in B["tomodd"]["classes"]],
                       outputs=[C["dir"] / "run_stage.json"],
                       params={"tomodd": B["tomodd"], "nproc": nproc, "nboot": nboot},
                       counts=dict(runs=len(tasks), done_now=len(todo), ok=n_ok, skipped=n_skip, failed=n_fail,
                                   errors_or_busy=n_bad, wall_min=round(wall, 1)))
    if n_bad:
        raise SystemExit(f"{n_bad} runs with an error or busy in another driver (see the log); rerun the stage")


# ------------------------------------------------------------------------------------------------ collect
def b4_dt_rms(C, ev, pk, dt, pos_b4, pos_b5, nproc):
    """dt residuals (s) in the B4 time grids at B4 and B5 positions, per-pair median removed (b5_compare.dt_misfit).
    Returns per-dt residual arrays (NaN where an event has no B5 position)."""
    import multiprocessing as mp
    st = pd.read_csv(C["dir"] / "prep" / "stations.csv")
    need = np.unique(np.r_[dt.pk1.values, dt.pk2.values])
    psp = pk.sp.values[need]
    o = np.argsort(psp, kind="stable")
    need, psp = need[o], psp[o]
    bounds = np.r_[0, np.where(np.diff(psp))[0] + 1, len(psp)]
    jobs = [(int(psp[bounds[i]]), need[bounds[i]:bounds[i + 1]]) for i in range(len(bounds) - 1)]
    _W.update(gC=C, gst=st, gpk_ev=pk.ev.values, gpos=(pos_b4, pos_b5))
    ctx = mp.get_context("fork")
    T4 = np.full(len(pk), np.nan)
    T5 = np.full(len(pk), np.nan)
    with ctx.Pool(nproc) as pool:
        for sel, a, b in pool.imap_unordered(_grid_job, jobs, chunksize=4):
            T4[sel], T5[sel] = a, b
    out = {}
    for name, TT in (("b4", T4), ("b5", T5)):
        r = (pk.tt.values[dt.pk1] - pk.tt.values[dt.pk2]) - (TT[dt.pk1] - TT[dt.pk2])
        r = pd.Series(r)
        med = r.groupby([dt.e1.values, dt.e2.values]).transform("median")
        out[name] = (r - med).values
    return out


def _grid_job(job):
    sp, sel = job
    C, st = _W["gC"], _W["gst"]
    G = _W.setdefault("G", Grids(C["grids"]))
    sta, ph = st.station.values[sp // 2], "S" if sp % 2 else "P"
    e = _W["gpk_ev"][sel]
    res = []
    for p in _W["gpos"]:
        x, y, zf = p["x"][e], p["y"][e], p["zf"][e]
        t = np.full(len(e), np.nan)
        ok = np.isfinite(x)
        if ok.any():
            t[ok] = G.time(ph, sta, x[ok], y[ok], zf[ok])
        res.append(t)
    return sel, res[0], res[1]


def boot_spread(cdir, core, E):
    """Bootstrap spread of a cluster (runs <run>/boot/<cid>/bNNN): each member's core centroid put back on B4 as in the main run,
    then per core event the rms horizontal distance from the member mean and the std of depth (km), over the members
    where the event was relocated. None if no member ran."""
    runs = sorted(cdir.glob("b[0-9][0-9][0-9]/reloc.csv"))
    if not runs:
        return None
    ids = sorted(core)
    X, wall = [], 0.0
    for f in runs:
        r = pd.read_csv(f).set_index("ID")
        rc = r[r.index.isin(core)]
        if len(rc) == 0:
            continue
        b = E.loc[rc.index]
        r["lat"] -= (rc.lat - b.lat).mean()
        r["lon"] -= (rc.lon - b.lon).mean()
        r["depth"] -= (rc.depth - b.depth).mean()
        X.append(r[["lat", "lon", "depth"]].reindex(ids))
        wall += json.loads((f.parent / "run.json").read_text()).get("wall_s", 0.0)
    if not X:
        return None
    la, lo, z = (np.stack([x[c].values for x in X], 1) for c in ("lat", "lon", "depth"))
    e = (lo - np.nanmean(lo, 1, keepdims=True)) * KM_DEG * np.cos(np.radians(np.nanmean(la)))
    n = (la - np.nanmean(la, 1, keepdims=True)) * KM_DEG
    out = pd.DataFrame({"ID": ids, "boot_sh": np.sqrt(np.nanmean(e ** 2 + n ** 2, 1)), "boot_sz": np.nanstd(z, 1),
                        "boot_n": np.isfinite(z).sum(1)})
    out.attrs.update(nboot=len(X), wall_s=wall)
    return out


def stage_collect(C, nproc, partial=False):
    sys.path.insert(0, str(HERE))
    from b4_catalog import V4, flags, surface_km
    d = C["dir"]
    ev = pd.read_parquet(d / "prep" / "events.parquet")
    cl = pd.read_csv(d / "split" / "clusters.csv").set_index("cid")
    mem = pd.read_parquet(d / "split" / "members.parquet")
    mem["ID"] = ev.ID.values[mem.ev]
    E = ev.set_index("ID")
    rows, sols, missing, aq = [], [], [], set()
    for cid in cl.index:
        f = d / "clusters" / cid / "run.json"
        if not f.exists():
            missing.append(cid)
            continue
        rj = json.loads(f.read_text())
        if "airquake_ids" not in rj:              # run.json of a first test version: take them from the log
            lg = (f.parent / "tomoDD-SE.log").read_text(errors="replace").splitlines()
            rj["airquake_ids"] = sorted({int(l.split("-")[-1]) for l in lg if "negative depth" in l})
            rj["airquakes"] = len(rj["airquake_ids"])
        aq |= set(rj["airquake_ids"])
        g = mem[mem.cid == cid]
        core = set(g.ID[g.core])
        rf = d / "clusters" / cid / "reloc.csv"
        r = pd.read_csv(rf) if rf.exists() else pd.DataFrame(columns=["ID", "lat", "lon", "depth", "dt_s"])
        r["core"] = r.ID.isin(core)
        r["cid"] = cid
        rc = r[r.core]
        b = E.loc[rc.ID]
        if len(rc):
            sh = dict(dlat=float((rc.lat.values - b.lat.values).mean()), dlon=float((rc.lon.values - b.lon.values).mean()),
                      ddep=float((rc.depth.values - b.depth.values).mean()), dt_s=float(rc.dt_s.mean()))
        else:
            sh = dict(dlat=0.0, dlon=0.0, ddep=0.0, dt_s=0.0)
        r["lat_f"], r["lon_f"], r["depth_f"] = r.lat - sh["dlat"], r.lon - sh["dlon"], r.depth - sh["ddep"]
        r["dt_f"] = r.dt_s - sh["dt_s"]
        rc = r[r.core]
        lat0 = float(b.lat.mean()) if len(rc) else float(cl.loc[cid, "lat"])
        row = dict(cid=cid, **{k: rj.get(k) for k in ("status", "wall_s", "cpu_s", "rss_gb", "inputs_s",
                                                       "iterations", "airquakes", "cnd_first", "cnd_last",
                                                       "rmsct_first_ms", "rmsct_last_ms", "ev_pct_last", "ct_pct_last",
                                                       "os_last_m", "min_dist_centre_km", "air_dep_km")},
                   relocated=len(r), relocated_core=len(rc),
                   shift_e_km=sh["dlon"] * KM_DEG * np.cos(np.radians(lat0)), shift_n_km=sh["dlat"] * KM_DEG,
                   shift_z_km=sh["ddep"], shift_t_s=sh["dt_s"],
                   ex_med_km=float(rc.ex.median()) if len(rc) else np.nan,
                   ey_med_km=float(rc.ey.median()) if len(rc) else np.nan,
                   ez_med_km=float(rc.ez.median()) if len(rc) else np.nan,
                   center_lat=rj.get("center", [np.nan])[0], center_lon=rj.get("center", [np.nan, np.nan])[1])
        if len(rc) and rj.get("center"):
            c0 = rj["center"]
            dd_ = np.hypot((rc.lat.values - c0[0]) * KM_DEG,
                           (rc.lon.values - c0[1]) * KM_DEG * np.cos(np.radians(c0[0])))
            row["min_dist_centre_final_km"] = float(dd_.min())
        if len(rc) >= B["collect"]["pca_min_events"]:
            X4 = np.c_[(b.lon.values - b.lon.mean()) * KM_DEG * np.cos(np.radians(lat0)),
                       (b.lat.values - b.lat.mean()) * KM_DEG, b.depth.values]
            X5 = np.c_[(rc.lon_f.values - b.lon.mean()) * KM_DEG * np.cos(np.radians(lat0)),
                       (rc.lat_f.values - b.lat.mean()) * KM_DEG, rc.depth_f.values]
            s4, _, _, _ = pca(X4)
            s5, _, _, _ = pca(X5)
            row.update(pca1_b4=s4[0], pca2_b4=s4[1], pca3_b4=s4[2], pca1_b5=s5[0], pca2_b5=s5[1], pca3_b5=s5[2])
        bt = boot_spread(d / "boot" / cid, core, E)
        if bt is not None:
            row.update(nboot=int(bt.attrs["nboot"]), boot_wall_s=bt.attrs["wall_s"],
                       boot_sh_med=float(bt.boot_sh.median()), boot_sz_med=float(bt.boot_sz.median()))
            r = r.merge(bt, on="ID", how="left")
        rows.append(row)
        sols.append(r)
    if missing and not partial:
        raise SystemExit(f"{len(missing)} clusters have no run.json (run stage not finished): {missing[:10]}")
    cs = pd.DataFrame(rows).set_index("cid")
    cs = cl.join(cs, how="inner")
    S = pd.concat(sols, ignore_index=True) if sols else pd.DataFrame()
    # overlap duplicates: events relocated as core in one chunk and as halo in another (split effect)
    dup = S[~S.core].merge(S[S.core][["ID", "lat_f", "lon_f", "depth_f"]], on="ID", suffixes=("_halo", ""))
    split_eff = {}
    if len(dup):
        dh = np.hypot((dup.lat_f_halo - dup.lat_f) * KM_DEG,
                      (dup.lon_f_halo - dup.lon_f) * KM_DEG * np.cos(np.radians(dup.lat_f)))
        dz = (dup.depth_f_halo - dup.depth_f).abs()
        hop = dup.merge(mem[["cid", "ID", "hop"]], on=["cid", "ID"]).hop
        split_eff = dict(n=len(dup), dh_med=float(dh.median()), dh_p90=float(dh.quantile(0.9)),
                         dz_med=float(dz.median()), dz_p90=float(dz.quantile(0.9)),
                         by_hop={int(h): dict(n=int((hop == h).sum()), dh_med=float(dh[(hop == h).values].median()),
                                              dz_med=float(dz[(hop == h).values].median())) for h in sorted(hop.unique())})
    core = S[S.core].set_index("ID")
    # ---- catalog
    cat = pd.read_csv(C["catalog_b4"])
    excl = [int(i) for i in B["exclude_ids"]]
    cat = cat[~cat.ID.isin(excl)].reset_index(drop=True)
    nll = (cat.loc_source == B["loc_source"]).values
    # full-precision B4 (nll3d) and v4 (tier U) positions, as b4_catalog.py
    cat.loc[nll, ["lat", "lon", "depth"]] = E.loc[cat.ID[nll], ["lat", "lon", "depth"]].values
    v4 = pd.read_csv(V4, skipinitialspace=True).set_index("ID")
    u = ~nll
    for k, s in (("lat", "latitude"), ("lon", "longitude"), ("depth", "depth(km)")):
        new = v4[s].reindex(cat.ID[u]).to_numpy(float)
        old = cat.loc[u, k].to_numpy(float)
        assert np.all(np.abs(new - old) <= 1e-4 * np.abs(old) + 1e-6), k
        cat.loc[u, k] = new
    cat.loc[nll, "time"] = E.loc[cat.ID[nll], "time"].values
    for k in ("time", "lat", "lon", "depth", "depth_quality", "event_flag"):
        if k in cat:
            cat[f"b4_{k}"] = cat[k]
    reloc = cat.ID.isin(core.index).values
    rr = core.reindex(cat.ID[reloc])
    cat.loc[reloc, "lat"] = rr.lat_f.values
    cat.loc[reloc, "lon"] = rr.lon_f.values
    cat.loc[reloc, "depth"] = rr.depth_f.values
    tn = E.loc[cat.ID[reloc], "t_ns"].values + np.rint(rr.dt_f.values * 1e9).astype(np.int64)
    cat.loc[reloc, "time"] = pd.to_datetime(tn).strftime("%Y-%m-%dT%H:%M:%S.%f").values
    cat["loc_source"] = np.where(reloc, "b5_tomodd", cat.loc_source)
    memc = mem[mem.core].set_index("ID")
    cat["b5_cluster"] = memc.cid.reindex(cat.ID).values
    cat["b5_component"] = E.comp.reindex(cat.ID).values
    comp_size = E[E.comp >= 0].groupby("comp").size()
    cat["b5_component_size"] = pd.Series(cat.b5_component).map(comp_size).values
    cat["b5_n_pairs"] = E.n_pairs.reindex(cat.ID).values
    for k in ("ex", "ey", "ez", "nctp", "ncts", "rct", "boot_sh", "boot_sz", "boot_n"):
        if k in core:
            cat[f"b5_{k}"] = core[k].reindex(cat.ID).values
    cat["b5_dt_s"] = core.dt_f.reindex(cat.ID).values
    # why an event was not relocated: tier_U (v4 kept), unlinked (no dt pair), airquake_by_tomodd (tomoDD removed
    # it: an update would have put it above Air_dep, i.e. above the cluster's highest station), dropped_by_tomodd (no
    # data left after re-weighting), cluster_<status> (cluster skipped/failed/not run)
    lk = E.comp.reindex(cat.ID).values
    nr = nll & (lk >= 0) & ~reloc
    cst = pd.Series(cs.status.reindex(cat.b5_cluster[nr]).values).fillna("not_run")
    st_ = np.full(len(cat), "relocated", dtype=object)
    st_[~nll] = "tier_U"
    st_[nll & ~(lk >= 0)] = "unlinked"
    st_[nr] = np.where(cst == "ok", "dropped_by_tomodd", "cluster_" + cst)
    st_[nr & cat.ID.isin(aq).values] = "airquake_by_tomodd"     # update would put it above Air_dep (tomoDD drops it)
    cat["b5_status"] = st_
    cat["b5_shift_h_km"] = np.where(reloc, np.hypot((cat.lat - cat.b4_lat) * KM_DEG, (cat.lon - cat.b4_lon) * KM_DEG
                                                    * np.cos(np.radians(cat.b4_lat))), np.nan)
    cat["b5_shift_z_km"] = np.where(reloc, cat.depth - cat.b4_depth, np.nan)
    cat["b4_surface_km"] = surface_km(cat.b4_lat, cat.b4_lon)       # at the start position (B4 or v4)
    cat["b4_above_surface"] = cat.b4_depth < -cat.b4_surface_km
    cat["surface_km"] = surface_km(cat.lat, cat.lon)
    cat["above_surface"] = cat.depth < -cat.surface_km
    cat["depth_frame"] = np.where(cat.loc_source == "v4", "v4", "true depth (km bsl; B4 EFT undone)")
    cat["depth_quality"], cat["event_flag"] = flags(cat)
    od = C["summary"]
    od.mkdir(parents=True, exist_ok=True)
    tag = "_partial" if partial else ""
    out = d / f"catalog_b5_{C['run']}{tag}.csv"
    cat.to_csv(out, index=False, float_format="%.8g")
    cs.to_csv(d / f"clusters_summary{tag}.csv", float_format="%.6g")
    # ---- dt residual rms in the B4 grids, before (B4) / after (B5), relocated core-core pairs
    pk = pd.read_parquet(d / "prep" / "picks.parquet", columns=["ev", "sp", "tt"])
    dt = pd.read_parquet(d / "prep" / "dt.parquet", columns=["e1", "e2", "sp", "pk1", "pk2"])
    fin = pd.Series(np.nan, index=ev.ID)
    pos = {}
    lat5 = core.lat_f.reindex(ev.ID).values
    lon5 = core.lon_f.reindex(ev.ID).values
    dep5 = core.depth_f.reindex(ev.ID).values
    x5, y5 = lambert_km(np.nan_to_num(lon5, nan=-125.0), np.nan_to_num(lat5, nan=45.0))
    x5[np.isnan(lon5)] = np.nan
    pos_b5 = dict(x=x5, y=y5, zf=eft_depth(dep5))
    pos_b4 = dict(x=ev.x_km.values, y=ev.y_km.values, zf=ev.zf_km.values)
    del fin, pos
    both = np.isfinite(x5[dt.e1.values]) & np.isfinite(x5[dt.e2.values])
    dts = dt[both].reset_index(drop=True)
    res = b4_dt_rms(C, ev, pk, dts, pos_b4, pos_b5, nproc) if len(dts) else {"b4": np.array([]), "b5": np.array([])}
    reg = ev.region.values[dts.e1.values]
    cid1 = memc.cid.reindex(ev.ID.values[dts.e1.values]).values
    R = pd.DataFrame({"region": reg, "cid": cid1, "ph": np.where(dts.sp.values % 2, "S", "P"), "r4": res["b4"],
                      "r5": res["b5"]})
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan  # noqa: E731
    dtr = R.groupby(["region", "ph"]).agg(n=("r4", "size"), rms_b4=("r4", rms), rms_b5=("r5", rms)).unstack("ph")
    cl_dt = R.groupby("cid").agg(dt_rms_b4=("r4", rms), dt_rms_b5=("r5", rms), n_dt_eval=("r4", "size"))
    cs = cs.join(cl_dt)
    cs.to_csv(d / f"clusters_summary{tag}.csv", float_format="%.6g")
    allr = {ph: (len(R[R.ph == ph]), rms(R.r4[R.ph == ph]), rms(R.r5[R.ph == ph])) for ph in ("P", "S")}
    write_compare(C, cat, cs, dtr, allr, split_eff, od, tag, missing)
    keep = ["cls", "n_events", "n_core", "n_halo", "n_sta", "n_dt", "n_abs", "status", "relocated_core", "wall_s",
            "rss_gb", "cnd_first", "cnd_last", "rmsct_first_ms", "rmsct_last_ms", "shift_e_km", "shift_n_km",
            "shift_z_km", "shift_t_s", "ex_med_km", "ez_med_km", "pca3_b4", "pca3_b5", "dt_rms_b4", "dt_rms_b5",
            "region"]
    cs[[k for k in keep if k in cs]].to_csv(od / f"clusters_summary{tag}.csv", float_format="%.5g")
    na = int(cat.above_surface.sum())
    counts = dict(events=len(cat), loc_source={k: int(v) for k, v in cat.loc_source.value_counts().items()},
                  b5_status={k: int(v) for k, v in cat.b5_status.value_counts().items()}, above_surface=na,
                  above_surface_b5=int((cat.above_surface & (cat.loc_source == "b5_tomodd")).sum()),
                  publishable=na == 0, clusters=len(cs), missing_clusters=len(missing), split_effect=split_eff)
    (od / f"collect{tag}.json").write_text(json.dumps(counts, indent=1, default=float))
    write_manifest(f"b5_collect_{C['run']}{tag}", od,
                   inputs=[C["catalog_b4"], C["locations"], V4, d / "split" / "clusters.csv"],
                   outputs=[out, d / f"clusters_summary{tag}.csv", od / f"compare{tag}.md",
                            od / f"clusters_summary{tag}.csv"],
                   params={"b5": B, "config": {k: str(v) for k, v in C.items()}}, counts=counts)
    log(f"collect: {counts}")


def write_compare(C, cat, cs, dtr, allr, split_eff, od, tag, missing):
    rel = cat.loc_source == "b5_tomodd"
    L = [f"# B5 relocation {C['run']} (tomoDD-SE relocation only, variant dd, model v{C['version']})", "",
         f"Start: B4 run {C['b4_run']} (NLLoc, full precision, EFT undone). Catalog: {C['dir']}/catalog_b5_{C['run']}"
         f"{tag}.csv ({len(cat)} events). Depths in km below sea level. B5 = tomoDD solution with each cluster's "
         "centroid put back on its B4 centroid (D-B053)."]
    if missing:
        L += ["", f"PARTIAL: {len(missing)} of {len(missing) + len(cs)} clusters not run; their events keep B4."]
    L += ["", "## Events by location source", "", "| region | events | b5_tomodd | nll3d (B4 kept) | v4 (tier U) | "
          "relocated % of B4 3D |", "|---|---|---|---|---|---|"]
    t = pd.crosstab(cat.region, cat.loc_source, margins=True, margins_name="all")
    for k, r in t.iterrows():
        b5, n4, v4 = (int(r.get(c, 0)) for c in ("b5_tomodd", "nll3d", "v4"))
        L.append(f"| {k} | {int(r['all'])} | {b5} | {n4} | {v4} | {100 * b5 / max(b5 + n4, 1):.1f} |")
    st = pd.crosstab(cat.region, cat.b5_status, margins=True, margins_name="all")
    L += ["", "## Why events were not relocated (b5_status)", "", "| region | " + " | ".join(st.columns) + " |",
          "|---" * (len(st.columns) + 1) + "|"]
    L += [f"| {k} | " + " | ".join(str(int(v)) for v in r) + " |" for k, r in st.iterrows()]
    s = cat[rel]
    g = s.groupby("region").agg(n=("b5_shift_h_km", "size"), dh_med=("b5_shift_h_km", "median"),
                                dh_p90=("b5_shift_h_km", lambda x: x.quantile(0.9)),
                                dz_med=("b5_shift_z_km", "median"),
                                adz_med=("b5_shift_z_km", lambda x: x.abs().median()),
                                adz_p90=("b5_shift_z_km", lambda x: x.abs().quantile(0.9)),
                                ex=("b5_ex", "median"), ez=("b5_ez", "median"))
    g.loc["all"] = [len(s), s.b5_shift_h_km.median(), s.b5_shift_h_km.quantile(0.9), s.b5_shift_z_km.median(),
                    s.b5_shift_z_km.abs().median(), s.b5_shift_z_km.abs().quantile(0.9), s.b5_ex.median(),
                    s.b5_ez.median()]
    L += ["", "## Shift from B4 (relocated events, km)", "",
          "| region | n | dh med / p90 | dz med | abs dz med / p90 | LSQR ex / ez med |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {int(v.n)} | {v.dh_med:.3f} / {v.dh_p90:.3f} | {v.dz_med:+.3f} | {v.adz_med:.3f} / {v.adz_p90:.3f}"
          f" | {v.ex:.3f} / {v.ez:.3f} |" for k, v in g.iterrows()]
    L += ["", "## dt residual rms in the B4 time grids (s), B4 -> B5", "",
          "Relocated core-core pairs; residual = observed minus B4-grid differential time, per-pair median removed "
          "(b5_compare.py metric of the D-B048 test).", "",
          "| region | n P | P B4 -> B5 | n S | S B4 -> B5 |", "|---|---|---|---|---|"]
    for k, r in dtr.iterrows():
        L.append(f"| {k} | {int(r[('n', 'P')]) if np.isfinite(r[('n', 'P')]) else 0} | {r[('rms_b4', 'P')]:.3f} -> "
                 f"{r[('rms_b5', 'P')]:.3f} | {int(r[('n', 'S')]) if np.isfinite(r[('n', 'S')]) else 0} | "
                 f"{r[('rms_b4', 'S')]:.3f} -> {r[('rms_b5', 'S')]:.3f} |")
    L.append(f"| all | {allr['P'][0]} | {allr['P'][1]:.3f} -> {allr['P'][2]:.3f} | {allr['S'][0]} | "
             f"{allr['S'][1]:.3f} -> {allr['S'][2]:.3f} |")
    ok = cs[cs.status == "ok"]
    L += ["", "## tomoDD dt rms (its own forward model, first -> last iteration, ms) and condition number", "",
          "| region | clusters | rmsct first med | rmsct last med | CND first med / max | CND last med / max |",
          "|---|---|---|---|---|---|"]
    for k, r in ok.groupby("region"):
        L.append(f"| {k} | {len(r)} | {r.rmsct_first_ms.median():.0f} | {r.rmsct_last_ms.median():.0f} | "
                 f"{r.cnd_first.median():.0f} / {r.cnd_first.max():.0f} | {r.cnd_last.median():.0f} / "
                 f"{r.cnd_last.max():.0f} |")
    pm = B["collect"]["pca_min_events"]
    p = cs[cs.pca3_b4.notna()] if "pca3_b4" in cs else cs.iloc[:0]
    L += ["", f"## Cluster shape (PCA of relocated core events, clusters with >= {pm} events, km)", "",
          "sigma3 = thickness across the best-fitting plane (smallest axis).", "",
          "| region | clusters | sigma3 B4 med | sigma3 B5 med | B5/B4 med | thinner % | sigma1 B4 / B5 med |",
          "|---|---|---|---|---|---|---|"]
    for k, r in list(p.groupby("region")) + [("all", p)]:
        if len(r) == 0:
            continue
        L.append(f"| {k} | {len(r)} | {r.pca3_b4.median():.3f} | {r.pca3_b5.median():.3f} | "
                 f"{(r.pca3_b5 / r.pca3_b4).median():.2f} | {100 * (r.pca3_b5 < r.pca3_b4).mean():.0f} | "
                 f"{r.pca1_b4.median():.2f} / {r.pca1_b5.median():.2f} |")
    sh = np.hypot(ok.shift_e_km, ok.shift_n_km)
    L += ["", "## Centroid shift removed (tomoDD drift of each cluster's relocated core events, D-B053)", "",
          f"Horizontal median {sh.median():.3f} km (p90 {sh.quantile(0.9):.3f}, max {sh.max():.3f}); vertical median "
          f"{ok.shift_z_km.median():+.3f} km (abs p90 {ok.shift_z_km.abs().quantile(0.9):.3f}, max abs "
          f"{ok.shift_z_km.abs().max():.3f}); origin time median {ok.shift_t_s.median():+.4f} s. Weighted by events: "
          f"horizontal {np.average(sh, weights=ok.relocated_core):.3f} km, vertical "
          f"{np.average(ok.shift_z_km.abs(), weights=ok.relocated_core):.3f} km.",
          f"Frame centre: every start position >= {B['frame_clear_km'] * 1e3:.0f} m from it (min "
          f"{cs.min_dist_centre_km.min() * 1e3:.0f} m); final relocated positions min "
          f"{cs.min_dist_centre_final_km.min() * 1e3:.0f} m."]
    if split_eff:
        L += ["", "## Split effect (events relocated as core in one chunk and as halo in another)", "",
              f"{split_eff['n']} duplicates: horizontal difference median {split_eff['dh_med']:.3f} km (p90 "
              f"{split_eff['dh_p90']:.3f}), vertical {split_eff['dz_med']:.3f} km (p90 {split_eff['dz_p90']:.3f}). "
              "By hop: " + "; ".join(f"hop {h}: n {v['n']}, dh {v['dh_med']:.3f}, dz {v['dz_med']:.3f}"
                                     for h, v in split_eff["by_hop"].items())]
    # D-B055
    a = pd.crosstab(cat.region, [cat.loc_source, cat.above_surface])
    L += ["", "## Above the solid surface (D-B055)", "",
          "surface_km = GMT earth_relief 01m (gridline, bilinear) at the final epicentre, above_surface = depth < "
          "-surface_km on the true depth (b4_catalog.py). Nothing is moved or dropped.", "",
          "| region | B5 relocated above | B4 kept above | tier U above | B5 events above at their B4 position |",
          "|---|---|---|---|---|"]
    for k, r in list(cat.groupby("region")) + [("all", cat)]:
        L.append(f"| {k} | {int((r.above_surface & (r.loc_source == 'b5_tomodd')).sum())} | "
                 f"{int((r.above_surface & (r.loc_source == 'nll3d')).sum())} | "
                 f"{int((r.above_surface & (r.loc_source == 'v4')).sum())} | "
                 f"{int((r.b4_above_surface & (r.loc_source == 'b5_tomodd')).sum())} |")
    na = int(cat.above_surface.sum())
    nb5 = int((cat.above_surface & rel).sum())
    L += ["", f"Total above the surface: {na} ({nb5} B5-relocated, "
          f"{int((cat.above_surface & (cat.loc_source == 'nll3d')).sum())} B4 kept, "
          f"{int((cat.above_surface & (cat.loc_source == 'v4')).sum())} tier U). "
          + ("Publishable under D-B055." if na == 0 else "NOT PUBLISHABLE under D-B055 (computation errors to find, "
                                                           "not to move).")]
    if nb5:
        x = cat[cat.above_surface & rel]
        hgt = -x.surface_km - x.depth
        L.append(f"B5 events above: height above the surface median {hgt.median() * 1e3:.0f} m, max {hgt.max() * 1e3:.0f} m;"
                 f" their B4 depth was below the surface for {int((x.b4_depth >= -x.surface_km).sum())} of them.")
    tw = cs.wall_s.sum() / 3600
    L += ["", "## Run", "", f"{len(cs)} clusters ({int((cs.part >= 0).sum())} chunks of split components), "
          f"binaries {cs.cls.value_counts().to_dict()}; status {cs.status.value_counts().to_dict()}; tomoDD "
          f"wall time total {tw:.1f} h (largest cluster {cs.wall_s.max() / 60:.1f} min, peak RSS "
          f"{cs.rss_gb.max():.2f} GB)."]
    (od / f"compare{tag}.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--version", required=True)
    ap.add_argument("--run")
    ap.add_argument("--b4-run")
    ap.add_argument("--iter0")
    ap.add_argument("--time")
    ap.add_argument("--nproc", type=int, default=B["nproc"])
    ap.add_argument("--only", help="comma-separated cluster ids (run stage, tests)")
    ap.add_argument("--nboot", type=int, default=B["nboot"])
    ap.add_argument("--partial", action="store_true", help="collect: allow clusters without a run")
    ap.add_argument("--reverse", action="store_true", help="run: smallest clusters first (a second, helper driver)")
    a = ap.parse_args()
    C = config(a)
    log(f"stage {a.stage}: {C['run']} (model v{C['version']}, B4 {C['b4_run']}, grids {C['time_grids']})")
    if a.stage == "prep":
        stage_prep(C)
    elif a.stage == "split":
        stage_split(C)
    elif a.stage == "run":
        stage_run(C, a.nproc, set(a.only.split(",")) if a.only else None, a.nboot, a.reverse)
    else:
        stage_collect(C, min(a.nproc, 32), a.partial)


if __name__ == "__main__":
    main()
