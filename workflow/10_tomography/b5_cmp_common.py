"""D-B078 comparison of B5 codes (tomoDD + D-B081 floor, GrowClust3D on the NonLinLoc grids, NLL-SSST-coherence):
common data, station folds, output layout and scoring, so that the three legs are judged on the same footing.

Data (the same for every leg):
* events of a test set: /wd1/mdenolle_data/tomo_v1/b5/cc_test/events_<set>.csv (catalog v5.1, b5_cc_sets.py);
* catalog picks of those events: the re-timed v4.1t pick file (D-B072/D-B075), as the B5 prep keeps them (prep of
  run b5_v5.1: picks.parquet, stations.csv);
* waveform cross-correlation differential times: /wd1/mdenolle_data/tomo_v1/cc_dt/cc_<set>.parquet (other session,
  D-B040), rows with status ok and above threshold, minus the triangle-closure cut (n_tri >= 2 and n_tri_fail20 >
  n_tri / 2), minus rows whose pair straddles a station clock epoch (column straddles_clock_epoch, when present).
  Given and added pairs. DT = (t1 - T0_1) - (t2 - T0_2) with v5.1 origin times.

Folds: the stations with picks in a set, in a seeded random order, dealt into `FOLDS` groups. Run `fold<k>` relocates
without the stations of group k (no picks, no catalog dt, no CC dt at them); `full` uses every station.

Output of every leg: <CMP>/<leg>/<set>/<run>/reloc.csv with columns ID, lat, lon, depth (km bsl, true depth) for each
relocated event, and run.json (code, version, parameters, wall time, notes).

Scores (score_set): for each fold, the CC dt rows at its held-out stations are predicted from the fold's relocated
positions in the NonLinLoc v5.1 grids (time_b4_v0.3.1m_sc21: Lambert x, y and Earth-flattened depth), the residual of
each row is taken relative to the median residual of the same pair at the training stations (which removes the pair's
origin-time and any common shift), and the held-out rms / MAD are pooled over the folds per phase; the same is done for
the v5.1 positions (no relocation) as the reference. Full runs: events above the merged solid surface (D-B055, as
b4_catalog.surface_km), cluster thickness (smallest principal axis, km), median shift from v5.1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_fwd_common import eft_depth, lambert_km  # noqa: E402

SETS = ("axial", "blanco", "msh", "jdf")
FOLDS = 3
SEED = 78
B5 = Path("/wd1/mdenolle_data/tomo_v1/b5")
TESTS = B5 / "cc_test"
CC = Path("/wd1/mdenolle_data/tomo_v1/cc_dt")
PREP = B5 / "b5_v5.1" / "prep"
GRIDS = "/wd1/mdenolle_data/reloc_v5/time_b4_v0.3.1m_sc21"
CMP = B5 / "cmp"
KM = 111.19


def events(s):
    return pd.read_csv(TESTS / f"events_{s}.csv")


def picks(s):
    """Catalog picks of the set's events as the B5 prep keeps them: ID, station, phase, tt (s after the v5.1 origin),
    pick_prob, sigma, w (D-B008 weight)."""
    ev = pd.read_parquet(PREP / "events.parquet", columns=["ID"])
    st = pd.read_csv(PREP / "stations.csv")
    pk = pd.read_parquet(PREP / "picks.parquet", columns=["ev", "sta", "phase", "tt", "pick_prob", "sigma", "w"])
    pk["ID"] = ev.ID.values[pk.ev.values]
    pk["station"] = st.station.values[pk.sta.values]
    ids = set(events(s).ID)
    return pk[pk.ID.isin(ids)][["ID", "station", "phase", "tt", "pick_prob", "sigma", "w"]].reset_index(drop=True)


def cc_rows(s, unretimable=True):
    """Accepted CC dt rows of a set after the common cuts: ID1, ID2, station, phase, dt_cc, cc_max, pair_source."""
    c = pd.read_parquet(CC / f"cc_{s}.parquet")
    keep = (c.status == "ok") & c.above.astype(bool)
    tri = (c.n_tri >= 2) & (c.n_tri_fail20 > c.n_tri / 2)
    keep &= ~tri
    if "straddles_clock_epoch" in c:
        keep &= ~c.straddles_clock_epoch.astype(bool)
    if not unretimable:
        keep &= ~c.unretimable_any.astype(bool)
    return c[keep][["ID1", "ID2", "station", "phase", "dt_cc", "cc_max", "pair_source"]].reset_index(drop=True)


def folds(s):
    """{station: fold} for the stations with picks in the set (seeded, reproducible)."""
    sta = np.array(sorted(picks(s).station.unique()))
    rng = np.random.default_rng(SEED + SETS.index(s))
    order = rng.permutation(len(sta))
    return {sta[i]: k % FOLDS for k, i in enumerate(order)}


def held_out(s, run):
    """Stations left out of a run ('full': none)."""
    if run == "full":
        return set()
    k = int(run.replace("fold", ""))
    return {st for st, f in folds(s).items() if f == k}


def runs():
    return ["full"] + [f"fold{k}" for k in range(FOLDS)]


def out_dir(leg, s, run):
    d = CMP / leg / s / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_reloc(leg, s, run):
    f = CMP / leg / s / run / "reloc.csv"
    return pd.read_csv(f).set_index("ID") if f.exists() else None


# ------------------------------------------------------------------------------------------------ scoring
_G = {}


def grid_times(st, ph, lat, lon, depth):
    if "G" not in _G:
        import b5_relocate as BR
        _G["G"] = BR.Grids(GRIDS)
    x, y = lambert_km(np.asarray(lon, float), np.asarray(lat, float))
    return _G["G"].time(ph, st, x, y, eft_depth(np.asarray(depth, float)))


def residuals(cc, pos):
    """Residual (s) of CC dt rows for the positions pos (index ID; lat, lon, depth): observed - predicted."""
    r = cc[cc.ID1.isin(pos.index) & cc.ID2.isin(pos.index)].copy()
    pred = np.full(len(r), np.nan)
    for (st, ph), g in r.groupby(["station", "phase"]):
        p1, p2 = pos.loc[g.ID1], pos.loc[g.ID2]
        t1 = grid_times(st, ph, p1.lat.values, p1.lon.values, p1.depth.values)
        t2 = grid_times(st, ph, p2.lat.values, p2.lon.values, p2.depth.values)
        pred[r.index.get_indexer(g.index)] = t1 - t2
    r["res"] = r.dt_cc.values - pred
    return r


def fold_score(cc, pos, out_sta):
    """Held-out residuals of one fold: residual minus the pair's median training residual."""
    r = residuals(cc, pos).dropna(subset=["res"])
    ho = r.station.isin(out_sta)
    med = r[~ho].groupby(["ID1", "ID2"]).res.median()
    h = r[ho].copy()
    h["ref"] = med.reindex(pd.MultiIndex.from_frame(h[["ID1", "ID2"]])).values
    return h.dropna(subset=["ref"]).assign(score=lambda x: x.res - x.ref)


def score_set(leg, s, unretimable=True):
    """Scores of a leg on a set (and of the v5.1 reference): held-out CC rms / MAD per phase pooled over folds; full-run
    surface, thickness and shift statistics."""
    cc = cc_rows(s, unretimable)
    ev = events(s).set_index("ID")
    out = {}
    for tag in ("v5.1", leg):
        rows = []
        for run in runs()[1:]:
            pos = ev if tag == "v5.1" else read_reloc(leg, s, run)
            if pos is None:
                continue
            rows.append(fold_score(cc, pos, held_out(s, run)))
        if not rows:
            continue
        H = pd.concat(rows)
        d = {"folds": len(rows), "relocated_per_fold": None if tag == "v5.1" else
             [int(len(read_reloc(leg, s, r))) for r in runs()[1:] if read_reloc(leg, s, r) is not None]}
        for ph, g in H.groupby("phase"):
            v = g.score.values
            d[ph] = dict(n=len(v), rms_ms=round(float(np.sqrt(np.mean(v ** 2))) * 1e3, 2),
                         mad_ms=round(float(np.median(np.abs(v - np.median(v)))) * 1e3, 2),
                         p90_ms=round(float(np.percentile(np.abs(v), 90)) * 1e3, 2))
        out[tag] = d
    full = read_reloc(leg, s, "full")
    if full is not None:
        from b4_catalog import surface_km
        f = full.join(ev[["lat", "lon", "depth"]], rsuffix="_v51")
        sk = surface_km(f.lat.values, f.lon.values)
        X = np.c_[(f.lon - f.lon.mean()) * KM * np.cos(np.radians(f.lat.mean())), (f.lat - f.lat.mean()) * KM, f.depth]
        X0 = np.c_[(f.lon_v51 - f.lon_v51.mean()) * KM * np.cos(np.radians(f.lat_v51.mean())),
                   (f.lat_v51 - f.lat_v51.mean()) * KM, f.depth_v51]
        thick = lambda A: float(np.sqrt(np.linalg.eigvalsh(np.cov((A - A.mean(0)).T))[0]))
        dh = np.hypot((f.lat - f.lat_v51) * KM, (f.lon - f.lon_v51) * KM * np.cos(np.radians(f.lat_v51)))
        out["full"] = dict(relocated=len(f), of=len(ev), above_surface=int((f.depth < -sk).sum()),
                           thickness_km=round(thick(X), 3), thickness_v51_km=round(thick(X0), 3),
                           shift_h_med_km=round(float(dh.median()), 3),
                           shift_z_med_km=round(float((f.depth - f.depth_v51).median()), 3))
    return out


if __name__ == "__main__":
    for s in SETS:
        f = folds(s)
        c = cc_rows(s)
        print(s, "stations", len(f), "per fold", np.bincount(list(f.values())).tolist(), "cc rows", len(c),
              {k: int(v) for k, v in c.phase.value_counts().items()}, "picks", len(picks(s)))
        print("  v5.1 reference:", json.dumps(score_set("none", s).get("v5.1")))
