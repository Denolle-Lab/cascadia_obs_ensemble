"""Nearest-neighbour clustering of catalogs v5.1 and v6.0 (Zaliapin & Ben-Zion method).

The method, parameters and classification rules are those of v4_clusters_zbz.py (catalog v4; read its
docstring for the method); the functions are imported from it so the runs stay comparable. Written for v5.1
(below); catalog v6.0 runs the same code with --catalog v6.0 (see "Catalog version"). What is new for v5.1:

Events
* Catalog v5.1 (NonLinLoc in model v0.3.1) and the v5.1 Route A magnitudes (column M): by default the
  converged table of D-B091 (cascadia_catalog_v5.1_magnitudes_converged.csv) and its Mc / b table; --mag stored
  gives the piece "mag" table (cascadia_catalog_v5.1_magnitudes.csv); see workflow/05_magnitude/mag_paths.py.
* Left out: rows without a hypocentre (loc_source none, 623) and the 3,313 events flagged as not
  earthquakes (is_earthquake False: ComCat explosions / quarry blasts / other, Texada quarry_blast_likely).
  They are kept in the event table with role "excluded" and the reason.
* floor_limited, far_only and deep_pass events enter through their epicentres: the main run uses
  epicentral (2-D) distance, as in v4. Their depths are not used in the main run or in any depth
  statistic: depth ranges, depth migration and the 3-D variant d3q use only events with a well-constrained
  depth (depth_good: tier A-C, depth_quality ok, not floor_limited / far_only / deep_pass). Exceptions,
  labelled as such: variant d3 (3-D distance with every depth, a sensitivity test only), the family column
  depth_main_any_quality (depth of the largest event, any quality; see depth_main_good) and the DBSCAN
  columns depth_any_quality_med / _iqr.
* Tier D events are in the main analysis set, as all events were in v4; variant tierABC leaves them out of
  it (they are then attached afterwards, like events below Mc). Tier D epicentres are poor outside Axial:
  the median 68 % horizontal semi-axis (h_max) of the tier D events above Mc is 4.8 km at Axial but 9 km at
  the Mendocino triple junction and 23-45 km elsewhere (GO 23, BL 28, ON 32, OR 34, EN 36, PU 38, WA 45);
  the five-phase events (one degree of freedom) have a median of 42 km. 16 % of the links of the main run
  have a tier D end (48 % in ON). Step checks writes these numbers (tierD_by_region.csv).
* Regions: the nine boxes of v4 applied to the v5.1 epicentres; Mc per box from the v5.1 magnitudes with the
  v4 rule (maximum curvature + 0.2, largest yearly value over years with >= 200 events). The rule bins M
  with np.arange(-2, 8, 0.1) edges, as in v4; with exact decimal edges BL and OR change by 0.1 (step checks,
  mc_edge_check.csv). Kept for comparability with v4.
* Inputs of the other v5.1 pieces (fault surfaces v1, the mag piece's Mc_best) are read from their pushed
  branches (FAULTS_REF, MAG_REF; commits written to input_commits.json), not from the other worktrees.

Per family, in addition to the v4 fields: magnitude range of all members, duration of all members, depth
range of the well-constrained depths, migration from the tier A-C members (radial speed, Theil-Sen velocity
vector and azimuth, depth migration) and a sqrt(t) diffusion fit (Shapiro et al. 1997: r = sqrt(4 pi D t);
D90 is the diffusivity of the front that encloses 90 % of the events, and the exponent p of the 90 %
distance envelope against time in log bins tells sqrt(t)-like (p near 0.5) from linear (p near 1)
migration). The migration test uses a permutation p-value (mig_p_perm): the origin is the centroid of the
first 10 % of the same events, which biases the Spearman p-value (mig_p, kept for reference) toward
migration.

Variants (sensitivity): d3 (3-D distance, all depths), d3q (3-D only between two depth_good events), mc+0.3,
mc_best (Mc_best of the mag piece), df_est, df1.3, df2.0, noM_floor, mc_year, eta0_shuf5, tierABC, v41M
(v4.1 magnitudes on the v5.1 locations: separates the location change from the magnitude change).

Steps (each caches its output; rerun skips what exists unless --force):
    main      main run, completeness table, events and families
    shuffle   time-shuffled catalogs and eta0_shuf5 thresholds
    variants  sensitivity variants (one file each on /wd1)
    faults    join with the fault surfaces v1 (piece "faults"), family flags (daytime-screened surfaces,
              waveform-review sets P1 Mount St Helens deep events and P2 Brookings 2012)
    nulls     null test of the links: time-shuffled catalogs (within region, region-year, region-month) at
              the real thresholds; linked fraction, families and swarms against the real catalog
    checks    tier D by region, Mc bin-edge check, swarm/mainshock-aftershock depth with families as the
              unit, waveform-review sets in the clustering
    compare   comparison with the v4 results, known sequences, volcano zoom, setting statistics

Catalog version (--catalog; configure() sets inputs, folders and file tags):
    v5.1  --mag stored reproduces the committed v5.1 run (outputs in data/catalog_v5/v5.1_analysis/clusters/ and
          /wd1/.../v5.1_analysis/clusters/, compared with the v4 run of the stored magnitudes). Any other --mag writes
          to folders with the suffix _Mconv (or --out / --big) and is compared with the v4.1 run on main (converged).
    v6.0  catalog v6.0 with its own magnitude column (converged, D-B091); not earthquakes = event_flag set; every
          output in /wd1/mdenolle_data/catalog_v6/v6.0_analysis/clusters/ (events_zbz_v60.csv.gz,
          families_zbz_v60.csv, ...); variant storedM replaces v41M; mc_best from the converged Mc map table
          (MCB_V60); compared with v5.1 (stored M), v5.1 locations with converged M (ref_v51_Mconv/, run first with
          --catalog v5.1 --mag converged --out/--big <v6.0 folder>/ref_v51_Mconv) and v4.1 (converged M). Step repo
          copies the small outputs to data/catalog_v6/v6.0_analysis/clusters/.

Run (from this folder):
    python v51_clusters_zbz.py --catalog v5.1 --mag stored --steps main,shuffle,variants,faults,nulls,checks,compare
    python v51_clusters_zbz.py --catalog v6.0 --steps main,shuffle,variants,faults,nulls,checks,compare,repo
Later catalogs (end-to-end pipeline, D-B110): --catalog v6.1 runs the v6.0 recipe on catalog v6.1 (magnitudes and
event_flag of the catalog; analog_test_signal is a flag, so those events are not earthquakes here), compared with v6.0
(primary) and v4.1 (converged M). --mcb names the Mc map table of the catalog's own hypocentres (v4_mc_maps.py
--catalog, mc_regional.csv) for variant mc_best, and --faults a folder holding the fault surfaces v1 of the catalog
(fault_surfaces_v1_assignment.csv.gz, fault_surfaces_v1_summary.csv) for step faults, instead of the v5.1 surfaces
read from git. Without these options v5.1 and v6.0 run exactly as before.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu, rankdata, spearmanr, theilslopes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "05_magnitude"))
import mag_paths  # noqa: E402  (converged Route A magnitudes by default, D-B091)
from v4_clusters_zbz import (REGIONS, REGION_NAMES, REG_ORDER, RE, Q, mc_maxc, b_aki, xyz,  # noqa: E402
                             gmm1d, tree_metrics, classify, rupture_cap, migration, completeness_table)

ROOT = HERE.parents[1]
WD1 = Path("/wd1/mdenolle_data")
# The catalog version (--catalog) sets the inputs, the output folders and the file tag; configure() sets the globals
# below. 'v5.1' with --mag stored reproduces the committed v5.1 results (PR #42); 'v6.0' is catalog v6.0.
CATALOGS = {
    "v5.1": dict(tag="v51", csv=WD1 / "catalog_v5/v5.1/cascadia_catalog_v5.1.csv",
                 out="data/catalog_v5/v5.1_analysis/clusters", big=WD1 / "catalog_v5/v5.1_analysis/clusters"),
    "v6.0": dict(tag="v60", csv=WD1 / "catalog_v6/v6.0/cascadia_catalog_v6.0.csv",
                 out=None, big=WD1 / "catalog_v6/v6.0_analysis/clusters"),
    # pipeline (D-B110): the v6.0 recipe, compared with v6.0 and v4.1; Mc_best from its own Mc maps (D-B104)
    "v6.1": dict(tag="v61", csv=WD1 / "catalog_v6/v6.1/cascadia_catalog_v6.1.csv",
                 out=None, big=WD1 / "catalog_v6/v6.1_analysis/clusters",
                 mcb=WD1 / "magnitude_converged/mc_maps_v61/mc_regional.csv", refs=["v6.0", "v4"]),
}
CATV = "v5.1"
TAG = "v51"
CAT = CATALOGS["v5.1"]["csv"]
MAG = mag_paths.v51()        # v5.1: --mag / $CASCADIA_MAG_V51: converged (default), stored or a path
# v6.0: the catalog's own M column (converged Route A magnitudes of D-B091, magnitude_source), not recomputed; variant
# storedM puts the stored (pre-D-B091) magnitudes on the v6.0 locations: the v5.1 stored table, else the v4.1 stored
# table, as v6.0 takes the v5.1 table, else the v4.1 table
MAG_V51_STORED = mag_paths.v51("stored")
MAG_V41_STORED = mag_paths.v41("stored")
# products of the other v5.1 pieces, read through git (git objects are shared by all worktrees of the repository)
MAG_REF = "origin/main"      # the v5.1 Mc / b tables (stored and converged) are on main (PRs #40, #45)
MCB = mag_paths.mcb_v51()
# Mc_best for variant mc_best of v6.0: the converged regional Mc map table (v4_mc_maps.py on the converged v4.1
# magnitudes, v4 locations; rows variant 'all', period 'all')
MCB_V60 = WD1 / "magnitude_converged/mc_maps_v41/mc_regional.csv"
FAULTS_REF = "origin/main"   # fault surfaces v1 (PR #41; same blobs as origin/analysis/v51-faults 6ebe0230)
FAULTS = "data/catalog_v5/v5.1_analysis/faults"
FAULTS_DIR = None            # --faults: a folder with the catalog's own fault surfaces v1 (read instead of git)
# waveform review of four v5.1 location problems (P1 Mount St Helens deep events, P2 Brookings 2012,
# P3 tensor events, P4 deep_pass / far_only), review/v51-locations
REVIEW = WD1 / "catalog_v5/v5.1_review/review_v51_events.csv"
# clustering runs to compare with (step compare): key -> folder, events file, families file, label. The first one is
# the primary comparison (file names without '_vs_<key>'). The v4 clustering of the stored magnitudes is the frozen
# copy of origin/main 5d6658bf (data/catalog_v4/clusters_zbz has held the converged run since PR #47).
V4_STORED = WD1 / "magnitude_converged/paper/stored/data/catalog_v4"
V4_CONV = ROOT / "data/catalog_v4"
REFS: dict = {}
OUT = ROOT / CATALOGS["v5.1"]["out"]
BIG = CATALOGS["v5.1"]["big"]
EVF, FAMF = "events_zbz_v51.csv.gz", "families_zbz_v51.csv"
T0 = pd.Timestamp("2010-01-01", tz="UTC")
YR = 365.25
KM = 111.195
VARIANTS_V51 = ["d3", "d3q", "mc+0.3", "mc_best", "df_est", "df1.3", "df2.0", "noM_floor", "mc_year",
                "eta0_shuf5", "tierABC", "v41M"]
VARIANTS = list(VARIANTS_V51)
VARIANTS_V60 = ["d3", "d3q", "mc+0.3", "mc_best", "df_est", "df1.3", "df2.0", "noM_floor", "mc_year",
                "eta0_shuf5", "tierABC", "storedM"]


def configure(catalog: str = "v5.1", mag: str | None = None, out: str | None = None, big: str | None = None,
              mcb: str | None = None, faults: str | None = None) -> None:
    """Set the catalog version, magnitudes, output folders and file names (module globals).

    v5.1: magnitudes from --mag (mag_paths: converged by default, 'stored' = the committed v5.1 results). The outputs
    of the stored magnitudes go to the committed folders; any other magnitude choice goes to folders with the suffix
    '_Mconv' (or to --out / --big), so a converged run never overwrites the committed stored-magnitude results.
    v6.0: magnitudes from the catalog (--mag is refused); every output in one folder on /wd1 (OUT = BIG) so the atlas
    finds the events, families and sensitivity tables together."""
    global CATV, TAG, CAT, MAG, MCB, OUT, BIG, EVF, FAMF, REFS, VARIANTS, FAULTS_DIR
    cfg = CATALOGS[catalog]
    FAULTS_DIR = Path(faults) if faults else None
    CATV, TAG, CAT = catalog, cfg["tag"], cfg["csv"]
    EVF, FAMF = f"events_zbz_{TAG}.csv.gz", f"families_zbz_{TAG}.csv"
    if catalog == "v5.1":
        MAG, MCB = mag_paths.v51(mag), mag_paths.mcb_v51(mag)
        stored = Path(MAG) == mag_paths.v51("stored")
        sfx = "" if stored else "_Mconv"
        OUT = Path(out) if out else ROOT / (cfg["out"] + sfx)
        BIG = Path(big) if big else Path(str(cfg["big"]) + sfx)
        v4 = V4_STORED if stored else V4_CONV
        REFS = {"v4": dict(dir=v4 / "clusters_zbz", events="events_zbz.csv.gz", families="families_zbz.csv",
                           swarms=v4 / "swarms", label="v4 (" + ("stored" if stored else "converged") + " M)")}
        VARIANTS = [v for v in VARIANTS_V51 if stored or v != "v41M"]
    else:
        if mag:
            raise SystemExit("--mag: catalog v6.0 carries its magnitudes (converged, D-B091); use variant storedM")
        MAG, MCB = CAT, str(mcb or cfg.get("mcb", MCB_V60))
        BIG = Path(big) if big else cfg["big"]
        OUT = Path(out) if out else BIG
        v51 = CATALOGS["v5.1"]
        if "refs" in cfg:        # later catalogs (v6.1): the earlier v6.x runs (events and DBSCAN scan in one folder), v4.1
            REFS = {}
            for r in cfg["refs"]:
                if r == "v4":
                    REFS["v4"] = dict(dir=V4_CONV / "clusters_zbz", events="events_zbz.csv.gz",
                                      families="families_zbz.csv", swarms=V4_CONV / "swarms", label="v4.1 (converged M)")
                else:
                    R = CATALOGS[r]
                    REFS[R["tag"]] = dict(dir=R["big"], big=R["big"], events=f"events_zbz_{R['tag']}.csv.gz",
                                          families=f"families_zbz_{R['tag']}.csv", swarms=R["big"],
                                          swarms_tag=R["tag"], label=f"{r} (catalog M)")
            VARIANTS = list(VARIANTS_V60)
            return
        REFS = {
            "v51": dict(dir=ROOT / v51["out"], big=v51["big"], events="events_zbz_v51.csv.gz",
                        families="families_zbz_v51.csv", swarms=ROOT / v51["out"], label="v5.1 (stored M)"),
            "v51Mc": dict(dir=BIG / "ref_v51_Mconv", big=BIG / "ref_v51_Mconv", events="events_zbz_v51.csv.gz",
                          families="families_zbz_v51.csv", label="v5.1 locations, converged M"),
            "v4": dict(dir=V4_CONV / "clusters_zbz", events="events_zbz.csv.gz", families="families_zbz.csv",
                       swarms=V4_CONV / "swarms", label="v4.1 (converged M)"),
        }
        VARIANTS = list(VARIANTS_V60)
# known events for validation (catalog v4 event IDs = v5.1 IDs; ComCat magnitudes); list of v4_clusters_zbz_report.py
KNOWN = [
    (262, "2010-01-10 Mw 6.5 Gorda", "mainshock-aftershock"),
    (83695, "2014-03-10 Mw 6.8 Gorda", "mainshock-aftershock"),
    (19960, "2011-09-09 Mw 6.4 Vancouver Island", "mainshock-aftershock"),
    (89617, "2014-04-24 Mw 6.5 Vancouver Island", "mainshock-aftershock"),
    (40500, "2012-11-08 Mw 6.1 Explorer plate", "mainshock-aftershock"),
    (117222, "2015-01-28 Mw 5.7 Mendocino fault zone", "mainshock-aftershock"),
    (150165, "2015-06-01 Mw 5.8, 5.4, 5.9 Blanco triplet", "swarm or multiplet"),
    (27442, "2012-02/04 Mw 5.8 and 6.0 Blanco", "swarm or multiplet"),
    (900001, "2016-12-08 Mw 6.6 Mendocino fracture zone (v4.1 addition)", "mainshock-aftershock"),
]
VOLCANOES = {"Mount Baker": (-121.813, 48.777), "Glacier Peak": (-121.113, 48.112),
             "Mount Rainier": (-121.760, 46.853), "Mount St. Helens": (-122.180, 46.200),
             "Mount Adams": (-121.490, 46.206), "Mount Hood": (-121.695, 45.374),
             "Mount Jefferson": (-121.800, 44.674), "Three Sisters": (-121.768, 44.103),
             "Newberry": (-121.229, 43.722), "Crater Lake": (-122.110, 42.940),
             "Medicine Lake": (-121.550, 41.590), "Mount Shasta": (-122.193, 41.409),
             "Lassen Peak": (-121.505, 40.492)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def git_csv(ref: str, path: str, **kw) -> pd.DataFrame:
    """Read a CSV committed on branch `ref` and record the commit in input_commits.json."""
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", ref], check=True, capture_output=True,
                         text=True).stdout.strip()
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{path}"], check=True,
                          capture_output=True).stdout
    f = OUT / "input_commits.json"
    try:
        rec = json.loads(f.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        rec = {}
    rec[path] = dict(ref=ref, commit=sha)
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return pd.read_csv(io.BytesIO(blob), compression="gzip" if path.endswith(".gz") else None, **kw)


# ---------------------------------------------------------------- loading
def load_all() -> pd.DataFrame:
    """All 172,342 catalog rows with the magnitudes and the exclusion reason ('' = used).

    The catalog column 'region' (box label of the v4 epicentre; identical in catalogs v5.1 and v6.0) is kept as
    region_v51 in both versions, so the output tables have one schema (the atlas reads region_v51)."""
    cols = ["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region", "h_min", "h_max", "sig_z",
            "depth_quality", "event_flag", "floor_limited", "far_only", "deep_pass", "five_phase", "M", "n_sta",
            "gap"]
    if CATV == "v5.1":
        c = pd.read_csv(CAT, usecols=cols, low_memory=False).rename(
            columns={"ID": "event_id", "region": "region_v51", "M": "M_v41"})
        m = pd.read_csv(MAG, usecols=["ID", "M", "M_type", "is_earthquake"]).rename(columns={"ID": "event_id"})
        c = c.merge(m, on="event_id", how="left", validate="1:1")
        c["is_earthquake"] = _bool(c["is_earthquake"])
    else:
        c = pd.read_csv(CAT, usecols=cols + ["M_type", "magnitude_source"], low_memory=False).rename(
            columns={"ID": "event_id", "region": "region_v51"})
        # not an earthquake: the catalog's own event_flag (ComCat explosion / quarry blast / other within 3 s and
        # 20 km, Texada quarry_blast_likely; D-B071), evaluated on the v6.0 hypocentres
        c["is_earthquake"] = c.event_flag.fillna("").astype(str).str.strip().eq("")
        # variant storedM: stored magnitudes on the v6.0 locations (v5.1 stored table, else v4.1 stored table)
        s51 = pd.read_csv(MAG_V51_STORED, usecols=["ID", "M"]).set_index("ID").M
        s41 = pd.read_csv(MAG_V41_STORED, usecols=["event_id", "M"]).drop_duplicates("event_id").set_index(
            "event_id").M
        src51 = c.magnitude_source.fillna("").str.startswith("v5.1")
        c["M_stored"] = np.where(src51, c.event_id.map(s51), c.event_id.map(s41))
        c["M_stored"] = np.where(c.M.notna(), c.M_stored, np.nan)   # the same events carry M as in the catalog
    for f in ["floor_limited", "far_only", "deep_pass", "five_phase"]:
        c[f] = _bool(c[f])
    c["excluded"] = np.where(c.loc_source.eq("none"), "no_hypocentre",
                             np.where(~c.is_earthquake, "not_earthquake", ""))
    return c


def box_of(lon, lat) -> np.ndarray:
    reg = np.full(len(lon), "ON", dtype=object)
    for key, _, (a, b, c, d) in REGIONS[::-1]:
        s = (lon >= a) & (lon <= b) & (lat >= c) & (lat <= d)
        reg[np.asarray(s)] = key
    return reg


def prepare(c: pd.DataFrame, mcol: str = "M") -> pd.DataFrame:
    df = c[c.excluded.eq("")].copy()
    df["M"] = df[mcol]
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["t_yr"] = (df["time"] - T0).dt.total_seconds() / (YR * 86400)
    df["year"] = df["time"].dt.year
    df["region"] = box_of(df.lon.values, df.lat.values)
    df["tierABC"] = df.tier.isin(["A", "B", "C"])
    df["depth_good"] = (df.tierABC & df.depth_quality.eq("ok") & ~df.floor_limited & ~df.far_only
                        & ~df.deep_pass)
    return df.sort_values("t_yr", kind="stable").reset_index(drop=True)


def completeness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctab, ytab = completeness_table(df)
    ctab["Mc_year"] = [ytab[ytab.region == r].set_index("year").Mc_maxc.to_dict() for r in ctab.region]
    # Mc_best for variant mc_best. v5.1: the mag piece's table (v5.1 magnitudes, v5.1 locations, whole period);
    # v6.0: the converged regional Mc map table (MCB_V60; no Mc map exists for the v6.0 locations)
    try:
        if CATV == "v5.1":
            mb = git_csv(MAG_REF, MCB)
            mb = mb[(mb.version == "C_v51M_v51loc") & (mb.period == "all")].set_index("region")
        else:
            mb = pd.read_csv(MCB)
            mb = mb[(mb.variant == "all") & (mb.period == "all")].set_index("region")
        ctab["Mc_best"] = ctab.region.map(mb.mc_best)
    except (subprocess.CalledProcessError, FileNotFoundError):
        ctab["Mc_best"] = np.nan
    bb = [b_aki(df.M.values[df.region.values == r], m)[0] if np.isfinite(m) else np.nan
          for r, m in zip(ctab.region, ctab.Mc_best)]
    ctab["b_at_Mc_best"] = bb
    return ctab, ytab


# ---------------------------------------------------------------- nearest neighbours
_G: dict = {}  # arrays shared with forked workers


def _nn_chunk(rng_):
    """Nearest earlier parent for children rng_[0]:rng_[1] (v4 _nn_chunk plus the depth-quality mask)."""
    a, b_ = rng_
    G = _G
    tc, uc, zc, dfc = G["tc"][a:b_], G["uc"][a:b_], G["zc"][a:b_], G["dfc"][a:b_]
    qc = G["qc"][a:b_] if G["qc"] is not None else None
    tp, up, zp, bmp, qp = G["tp"], G["up"], G["zp"], G["bmp"], G["qp"]
    prefix, rmin, d3 = G["prefix"], G["rmin"], G["d3"]
    regc, regp, capp = G["regc"][a:b_], G["regp"], G["capp"]
    nc = b_ - a
    par = np.full(nc, -1)
    out = np.full((nc, 5), np.nan)
    sub = 256
    for s0 in range(0, nc, sub):
        sl = slice(s0, min(s0 + sub, nc))
        npar = (a + sl.stop) if prefix else np.searchsorted(tp, tc[sl].max(), side="left")
        if npar == 0:
            continue
        dt = tc[sl, None] - tp[None, :npar]
        if prefix:
            bad = np.arange(npar)[None, :] >= np.arange(a + sl.start, a + sl.stop)[:, None]
        else:
            bad = dt <= 0
        dt = np.where(bad, 1.0, np.maximum(dt, 1e-9))
        ch = np.sqrt(((uc[sl, None, :] - up[None, :npar, :]) ** 2).sum(-1))
        r = 2 * RE * np.arcsin(np.clip(ch / 2, 0, 1))
        if d3:
            dz2 = (zc[sl, None] - zp[None, :npar]) ** 2
            if qc is not None:  # 3-D only between two well-constrained depths
                dz2 = np.where(qc[sl, None] & qp[None, :npar], dz2, 0.0)
            r = np.sqrt(r ** 2 + dz2)
        r = np.maximum(r, rmin)
        bad = bad | ((regc[sl, None] != regp[None, :npar]) & (r > capp[None, :npar]))
        dt = np.where(bad, 1.0, dt)
        lr_all = np.log10(r)
        le = np.log10(dt) + dfc[sl, None] * lr_all - bmp[None, :npar]
        le[bad] = np.inf
        k = np.argmin(le, axis=1)
        rows = np.arange(le.shape[0])
        lev = le[rows, k]
        ok = np.isfinite(lev)
        idx = np.arange(sl.start, sl.stop)[ok]
        kk = k[ok]
        par[idx] = kk
        lt = np.log10(dt[rows, k])[ok]
        lr = lr_all[rows, k][ok]
        out[idx, 0] = lev[ok]
        out[idx, 1] = lt - Q * bmp[kk]
        out[idx, 2] = dfc[sl][ok] * lr - (1 - Q) * bmp[kk]
        out[idx, 3] = r[rows, k][ok]
        out[idx, 4] = dt[rows, k][ok]
    return par, out


def nn_search(tc, uc, zc, dfc, tp, up, zp, bmp, rmin, d3, prefix, workers, regc, regp, capp,
              qc=None, qp=None, chunk=2000):
    """As v4_clusters_zbz.nn_search; qc/qp (bool) restrict the depth term of d3 to well-constrained pairs."""
    _G.update(regc=regc, regp=regp, capp=capp, tc=tc, uc=uc, zc=zc, dfc=dfc, tp=tp, up=up, zp=zp, bmp=bmp,
              prefix=prefix, rmin=rmin, d3=d3, qc=qc if qc is None else qc[:len(tc)], qp=qp)
    n = len(tc)
    rngs = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_nn_chunk, rngs))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


# ---------------------------------------------------------------- one run
def select(df, ctab, variant):
    c = ctab.set_index("region")
    mm = df.M.values.copy()
    mc = df.region.map(c.Mc).values.astype(float)
    if variant == "mc+0.3":
        mc = mc + 0.3
    if variant == "mc_best":
        mc = df.region.map(c.Mc_best).values.astype(float)
    if variant == "noM_floor":
        mm = np.where(np.isnan(mm), mc, mm)
    if variant == "mc_year":
        mcy = np.array([c.Mc_year[r].get(y, np.nan) for r, y in zip(df.region, df.year)])
        mc = np.where(np.isfinite(mcy), mcy, mc)
    sel = np.isfinite(mm) & (mm >= mc - 1e-9)
    if variant == "tierABC":
        sel &= df.tierABC.values
    if variant in ("df1.3", "df2.0"):
        dfe = np.full(len(df), float(variant[2:]))
    elif variant == "df_est":
        dfe = df.region.map(c.df_est_clip).values
    else:
        dfe = df.region.map(c.df).values
    be = df.region.map(c.b_at_Mc_best if variant == "mc_best" else c.b).values
    return sel, mm, mc, dfe, be


def build(df, ctab, variant, a, thr_override=None, shuffle_seed=None, null="region", null_families=False):
    """v4_clusters_zbz.build for v5.1: same links, families and classification; extra event columns.

    shuffle_seed: permute the event times within groups (null 'region': whole period, as v4; 'region_year',
    'region_month': within the same calendar year or month, which keeps slow rate changes). null_families: also
    form and classify the families of the shuffled catalog (returned as 'fam')."""
    sel, mm, mcv, dfe, be = select(df, ctab, variant)
    d3 = variant in ("d3", "d3q")
    A = df[sel].copy()
    A["m"] = mm[sel]
    A["dfe"], A["be"] = dfe[sel], be[sel]
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        key = {"region": A.region,
               "region_year": A.region + "_" + A.year.astype(str),
               "region_month": A.region + "_" + A.time.dt.strftime("%Y-%m")}[null]
        for _, g in A.groupby(key):
            A.loc[g.index, "t_yr"] = rng.permutation(g.t_yr.values)
        A = A.sort_values("t_yr", kind="stable")
    A = A.reset_index().rename(columns={"index": "row"})
    u = xyz(A.lat.values, A.lon.values)
    t, z, m = A.t_yr.values, A.depth.values, A.m.values
    q = A.depth_good.values if variant == "d3q" else None
    bm = A.be.values * m
    rc = A.region.map({r: k for k, r in enumerate(REG_ORDER)}).values
    capm = rupture_cap(m, a.rcap_min)
    par, o = nn_search(t, u, z, A.dfe.values, t, u, z, bm, a.rmin, d3, True, a.workers, rc, rc, capm, q, q)
    reg = A.region.values
    thr = {}
    for r in REG_ORDER:
        g = gmm1d(o[reg == r, 0])
        if thr_override is not None:
            g["log_eta0"] = thr_override[r]
        thr[r] = g
    le0 = np.array([thr[r]["log_eta0"] for r in reg])
    cap = rupture_cap(m[np.maximum(par, 0)], a.rcap_min)
    below = np.isfinite(o[:, 0]) & (o[:, 0] < le0)
    linked = below & (o[:, 3] <= cap)
    if shuffle_seed is not None and not null_families:
        return dict(log_eta=o[:, 0], log_T=o[:, 1], log_R=o[:, 2], region=reg, below=below, linked=linked)
    n = len(A)
    ii = np.where(linked)[0]
    _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, par[ii])), shape=(n, n)), directed=False)
    sizes = np.bincount(lab)
    fam_local = np.where(sizes[lab] >= 2, lab, -1)
    role = np.array(["single"] * n, dtype=object)
    fid = np.full(n, -1)
    fam_rows = []
    order = pd.Series(np.arange(n))[fam_local >= 0].groupby(fam_local[fam_local >= 0])
    starts = sorted(((g.values[0], c) for c, g in order), key=lambda x: x[0])
    groups = {c: g.values for c, g in order}
    for F, (_, c) in enumerate(starts, start=1):
        mem = groups[c]
        tm = tree_metrics(mem, par, linked, m)
        ms = np.sort(m[mem])[::-1]
        dM = float(ms[0] - ms[1])
        k = tm["imax_local"]
        frac_before = k / len(mem)
        typ, flags = classify(len(mem), dM, frac_before, tm)
        fid[mem] = F
        if typ in ("swarm", "multiplet"):
            role[mem] = typ
        else:
            role[mem[:k]] = "foreshock"
            role[mem[k + 1:]] = "aftershock"
            role[mem[k]] = "mainshock"
        la, lo = A.lat.values[mem], A.lon.values[mem]
        x = (lo - lo.mean()) * KM * np.cos(np.radians(la.mean()))
        y = (la - la.mean()) * KM
        dc = np.hypot(x, y)
        tday = (t[mem] - t[mem[0]]) * YR
        v, rho, p = migration(tday, x, y)
        regs = pd.Series(reg[mem]).value_counts()
        fam_rows.append(dict(
            family_id=F, region=reg[mem[k]], region_v51=A.region_v51.values[mem[k]],
            regions=";".join(f"{q_}:{w}" for q_, w in regs.items()),
            n=len(mem), Mmax=float(ms[0]), M2=float(ms[1]), dM=dM, Mmin=float(ms[-1]),
            mainshock_id=int(A.event_id.values[mem[k]]),
            start=A.time.values[mem[0]], end=A.time.values[mem[-1]], duration_days=float(tday[-1]),
            duration90_days=float(np.quantile(tday, 0.9)),
            lon=float(lo.mean()), lat=float(la.mean()), lon_main=float(lo[k]), lat_main=float(la[k]),
            depth_main_any_quality=float(A.depth.values[mem[k]]),
            depth_main_good=bool(A.depth_good.values[mem[k]]),
            extent_km=float(2 * np.quantile(dc, 0.95)), type=typ,
            S1_no_dominant=flags[0], S2_late_peak=flags[1], S3_chain=flags[2],
            frac_before_largest=frac_before, t_largest_days=float(tday[k]),
            mean_leaf_depth=tm["mean_leaf_depth"], max_depth=tm["max_depth"],
            main_offspring_frac=tm["main_offspring_frac"], mean_branching=tm["mean_branching"],
            migration_km_day=v, migration_rho=rho, migration_p=p,
            migrating=bool(np.isfinite(rho) and rho >= 0.5 and p < 0.01 and v > 0)))
    fam = pd.DataFrame(fam_rows)
    if shuffle_seed is not None:
        # event_id / family_id: family membership of the shuffled catalog (analysis set), for per-unit null shares
        # (workflow/11_paper_plan/q1_recluster.py); the other keys are unchanged
        return dict(log_eta=o[:, 0], log_T=o[:, 1], log_R=o[:, 2], region=reg, below=below, linked=linked,
                    fam=fam, n_in_family=pd.Series(reg[fid > 0]).value_counts(),
                    event_id=A.event_id.values, family_id=fid)
    ev = pd.DataFrame(dict(
        event_id=A.event_id.values, region=reg, in_analysis=True, M=m,
        parent_id=np.where(par >= 0, A.event_id.values[np.maximum(par, 0)], -1),
        log_eta=o[:, 0], log_T=o[:, 1], log_R=o[:, 2], nn_dist_km=o[:, 3], nn_dt_days=o[:, 4] * YR,
        linked=linked, family_id=fid, role=role))
    # ---- attach events below Mc / without M (and tier D in variant tierABC)
    O = df[~sel]
    if len(O):
        qo = O.depth_good.values if variant == "d3q" else None
        pp, oo = nn_search(O.t_yr.values, xyz(O.lat.values, O.lon.values), O.depth.values, dfe[~sel],
                           t, u, z, bm, a.rmin, d3, False, a.workers,
                           O.region.map({r: k for k, r in enumerate(REG_ORDER)}).values, rc, capm, qo, q)
        le0o = O.region.map({r: thr[r]["log_eta0"] for r in thr}).values
        capo = rupture_cap(m[np.maximum(pp, 0)], a.rcap_min)
        lk = np.isfinite(oo[:, 0]) & (oo[:, 0] < le0o) & (oo[:, 3] <= capo)
        pf = np.where(pp >= 0, fid[np.maximum(pp, 0)], -1)
        ftype = dict(zip(fam.family_id, fam.type)) if len(fam) else {}
        tmain = df.set_index("event_id").t_yr
        mt = dict(zip(fam.family_id, tmain.reindex(fam.mainshock_id).values)) if len(fam) else {}
        orole = np.where(~lk, "single_belowMc", "linked_to_single_belowMc").astype(object)
        to = O.t_yr.values
        for q_ in np.where(lk & (pf >= 0))[0]:
            ty = ftype[pf[q_]]
            if ty == "swarm":
                orole[q_] = "swarm_belowMc"
            elif ty == "multiplet":
                orole[q_] = "multiplet_belowMc"
            else:
                orole[q_] = "foreshock_belowMc" if to[q_] < mt[pf[q_]] else "aftershock_belowMc"
        ev2 = pd.DataFrame(dict(
            event_id=O.event_id.values, region=O.region.values, in_analysis=False, M=O.M.values,
            parent_id=np.where(pp >= 0, A.event_id.values[np.maximum(pp, 0)], -1),
            log_eta=oo[:, 0], log_T=oo[:, 1], log_R=oo[:, 2], nn_dist_km=oo[:, 3],
            nn_dt_days=oo[:, 4] * YR, linked=lk, family_id=np.where(lk, pf, -1), role=orole))
        ev = pd.concat([ev, ev2], ignore_index=True)
    ftype = dict(zip(fam.family_id, fam.type)) if len(fam) else {}
    ev["family_type"] = ev.family_id.map(ftype).fillna("")
    ev["background"] = (ev.family_id <= 0) & ~ev.linked
    ev["cls"] = np.where(ev.family_id > 0, ev.family_type,
                         np.where(ev.linked, "linked_to_single", "background"))
    regt = pd.DataFrame([dict(region=r, n_analysis=int((reg == r).sum()),
                              Mc=float(np.nanmax(mcv[sel & (df.region.values == r)]))
                              if variant != "mc_year" and (sel & (df.region.values == r)).any() else np.nan,
                              n_below_eta0=int(below[reg == r].sum()),
                              n_cut_by_cap=int((below & ~linked)[reg == r].sum()), **thr[r])
                         for r in REG_ORDER])
    regt = regt.merge(ctab[["region", "b", "df"]], on="region")
    return ev, fam, regt, thr


def summarize(ev, fam, reg):
    rows = []
    for r in REG_ORDER:
        e = ev[(ev.region == r) & ev.in_analysis]
        f = fam[fam.region == r] if len(fam) else fam
        rr = reg[reg.region == r].iloc[0]
        rows.append(dict(
            region=r, n_analysis=len(e), Mc=round(rr.Mc, 2), b=round(rr.b, 2), df=round(rr.df, 2),
            log_eta0=round(rr.log_eta0, 2), ashman_D=round(rr.ashman_D, 2),
            pct_background=round(100 * (e.role == "single").mean(), 1),
            pct_MA=round(100 * e.role.isin(["mainshock", "foreshock", "aftershock"]).mean(), 1),
            pct_swarm=round(100 * (e.role == "swarm").mean(), 1),
            pct_multiplet=round(100 * (e.role == "multiplet").mean(), 1),
            n_families=len(f), n_MA=int((f.type == "mainshock-aftershock").sum()) if len(f) else 0,
            n_MA_ge5=int(((f.type == "mainshock-aftershock") & (f.n >= 5)).sum()) if len(f) else 0,
            n_swarm=int((f.type == "swarm").sum()) if len(f) else 0,
            n_multiplet=int((f.type == "multiplet").sum()) if len(f) else 0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per-family details
def perm_p_rho(t, x, y, k0, rho_obs, nperm=2000, seed=1, batch=250):
    """One-sided permutation p-value of the radial Spearman rho: event positions are permuted against the times,
    and the origin (centroid of the first k0 events) is recomputed for every permutation, so the test includes
    the bias of taking the origin from the same events."""
    rng = np.random.default_rng(seed)
    n = len(t)
    rt = rankdata(t)
    rt = rt - rt.mean()
    rt = rt / np.sqrt((rt ** 2).sum())
    hits = 0
    for b0 in range(0, nperm, batch):
        m = min(batch, nperm - b0)
        P = np.argsort(rng.random((m, n)), axis=1)
        xs, ys = x[P], y[P]
        r = np.hypot(xs - xs[:, :k0].mean(1, keepdims=True), ys - ys[:, :k0].mean(1, keepdims=True))
        rr = rankdata(r, axis=1)
        rr = rr - rr.mean(1, keepdims=True)
        nr = np.sqrt((rr ** 2).sum(1))
        rho = np.where(nr > 0, (rr @ rt) / np.where(nr > 0, nr, 1), 0.0)
        hits += int((rho >= rho_obs - 1e-12).sum())
    return (hits + 1) / (nperm + 1)


def migration_details(s: pd.DataFrame, nmin: int = 10, nmax: int = 3000) -> dict:
    """Migration and sqrt(t) diffusion fit on the members s (time-sorted; tier A-C epicentres)."""
    out = dict(mig_n=len(s))
    if len(s) < nmin:
        return out
    la, lo = s.lat.values, s.lon.values
    x = (lo - lo.mean()) * KM * np.cos(np.radians(la.mean()))
    y = (la - la.mean()) * KM
    t = (s.t_yr.values - s.t_yr.values[0]) * YR
    if np.ptp(t) <= 0:
        return out
    k0 = max(3, len(s) // 10)
    x0, y0 = x[:k0].mean(), y[:k0].mean()
    r = np.hypot(x - x0, y - y0)
    w, vec = np.linalg.eigh(np.cov(np.c_[x, y].T))
    out["axis_az"] = float(np.degrees(np.arctan2(vec[0, 1], vec[1, 1])) % 180)
    out["axis_aspect"] = float(np.sqrt(w[1] / max(w[0], 1e-6)))
    h = s.h_max.values
    out["h_err_med_km"] = float(np.nanmedian(h))
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(s), nmax, replace=False)) if len(s) > nmax else np.arange(len(s))
    rho, p = spearmanr(t, r)
    out.update(mig_rho=float(rho), mig_p=float(p), mig_v_radial_km_day=float(theilslopes(r[idx], t[idx])[0]))
    # permutation p-value, on the subsample idx when there are more than nmax events
    if np.isfinite(rho):
        ts, xs, ys = t[idx], x[idx], y[idx]
        ks = max(3, len(idx) // 10)
        rho_s = rho if len(idx) == len(s) else spearmanr(ts, np.hypot(xs - xs[:ks].mean(), ys - ys[:ks].mean()))[0]
        out["mig_p_perm"] = float(perm_p_rho(ts, xs, ys, ks, rho_s))
    else:
        out["mig_p_perm"] = np.nan
    vx, vy = theilslopes(x[idx], t[idx])[0], theilslopes(y[idx], t[idx])[0]
    out.update(mig_vx_km_day=float(vx), mig_vy_km_day=float(vy), mig_v_dir_km_day=float(np.hypot(vx, vy)),
               mig_az=float(np.degrees(np.arctan2(vx, vy)) % 360))
    zg = s.depth_good.values
    if zg.sum() >= nmin and np.ptp(t[zg]) > 0:
        iz = np.where(zg)[0]
        iz = np.sort(rng.choice(iz, nmax, replace=False)) if len(iz) > nmax else iz
        out["mig_vz_km_day"] = float(theilslopes(s.depth.values[iz], t[iz])[0])
        out["mig_rho_z"] = float(spearmanr(t[iz], s.depth.values[iz])[0])
    r90 = float(np.quantile(r, 0.9))
    out["mig_r90_km"] = r90
    out["mig_resolvable"] = bool(r90 >= 2 * out["h_err_med_km"])
    tt, rr = t[k0:], r[k0:]
    ok = tt > 0
    if ok.sum() >= 5:
        Di = rr[ok] ** 2 / (4 * np.pi * tt[ok])          # km^2/day
        conv = 1e6 / 86400.0                              # km^2/day -> m^2/s
        out["D90_m2s"] = float(np.quantile(Di, 0.9) * conv)
        out["D50_m2s"] = float(np.quantile(Di, 0.5) * conv)
    if ok.sum() >= 12:
        lt = np.log10(tt[ok])
        edges = np.linspace(lt.min(), lt.max() + 1e-9, 9)
        bx, by = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            k = (lt >= e0) & (lt < e1)
            if k.sum() >= 3:
                bx.append(np.median(lt[k]))
                by.append(np.log10(max(np.quantile(rr[ok][k], 0.9), 0.05)))
        bx, by = np.array(bx), np.array(by)
        out["env_nbins"] = len(bx)
        if len(bx) >= 4 and np.ptp(bx) > 0.5:
            pfit, cfit = np.polyfit(bx, by, 1)
            out["env_p"] = float(pfit)
            out["env_rms_free"] = float(np.sqrt(np.mean((by - (pfit * bx + cfit)) ** 2)))
            for tag, sl in [("sqrt", 0.5), ("lin", 1.0)]:
                c0 = np.mean(by - sl * bx)
                out[f"env_rms_{tag}"] = float(np.sqrt(np.mean((by - (sl * bx + c0)) ** 2)))
    passes = (np.isfinite(out.get("mig_rho", np.nan)) and out["mig_rho"] >= 0.5
              and out["mig_v_radial_km_day"] > 0 and out["mig_resolvable"])
    mig = bool(passes and out["mig_p_perm"] < 0.01)
    out["mig_migrating"] = mig
    # the builder's first test (Spearman p), kept for comparison
    out["mig_migrating_spearman_p"] = bool(passes and out["mig_p"] < 0.01)
    # style of a migrating family from the exponent p of its 90 % distance envelope, r90 ~ t^p
    style = ""
    if mig:
        pe = out.get("env_p", np.nan)
        style = ("too few events for the envelope" if not np.isfinite(pe) else
                 "slower than sqrt(t)" if pe < 0.3 else "sqrt(t)-like" if pe < 0.75 else "linear-like")
    out["mig_style"] = style
    return out


def family_details(ev: pd.DataFrame, fam: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    cols = ["event_id", "t_yr", "time", "lat", "lon", "depth", "tier", "tierABC", "depth_good", "h_max"]
    e = ev[ev.family_id > 0][["event_id", "family_id", "M", "in_analysis"]].merge(df[cols], on="event_id")
    rows = []
    for F, g in e.groupby("family_id"):
        g = g.sort_values("t_yr")
        d = dict(family_id=F, n_all=len(g), n_tierABC=int(g.tierABC.sum()), n_tierD=int((g.tier == "D").sum()),
                 n_depth_good=int(g.depth_good.sum()), n_with_M=int(g.M.notna().sum()),
                 Mmin_all=float(g.M.min()) if g.M.notna().any() else np.nan,
                 start_all=g.time.iloc[0], end_all=g.time.iloc[-1],
                 duration_all_days=float((g.t_yr.iloc[-1] - g.t_yr.iloc[0]) * YR))
        z = g.depth[g.depth_good]
        if len(z):
            d.update(depth_good_min=float(z.min()), depth_good_q10=float(z.quantile(0.1)),
                     depth_good_med=float(z.median()), depth_good_q90=float(z.quantile(0.9)),
                     depth_good_max=float(z.max()))
        d.update(migration_details(g[g.tierABC]))
        rows.append(d)
    return fam.merge(pd.DataFrame(rows), on="family_id", how="left")


# ---------------------------------------------------------------- steps
def step_main(c, a):
    df = prepare(c)
    log(f"{len(c)} rows; {len(df)} located earthquakes, {df.M.notna().sum()} with M; "
        f"excluded {c.excluded.value_counts().to_dict()}")
    ctab, ytab = completeness(df)
    ctab.drop(columns="Mc_year").to_csv(OUT / "completeness_b_df.csv", index=False, float_format="%.3f")
    ytab.to_csv(OUT / "completeness_by_year.csv", index=False, float_format="%.2f")
    log("\n" + ctab.drop(columns="Mc_year").to_string())
    ev, fam, reg, thr = build(df, ctab, "main", a)
    log(f"main: {len(fam)} families")
    fam = family_details(ev, fam, df)
    summ = summarize(ev, fam, reg)
    log("\n" + summ.to_string())
    reg.to_csv(OUT / "thresholds_zbz.csv", index=False, float_format="%.4f")
    summ.to_csv(OUT / "regions_zbz.csv", index=False)
    # event table: every catalog row
    dd = df[["event_id", "time", "lat", "lon", "depth", "tier", "region_v51", "depth_good",
             "floor_limited", "far_only", "deep_pass", "five_phase"]].copy()
    dd["time"] = dd.time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    ev = ev.merge(dd, on="event_id", how="left")
    ev["excluded"] = ""
    x = c[c.excluded.ne("")]
    exc = pd.DataFrame(dict(event_id=x.event_id.values, region=box_of(x.lon.values, x.lat.values),
                            in_analysis=False, M=x.M.values, parent_id=-1, linked=False, family_id=-1,
                            role="excluded", family_type="", background=False, cls="excluded",
                            time=x.time.values, lat=x.lat.values,
                            lon=x.lon.values, depth=x.depth.values, tier=x.tier.values,
                            region_v51=x.region_v51.values, excluded=x.excluded.values))
    exc.loc[exc.excluded.eq("no_hypocentre"), "region"] = ""
    ev = pd.concat([ev, exc], ignore_index=True).sort_values("event_id")
    BIG.mkdir(parents=True, exist_ok=True)
    ev.to_csv(BIG / EVF, index=False, float_format="%.4f")
    if OUT.resolve() != BIG.resolve():
        ev[SLIM].to_csv(OUT / EVF, index=False, float_format="%.4f")
    fam.to_csv(OUT / FAMF, index=False, float_format="%.4f")
    extra = {} if CATV == "v5.1" else dict(
        catalog_version=CATV, magnitudes_column="M (catalog, D-B091 converged)", mc_best_source=str(MCB),
        M_source=c.magnitude_source.value_counts().to_dict())
    json.dump(dict(rmin=a.rmin, rcap_min=a.rcap_min, q=Q, df_main=1.6, catalog=str(CAT), **mag_paths.describe(MAG),
                   mc_b=f"{MAG_REF}:{MCB}" if CATV == "v5.1" else str(MCB),
                   n_rows=len(c), n_used=len(df), excluded=c.excluded.value_counts().to_dict(), **extra),
              open(OUT / "run_params.json", "w"), indent=1)
    return df, ctab, thr


SLIM = ["event_id", "region", "in_analysis", "M", "parent_id", "log_eta", "log_T", "log_R", "nn_dist_km",
        "nn_dt_days", "linked", "family_id", "family_type", "role", "background", "cls", "excluded"]


def step_shuffle(df, ctab, thr, a):
    srows, shuf_all = [], {}
    for sd in range(a.nshuffle):
        s = build(df, ctab, "main", a, shuffle_seed=100 + sd, thr_override={r: thr[r]["log_eta0"] for r in thr})
        for r in REG_ORDER:
            k = s["region"] == r
            le = s["log_eta"][k]
            g = gmm1d(le)
            srows.append(dict(region=r, seed=sd, n=int(k.sum()),
                              frac_below_eta0=float(np.nanmean(le < thr[r]["log_eta0"])),
                              frac_linked=float(s["linked"][k].mean()),
                              ashman_D=g["ashman_D"], mu1=g["mu1"], mu2=g["mu2"], w1=g["w1"]))
            shuf_all.setdefault(r, []).append(le[np.isfinite(le)])
        if sd == 0:
            np.savez_compressed(OUT / "shuffle_eta_seed0.npz", log_eta=s["log_eta"], log_T=s["log_T"],
                                log_R=s["log_R"], region=s["region"].astype(str))
        log(f"shuffle {sd} done")
    sh = pd.DataFrame(srows)
    sh.to_csv(OUT / "shuffle_test.csv", index=False, float_format="%.4f")
    thr_shuf = {r: float(np.quantile(np.concatenate(v), 0.05)) for r, v in shuf_all.items()}
    reg = pd.read_csv(OUT / "thresholds_zbz.csv")
    reg["log_eta0_shuf5"] = reg.region.map(thr_shuf)
    reg.to_csv(OUT / "thresholds_zbz.csv", index=False, float_format="%.4f")
    log("\n" + sh.groupby("region").mean(numeric_only=True).to_string())


def kde_valley_interior(g: dict, tol: float = 1e-6) -> bool:
    """gmm1d puts log_eta0_kde at the KDE minimum between the two means; at an end point there is no valley."""
    return bool(abs(g["log_eta0_kde"] - g["mu1"]) > tol and abs(g["log_eta0_kde"] - g["mu2"]) > tol)


def step_nulls(df, ctab, a, seeds=(10, 5, 5)):
    """Null test of the links. Time-shuffled catalogs, run at the REAL thresholds and caps: the share of events
    linked, the mixture separation D, and the families and swarms they form. Shuffles: within region over the
    whole period (as v4 and the shuffle step), within region and calendar year, within region and month (these
    two keep slow rate changes such as network changes and long sequences)."""
    reg = pd.read_csv(OUT / "thresholds_zbz.csv").set_index("region")
    thr0 = reg.log_eta0.to_dict()
    ev = pd.read_csv(OUT / EVF, usecols=["event_id", "region", "in_analysis", "linked",
                                                            "family_id"])
    ea = ev[ev.in_analysis.astype(bool)]
    fam = pd.read_csv(OUT / FAMF, usecols=["family_id", "region", "type"])
    rows = []
    for r in REG_ORDER:
        e = ea[ea.region == r]
        f = fam[fam.region == r]
        rows.append(dict(null="real", seed=-1, region=r, n=len(e), frac_linked=float(e.linked.astype(bool).mean()),
                         frac_in_family=float((e.family_id > 0).mean()), ashman_D=reg.ashman_D[r],
                         kde_valley=bool(abs(reg.log_eta0_kde[r] - reg.mu1[r]) > 1e-6
                                         and abs(reg.log_eta0_kde[r] - reg.mu2[r]) > 1e-6),
                         n_families=len(f), n_swarm=int((f.type == "swarm").sum()),
                         n_MA=int((f.type == "mainshock-aftershock").sum()), n_multiplet=int((f.type == "multiplet").sum())))
    for k_null, (null, ns) in enumerate(zip(["region", "region_year", "region_month"], seeds)):
        for sd in range(ns):
            t0 = time.time()
            s = build(df, ctab, "main", a, thr_override=thr0, shuffle_seed=2000 + 1000 * k_null + sd, null=null,
                      null_families=True)
            fs = s["fam"]
            for r in REG_ORDER:
                k = s["region"] == r
                g = gmm1d(s["log_eta"][k])
                f = fs[fs.region == r] if len(fs) else fs
                rows.append(dict(null=null, seed=sd, region=r, n=int(k.sum()), frac_linked=float(s["linked"][k].mean()),
                                 frac_in_family=float(s["n_in_family"].get(r, 0) / max(k.sum(), 1)),
                                 ashman_D=g["ashman_D"], kde_valley=kde_valley_interior(g), n_families=len(f),
                                 n_swarm=int((f.type == "swarm").sum()) if len(f) else 0,
                                 n_MA=int((f.type == "mainshock-aftershock").sum()) if len(f) else 0,
                                 n_multiplet=int((f.type == "multiplet").sum()) if len(f) else 0))
            log(f"null {null} seed {sd}: {time.time() - t0:.0f} s")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "null_test.csv", index=False, float_format="%.4f")
    real = R[R.null == "real"].set_index("region")
    S = R[R.null != "real"].groupby(["null", "region"]).agg(
        n_seeds=("seed", "size"), frac_linked_mean=("frac_linked", "mean"), frac_linked_max=("frac_linked", "max"),
        frac_in_family_mean=("frac_in_family", "mean"), D_mean=("ashman_D", "mean"), D_min=("ashman_D", "min"),
        D_max=("ashman_D", "max"), kde_valley_frac=("kde_valley", "mean"), n_families_mean=("n_families", "mean"),
        n_swarm_mean=("n_swarm", "mean"), n_swarm_max=("n_swarm", "max"), n_MA_mean=("n_MA", "mean")).reset_index()
    for col in ["frac_linked", "frac_in_family", "ashman_D", "kde_valley", "n_families", "n_swarm", "n_MA"]:
        S[f"real_{col}"] = S.region.map(real[col])
    S["link_ratio_null_over_real"] = S.frac_linked_mean / S.real_frac_linked
    S["swarm_ratio_null_over_real"] = S.n_swarm_mean / S.real_n_swarm.replace(0, np.nan)
    S["family_ratio_null_over_real"] = S.n_families_mean / S.real_n_families.replace(0, np.nan)
    S.to_csv(OUT / "null_test_summary.csv", index=False, float_format="%.4f")
    log("\n" + S[["null", "region", "frac_linked_mean", "real_frac_linked", "link_ratio_null_over_real", "D_mean",
                  "D_max", "real_ashman_D", "kde_valley_frac", "n_swarm_mean", "real_n_swarm"]].round(3).to_string())


def step_variants(c, df, ctab, a):
    vdir = BIG / "variants"
    vdir.mkdir(parents=True, exist_ok=True)
    main_ev = pd.read_csv(OUT / EVF, usecols=["event_id", "in_analysis", "background"])
    main_bg = main_ev[main_ev.in_analysis].set_index("event_id").background
    regs = pd.read_csv(OUT / "thresholds_zbz.csv")
    thr_shuf = dict(zip(regs.region, regs.log_eta0_shuf5))
    rows = [pd.read_csv(OUT / "regions_zbz.csv").assign(variant="main")]
    for v in [x for x in a.variants.split(",") if x]:
        f = vdir / f"events_{v}.csv.gz"
        fs = vdir / f"summary_{v}.csv"
        if f.exists() and fs.exists() and not a.force:
            rows.append(pd.read_csv(fs))
            continue
        t0 = time.time()
        if v in ("v41M", "storedM"):
            # other magnitudes on the same locations: Mc, b and eta0 recomputed from them (v41M: the v4.1 magnitudes
            # of the catalog v5.1 column M; storedM: the stored, pre-D-B091 magnitudes on the v6.0 locations)
            dfv = prepare(c, "M_v41" if v == "v41M" else "M_stored")
            ctv, _ = completeness(dfv)
            ctv.drop(columns="Mc_year").to_csv(vdir / f"completeness_{v}.csv", index=False, float_format="%.3f")
            ev_v, fam_v, reg_v, _ = build(dfv, ctv, "main", a)
        else:
            ev_v, fam_v, reg_v, _ = build(df, ctab, v, a, thr_override=thr_shuf if v == "eta0_shuf5" else None)
        sv = summarize(ev_v, fam_v, reg_v).assign(variant=v)
        e2 = ev_v[ev_v.in_analysis].set_index("event_id")
        com = main_bg.index.intersection(e2.index)
        same = pd.Series(main_bg[com].values == e2.background[com].values, index=com)
        sv["agree_bg_vs_clustered_with_main"] = sv.region.map(same.groupby(e2.region[com]).mean()).round(3)
        sv["n_common_with_main"] = sv.region.map(same.groupby(e2.region[com]).size())
        ev_v[["event_id", "region", "in_analysis", "M", "family_id", "family_type", "role", "background",
              "linked", "log_eta"]].to_csv(f, index=False, float_format="%.4f")
        fam_v.to_csv(vdir / f"families_{v}.csv", index=False, float_format="%.4f")
        sv.to_csv(fs, index=False)
        rows.append(sv)
        log(f"variant {v}: {time.time() - t0:.0f} s\n" + sv.to_string())
    S = pd.concat(rows, ignore_index=True)
    # Mc to 0.01 (Mc + 0.3 carries a float error, e.g. 2.8499999999999996) and the event count as an integer
    # (empty for the main rows); the cached summaries of earlier runs are written the same way
    S["Mc"] = S.Mc.round(2)
    S["n_common_with_main"] = S.n_common_with_main.round().astype("Int64")
    S.to_csv(OUT / "sensitivity_zbz.csv", index=False)


FLAG_COLS = ["frac_on_faults", "frac_on_faults_loose", "frac_not_fitted", "main_fault", "main_fault_frac",
             "fault_ids", "n_daytime_screened", "n_review_P1_msh_deep", "n_review_P2_brookings"]


def step_faults(ev, fam):
    """Join families with the fault surfaces v1 and flag families with events of the daytime-screened surfaces or
    of the waveform-review sets P1 and P2 (catalog_note of the assignment table).

    On a surface = on_surface_strict of the faults piece (fit members plus tier A-C 'assigned' events within 2 km,
    without floor_limited, far_only or deep_pass), as its README asks; the looser join of the first build (fit
    members plus every 'assigned' event) is kept as frac_on_faults_loose. Fractions leave out events at Axial
    Seamount, which was not fitted (NaN for families with no member in a fitted region)."""
    acols = ["event_id", "assignment", "surface_id", "on_surface_strict", "removed_surface", "catalog_note"]
    scols = ["surface_id", "region", "n_events", "strike", "dip", "kind", "stable", "flags", "change_from_v0",
             "v0_surfaces", "lit_match"]
    if FAULTS_DIR is not None:   # --faults: the catalog's own surfaces (pipeline, D-B110)
        asg = pd.read_csv(FAULTS_DIR / "fault_surfaces_v1_assignment.csv.gz", usecols=acols)
        summ = pd.read_csv(FAULTS_DIR / "fault_surfaces_v1_summary.csv", usecols=scols)
    else:
        asg = git_csv(FAULTS_REF, f"{FAULTS}/fault_surfaces_v1_assignment.csv.gz", usecols=acols)
        summ = git_csv(FAULTS_REF, f"{FAULTS}/fault_surfaces_v1_summary.csv", usecols=scols)
    asg["strict"] = _bool(asg.on_surface_strict)
    a2 = asg.set_index("event_id")
    on = asg[asg.strict][["event_id", "surface_id", "assignment"]]
    loose = asg[asg.assignment.isin(["fit member", "assigned"])].set_index("event_id").surface_id
    e = ev[ev.excluded.fillna("").eq("")].merge(on, on="event_id", how="left")
    e["surface_loose"] = e.event_id.map(loose)
    e["not_fitted"] = e.event_id.map(a2.assignment).fillna("").str.startswith("not fitted")
    e["daytime"] = e.event_id.map(a2.assignment).fillna("").str.startswith("non-earthquake timing")
    note = e.event_id.map(a2.catalog_note).fillna("")
    e["P1"] = note.str.contains("review P1")
    e["P2"] = note.str.contains("review P2")
    rows = []
    for F, g in e[e.family_id > 0].groupby("family_id"):
        gf = g[~g.not_fitted]
        vc = gf.surface_id.value_counts()
        fr = (vc / len(gf)).round(3) if len(gf) else vc
        rows.append(dict(family_id=F, frac_on_faults=round(gf.surface_id.notna().mean(), 3) if len(gf) else np.nan,
                         frac_on_faults_loose=round(gf.surface_loose.notna().mean(), 3) if len(gf) else np.nan,
                         frac_not_fitted=round(g.not_fitted.mean(), 3),
                         main_fault=vc.index[0] if len(vc) else "",
                         main_fault_frac=float(fr.iloc[0]) if len(fr) else 0.0,
                         fault_ids=";".join(f"{k}:{v}" for k, v in fr.items()),
                         n_daytime_screened=int(g.daytime.sum()), n_review_P1_msh_deep=int(g.P1.sum()),
                         n_review_P2_brookings=int(g.P2.sum())))
    fj = pd.DataFrame(rows)
    fam2 = fam.drop(columns=[c for c in FLAG_COLS if c in fam]).merge(fj, on="family_id", how="left")
    ea = e[e.surface_id.notna()]
    pf = []
    for sid, g in ea.groupby("surface_id"):
        an = g[g.in_analysis.astype(bool)]
        d = dict(surface_id=sid, n_events=len(g), n_analysis=len(an))
        for cc in ["background", "mainshock-aftershock", "swarm", "multiplet"]:
            d[f"n_{cc}"] = int((an.cls == cc).sum())
        d["pct_background"] = round(100 * d["n_background"] / len(an), 1) if len(an) else np.nan
        d["pct_swarm"] = round(100 * d["n_swarm"] / len(an), 1) if len(an) else np.nan
        d["n_families"] = int(an.family_id[an.family_id > 0].nunique())
        sw = an[an.cls == "swarm"].family_id.value_counts()
        d["n_swarm_families"] = len(sw)
        # how concentrated the swarm events of a surface are: share of its swarm events above Mc in its
        # largest one and two swarm families
        d["swarm_top1_family"] = int(sw.index[0]) if len(sw) else -1
        d["swarm_share_top1"] = round(sw.iloc[0] / sw.sum(), 3) if len(sw) else np.nan
        d["swarm_share_top2"] = round(sw.iloc[:2].sum() / sw.sum(), 3) if len(sw) else np.nan
        d["n_swarm_families_all"] = int(g[g.family_type.eq("swarm")].family_id.nunique())
        d["n_analysis_loose"] = int((e.surface_loose.eq(sid) & e.in_analysis.astype(bool)).sum())
        pf.append(d)
    pf = pd.DataFrame(pf).rename(columns={"n_events": "n_events_joined"}).merge(summ, on="surface_id", how="left")
    pf.to_csv(OUT / "faults_v1_cluster_counts.csv", index=False)
    # events of surfaces the faults piece removed for weekday-daytime timing (possible unflagged blasts): the
    # weekday-daytime members of those surfaces; their other members are assigned like any event
    ndt = asg[asg.assignment.str.startswith("non-earthquake timing")][["event_id"]]
    x = ndt.merge(ev, on="event_id", how="left")
    pd.DataFrame([dict(n=len(x), in_analysis=int(x.in_analysis.astype(bool).sum()),
                       in_families=int((x.family_id > 0).sum()),
                       n_families=int(x.family_id[x.family_id > 0].nunique()),
                       **{f"cls_{k}": int(v) for k, v in x.cls.value_counts().items()})]).to_csv(
        OUT / "faults_v1_daytime_surfaces_events.csv", index=False)
    return fam2, pf


# ---------------------------------------------------------------- checks asked by the review
def review_ids(problem: str, role: str | None = None) -> set:
    try:
        rv = pd.read_csv(REVIEW, usecols=["ID", "problem", "role"])
    except FileNotFoundError:
        return set()
    rv = rv[rv.problem.eq(problem) & (rv.role.eq(role) if role else True)]
    return set(rv.ID.astype(int))


def maxc_edges(m: np.ndarray, edges: np.ndarray) -> float:
    """v4 maximum-curvature Mc (left edge of the modal bin + 0.05 + 0.2) with the given bin edges."""
    m = m[np.isfinite(m)]
    if len(m) < 50:
        return np.nan
    h, e = np.histogram(m, bins=edges)
    return float(e[np.argmax(h)] + 0.05 + 0.2)


def step_checks(c, df):
    ev = pd.read_csv(BIG / EVF, low_memory=False)
    fam = pd.read_csv(OUT / FAMF)
    cc = c.set_index("event_id")
    ev["h_max"] = ev.event_id.map(cc.h_max)
    ev["five_phase"] = ev.event_id.map(cc.five_phase).fillna(False).astype(bool)
    ev["depth_good"] = _bool(ev.depth_good)
    an = ev[ev.in_analysis.astype(bool)]
    # ---- tier D by region: epicentre errors and links with a tier D end
    tier, hm = ev.set_index("event_id").tier, ev.set_index("event_id").h_max
    lk = an[an.linked.astype(bool)].copy()
    lk["tier_par"] = lk.parent_id.map(tier)
    lk["h_par"] = lk.parent_id.map(hm)
    lk["anyD"] = lk.tier.eq("D") | lk.tier_par.eq("D")
    lk["h10"] = np.maximum(lk.h_max, lk.h_par) >= 10
    rows = []
    for r in REG_ORDER + ["all"]:
        e = an if r == "all" else an[an.region == r]
        li = lk if r == "all" else lk[lk.region == r]
        dd, ab = e[e.tier == "D"], e[e.tier != "D"]
        rows.append(dict(region=r, n_analysis=len(e), n_tierD=len(dd), pct_tierD=round(100 * len(dd) / max(len(e), 1), 1),
                         h_max_med_tierD_km=round(dd.h_max.median(), 1), h_max_med_tierABC_km=round(ab.h_max.median(), 1),
                         n_five_phase=int(e.five_phase.sum()),
                         h_max_med_five_phase_km=round(e[e.five_phase].h_max.median(), 1),
                         pct_background_tierD=round(100 * dd.background.mean(), 1) if len(dd) else np.nan,
                         pct_background_tierABC=round(100 * ab.background.mean(), 1) if len(ab) else np.nan,
                         n_links=len(li), pct_links_tierD_end=round(100 * li.anyD.mean(), 1) if len(li) else np.nan,
                         pct_links_hmax_ge10=round(100 * li.h10.mean(), 1) if len(li) else np.nan,
                         n_families_tierD_largest=int((fam[fam.region == r] if r != "all" else fam).mainshock_id
                                                      .map(tier).eq("D").sum())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tierD_by_region.csv", index=False)
    log("\n" + T.to_string())
    # ---- Mc: v4 floating-point bin edges against exact decimal edges
    edges = {"v4_float": np.arange(-2, 8, 0.1), "exact": np.round(np.arange(-20, 81) / 10, 1)}
    rows = []
    for r in REG_ORDER:
        d = df[df.region == r]
        rr = dict(region=r)
        for tag, ed in edges.items():
            mcy = {y: maxc_edges(g.M.values, ed) for y, g in d.groupby("year") if len(g) >= 200}
            mcy = {y: v for y, v in mcy.items() if np.isfinite(v)}
            y0 = max(mcy, key=mcy.get)
            mc = round(mcy[y0], 2)
            # the two largest bins of the year that sets Mc with the v4 edges, counted with these edges
            yv4 = rr.get("year_v4_float", y0)
            hh, _ = np.histogram(d[d.year == yv4].M.dropna().values, bins=ed)
            top2 = np.sort(hh)[::-1][:2]
            rr.update({f"Mc_{tag}": mc, f"year_{tag}": y0, f"n_above_{tag}": int((d.M >= mc - 1e-9).sum()),
                       f"modal_counts_{tag}": f"{top2[0]};{top2[1]}"})
        # events exactly on the 0.1 grid (they fall in the bin below with the float edges)
        m = d.M.dropna().values
        rr["frac_M_on_0.1_grid"] = round(float(np.isclose(np.round(m, 1), m, atol=1e-9).mean()), 3)
        rows.append(rr)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "mc_edge_check.csv", index=False)
    log("\n" + M.to_string())
    # ---- depth of swarm and mainshock-aftershock families, families as the unit (Mann-Whitney on family
    # medians of depth_good members, families with >= 3 of them); also without the review sets P1 and P3
    p1, p3 = review_ids("P1", "deep"), review_ids("P3")
    freg = fam.set_index("family_id").region
    g0 = ev[ev.excluded.fillna("").eq("") & ev.depth_good & (ev.family_id > 0)]
    rows = []
    for tag, g in [("all", g0), ("without_P1_P3", g0[~g0.event_id.isin(p1 | p3)])]:
        f = g.groupby("family_id").agg(typ=("family_type", "first"), z=("depth", "median"), k=("depth", "size"))
        f["region"] = f.index.map(freg)
        f = f[f.k >= 3]
        rr = []
        for r in REG_ORDER:
            s_ = f[(f.region == r) & (f.typ == "swarm")].z
            m_ = f[(f.region == r) & (f.typ == "mainshock-aftershock")].z
            ge = g[g.family_id.map(freg).eq(r)]
            p = mannwhitneyu(s_, m_, alternative="two-sided").pvalue if len(s_) >= 2 and len(m_) >= 2 else np.nan
            rr.append(dict(events=tag, region=r, n_swarm_families=len(s_), n_MA_families=len(m_),
                           swarm_family_median_km=round(s_.median(), 1) if len(s_) else np.nan,
                           MA_family_median_km=round(m_.median(), 1) if len(m_) else np.nan,
                           swarm_event_median_km=round(ge[ge.family_type == "swarm"].depth.median(), 1),
                           MA_event_median_km=round(ge[ge.family_type == "mainshock-aftershock"].depth.median(), 1),
                           p_mannwhitney=p))
        rr = pd.DataFrame(rr)
        ntest = int(rr.p_mannwhitney.notna().sum())
        rr["n_tests"] = ntest
        rr["p_bonferroni"] = np.minimum(1.0, rr.p_mannwhitney * ntest)
        rows.append(rr)
    Dp = pd.concat(rows, ignore_index=True)
    Dp.to_csv(OUT / "depth_swarm_vs_ma_families.csv", index=False, float_format="%.4f")
    log("\n" + Dp.to_string())
    # ---- the waveform-review sets in the clustering
    try:
        rv = pd.read_csv(REVIEW, usecols=["ID", "problem", "role"])
    except FileNotFoundError:
        rv = pd.DataFrame(columns=["ID", "problem", "role"])
    rows = []
    ei = ev.set_index("event_id")
    for (p, role), g in rv.groupby(["problem", "role"]):
        x = ei.reindex(g.ID.astype(int))
        inf = x[x.family_id > 0]
        rows.append(dict(problem=p, role=role, n=len(g), excluded=int(x.excluded.fillna("").ne("").sum()),
                         in_analysis=int(x.in_analysis.astype(bool).sum()), depth_good=int(x.depth_good.sum()),
                         in_families=len(inf), n_families=inf.family_id.nunique(),
                         families_by_type=";".join(f"{k}:{v}" for k, v in
                                                   fam[fam.family_id.isin(inf.family_id)].type.value_counts().items())))
    pd.DataFrame(rows).to_csv(OUT / "review_sets_in_clusters.csv", index=False)
    # families holding Mount St Helens deep events (P1): start weekday and local time (the review found a weekly
    # artificial signal on Wednesdays on the analog UW channels HSR, EDM and STD)
    x = ev[ev.event_id.isin(p1) & (ev.family_id > 0)]
    f = fam[fam.family_id.isin(x.family_id)][["family_id", "region", "type", "n", "n_all", "Mmax", "start", "lon",
                                               "lat"]].copy()
    st = pd.to_datetime(f.start, utc=True, format="ISO8601").dt.tz_convert("America/Los_Angeles")
    f["weekday_local"] = st.dt.day_name()
    f["time_local"] = st.dt.strftime("%H:%M")
    f["n_P1_deep"] = f.family_id.map(x.family_id.value_counts())
    hh = st.dt.hour + st.dt.minute / 60
    f["wednesday_local_noon"] = (st.dt.dayofweek == 2) & (hh >= 11.75) & (hh < 12.5)
    f.to_csv(OUT / "review_P1_families.csv", index=False)
    log(f"P1 families: {len(f)}, Wednesday 11:45-12:30 local: {int(f.wednesday_local_noon.sum())}")


# ---------------------------------------------------------------- comparison with v4
def cls_of(e):
    return np.where(e.family_id > 0, "family", np.where(e.linked.astype(bool), "linked_to_single", "background"))


def match_families(m4: pd.DataFrame, m5: pd.DataFrame, f4: pd.DataFrame, f5: pd.DataFrame,
                   min_share: float = 0.2, ref: str = "v4", new: str = "v51"
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """m4, m5: event_id -> family id (members incl. attached) of the reference run (ref) and of the new run (new).
    Status of each reference and each new family; ref / new only name the output columns."""
    fin, npt = f"frac_in_{new}_families", f"n_{new}_parts"
    ffr, nrp = f"frac_from_{ref}_families", f"n_{ref}_parts"
    j = m4.merge(m5, on="event_id")
    ov = j.groupby(["f4", "f5"]).size().rename("n_shared").reset_index()
    n4, n5 = m4.f4.value_counts(), m5.f5.value_counts()
    ov["n4"], ov["n5"] = ov.f4.map(n4), ov.f5.map(n5)
    ov["sig4"] = ov.n_shared >= np.maximum(2, min_share * ov.n4)     # significant part of the v4 family
    ov["sig5"] = ov.n_shared >= np.maximum(2, min_share * ov.n5)     # significant part of the v5.1 family
    s4 = ov[ov.sig4].groupby("f4").f5.nunique()
    s5 = ov[ov.sig5].groupby("f5").f4.nunique()
    best4 = ov.sort_values("n_shared", ascending=False).drop_duplicates("f4").set_index("f4")
    best5 = ov.sort_values("n_shared", ascending=False).drop_duplicates("f5").set_index("f5")
    tot4 = ov.groupby("f4").n_shared.sum()
    tot5 = ov.groupby("f5").n_shared.sum()
    r4 = []
    for F, nF in n4.items():
        d = {"f4": F, "n4_all": int(nF), fin: float(tot4.get(F, 0) / nF), npt: int(s4.get(F, 0))}
        if F in best4.index:
            b = best4.loc[F]
            d.update(f5_main=int(b.f5), shared_main=int(b.n_shared), n5_main=int(b.n5))
        if d[fin] < 0.5:
            st = "dissolved"
        elif d[npt] >= 2:
            st = "split"
        elif s5.get(d["f5_main"], 0) >= 2:
            st = "merged"
        elif d["shared_main"] >= 0.5 * nF and d["shared_main"] >= 0.5 * d["n5_main"]:
            st = "kept"
        elif d["shared_main"] >= 0.5 * nF:
            st = "grown"
        else:
            st = "changed"
        d["status"] = st
        r4.append(d)
    r4 = pd.DataFrame(r4)
    st4 = r4.set_index("f4").status
    r5 = []
    for G, nG in n5.items():
        d = {"f5": G, "n5_all": int(nG), ffr: float(tot5.get(G, 0) / nG), nrp: int(s5.get(G, 0))}
        if G in best5.index:
            b = best5.loc[G]
            d.update(f4_main=int(b.f4), shared_main=int(b.n_shared), n4_main=int(b.n4))
        if d[nrp] == 0:
            st = "new"
        elif d[nrp] >= 2:
            st = "merged"
        elif st4.get(d.get("f4_main", -1), "") == "split":
            st = "split_part"
        elif st4.get(d.get("f4_main", -1), "") == "kept":
            st = "kept"
        elif st4.get(d.get("f4_main", -1), "") == "grown":
            st = "grown"
        else:
            st = "changed"
        d["status"] = st
        r5.append(d)
    r5 = pd.DataFrame(r5)
    return ov, r4, r5


def ref_tables(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Events (with positions when the reference has them) and families of comparison run REFS[key]."""
    R = REFS[key]
    want = ["event_id", "region", "in_analysis", "M", "family_id", "role", "linked", "lat", "lon", "depth",
            "excluded"]
    src = Path(R.get("big", R["dir"])) / R["events"]
    if not src.exists():
        src = Path(R["dir"]) / R["events"]
    e = pd.read_csv(src, usecols=lambda x: x in want, low_memory=False)
    f = pd.read_csv(Path(R["dir"]) / R["families"])
    return e, f


def step_compare(c, df, ctab, a):
    """Compare with every run of REFS: event classes, analysis-set membership, family matching (status of each
    family on both sides), fate of dissolved families, location shift. The first entry of REFS is the primary
    comparison (status files without '_vs_<key>')."""
    ev = pd.read_csv(BIG / EVF, low_memory=False)
    fam = pd.read_csv(OUT / FAMF)
    ev["cls5"] = np.where(ev.excluded.fillna("").ne(""), "excluded_" + ev.excluded.fillna(""), cls_of(ev))
    e4v = None
    for k_ref, ref in enumerate(REFS):
        e4, f4 = ref_tables(ref)
        sfx = "" if k_ref == 0 else f"_vs_{ref}"
        # ---- event-level transitions
        e4["cls4"] = cls_of(e4)
        e4["type4"] = e4.family_id.map(f4.set_index("family_id").type).fillna("")
        j = e4.merge(ev[["event_id", "cls5", "in_analysis", "family_type", "region", "M"]], on="event_id",
                     how="outer", suffixes=("_4", "_5"))
        j["cls4"] = j.cls4.fillna(f"not in {ref}")
        j["cls5"] = j.cls5.fillna(f"not in {CATV}")
        if "excluded" in e4:     # excluded rows of the reference run (no hypocentre, not an earthquake)
            ex4 = j.excluded.fillna("").astype(str)
            j["cls4"] = np.where(ex4.ne("") & ex4.ne("nan"), "excluded_" + ex4, j.cls4)
        pd.crosstab(j.cls4, j.cls5, margins=True).to_csv(OUT / f"{ref}_vs_{TAG}_event_classes.csv")
        both = j[j.in_analysis_4.eq(True) & j.in_analysis_5.eq(True)]
        tr = pd.crosstab([both.region_4], [both.cls4 + "->" + both.cls5])
        tr.to_csv(OUT / f"{ref}_vs_{TAG}_event_classes_analysis_by_region.csv")
        # analysis-set membership changes (Mc and magnitude changes)
        j["an4"] = j.in_analysis_4.eq(True)
        j["an5"] = j.in_analysis_5.eq(True)
        pd.crosstab(j.region_4.fillna(j.region_5), [j.an4, j.an5]).to_csv(
            OUT / f"{ref}_vs_{TAG}_analysis_membership.csv")
        # ---- family matching
        m4 = e4[e4.family_id > 0][["event_id", "family_id"]].rename(columns={"family_id": "f4"})
        m5 = ev[ev.family_id > 0][["event_id", "family_id"]].rename(columns={"family_id": "f5"})
        ov, r4, r5 = match_families(m4, m5, f4, fam, ref=ref, new=TAG)
        # where did the members of dissolved reference families go
        dis = m4[m4.f4.isin(r4[r4.status == "dissolved"].f4)].merge(ev[["event_id", "cls5"]], on="event_id",
                                                                     how="left")
        r4 = r4.merge(f4[["family_id", "region", "type", "n", "Mmax", "start", "lon", "lat", "extent_km",
                          "duration90_days", "mean_leaf_depth"]].rename(columns={"family_id": "f4"}), on="f4",
                      how="left")
        r5 = r5.merge(fam[["family_id", "region", "type", "n", "Mmax", "start", "lon", "lat", "extent_km",
                           "duration90_days", "mean_leaf_depth"]].rename(columns={"family_id": "f5"}), on="f5",
                      how="left")
        tmap = r5.set_index("f5").type
        r4[f"type_{TAG}_main"] = r4.f5_main.map(tmap)
        r4.to_csv(OUT / f"{ref}_families_status.csv", index=False, float_format="%.3f")
        r5[f"type_{ref}_main"] = r5.f4_main.map(f4.set_index("family_id").type)
        r5.to_csv(OUT / f"{TAG}_families_status{sfx}.csv", index=False, float_format="%.3f")
        cols = [f"{ref}_status", f"{ref}_family_main", f"{ref}_shared_main"]
        fam = fam.drop(columns=[x for x in cols if x in fam]).merge(
            r5[["f5", "status", "f4_main", "shared_main"]].rename(columns={
                "f5": "family_id", "status": cols[0], "f4_main": cols[1], "shared_main": cols[2]}),
            on="family_id", how="left")
        ov.to_csv(OUT / f"{ref}_{TAG}_family_overlap.csv", index=False)
        dis_counts = dis.cls5.fillna(f"not in {CATV}").value_counts()
        dis_counts.rename("n").to_csv(OUT / f"{ref}_dissolved_members_fate.csv")
        # ---- location change of the events between the reference run and this one
        if {"lat", "lon", "depth"} <= set(e4.columns):
            sh = df[["event_id", "lat", "lon", "depth", "depth_good"]].merge(
                e4[["event_id", "lat", "lon", "depth"]].dropna(subset=["lat"]), on="event_id", suffixes=("", "_r"))
            sh["dh_km"] = np.hypot((sh.lon - sh.lon_r) * KM * np.cos(np.radians(sh.lat)), (sh.lat - sh.lat_r) * KM)
            sh["dz_km"] = sh.depth - sh.depth_r
            sh[["event_id", "dh_km", "dz_km", "depth_good"]].to_csv(BIG / f"{ref}_{TAG}_shift.csv.gz", index=False,
                                                                    float_format="%.3f")
        if ref == "v4":
            e4v = e4
        log(f"compare with {ref}: {TAG} families {r5.status.value_counts().to_dict()}; "
            f"{ref} families {r4.status.value_counts().to_dict()}")
    fam.to_csv(OUT / FAMF, index=False, float_format="%.4f")
    # ---- validation, volcano zoom, setting stats
    validate(ev, fam, c, e4v if e4v is not None else e4)
    volcano_local(df, a)
    setting_stats(ev, fam, ctab)


def validate(ev, fam, c, e4):
    rows = []
    fi = fam.set_index("family_id")
    ev = ev.copy()
    ev["time"] = pd.to_datetime(ev.time, utc=True, format="ISO8601")
    cc = c.set_index("event_id")
    p4 = e4.set_index("event_id")[["lat", "lon"]]
    pv = ev.set_index("event_id")[["lat", "lon"]]
    # the same families in variant tierABC (tier D out of the analysis set)
    try:
        vt = pd.read_csv(BIG / "variants/events_tierABC.csv.gz", usecols=["event_id", "family_id"]).set_index(
            "event_id").family_id
        fv = pd.read_csv(BIG / "variants/families_tierABC.csv").set_index("family_id")
    except FileNotFoundError:
        vt, fv = pd.Series(dtype=int), pd.DataFrame()

    def dist(i, j, P):
        if i not in P.index or j not in P.index:
            return np.nan
        return round(float(km(P.lon[i], P.lat[i], P.lon[j], P.lat[j])), 1)

    for eid, label, expect in KNOWN:
        e = ev[ev.event_id == eid]
        if not len(e):
            continue
        e = e.iloc[0]
        d = dict(event_id=eid, label=label, expected=expect, region=e.region, M=e.M, role=e.role,
                 family_id=int(e.family_id), excluded=e.excluded)
        if e.family_id > 0:
            f = fi.loc[e.family_id]
            d.update(type=f.type, n=int(f.n), n_all=int(f.n_all), Mmax=f.Mmax, M2=f.M2,
                     duration90_days=round(f.duration90_days, 1), extent_km=round(f.extent_km, 1),
                     mean_leaf_depth=round(f.mean_leaf_depth, 2), regions=f.regions)
            # largest other member: its location quality and distance from this event in v5.1 and v4
            mem = ev[(ev.family_id == e.family_id) & (ev.event_id != eid)].sort_values("M", ascending=False)
            if len(mem):
                o = int(mem.event_id.iloc[0])
                d.update(other_id=o, other_M=mem.M.iloc[0], other_time=str(mem.time.iloc[0])[:19],
                         other_tier=cc.tier.get(o), other_h_max_km=round(float(cc.h_max.get(o, np.nan)), 1),
                         other_n_sta=cc.n_sta.get(o), other_gap=round(float(cc.gap.get(o, np.nan)), 0),
                         **{f"other_dist_{TAG}_km": dist(eid, o, pv), "other_dist_v4_km": dist(eid, o, p4)})
            Fv = vt.get(eid, -1)
            if Fv > 0 and len(fv):
                g = fv.loc[Fv]
                d.update(tierABC_type=g.type, tierABC_n=int(g.n), tierABC_M2=g.M2,
                         tierABC_duration90_days=round(g.duration90_days, 1),
                         tierABC_mean_leaf_depth=round(g.mean_leaf_depth, 2))
        rows.append(d)
    ax = ev[(ev.region == "AX") & ev.in_analysis.astype(bool) & (ev.time >= "2015-04-23") & (ev.time < "2015-05-31")]
    big = ax[ax.family_id > 0].family_id.value_counts().head(1)
    for F, nn in big.items():
        f = fi.loc[F]
        rows.append(dict(event_id=int(f.mainshock_id), label="2015-04-24 Axial Seamount eruption onset",
                         expected="swarm", region="AX", M=f.Mmax, role="largest event", family_id=int(F),
                         type=f.type, n=int(f.n), n_all=int(f.n_all), Mmax=f.Mmax, M2=f.M2,
                         duration90_days=round(f.duration90_days, 1), extent_km=round(f.extent_km, 1),
                         mean_leaf_depth=round(f.mean_leaf_depth, 2), regions=f.regions,
                         note=f"start {f.start}; {100 * nn / len(ax):.0f}% of AX analysis events "
                              f"2015-04-23 to 05-31 in this family"))
    pd.DataFrame(rows).to_csv(OUT / "validation_sequences.csv", index=False)
    vrows = []
    for v, (lo, la) in VOLCANOES.items():
        f = fam[km(fam.lon, fam.lat, lo, la) <= 15]
        e2 = ev[(km(ev.lon, ev.lat, lo, la) <= 15) & ev.in_analysis.astype(bool)]
        if len(e2) == 0:
            continue
        vrows.append(dict(volcano=v, n_events_15km=len(e2), pct_background=round(100 * e2.background.mean(), 1),
                          n_families=len(f), n_swarm=int((f.type == "swarm").sum()),
                          n_MA=int((f.type == "mainshock-aftershock").sum()),
                          n_multiplet=int((f.type == "multiplet").sum()),
                          largest_family_n=int(f.n.max()) if len(f) else 0,
                          largest_family_type=f.sort_values("n").type.iloc[-1] if len(f) else ""))
    pd.DataFrame(vrows).to_csv(OUT / "volcano_families.csv", index=False)


def km(lon, lat, lon0, lat0):
    return np.hypot((lon - lon0) * KM * np.cos(np.radians(lat0)), (lat - lat0) * KM)


def volcano_local(df, a, radius=20.0, names=("Mount St. Helens", "Mount Hood", "Mount Rainier"), nshuf=10):
    """Zoomed analysis around volcanoes with a local Mc (as in v4_clusters_zbz_report.volcano_local). The
    shuffled D is given for the v4 seed and as median and maximum over nshuf seeds.

    The local Mc is rounded to 0.01 and applied with the rule of the main run (M >= Mc - 1e-9, as in select):
    the unrounded maximum-curvature value carries a float error (0.35000000000000187 at Mount St Helens) that
    left out the events with M exactly at Mc. The v4 zoom compared M with the unrounded value."""
    p1 = review_ids("P1", "deep")
    rows, frows = [], []
    for v in names:
        lo, la = VOLCANOES[v]
        sub = df[km(df.lon.values, df.lat.values, lo, la) <= radius]
        mc = round(mc_maxc(sub.M.values), 2)
        b, _, _ = b_aki(sub.M.values, mc - 1e-9)
        A = sub[sub.M >= mc - 1e-9].sort_values("t_yr").reset_index(drop=True)
        n = len(A)
        if n < 50:
            continue
        u = xyz(A.lat.values, A.lon.values)
        t, z, m = A.t_yr.values, A.depth.values, A.M.values
        zero = np.zeros(n, int)
        cap = rupture_cap(m)
        res = {}
        for tag, tt in [("real", t), ("shuffled", np.sort(np.random.default_rng(5).permutation(t)))]:
            uu = u if tag == "real" else u[np.random.default_rng(6).permutation(n)]
            res[tag] = nn_search(tt, uu, z, np.full(n, 1.6), tt, uu, z, b * m, 0.5, False, True, min(a.workers, 8),
                                 zero, zero, cap)
        par, o = res["real"]
        g = gmm1d(o[:, 0])
        gs = gmm1d(res["shuffled"][1][:, 0])
        dsh = [gs["ashman_D"]]
        for k in range(1, nshuf):  # more shuffles: positions permuted against the times
            uu = u[np.random.default_rng(100 + k).permutation(n)]
            dsh.append(gmm1d(nn_search(t, uu, z, np.full(n, 1.6), t, uu, z, b * m, 0.5, False, True,
                                       min(a.workers, 8), zero, zero, cap)[1][:, 0])["ashman_D"])
        linked = np.isfinite(o[:, 0]) & (o[:, 0] < g["log_eta0"]) & (o[:, 3] <= cap[np.maximum(par, 0)])
        ii = np.where(linked)[0]
        _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, par[ii])), shape=(n, n)), directed=False)
        sz = np.bincount(lab)
        types = []
        for cc in np.unique(lab[sz[lab] >= 2]):
            mem = np.where(lab == cc)[0]
            tm = tree_metrics(mem, par, linked, m)
            ms = np.sort(m[mem])[::-1]
            typ, _ = classify(len(mem), ms[0] - ms[1], tm["imax_local"] / len(mem), tm)
            types.append(typ)
            zg = A.depth.values[mem][A.depth_good.values[mem]]
            frows.append(dict(volcano=v, n=len(mem), Mmax=ms[0], M2=ms[1], type=typ, start=A.time.values[mem[0]],
                              duration_days=(t[mem[-1]] - t[mem[0]]) * YR, mean_leaf_depth=tm["mean_leaf_depth"],
                              depth_good_med=float(np.median(zg)) if len(zg) else np.nan, n_depth_good=len(zg),
                              n_review_P1_msh_deep=int(A.event_id.iloc[mem].isin(p1).sum())))
        ty = pd.Series(types, dtype=object)
        inP1 = A.event_id.isin(p1).values
        rows.append(dict(volcano=v, radius_km=radius, n_all=len(sub), Mc_local=mc, b_local=round(b, 2), n_above=n,
                         log_eta0=round(g["log_eta0"], 2), ashman_D=round(g["ashman_D"], 2),
                         ashman_D_shuffled=round(gs["ashman_D"], 2),
                         ashman_D_shuffled_med=round(float(np.median(dsh)), 2),
                         ashman_D_shuffled_max=round(float(np.max(dsh)), 2),
                         n_shuffles_D_ge_real=int((np.array(dsh) >= g["ashman_D"]).sum()), n_shuffles=len(dsh),
                         n_review_P1_above=int(inP1.sum()), n_review_P1_in_families=int((inP1 & (sz[lab] >= 2)).sum()),
                         pct_background=round(100 * (sz[lab] == 1).mean(), 1),
                         n_families=len(types), n_swarm=int((ty == "swarm").sum()),
                         n_MA=int((ty == "mainshock-aftershock").sum()), n_multiplet=int((ty == "multiplet").sum())))
    pd.DataFrame(rows).to_csv(OUT / "volcano_local_zbz.csv", index=False)
    pd.DataFrame(frows).to_csv(OUT / "volcano_local_families.csv", index=False, float_format="%.3f")


def setting_stats(ev, fam, ctab):
    rows = []
    mc = ctab.set_index("region").Mc
    ea = ev[ev.in_analysis.astype(bool)]
    off = ea[ea.linked.astype(bool)].groupby("parent_id").size()
    e_fs = ea[ea.role == "foreshock"].groupby("family_id").size()
    for r in REG_ORDER:
        f = fam[fam.region == r]
        f5 = f[f.n >= 5]
        ma = f[f.type == "mainshock-aftershock"]
        band = ea[(ea.region == r) & (ea.M - mc[r] >= 1.0) & (ea.M - mc[r] < 2.0)]
        e_as = band.event_id.map(off).fillna(0)
        e = ea[ea.region == r]
        big = e[e.M >= mc[r] + 1.0]
        f10 = f[f.n_all >= 10]
        rows.append(dict(
            region=r, n_fam_ge5=len(f5),
            pct_swarm_of_ge5=round(100 * (f5.type == "swarm").mean(), 1) if len(f5) else np.nan,
            median_leaf_depth_ge5=round(f5.mean_leaf_depth.median(), 2) if len(f5) else np.nan,
            n_events_Mc1to2=len(band),
            mean_offspring_Mc1to2=round(e_as.mean(), 2) if len(band) else np.nan,
            pct_Mc1to2_no_offspring=round(100 * (e_as == 0).mean(), 1) if len(band) else np.nan,
            n_events_ge_Mc1=len(big), pct_ge_Mc1_single=round(100 * (big.role == "single").mean(), 1) if len(big)
            else np.nan,
            pct_MA_with_foreshocks=round(100 * (ma.family_id.map(e_fs).fillna(0) > 0).mean(), 1) if len(ma)
            else np.nan,
            pct_fam_migrating_ge10=round(100 * f[f.n >= 10].migrating.mean(), 1) if (f.n >= 10).any() else np.nan,
            n_fam_ge10=int((f.n >= 10).sum()),
            n_fam_all_ge10=len(f10),
            **{f"pct_fam_all_ge10_migrating_{TAG}": round(100 * f10.mig_migrating.fillna(False).astype(bool).mean(), 1)
               if len(f10) else np.nan}))
    pd.DataFrame(rows).to_csv(OUT / "setting_stats.csv", index=False)


# ---------------------------------------------------------------- copy to the repository (v6.0)
REPO_V60 = "data/catalog_v6/v6.0_analysis/clusters"


def step_repo():
    """v6.0 writes everything to BIG on /wd1; copy the small outputs (tables, figures, report, the slim event table)
    to the repository folder REPO_V60 for the commit. Large files (full event table, variants, shifts) stay on /wd1."""
    import shutil
    dst = ROOT / REPO_V60
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.suffix in (".csv", ".json", ".png", ".html", ".md", ".npz") and f.name != "RESUME.md" \
                and f.stat().st_size < 5e6:
            shutil.copy2(f, dst / f.name)
            n += 1
    ev = pd.read_csv(BIG / EVF, usecols=SLIM, low_memory=False)
    ev.to_csv(dst / EVF, index=False, float_format="%.4f")
    log(f"repo: {n} files and the slim {EVF} copied to {dst}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", default="v5.1", choices=sorted(CATALOGS),
                    help="catalog version (v5.1: committed run with --mag stored; v6.0: magnitudes of the catalog)")
    ap.add_argument("--steps", default="main,shuffle,variants,faults,nulls,checks,compare")
    ap.add_argument("--rmin", type=float, default=0.5, help="distance floor (km)")
    ap.add_argument("--rcap-min", type=float, default=10.0, help="minimum link distance cap (km)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nshuffle", type=int, default=3)
    ap.add_argument("--variants", default=None, help="comma-separated (default: all variants of the catalog)")
    ap.add_argument("--out", default=None, help="folder of the tables (default: set by --catalog and --mag)")
    ap.add_argument("--big", default=None, help="folder of the large outputs (default: set by --catalog and --mag)")
    ap.add_argument("--force", action="store_true", help="recompute cached steps")
    ap.add_argument("--mcb", default=None, help="v6.x: Mc map table (mc_regional.csv) for variant mc_best "
                    "(default: the catalog's entry in CATALOGS)")
    ap.add_argument("--faults", default=None, help="folder of the catalog's fault surfaces v1 for step faults "
                    "(default: the v5.1 surfaces committed on origin/main)")
    mag_paths.add_argument(ap, "v5.1")
    a = ap.parse_args()
    configure(a.catalog, a.mag, a.out, a.big, a.mcb, a.faults)
    if a.variants is None:
        a.variants = ",".join(VARIANTS)
    print(f"catalog {CATV}: {CAT}; magnitudes: {MAG}; Mc_best: {MCB}; OUT {OUT}; BIG {BIG}; "
          f"compare with {list(REFS)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    BIG.mkdir(parents=True, exist_ok=True)
    steps = a.steps.split(",")
    c = load_all()
    if "main" in steps and (a.force or not (OUT / FAMF).exists()):
        step_main(c, a)
    df = prepare(c)
    ctab, _ = completeness(df)
    reg = pd.read_csv(OUT / "thresholds_zbz.csv")
    thr = {r: dict(log_eta0=v) for r, v in zip(reg.region, reg.log_eta0)}
    if "shuffle" in steps and (a.force or "log_eta0_shuf5" not in reg.columns):
        step_shuffle(df, ctab, thr, a)
    if "variants" in steps:
        step_variants(c, df, ctab, a)
    if "faults" in steps:
        ev = pd.read_csv(BIG / EVF, low_memory=False)
        fam = pd.read_csv(OUT / FAMF)
        fam, pf = step_faults(ev, fam)
        fam.to_csv(OUT / FAMF, index=False, float_format="%.4f")
        log(f"faults: {len(pf)} surfaces with events")
    if "nulls" in steps and (a.force or not (OUT / "null_test_summary.csv").exists()):
        step_nulls(df, ctab, a)
    if "checks" in steps:
        step_checks(c, df)
    if "compare" in steps:
        step_compare(c, df, ctab, a)
        log("compare done")
    if "repo" in steps and CATV == "v6.0":
        step_repo()


if __name__ == "__main__":
    main()
