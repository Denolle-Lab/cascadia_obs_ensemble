"""Track B, catalog v6.0 (D-B079, D-B096, D-B097) and v6.1 (D-B103, D-B104): the catalog of the v6 relocation
(b4_v6_reloc.py, run ROOT), with the v5.1 columns (b4_catalog.py / a2_build_catalog.py rules) and the v6.0 columns of
notes/v6_plan.md. The final SSST pass is it<NPASS> of b4_v6_reloc.SCHEDULE (V6_SCHEDULE: v6.0 it4, CharDist 4 km; v6.1
it2, 16 km); its label is ssst_it<NPASS>.

Hypocentre of each event (one row per event of the catalog v4.1 event list), first that applies:
  deep pass LOCATED (deep_pass; D-B067) > edge pass LOCATED (edge_pass) > coherence stage LOCATED (coherence_relocated)
  > final SSST pass (v6.0 it4, v6.1 it2) LOCATED > no hypocentre (loc_source none, tier U, u_reason).
Depths are true depths (the Earth-flattening of the grids undone, z = Re (1 - exp(-z'/Re))); nothing is moved or clamped
(D-B055): surface_km (merged surface, D-B068) and above_surface are reported, and the acceptance stage stops the chain if
any event is above the surface.

Columns (v5.1 ones as b4_catalog.py; new ones):
  model_version            velocity model of the travel-time grids (v6.0 v0.4.1, v6.1 v0.4.2)
  loc_pass                 ssst_it<NPASS> (v6.0 ssst_it4, v6.1 ssst_it2), coherence, deep or edge (the NLLoc run of the
                           hypocentre)
  station_terms            ssst (NLL-SSST grids, CharDist schedule 9999-16-8-4 km in v6.0, 9999-16 km in v6.1) or static
                           (deep and edge passes: static terms of it0, the SSST corrections stop at z' = 80 km and at the
                           tile box)
  ssst_char_km             final Loc2ssst CharDist of the event's grids (v6.0 4 km, v6.1 16 km; empty for static rows)
  rms_unweighted           unweighted rms of the residuals of the used arrivals (weight > 0), s (NLLoc's RMS is
                           EDT-weighted, D-B069)
  n_picks_retimed          picks of the event re-timed from a short read (D-B072/D-B075; timing_corrected)
  n_picks_clock_corrected  picks corrected for a confirmed station clock drift (D-B090; clock_correction_s != 0)
  shift_h_v51_km, shift_z_v51_km  change from catalog v5.1 (horizontal; depth, positive = deeper), both located
  coherence_relocated      relocated by the NLL-coherence stage (compact pdf, Len3 <= 5 km)
  coherence_partners       partners with pair coherence >= 0.5 and a compact pdf (stackable) of a coherence-stage event
  coherence_skipped_region compact event not coherence-relocated because it lies in the Mendocino region (D-B098); SSST location kept
  edge_pass                hypocentre from the edge pass (event rejected at a lateral face of its SSST tile box)
  review_problems, review_auto_indicators   v5.1 location-review package (review_v51_events.csv): problem groups and
                           the automatic indicators (auto_indicator_rule_not_verdict), joined with ';' (D-B097: carried
                           as columns; no reviewer verdict is applied)
  review_not_earthquake, review_depth_unreliable, review_relocate, review_depth_km   reviewer columns: empty (D-B097),
                           or, with V6_REVIEW_DIR (the outputs of review_v51/ingest_review.py: review_answers_merged.csv,
                           review_flag_not_earthquake.csv, review_depth_unreliable.csv, review_relocate.csv), True for the
                           listed events, False for the other reviewed events, empty for events nobody reviewed;
                           review_depth_km = the corrected depth of review_relocate.csv when one value is given
  event_flag               v5.1 rules (b4_catalog.flags), plus the flags of V6_EVENT_FLAGS (CSV with ID, event_flag;
                           e.g. the Mount St Helens analog-signal events; ';'-joined, no event removed)
  M, M_unc, M_type, ML, ML_unc, n_ML, ML_method, Mw_mt, comcat_id, magnitude_source   converged Route A magnitudes
                           (D-B091): the v5.1 table (computed with v5.1 hypocentres) where it has the event, else the
                           v4.1 table (v4 hypocentres); magnitude_source names which, with the short labels of
                           catalog_version.py (v5.1_converged, v4.1_converged; PR #58); not recomputed on v6 hypocentres

  python b4_v6_catalog.py ROOT --out CSV [--mag-tables LABEL=CSV,LABEL=CSV]
--mag-tables: the ordered magnitude tables (first with an M per event; key column ID or event_id), each with its
magnitude_source label. Default v5.1_converged (MAG_V51), v4.1_converged (MAG_V41). Catalog v6.1 (D-B105): the chain's
magnitudes stage computes the converged Route A magnitudes on the v6.1 hypocentres and rebuilds the catalog with
v6.1_converged=<that table>,v4.1_converged=MAG_V41 (hypocentre columns unchanged).
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_relocation"))
import a2_parse_hyp as PH  # noqa: E402
from a2_build_catalog import tier  # noqa: E402
import b4_catalog as BC  # noqa: E402
import b4_v6_reloc as V  # noqa: E402

V51 = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
REVIEW = Path("/wd1/mdenolle_data/catalog_v5/v5.1_review/review_v51_events.csv")
MAG_V51 = Path("/wd1/mdenolle_data/magnitude_converged/v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv")
MAG_V41 = Path("/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv")
# the chain's picks (b4_v6_relocate.sh PICKS, exported as V6_PICKS)
PICKS = Path(os.environ.get("V6_PICKS", "/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv"))
MAG_COLS = ["M", "M_unc", "M_type", "ML", "ML_unc", "n_ML", "ML_method", "Mw_mt", "comcat_id"]
PRIO = {("deep", True): 4, ("edge", True): 3, ("coh", True): 2, ("it4", True): 1, ("it4", False): 0}
# "it4" = the run label of the final SSST pass inside this script (it<NPASS>); its catalog label names the pass
LOC_PASS = {"it4": f"ssst_it{V.NPASS}", "coh": "coherence", "deep": "deep", "edge": "edge"}


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def parse_sum_file(args):
    """rows of one summary file: a2_parse_hyp.parse fields + the per-event file root."""
    path, run = args
    rows = PH.parse(path)
    roots = {r["ID"]: r["root"] for r in V.sum_rows(path) if "ID" in r}
    for r in rows:
        r["root"] = roots.get(r.get("ID"))
        r["run"] = run
        r["workdir"] = str(Path(path).parent)
    return rows


def phase_stats(root):
    """(unweighted rms, used arrivals) of a per-event hyp file (summary root + '.loc.hyp'): residuals of the arrivals with
    weight > 0."""
    res, on = [], False
    try:
        fh = open(f"{root}.loc.hyp", errors="replace")      # NLLOC line root: <path>.grid0
    except OSError:
        return np.nan, 0
    with fh:
        for line in fh:
            if line.startswith("PHASE "):
                on = True
                continue
            if line.startswith("END_PHASE"):
                break
            if on and ">" in line:
                u = line.split(">")[1].split()
                if float(u[2]) > 0:
                    res.append(float(u[1]))
    if not res:
        return np.nan, 0
    r = np.asarray(res)
    return float(np.sqrt(np.mean(r ** 2))), len(r)


def _phase_many(roots):
    return [phase_stats(r) for r in roots]


def km_h(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    return 2 * R * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


def read_runs(root):
    files = [(f, "it4") for f in sorted((root / f"it{V.NPASS}").glob("*/loc.sum.grid0.loc.hyp"))]
    files += [(f, "coh") for f in sorted((root / "coh").glob("*/e*/loc.sum.grid0.loc.hyp"))]
    for k in ("deep", "edge"):
        files += [(f, k) for f in sorted((root / k).glob("*/loc.sum.grid0.loc.hyp"))]
    with ProcessPoolExecutor(16) as ex:
        rows = [r for part in ex.map(parse_sum_file, [(str(f), k) for f, k in files], chunksize=64) for r in part]
    loc = pd.DataFrame(rows)
    return loc


def check_floor(root):
    """every NLLoc run of the catalog used the configured floor (LOCTOPO_SURFACE of its nlloc.in)."""
    seen = set()
    nl = list((root / f"it{V.NPASS}").glob("*/nlloc.in")) + list((root / "deep").glob("*/nlloc.in")) + \
        list((root / "edge").glob("*/nlloc.in"))
    coh = sorted((root / "coh").glob("*/e*/nlloc.in"))
    nl += coh[:: max(1, len(coh) // 2000)]
    for f in nl:
        v = [l.split()[1] for l in f.read_text().splitlines() if l.startswith("LOCTOPO_SURFACE")]
        seen.update(v)
    if seen != {V.FLOOR}:
        raise SystemExit(f"NLLoc floors {seen}, expected {{{V.FLOOR}}} (D-B056/D-B068: one floor)")
    return len(nl)


def arrivals_check(root, chosen):
    """the arrivals-used accounting (a2_run_nlloc.account) has a row for every event of the final pass, and every deep or
    edge hypocentre has its row in that pass's arrivals_used.csv."""
    au = pd.read_csv(root / f"it{V.NPASS}" / "arrivals_used.csv")
    ev = pd.read_parquet(V.plan_dir(root) / "events.parquet")
    miss = sorted(set(ev.ID) - set(au.ID))
    if miss:
        raise SystemExit(f"{len(miss)} events of the plan without a row in it{V.NPASS}/arrivals_used.csv: {miss[:5]}")
    for k in ("deep", "edge"):
        ids = set(chosen.ID[chosen.run == k])
        if ids:
            a = pd.read_csv(root / k / "arrivals_used.csv")
            m = sorted(ids - set(a.ID))
            if m:
                raise SystemExit(f"{len(m)} {k}-pass hypocentres without a row in {k}/arrivals_used.csv: {m[:5]}")
    return au


REVIEW_DIR = os.environ.get("V6_REVIEW_DIR", "")      # ingest_review.py outputs (one-version v6.1); empty = D-B097
EVENT_FLAGS = os.environ.get("V6_EVENT_FLAGS", "")    # CSV ID, event_flag (e.g. Mount St Helens analog signal)


def review_columns(ids):
    r = pd.read_csv(REVIEW, usecols=["ID", "problem", "auto_indicator_rule_not_verdict"])
    g = r.groupby("ID").agg(review_problems=("problem", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
                            review_auto_indicators=("auto_indicator_rule_not_verdict",
                                                    lambda s: ";".join(sorted({x for v in s.dropna().astype(str)
                                                                               for x in v.split(";") if x}))))
    out = pd.DataFrame(index=pd.Index(ids, name="ID"))
    out = out.join(g)
    for k in ("review_problems", "review_auto_indicators"):
        out[k] = out[k].fillna("")
    for k in ("review_not_earthquake", "review_depth_unreliable", "review_relocate"):
        out[k] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["review_depth_km"] = np.nan
    if REVIEW_DIR:                                   # reviewer verdicts (ingest_review.py rules)
        d = Path(REVIEW_DIR)
        seen = set(pd.read_csv(d / "review_answers_merged.csv", dtype={"ID": "Int64"}).ID.dropna().astype(int))
        rev = out.index.isin(list(seen))
        for k, f in (("review_not_earthquake", "review_flag_not_earthquake.csv"),
                     ("review_depth_unreliable", "review_depth_unreliable.csv"), ("review_relocate", "review_relocate.csv")):
            lst = set(pd.read_csv(d / f, dtype={"ID": "Int64"}).ID.dropna().astype(int))
            out.loc[rev, k] = out.index[rev].isin(list(lst))
        rl = pd.read_csv(d / "review_relocate.csv", dtype=str, keep_default_na=False)
        one = rl[rl.corrected_depth_km.str.len().gt(0) & ~rl.corrected_depth_km.str.contains(";")]
        z = pd.Series(pd.to_numeric(one.corrected_depth_km).to_numpy(), index=one.ID.astype(int))
        out["review_depth_km"] = out.index.map(z)
    return out.reset_index()


MAG_TABLES = [("v5.1_converged", MAG_V51), ("v4.1_converged", MAG_V41)]   # default order (D-B091, D-B102)


def magnitude_columns(ids, tables=None):
    """magnitude columns from the first table (in order) with an M for the event; magnitude_source = its short label
    (catalog_version.MAG_TABLES, PR #58: v5.1_converged = v5.1 table on v5.1 hypocentres, v4.1_converged = v4.1 table
    on v4 hypocentres, v6.1_converged = computed on the v6.1 hypocentres, D-B105)."""
    out = pd.DataFrame({"ID": ids})
    got = pd.Series(False, index=out.index)
    src = pd.Series("", index=out.index, dtype=object)
    vals = {k: pd.Series(np.nan, index=out.index, dtype=object) for k in MAG_COLS}
    info = {}
    for lab, path in (tables or MAG_TABLES):
        m = pd.read_csv(path, low_memory=False)
        m = m.set_index("ID" if "ID" in m else "event_id")
        take = ~got & out.ID.map(m.M).notna()
        for k in MAG_COLS:
            if k in m:
                vals[k] = vals[k].where(~take, out.ID.map(m[k]))
        src[take] = lab
        got |= take
        info[lab] = dict(events=int(take.sum()), path=str(path), md5=V.md5(path))
    for k in MAG_COLS:
        v = vals[k]
        out[k] = v if v.dropna().map(lambda x: isinstance(x, str)).any() else pd.to_numeric(v)
    out["magnitude_source"] = src
    # v51 / v41 counts as before (acceptance text) plus every table
    info.update(v51=info.get("v5.1_converged", {}).get("events", 0), v41=info.get("v4.1_converged", {}).get("events", 0))
    return out, info


def pick_counts():
    p = pd.read_csv(PICKS, skipinitialspace=True, usecols=["event_ID", "timing_corrected", "clock_correction_s"],
                    dtype={"timing_corrected": str})
    tc = p.timing_corrected.astype(str).str.strip().str.lower().eq("true")
    cc = pd.to_numeric(p.clock_correction_s, errors="coerce").fillna(0).ne(0)
    g = pd.DataFrame({"ID": p.event_ID, "rt": tc, "cc": cc}).groupby("ID").sum()
    return g.rt.astype(int), g.cc.astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mag-tables", help="LABEL=CSV,... ordered magnitude tables (default v5.1_converged, v4.1_converged)")
    a = ap.parse_args()
    tables = [tuple(x.split("=", 1)) for x in a.mag_tables.split(",")] if a.mag_tables else None
    root, out = Path(a.root), Path(a.out)
    t0 = time.time()
    plan = json.loads((V.plan_dir(root) / "plan.json").read_text())
    ev = pd.read_parquet(V.plan_dir(root) / "events.parquet").set_index("ID")
    loc = read_runs(root)
    loc["ok"] = loc.status.eq("LOCATED")
    loc["prio"] = [PRIO.get((r, o), -1) for r, o in zip(loc.run, loc.ok)]
    it4 = loc[loc.run == "it4"]
    if it4.ID.duplicated().any():
        raise SystemExit("it4: several rows per event")
    chosen = loc.sort_values(["ID", "prio"]).drop_duplicates("ID", keep="last")
    chosen = chosen[chosen.prio >= 0].set_index("ID")
    nfl = check_floor(root)
    arrivals_check(root, chosen.reset_index())
    # coherence: events the stage ran on, partners
    coh_tab = (pd.read_csv(root / "coh" / "coherence_events.csv").set_index("ID")
               if (root / "coh" / "coherence_events.csv").exists() else pd.DataFrame(columns=["partners_compact"]))
    v4 = PH.read_v4()
    c = v4[["ID", "raw_ID", "v4_time", "v4_lat", "v4_lon", "v4_depth", "v4_npick", "v4_sigma_m"]].copy()
    L = chosen[chosen.ok].copy()
    with ProcessPoolExecutor(16) as ex:
        roots = L.root.tolist()
        parts = [roots[i:i + 2000] for i in range(0, len(roots), 2000)]
        st = [r for part in ex.map(_phase_many, parts) for r in part]
    L["rms_unweighted"] = [s[0] for s in st]
    L["n_used_w"] = [s[1] for s in st]
    cols = ["time", "lat", "lon", "depth", "exp_depth", "nphs", "n_sta", "gap", "rms", "h_min", "h_max", "sig_z",
            "ell_len3", "dist_min", "dist_med", "dist_max", "ns_z", "run", "workdir", "rms_unweighted"]
    c = c.merge(L[cols], left_on="ID", right_index=True, how="left")
    nll = c.time.notna()
    c["loc_source"] = np.where(nll, "nll3d", "none")
    c["depth_flat"] = c.depth
    for k in ("depth", "exp_depth"):
        c.loc[nll, k] = BC.unflatten(c.loc[nll, k])
    c["surface_km"] = np.nan
    c.loc[nll, "surface_km"] = BC.surface_km(c.lat[nll], c.lon[nll])
    # tiers: a2_build_catalog.tier on the run-frame depth (as v5.1), merged surface
    tt = c[nll].assign(depth=c.depth_flat[nll])
    t, ratio = tier(tt)
    c["tier"] = "U"
    c.loc[nll, "tier"] = t
    c["dist_over_depth"] = np.nan
    c.loc[nll, "dist_over_depth"] = ratio.to_numpy()
    c["above_surface"] = pd.array([pd.NA] * len(c), dtype="boolean")
    c.loc[nll, "above_surface"] = (c.depth[nll] < -c.surface_km[nll]).to_numpy()
    c["shift_h_km"] = np.where(nll, km_h(c.lat, c.lon, c.v4_lat, c.v4_lon), np.nan)
    c["shift_z_km"] = np.where(nll, c.depth - c.v4_depth, np.nan)
    c["tile"] = c.ID.map(ev.tile).where(nll, "")
    # why not located (a2_build_catalog.py wording)
    i4 = it4.set_index("ID")
    com = i4.status.str.lower() + ": " + i4.comment.fillna("").astype(str)
    nobs = c.ID.map(ev.n_obs)
    why = np.select([c.ID.isin(com.index), nobs.isna(), nobs < 5],
                    [c.ID.map(com).fillna(""), "not in the run's observation files", "fewer than 5 picks"],
                    "too few usable arrivals")
    c["u_reason"] = np.where(nll, "", why)
    c["region"] = [PH.region(x, y) for x, y in zip(c.v4_lat, c.v4_lon)]
    c["depth_frame"] = np.where(nll, "true depth (EFT undone, b4_v6_catalog.py)", "")
    q, f = BC.flags(c.assign(time=c.time.fillna("1970-01-01T00:00:00")))
    c["depth_quality"] = np.where(nll, q, "")
    c["event_flag"] = np.where(nll, f, "")
    if EVENT_FLAGS:                                  # extra flags (ID, event_flag), appended; no event removed
        ef = pd.read_csv(EVENT_FLAGS, dtype={"event_flag": str}).groupby("ID").event_flag.agg(lambda s: ";".join(sorted(set(s))))
        add = c.ID.map(ef).fillna("")
        c["event_flag"] = [";".join(x for x in (a_, b_) if x) for a_, b_ in zip(c.event_flag, add)]
    # v5.1 flags
    fl = BC.read_floor(V.FLOOR)
    h_fl = c.depth_flat + BC.floor_at(fl, c.lat.fillna(0), c.lon.fillna(0)) / 1e3
    n_above_floor = int((h_fl[nll] < 0).sum())
    flags = {"floor_limited": (h_fl < BC.FLOOR_CELLS * c.ns_z),
             "far_only": c.ID.map(ev.n_fine_base).fillna(0).eq(0),
             "deep_pass": c.run.eq("deep"), "edge_pass": c.run.eq("edge"), "five_phase": c.nphs.eq(5)}
    for k, v in flags.items():
        c[k] = pd.array([pd.NA] * len(c), dtype="boolean")
        c.loc[nll, k] = v[nll].to_numpy()
    # v6.0 columns
    c["model_version"] = f"v{V.VERSION}"
    c["loc_pass"] = c.run.map(LOC_PASS).fillna("")
    ssst = c.run.isin(["it4", "coh"])
    c["station_terms"] = np.where(ssst, "ssst", np.where(nll, "static", ""))
    c["ssst_char_km"] = np.where(ssst, V.SCHEDULE[-1], np.nan)
    rt, cc = pick_counts()
    c["n_picks_retimed"] = c.ID.map(rt).fillna(0).astype(int)
    c["n_picks_clock_corrected"] = c.ID.map(cc).fillna(0).astype(int)
    v51 = pd.read_csv(V51, usecols=["ID", "lat", "lon", "depth", "loc_source", "tier"]).set_index("ID")
    both = nll & c.ID.map(v51.loc_source).eq("nll3d")
    c["shift_h_v51_km"] = np.where(both, km_h(c.lat, c.lon, c.ID.map(v51.lat), c.ID.map(v51.lon)), np.nan)
    c["shift_z_v51_km"] = np.where(both, c.depth - c.ID.map(v51.depth), np.nan)
    c["coherence_relocated"] = pd.array([pd.NA] * len(c), dtype="boolean")
    c.loc[nll, "coherence_relocated"] = c.run[nll].eq("coh").to_numpy()
    c["coherence_partners"] = c.ID.map(coh_tab.partners_compact).astype("Int64") if len(coh_tab) else pd.NA
    skf = root / "coh" / "coherence_skipped_region.csv"     # D-B098: compact events not coherence-relocated (Mendocino)
    sk = set(pd.read_csv(skf).ID) if skf.exists() else set()
    c["coherence_skipped_region"] = c.ID.isin(sk)
    c = c.merge(review_columns(c.ID), on="ID", how="left")
    mags, minfo = magnitude_columns(c.ID, tables)
    c = c.merge(mags, on="ID", how="left")
    # v4.1 additions NLLoc did not locate: left out (catalog_version.py adds them with the ComCat hypocentre)
    drop_add = c.ID.isin(BC.CV["v4_1_added"]) & ~nll
    c = c[~drop_add].reset_index(drop=True)
    nll = c.loc_source == "nll3d"
    keep = ["ID", "raw_ID", "v4_time", "v4_lat", "v4_lon", "v4_depth", "v4_npick", "v4_sigma_m", "time", "lat", "lon",
            "depth", "exp_depth", "loc_source", "tier", "nphs", "n_sta", "gap", "rms", "rms_unweighted", "h_min", "h_max",
            "sig_z", "ell_len3", "dist_min", "dist_med", "dist_max", "dist_over_depth", "surface_km", "above_surface",
            "shift_h_km", "shift_z_km", "shift_h_v51_km", "shift_z_v51_km", "tile", "u_reason", "region", "depth_frame",
            "depth_quality", "event_flag", "floor_limited", "far_only", "deep_pass", "edge_pass", "five_phase",
            "model_version", "loc_pass", "station_terms", "ssst_char_km", "coherence_relocated", "coherence_partners", "coherence_skipped_region",
            "n_picks_retimed", "n_picks_clock_corrected", "review_problems", "review_auto_indicators",
            "review_not_earthquake", "review_depth_unreliable", "review_relocate", "review_depth_km"] + MAG_COLS + \
           ["magnitude_source"]
    c = c[keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(out, index=False, float_format="%.8g")
    ab = c.above_surface.fillna(False).astype(bool)
    counts = dict(events=len(c), located=int(nll.sum()), none=int((~nll).sum()), v4_1_added_left_out=int(drop_add.sum()),
                  above_surface=int(ab.sum()), above_floor_file=n_above_floor,
                  loc_pass={k: int(v) for k, v in c.loc_pass[nll].value_counts().items()},
                  tiers={k: int(v) for k, v in c.tier.value_counts().sort_index().items()},
                  flags={k: int(c[k].fillna(False).astype(bool).sum()) for k in
                         ("floor_limited", "far_only", "deep_pass", "edge_pass", "five_phase", "coherence_relocated")},
                  depth_quality={k: int(v) for k, v in c.depth_quality.value_counts().items()},
                  with_magnitude=int(c.M.notna().sum()), magnitudes=minfo,
                  review_rows=int((c.review_problems != "").sum()),
                  u_reason={k: int(v) for k, v in c.u_reason[~nll].str.replace(
                      r"rejected: .*boundary (\d+).*", r"rejected: max likelihood on box face \1", regex=True).value_counts().items()},
                  nlloc_in_checked=nfl, publishable=int(ab.sum()) == 0)
    man = dict(catalog=str(out), catalog_md5=V.md5(out), root=str(root), plan=plan, model=f"v{V.VERSION}",
               grid_manifest=str(V.GRID_MD5), grid_manifest_md5=V.grid_manifest_md5(), floor=V.FLOOR,
               floor_md5=V.md5(V.FLOOR), picks=str(PICKS), surface=BC.SURFACE, surface_source=BC.surface_source(),
               v51=str(V51), review=str(REVIEW), review_verdicts=REVIEW_DIR or None, event_flags=EVENT_FLAGS or None,
               event_flags_md5=V.md5(EVENT_FLAGS) if EVENT_FLAGS else None, counts=counts, wall_min=round((time.time() - t0) / 60, 1),
               runs={k: json.loads((root / k / "run.json").read_text()) for k in
                     [f"it{i}" for i in range(V.NPASS + 1)] + ["coh", "deep", "edge"] if (root / k / "run.json").exists()},
               loc2ssst={f"g{i}": json.loads((root / f"g{i}" / "summary.json").read_text())
                         for i in range(V.NPASS) if (root / f"g{i}" / "summary.json").exists()},
               utc=time.strftime("%FT%T%z"))
    out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"catalog {out}: {json.dumps(counts, default=str)}")


if __name__ == "__main__":
    main()
