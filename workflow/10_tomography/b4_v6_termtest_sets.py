"""Track B, catalog v6.0 station-term test (PI, 2026-11-25 14:29): three independent test clusters.

The PI asked for a decisive test between two station-term recipes for the v6.0 NonLinLoc relocation:
(A) static station terms (as v5.1) + the NLL-coherence stage (D-B093), (B) NLL-SSST with the CharDist schedule
stopped at 2 km + the coherence stage, on three clusters independent of the four D-B078 sets (axial, blanco, msh,
jdf of /wd1/mdenolle_data/tomo_v1/b5/cc_test), taken from the v5.1 clustering (events_zbz_v51.csv.gz,
families_zbz_v51.csv) and chosen like -18's sets (b5_cc_sets.py): the core events nearest the median epicentre.

Sets (located v5.1 events, loc_source nll3d, any tier, as -18's sets):
* puget  - onshore, Puget Lowland (Kitsap Peninsula / Hood Canal, deep crust ~17 km): the densest v5.1 spot of the
           Puget/Georgia region (most epicentres within 8 km); ZBZ families there are small (largest 1027, 205
           events) so the cluster is spatial: the N_PUGET events of the region nearest the median epicentre of the
           events within 10 km of the peak (families 1027, 2048, ... and background events);
* mtj    - Mendocino triple junction, mixed NC/BK/PB land and 7D/Z5 OBS: ZBZ family 1536 (2015-01-28 M5.7
           offshore Cape Mendocino and its aftershocks, Cascadia Initiative year 4 OBS on the seafloor), the
           N_MTJ family events nearest the family's median epicentre;
* blancow - offshore OBS, a second Blanco segment (western Blanco, WBD/EBD of Lange et al. 2026; -18's blanco set
           is at -128.7 E): ZBZ family 427 (2012-10 swarm, Blanco OBS experiment X9 + 7D), the N_BLANCOW family
           events nearest the family's median epicentre.
Pairs (for the CC "given" pairs, as b5_cc_sets.py): the catalog differential-time pairs among the set's events in
the B5 prep (b5_v5.1_s3/prep/pairs.parquet: 10 nearest neighbours within 10 km, >= 8 common station-phases).

  python b4_v6_termtest_sets.py   -> /wd1/mdenolle_data/reloc_v6/termtest/sets/{events,pairs}_<set>.csv, README.md
  python b4_v6_termtest_sets.py --group 2   -> georgia, gorda (stopping-distance test), README2.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TT = Path("/wd1/mdenolle_data/reloc_v6/termtest")
OUT = TT / "sets"
CLU = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/clusters/events_zbz_v51.csv.gz")
CAT = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
PREP = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v5.1_s3/prep")
OLD = Path("/wd1/mdenolle_data/tomo_v1/b5/cc_test")
N_PUGET, N_MTJ, N_BLANCOW = 1000, 600, 1200
PUGET_PEAK_R_KM, PUGET_POOL_R_KM = 8.0, 10.0
SETS = ("puget", "mtj", "blancow")


def epi(lat, lon, la0, lo0):
    return np.hypot((lat - la0) * 111.195, (lon - lo0) * 111.195 * np.cos(np.radians(la0)))


def nearest(pool, n):
    la0, lo0 = pool.lat.median(), pool.lon.median()
    d = epi(pool.lat.values, pool.lon.values, la0, lo0)
    o = np.argsort(d, kind="stable")[:n]
    s = pool.iloc[np.sort(o)].copy()
    s["d_med_km"] = d[np.sort(o)]
    return s, (la0, lo0)


def main():
    from scipy.spatial import cKDTree
    OUT.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(CLU)
    c = pd.read_csv(CAT, usecols=["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region"])
    c = c[c.loc_source == "nll3d"]
    e = e[["event_id", "family_id", "cls", "region_v51"]].rename(columns={"event_id": "ID"}).merge(c, on="ID")
    old = set()
    for s in ("axial", "blanco", "msh", "jdf"):
        old |= set(pd.read_csv(OLD / f"events_{s}.csv").ID)
    info = {}
    # puget: density peak of the Puget/Georgia region, then the N nearest the pool's median epicentre
    p = e[e.region_v51 == "Puget/Georgia"].reset_index(drop=True)
    xy = np.c_[(p.lon + 123) * 111.195 * np.cos(np.radians(47.5)), p.lat * 111.195]
    t = cKDTree(xy)
    cnt = np.array([len(v) for v in t.query_ball_point(xy, PUGET_PEAK_R_KM)])
    k = int(np.argmax(cnt))
    la_pk, lo_pk = float(p.lat[k]), float(p.lon[k])
    pool = p[epi(p.lat.values, p.lon.values, la_pk, lo_pk) <= PUGET_POOL_R_KM]
    sets = {}
    d = epi(p.lat.values, p.lon.values, pool.lat.median(), pool.lon.median())
    o = np.sort(np.argsort(d, kind="stable")[:N_PUGET])
    sets["puget"] = p.iloc[o].assign(d_med_km=d[o])
    info["puget"] = dict(rule=f"Puget/Georgia density peak ({PUGET_PEAK_R_KM:g} km count {int(cnt[k])} at "
                              f"{la_pk:.3f}, {lo_pk:.3f}); {N_PUGET} region events nearest the median epicentre of "
                              f"the {len(pool)} events within {PUGET_POOL_R_KM:g} km")
    f = e[e.family_id == 1536]
    sets["mtj"], med = nearest(f, N_MTJ)
    info["mtj"] = dict(rule=f"ZBZ family 1536 ({len(f)} located events): the {N_MTJ} nearest its median epicentre")
    f = e[e.family_id == 427]
    sets["blancow"], med = nearest(f, N_BLANCOW)
    info["blancow"] = dict(rule=f"ZBZ family 427 ({len(f)} located events): the {N_BLANCOW} nearest its median "
                                f"epicentre")
    pev = pd.read_parquet(PREP / "events.parquet", columns=["ID"])
    pr = pd.read_parquet(PREP / "pairs.parquet")
    pr["ID1"], pr["ID2"] = pev.ID.values[pr.e1.values], pev.ID.values[pr.e2.values]
    L = ["# Station-term test sets (b4_v6_termtest_sets.py)", "",
         "| set | rule | events | pairs | lat | lon | depth median (km) | radius (km) | tiers | families |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        ev = sets[s]
        assert not (set(ev.ID) & old), f"{s}: events shared with the D-B078 sets"
        ids = set(ev.ID)
        pp = pr[pr.ID1.isin(ids) & pr.ID2.isin(ids)]
        ev[["ID", "time", "lat", "lon", "depth", "region", "tier"]].to_csv(OUT / f"events_{s}.csv", index=False,
                                                                          float_format="%.8g")
        pp[["ID1", "ID2", "n_dt", "sep_km"]].to_csv(OUT / f"pairs_{s}.csv", index=False, float_format="%.6g")
        fam = ev.family_id.value_counts().head(4).to_dict()
        info[s].update(events=len(ev), pairs=len(pp), lat=float(ev.lat.median()), lon=float(ev.lon.median()),
                       depth=float(ev.depth.median()), radius_km=float(ev.d_med_km.max()),
                       tiers=ev.tier.value_counts().sort_index().to_dict(), families={int(a): int(b) for a, b in fam.items()},
                       time_min=str(ev.time.min()), time_max=str(ev.time.max()))
        L.append(f"| {s} | {info[s]['rule']} | {len(ev):,} | {len(pp):,} | {ev.lat.median():.3f} | {ev.lon.median():.3f} "
                 f"| {ev.depth.median():.2f} | {ev.d_med_km.max():.1f} | {info[s]['tiers']} | {fam} |")
    L += ["", "Events: ID, v5.1 origin time and hypocentre (km bsl), region, tier. Pairs: B5 prep catalog-dt pairs among "
          "them (10 nearest neighbours within 10 km, >= 8 common station-phases). No event is in the D-B078 sets "
          "(axial, blanco, msh, jdf). Family -1 = background (no ZBZ family)."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    (OUT / "sets.json").write_text(json.dumps(info, indent=1))
    print("\n".join(L))


# ------------------------------------------------------------------------------------------- group 2 (stop distance)
SETS2 = ("georgia", "gorda")
N_GEORGIA, N_GORDA = 700, 800
GEORGIA_PEAK = (48.624, -123.075)      # Strait of Georgia / Haro Strait density peak (8 km count 691, region Puget/Georgia)


def main2():
    """Confirmatory test of the SSST stopping distance (D-B095 follow-up): two more clusters sharing no event with
    axial, blanco, msh, jdf, puget, mtj, blancow.
    * georgia - onshore, Strait of Georgia / Haro Strait (CN, PB, UW, TA stations, ~15 km deep): the N_GEORGIA events
      of the Puget/Georgia region nearest the median epicentre of the events within 10 km of GEORGIA_PEAK (the
      densest spot north of 48.3 N below 5 km depth; the 49.1 N Fraser Valley spot is at 2 km depth and left out);
    * gorda - offshore, Gorda plate interior: ZBZ family 1162 (2014-03-10 M6.8 and aftershocks; Z5 and 7D OBS + NC,
      PB, BK, TA land stations), the N_GORDA family events nearest the family's median epicentre."""
    e = pd.read_csv(CLU)
    c = pd.read_csv(CAT, usecols=["ID", "time", "lat", "lon", "depth", "loc_source", "tier", "region"])
    c = c[c.loc_source == "nll3d"]
    e = e[["event_id", "family_id", "cls", "region_v51"]].rename(columns={"event_id": "ID"}).merge(c, on="ID")
    old = set()
    for s in ("axial", "blanco", "msh", "jdf"):
        old |= set(pd.read_csv(OLD / f"events_{s}.csv").ID)
    for s in SETS:
        old |= set(pd.read_csv(OUT / f"events_{s}.csv").ID)
    info = json.loads((OUT / "sets.json").read_text())
    sets = {}
    p = e[e.region_v51 == "Puget/Georgia"].reset_index(drop=True)
    pool = p[epi(p.lat.values, p.lon.values, *GEORGIA_PEAK) <= PUGET_POOL_R_KM]
    d = epi(p.lat.values, p.lon.values, pool.lat.median(), pool.lon.median())
    o = np.sort(np.argsort(d, kind="stable")[:N_GEORGIA])
    sets["georgia"] = p.iloc[o].assign(d_med_km=d[o])
    info["georgia"] = dict(rule=f"Strait of Georgia peak ({GEORGIA_PEAK[0]}, {GEORGIA_PEAK[1]}); {N_GEORGIA} region "
                                f"events nearest the median epicentre of the {len(pool)} events within "
                                f"{PUGET_POOL_R_KM:g} km")
    f = e[e.family_id == 1162]
    sets["gorda"], _ = nearest(f, N_GORDA)
    info["gorda"] = dict(rule=f"ZBZ family 1162 ({len(f)} located events): the {N_GORDA} nearest its median epicentre")
    pev = pd.read_parquet(PREP / "events.parquet", columns=["ID"])
    pr = pd.read_parquet(PREP / "pairs.parquet")
    pr["ID1"], pr["ID2"] = pev.ID.values[pr.e1.values], pev.ID.values[pr.e2.values]
    L = ["# Station-term test sets, group 2: SSST stopping distance (b4_v6_termtest_sets.py --group 2)", "",
         "| set | rule | events | pairs | lat | lon | depth median (km) | radius (km) | tiers | families |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETS2:
        ev = sets[s]
        assert not (set(ev.ID) & old), f"{s}: events shared with the earlier sets"
        ids = set(ev.ID)
        pp = pr[pr.ID1.isin(ids) & pr.ID2.isin(ids)]
        ev[["ID", "time", "lat", "lon", "depth", "region", "tier"]].to_csv(OUT / f"events_{s}.csv", index=False,
                                                                          float_format="%.8g")
        pp[["ID1", "ID2", "n_dt", "sep_km"]].to_csv(OUT / f"pairs_{s}.csv", index=False, float_format="%.6g")
        fam = ev.family_id.value_counts().head(4).to_dict()
        info[s].update(events=len(ev), pairs=len(pp), lat=float(ev.lat.median()), lon=float(ev.lon.median()),
                       depth=float(ev.depth.median()), radius_km=float(ev.d_med_km.max()),
                       tiers=ev.tier.value_counts().sort_index().to_dict(),
                       families={int(a): int(b) for a, b in fam.items()},
                       time_min=str(ev.time.min()), time_max=str(ev.time.max()), group=2)
        L.append(f"| {s} | {info[s]['rule']} | {len(ev):,} | {len(pp):,} | {ev.lat.median():.3f} | {ev.lon.median():.3f} "
                 f"| {ev.depth.median():.2f} | {ev.d_med_km.max():.1f} | {info[s]['tiers']} | {fam} |")
    L += ["", "No event is in axial, blanco, msh, jdf (D-B078) or puget, mtj, blancow (group 1)."]
    (OUT / "README2.md").write_text("\n".join(L) + "\n")
    (OUT / "sets.json").write_text(json.dumps(info, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=1)
    {1: main, 2: main2}[ap.parse_args().group]()
