"""Track B, catalog v6.1 depth test (D-B100): the fourth test cluster, Gorda Depression (gordad).

The events of catalog v4 paired with Lange et al. (2026)'s absolute OBS locations (pair table of b4_lit_paired.py,
blanco_lange2026_abs_pairs.csv) whose published epicentre lies in the Gorda Depression segment (longitude -127.1 to
-126.0, the segment of b4_lit_paired.SEGS): 503 events, 2012-09 to 2013-10. Written in the layout b4_v6_heldout.SETS
expects:
* sets/events_gordad.csv   v5.1 rows (ID, time, lat, lon, depth, region, tier): box and fold seeds
* inputs/picks_gordad.csv  picks of the events from the D-B090 clock-corrected pick file (md5 checked)
* inputs/events_gordad.csv catalog v4 rows (a2_make_obs.py input)
* cc/cc_gordad.parquet     rows of the catalog-wide CC job (tomo_v1/cc_dt_all/cc_blanco.parquet) with both events in
                           the set (same code and columns as the other test sets)

  python b4_v61_depth_sets.py -> /wd1/mdenolle_data/reloc_v61/depth_test/{sets,inputs,cc}
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_v6_ssst as S  # noqa: E402

OUT = Path("/wd1/mdenolle_data/reloc_v61/depth_test")
PAIRS = Path("/wd1/mdenolle_data/tomo_v1/depth_literature/pairs/blanco_lange2026_abs_pairs.csv")
V51 = Path("/wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_b4_v51_iter1.csv")
CC_ALL = Path("/wd1/mdenolle_data/tomo_v1/cc_dt_all/cc_blanco.parquet")
LON0, LON1 = -127.1, -126.0


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    for d in ("sets", "inputs", "cc"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    la = pd.read_csv(PAIRS)
    ids = set(la[(la.ref_lon >= LON0) & (la.ref_lon < LON1)].ID)
    v51 = pd.read_csv(V51, usecols=["ID", "time", "lat", "lon", "depth", "region", "tier", "loc_source"])
    ev = v51[v51.ID.isin(ids)].drop(columns="loc_source").sort_values("ID")
    assert len(ev) == len(ids), f"{len(ids) - len(ev)} paired events missing from v5.1"
    ev.to_csv(OUT / "sets" / "events_gordad.csv", index=False)
    h = md5(S.PICKS)
    assert h == S.PICKS_MD5, f"{S.PICKS} md5 {h}, expected {S.PICKS_MD5}"
    p = pd.concat([ch[ch.event_ID.isin(ids)] for ch in pd.read_csv(S.PICKS, skipinitialspace=True,
                                                                     chunksize=500_000)])
    p.to_csv(OUT / "inputs" / "picks_gordad.csv", index=False)
    v4 = pd.read_csv(S.EVENTS, skipinitialspace=True)
    v4 = v4[v4.ID.isin(ids)]
    assert len(v4) == len(ids)
    v4.to_csv(OUT / "inputs" / "events_gordad.csv", index=False)
    t = ds.dataset(CC_ALL).to_table(filter=ds.field("ID1").isin(list(ids)) & ds.field("ID2").isin(list(ids)))
    cc = t.to_pandas()
    cc.to_parquet(OUT / "cc" / "cc_gordad.parquet", index=False)
    rec = dict(events=len(ev), picks=len(p), stations=int(p.station.nunique()), cc_rows=len(cc),
               cc_ok_above=int(((cc.status == "ok") & cc.above.astype(bool)).sum()),
               cc_pairs=int(cc[["ID1", "ID2"]].drop_duplicates().shape[0]), picks_md5=h, pairs=str(PAIRS),
               v51=str(V51), cc_source=str(CC_ALL), lon_range=[LON0, LON1])
    (OUT / "sets" / "gordad.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
