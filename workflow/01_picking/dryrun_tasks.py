"""Station-days for the v5 base-model dry run (V5_PLAN §7), written to a tasks csv.

Three groups, sampled with a fixed seed:
  seafloor_picked    250 seafloor station-days with v3 catalog arrivals, spread over years
                     and networks; 7D station-days in the OBSTransformer training set
                     (Niksejel & Zhang 2024, SeisBench dataset OBST2024) are excluded
  seafloor_unpicked   50 seafloor station-days of stations never picked in v3 (pnwstore index)
  land               100 land station-days with v3 arrivals (NC/BK excluded: not in pnwstore)

    python workflow/01_picking/dryrun_tasks.py --obst-meta obst2024_metadata.csv --out tasks.csv
"""
import argparse
import sqlite3

import pandas as pd

ROOT = __file__.rsplit("/workflow/01_picking/", 1)[0]
ARRIVALS = f"{ROOT}/data/catalog_v3/arrivals_v3.csv"
STATIONS = f"{ROOT}/data/station_coverage_review_2026-09-30.csv"
INDEX = "/wd1/PNWstore_sqlite/{year}.sqlite"
YEARS = range(2010, 2016)


def training_station_days(meta):
    m = pd.read_csv(meta, usecols=["station_network_code", "station_code", "trace_start_time"])
    m = m[m.station_network_code == "7D"]
    day = pd.to_datetime(m.trace_start_time, utc=True).dt.strftime("%Y-%m-%d")
    return set(zip(m.station_code, day))


def sample(df, n, seed, by):
    """About n rows, spread evenly over the groups in ``by``."""
    groups = df.groupby(by)
    k = max(1, n // groups.ngroups)
    out = groups.apply(lambda g: g.sample(min(len(g), k), random_state=seed)).reset_index(drop=True)
    if len(out) < n:
        extra = df[~df.set_index(["network", "station", "day"]).index.isin(
            out.set_index(["network", "station", "day"]).index)]
        out = pd.concat([out, extra.sample(min(n - len(out), len(extra)), random_state=seed)])
    return out.sample(min(n, len(out)), random_state=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obst-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sta = pd.read_csv(STATIONS)
    elev = sta.groupby(["network", "station"]).elev.first()
    picked = sta.groupby(["network", "station"]).ver3_picks.max()

    arr = pd.read_csv(ARRIVALS, usecols=["network", "station", "time"])
    arr["day"] = arr.time.str[:10]
    sd = arr.groupby(["network", "station", "day"]).size().rename("n_arr").reset_index()
    sd = sd[~sd.network.isin(["NC", "BK"])]
    sd["elev"] = [elev.get((n, s)) for n, s in zip(sd.network, sd.station)]
    sd = sd.dropna(subset=["elev"])
    sd["year"] = sd.day.str[:4]

    train = training_station_days(a.obst_meta)
    sea = sd[sd.elev < 0]
    in_train = [(n == "7D" and (s, d) in train) for n, s, d in zip(sea.network, sea.station, sea.day)]
    print(f"seafloor station-days with arrivals: {len(sea)}, in OBST2024 training: {sum(in_train)} (excluded)")
    sea = sea[[not x for x in in_train]]
    g1 = sample(sea, 250, a.seed, ["network", "year"]).assign(group="seafloor_picked")
    g3 = sample(sd[sd.elev >= 0], 100, a.seed, ["year"]).assign(group="land")

    never = [k for k, v in picked.items() if v == 0 and elev.get(k, 1) < 0]
    rows = []
    for y in YEARS:
        con = sqlite3.connect(INDEX.format(year=y))
        q = ("SELECT DISTINCT network, station, substr(starttime,1,10) FROM tsindex "
             "WHERE (channel LIKE '_HZ' OR channel LIKE '_H3') AND network IN ({})").format(
            ",".join(f"'{n}'" for n in sorted({k[0] for k in never})))
        rows += [r for r in con.execute(q) if (r[0], r[1]) in set(never)]
    un = pd.DataFrame(rows, columns=["network", "station", "day"]).drop_duplicates()
    un["year"] = un.day.str[:4]
    print(f"seafloor station-days never picked in v3 (pnwstore index): {len(un)}")
    g2 = sample(un, 50, a.seed, ["network"]).assign(group="seafloor_unpicked", n_arr=0)

    out = pd.concat([g1, g2, g3])[["network", "station", "day", "group", "n_arr"]]
    out.to_csv(a.out, index=False)
    print(out.groupby("group").agg(n=("day", "size"), stations=("station", "nunique")))
    print(out.groupby(["group", "network"]).size().to_string())


if __name__ == "__main__":
    main()
