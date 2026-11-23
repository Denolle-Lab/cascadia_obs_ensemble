"""v6 gap fill: pick a station-day and measure the Route A amplitudes of its picks in the same task.

The v5 fill picked first and measured amplitudes two weeks later in a second pass over all
picks (route_a_day_amplitudes.py). Here one worker does both for each station-day:

  1. picks with option (a), the v3/v4 model set, exactly as the v5 offshore fill
     (dryrun_models.pick_day, configs a);
  2. measures every new pick with the Route A day engine of route_a_day_amplitudes.py
     (run_task: one sensor, response removed on each contiguous segment, Wood-Anderson
     1 Hz high-pass and 0.05-2 Hz displacement, running maxima over the WINDOWS grid).

The picker reads the day with the v5 channel rule (HH > BH > EH, 4-15 Hz); the amplitude
engine reads it again with the Route A sensor rule (the best sensor with a response). The
second read costs ~0.2 s from pnwstore, so the two validated code paths are kept as they are.

Subcommands:
  tasks      station-days in pnwstore inside a lon/lat/year box that have no pick in ver5
  inventory  StationXML with responses for the stations of the task list (EarthScope FDSN)
  run        pick + amplitudes, resumable; OUT/picks/NET.STA.DAY.csv, OUT/amps/NET.STA.DAY.parquet
             (amplitude rows in the order of the pick rows, `row` = 0..n-1; pick_id is
             assigned when the v6 table is built), OUT/log.csv, OUT/DONE

    python workflow/01_picking/pick_and_amps.py tasks --out DIR \\
        --lon -123 -122 --lat 40 46 --years 2010 2016
    python workflow/01_picking/pick_and_amps.py inventory --out DIR
    python workflow/01_picking/pick_and_amps.py run --out DIR --workers 40

scale/run_pick_amps.sh drives `run` with a hang watchdog.
"""
import argparse
import os
import sqlite3
import sys
import time
import traceback
import types
import urllib.request
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "workflow", "05_magnitude"))

INDEX = "/wd1/PNWstore_sqlite/{year}.sqlite"
VER5_DAYS = "/wd1/mdenolle_data/amps_v5/tasks.csv"     # every ver5 station-day (pick-time day)
FDSN = "https://service.earthscope.org/fdsnws/station/1/query"
AMP_ARGS = types.SimpleNamespace(prefer_complete=True, highpass=1.0, disp_lo=0.05, disp_hi=2.0)
LOG_COLS = ["network", "station", "day", "status", "band", "z_copied", "n_a", "amp_status",
            "amp_sensor", "n_amp_ok", "t_pre", "t_infer", "t_semb", "t_amp", "t_total", "finished"]


# ---------------------------------------------------------------------------- tasks
def station_coords(nets):
    url = (f"{FDSN}?net={','.join(sorted(nets))}&starttime=2010-01-01&endtime=2017-01-01"
           "&level=station&format=text")
    with urllib.request.urlopen(url, timeout=300) as r:
        lines = r.read().decode().splitlines()[1:]
    rows = [ln.split("|")[:5] for ln in lines if ln.strip()]
    df = pd.DataFrame(rows, columns=["network", "station", "lat", "lon", "elev"])
    df[["lat", "lon", "elev"]] = df[["lat", "lon", "elev"]].astype(float)
    return df.drop_duplicates(["network", "station"], keep="last")


def cmd_tasks(a):
    rows = []
    for y in range(a.years[0], a.years[1] + 1):
        con = sqlite3.connect(INDEX.format(year=y))
        rows += con.execute(
            "SELECT DISTINCT network, station, substr(starttime,1,10) FROM tsindex "
            "WHERE (channel LIKE 'HH_' OR channel LIKE 'BH_' OR channel LIKE 'EH_') "
            "AND (channel LIKE '__Z' OR channel LIKE '__3')").fetchall()
    sd = pd.DataFrame(rows, columns=["network", "station", "day"]).drop_duplicates()
    sd = sd[sd.day.str[:4].astype(int).between(*a.years)]
    co = station_coords(set(sd.network))
    box = co[co.lon.between(*a.lon) & co.lat.between(*a.lat)]
    sd = sd.merge(box, on=["network", "station"])
    v5 = pd.read_csv(VER5_DAYS, usecols=["network", "station", "task"])
    v5 = set(zip(v5.network, v5.station, v5.task.str[-10:]))
    sd["in_ver5"] = [k in v5 for k in zip(sd.network, sd.station, sd.day)]
    os.makedirs(a.out, exist_ok=True)
    sd.to_csv(os.path.join(a.out, "candidates.csv"), index=False)
    todo = sd[~sd.in_ver5].sample(frac=1, random_state=0)        # random order, as the v5 fill
    todo[["network", "station", "day"]].to_csv(os.path.join(a.out, "tasks.csv"), index=False)
    print(f"box lon {a.lon} lat {a.lat} years {a.years}: {len(sd):,} station-days in pnwstore, "
          f"{sd.in_ver5.sum():,} already in ver5, {len(todo):,} to do at "
          f"{todo[['network', 'station']].drop_duplicates().shape[0]} stations")
    print(todo.groupby([todo.day.str[:4], "network"]).size().unstack(0, fill_value=0).to_string())


# ------------------------------------------------------------------------ inventory
def cmd_inventory(a):
    from obspy import Inventory, read_inventory
    from obspy.clients.fdsn import Client

    t = pd.read_csv(os.path.join(a.out, "tasks.csv"), dtype=str)
    pairs = sorted(set(zip(t.network, t.station)))
    cl = Client("EARTHSCOPE", timeout=300)
    inv = Inventory(networks=[], source="EarthScope FDSN")
    for net in sorted({n for n, _ in pairs}):
        stas = ",".join(s for n, s in pairs if n == net)
        for k in range(4):
            try:
                inv += cl.get_stations(network=net, station=stas, channel="?H?,?L?,?N?,?P?",
                                       starttime="2010-01-01", endtime="2017-01-01",
                                       level="response")
                break
            except Exception as e:
                print(f"{net}: try {k + 1}: {e}", flush=True)
                time.sleep(30)
    n0 = sum(len(s.channels) for n in inv for s in n)
    for net in inv:                    # slim_inventory.py's filter
        for s in net:
            s.channels = [c for c in s.channels if len(c.code) == 3 and c.code[1] in "HLNP"
                          and (c.sample_rate or 0) >= 10 and c.response is not None]
    have = {(n.code, s.code) for n in inv for s in n if s.channels}
    missing = [p for p in pairs if p not in have]
    out = os.path.join(a.out, "station_inventory.xml")
    inv.write(out, format="STATIONXML")
    read_inventory(out)                # parses back
    print(f"{len(pairs)} stations; channel-epochs {n0} -> "
          f"{sum(len(s.channels) for n in inv for s in n)}; no response: {missing}; wrote {out}")


# ------------------------------------------------------------------------------ run
_reader = _inv = None


def init_worker(inventory):
    global _reader, _inv
    import dryrun_models as dm
    from obspy import read_inventory
    from route_a_day_amplitudes import DayReader
    dm.init_worker()
    _inv = read_inventory(inventory)
    _reader = DayReader(_inv)


def run_one(task):
    import dryrun_models as dm
    from route_a_day_amplitudes import DATA_COLUMNS, run_task
    net, sta, day, outdir = task
    rec = {"network": net, "station": sta, "day": day, "status": "ok"}
    t0 = time.time()
    try:
        df = dm.pick_day(net, sta, day, rec)
        if df is not None and len(df) == 0:
            rec["status"] = "no_picks"
        elif df is not None:
            df = df.drop(columns="config").reset_index(drop=True)
            fp = os.path.join(outdir, "picks", f"{net}.{sta}.{day}.csv")
            fa = os.path.join(outdir, "amps", f"{net}.{sta}.{day}.parquet")
            t = time.time()
            p = pd.DataFrame({"pick_id": np.arange(len(df)), "label": df.label.astype(str),
                              "t": pd.to_datetime(df.pick_time.astype(str), utc=True)
                                     .astype("int64").to_numpy() / 1e9})
            res, why, sensor = run_task(_reader, _inv, net, sta,
                                        int(pd.Timestamp(day).value // 86400e9), p, AMP_ARGS)
            res = res[DATA_COLUMNS].rename(columns={"pick_id": "row"})
            rec.update(amp_status=str(why).replace(",", ";")[:80], amp_sensor=sensor,
                       n_amp_ok=int((res.status == "ok").sum()),
                       t_amp=round(time.time() - t, 1))
            # amplitudes first, then picks: a station-day counts as done once its picks exist
            res.to_parquet(fa + ".tmp", index=False, compression="zstd")
            os.replace(fa + ".tmp", fa)
            df.to_csv(fp + ".tmp", index=False)
            os.replace(fp + ".tmp", fp)
    except Exception as e:  # keep going; the log records why
        rec["status"] = f"error: {type(e).__name__}: {e}"[:300]
        traceback.print_exc()
    rec["t_total"] = round(time.time() - t0, 1)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def cmd_run(a):
    import dryrun_models as dm
    dm.CONFIGS = {"a": dm.CONFIGS["a"]}           # option (a) only, as the v5 fill
    dm.MODELS = [m for m in dm.MODELS if m in dm.CONFIGS["a"]]
    inventory = a.inventory or os.path.join(a.out, "station_inventory.xml")
    tasks = pd.read_csv(os.path.join(a.out, "tasks.csv"), dtype=str)
    logf = os.path.join(a.out, "log.csv")
    done = set()
    if os.path.exists(logf):
        lg = pd.read_csv(logf, dtype=str)
        done = set(zip(lg.network, lg.station, lg.day))
    todo = [(r.network, r.station, r.day, a.out) for r in tasks.itertuples()
            if (r.network, r.station, r.day) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} station-days to do ({len(done)} done)", flush=True)
    for d in ("picks", "amps"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    new = not os.path.exists(logf)
    with Pool(a.workers, initializer=init_worker, initargs=(inventory,),
              maxtasksperchild=a.max_tasks) as pool, open(logf, "a") as f:
        if new:
            f.write(",".join(LOG_COLS) + "\n")
        for k, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            f.write(",".join(str(rec.get(c, "")).replace(",", ";") for c in LOG_COLS) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"{time.strftime('%H:%M:%S')} {k}/{len(todo)}", flush=True)
    if not a.limit:
        open(os.path.join(a.out, "DONE"), "w").close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tasks")
    p.add_argument("--out", required=True)
    p.add_argument("--lon", type=float, nargs=2, required=True)
    p.add_argument("--lat", type=float, nargs=2, required=True)
    p.add_argument("--years", type=int, nargs=2, default=[2010, 2016])
    p = sub.add_parser("inventory")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--inventory", default=None, help="default: OUT/station_inventory.xml")
    p.add_argument("--max-tasks", type=int, default=50, help="tasks per worker before it is replaced")
    p.add_argument("--limit", type=int, default=0, help="at most N station-days (tests; no DONE)")
    a = ap.parse_args()
    {"tasks": cmd_tasks, "inventory": cmd_inventory, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
