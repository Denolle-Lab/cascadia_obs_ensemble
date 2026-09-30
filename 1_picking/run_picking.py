"""Run ELEP picking for one row of picking_config.csv (a year and a region).

Replaces the 24 parallel_pick_{year}[_region].py scripts that wrote the ver3
picks (now in legacy/). Each task is one station-day; its picks go to
``OUTROOT/picks_{year}_{region}/NET_STA_YYYYMMDD_YYYYMMDD.csv``. Station-days
already written are skipped, so the run can be restarted.

    python run_picking.py --year 2011 --region 122-129 --outroot /wd1/hbito_data/data
    python run_picking.py --year 2011 --region 123-127_EH --station UW.NLO --day 2011-02-11

Environment: pixi ``internal`` (ELEP, SeisBench, pnwstore).
"""
import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from obspy.clients.fdsn import Client

import elep_picker

HERE = os.path.dirname(os.path.abspath(__file__))
NETWORKS = "C8,7D,7A,CN,NV,UW,UO,NC,BK,TA,OO,PB,X6,Z5,X9"

Logger = logging.getLogger(__name__)


def load_config(year, region, path=os.path.join(HERE, 'picking_config.csv')):
    cfg = pd.read_csv(path, dtype={'region': str, 'skip_if_in': str}, keep_default_na=False)
    row = cfg[(cfg.year == year) & (cfg.region == region)]
    if len(row) != 1:
        raise SystemExit(f"no config row for year={year} region={region}; see {path}")
    return row.iloc[0]


def station_list(cfg, time1, time2):
    """Stations of NETWORKS in the region's box, operating during [time1, time2)."""
    inventory = Client('IRIS').get_stations(
        network=NETWORKS, station="*",
        minlatitude=cfg.minlat, maxlatitude=cfg.maxlat,
        minlongitude=cfg.minlon, maxlongitude=cfg.maxlon,
        starttime=time1.strftime('%Y%m%d'), endtime=time2.strftime('%Y%m%d'))
    return [(net.code, sta.code) for net in inventory for sta in net]


@delayed
def pick_station_day(network, station, day, outdir, cfg, skip_dir, source):
    t1 = day.to_pydatetime()
    t2 = t1 + datetime.timedelta(days=1)
    try:
        return elep_picker.run_detection(network, station, t1, t2, outdir,
                                         channel_mode=cfg.channel_mode, vertical=cfg.vertical,
                                         skip_if_in=skip_dir, source=source)
    except Exception as e:  # one bad station-day must not stop the year
        print(f"Error {network}.{station} {t1:%Y-%m-%d}: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--region', required=True, help="region column of picking_config.csv")
    p.add_argument('--outroot', default=os.path.join(HERE, '..', 'data'),
                   help="parent of the picks_{year}_{region}/ directories")
    p.add_argument('--station', help="NET.STA: pick only this station (testing)")
    p.add_argument('--day', help="YYYY-MM-DD: pick only this day (testing)")
    p.add_argument('--workers', type=int, help="override the config's worker count")
    p.add_argument('--source', default='pnwstore', choices=['pnwstore', 'fdsn'],
                   help="waveforms from the UW pnwstore archive (ver3) or public FDSN")
    args = p.parse_args()

    cfg = load_config(args.year, args.region)
    outdir = os.path.join(args.outroot, f"picks_{args.year}_{args.region}") + '/'
    skip_dir = (os.path.join(args.outroot, f"picks_{args.year}_{cfg.skip_if_in}") + '/'
                if cfg.skip_if_in else None)
    os.makedirs(outdir, exist_ok=True)

    time1 = datetime.datetime(year=args.year, month=1, day=1)
    time2 = datetime.datetime(year=args.year + 1, month=1, day=1)
    days = pd.to_datetime(np.arange(time1, time2, pd.Timedelta(1, 'days')))
    if args.day:
        days = pd.to_datetime([args.day])
    if args.station:
        stations = [tuple(args.station.split('.'))]
    else:
        stations = station_list(cfg, time1, time2)

    tasks = [pick_station_day(net, sta, d, outdir, cfg, skip_dir, args.source) for net, sta in stations for d in days]
    print(f"{args.year} {args.region}: {len(stations)} stations x {len(days)} days -> {outdir}")
    with ProgressBar():
        compute(tasks, scheduler='processes', num_workers=args.workers or int(cfg.workers))


if __name__ == "__main__":
    main()
