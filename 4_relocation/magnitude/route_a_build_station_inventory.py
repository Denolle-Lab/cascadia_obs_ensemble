#!/usr/bin/env python3
"""
Route A, step 0: build a station inventory (coordinates + instrument response +
operational epochs) for every network in the pick catalog.

The inventory serves two purposes downstream:
  * response removal for Wood-Anderson simulation (route_a_wa_amplitudes.py);
  * exposing per-station response *epochs* so ocean-bottom seismometers that were
    redeployed with different instruments under a reused station code get an
    epoch-specific station term in the inversion (the main Method-B soundness fix).

NC/BK are served by NCEDC (see utils/data_client.py); all others by the chosen FDSN
datacenter. Run on a host with FDSN + NCEDC reachable (any machine with internet).

Outputs:
  station_inventory.xml   StationXML (input to route_a_wa_amplitudes.py)
  station_epochs.csv      one row per (network, station, channel, epoch)

Usage:
  python route_a_build_station_inventory.py \
      --picks /path/Cascadia_updated_catalog_picks_assignment_ver_3.csv \
      --out-xml station_inventory.xml --out-csv station_epochs.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from obspy import Inventory, UTCDateTime
from obspy.clients.fdsn import Client

NCEDC_NETWORKS = frozenset(["NC", "BK"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", required=True, help="picks CSV with a NET.STA 'station' column")
    ap.add_argument("--out-xml", default="station_inventory.xml")
    ap.add_argument("--out-csv", default="station_epochs.csv")
    ap.add_argument("--t0", default="2010-01-01")
    ap.add_argument("--t1", default="2016-01-01")
    ap.add_argument("--fdsn", default="IRIS")
    ap.add_argument("--channels", default="?H?,?N?")
    args = ap.parse_args(argv)

    picks = pd.read_csv(args.picks)
    picks.columns = [c.strip() for c in picks.columns]
    networks = sorted({str(s).split(".")[0].strip()
                       for s in picks["station"].dropna()})
    t0, t1 = UTCDateTime(args.t0), UTCDateTime(args.t1)
    print("networks:", networks)

    inv = Inventory(networks=[], source="route_a_build_station_inventory")
    rows = []
    for net in networks:
        client = Client("NCEDC") if net in NCEDC_NETWORKS else Client(args.fdsn)
        try:
            sub = client.get_stations(network=net, station="*", location="*",
                                      channel=args.channels, starttime=t0, endtime=t1,
                                      level="response")
        except Exception as e:
            print(f"{net}: get_stations failed: {e}")
            continue
        inv += sub
        n_ch = 0
        for n in sub:
            for s in n:
                for c in s.channels:
                    n_ch += 1
                    rows.append(dict(
                        network=n.code, station=s.code, location=c.location_code,
                        channel=c.code, latitude=c.latitude, longitude=c.longitude,
                        elevation=c.elevation, depth=c.depth,
                        start_date=str(c.start_date), end_date=str(c.end_date),
                        sample_rate=c.sample_rate,
                        has_response=c.response is not None))
        print(f"{net}: {n_ch} channel-epochs")

    inv.write(args.out_xml, format="STATIONXML")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    # flag stations with >1 epoch per channel band (redeployments -> epoch-keyed terms)
    if len(df):
        multi = (df.groupby(["network", "station"])["start_date"].nunique() > 1).sum()
        print(f"wrote {args.out_xml} and {args.out_csv}: "
              f"{len(df)} channel-epochs, {df.groupby(['network','station']).ngroups} stations, "
              f"{multi} stations with multiple epochs (redeployments)")


if __name__ == "__main__":
    main()
