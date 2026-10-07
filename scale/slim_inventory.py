#!/usr/bin/env python3
"""
Shrink a Route A StationXML to what route_a_wa_amplitudes.py can use: only the
stations that appear in the picks, and only ground-motion channels at >= --min-sr
that carry a response (the same filter the amplitude script applies). The full inventory holds every
station of every network in the picks (all of TA, nationwide) and parses to
~11.7 GB per process, so 20 shards filled a 250 GB host.

  # from workflow/05_magnitude:
  python ../../scale/slim_inventory.py --picks <picks.csv> --inventory station_inventory_v2.xml \
      --out station_inventory_v2_slim.xml
"""
import argparse

import pandas as pd
from obspy import read_inventory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-sr", type=float, default=10.0)
    args = ap.parse_args()

    codes = pd.read_csv(args.picks, usecols=["station"])["station"].dropna()
    want = {tuple(x.strip() for x in c.split(".")[:2]) for c in codes.astype(str).unique()}
    inv = read_inventory(args.inventory, format="STATIONXML")
    n0 = sum(len(s.channels) for n in inv for s in n)
    for net in inv:
        net.stations = [s for s in net if (net.code, s.code) in want]
        for s in net:
            s.channels = [c for c in s.channels if len(c.code) == 3
                          and c.code[1] in "HLNP" and (c.sample_rate or 0) >= args.min_sr
                          and c.response is not None]      # unusable for deconvolution
    inv.networks = [n for n in inv if n.stations]
    n1 = sum(len(s.channels) for n in inv for s in n)
    inv.write(args.out, format="STATIONXML")
    print(f"{len(want)} pick stations; channel-epochs {n0} -> {n1}; wrote {args.out}")


if __name__ == "__main__":
    main()
