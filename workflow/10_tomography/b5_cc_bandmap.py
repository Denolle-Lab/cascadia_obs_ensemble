"""Band per station for a set: the band (HH/BH/EH) whose vertical was read at the most picks of that station
(from the snippet meta written with the picker's rule HH > BH > EH). With it, both events of a pair use the same
channel; the picker's rule alone switches XD stations between HH (100 Hz, some days) and BH (50 Hz, every day),
and those cross-band pairs miss triangle closure 4-5 times more often (msh: 6.8 % vs 1.5 % > 20 ms).

  python b5_cc_bandmap.py --set msh   -> snippets/<set>/band_map.json, band_counts.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    a = ap.parse_args()
    d = C.SNIP / a.set
    m = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*.meta.csv"))], ignore_index=True)
    m = m[(m.comp == "Z") & m.band.notna()]
    c = m.groupby(["station", "band"]).size().rename("n").reset_index()
    c.to_csv(d / "band_counts.csv", index=False)
    bm = c.sort_values("n", ascending=False).drop_duplicates("station").set_index("station").band.to_dict()
    (d / "band_map.json").write_text(json.dumps(bm, indent=1, sort_keys=True))
    multi = c.groupby("station").filter(lambda g: len(g) > 1)
    print(f"{a.set}: {len(bm)} stations, {multi.station.nunique()} with more than one band; picks off the map band: "
          f"{int((m.band != m.station.map(bm)).sum())}")
    if len(multi):
        print(multi.pivot(index="station", columns="band", values="n").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
