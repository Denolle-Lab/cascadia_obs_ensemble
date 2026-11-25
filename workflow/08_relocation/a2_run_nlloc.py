"""Track A, step A2: run NLLoc tile by tile (obs from a2_make_obs.py), in parallel.

Each tile's search box (LOCGRID) is the tile +- 75 km, z -5..80 km, 1 km; the oct-tree starts with ~10 km
cells. Tiles with more than CHUNK events are split so they run in parallel.

Usage: python a2_run_nlloc.py RUN_NAME [--time time_vpvs176] [--tiles t13_29,t14_29] [--max-events N] [--nproc 100] [--stat FILE]
  --stat: NLL LOCDELAY station-delay file (from Loc2ssst / station terms) to include.
Output: /wd1/mdenolle_data/reloc_v5/runs/RUN_NAME/<tile>[_cNN]/loc.*
"""
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

B = Path("/wd1/mdenolle_data/reloc_v5")
NLL = B / "src/NonLinLoc/src/bin/NLLoc"
TEMPLATE = Path(__file__).with_name("nlloc_template.in")
CHUNK = 500
Z0, NZ = -5.0, 86


def split_obs(text, max_events=None):
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if max_events:
        blocks = blocks[:max_events]
    return [blocks[i:i + CHUNK] for i in range(0, len(blocks), CHUNK)]


def control(tile, obs, out, stat, time_dir):
    nx = int(round(tile.box_x1 - tile.box_x0)) + 1
    ny = int(round(tile.box_y1 - tile.box_y0)) + 1
    locgrid = (f"LOCGRID {nx} {ny} {NZ} {tile.box_x0:.1f} {tile.box_y0:.1f} {Z0:.1f} 1.0 1.0 1.0 "
               "PROB_DENSITY SAVE")
    search = f"LOCSEARCH OCT {max(nx // 10, 2)} {max(ny // 10, 2)} 9 0.02 40000 2000 0 1"
    t = TEMPLATE.read_text()
    t = t.replace("__TIME__", time_dir).replace("__OBS__", str(obs)).replace("__OUT__", str(out))
    t = t.replace("__STATERMS__", f"INCLUDE {stat}" if stat else "")
    lines = [l for l in t.splitlines() if not l.startswith(("LOCGRID", "LOCSEARCH"))]
    return "\n".join(lines + [locgrid, search]) + "\n"


def run_one(job):
    wd, ctl = job
    (wd / "nlloc.in").write_text(ctl)
    with open(wd / "nlloc.log", "w") as log:
        r = subprocess.run([str(NLL), "nlloc.in"], cwd=wd, stdout=log, stderr=subprocess.STDOUT)
    return wd.name, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tiles")
    ap.add_argument("--max-events", type=int)
    ap.add_argument("--nproc", type=int, default=100)
    ap.add_argument("--stat")
    ap.add_argument("--time", default="time_vpvs176", help="travel-time grid folder under reloc_v5")
    a = ap.parse_args()
    tiles = pd.read_csv(B / "obs_tiles/tiles.csv")
    if a.tiles:
        tiles = tiles[tiles.tile.isin(a.tiles.split(","))]
    jobs = []
    for tile in tiles.itertuples():
        chunks = split_obs((B / f"obs_tiles/{tile.tile}.obs").read_text(), a.max_events)
        for c, blocks in enumerate(chunks):
            name = tile.tile if len(chunks) == 1 else f"{tile.tile}_c{c:02d}"
            wd = B / "runs" / a.run / name
            wd.mkdir(parents=True, exist_ok=True)
            if (wd / "loc.sum.grid0.loc.hyp").exists():
                continue
            (wd / "events.obs").write_text("\n\n".join(blocks) + "\n\n")
            jobs.append((wd, control(tile, wd / "events.obs", wd / "loc", a.stat, a.time)))
    print(f"{len(jobs)} NLLoc jobs")
    bad = []
    with ProcessPoolExecutor(a.nproc) as ex:
        for name, rc in ex.map(run_one, jobs):
            if rc != 0:
                bad.append(name)
    print("failed:", bad if bad else "none")


if __name__ == "__main__":
    main()
