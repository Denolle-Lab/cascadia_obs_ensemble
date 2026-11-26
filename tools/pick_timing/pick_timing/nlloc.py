"""NonLinLoc helpers: phase residuals from hypocenter-phase files, and re-location of a subset of events.

``extract-nlloc`` turns NLLoc per-event ``*.grid0.loc.hyp`` files (NLL_FORMAT_VER_2 phase lines) into one
residual table with canonical columns: event_id, station, phase, time (pick, POSIX s), residual (s, NLLoc
'Res', i.e. after the station correction Tcorr), weight, ttpred, tcorr, sdist (km), x_km/y_km/z_km of the
station in the NLLoc frame, and the event's origin time, hypocentre (lat, lon, depth; x, y, z in the frame).
Only LOCATED events are kept. When several globs are given, a later one wins for the same event_id (e.g. a
deep pass that re-located some events).
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .tables import posix_to_iso

_EPOCH = np.datetime64("1970-01-01T00:00:00", "s")


def _posix(date8: str, hhmm: str, sec: float) -> float:
    d = np.datetime64(f"{date8[:4]}-{date8[4:6]}-{date8[6:8]}T{hhmm[:2]}:{hhmm[2:4]}:00", "s")
    return float((d - _EPOCH).astype(np.int64)) + sec


def parse_hyp(path: str, id_regex: str = r"(\d+)$") -> tuple[dict | None, list]:
    """One NLLoc hyp file -> (event dict, phase rows). Event None if not LOCATED."""
    ev, rows, inphase = {}, [], False
    rx = re.compile(id_regex)
    with open(path) as fh:
        for line in fh:
            if line.startswith("NLLOC "):
                ev["status"] = line.split('"')[3] if line.count('"') >= 4 else ""
            elif line.startswith("PUBLIC_ID"):
                m = rx.search(line.split()[1])
                ev["event_id"] = int(m.group(1)) if m and m.group(1).isdigit() else (m.group(1) if m else line.split()[1])
            elif line.startswith("HYPOCENTER"):
                t = line.split()
                ev["ev_x_km"], ev["ev_y_km"], ev["ev_z_km"] = float(t[2]), float(t[4]), float(t[6])
            elif line.startswith("GEOGRAPHIC"):
                t = line.split()
                ev["origin_s"] = _posix(f"{t[2]}{int(t[3]):02d}{int(t[4]):02d}", f"{int(t[5]):02d}{int(t[6]):02d}",
                                        float(t[7]))
                ev["ev_lat"], ev["ev_lon"], ev["ev_depth_km"] = float(t[9]), float(t[11]), float(t[13])
            elif line.startswith("PHASE "):
                inphase = True
            elif line.startswith("END_PHASE"):
                inphase = False
            elif inphase and ">" in line:
                a, b = line.split(">", 1)
                ta, tb = a.split(), b.split()
                if len(ta) < 9 or len(tb) < 12:
                    continue
                rows.append((ta[0], ta[4], _posix(ta[6], ta[7], float(ta[8])), float(tb[0]), float(tb[1]),
                             float(tb[2]), float(tb[3]), float(tb[4]), float(tb[5]), float(tb[6]),
                             float(tb[11]), float(ta[10]) if len(ta) > 10 else np.nan))
    if ev.get("status") != "LOCATED":
        return None, []
    return ev, rows


_COLS = ["station", "phase", "time_s", "ttpred", "residual", "weight", "x_km", "y_km", "z_km", "sdist",
         "tcorr", "pick_err"]


def _parse_many(args):
    files, id_regex, tag = args
    out = []
    for f in files:
        ev, rows = parse_hyp(f, id_regex)
        if ev is None or not rows:
            continue
        d = pd.DataFrame(rows, columns=_COLS)
        for k, v in ev.items():
            if k != "status":
                d[k] = v
        d["hyp_file"] = os.path.relpath(f, os.path.dirname(os.path.dirname(f)))
        d["run"] = tag
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def extract(globs: list[str], id_regex: str = r"(\d+)$", workers: int = 4, chunk: int = 500) -> pd.DataFrame:
    """Parse every hyp file matched by the globs (in order; later globs win per event_id)."""
    parts = []
    for gi, g in enumerate(globs):
        files = sorted(f for f in glob.glob(g) if ".sum." not in os.path.basename(f))
        jobs = [(files[i:i + chunk], id_regex, f"glob{gi}") for i in range(0, len(files), chunk)]
        if workers > 1:
            with Pool(workers) as p:
                res = p.map(_parse_many, jobs)
        else:
            res = [_parse_many(j) for j in jobs]
        d = pd.concat([r for r in res if len(r)], ignore_index=True) if any(len(r) for r in res) else pd.DataFrame()
        d["glob_index"] = gi
        parts.append(d)
        print(f"  {g}: {len(files)} files, {len(d)} phase rows, {d.event_id.nunique() if len(d) else 0} located events",
              flush=True)
    df = pd.concat(parts, ignore_index=True)
    # later glob wins per event
    last = df.groupby("event_id").glob_index.transform("max")
    df = df[df.glob_index == last].drop(columns="glob_index").reset_index(drop=True)
    df["time"] = [posix_to_iso(t, 4) for t in df.time_s.values]
    return df


def stations_from_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Station table (median frame coordinates) from the residual rows."""
    return df.groupby("station")[["x_km", "y_km", "z_km"]].median().reset_index()


# ------------------------------------------------------------------------------------- subset re-location
_OBS_RX = re.compile(r"^(\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+)(\d{8})\s+(\d{4})\s+(-?[\d.]+)(\s.*)$")


def shift_obs_line(line: str, dt: float) -> str:
    """Shift the arrival time of one NLLOC_OBS phase line by dt seconds (4 decimals kept)."""
    m = _OBS_RX.match(line.rstrip("\n"))
    if not m:
        raise ValueError(f"not an NLLOC_OBS phase line: {line!r}")
    t = _posix(m.group(2), m.group(3), float(m.group(4))) + dt
    iso = posix_to_iso(t, 4, "")
    date8 = iso[:10].replace("-", "")
    hhmm = iso[11:13] + iso[14:16]
    sec = float(iso[17:])
    return f"{m.group(1)}{date8} {hhmm} {sec:7.4f}{m.group(5)}\n"


def read_obs_blocks(path: str) -> dict:
    """NLLOC_OBS file with PUBLIC_ID headers -> {public_id: [lines]} (blank-line separated blocks)."""
    blocks, cur, key = {}, [], None
    for line in open(path):
        if line.startswith("PUBLIC_ID"):
            if key is not None:
                blocks[key] = cur
            key, cur = line.split()[1], [line]
        elif line.strip() == "":
            if key is not None:
                blocks[key] = cur
                key, cur = None, []
        elif key is not None:
            cur.append(line)
    if key is not None:
        blocks[key] = cur
    return blocks


def relocate_subset(run_dir: str, tiles_events: dict, shifts: dict, out_dir: str, nlloc_bin: str,
                    id_prefix: str = "", workers: int = 4) -> pd.DataFrame:
    """Re-locate events with shifted picks, re-using each tile's NLLoc control file (read only).

    tiles_events: {tile: [public_id, ...]}; shifts: {(public_id, station, phase): dt_s}. For each tile, writes
    out_dir/<tile>/events.obs (only those events, picks shifted) and nlloc.in (the tile's file with LOCFILES
    pointing to them and the output root moved), runs NLLoc and returns the residual table of the new
    hyp files. Grids, model and station terms are the run's own (read only)."""
    out_dir = Path(out_dir)
    jobs = []
    for tile, ids in tiles_events.items():
        td = out_dir / tile
        td.mkdir(parents=True, exist_ok=True)
        src = Path(run_dir) / tile
        blocks = read_obs_blocks(str(src / "events.obs"))
        with open(td / "events.obs", "w") as f:
            for pid in ids:
                if pid not in blocks:
                    continue
                for line in blocks[pid]:
                    if line.startswith("PUBLIC_ID"):
                        f.write(line)
                        continue
                    sta, ph = line.split()[0], line.split()[4]
                    dt = shifts.get((pid, sta, ph), 0.0)
                    f.write(shift_obs_line(line, dt) if dt else line)
                f.write("\n")
        ctl = []
        for line in open(src / "nlloc.in"):
            if line.startswith("LOCFILES"):
                t = line.split()
                t[1] = str(td / "events.obs")
                t[4] = str(td / "loc")
                line = " ".join(t) + "\n"
            ctl.append(line)
        (td / "nlloc.in").write_text("".join(ctl))
        jobs.append((nlloc_bin, str(td)))
    if workers > 1:
        with Pool(workers) as p:
            p.map(_run_nlloc, jobs)
    else:
        for j in jobs:
            _run_nlloc(j)
    return extract([str(out_dir / "*" / "loc.*.grid0.loc.hyp")], workers=workers)


def _run_nlloc(job):
    nll, td = job
    if (Path(td) / "loc.sum.grid0.loc.hyp").exists():
        return
    with open(Path(td) / "nlloc.log", "w") as log:
        subprocess.run([nll, str(Path(td) / "nlloc.in")], stdout=log, stderr=subprocess.STDOUT, cwd=td, check=False)


# ------------------------------------------------------------------------------------- command line
def cli_extract(cfg, a):
    from .config import out_path
    from .tables import provenance, write_json, write_table
    out = a.output or str(out_path(cfg, "residuals.parquet"))
    print(f"extract-nlloc: {a.hyp_glob}", flush=True)
    df = extract(a.hyp_glob, a.id_regex, workers=cfg["workers"])
    p = write_table(df, out)
    st = stations_from_residuals(df)
    if a.stations_glob:
        rows = []
        for f in sorted(glob.glob(a.stations_glob)):
            try:
                s = pd.read_csv(f, sep=r"\s+")
                rows.append(s.rename(columns={"Label": "station"})[["station", "lat", "lon"]])
            except Exception:  # noqa: BLE001
                continue
        if rows:
            ll = pd.concat(rows).groupby("station")[["lat", "lon"]].median().reset_index()
            st = st.merge(ll, on="station", how="left")
    sp = write_table(st, Path(p).parent / "stations.csv")
    write_json(provenance({}, {"hyp_globs": a.hyp_glob, "rows": len(df), "events": int(df.event_id.nunique()),
                               "stations": int(df.station.nunique()), "output": str(p), "stations_table": str(sp)},
                          md5_inputs=False), str(p) + ".provenance.json")
    print(f"{len(df)} phase rows, {df.event_id.nunique()} events, {df.station.nunique()} stations -> {p}", flush=True)
    return 0
