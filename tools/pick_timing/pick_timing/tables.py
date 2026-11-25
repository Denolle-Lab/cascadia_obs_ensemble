"""Table input/output through a column-mapping config, time parsing, provenance helpers.

Every table the package reads (picks, origins, residuals, stations, CC differential times) is described in
the YAML config by a path, optional pandas read options and a ``columns`` map from the package's canonical
names to the file's own column names. Canonical names:

picks      : event_id, station, phase, time  [travel_time, timing_correction, network, station_code]
origins    : event_id, time, lat, lon, depth  [x_km, y_km, z_km]
residuals  : event_id, station, phase, time, residual  [weight, x_km, y_km, lat, lon]
stations   : station, lat, lon  [x_km, y_km, elevation]
cc         : pair, event1, event2, station, phase, dt_cc, time1, time2  [accepted, cc]

``station`` is a single string id. ``station_id`` in the config gives its pattern, e.g. "{sta}.{net}" (default)
or "{net}.{sta}"; when the file has separate network and station columns, map ``network`` and
``station_code`` and the id is built from the pattern.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "picks": ["event_id", "station", "phase", "time"],
    "origins": ["event_id", "time"],
    "residuals": ["event_id", "station", "phase", "time", "residual"],
    "stations": ["station"],
    "cc": ["event1", "event2", "station", "phase", "dt_cc", "time1", "time2"],
}


# ----------------------------------------------------------------------------------------------- time
def to_posix(values, fmt: str = "iso") -> np.ndarray:
    """Times -> float POSIX seconds (UTC). fmt: 'iso' (ISO 8601 strings, 'Z' optional), 'epoch' (seconds)."""
    if fmt == "epoch":
        return np.asarray(values, dtype=float)
    s = pd.Series(values).astype(str).str.strip().str.rstrip("Z")
    t = pd.to_datetime(s, format="ISO8601", utc=True)
    # integer nanoseconds -> seconds, keeping the sub-microsecond part exact enough for 0.1 ms work
    ns = t.astype("int64").to_numpy()
    return (ns // 1_000_000_000).astype(float) + (ns % 1_000_000_000) / 1e9


def posix_to_iso(t, decimals: int = 6, suffix: str = "Z") -> str:
    """POSIX seconds -> 'YYYY-MM-DDTHH:MM:SS.ffffff' + suffix, rounded to `decimals`."""
    q = 10 ** decimals
    tot = int(round(float(t) * q))
    sec, frac = divmod(tot, q)
    base = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=sec)
    s = base.strftime("%Y-%m-%dT%H:%M:%S")
    if decimals > 0:
        s += "." + str(frac).rjust(decimals, "0")
    return s + suffix


# ----------------------------------------------------------------------------------------------- ids
def station_pattern(cfg: dict) -> str:
    return (cfg or {}).get("station_id", "{sta}.{net}")


def split_station(ids, pattern: str = "{sta}.{net}"):
    """Station ids -> (net, sta) arrays following the pattern."""
    rx = re.compile("^" + re.escape(pattern).replace(r"\{sta\}", r"(?P<sta>[^.]+)")
                    .replace(r"\{net\}", r"(?P<net>[^.]+)") + "$")
    nets, stas = [], []
    for s in ids:
        m = rx.match(str(s))
        if m:
            nets.append(m.group("net"))
            stas.append(m.group("sta"))
        else:
            nets.append("")
            stas.append(str(s))
    return np.array(nets), np.array(stas)


def join_station(net, sta, pattern: str = "{sta}.{net}"):
    return [pattern.format(net=n, sta=s) for n, s in zip(net, sta)]


# ----------------------------------------------------------------------------------------------- tables
def read_raw(spec: dict) -> pd.DataFrame:
    """Read a table as stored (no renaming): csv/csv.gz/txt or parquet. ``path`` may be a list: the tables are
    concatenated with a column _source (file index); a 'pair' column is made unique per file."""
    path = spec["path"]
    if isinstance(path, (list, tuple)):
        parts = []
        for k, pth in enumerate(path):
            d = read_raw(dict(spec, path=pth))
            d["_source"] = k
            pc = spec.get("columns", {}).get("pair", "pair")
            if pc in d.columns:
                d[pc] = str(k) + ":" + d[pc].astype(str)
            parts.append(d)
        return pd.concat(parts, ignore_index=True)
    kw = dict(spec.get("read_kwargs", {}))
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, **kw)
    kw.setdefault("low_memory", False)
    return pd.read_csv(path, **kw)


def canonical(df: pd.DataFrame, spec: dict, kind: str, cfg: dict | None = None) -> pd.DataFrame:
    """Rename a raw table to canonical names (only mapped columns are renamed; others are kept), check the
    required ones, add 'time_s' (POSIX float) for time columns and build 'station' from network/code if needed."""
    cols = spec.get("columns", {})
    inv = {v: k for k, v in cols.items()}
    out = df.rename(columns=inv)
    if kind in ("picks", "residuals", "stations", "cc") and "station" not in out.columns \
            and {"network", "station_code"} <= set(out.columns):
        out["station"] = join_station(out["network"], out["station_code"], station_pattern(cfg))
    missing = [c for c in REQUIRED.get(kind, []) if c not in out.columns]
    if missing:
        raise KeyError(f"{kind} table {spec.get('path')}: missing columns {missing} "
                       f"(map them in the config under {kind}.columns)")
    fmt = spec.get("time_format", "iso")
    for c in ("time", "time1", "time2"):
        if c in out.columns and c + "_s" not in out.columns:
            numeric = np.issubdtype(out[c].dtype, np.number)
            out[c + "_s"] = to_posix(out[c].values, "epoch" if numeric else fmt)
    return out


def load(cfg: dict, kind: str, path: str | None = None) -> pd.DataFrame:
    spec = dict(cfg[kind])
    if path:
        spec["path"] = path
    return canonical(read_raw(spec), spec, kind, cfg)


def to_file_columns(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Inverse of canonical(): canonical names back to the file's names (drops helper *_s columns)."""
    cols = spec.get("columns", {})
    out = df.drop(columns=[c for c in df.columns if c.endswith("_s") and c[:-2] in ("time", "time1", "time2")])
    return out.rename(columns=cols)


def write_table(df: pd.DataFrame, path: str | Path, **kw):
    """Write csv / csv.gz / parquet by extension (parquet needs pyarrow; falls back to .csv.gz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            path = path.with_suffix(".csv.gz")
    df.to_csv(path, index=False, **kw)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    if path.suffix == ".parquet" and path.with_suffix(".csv.gz").exists():
        path = path.with_suffix(".csv.gz")
    return pd.read_csv(path, low_memory=False)


# ----------------------------------------------------------------------------------------------- provenance
def md5(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def code_version() -> dict:
    """Package version and the git commit of the source tree (if it is a git checkout)."""
    from . import __version__
    here = Path(__file__).resolve().parent
    out = {"package": "pick_timing", "version": __version__, "source": str(here)}
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", str(here)],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        out["git_dirty"] = bool(dirty)
    except Exception:  # noqa: BLE001
        out["git_commit"] = None
    return out


def provenance(inputs: dict, extra: dict | None = None, md5_inputs: bool = True) -> dict:
    """Provenance record: inputs (path, md5, size), code version, command line, time, host-free."""
    rec = {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "command": [os.path.basename(sys.argv[0])] + sys.argv[1:], "code": code_version(), "inputs": {}}
    flat = {}
    for k, p in inputs.items():
        if isinstance(p, (list, tuple)):
            flat.update({f"{k}[{i}]": q for i, q in enumerate(p)})
        else:
            flat[k] = p
    for k, p in flat.items():
        if p is None:
            continue
        p = Path(p)
        rec["inputs"][k] = {"path": str(p), "bytes": p.stat().st_size if p.exists() else None,
                            "md5": md5(p) if (md5_inputs and p.exists() and p.is_file()) else None}
    if extra:
        rec.update(extra)
    return rec


def write_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, _dt.datetime)):
            return o.isoformat()
        return str(o)
    path.write_text(json.dumps(obj, indent=1, default=_default))
    return path
