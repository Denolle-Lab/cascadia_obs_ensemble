"""Pluggable waveform access returning raw MiniSEED bytes (record headers must stay untouched).

Every reader implements:
  day_blocks(net, sta, day0, bands=None) -> list[Block]   the byte blocks a whole-day read returns, in the order
      the archive returns them (one ObsPy read per block, as a day-reading picker does)
  day_heads(net, sta, day0, bands=None) -> DataFrame      record headers of that day (header-only where possible)
  window_blocks(net, sta, t0, t1, bands=None) -> list[Block]

Readers (config ``waveforms.reader``):
  sqlite_index  an mseedindex/tsindex sqlite archive (one file per year or one file), with an optional path rewrite
                (``path_map``). pnwstore is this reader with ``path_map: {type: wd_by_year}`` (the year -> disk
                table is pnwstore.constants.wd_mapper; it is read from pnwstore when installed, else from the config).
  fdsn          ObsPy FDSN dataselect (raw bytes kept); ``client`` is a data-centre name or base URL.
  sds           an SDS tree: {root}/{year}/{net}/{sta}/{cha}.D/{net}.{sta}.{loc}.{cha}.D.{year}.{doy}
  routed        ``routes``: list of {networks: [...], reader: {...}}; ``default``: reader config; ``fallback``:
                reader config used when the routed reader returns nothing.
"""
from __future__ import annotations

import glob
import io
import os
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mseed import parse_headers, read_heads_file

DAY = 86400.0


@dataclass
class Block:
    data: bytes
    source: str
    loc: str = ""
    cha: str = ""


def _iso(t: float) -> str:
    return pd.Timestamp(t, unit="s").strftime("%Y-%m-%dT%H:%M:%S.%f")


def _band_ok(cha: str, bands) -> bool:
    return bands is None or cha[:2] in bands


class Reader:
    name = "base"

    def day_blocks(self, net, sta, day0, bands=None):  # pragma: no cover - interface
        raise NotImplementedError

    def day_heads(self, net, sta, day0, bands=None) -> pd.DataFrame:
        hs = []
        for b in self.day_blocks(net, sta, day0, bands):
            h = parse_headers(b.data)
            h["source"] = b.source
            hs.append(h)
        return pd.concat(hs, ignore_index=True) if hs else pd.DataFrame()

    def window_blocks(self, net, sta, t0, t1, bands=None):
        out = []
        d = np.floor(t0 / DAY) * DAY
        while d < t1:
            out += self.day_blocks(net, sta, d, bands)
            d += DAY
        return out


# ------------------------------------------------------------------------------------------ sqlite index
class SqliteIndexReader(Reader):
    """mseedindex 'tsindex' table: network, station, location, channel, starttime, endtime, samplerate,
    filename, byteoffset, bytes. ``index`` may contain {year} (one database per year, as pnwstore)."""
    name = "sqlite_index"

    def __init__(self, index: str, path_map: dict | None = None, day_mode: str = "filename", **_):
        self.index = index
        self.path_map = path_map or {}
        self.day_mode = day_mode
        self._dbs = {}
        if self.path_map.get("type") == "wd_by_year" and not self.path_map.get("wd_by_year"):
            try:  # the archive's own table when pnwstore is installed
                from pnwstore.constants import wd_mapper  # type: ignore
                self.path_map["wd_by_year"] = dict(wd_mapper)
            except ImportError as e:
                raise ImportError("path_map type wd_by_year needs wd_by_year in the config or pnwstore installed") from e

    def _db(self, year):
        key = year if "{year}" in self.index else 0
        if key not in self._dbs:
            fn = self.index.format(year=year)
            if not os.path.exists(fn):
                self._dbs[key] = None
            else:
                self._dbs[key] = sqlite3.connect(f"file:{fn}?mode=ro", uri=True, check_same_thread=False)
        return self._dbs[key]

    def real_path(self, fn: str) -> str:
        pm = self.path_map
        t = pm.get("type")
        if not t:
            return fn
        if t == "wd_by_year":
            # as pnwstore.utils.pnwstore_filename_mapper: <prefix>/PNWyyyy/... -> root.format(wd=wd_by_year[yyyy]) + rest
            year = int(fn.split("/")[pm.get("year_part", 5)])
            wd = {int(k): v for k, v in pm["wd_by_year"].items()}[year]
            return pm["root"].format(wd=wd) + fn[pm.get("strip", 10):]
        if t == "replace":
            return fn.replace(pm["from"], pm["to"], 1)
        raise ValueError(f"unknown path_map type {t}")

    def rows_day(self, net, sta, day0) -> pd.DataFrame:
        t = pd.Timestamp(day0, unit="s")
        db = self._db(t.year)
        if db is None:
            return pd.DataFrame()
        cols = "network,station,location,channel,starttime,endtime,samplerate,filename,byteoffset,bytes"
        if self.day_mode == "filename":
            # what pnwstore.WaveformClient.get_waveforms(year, month, day) selects
            q = (f"select {cols} from tsindex where network=? and station=? and channel like '%' "
                 "and filename like ? and filename like ?")
            args = [net, sta, f"%/{t.year}/%", f"%/{t.dayofyear:03d}/%"]
        else:
            q = f"select {cols} from tsindex where network=? and station=? and starttime<? and endtime>=?"
            args = [net, sta, _iso(day0 + DAY), _iso(day0)]
        return pd.read_sql(q, db, params=args)

    def day_blocks(self, net, sta, day0, bands=None):
        out = []
        for r in self.rows_day(net, sta, day0).itertuples():
            if not _band_ok(r.channel, bands):
                continue
            with open(self.real_path(r.filename), "rb") as f:
                f.seek(int(r.byteoffset))
                out.append(Block(f.read(int(r.bytes)), f"{r.filename}@{int(r.byteoffset)}", r.location, r.channel))
        return out

    def day_heads(self, net, sta, day0, bands=None):
        hs = []
        for r in self.rows_day(net, sta, day0).itertuples():
            if not _band_ok(r.channel, bands):
                continue
            h = read_heads_file(self.real_path(r.filename), int(r.byteoffset), int(r.bytes))
            h["source"] = r.filename
            hs.append(h)
        return pd.concat(hs, ignore_index=True) if hs else pd.DataFrame()


# ------------------------------------------------------------------------------------------ FDSN
class FDSNReader(Reader):
    name = "fdsn"

    def __init__(self, client: str = "EARTHSCOPE", channels: str = "HH?,BH?,EH?", timeout: float = 120, **_):
        self.label, self.timeout, self.channels = client, timeout, channels
        self._client = None

    @property
    def client(self):
        """Created on first use (creating an ObsPy FDSN client queries the data centre)."""
        if self._client is None:
            from obspy.clients.fdsn import Client
            self._client = Client(self.label, timeout=self.timeout)
        return self._client

    def _get(self, net, sta, t0, t1, bands):
        from obspy import UTCDateTime
        cha = ",".join(f"{b}?" for b in bands) if bands else self.channels
        buf = io.BytesIO()
        try:
            self.client.get_waveforms(net, sta, "*", cha, UTCDateTime(t0), UTCDateTime(t1), filename=buf)
        except Exception:  # noqa: BLE001  (no data -> FDSNNoDataException)
            return []
        data = buf.getvalue()
        return [Block(data, f"fdsn:{self.label}:{net}.{sta}:{_iso(t0)}")] if data else []

    def day_blocks(self, net, sta, day0, bands=None):
        return self._get(net, sta, day0, day0 + DAY, bands)

    def window_blocks(self, net, sta, t0, t1, bands=None):
        return self._get(net, sta, t0, t1, bands)


# ------------------------------------------------------------------------------------------ SDS
class SDSReader(Reader):
    name = "sds"

    def __init__(self, root: str, **_):
        self.root = root

    def day_blocks(self, net, sta, day0, bands=None):
        t = pd.Timestamp(day0, unit="s")
        pat = os.path.join(self.root, f"{t.year}", net, sta, "*.D", f"{net}.{sta}.*.*.D.{t.year}.{t.dayofyear:03d}")
        out = []
        for fn in sorted(glob.glob(pat)):
            parts = os.path.basename(fn).split(".")
            loc, cha = parts[2], parts[3]
            if not _band_ok(cha, bands):
                continue
            with open(fn, "rb") as f:
                out.append(Block(f.read(), fn, loc, cha))
        return out


# ------------------------------------------------------------------------------------------ router
class RoutedReader(Reader):
    name = "routed"

    def __init__(self, routes=None, default=None, fallback=None, **_):
        self.routes = [(set(r["networks"]), make_reader(r["reader"])) for r in (routes or [])]
        self.default = make_reader(default) if default else None
        self.fallback = make_reader(fallback) if fallback else None

    def _pick(self, net):
        for nets, r in self.routes:
            if net in nets:
                return r
        return self.default

    def _call(self, meth, net, *args):
        r = self._pick(net)
        out = getattr(r, meth)(net, *args) if r is not None else []
        if (out is None or len(out) == 0) and self.fallback is not None:
            out = getattr(self.fallback, meth)(net, *args)
        return out

    def day_blocks(self, net, sta, day0, bands=None):
        return self._call("day_blocks", net, sta, day0, bands)

    def day_heads(self, net, sta, day0, bands=None):
        return self._call("day_heads", net, sta, day0, bands)

    def window_blocks(self, net, sta, t0, t1, bands=None):
        return self._call("window_blocks", net, sta, t0, t1, bands)


def make_reader(spec: dict) -> Reader:
    spec = dict(spec or {})
    kind = spec.pop("reader", spec.pop("type", "fdsn"))
    if kind == "sqlite_index":
        return SqliteIndexReader(**spec)
    if kind == "fdsn":
        sub = spec.pop("fdsn", {})
        return FDSNReader(**{**spec, **sub})
    if kind == "sds":
        return SDSReader(**spec)
    if kind == "routed":
        return RoutedReader(**spec)
    raise ValueError(f"unknown waveform reader {kind}")
