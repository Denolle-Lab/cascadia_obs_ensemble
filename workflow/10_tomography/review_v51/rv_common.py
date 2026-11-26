"""Review reports of catalog v5.1 location problems (PI request 2026-11-25 07:08 PST): shared helpers.

Read-only on every input: catalog v5.1, re-timed picks, the NonLinLoc runs of v5.1 (b4_v51_iter1, b4_v51_deep),
their travel-time grids, the analysis outputs of the v5.1 pieces and pnwstore. Writes only under OUT.

* catalog, magnitudes, ComCat (ANSS cache of the mag piece), Slab2, the D-B068 solid surface (b4_surface.py)
* NonLinLoc: per-event .hyp (hypocentre, per-phase TTpred / Res / Tcorr) and .scat (scatter samples) of v5.1
* travel times in the v5.1 grids (fine 1 km grids within ~250 km of a station, wide 2 km grids to z' 121 km), chosen
  per arrival by NLLoc's header rule (first LOCFILES root whose grid contains the event's LOCGRID box), trilinear as
  NLLoc; true depth z and flattened depth z' are related by z = Re (1 - exp(-z'/Re)), Re = 6371 km
* waveforms: pnwstore short reads on the record-header time base (copy of b5_cc_common.py of branch tomo/cc-dt,
  itself from the D-B072 re-timing scripts): never a merged day; NC and BK are not in pnwstore
"""
import functools
import io
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OUT = Path("/wd1/mdenolle_data/catalog_v5/v5.1_review")
WORK = OUT / "work"
CAT = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
PICKS = Path("/wd1/mdenolle_data/catalog_v5/v5.1/Cascadia_picks_catalog_v4.1t.csv.gz")
RELOC = Path("/wd1/mdenolle_data/reloc_v5")
RUN_MAIN, RUN_DEEP = RELOC / "runs/b4_v51_iter1", RELOC / "runs/b4_v51_deep"
FINE = RELOC / "time_b4_v0.3.1m_sc21/cascadia3d"
WIDE = RELOC / "wide_time_b4_v0.3.1m_sc21/cascadia3d"
STATIONS = RELOC / "model_vpvs176/stations_GTSRCE.in"
STAT_TERMS = RELOC / "runs/b4_v51_iter0/station_terms.in"
MAGS = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/cascadia_catalog_v5.1_magnitudes.csv")
MT_MATCH = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/comcat_mt_matched_v51.csv")
ANSS = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/anss_2010-16.csv")
FAULTS = Path("/home/mdenolle/wt_a51_faults/data/catalog_v5/v5.1_analysis/faults")
SLAB2 = Path("/home/mdenolle/cascadia_obs_ensemble/data/slab2/cas_slab2_dep.xyz")
INDEX = "/wd1/PNWstore_sqlite/{year}.sqlite"
NO_PNWSTORE = {"NC", "BK"}
RE = 6371.0
MODEL_ERR = 0.2          # LOCGAU 0.2 0.0 of the v5.1 runs (D-B069)
MSH_BOX = dict(lat=(46.05, 46.30), lon=(-122.35, -122.10))   # faults piece (v1_report.py)
MSH_SUMMIT = (46.191, -122.194)


def eft(z):
    """true depth (km bsl) -> Earth-flattened z' (km)."""
    return RE * np.log(RE / (RE - np.asarray(z, float)))


def ieft(zp):
    """flattened z' -> true depth."""
    return RE * (1.0 - np.exp(-np.asarray(zp, float) / RE))


@functools.lru_cache(maxsize=1)
def proj():
    from pyproj import Proj
    return Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")


def lambert(lon, lat):
    x, y = proj()(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x) / 1e3, np.asarray(y) / 1e3


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((np.asarray(lat2) - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(np.asarray(lat2) * p) * np.sin((np.asarray(lon2) - lon1) * p / 2) ** 2)
    return 2 * RE * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def azimuth(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    y = np.sin((np.asarray(lon2) - lon1) * p) * np.cos(np.asarray(lat2) * p)
    x = np.cos(lat1 * p) * np.sin(np.asarray(lat2) * p) - np.sin(lat1 * p) * np.cos(np.asarray(lat2) * p) * np.cos((np.asarray(lon2) - lon1) * p)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


# ------------------------------------------------------------------------------------------------ tables
def _bool(s):
    return s.astype(str).str.strip().eq("True")


@functools.lru_cache(maxsize=1)
def catalog():
    c = pd.read_csv(CAT, low_memory=False)
    for k in ("floor_limited", "far_only", "deep_pass", "five_phase", "above_surface"):
        c[k] = _bool(c[k])
    c["t"] = pd.to_datetime(c.time, utc=True, format="ISO8601")
    c["t_v4"] = pd.to_datetime(c.v4_time, utc=True, format="ISO8601")
    return c.set_index("ID", drop=False)


@functools.lru_cache(maxsize=1)
def mags():
    return pd.read_csv(MAGS, low_memory=False).set_index("ID", drop=False)


@functools.lru_cache(maxsize=1)
def anss():
    a = pd.read_csv(ANSS, low_memory=False)
    a["t"] = pd.to_datetime(a.time, utc=True, format="ISO8601")
    return a.drop_duplicates("id").set_index("id", drop=False)


@functools.lru_cache(maxsize=1)
def mt_match():
    m = pd.read_csv(MT_MATCH)
    m["t"] = pd.to_datetime(m.time, utc=True, format="ISO8601")
    return m


def comcat_for(eid):
    """ComCat hypocentre of an event, or None: the tensor match of the mag piece (v5.1 match, else the v4.1 tensor
    carried by its comcat_id), else the ANSS match of the magnitude table (anss_id). Values from the mag piece's
    ANSS cache (anss_2010-16.csv; tensor file values where the cache lacks the id)."""
    c, m = catalog().loc[eid], mags()
    mm = mt_match()
    src, cid = None, None
    r = mm[mm.event_id == eid]
    if len(r):
        cid, src = r.comcat_id.iloc[0], "tensor match (v5.1)"
    elif isinstance(c.comcat_id, str) and c.comcat_id:
        cid, src = c.comcat_id, "tensor match (v4.1, carried)"
    elif eid in m.index and isinstance(m.loc[eid].anss_id, str):
        cid, src = m.loc[eid].anss_id, "ANSS match (mag piece)"
    if cid is None:
        return None
    a = anss()
    if cid in a.index:
        x = a.loc[cid]
        return dict(id=cid, src=src, t=x.t, lat=float(x.latitude), lon=float(x.longitude), depth=float(x.depth),
                    mag=float(x.mag), magtype=str(x.magType), type=str(x["type"]), place=str(x.place))
    if len(r):
        x = r.iloc[0]
        return dict(id=cid, src=src, t=x.t, lat=float(x.lat), lon=float(x.lon), depth=float(x.depth),
                    mag=float(x.mag), magtype=str(x.magtype), type="earthquake", place="")
    return dict(id=cid, src=src + "; not in the ANSS cache", t=None, lat=np.nan, lon=np.nan, depth=np.nan,
                mag=np.nan, magtype="", type="", place="")


@functools.lru_cache(maxsize=1)
def stations():
    rows = []
    for line in open(STATIONS):
        t = line.split()
        if t and t[0] == "GTSRCE":
            rows.append(dict(sta=t[1], lat=float(t[3]), lon=float(t[4]), z_km=float(t[5])))
    s = pd.DataFrame(rows).drop_duplicates("sta").set_index("sta", drop=False)
    s["net"] = s.sta.str.split(".").str[1]
    s["code"] = s.sta.str.split(".").str[0]
    return s


@functools.lru_cache(maxsize=1)
def station_terms():
    d = {}
    for line in open(STAT_TERMS):
        t = line.split()
        if t and t[0] == "LOCDELAY":
            d[(t[1], t[2])] = float(t[4])
    return d


@functools.lru_cache(maxsize=1)
def slab2():
    cache = WORK / "slab2_grid.npz"
    if cache.exists():
        z = np.load(cache)
        return z["lons"], z["lats"], z["Z"]
    a = pd.read_csv(SLAB2, header=None).values.astype(float)
    lon = np.where(a[:, 0] > 180, a[:, 0] - 360, a[:, 0])
    lons, lats = np.unique(lon), np.unique(a[:, 1])
    Z = np.full((len(lats), len(lons)), np.nan)
    Z[np.searchsorted(lats, a[:, 1]), np.searchsorted(lons, lon)] = -a[:, 2]     # depth km positive down
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(cache, lons=lons, lats=lats, Z=Z)
    return lons, lats, Z


def slab2_depth(lon, lat):
    """Slab2 (Hayes et al. 2018, cas_slab2_dep 02.24.18) slab-top depth, km positive down; NaN outside."""
    from scipy.interpolate import RegularGridInterpolator
    lons, lats, Z = slab2()
    f = RegularGridInterpolator((lats, lons), Z, bounds_error=False, fill_value=np.nan)
    return f(np.c_[np.ravel(lat), np.ravel(lon)]).reshape(np.shape(lon))


def surface_km(lon, lat):
    """D-B068 merged solid surface (km, positive up) through b4_surface.surface_m."""
    import b4_surface
    return b4_surface.surface_m(np.asarray(lon, float), np.asarray(lat, float)) / 1e3


# ------------------------------------------------------------------------------------------------ NonLinLoc
PH_COLS = ["sta", "ins", "cmp", "on", "phase", "fm", "date", "hrmn", "sec", "errtype", "err", "coda", "amp", "per",
           "priorwt", "gt", "ttpred", "res", "weight", "sx", "sy", "sz", "sdist", "sazim", "raz", "rdip", "rqual",
           "tcorr", "tterr"]


def parse_hyp(path):
    """One NLLoc .hyp (single event) -> dict (status, public_id, OT (pandas UTC), x, y, zp, lat, lon, depth_flat,
    grid, phases DataFrame with obs time 'tobs' (UTC Timestamp))."""
    out = dict(path=str(path))
    ph = []
    inph = False
    for line in open(path):
        t = line.split()
        if not t:
            continue
        k = t[0]
        if k == "NLLOC":
            out["status"] = t[2].strip('"')
        elif k == "PUBLIC_ID":
            out["public_id"] = t[1]
        elif k == "GRID":
            out["grid"] = [float(v) for v in t[1:10]]
        elif k == "HYPOCENTER":
            out["x"], out["y"], out["zp"] = float(t[2]), float(t[4]), float(t[6])
        elif k == "GEOGRAPHIC":
            y, mo, d, h, mi = (int(v) for v in t[2:7])
            out["OT"] = pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, tz="UTC") + pd.to_timedelta(float(t[7]), "s")
            out["lat"], out["lon"], out["depth_flat"] = float(t[9]), float(t[11]), float(t[13])
        elif k == "QUALITY":
            out["rms"], out["nphs"], out["gap"], out["dist"] = float(t[8]), int(t[10]), float(t[12]), float(t[14])
        elif k == "STATISTICS":
            out["cov"] = dict(XX=float(t[8]), XY=float(t[10]), XZ=float(t[12]), YY=float(t[14]), YZ=float(t[16]),
                              ZZ=float(t[18]))
        elif k == "PHASE":
            inph = True
        elif k == "END_PHASE":
            inph = False
        elif inph:
            if len(t) >= 29:
                ph.append(t[:29])
    P = pd.DataFrame(ph, columns=PH_COLS)
    for col in ("sec", "err", "ttpred", "res", "weight", "sx", "sy", "sz", "sdist", "sazim", "tcorr", "tterr"):
        P[col] = P[col].astype(float)
    if len(P):
        base = pd.to_datetime(P.date + P.hrmn.str.zfill(4), format="%Y%m%d%H%M", utc=True)
        P["tobs"] = base + pd.to_timedelta(P.sec, "s")
    out["phases"] = P
    return out


def read_scatter(path, max_n=4000):
    """NLLoc .scat: int n + 3 floats header, then n x (x, y, z', pdf) float32. Returns (n, 4) array (z' flattened)."""
    b = Path(path).read_bytes()
    n = int(np.frombuffer(b[:4], "<i4")[0])
    a = np.frombuffer(b[16:16 + 16 * n], "<f4").reshape(-1, 4)
    if len(a) > max_n:
        a = a[np.linspace(0, len(a) - 1, max_n).astype(int)]
    return a


@functools.lru_cache(maxsize=64)
def _wd_index(wd):
    """{event ID: per-event .hyp path} of a workdir (PUBLIC_ID line of each loc.*.grid0.loc.hyp)."""
    out = {}
    for f in sorted(Path(wd).glob("loc.*.grid0.loc.hyp")):
        if f.name.startswith("loc.sum"):
            continue
        with open(f) as fh:
            for _ in range(3):
                line = fh.readline()
                if line.startswith("PUBLIC_ID v4_"):
                    out.setdefault(int(line.split()[1][3:]), []).append(f)
                    break
    return out


def find_hyp(eid, row):
    """(.hyp path, .scat path or None, workdir) of an event in its v5.1 workdir (catalog 'tile' and pass). Per-event
    files are named by the first arrival time to the second; two events with the same name in a workdir overwrite each
    other, so the PUBLIC_ID v4_<ID> line decides; None when the event has no per-event file."""
    run = RUN_DEEP if bool(row.deep_pass) else RUN_MAIN
    wd = run / str(row.tile)
    if not wd.is_dir():
        return None, None, wd
    fs = _wd_index(str(wd)).get(int(eid), [])
    if not fs:
        return None, None, wd
    f = fs[-1]
    s = f.with_suffix(".scat")
    return f, (s if s.exists() else None), wd


def sum_block(eid, wd):
    """Fallback: the event's block (with phases if present) from loc.sum.grid0.loc.hyp, written to WORK/tmp."""
    f = wd / "loc.sum.grid0.loc.hyp"
    if not f.exists():
        return None
    blocks = f.read_text().split("END_NLLOC")
    for b in blocks:
        if f"PUBLIC_ID v4_{eid}\n" in b:
            p = WORK / "tmp" / f"sum_{eid}.hyp"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(b.strip("\n") + "\nEND_NLLOC\n")
            return p
    return None


def control_box(wd):
    """LOCGRID box (x0, x1, y0, y1, z0, z1) of a workdir's nlloc.in (a2_run_nlloc.read_control)."""
    text = (wd / "nlloc.in").read_text()
    g = re.search(r"^LOCGRID +(\d+) +(\d+) +(\d+) +(\S+) +(\S+) +(\S+) +(\S+) +(\S+) +(\S+)", text, re.M)
    nx, ny, nz = (int(g.group(i)) for i in (1, 2, 3))
    x0, y0, z0, dx, dy, dz = (float(g.group(i)) for i in range(4, 10))
    return (x0, x0 + (nx - 1) * dx, y0, y0 + (ny - 1) * dy, z0, z0 + (nz - 1) * dz)


# ------------------------------------------------------------------------------------------------ travel times
@functools.lru_cache(maxsize=4096)
def grid_hdr(root, phase, sta):
    h = Path(f"{root}.{phase}.{sta}.time.hdr")
    if not h.exists():
        return None
    t = h.read_text().split()
    nx, ny, nz = (int(v) for v in t[:3])
    x0, y0, z0, dx = (float(v) for v in t[3:7])
    return dict(nx=nx, ny=ny, nz=nz, x0=x0, y0=y0, z0=z0, dx=dx,
                ext=(x0, x0 + (nx - 1) * dx, y0, y0 + (ny - 1) * dx, z0, z0 + (nz - 1) * dx))


_MM = {}


def _mm(root, phase, sta):
    k = (str(root), phase, sta)
    if k not in _MM:
        g = grid_hdr(str(root), phase, sta)
        _MM[k] = np.memmap(f"{root}.{phase}.{sta}.time.buf", dtype="<f4", mode="r", shape=(g["nx"], g["ny"], g["nz"]))
        if len(_MM) > 600:
            _MM.pop(next(iter(_MM)))
    return _MM[k]


def inside(box, g):
    return not (box[0] < g[0] or box[1] > g[1] or box[2] < g[2] or box[3] > g[3] or box[4] < g[4] or box[5] > g[5])


def serving_root(box, phase, sta):
    """0 (fine) or 1 (wide) by NLLoc's header rule for this LOCGRID box, None = box rejection."""
    for i, r in enumerate((FINE, WIDE)):
        g = grid_hdr(str(r), phase, sta)
        if g is not None and inside(box, g["ext"]):
            return i
    return None


def tt_point(root, phase, sta, x, y, zp):
    """Trilinear travel time (s) at Lambert x, y (km) and flattened z' (km); NaN outside the grid."""
    g = grid_hdr(str(root), phase, sta)
    if g is None:
        return np.nan
    a = _mm(root, phase, sta)
    fx, fy, fz = (x - g["x0"]) / g["dx"], (y - g["y0"]) / g["dx"], (zp - g["z0"]) / g["dx"]
    if not (0 <= fx <= g["nx"] - 1 and 0 <= fy <= g["ny"] - 1 and 0 <= fz <= g["nz"] - 1):
        return np.nan
    ix, iy, iz = min(int(fx), g["nx"] - 2), min(int(fy), g["ny"] - 2), min(int(fz), g["nz"] - 2)
    wx, wy, wz = fx - ix, fy - iy, fz - iz
    c = np.asarray(a[ix:ix + 2, iy:iy + 2, iz:iz + 2], dtype=float)
    cx = c[0] * (1 - wx) + c[1] * wx
    cy = cx[0] * (1 - wy) + cx[1] * wy
    return float(cy[0] * (1 - wz) + cy[1] * wz)


def tt(phase, sta, x, y, zp, prefer):
    """Travel time from the preferred root (serving root of the v5.1 box), else the other one. (t, root used)."""
    order = (FINE, WIDE) if prefer in (0, None) else (WIDE, FINE)
    for i, r in zip((0, 1) if prefer in (0, None) else (1, 0), order):
        v = tt_point(r, phase, sta, x, y, zp)
        if np.isfinite(v):
            return v, i
    return np.nan, None


def tt_profile(phase, sta, x, y, zps):
    """Travel times along depth at one epicentre (wide grid where the fine one does not reach)."""
    return np.array([tt(phase, sta, x, y, z, 0)[0] for z in zps])


# ------------------------------------------------------------------------------------------------ pnwstore
# Copy of b5_cc_common.py (branch tomo/cc-dt, worktree ~/wt_ccdt), itself from the D-B072 re-timing prt_common.py.
_dbs = {}


def db(year):
    k = (os.getpid(), year)          # one connection per process: never share a connection across fork
    if k not in _dbs:
        _dbs[k] = sqlite3.connect(f"file:{INDEX.format(year=year)}?mode=ro", uri=True)
    return _dbs[k]


WD = {2000: "wd05", 2001: "wd08", 2002: "wd07", 2003: "wd06", 2004: "wd05", 2005: "wd04", 2006: "wd03",
      2007: "wd02", 2008: "wd01", 2009: "wd00", 2010: "wd08", 2011: "wd02", 2012: "wd08", 2013: "wd07",
      2014: "wd06", 2015: "wd05", 2016: "wd05", 2017: "wd07", 2018: "wd02", 2019: "wd01", 2020: "wd00",
      2021: "wd10", 2022: "wd03", 2023: "wd09"}


def real_path(fn):
    return "/auto/pnwstore1-" + WD[int(fn.split("/")[5])] + fn[10:]


def index_rows(net, sta, t_lo, t_hi):
    """tsindex rows of a station overlapping [t_lo, t_hi] (pandas Timestamps), all channels."""
    out = []
    for y in sorted({t_lo.year, t_hi.year}):
        if not os.path.exists(INDEX.format(year=y)):
            continue
        q = ("select network,station,location,channel,starttime,endtime,samplerate,filename,byteoffset,bytes,"
             "timeindex from tsindex where network=? and station=? and starttime<=? and endtime>=?")
        out.append(pd.read_sql(q, db(y), params=[net, sta, t_hi.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                                                 t_lo.strftime("%Y-%m-%dT%H:%M:%S.%f")]))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


BANDS = ("HH", "BH", "EH")


def choose_band(rows):
    """Picker's rule (elep_picker.select_channels hh_bh_eh): first band with a vertical. (band, Z, [horizontals])."""
    if len(rows) == 0:
        return None
    ch = set(rows.channel)
    for b in BANDS:
        z = [c for c in ch if c[:2] == b and c[2] in "Z3"]
        if z:
            zc = b + "Z" if b + "Z" in ch else sorted(z)[0]
            hz = sorted(c for c in ch if c[:2] == b and c[2] in "EN12")
            return b, zc, hz
    return None


def _rate(f, m):
    f = f.astype(float)
    m = m.astype(float)
    return np.where((f > 0) & (m > 0), f * m,
           np.where((f > 0) & (m < 0), -f / np.where(m == 0, 1, m),
           np.where((f < 0) & (m > 0), -m / np.where(f == 0, 1, f),
           np.where((f < 0) & (m < 0), 1.0 / (f * m), 0.0))))


def _b1000_len(b, off):
    n = len(b)
    nb = b[off + 39]
    bo = int(b[off + 46]) << 8 | int(b[off + 47])
    for _ in range(int(nb) + 1):
        if bo == 0 or bo + off + 8 > n:
            break
        btype = int(b[off + bo]) << 8 | int(b[off + bo + 1])
        if btype == 1000:
            return 2 ** int(b[off + bo + 6])
        bo = int(b[off + bo + 2]) << 8 | int(b[off + bo + 3])
    return None


def parse_headers(buf):
    b = np.frombuffer(buf, dtype=np.uint8)
    n = len(b)
    offs, o = [], 0
    while o + 48 <= n:
        rl = _b1000_len(b, o)
        if not rl or o + rl > n:
            break
        offs.append(o)
        o += rl
    if not offs:
        return pd.DataFrame(columns=["offset", "reclen", "t", "npts", "rate", "cha"])
    offs = np.array(offs)
    H = b[offs[:, None] + np.arange(64)]
    be_year = (H[:, 20].astype(int) << 8) | H[:, 21]
    big = (be_year > 1900) & (be_year < 2100)

    def u16(i):
        hi = np.where(big, H[:, i], H[:, i + 1]).astype(np.int64)
        lo = np.where(big, H[:, i + 1], H[:, i]).astype(np.int64)
        return (hi << 8) | lo

    def i16(i):
        v = u16(i)
        return np.where(v >= 32768, v - 65536, v)

    def i32(i):
        a = [H[:, i + k].astype(np.int64) for k in range(4)]
        v = np.where(big, (a[0] << 24) | (a[1] << 16) | (a[2] << 8) | a[3],
                     (a[3] << 24) | (a[2] << 16) | (a[1] << 8) | a[0])
        return np.where(v >= 2 ** 31, v - 2 ** 32, v)

    year, doy = u16(20), u16(22)
    hh, mm, ss = H[:, 24].astype(np.int64), H[:, 25].astype(np.int64), H[:, 26].astype(np.int64)
    fract, npts = u16(28), u16(30)
    rate = _rate(i16(32), i16(34))
    act = H[:, 36]
    tcorr = i32(40)
    bo = u16(46)
    base = (pd.to_datetime((year * 1000 + doy).astype(str), format="%Y%j").values.astype("datetime64[s]")
            .astype(np.int64))
    sec = base + hh * 3600 + mm * 60 + ss
    usec = np.zeros(len(offs))
    rate100 = np.full(len(offs), np.nan)
    cur = bo.copy()
    for _ in range(4):
        valid = (cur > 0) & (cur + 8 <= 4096)
        if not valid.any():
            break
        idx = np.where(valid)[0]
        pos = np.minimum(offs[idx] + cur[idx], n - 8)
        bt = np.where(big[idx], (b[pos].astype(int) << 8) | b[pos + 1], (b[pos + 1].astype(int) << 8) | b[pos])
        nxt = np.where(big[idx], (b[pos + 2].astype(int) << 8) | b[pos + 3], (b[pos + 3].astype(int) << 8) | b[pos + 2])
        m1001 = bt == 1001
        if m1001.any():
            usec[idx[m1001]] = b[pos[m1001] + 5].astype(np.int8)
        m100 = bt == 100
        if m100.any():
            ii, pp = idx[m100], pos[m100]
            raw = np.stack([b[pp + 4 + k] for k in range(4)], 1)
            raw = np.where(big[ii][:, None], raw, raw[:, ::-1]).astype(np.uint8).copy()
            rate100[ii] = raw.view(">f4").ravel()
        newcur = np.zeros_like(cur)
        newcur[idx] = nxt
        cur = newcur
    applied = (act & 0x02) != 0
    corr = np.where(applied, 0, tcorr) * 1e-4
    rate = np.where(np.isfinite(rate100), rate100, rate)
    t = sec.astype(np.float64) + fract * 1e-4 + usec * 1e-6 + corr
    reclen = np.diff(np.append(offs, o))
    return pd.DataFrame({"offset": offs, "reclen": reclen, "t": t, "npts": npts, "rate": rate,
                         "cha": [bytes(r[15:18]).decode("ascii", "replace").strip() for r in H]})


def _timeindex(row):
    keys, offs = [], []
    for x in str(row.timeindex).split(","):
        if "=>" not in x or x.startswith("latest"):
            continue
        a, b_ = x.split("=>")
        keys.append(float(a))
        offs.append(int(b_))
    if not keys:
        keys, offs = [pd.Timestamp(row.starttime, tz="UTC").timestamp()], [int(row.byteoffset)]
    return np.array(keys), np.array(offs)


def _read(fn, off, nb):
    with open(fn, "rb") as f:
        f.seek(off)
        return f.read(nb)


def records_window(row, t_lo, t_hi, chunk=65536):
    """Bytes and headers of the records of one index row overlapping [t_lo, t_hi] (POSIX floats)."""
    fn = real_path(row.filename)
    r0, r1 = int(row.byteoffset), int(row.byteoffset) + int(row.bytes)
    keys, offs = _timeindex(row)
    i = max(0, int(np.searchsorted(keys, t_lo, side="right")) - 1)
    j = int(np.searchsorted(keys, t_hi, side="right"))
    lo_b = int(offs[i])
    hi_b = int(offs[j]) if j < len(offs) else r1
    hi_b = min(max(hi_b, lo_b), r1)
    head = _read(fn, lo_b, 64)
    rl = _b1000_len(np.frombuffer(head, dtype=np.uint8), 0) or 512
    if hi_b - lo_b <= 4 * chunk:
        a, b_ = lo_b, hi_b
    else:
        k1 = keys[i + 1] if i + 1 < len(keys) else pd.Timestamp(row.endtime, tz="UTC").timestamp()
        frac = (t_lo - keys[i]) / max(k1 - keys[i], 1e-3)
        g = lo_b + int(frac * (hi_b - lo_b))
        a = lo_b + ((max(g - chunk, lo_b) - lo_b) // rl) * rl
        b_ = min(a + 2 * chunk + int((t_hi - t_lo) / max(k1 - keys[i], 1e-3) * (hi_b - lo_b)), hi_b)
    h = None
    buf = b""
    for _ in range(12):
        buf = _read(fn, a, b_ - a)
        h = parse_headers(buf)
        if len(h) == 0 and (a, b_) != (lo_b, hi_b):
            a, b_ = lo_b, hi_b
            continue
        if len(h) == 0:
            return b"", h
        tend = h.t.values + h.npts.values / np.where(h.rate.values > 0, h.rate.values, np.nan)
        ok_lo = h.t.values[0] <= t_lo or a <= lo_b
        ok_hi = tend[-1] >= t_hi or b_ >= hi_b
        if ok_lo and ok_hi:
            break
        span = b_ - a
        if not ok_lo:
            a = max(lo_b, a - max(span, chunk))
            a = lo_b + ((a - lo_b) // rl) * rl
        if not ok_hi:
            b_ = min(hi_b, b_ + max(span, chunk))
    tend = h.t.values + h.npts.values / np.where(h.rate.values > 0, h.rate.values, np.nan)
    sel = np.where((h.t.values <= t_hi) & (tend >= t_lo))[0]
    if len(sel) == 0:
        return b"", h.iloc[:0]
    s0, s1 = sel[0], sel[-1]
    sub = h.iloc[s0:s1 + 1].copy()
    o0 = int(sub.offset.iloc[0])
    o1 = int(sub.offset.iloc[-1] + sub.reclen.iloc[-1])
    sub["offset"] -= o0
    return buf[o0:o1], sub


def read_window(rows, cha, t_anchor, t_lo, t_hi):
    """Traces of one channel over [t_lo, t_hi] (POSIX floats) on the record-header time base: the trace holding
    t_anchor is shifted so that the record holding the sample at t_anchor starts at its own header time (D-B072
    definition, as b5_cc_common.read_channel); other segments keep their ObsPy start times. Returns ([(t0, sr,
    data float32)], info)."""
    import obspy
    from obspy import Stream
    info = dict(status="ok", cha=cha)
    rr = rows[rows.channel == cha]
    if len(rr) == 0:
        info["status"] = "no_channel"
        return [], info
    st = Stream()
    hdrs = []
    for _, row in rr.iterrows():
        try:
            buf, h = records_window(row, t_lo, t_hi)
        except Exception as e:
            info["status"] = f"read_error:{type(e).__name__}"
            continue
        if not len(buf):
            continue
        try:
            s = obspy.read(io.BytesIO(buf), format="MSEED")
        except Exception as e:
            info["status"] = f"read_error:{type(e).__name__}"
            continue
        st += s.select(channel=cha)
        hdrs.append(h[h.cha == cha])
    if len(st) == 0:
        if info["status"] == "ok":
            info["status"] = "no_records"
        return [], info
    locs = sorted({tr.stats.location for tr in st})
    if len(locs) > 1:
        info["multi_loc"] = ",".join(locs)
        st = st.select(location=locs[0])
    st.merge(method=-1)          # join contiguous records only (no fill, no overlap change)
    st.sort(["starttime"])
    h = pd.concat(hdrs, ignore_index=True).sort_values("t").drop_duplicates("t")
    shift = 0.0
    cand = [tr for tr in st if tr.stats.starttime.timestamp <= t_anchor <= tr.stats.endtime.timestamp]
    if cand:
        tr = cand[0]
        sr = tr.stats.sampling_rate
        t0 = tr.stats.starttime.timestamp
        hh = h[(h.t >= t0 - 2e-6) & (h.t <= tr.stats.endtime.timestamp + 1.0 / sr)]
        j0 = np.where(np.abs(hh.t.values - t0) < 2e-6)[0]
        if len(j0):
            hh = hh.iloc[j0[0]:]
            cum = np.concatenate([[0], np.cumsum(hh.npts.values)])[:len(hh)]
            keep = cum < tr.stats.npts
            hh, cum = hh[keep], cum[keep]
            dev = hh.t.values - (t0 + cum / sr)
            r = int(np.searchsorted(t0 + cum / sr, t_anchor, side="right") - 1)
            r = min(max(r, 0), len(dev) - 1)
            shift = float(dev[r])
            tr.stats.starttime = tr.stats.starttime + shift
            info["hdr_dev_max_ms"] = float(np.abs(dev - shift).max()) * 1e3
        else:
            info["anchor"] = "anchor_fail"
    else:
        info["anchor"] = "anchor_in_gap"
    info["anchor_shift_ms"] = shift * 1e3
    out = []
    for tr in st:
        tr.trim(obspy.UTCDateTime(t_lo), obspy.UTCDateTime(t_hi))
        if tr.stats.npts < 10:
            continue
        out.append((tr.stats.starttime.timestamp, float(tr.stats.sampling_rate), tr.data.astype(np.float32)))
    if not out:
        info["status"] = "no_samples"
    info["n_seg"] = len(out)
    return out, info


def bandpass(x, sr, band, taper_s=1.0):
    """Demean, detrend, cosine taper, zero-phase 2-corner Butterworth band-pass applied forward and backward."""
    from scipy.signal import butter, sosfiltfilt, detrend
    y = detrend(np.asarray(x, dtype=np.float64), type="linear")
    n = len(y)
    m = int(min(taper_s * sr, n // 4))
    if m > 1:
        w = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        y[:m] *= w
        y[-m:] *= w[::-1]
    fmax = min(band[1], 0.45 * sr)
    sos = butter(2, [band[0], fmax], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def local_time(ts):
    """pandas UTC Timestamp -> 'YYYY-MM-DD HH:MM:SS PST/PDT' (America/Los_Angeles)."""
    if ts is None or pd.isna(ts):
        return ""
    lt = ts.tz_convert("America/Los_Angeles")
    return f"{lt:%Y-%m-%d %H:%M:%S} {lt.tzname()} ({lt:%a})"


# ------------------------------------------------------------------------------------------------ NCEDC (NC, BK)
def read_ncedc(stas, t0, t1):
    """NC/BK stations through NCEDC FDSN, the project's route for these networks (utils/data_client.py: pnwstore does
    not hold NC/BK). One bulk request; per station the picker's band rule (HH > BH > EH). Returns
    {sta: {cha: [(t0, sr, data)]}}, {sta: info}. Times are NCEDC's record times (no re-timing)."""
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    out, info = {}, {}
    if not stas:
        return out, info
    bulk = []
    for s in stas:
        code, net = s.split(".")
        for b in BANDS:
            bulk.append((net, code, "*", b + "?", UTCDateTime(t0), UTCDateTime(t1)))
    st = None
    err = ""
    for attempt in range(2):
        try:
            st = Client("NCEDC", timeout=90).get_waveforms_bulk(bulk)
            break
        except Exception as e:     # FDSNNoDataException included
            err = f"{type(e).__name__}"
            if "NoData" in err:
                break
    for s in stas:
        code, net = s.split(".")
        if st is None:
            info[s] = dict(sta=s, status=f"no data: NCEDC request failed ({err})", source="NCEDC")
            continue
        sub = st.select(network=net, station=code)
        chans = {tr.stats.channel for tr in sub}
        rows = pd.DataFrame({"channel": sorted(chans)})
        cb = choose_band(rows)
        if cb is None:
            info[s] = dict(sta=s, status="no data: NCEDC returned no HH/BH/EH vertical", source="NCEDC")
            continue
        band, zc, hz = cb
        d = {}
        for cha in [zc] + hz:
            ss = sub.select(channel=cha)
            locs = sorted({tr.stats.location for tr in ss})
            ss = ss.select(location=locs[0]) if locs else ss
            ss.merge(method=-1)
            segs = [(tr.stats.starttime.timestamp, float(tr.stats.sampling_rate), tr.data.astype(np.float32))
                    for tr in ss if tr.stats.npts >= 10]
            if segs:
                d[cha] = segs
        out[s] = d
        info[s] = dict(sta=s, band=band, status="ok (NCEDC)" if any(k[-1] in "Z3" for k in d) else "no data: NCEDC",
                       channels=" ".join(sorted(d)), source="NCEDC")
    return out, info
