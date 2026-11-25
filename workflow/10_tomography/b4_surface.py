"""Track B, v5.1: the merged solid surface (D-B068) and the conservative NLLoc topography floor built from it (D-B056).

SURFACE (interface I1). surface_m(lon, lat) -> numpy array, elevation in m, positive up, of ONE solid surface: GMRT v4
patches where available, GMT earth_relief 01m (gridline) elsewhere. Every user (the NLLoc floor, the D-B055 above-
surface test of b4_catalog.py, the water/rock boundary of the B4 slowness fields) must call this function, so that all
of them see the same surface. Rule, point by point (vectorized):
* sources in priority order: the GMRT patches sorted by nominal cell size sqrt(dx dy) in metres at the patch centre
  latitude, finest first (file name breaks ties), then earth_relief 01m;
* a source covers a point when lon0 <= lon <= lon1 and lat0 <= lat <= lat1 (its first and last grid lines, inclusive)
  and the four nodes of its bilinear cell around the point are finite. The cell is the one whose lower-left node is
  the last grid line <= the point (numpy searchsorted 'right'), except on the last grid line, which uses the cell
  below/left of it;
* the value is the bilinear interpolation of the FIRST covering source on its own grid (no blending).
Patch edges and overlaps: where patches overlap, the finest one that covers the point wins; across the edge of a patch
(its bounding grid lines, or the edge of its finite rows: the first two and sometimes the last row of every GMRT file
are all NaN) the surface jumps to the next source. The surface is therefore continuous inside each source and
discontinuous along patch edges (GMRT minus earth_relief there: b4_surface_check.py stats). Points outside the
earth_relief region (params b4.surface.earth_relief.region) give NaN.
Sources are pinned copies, never the atlas folder: `install` copies the GMRT files (md5 checked against params
b4.surface.gmrt_md5) to <dir>/gmrt/ and writes the earth_relief subset to <dir>/<earth_relief.cache> (md5 of its values
checked against params); the module then reads only those copies (netCDF4 + numpy; no pygmt, so it also runs in the
models env). <dir> = params b4.surface.dir, or the environment variable B4_SURFACE_DIR (tests).
SOURCE (str, for manifests): the rule, the directory and every file with its md5; computed on first access.

FLOOR (D-B056 applied to the merged surface). NLLoc reads LOCTOPO_SURFACE by nearest node (velmod.c get_surface_z,
pix_shift 0.5, node = (int)(0.5 + (x - x_min) / x_inc), float zdata) and rejects a trial point whose elevation is above
that value. So every node must hold the minimum of the surface over the closed rectangle of points NLLoc maps to it
(node +- half a spacing; clipped to the earth_relief region, which contains every LOCGRID box with ~0.9 deg to spare),
minus params b4.surface.floor.margin_m (float32 storage, coordinate rounding), then Earth-flattened:
h' = Re ln(1 + h / Re) (Re = 6371 km), never above h, so the same file is safe in the flattened (B4) and true frames.
The minimum is EXACT: inside a node rectangle, the rectangle's edges and every grid line of every source that crosses
it cut it into sub-rectangles, open segments and points, on each of which the selected source and its bilinear cell
are constant. A bilinear function is linear along each axis, so its minimum over a closed sub-rectangle, a closed
segment or a point is at a corner. The floor value is the minimum over the corners of every closed sub-rectangle
(source selected inside it), every closed segment (source selected on its open part) and every cut point (source
selected there): each piece of the piecewise-bilinear, discontinuous surface contributes its closure, so both sides of
every patch edge are included. The selection uses the same code path as surface_m. Checked by brute force and against a
reproduction of NLLoc's float32 lookup in b4_surface_check.py.
Output (as workflow/08_relocation/a2_make_topo_surface_cons.py): <out>.bin little-endian doubles, rows north -> south
(GMT grd2xyz -ZTLd order), and <out>.bin.hdr (GMT-5 style header parsed by NLLoc read_grd), for the flattened frame
(<name>_eft.bin, the one NLLoc uses for B4) and the true frame (<name>.bin).

Usage (default env python; nice -n 10):
  python b4_surface.py install [--gmrt-src DIR] [--dir DIR]       # copies GMRT + earth_relief cache into DIR
  python b4_surface.py floor [--spacing 15] [--out PATH_eft.bin] [--nproc 12] [--manifest full|local]
  python b4_surface.py info
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params  # noqa: E402

RE_M = 6371000.0
P = (load_params().get("b4") or {}).get("surface") or {}


def _dir():
    return Path(os.environ.get("B4_SURFACE_DIR") or P["dir"])


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def values_md5(z):
    return hashlib.md5(np.ascontiguousarray(np.asarray(z, "<f8")).tobytes()).hexdigest()


@dataclass
class Src:
    name: str
    lon: np.ndarray      # ascending node coordinates (deg)
    lat: np.ndarray      # ascending
    z: np.ndarray        # (nlat, nlon) float64, m, positive up
    cellok: np.ndarray   # (nlat-1, nlon-1) bool: the four nodes of the cell are finite
    res_m: float         # nominal cell size, for the priority order
    md5: str

    @property
    def box(self):
        return self.lon[0], self.lon[-1], self.lat[0], self.lat[-1]

    def cx(self, x):
        return np.clip(np.searchsorted(self.lon, x, "right") - 1, 0, self.lon.size - 2)

    def cy(self, y):
        return np.clip(np.searchsorted(self.lat, y, "right") - 1, 0, self.lat.size - 2)


def _read_nc(path, var):
    import netCDF4
    with netCDF4.Dataset(path) as n:
        n.set_auto_mask(False)
        lon = np.asarray(n.variables["lon"][:], "f8")
        lat = np.asarray(n.variables["lat"][:], "f8")
        z = np.asarray(n.variables[var][:], "f8")
        fill = getattr(n.variables[var], "_FillValue", None)
    if fill is not None and np.isfinite(fill):
        z = np.where(z == fill, np.nan, z)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    if lon[0] > lon[-1]:
        lon, z = lon[::-1], z[:, ::-1]
    assert np.all(np.diff(lon) > 0) and np.all(np.diff(lat) > 0), f"{path}: coordinates not monotonic"
    assert z.shape == (lat.size, lon.size), f"{path}: {z.shape} vs {lat.size} x {lon.size}"
    return lon, lat, np.ascontiguousarray(z)


def _make(name, lon, lat, z, digest):
    ok = np.isfinite(z)
    cellok = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    latc = np.radians(0.5 * (lat[0] + lat[-1]))
    dx = np.median(np.diff(lon)) * 111194.9 * np.cos(latc)
    dy = np.median(np.diff(lat)) * 111194.9
    return Src(name, lon, lat, z, cellok, float(np.sqrt(dx * dy)), digest)


_SOURCES = None


def sources():
    """GMRT patches finest first, then earth_relief (cached per process). Checks every md5 against params."""
    global _SOURCES
    if _SOURCES is not None:
        return _SOURCES
    d = _dir()
    gm = P.get("gmrt_md5") or {}
    patches = []
    for name, want in sorted(gm.items()):
        f = d / "gmrt" / name
        got = md5(f)
        assert got == want, f"{f}: md5 {got} != params b4.surface.gmrt_md5 {want}"
        lon, lat, z = _read_nc(f, P.get("gmrt_var", "altitude"))
        patches.append(_make(name, lon, lat, z, got))
    extra = sorted(set(p.name for p in (d / "gmrt").glob("*.nc")) - set(gm)) if (d / "gmrt").is_dir() else []
    assert not extra, f"{d / 'gmrt'}: files not in params b4.surface.gmrt_md5 (not used, remove or add): {extra}"
    patches.sort(key=lambda s: (s.res_m, s.name))
    er = P["earth_relief"]
    f = d / er["cache"]
    lon, lat, z = _read_nc(f, "z")
    got = values_md5(z)
    assert got == er["values_md5"], f"{f}: values md5 {got} != params b4.surface.earth_relief.values_md5"
    assert np.isfinite(z).all(), f"{f}: NaN in earth_relief"
    W, E, S, N = er["region"]
    assert np.isclose(lon[0], W) and np.isclose(lon[-1], E) and np.isclose(lat[0], S) and np.isclose(lat[-1], N)
    _SOURCES = patches + [_make(f"earth_relief_{er['resolution']}_g", lon, lat, z, got)]
    return _SOURCES


def _eval(srcs, x, y):
    """Pointwise rule of the module docstring. x, y: 1-D float64. Returns (value m, index of the source used or -1)."""
    out = np.full(x.shape, np.nan)
    used = np.full(x.shape, -1, np.int16)
    todo = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    for k, s in enumerate(srcs):
        if todo.size == 0:
            break
        x0, x1, y0, y1 = s.box
        xs, ys = x[todo], y[todo]
        inb = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        if not inb.any():
            continue
        p, xs, ys = todo[inb], xs[inb], ys[inb]
        ix, iy = s.cx(xs), s.cy(ys)
        ok = s.cellok[iy, ix]
        if not ok.any():
            continue
        p, xs, ys, ix, iy = p[ok], xs[ok], ys[ok], ix[ok], iy[ok]
        t = (xs - s.lon[ix]) / (s.lon[ix + 1] - s.lon[ix])
        u = (ys - s.lat[iy]) / (s.lat[iy + 1] - s.lat[iy])
        z = s.z
        out[p] = ((1 - u) * ((1 - t) * z[iy, ix] + t * z[iy, ix + 1]) + u * ((1 - t) * z[iy + 1, ix] + t * z[iy + 1, ix + 1]))
        used[p] = k
        done = np.zeros(todo.size, bool)
        done[np.flatnonzero(inb)[ok]] = True
        todo = todo[~done]
    return out, used


def surface_m(lon, lat):
    """Merged solid surface (m, positive up) at (lon, lat) (broadcast arrays, degrees). NaN outside the region."""
    x, y = np.broadcast_arrays(np.asarray(lon, "f8"), np.asarray(lat, "f8"))
    v, _ = _eval(sources(), x.ravel(), y.ravel())
    return v.reshape(x.shape)


def surface_source(lon, lat):
    """Name of the source surface_m uses at each point ('' outside the region)."""
    x, y = np.broadcast_arrays(np.asarray(lon, "f8"), np.asarray(lat, "f8"))
    srcs = sources()
    _, k = _eval(srcs, x.ravel(), y.ravel())
    names = np.array([s.name for s in srcs] + [""], dtype=object)
    return names[k].reshape(x.shape)


def __getattr__(name):
    if name == "SOURCE":
        srcs = sources()
        files = "; ".join(f"{s.name} ({s.res_m:.0f} m, md5 {s.md5})" for s in srcs)
        return (f"D-B068 merged solid surface (b4_surface.py, {_dir()}): first covering source of [{files}] "
                "(GMRT finest first, then earth_relief 01m gridline; covering = inside the source's grid lines with "
                "4 finite bilinear nodes), bilinear on that source's grid, no blending")
    raise AttributeError(name)


# ---------------------------------------------------------------------------------------------------- exact floor
def _minval(srcs, xr, xlo, xhi, yr, ylo, yhi):
    """Tensor grid of reference points (yr x xr): source and bilinear cell selected at each reference point (same rule
    as _eval), and the minimum of that cell's bilinear function over the corners (xlo|xhi) x (ylo|yhi)."""
    out = np.full((yr.size, xr.size), np.inf)
    free = np.ones(out.shape, bool)
    for s in srcs:
        x0, x1, y0, y1 = s.box
        ca = np.flatnonzero((xr >= x0) & (xr <= x1))
        rb = np.flatnonzero((yr >= y0) & (yr <= y1))
        if ca.size == 0 or rb.size == 0:
            continue
        ix, iy = s.cx(xr[ca]), s.cy(yr[rb])
        sel = s.cellok[np.ix_(iy, ix)] & free[np.ix_(rb, ca)]
        if not sel.any():
            continue
        lx, dx = s.lon[ix], s.lon[ix + 1] - s.lon[ix]
        ly, dy = s.lat[iy], s.lat[iy + 1] - s.lat[iy]
        t0, t1 = (xlo[ca] - lx) / dx, (xhi[ca] - lx) / dx
        u0, u1 = (ylo[rb] - ly) / dy, (yhi[rb] - ly) / dy
        for a in (t0, t1, u0, u1):   # the reference cell must lie inside the source cell (lattice holds its lines)
            assert a.min() > -1e-9 and a.max() < 1 + 1e-9, "lattice misses a source grid line"
            np.clip(a, 0.0, 1.0, out=a)
        u0, u1 = u0[:, None], u1[:, None]
        z = s.z
        z00, z10 = z[np.ix_(iy, ix)], z[np.ix_(iy, ix + 1)]
        z01, z11 = z[np.ix_(iy + 1, ix)], z[np.ix_(iy + 1, ix + 1)]
        b0, b1 = z00 + (z10 - z00) * t0, z00 + (z10 - z00) * t1      # lower row of the cell at t0, t1
        c0, c1 = z01 + (z11 - z01) * t0, z01 + (z11 - z01) * t1      # upper row
        v = np.minimum(np.minimum(b0 + (c0 - b0) * u0, b0 + (c0 - b0) * u1),
                       np.minimum(b1 + (c1 - b1) * u0, b1 + (c1 - b1) * u1))
        blk, fr = out[np.ix_(rb, ca)], free[np.ix_(rb, ca)]
        blk[sel] = v[sel]
        fr[sel] = False
        out[np.ix_(rb, ca)], free[np.ix_(rb, ca)] = blk, fr
        if not free.any():
            break
    assert not free.any(), f"{int(free.sum())} reference points covered by no source"
    return out


def _reduce(a, X, e, axis, lines):
    """Min of `a` over each closed node cell [e_i, e_i+1] along `axis`. Elements along that axis are the lattice
    lines X (lines=True; a line on a cell edge belongs to both cells) or the intervals between them (lines=False)."""
    ie = np.searchsorted(X, e)
    assert np.array_equal(X[ie], e), "cell edges missing from the lattice"
    r = np.minimum.reduceat(a, ie[:-1], axis=axis)
    if lines:
        r = np.minimum(r, np.take(a, ie[1:], axis=axis))
    return r


def _floor_block(args):
    """Exact minimum of the merged surface over the closed node cells with x edges ex and y edges ey."""
    ex, ey = args
    srcs = sources()
    X, Y = [ex], [ey]
    for s in srcs:
        x0, x1, y0, y1 = s.box
        if x1 < ex[0] or x0 > ex[-1] or y1 < ey[0] or y0 > ey[-1]:
            continue
        X.append(s.lon[(s.lon > ex[0]) & (s.lon < ex[-1])])
        Y.append(s.lat[(s.lat > ey[0]) & (s.lat < ey[-1])])
    X, Y = np.unique(np.concatenate(X)), np.unique(np.concatenate(Y))
    xc, yc = 0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:])
    Xl, Xh, Yl, Yh = X[:-1], X[1:], Y[:-1], Y[1:]
    parts = [  # (array over (y refs, x refs), y are lines?, x are lines?)
        (_minval(srcs, xc, Xl, Xh, yc, Yl, Yh), False, False),   # closed sub-rectangles
        (_minval(srcs, xc, Xl, Xh, Y, Y, Y), True, False),       # closed segments on y lines
        (_minval(srcs, X, X, X, yc, Yl, Yh), False, True),       # closed segments on x lines
        (_minval(srcs, X, X, X, Y, Y, Y), True, True),           # cut points
    ]
    res = None
    for a, ylines, xlines in parts:
        r = _reduce(_reduce(a, X, ex, 1, xlines), Y, ey, 0, ylines)
        res = r if res is None else np.minimum(res, r)
    return res, X.size, Y.size


def node_axis(lo, hi, spacing_s):
    n = int(round((hi - lo) * 3600 / spacing_s)) + 1
    nodes = lo + np.arange(n) * spacing_s / 3600.0
    edges = np.clip(lo + (np.arange(n + 1) - 0.5) * spacing_s / 3600.0, lo, hi)
    return nodes, edges


def floor_min(spacing_s, nproc=1, block=None):
    """Exact minimum of surface_m over every NLLoc node cell of a spacing_s-arcsecond node grid on the earth_relief
    region. Returns (nodes_lon, nodes_lat, zmin (nlat, nlon), lattice size)."""
    W, E, S, N = P["earth_relief"]["region"]
    xn, xe = node_axis(W, E, spacing_s)
    yn, ye = node_axis(S, N, spacing_s)
    block = block or max(16, int(round(960 / spacing_s)))   # ~0.27 deg blocks
    jobs, where = [], []
    for j0 in range(0, yn.size, block):
        j1 = min(j0 + block, yn.size)
        for i0 in range(0, xn.size, block):
            i1 = min(i0 + block, xn.size)
            jobs.append((xe[i0:i1 + 1], ye[j0:j1 + 1]))
            where.append((j0, j1, i0, i1))
    out = np.full((yn.size, xn.size), np.nan)
    nlat = 0
    if nproc > 1:
        with ProcessPoolExecutor(nproc) as ex:
            res = ex.map(_floor_block, jobs, chunksize=4)
            for (j0, j1, i0, i1), (r, nx, ny) in zip(where, res):
                out[j0:j1, i0:i1] = r
                nlat += (nx - 1) * (ny - 1)
    else:
        for (j0, j1, i0, i1), job in zip(where, jobs):
            r, nx, ny = _floor_block(job)
            out[j0:j1, i0:i1] = r
            nlat += (nx - 1) * (ny - 1)
    assert np.isfinite(out).all()
    return xn, yn, out, nlat


def write_grid(path, zgrid, region, inc_deg, title, remark):
    """zgrid: lat ascending. Writes <path> (doubles, north -> south) and <path>.hdr in NLLoc read_grd's format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    W, E, S, N = region
    ny, nx = zgrid.shape
    f = p.name
    hdr = [f"{f}: Title: {title}", f"{f}: Command: b4_surface.py floor", f"{f}: Remark: {remark}",
           f"{f}: Normal node registration used", f"{f}: grdfile format # double",
           f"{f}: x_min: {W:.6f} x_max: {E:.6f} x_inc: {inc_deg:.15f} name: deg nx: {nx}",
           f"{f}: y_min: {S:.6f} y_max: {N:.6f} y_inc: {inc_deg:.15f} name: deg ny: {ny}",
           f"{f}: z_min: {zgrid.min():.3f} z_max: {zgrid.max():.3f} name: m",
           f"{f}: scale_factor: 1 add_offset: 0"]
    Path(f"{p}.hdr").write_text("\n".join(hdr) + "\n")
    np.ascontiguousarray(zgrid[::-1], "<f8").tofile(p)
    print(p, nx, ny, f"{p.stat().st_size / 1e6:.1f} MB on disk, {nx * ny * 4 / 1e6:.1f} MB in NLLoc (float)")
    return p


def eft(h):
    """Earth-flattened elevation h' = Re ln(1 + h/Re) (m); never above h."""
    return RE_M * np.log1p(np.asarray(h, "f8") / RE_M)


# ---------------------------------------------------------------------------------------------------- CLI
def cmd_install(a):
    """Pinned copies: GMRT files (md5 checked) and the earth_relief subset (values md5 checked when params has it)."""
    d = Path(a.dir or _dir())
    (d / "gmrt").mkdir(parents=True, exist_ok=True)
    src = Path(a.gmrt_src or P["gmrt_src"])
    for name, want in sorted((P.get("gmrt_md5") or {}).items()):
        got = md5(src / name)
        assert got == want, f"{src / name}: md5 {got} != params {want} (the atlas data changed: check before use)"
        dst = d / "gmrt" / name
        if not (dst.exists() and md5(dst) == want):
            shutil.copy2(src / name, dst)
        assert md5(dst) == want
        print("gmrt", dst, want)
    er = P["earth_relief"]
    import pygmt
    import netCDF4
    rel = pygmt.datasets.load_earth_relief(er["resolution"], region=er["region"], registration=er["registration"])
    lon, lat, z = rel.lon.values.astype("f8"), rel.lat.values.astype("f8"), rel.values.astype("f8")
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    got = values_md5(z)
    want = er.get("values_md5")
    assert want in (None, got), f"earth_relief values md5 {got} != params {want} (GMT server data changed?)"
    out = d / er["cache"]
    tmp = out.with_suffix(".tmp.nc")
    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as n:
        n.createDimension("lon", lon.size)
        n.createDimension("lat", lat.size)
        n.createVariable("lon", "f8", ("lon",))[:] = lon
        n.createVariable("lat", "f8", ("lat",))[:] = lat
        v = n.createVariable("z", "f8", ("lat", "lon"), zlib=True)
        v[:] = z
        v.units = "m"
        n.source = (f"pygmt.datasets.load_earth_relief('{er['resolution']}', region={er['region']}, "
                    f"registration='{er['registration']}') (pygmt {pygmt.__version__}); {rel.attrs.get('history', '')}")
        n.vertical_datum = str(rel.attrs.get("vertical_datum", ""))
        n.values_md5 = got
    tmp.replace(out)
    print("earth_relief", out, z.shape, "values md5", got, "file md5", md5(out))


def cmd_floor(a):
    S_ = P["floor"]
    spacing = float(a.spacing or S_["spacing_s"])
    margin = float(S_["margin_m"] if a.margin is None else a.margin)
    out_eft = Path(a.out or S_["file"])
    assert out_eft.name.endswith("_eft.bin"), "--out must end in _eft.bin (the true-frame file drops _eft)"
    out_true = out_eft.with_name(out_eft.name.replace("_eft.bin", ".bin"))
    srcs = sources()
    t = time.time()
    xn, yn, zmin, nlat = floor_min(spacing, a.nproc)
    region = P["earth_relief"]["region"]
    inc = spacing / 3600.0
    cons = zmin - margin
    what = f"D-B068 merged surface (GMRT + earth_relief 01m), exact min over NLLoc node cell, -{margin} m"
    write_grid(out_true, cons, region, inc, what, "elevation m (true)")
    write_grid(out_eft, eft(cons), region, inc, what + ", Earth-flattened",
               "flattened elevation m: h' = Re ln(1 + h/Re), Re = 6371 km")
    counts = {"spacing_s": spacing, "margin_m": margin, "nx": int(xn.size), "ny": int(yn.size),
              "lattice_subrectangles": int(nlat), "seconds": round(time.time() - t, 1),
              "nlloc_float_MB": round(xn.size * yn.size * 4 / 1e6, 1),
              "min_surface_m": float(zmin.min()), "max_surface_m": float(zmin.max())}
    print(counts)
    params = {"b4.surface": P, "SOURCE": __getattr__("SOURCE")}
    ins = [Path(_dir()) / "gmrt" / s.name for s in srcs[:-1]] + [Path(_dir()) / P["earth_relief"]["cache"]]
    if a.manifest == "full":
        from manifest import write_manifest
        write_manifest(out_eft.stem, out_eft.parent, inputs=ins, outputs=[out_eft, out_true], params=params,
                       counts=counts)
    else:   # tests: manifest next to the outputs only (no append to the shared manifests.jsonl)
        import json
        rec = {"name": out_eft.stem, "argv": sys.argv, "params": params, "counts": counts,
               "inputs": {str(p): md5(p) for p in ins},
               "outputs": {str(p): md5(p) for p in (out_eft, out_true)}}
        (out_eft.parent / f"{out_eft.stem}.manifest.json").write_text(json.dumps(rec, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("install")
    s.add_argument("--gmrt-src")
    s.add_argument("--dir")
    s = sub.add_parser("floor")
    s.add_argument("--spacing", type=float)
    s.add_argument("--margin", type=float)
    s.add_argument("--out")
    s.add_argument("--nproc", type=int, default=12)
    s.add_argument("--manifest", choices=["full", "local"], default="full")
    sub.add_parser("info")
    a = ap.parse_args()
    if a.cmd == "install":
        cmd_install(a)
    elif a.cmd == "floor":
        cmd_floor(a)
    else:
        for k, s in enumerate(sources()):
            print(k, s.name, f"{s.res_m:.1f} m", s.box, s.z.shape, f"cells ok {s.cellok.mean():.4f}", s.md5)
        print(__getattr__("SOURCE"))


if __name__ == "__main__":
    main()
