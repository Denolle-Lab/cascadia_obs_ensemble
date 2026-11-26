"""Conservative LOCTOPO_SURFACE files for NLLoc's nearest-node lookup (D-B055/D-B056; built and tested 2026-11-24 in the
above-surface root-cause workflow, which found that a2_make_topo_surface.py's 01m file, read by nearest node, let
1,079 B4 and 594 Track A locations sit up to 0.41 km above the bilinear surface).

NLLoc (velmod.c get_surface_z, pix_shift 0.5 for "Normal node registration") returns, for any epicentre, the value
of the nearest node. A location is rejected when elev > value. So if every node holds the MINIMUM of the bilinear
earth_relief 01m surface over the area that NLLoc maps to that node ([i-1/2, i+1/2] x [j-1/2, j+1/2] node spacings),
an accepted location can never be above the bilinear surface. A bilinear function on a rectangle has its minimum at a
corner of the sub-rectangles cut by the 01m grid lines, so the minimum is exact on a lattice of 1/8 of the 01m spacing
(7.5 s) for target spacings of 15 s (3x3 lattice points per node), 30 s (5x5) and 01m (9x9).
A margin (MARGIN_M, default 1 m) covers float32 storage in NLLoc (ps->zdata is float), the %lf (1e-6 deg) and %.8g
rounding of written coordinates, and header x_inc rounding.
EFT version (B4 Earth-flattened frame): NLLoc compares the flattened depth z' with the file; the file holds
h' = Re ln(1 + h/Re) (m, Re = 6371 km), so z' >= -h' <=> z = Re(1 - exp(-z'/Re)) >= -h (monotonic map).

Writes <name>.bin (little-endian doubles, rows north -> south, grd2xyz -ZTLd order, as a2_make_topo_surface.py) and
<name>.bin.hdr (the header read_grd parses) into OUT.
Usage: python a2_make_topo_surface_cons.py [--margin 1.0]   (writes the 15 s files; nlloc_template.in uses the EFT one
for every run: h' <= h everywhere, so it is also safe, by at most 2.2 m, for runs in the true frame)
"""
import argparse
from pathlib import Path
import numpy as np
import pygmt

OUT = Path("/wd1/mdenolle_data/reloc_v5/topo")
W, E, S, N = -134.0, -118.0, 37.5, 52.5
RE_M = 6371000.0


def upsample8(z):
    """Bilinear surface of the 01m grid z (lat ascending, lon ascending) on the 1/8-spacing lattice."""
    ny, nx = z.shape
    t = np.arange(8) / 8.0
    # along lon
    a = z[:, :-1, None] * (1 - t) + z[:, 1:, None] * t          # (ny, nx-1, 8)
    zx = np.concatenate([a.reshape(ny, -1), z[:, -1:]], axis=1)    # (ny, 8(nx-1)+1)
    b = zx[:-1, None, :] * (1 - t)[None, :, None] + zx[1:, None, :] * t[None, :, None]
    return np.concatenate([b.reshape(-1, zx.shape[1]), zx[-1:]], axis=0)


def block_min(z8, step, half):
    """Min of z8 over [s*k - half, s*k + half] (lattice units) for every target node k (edges clipped)."""
    from scipy.ndimage import minimum_filter
    m = minimum_filter(z8, size=2 * half + 1, mode="nearest")
    return m[::step, ::step]


def write(name, zgrid, inc_deg, title, remark):
    """zgrid: lat ascending. Writes the .bin (north -> south) and .hdr in read_grd's format."""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.bin"
    ny, nx = zgrid.shape
    f = p.name
    hdr = [f"{f}: Title: {title}", f"{f}: Command: s2_build_surfaces.py", f"{f}: Remark: {remark}",
           f"{f}: Normal node registration used", f"{f}: grdfile format # double",
           f"{f}: x_min: {W:.6f} x_max: {E:.6f} x_inc: {inc_deg:.15f} name: deg nx: {nx}",
           f"{f}: y_min: {S:.6f} y_max: {N:.6f} y_inc: {inc_deg:.15f} name: deg ny: {ny}",
           f"{f}: z_min: {zgrid.min():.3f} z_max: {zgrid.max():.3f} name: m",
           f"{f}: scale_factor: 1 add_offset: 0"]
    Path(f"{p}.hdr").write_text("\n".join(hdr) + "\n")
    zgrid[::-1].astype("<f8").tofile(p)
    print(p, nx, ny, f"{p.stat().st_size / 1e6:.1f} MB on disk, {nx * ny * 4 / 1e6:.1f} MB in NLLoc (float)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=1.0)
    a = ap.parse_args()
    rel = pygmt.datasets.load_earth_relief("01m", region=[W, E, S, N], registration="gridline")
    assert np.all(np.diff(rel.lat.values) > 0) and np.all(np.diff(rel.lon.values) > 0)
    z = rel.values.astype("f8")
    z8 = upsample8(z)
    print("01m grid", z.shape, "-> 1/8 lattice", z8.shape)
    # sanity: lattice points that are 01m nodes reproduce the grid; a random lattice point matches xarray interp
    assert np.array_equal(z8[::8, ::8], z)
    rng = np.random.default_rng(0)
    iy, ix = rng.integers(0, z8.shape[0], 2000), rng.integers(0, z8.shape[1], 2000)
    ref = rel.interp(lat=("p", S + iy / 480.0), lon=("p", W + ix / 480.0)).values
    print("lattice vs xarray bilinear: max |diff| m =", float(np.nanmax(np.abs(ref - z8[iy, ix]))))
    for name, step, inc in (("15s", 2, 1 / 240),):
        cons = block_min(z8, step, step // 2) - a.margin
        write(f"topo_{name}_cons", cons, inc, f"earth_relief_01m bilinear, min over NLLoc node area, -{a.margin} m",
              "elevation m (true)")
        eft = RE_M * np.log1p(cons / RE_M)
        write(f"topo_{name}_cons_eft", eft, inc,
              f"earth_relief_01m bilinear, min over NLLoc node area, -{a.margin} m, Earth-flattened",
              "flattened elevation m: h' = Re ln(1 + h/Re), Re = 6371 km")


if __name__ == "__main__":
    main()
