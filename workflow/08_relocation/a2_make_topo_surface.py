"""LOCTOPO_SURFACE file for NLLoc: GMT earth_relief 01m (m), binary doubles in grd2xyz -ZTLd order, with
the GMT-5 style grdinfo header that NLLoc's read_grd() parses (velmod.c). Masks the location prior above
the solid surface (land or seafloor)."""
from pathlib import Path
import numpy as np, pygmt
OUT = Path("/wd1/mdenolle_data/reloc_v5/topo/topo_01m.bin")
W, E, S, N = -134.0, -118.0, 37.5, 52.5
rel = pygmt.datasets.load_earth_relief("01m", region=[W, E, S, N], registration="gridline")
z = rel.values.astype("f8")[::-1]                 # rows north -> south (T), columns west -> east (L)
ny, nx = z.shape
lon, lat = rel.lon.values, rel.lat.values
dx, dy = float(np.diff(lon).mean()), float(np.diff(lat).mean())
f = OUT.name
hdr = [f"{f}: Title: earth_relief_01m", f"{f}: Command: pygmt load_earth_relief 01m", f"{f}: Remark: elevation m",
       f"{f}: Normal node registration used", f"{f}: grdfile format # double",
       f"{f}: x_min: {lon[0]:.6f} x_max: {lon[-1]:.6f} x_inc: {dx:.10f} name: deg nx: {nx}",
       f"{f}: y_min: {lat[0]:.6f} y_max: {lat[-1]:.6f} y_inc: {dy:.10f} name: deg ny: {ny}",
       f"{f}: z_min: {z.min():.1f} z_max: {z.max():.1f} name: m",
       f"{f}: scale_factor: 1 add_offset: 0"]
Path(f"{OUT}.hdr").write_text("\n".join(hdr) + "\n")
z.astype("<f8").tofile(OUT)
print(OUT, nx, ny)
