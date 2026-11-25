"""D-B028 P-2: map tomoDD outputs from the rescaled (ellipsoid-distance) coordinates back to geographic ones.

Inputs built with forward.coords = ellipsoid (b2_tomodd_inputs.py) carry lat' = c_lat + (M/R)(lat - c_lat) and
lon' = c_lon + (N/R)(lon - c_lon) (inputs/forward.json). This script inverts the map for one run:
  Output_Files/tomoDD.reloc, tomoDD.loc  -> *_geo (columns 2-3 = lat, lon replaced; rest unchanged; x/y/z columns
                                            stay in tomoDD's local Cartesian frame, which is already metric)
  Output_Files/tomoDD.sta                -> tomoDD.sta_geo
  Vp_model.dat, Vs_model.dat + MOD nodes -> model_geo.npz (lon, lat, z of all nodes incl. boundary nodes;
                                            vp, vs [nx, ny, nz]; interior = [1:-1, 1:-1, 1:-1])
For coords = sphere the files are copied with the same names (identity), so downstream code can always read *_geo.

  python b2_tomodd_backtransform.py REGION TAG
"""
import json
import sys
from pathlib import Path

import numpy as np

from b2_fwd_common import read_mod
from manifest import load_params

T = load_params()["tomodd"]


def inverse(lat, lon, fw):
    if fw.get("coords", "sphere") != "ellipsoid":
        return np.asarray(lat, float), np.asarray(lon, float)
    clat, clon = fw["center"]
    return (clat + (np.asarray(lat, float) - clat) / fw["lat_scale_M_over_R"],
            clon + (np.asarray(lon, float) - clon) / fw["lon_scale_N_over_R"])


def forward(lat, lon, fw):
    if fw.get("coords", "sphere") != "ellipsoid":
        return np.asarray(lat, float), np.asarray(lon, float)
    clat, clon = fw["center"]
    return (clat + (np.asarray(lat, float) - clat) * fw["lat_scale_M_over_R"],
            clon + (np.asarray(lon, float) - clon) * fw["lon_scale_N_over_R"])


def table(src, dst, ilat, ilon, fw, fmt="{:.6f}"):
    out = []
    for line in src.read_text().splitlines():
        t = line.split()
        if len(t) <= max(ilat, ilon):
            out.append(line)
            continue
        la, lo = inverse(float(t[ilat]), float(t[ilon]), fw)
        t[ilat], t[ilon] = fmt.format(float(la)), fmt.format(float(lo))
        out.append(" ".join(t))
    dst.write_text("\n".join(out) + "\n")


def read_model_dat(path, nx, ny, nz):
    v = np.array(path.read_text().split(), float)
    return v.reshape(nz, ny, nx).transpose(2, 1, 0)


def main():
    region, tag = sys.argv[1], sys.argv[2]
    base = Path(T["runs"]) / region
    run = base / tag
    fw = json.loads((base / "inputs" / "forward.json").read_text()) if (base / "inputs" / "forward.json").exists() \
        else {"coords": "sphere"}
    o = run / "Output_Files"
    for name in ("tomoDD.reloc", "tomoDD.loc"):
        if (o / name).exists():
            table(o / name, o / f"{name}_geo", 1, 2, fw)
    if (o / "tomoDD.sta").exists():
        table(o / "tomoDD.sta", o / "tomoDD.sta_geo", 1, 2, fw, "{:.5f}")
    m = read_mod(run / "MOD")
    c = fw.get("center", [0.0, 0.0])                 # the map is separable: lat' depends on lat only, lon' on lon
    lat = inverse(m["lat"], np.full_like(m["lat"], c[1]), fw)[0]
    lon = inverse(np.full_like(m["lon"], c[0]), m["lon"], fw)[1]
    nx, ny, nz = len(m["lon"]), len(m["lat"]), len(m["z"])
    out = dict(lon=lon, lat=lat, z=m["z"])
    for ph in ("Vp", "Vs"):
        f = o / f"{ph}_model.dat"
        if f.exists():
            out[ph.lower()] = read_model_dat(f, nx, ny, nz)
    np.savez_compressed(o / "model_geo.npz", **out)
    print(f"{region}/{tag}: coords {fw.get('coords')}; wrote *_geo and model_geo.npz "
          f"(lon {lon[1]:.4f}..{lon[-2]:.4f}, lat {lat[1]:.4f}..{lat[-2]:.4f})")


if __name__ == "__main__":
    main()
