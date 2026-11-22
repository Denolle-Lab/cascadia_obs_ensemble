"""Track A, step A1: 3D P and S velocity grids for NonLinLoc, with topography and bathymetry.

Model
-----
* Vs below the solid surface: CRESCENT CVM Gen0 (data/tomography/CRESCENT_Gen0.nc; 0.2 deg, 0.5 km,
  depth in km below sea level).
  - Offshore, Gen0 fills the water column with a linear ramp from Vs=0 at sea level to the first solid
    value at the seafloor, and the depth-0 slice is 0 offshore. Cells with Vs < 0.05 km/s and cells
    shallower than the cell's mean seafloor are discarded; each column is then extended upward with its
    shallowest kept value.
  - Gen0 does not cover west of ~128.5 W (Axial, Endeavour, western Blanco) or the north/south edges.
    Deep-water nodes outside it use a 1D young-oceanic-crust model referenced to the seafloor (YOUNG_OCEAN);
    other nodes outside it take the nearest Gen0 column. An 80 km linear taper blends the two inside the
    Gen0 footprint. Near Axial, Endeavour and the Blanco transform the 1D model is the published regional
    location model (REGIONAL: Wilcock et al. 2016; Weekly et al. 2013; Kuna et al. 2019), tapered into YOUNG_OCEAN.
* Vp from Vs with Brocher (2005) eq. 9, Vp/Vs floored at VPVS_MIN (default 1.76; pilot Wadati ratios
  1.74-1.77 onshore, Brocher alone gives ~1.70 in the mid crust and put Mount Hood events 14 km too deep). YOUNG_OCEAN carries its own Vp.
* Surface from GMT earth_relief 01m: water (Vp 1.5, Vs 0.4 so no S ray uses it) and air (0.3 km/s)
  above it. Stations sit at their real elevation/depth (GTSRCE zSrce = -elev).
* Depth in the grid is km below sea level. No Earth-flattening transform (paths <= 250 km; see README).

Grid: LAMBERT (WGS-84), 1 km cubic cells, z from -5 to 80 km. Written as NLL SLOW_LEN grids
(.hdr/.buf, z fastest) to OUT/model/cascadia3d.{P,S}.mod.*, plus the station GTSRCE file and QC figures.

Run:  LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/08_relocation/a1_build_model.py
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[2]
OUT = Path("/wd1/mdenolle_data/reloc_v5")
VPVS_MIN = float(os.environ.get("VPVS_MIN", "1.76"))
MODEL = os.environ.get("MODEL", f"model_vpvs{round(VPVS_MIN * 100)}")   # output folder under OUT
QC = REPO / "data/reloc_v5/a1_model"  # QC of the last model built
CRESCENT = REPO / "data/tomography/CRESCENT_Gen0.nc"
STATIONS = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
PICKS = REPO / "data/catalog_v4/Cascadia_picks_catalog_v4.csv"
CATALOG = REPO / "data/catalog_v4/Cascadia_relocated_catalog_v4.csv"

LAT0, LON0, SP1, SP2 = 45.0, -125.8, 41.0, 49.0
DX = 1.0                       # km, cubic cells (Grid2Time needs dx=dy=dz)
Z0, Z1 = -5.0, 80.0            # km below sea level
PAD = 30.0                     # km around events and stations
TAPER = 80.0                   # km, blend width inside the Gen0 footprint
V_WATER_P, V_WATER_S, V_AIR = 1.5, 0.4, 0.3

# 1D young oceanic lithosphere, depth below seafloor (km): Vp, Vs. Layer 2A/2B/3 and mantle from
# typical Juan de Fuca / Gorda refraction profiles (e.g. Canales et al. 2005; Horning et al. 2016).
YOUNG_OCEAN = np.array([
    # h,   Vp,   Vs
    [0.0, 2.6, 1.0],
    [0.5, 4.6, 2.3],
    [1.5, 6.0, 3.3],
    [3.0, 6.6, 3.7],
    [6.0, 7.0, 3.9],
    [6.5, 7.8, 4.4],
    [30., 8.0, 4.5],
    [200, 8.1, 4.55],
])


def _layers(tops, vp, vs, below):
    """Layer table (tops below seafloor) -> node table with steps, continued by `below` (h, Vp, Vs)."""
    rows = []
    for i, (t, p, s) in enumerate(zip(tops, vp, vs)):
        rows.append([t, p, s])
        bot = tops[i + 1] - 1e-3 if i + 1 < len(tops) else None
        if bot is not None:
            rows.append([bot, p, s])
    last = rows[-1][0]
    rows += [r for r in below.tolist() if r[0] > last + 1.0]
    return np.array(rows)


# Regional 1D models (depth below seafloor). Used instead of YOUNG_OCEAN within R_IN km of the site,
# tapering to YOUNG_OCEAN at R_OUT km; only where Gen0 is absent (same weight w_c as YOUNG_OCEAN).
# Axial: Wilcock et al. 2016 Science (Hypoinverse model, MGDS 23844), 0.2 km nodes to 7.6 km.
_ax_z = np.round(0.2 * np.arange(39), 1)
_ax_vp = [2.075, 3.003, 3.493, 4.274, 4.890, 5.152, 5.343, 5.420, 5.466, 5.511, 5.557, 5.602, 5.648, 5.693,
          5.751, 5.814, 5.877, 5.940, 6.003, 6.066, 6.129, 6.192, 6.255, 6.318, 6.381, 6.444, 6.507, 6.570,
          6.633, 6.696, 6.759, 6.822, 6.885, 6.948, 7.011, 7.074, 7.137, 7.592, 8.000]
_ax_vs = [0.835, 1.306, 1.587, 2.090, 2.744, 2.995, 3.106, 3.151, 3.178, 3.204, 3.231, 3.257, 3.284, 3.310,
          3.344, 3.380, 3.417, 3.454, 3.490, 3.527, 3.564, 3.600, 3.637, 3.673, 3.710, 3.747, 3.783, 3.820,
          3.857, 3.893, 3.930, 3.966, 4.003, 4.040, 4.076, 4.113, 4.149, 4.414, 4.651]
# Endeavour: Weekly et al. 2013 G-Cubed (text02, after Cudrak & Clowes 1993), 0.2 km nodes to 6.6 km.
_en_z = np.round(0.2 * np.arange(34), 1)
_en_vp = [2.61, 2.71, 4.7879, 5.0552, 5.2254, 5.6554, 5.9396, 6.0524, 6.1652, 6.2780, 6.5002, 6.5505, 6.5854,
          6.6202, 6.6551, 6.6900, 6.7248, 6.7597, 6.7945, 6.8294, 6.8643, 6.8991, 6.9340, 6.9688, 7.0037,
          7.0385, 7.0734, 7.1083, 7.1431, 7.1780, 7.2128, 7.2477, 7.2826, 8.2000]
_en_vs = [0.87, 0.9033, 2.7836, 2.9391, 3.0380, 3.2880, 3.4533, 3.5188, 3.5844, 3.6500, 3.7792, 3.8084, 3.8287,
          3.8490, 3.8692, 3.8895, 3.9098, 3.9300, 3.9503, 3.9706, 3.9908, 4.0111, 4.0314, 4.0516, 4.0719,
          4.0922, 4.1124, 4.1327, 4.1530, 4.1732, 4.1935, 4.2138, 4.2340, 4.7674]
REGIONAL = {
    # name: (node table, site as a point or a polyline [(lat, lon), ...], R_IN, R_OUT km)
    "axial": (np.c_[_ax_z, _ax_vp, _ax_vs], [(45.95, -130.00)], 25.0, 50.0),
    "endeavour": (np.c_[_en_z, _en_vp, _en_vs], [(47.95, -129.10)], 25.0, 50.0),
    # Blanco: Kuna et al. 2019 Nat. Geosci. Supp. Table 1 (Braunmiller & Nabelek 2008, mantle Vp -3%);
    # trace from the Juan de Fuca Ridge (Cleft) to the Gorda Ridge.
    "blanco": (np.array([[0.0, 4.20, 1.56], [1.5, 6.00, 3.24], [3.3, 6.90, 3.73], [7.0, 7.87, 4.55]]),
               [(44.45, -130.45), (43.95, -128.85), (43.45, -127.55), (43.00, -126.60)], 20.0, 45.0),
}
for _k, (_t, *_r) in REGIONAL.items():
    if _k == "blanco":
        REGIONAL[_k] = (_layers(_t[:, 0], _t[:, 1], _t[:, 2], YOUNG_OCEAN), *_r)
    else:
        REGIONAL[_k] = (np.vstack([_t, YOUNG_OCEAN[YOUNG_OCEAN[:, 0] > _t[-1, 0] + 1.0]]), *_r)

PROJ = Proj(proj="lcc", lat_0=LAT0, lon_0=LON0, lat_1=SP1, lat_2=SP2, ellps="WGS84")
TRANS = (f"TRANSFORM  LAMBERT RefEllipsoid WGS-84  LatOrig {LAT0:f}  LongOrig {LON0:f}  "
         f"FirstStdParal {SP1:f}  SecondStdParal {SP2:f}  RotCW 0.000000")


def brocher_vp(vs):
    """Brocher (2005) eq. 9, Vp from Vs (km/s), valid 0 < Vs < 4.5."""
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    return np.maximum(vp, VPVS_MIN * vs)


def stations_used():
    st = pd.read_csv(STATIONS)
    st["code"] = st.station + "." + st.network
    used = pd.read_csv(PICKS, skipinitialspace=True, usecols=["station"]).station.unique()
    st = st[st.code.isin(used)].copy()
    assert len(st) == len(used), "stations missing from the station table"
    return st


def grid_extent(st):
    cat = pd.read_csv(CATALOG, skipinitialspace=True, usecols=["latitude", "longitude"])
    lon = np.r_[cat.longitude, st.lon]
    lat = np.r_[cat.latitude, st.lat]
    x, y = PROJ(lon, lat)
    x, y = x / 1e3, y / 1e3
    x0, x1 = np.floor(x.min() - PAD), np.ceil(x.max() + PAD)
    y0, y1 = np.floor(y.min() - PAD), np.ceil(y.max() + PAD)
    return x0, x1, y0, y1


def surface_elevation_km(lon, lat):
    import pygmt
    region = [lon.min() - 0.1, lon.max() + 0.1, lat.min() - 0.1, lat.max() + 0.1]
    rel = pygmt.datasets.load_earth_relief("01m", region=region, registration="gridline")
    f = RegularGridInterpolator((rel.lat.values, rel.lon.values), rel.values.astype("f4"),
                                bounds_error=False, fill_value=None)
    return f(np.c_[lat.ravel(), lon.ravel()]).reshape(lat.shape) / 1e3, rel


def clean_crescent(rel):
    """Gen0 Vs with water-column fill removed and columns extended upward; NaN outside its footprint."""
    d = xr.open_dataset(CRESCENT)
    vs = d.Vs.transpose("depth", "latitude", "longitude").values.astype("f8")
    z, la, lo = d.depth.values, d.latitude.values, d.longitude.values
    vs[vs < 0.05] = np.nan
    # mean seafloor/land elevation in each 0.2 deg cell (km, positive up)
    cellmean = ndimage.uniform_filter(rel.values.astype("f8"), size=13)  # ~0.2 deg box at 1'
    e = xr.DataArray(cellmean, coords=rel.coords).interp(lat=la, lon=lo).values / 1e3
    offshore = e < 0
    above_floor = z[:, None, None] < (-e)[None] + 0.25
    vs[above_floor & offshore[None]] = np.nan
    # extend each column upward with its shallowest valid value
    for i in range(len(z) - 2, -1, -1):
        m = np.isnan(vs[i]) & ~np.isnan(vs[i + 1])
        vs[i][m] = vs[i + 1][m]
    foot = ~np.isnan(vs[np.searchsorted(z, 10.0)])
    # also need z beyond 80? Gen0 goes to 100 km, fine
    return z, la, lo, vs, foot


def main():
    OUT.joinpath(MODEL).mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    st = stations_used()
    x0, x1, y0, y1 = grid_extent(st)
    xs = np.arange(x0, x1 + DX / 2, DX)
    ys = np.arange(y0, y1 + DX / 2, DX)
    zs = np.arange(Z0, Z1 + DX / 2, DX)
    nx, ny, nz = len(xs), len(ys), len(zs)
    print(f"grid {nx} x {ny} x {nz} = {nx*ny*nz/1e6:.0f} M nodes, x {x0}..{x1} y {y0}..{y1} km")

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    LON, LAT = PROJ(X * 1e3, Y * 1e3, inverse=True)
    elev, rel = surface_elevation_km(LON, LAT)   # (nx, ny), km positive up

    z, la, lo, vs_c, foot = clean_crescent(rel)

    # footprint distance (km) on the model grid: >0 inside Gen0, <0 outside
    fi = RegularGridInterpolator((la, lo), foot.astype("f4"), bounds_error=False, fill_value=0.0)
    inside = fi(np.c_[LAT.ravel(), LON.ravel()]).reshape(LAT.shape) > 0.5
    d_in = ndimage.distance_transform_edt(inside) * DX
    w_c = np.clip(d_in / TAPER, 0, 1)               # weight of Gen0

    # nearest-footprint fill of Gen0 for nodes outside it (used on land/shelf outside the footprint)
    fill_idx = ndimage.distance_transform_edt(~foot, return_distances=False, return_indices=True)
    vs_nn = vs_c[:, fill_idx[0], fill_idx[1]]
    vs_nn[:, foot] = vs_c[:, foot]
    fC = RegularGridInterpolator((z, la, lo), vs_nn, bounds_error=False, fill_value=None)

    deep_ocean = elev < -1.5
    # weights of the regional 1D models (distance to site point or polyline, km)
    w_reg = {}
    for name, (_, site, r_in, r_out) in REGIONAL.items():
        px, py = PROJ([s[1] for s in site], [s[0] for s in site])
        px, py = np.asarray(px) / 1e3, np.asarray(py) / 1e3
        if len(site) == 1:
            dist = np.hypot(X - px[0], Y - py[0])
        else:
            dist = np.full(X.shape, np.inf)
            for a in range(len(site) - 1):
                ax_, ay_, bx_, by_ = px[a], py[a], px[a + 1], py[a + 1]
                u = np.clip(((X - ax_) * (bx_ - ax_) + (Y - ay_) * (by_ - ay_)) / ((bx_ - ax_)**2 + (by_ - ay_)**2), 0, 1)
                dist = np.minimum(dist, np.hypot(X - ax_ - u * (bx_ - ax_), Y - ay_ - u * (by_ - ay_)))
        w_reg[name] = np.clip((r_out - dist) / (r_out - r_in), 0, 1)
    wsum = sum(w_reg.values())
    scale = np.where(wsum > 1, 1 / np.maximum(wsum, 1e-9), 1.0)
    w_reg = {k: v * scale for k, v in w_reg.items()}
    w_gen = 1 - sum(w_reg.values())
    vp = np.empty((nx, ny, nz), "f4")
    vs = np.empty((nx, ny, nz), "f4")
    # clamp to the Gen0 box (36-50 N, 132-111.2 W): constant extension at its edges, no linear extrapolation
    pts_ll = np.c_[np.clip(LAT.ravel(), la[0], la[-1]), np.clip(LON.ravel(), lo[0], lo[-1])]
    for k, zk in enumerate(zs):
        zc = np.clip(zk, z[0], z[-1])
        vsc = fC(np.c_[np.full(len(pts_ll), zc), pts_ll]).reshape(LAT.shape)
        vpc = brocher_vp(vsc)
        h = zk + elev                                # depth below the solid surface (km)
        vpo = w_gen * np.interp(h, YOUNG_OCEAN[:, 0], YOUNG_OCEAN[:, 1])
        vso = w_gen * np.interp(h, YOUNG_OCEAN[:, 0], YOUNG_OCEAN[:, 2])
        for name, (tab, *_) in REGIONAL.items():
            vpo = vpo + w_reg[name] * np.interp(h, tab[:, 0], tab[:, 1])
            vso = vso + w_reg[name] * np.interp(h, tab[:, 0], tab[:, 2])
        wc = np.where(deep_ocean, w_c, 1.0)          # outside Gen0 on land: nearest fill only
        vpk = wc * vpc + (1 - wc) * vpo
        vsk = wc * vsc + (1 - wc) * vso
        solid = h >= 0
        water = ~solid & (zk >= 0)
        vpk = np.where(solid, vpk, np.where(water, V_WATER_P, V_AIR))
        vsk = np.where(solid, vsk, np.where(water, V_WATER_S, V_AIR))
        vp[:, :, k], vs[:, :, k] = vpk, vsk

    # stations: GTSRCE at true position (zSrce = -elevation), or 50 m below the grid surface when the
    # 1' surface is lower than the station (peaks, canyon walls); station terms absorb the difference.
    fe = RegularGridInterpolator((xs, ys), elev)
    sx, sy = PROJ(st.lon.values, st.lat.values)
    st["grid_surface_km"] = fe(np.c_[sx / 1e3, sy / 1e3])
    st["surface_minus_station_m"] = (st.grid_surface_km * 1e3 - st.elev_m).round(0)
    st["zsrce_km"] = np.maximum(-st.elev_m / 1e3, -st.grid_surface_km + 0.05)
    lines = [f"GTSRCE {r.code} LATLON {r.lat:.6f} {r.lon:.6f} {r.zsrce_km:.4f} 0.0" for r in st.itertuples()]
    (OUT / MODEL / "stations_GTSRCE.in").write_text("\n".join(lines) + "\n")
    st[["code", "lat", "lon", "elev_m", "seafloor", "grid_surface_km", "surface_minus_station_m", "zsrce_km"]].to_csv(
        QC / "stations_vs_grid_surface.csv", index=False)

    # Make the cells around each station solid from the node at/above the station down, so the
    # Grid2Time source cell has rock velocity (a water/air top node would add ~0.4 s to P, more to S).
    n_patched = 0
    for r, x_, y_ in zip(st.itertuples(), sx / 1e3, sy / 1e3):
        k0 = int(np.floor((r.zsrce_km - Z0) / DX))
        i0, j0 = int(np.floor((x_ - x0) / DX)), int(np.floor((y_ - y0) / DX))
        for i in range(max(i0 - 1, 0), min(i0 + 3, nx)):
            for j in range(max(j0 - 1, 0), min(j0 + 3, ny)):
                solid = zs + elev[i, j] >= 0
                kf = int(np.argmax(solid))
                if kf > k0:
                    vp[i, j, k0:kf] = vp[i, j, kf]
                    vs[i, j, k0:kf] = vs[i, j, kf]
                    n_patched += kf - k0
    print("station-cell nodes made solid:", n_patched)

    for ph, v in (("P", vp), ("S", vs)):
        root = OUT / MODEL / f"cascadia3d.{ph}.mod"
        (DX / v).astype("<f4").tofile(f"{root}.buf")    # SLOW_LEN, ix outer, iz fastest
        print("wrote", root)
        Path(f"{root}.hdr").write_text(
            f"{nx} {ny} {nz}  {x0:f} {y0:f} {Z0:f}  {DX:f} {DX:f} {DX:f} SLOW_LEN FLOAT\n{TRANS}\n")

    meta = dict(nx=nx, ny=ny, nz=nz, x0=x0, y0=y0, z0=Z0, dx=DX, trans=TRANS, lat0=LAT0, lon0=LON0,
                sp1=SP1, sp2=SP2, n_stations=len(st), taper_km=TAPER, young_ocean=YOUNG_OCEAN.tolist())
    (QC / "grid_meta.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(QC / "qc_slices.npz", w_reg=sum(w_reg.values()), xs=xs, ys=ys, zs=zs, elev=elev, w_c=w_c, LON=LON, LAT=LAT,
                        vp=vp[:, :, [np.searchsorted(zs, d) for d in (2, 5, 10, 20, 30, 45)]],
                        vs=vs[:, :, [np.searchsorted(zs, d) for d in (2, 5, 10, 20, 30, 45)]],
                        vp_xs=vp[:, np.searchsorted(ys, 0.0), :], vs_xs=vs[:, np.searchsorted(ys, 0.0), :],
                        vp_xs2=vp[:, np.argmin(np.abs(LAT[nx // 2] - 41.3)), :],
                        vs_xs2=vs[:, np.argmin(np.abs(LAT[nx // 2] - 41.3)), :])
    print(st.surface_minus_station_m.describe())
    print("done", OUT / "model")


if __name__ == "__main__":
    main()
