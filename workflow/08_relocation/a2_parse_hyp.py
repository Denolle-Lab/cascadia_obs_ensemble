"""Parse NLLoc *.hyp summary files of a run into one CSV and compare with catalog v4.

Usage: python a2_parse_hyp.py RUN_NAME  ->  /wd1/mdenolle_data/reloc_v5/runs/RUN_NAME/locations.csv
Columns: v4 ID, status, maximum-likelihood origin, expectation depth, RMS, phases used, gap, distances,
68% horizontal ellipse (min/max), 3D ellipsoid semi-axes, depth sigma (sqrt CovZZ), v4 values, surface check.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
B = Path("/wd1/mdenolle_data/reloc_v5")


def kv(tokens):
    return {tokens[i]: tokens[i + 1] for i in range(len(tokens) - 1)}


def parse(path):
    rows, cur = [], None
    for line in open(path):
        tok = line.split()
        if not tok:
            continue
        k = tok[0]
        if k == "NLLOC":
            cur = {"status": tok[2].strip('"')}
        elif cur is None:
            continue
        elif k == "PUBLIC_ID":
            cur["ID"] = int(tok[1].split("_")[1])
        elif k == "GEOGRAPHIC":
            cur["time"] = f"{tok[2]}-{tok[3]}-{tok[4]}T{tok[5]}:{tok[6]}:{float(tok[7]):09.6f}"
            d = kv(tok[8:])
            cur.update(lat=float(d["Lat"]), lon=float(d["Long"]), depth=float(d["Depth"]))
        elif k == "QUALITY":
            d = kv(tok[1:])
            cur.update(rms=float(d["RMS"]), nphs=int(d["Nphs"]), gap=float(d["Gap"]), dist_min=float(d["Dist"]))
        elif k == "STATISTICS":
            d = kv(tok[1:])
            cur.update(sig_z=np.sqrt(float(d["ZZ"])), ell_len3=float(d["Len3"]))
        elif k == "STAT_GEOG":
            d = kv(tok[1:])
            cur.update(exp_lat=float(d["ExpectLat"]), exp_lon=float(d["Long"]), exp_depth=float(d["Depth"]))
        elif k == "QML_OriginQuality":
            d = kv(tok[1:])
            cur.update(n_sta=int(d["usedStaCt"]), dist_max=float(d["maxDist"]), dist_med=float(d["medDist"]))
        elif k == "QML_OriginUncertainty":
            d = kv(tok[1:])
            cur.update(h_min=float(d["minHorUnc"]), h_max=float(d["maxHorUnc"]))
        elif k == "END_NLLOC":
            rows.append(cur)
            cur = None
    return rows


REGIONS = [  # name, lat0, lat1, lon0, lon1 (first match wins; v4 epicentre)
    ("Axial", 45.7, 46.2, -130.3, -129.7), ("Endeavour", 47.7, 48.3, -129.5, -128.7),
    ("Explorer", 48.3, 51.5, -131.5, -127.5), ("Blanco", 42.8, 44.7, -130.8, -126.4),
    ("Gorda", 40.5, 42.8, -128.0, -124.9), ("Mendocino", 39.5, 40.5, -128.5, -123.5),
    ("JdF ridge/plate", 44.7, 48.3, -131.5, -127.3), ("Offshore forearc", 40.5, 51.5, -127.3, -124.0),
    ("Puget/Georgia", 46.8, 49.5, -123.5, -121.8), ("Cascades", 40.5, 49.5, -122.4, -120.5),
]


def region(lat, lon):
    for name, a, b, c, d in REGIONS:
        if a <= lat < b and c <= lon < d:
            return name
    return "Other onshore"


def main(run):
    rd = B / "runs" / run
    rows = []
    for f in sorted(rd.glob("*/loc.sum.grid0.loc.hyp")):
        for r in parse(f):
            r["tile"] = f.parent.name
            rows.append(r)
    loc = pd.DataFrame(rows)
    v4 = pd.read_csv(REPO / "data/catalog_v4/Cascadia_relocated_catalog_v4.csv", skipinitialspace=True)
    v4 = v4.rename(columns={"latitude": "v4_lat", "longitude": "v4_lon", "depth(km)": "v4_depth",
                            "spatial_sigma(m)": "v4_sigma_m"})
    v4["v4_npick"] = v4.Num_p_picks + v4.Num_s_picks
    loc = loc.merge(v4[["ID", "v4_lat", "v4_lon", "v4_depth", "v4_npick", "v4_sigma_m"]], on="ID", how="left")
    # surface elevation under the new and v4 epicentres
    import pygmt
    rel = pygmt.datasets.load_earth_relief("01m", region=[-134, -118, 37.5, 52.5], registration="gridline")
    for pre, la, lo in (("", "lat", "lon"), ("v4_", "v4_lat", "v4_lon")):
        e = rel.interp(lat=("p", loc[la].values), lon=("p", loc[lo].values)).values / 1e3
        loc[f"{pre}surface_km"] = e
    loc["above_surface"] = loc.depth < -loc.surface_km - 1e-3
    loc["v4_above_surface"] = loc.v4_depth < -loc.v4_surface_km - 1e-3
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (loc.lat, loc.lon, loc.v4_lat, loc.v4_lon))
    loc["shift_h_km"] = 2 * R * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2)**2 +
                                                 np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2)**2))
    loc["shift_z_km"] = loc.depth - loc.v4_depth
    loc["region"] = [region(a, b) for a, b in zip(loc.v4_lat, loc.v4_lon)]
    loc.to_csv(rd / "locations.csv", index=False)
    ok = loc[loc.status == "LOCATED"]
    print(f"{len(loc)} events, {len(ok)} located; above surface: new {ok.above_surface.mean():.1%}, "
          f"v4 (same events) {ok.v4_above_surface.mean():.1%}")
    key = "tile" if loc.tile.nunique() <= 12 else "region"
    print(ok.groupby(key).agg(n=("ID", "size"), rms=("rms", "median"), nphs=("nphs", "median"),
                                 v4_npick=("v4_npick", "median"), h_max=("h_max", "median"),
                                 sig_z=("sig_z", "median"), dz=("shift_z_km", "median"),
                                 dh=("shift_h_km", "median"), depth=("depth", "median"),
                                 v4_depth=("v4_depth", "median"), above=("above_surface", "mean"),
                                 v4_above=("v4_above_surface", "mean")).round(2).to_string())
    anss_check(ok, rd)


def anss_check(loc, rd):
    """Depth vs ANSS/ComCat (PNSN, NCSN) for events matched within 4 s and 30 km."""
    a = pd.read_csv(REPO / "data/magnitude_v4/anss_2010-16.csv")
    a["t"] = pd.to_datetime(a.time.str.rstrip("Z"), format="ISO8601", utc=True)
    a = a.rename(columns={"latitude": "a_lat", "longitude": "a_lon", "depth": "a_depth"}).sort_values("t")
    l = loc.assign(t=pd.to_datetime(loc.time, utc=True)).sort_values("t")
    m = pd.merge_asof(l, a[["t", "a_lat", "a_lon", "a_depth", "net"]], on="t",
                      tolerance=pd.Timedelta("4s"), direction="nearest").dropna(subset=["a_lat"])
    d = np.hypot((m.lat - m.a_lat) * 111.2, (m.lon - m.a_lon) * 111.2 * np.cos(np.radians(m.lat)))
    m = m[d < 30].copy()
    m["dz_new"] = m.depth - m.a_depth
    m["dz_v4"] = m.v4_depth - m.a_depth
    m.to_csv(rd / "anss_matched.csv", index=False)
    print(f"\nANSS depth check: {len(m)} matched (median depth; median |dz|)")
    key = "tile" if m.tile.nunique() <= 12 else "region"
    print(m.groupby(key).agg(n=("ID", "size"), anss=("a_depth", "median"), v4=("v4_depth", "median"),
                                new=("depth", "median"), v4_absdz=("dz_v4", lambda s: s.abs().median()),
                                new_absdz=("dz_new", lambda s: s.abs().median())).round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
