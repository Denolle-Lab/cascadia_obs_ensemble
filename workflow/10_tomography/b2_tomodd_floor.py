"""Track B v0.4.1 (D-B083): surface.dat of the D-B081 tomoDD floor for one inversion stage (M0 or a tile), and the
start-hypocentre check.

The floor (D-B081, tomoDD-SE 2026-11-25 freeze, change 9): when surface.dat is in the run folder, an event update that
would put the source above the floor depth of the nearest surface.dat node stops 1 m below it; the event is kept and
logged (>>>Floor). surface.dat: header `nla nlo la0 dla lo0 dlo`, then nla rows (south to north) of nlo floor depths
(km, positive down), in tomoDD's own lat/lon frame; at most 1001 x 1001 nodes.

Floor values (params tomodd.floor): the conservative floor of the D-B068 merged solid surface, as D-B056 does for
NonLinLoc: each node holds the exact minimum of b4_surface.surface_m over the node's cell (node +- half a spacing)
minus 1 m (b4_surface.py floor --spacing S, true frame, the .bin without _eft). The 1001-node limit sets the spacing:
the stage's interior node box + margin_deg must fit (tiles: 30 s, about 0.9 x 0.65 km; M0: 45 s).
Node placement: the floor grid's nodes x0 + k dx (NonLinLoc nearest-node rule, read_floor shift 0.5) are written as
they are, then rescaled about the stage centre exactly like every other tomoDD input (D-B028 P-2,
b2_fwd_mods.ellipsoid_coords, centre from inputs/forward.json). The map is linear, so tomoDD's nearest node
(floordep: floor((lat - la0) / dla + 0.5)) is the node NonLinLoc's rule picks at the true position. Depths are written
in km with 4 decimals rounded down (deeper): never above the cell minimum - 1 m. Depth is not rescaled (tomoDD depth =
true depth below sea level, as event.dat).

  python b2_tomodd_floor.py write REGION [--inputs DIR] [--out DIR]   # <runs>/REGION/inputs/surface.dat, floor.json
  python b2_tomodd_floor.py check REGION [--inputs DIR] [--out DIR]   # start hypocentres (event.dat) vs the true
                                             # surface and the floor -> floor_check.json, floor_check_events.csv
Run with the default env python (LD_LIBRARY_PATH=.pixi/envs/default/lib).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402
from b2_fwd_mods import ellipsoid_coords  # noqa: E402
from b4_catalog import floor_at, read_floor  # noqa: E402

P = load_params()
T = P["tomodd"]
FL = T["floor"]
RUNS = Path(T["runs"])


def stage(region, inputs=None):
    R = T["regions"][region]
    s = R.get("floor_spacing_s")
    if s is None:
        raise SystemExit(f"tomodd.regions.{region} has no floor_spacing_s: not a floored stage")
    inp = Path(inputs) if inputs else RUNS / region / "inputs"
    fw = json.loads((inp / "forward.json").read_text())
    assert fw["coords"] == "ellipsoid", f"{region}: the writer assumes the P-2 rescale (forward.coords ellipsoid)"
    return R, inp, tuple(fw["center"]), Path(FL["files"][int(s)])


def floor_nodes(fl, lon0, lon1, lat0, lat1):
    """Index range of the floor grid's nodes whose cells cover [lon0, lon1] x [lat0, lat1] (NonLinLoc's rule)."""
    ix0 = int(np.floor(fl["shift"] + (lon0 - fl["x0"]) / fl["dx"]))
    ix1 = int(np.floor(fl["shift"] + (lon1 - fl["x0"]) / fl["dx"]))
    k0 = int(np.floor(fl["shift"] + (lat0 - fl["y0"]) / fl["dy"]))
    k1 = int(np.floor(fl["shift"] + (lat1 - fl["y0"]) / fl["dy"]))
    assert ix0 >= 0 and k0 >= 0 and ix1 < fl["nx"] and k1 < fl["ny"], "stage box outside the floor grid"
    return ix0, ix1, k0, k1


def cmd_write(region, inputs=None, outdir=None):
    info, (path, inp, out, od) = write_surface(region, inputs, outdir)
    print(json.dumps(info, indent=1))
    write_manifest(f"b2_tomodd_floor_{region}", od, inputs=[path, Path(f"{path}.hdr"), inp / "forward.json"],
                   outputs=[out, od / "floor.json"], params={"tomodd.floor": FL, "region": region}, counts=info)


_FLOORS = {}


def write_surface(region, inputs=None, outdir=None, cache=False, floor_file=None):
    """surface.dat + floor.json of a stage in its rescaled frame (centre from <inputs>/forward.json), no manifest
    (the ensemble writes one per member stage). cache: keep the floor grid in memory between calls. floor_file: a copy
    of the configured floor grid to read instead (the ensemble's pinned copy; same .bin/.hdr pair)."""
    R, inp, center, path = stage(region, inputs)
    if floor_file is not None:
        path = Path(floor_file)
    od = Path(outdir) if outdir else inp
    od.mkdir(parents=True, exist_ok=True)
    fl = _FLOORS.get(path) if cache else None
    if fl is None:
        fl = read_floor(path)
        if cache:
            _FLOORS[path] = fl
    m = float(FL["margin_deg"])
    lo, la = R["node_lon"], R["node_lat"]                 # the stage's interior node box: every reachable position
    ix0, ix1, k0, k1 = floor_nodes(fl, lo[0] - m, lo[1] + m, la[0] - m, la[1] + m)
    nlo, nla = ix1 - ix0 + 1, k1 - k0 + 1
    if not (2 <= nla <= 1001 and 2 <= nlo <= 1001):
        raise SystemExit(f"{region}: {nla} x {nlo} floor nodes at {path.name}: tomoDD reads at most 1001 x 1001")
    lon_n = fl["x0"] + (np.arange(ix0, ix1 + 1) + 0.5 - fl["shift"]) * fl["dx"]
    lat_n = fl["y0"] + (np.arange(k0, k1 + 1) + 0.5 - fl["shift"]) * fl["dy"]
    z = fl["z"][fl["ny"] - 1 - np.arange(k0, k1 + 1)][:, ix0:ix1 + 1].astype(float)   # rows south -> north
    assert np.isfinite(z).all(), "floor grid has no value over the stage"
    dep = np.ceil(-z / 1000.0 * 1e4 - 1e-6) / 1e4      # km, positive down, 4 decimals rounded down (deeper)
    la_w, lo_w = ellipsoid_coords(lat_n, lon_n, *center)
    dla, dlo = (la_w[-1] - la_w[0]) / (nla - 1), (lo_w[-1] - lo_w[0]) / (nlo - 1)
    head = f"{nla} {nlo} {la_w[0]:.8f} {dla:.10f} {lo_w[0]:.8f} {dlo:.10f}"
    out = od / "surface.dat"
    out.write_text(head + "\n" + "\n".join(" ".join(f"{v:.4f}" for v in row) for row in dep) + "\n")
    info = dict(region=region, file=str(path), spacing_s=float(fl["dx"] * 3600), nodes=[nla, nlo],
                true_lat=[float(lat_n[0]), float(lat_n[-1])], true_lon=[float(lon_n[0]), float(lon_n[-1])],
                tomodd_header=head, center=list(center), margin_deg=m, air_dep_km=float(FL["air_dep_km"]),
                first_node_index=[int(k0), int(ix0)],
                floor_depth_km_min_max=[float(dep.min()), float(dep.max())])
    info["inputs"] = str(inp)
    (od / "floor.json").write_text(json.dumps(info, indent=1))
    return info, (path, inp, out, od)


def read_surface_dat(path):
    """surface.dat as tomoDD reads it (REAL*4) and its floordep rule."""
    lines = Path(path).read_text().splitlines()
    h = lines[0].split()
    nla, nlo = int(h[0]), int(h[1])
    la0, dla, lo0, dlo = (np.float32(v) for v in h[2:6])
    z = np.array([[np.float32(v) for v in r.split()] for r in lines[1:1 + nla]], dtype=np.float32)
    assert z.shape == (nla, nlo)
    return dict(nla=nla, nlo=nlo, la0=la0, dla=dla, lo0=lo0, dlo=dlo, z=z)


def floordep(S, lat, lon):
    """tomoDD-SE floordep (change 10): nearest node, edge node outside the grid; float32 arithmetic."""
    lat, lon = np.asarray(lat, np.float32), np.asarray(lon, np.float32)
    i = np.floor((lat - S["la0"]) / S["dla"] + np.float32(0.5)).astype(int)
    j = np.floor((lon - S["lo0"]) / S["dlo"] + np.float32(0.5)).astype(int)
    i, j = np.clip(i, 0, S["nla"] - 1), np.clip(j, 0, S["nlo"] - 1)
    return S["z"][i, j].astype(float), i, j


def read_event_dat(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return pd.DataFrame(dict(ID=[int(r[9]) for r in rows], lat_w=[float(r[2]) for r in rows],
                             lon_w=[float(r[3]) for r in rows], depth_w=[float(r[4]) for r in rows]))


def cmd_check(region, inputs=None, outdir=None):
    """Every start hypocentre (event.dat, tomoDD frame) against the floor tomoDD will use, and the same event
    (events.csv, true frame) against the merged surface itself (b4_surface.surface_m, exact) and the floor grid at
    its true position (NonLinLoc's rule). Lists start hypocentres below the true surface but above the floor."""
    from b4_surface import surface_m
    R, inp, center, path = stage(region, inputs)
    od = Path(outdir) if outdir else inp
    S = read_surface_dat(od / "surface.dat")
    ev = pd.read_csv(inp / "events.csv")[["ID", "lat", "lon", "depth_bsl_km", "region"]]
    ew = read_event_dat(inp / "event.dat")
    d = ev.merge(ew, on="ID", how="outer", validate="1:1")
    assert d.notna().all().all(), "events.csv and event.dat differ"
    d["surface_depth_km"] = -surface_m(d.lon.values, d.lat.values) / 1000.0      # true frame, exact surface
    fz, i, j = floordep(S, d.lat_w.values, d.lon_w.values)                       # tomoDD frame, as tomoDD computes it
    d["floor_tomodd_km"] = fz
    fl = read_floor(path)
    d["floor_true_km"] = -floor_at(fl, d.lat.values, d.lon.values) / 1000.0     # true frame, NLLoc rule
    k0, ix0 = json.loads((od / "floor.json").read_text())["first_node_index"]
    ut = fl["shift"] + (d.lat.values - fl["y0"]) / fl["dy"] - k0              # NLLoc's node coordinate, as written
    vt = fl["shift"] + (d.lon.values - fl["x0"]) / fl["dx"] - ix0
    it, jt = np.floor(ut).astype(int), np.floor(vt).astype(int)
    d["edge_node"] = (i == 0) | (i == S["nla"] - 1) | (j == 0) | (j == S["nlo"] - 1)
    d["below_surface"] = d.depth_bsl_km > d.surface_depth_km
    d["below_floor"] = d.depth_w >= d.floor_tomodd_km                            # tomoDD's test: tdep >= zfloor
    d["floor_minus_surface_m"] = 1000 * (d.floor_tomodd_km - d.surface_depth_km)  # >= 1 m (conservative)
    d["depth_below_surface_km"] = d.depth_bsl_km - d.surface_depth_km
    bad = d[d.below_surface & ~d.below_floor]
    above = d[~d.below_surface]
    # tomoDD's node vs NonLinLoc's at the true position. tomoDD works in REAL*4 (lat/lon ulp ~ 4-8e-6 deg, about 0.001
    # of a 30 s cell) on event.dat's 6-decimal rescaled coordinates, so a position within ~0.002 cells of a cell edge can
    # take the neighbouring node: a tie, covered by the floor's 1 m margin unless the surface rises > 1 m within that
    # distance. Any other mismatch is an error.
    mis = (it != i) | (jt != j)
    edge = np.minimum(np.abs(ut - np.rint(ut)), np.abs(vt - np.rint(vt)))   # distance to the nearest cell edge (cells)
    tie = mis & (edge < 0.005)
    node_mismatch = int((mis & ~tie).sum())
    q = lambda s: [round(float(np.percentile(s, p)), 1) for p in (50, 90, 99, 100)] if len(s) else None
    out = dict(region=region, events=len(d), surface_dat=str(od / "surface.dat"), floor_file=str(path),
               below_true_surface=int(d.below_surface.sum()), above_true_surface=len(above),
               below_surface_but_above_floor=len(bad),
               above_surface_and_above_floor=int((~d.below_surface & ~d.below_floor).sum()),
               floor_node_tomodd_vs_true_mismatch=node_mismatch, floor_node_edge_ties=int(tie.sum()),
               floor_node_edge_tie_dist_cells_max=float(edge[tie].max()) if tie.any() else None,
               floor_node_edge_tie_floor_diff_m=[round(1000 * float(v), 2) for v in
                                                 (d.floor_tomodd_km - d.floor_true_km)[tie]],
               events_on_edge_nodes=int(d.edge_node.sum()),
               floor_minus_surface_m_p50_p90_p99_max=q(d.floor_minus_surface_m),
               floor_minus_surface_min_m=round(float(d.floor_minus_surface_m.min()), 3),
               bad_by_region=bad.region.value_counts().to_dict(),
               bad_depth_below_surface_m=q(1000 * bad.depth_below_surface_km),
               bad_height_above_floor_m=q(1000 * (bad.floor_tomodd_km - bad.depth_w)),
               above_surface_by_region=above.region.value_counts().to_dict(),
               above_surface_height_m=q(-1000 * above.depth_below_surface_km))
    cols = ["ID", "region", "lat", "lon", "depth_bsl_km", "surface_depth_km", "floor_tomodd_km", "floor_true_km",
            "below_surface", "below_floor", "floor_minus_surface_m"]
    flag = d[(d.below_surface & ~d.below_floor) | ~d.below_surface]
    flag[cols].round(5).to_csv(od / "floor_check_events.csv", index=False)
    (od / "floor_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if node_mismatch:
        raise SystemExit(f"{node_mismatch} events: tomoDD's floor node differs from NonLinLoc's at the true position")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "check"])
    ap.add_argument("region")
    ap.add_argument("--inputs", help="inputs folder of another region with the same frame (test: blanco_m0t/inputs)")
    ap.add_argument("--out", help="write surface.dat / floor.json / the check here instead of the inputs folder")
    a = ap.parse_args()
    {"write": cmd_write, "check": cmd_check}[a.cmd](a.region, a.inputs, a.out)


if __name__ == "__main__":
    main()
