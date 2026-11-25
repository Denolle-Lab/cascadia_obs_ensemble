"""Track B, step B4: wide travel-time grids of every station-phase, consistent with the fine grids and extended to
z' = 121 km, for NLLoc's station-reach fallback and the deep pass (D-B067; v5.1 build, piece 'wide').

Why. NLLoc uses an arrival only if the search box (LOCGRID) lies inside that station-phase's time grid; the fine grids
of b4_time_grids.py cover +-250 km around the station and z' -5..80 km, the box is the 25 km tile +-75 km, so a station
counted only within ~150-175 km of the event, and no event could go below z' = 80 km (round 2, investigation T). With
these whole-lattice grids listed after the fine ones in LOCFILES (a2_run_nlloc.py --time-wide), NLLoc takes per arrival
the first grid that contains the box: the fine grid serves every arrival it can, the wide grid the rest; in the deep
pass (LOCGRID z' -5..120) no fine grid contains the box and the wide grid serves every arrival.

Consistency with the fine grid (round-2 verifier: the first wide grids, a Richardson step on 1 and 2 km runs, differed
from the fine grids by -0.36 to +0.62 s per station-phase at real hypocentres, so a station's predicted time jumped
between neighbouring tiles). Here, per station-phase:
1. raw wide time R on 2 km nodes: Grid2Time (Podvin-Lecomte, no angles) on the whole padded lattice at 1 km and 2 km
   cells, R = T1 + k (T1 - T2) with b4_time_grids.py's k (tomodd.p1.richardson_k), T1 alone within taper_km[0] of the
   source, linear to the combination at taper_km[1];
2. where the fine grid exists (its +-250 km box, z' <= 79 on the wide nodes) the wide grid holds the fine grid's own
   values at the coincident nodes (every 2 km node is a 1 km node of the D-B005 lattice): wide = fine there, up to
   the interpolation between 2 km instead of 1 km nodes;
3. outside it, R + C, C = the continuity correction D = fine - R of the nearest node of the fine region (each index
   clamped into the region: laterally the face node at the same depth, below z' = 79 the node above), with D taken
   on rock nodes only (rock_wide: at least 0.25 km below the model surface; a water, air or near-surface node carries
   D of the nearest rock node below it). D is mostly R's start-up error at the source carried along the rays
   (direction-dependent, -0.25 to +0.39 s within one grid in the tests), so it is carried outward, not recomputed.
   Test (tomo_v1/v51_build/wide, w02/w03; 8 OBS and land station-phases, v0.3.1) against a whole-lattice reference
   built with the fine recipe (0.5 and 1 km runs; it equals the fine grid inside its box), rock nodes outside the
   fine region, |W - ref| p95 (worst grid): 0-30 km beyond the region 0.03 s (z' >= 5) / 0.05 s (z' < 5), 30-120 km
   0.09 / 0.10 s, beyond 120 km (stations 370+ km away) 0.22 / 0.26 s; jump across the region's faces p95 0.003 s;
   real hypocentres outside the fine region (wide-served arrivals of production v0.3.1 events) p95 0.013 s, against
   0.05-0.38 s for R alone. Blending C into per-sector or per-column medians was worse near the region and at the
   hypocentres, better only beyond 120 km.
4. surface_rule: a node just above the solid (not rock, rock below) holds the rock time continued upward (inside the
   fine region through the fine grid's 1 km node in between, so wide = fine on the lower half of that 2 km cell):
   with 2 km nodes the water/air time of such a node pulled interpolated times of events 1-2 km below the surface
   late (wide-served arrivals at events with z' <= 0: median +0.28 s against the fine grid before this rule).
Tests at real hypocentres (w05: 208 station-phases, production B4 iter1 v0.3.1 located events inside the fine grid):
the 17,084 arrivals the wide grid serves |wide - fine| median 0.0014 s, p95 0.009 s, p99 0.13 s (the tail: events
above sea level, where the fine grid's own node above is in the air); per station-phase median <= 0.005 s. Deep pass
of round 2's 30 deep events and 10 controls (w08/w09): 29 of 30 located (16 of the 17 production-REJECTED, at
79-103 km), controls |dz| <= 0.12 km. Below z' 80 against the fine recipe extended to depth on a +-150 km crop
(w06, JCW.UW P and S): |wide - ref| p95 <= 0.05 s, max 0.09 s.
Velocity field: the fine build's whole-lattice 1 km field (work_<TIME_NAME>/field_1.<ph>.buf of b4_time_grids.py:
cell-centred, Earth-flattened SLOW_LEN, merged-surface water/air mask of that build) copied unchanged for z' -5..81 km
(cells 0..85), extended to z' = 123 km (params b4.wide.z_max_km + 2) with the same recipe: inside the tomography box
the model version of that build (v<version>.nc reaches 125 km), outside it the deepest D-B005 layer (80 km) blended
linearly in true depth into ak135 (Kennett et al. 1995; data/tomo_v1/ref_models/ak135_0-410km.csv) over b4.wide.blend_km; the 2 km
field is the mean slowness of the 2 x 2 x 2 1 km cells. Earth-flattening as b4_time_grids.py (z' = Re ln(Re/(Re-z)),
v' = v exp(z'/Re)).
Grid: 2 km nodes of the D-B005 lattice x -513..457, y -709..725, z' -5..121 (486 x 718 x 64, 89 MB).

Builds are never mixed: the grids go to /wd1/mdenolle_data/reloc_v5/wide_<TIME_NAME>/ with a manifest
b4_wide_grids_<TIME_NAME>.manifest.json whose params name the fine build (time_name, its manifest's utc and git commit,
md5 of its fields) and the md5 of this script; a rerun into a folder stamped for another fine build, other
parameters or other code, a fine build that is incomplete, or a fine field that changed since the stamp, is refused.
Needs from the fine build: reloc_v5/<TIME_NAME>/b4_time_grids_<TIME_NAME>.manifest.json (counts ok = station_phases),
work_<TIME_NAME>/params.json (= the manifest's params; "version"), field_1.<P|S>.buf (971 x 1434 x 86) and
field_0.5.P.buf (1941 x 2867 x 171), and the fine grids (their headers give each source position).
Cost (test, 11 processes on a loaded machine): 95-148 s per station-phase (median 114 s), ~2 GB memory per process,
0.8 GB transient disk per running job, 89.3 MB per grid (1,392 station-phases: 124 GB), fields 1.6 GB.

  pixi run -e models python workflow/10_tomography/b4_wide_grids.py TIME_NAME [--nproc 12] [--stations A,B:P]
                                                                               [--phases P,S] [--out DIR]
  -> /wd1/mdenolle_data/reloc_v5/wide_<TIME_NAME>/cascadia3d.<P|S>.<STA>.time.{hdr,buf}, the fields in work/, manifest
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b2_fwd_common import NLL_BIN, NLL_MODEL, RE, mod_velocity, read_nll_hdr  # noqa: E402

P = load_params()
WP = P["b4"]["wide"]
K = P["tomodd"]["p1"]["richardson_k"]
X = P["tomodd"]["models"]
RELOC = Path(P["paths"]["reloc_v5"])
TRANS = ("TRANSFORM  LAMBERT RefEllipsoid WGS-84  LatOrig 45.000000  LongOrig -125.800000  FirstStdParal 41.000000"
         "  SecondStdParal 49.000000  RotCW 0.000000")
G = read_nll_hdr(NLL_MODEL / "cascadia3d.P.mod")      # the D-B005 lattice (1 km, 971 x 1434 x 86, z' -5..80)
assert G["dx"] == 1.0 and G["nz"] == 86, G
DXW = float(WP["node_km"])
assert DXW == 2.0, "the 1 km -> 2 km decimation below assumes 2 km nodes"
NXW, NYW = -(-(G["nx"] - 1) // 2) + 1, -(-(G["ny"] - 1) // 2) + 1   # 486 x 718: x -513..457, y -709..725
NZW = int(round((float(WP["z_max_km"]) - G["z0"]) / DXW)) + 1       # 64: z' -5..121
N1 = (2 * NXW, 2 * NYW, 2 * NZW)                      # 1 km field cells (972 x 1436 x 128), last ones edge copies
NZF = G["nz"]                                         # fine field / fine grid depth cells and nodes (z' -5..80)
KF = (NZF - 1) // 2                                   # last wide node inside the fine depth range (z' 79)
R_T1, R_COMB = (float(v) for v in WP["taper_km"])
AK135 = REPO / WP["ak135"]
assert WP["correction"] == "nearest_rock", WP["correction"]   # assemble() implements only this


def proj():
    from pyproj import Proj
    return Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ak135():
    """(depth, vp, vs) of the ak135 table; first-order discontinuities as repeated depths."""
    a = np.loadtxt(AK135, delimiter=",", comments="#", skiprows=8)
    return a[:, 0], a[:, 1], a[:, 2]


def fine_identity(time_name):
    """The fine build the wide grids belong to: refuse an incomplete one. Returns (identity, work folder, version)."""
    fine, work = RELOC / time_name, RELOC / f"work_{time_name}"
    man = fine / f"b4_time_grids_{time_name}.manifest.json"
    if not man.exists():
        sys.exit(f"{man} missing: the fine build {time_name} is not finished")
    m = json.loads(man.read_text())
    c = m.get("counts", {})
    if c.get("ok") != c.get("station_phases") or not c.get("station_phases"):
        sys.exit(f"fine build {time_name} is incomplete: {c}")
    need = [work / "params.json", *(work / f"field_1.{ph}.buf" for ph in ("P", "S")), work / "field_0.5.P.buf"]
    if not all(f.exists() for f in need):
        sys.exit(f"{work}: the fine build's work files {[f.name for f in need if not f.exists()]} are missing")
    prm = json.loads((work / "params.json").read_text())
    if prm != m["params"]:
        sys.exit(f"{work}/params.json differs from the fine manifest's params: not the field of this build")
    ident = {"time_name": time_name, "fine_manifest_utc": m["utc"], "fine_git_commit": m["git_commit"],
             "fine_params": m["params"],
             "field_md5": {ph: md5(work / f"field_1.{ph}.buf") for ph in ("P", "S")},
             "field_05_P_md5": md5(work / "field_0.5.P.buf")}           # rock_wide's surface test
    return ident, work, prm["version"]


class DeepModel:
    """Velocities below the deepest D-B005 layer (true depth > 80 km), by b4_time_grids.build_field's rule: the model
    version inside the tomography box (lon/lat of its grid; it reaches 125 km), outside it D-B005's 80 km layer blended
    linearly in true depth into ak135 over blend_km (D-B067)."""

    def __init__(self, version):
        import xarray as xr
        from b4_time_grids import dB005
        ds = xr.open_dataset(Path(X["store"]) / X["name"] / f"v{version}.nc")
        self.m = dict(lon=ds.longitude.values, lat=ds.latitude.values, z=ds.depth.values)
        self.mm = {ph: dict(self.m, vp=ds[v].values.transpose(2, 1, 0), vs=ds[v].values.transpose(2, 1, 0))
                   for ph, v in (("P", "Vp"), ("S", "Vs"))}
        self.bg = {"P": dB005("P"), "S": dB005("S")}
        zak, vp, vs = ak135()
        self.ak = (zak, {"P": vp, "S": vs})
        self.z_db, self.z_ak = (float(v) for v in WP["blend_km"])
        assert self.z_db == G["z0"] + G["nz"] - 1, (self.z_db, G)   # D-B005's deepest layer

    def inside(self, lon, lat):
        m = self.m
        return (lon >= m["lon"][0]) & (lon <= m["lon"][-1]) & (lat >= m["lat"][0]) & (lat <= m["lat"][-1])

    def velocity(self, ph, x, y, lon, lat, inside, zt):
        """Velocity (km/s, true) at Lambert x, y (km) / lon, lat of one depth zt (true, km, > 80) layer of points."""
        assert zt > self.z_db, zt
        w = float(np.clip((zt - self.z_db) / (self.z_ak - self.z_db), 0.0, 1.0))
        flat = np.full(x.size, zt)
        vb = {q: (1 - w) * self.bg[q](x.ravel(), y.ravel(), flat).reshape(x.shape)      # D-B005 at 80 km (clamped)
              + w * float(np.interp(zt, self.ak[0], self.ak[1][q])) for q in "PS"}
        v = vb[ph].copy()
        use = inside & (vb["P"] >= 1.6) & (zt >= self.m["z"][0]) & (zt <= self.m["z"][-1])   # build_field's rule
        v[use] = mod_velocity(self.mm[ph], "P", lon[use], lat[use], np.full(use.sum(), zt), clamp_surface=False)
        return v, w


def build_fields(work_fine, version, work):
    """1 km (N1) and 2 km (NXW x NYW x NZW) SLOW_LEN Grid2Time models of the whole lattice to z' = 2 NZW - 5 km:
    cells 0..85 = the fine build's field_1 unchanged, deeper cells by DeepModel (cell centres, Earth-flattened as
    b4_time_grids.build_field). Written as work/mod_<1|2>.<ph>.mod.{hdr,buf}, each renamed when complete."""
    roots = {dx: work / f"mod_{dx}" for dx in (1, 2)}
    if all(Path(f"{roots[dx]}.{ph}.mod.buf").exists() for dx in (1, 2) for ph in ("P", "S")):
        return roots
    dm = DeepModel(version)
    xs, ys = (G[k0] + np.arange(n) + 0.5 for k0, n in (("x0", G["nx"]), ("y0", G["ny"])))
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    lon, lat = proj()(X2 * 1e3, Y2 * 1e3, inverse=True)
    inside = dm.inside(lon, lat)
    stats = {}
    for ph in ("P", "S"):
        f1 = Path(f"{roots[1]}.{ph}.mod.buf")
        if not f1.exists():
            part = work / f"mod_1.{ph}.partial"
            a = np.memmap(part, mode="w+", dtype="<f4", shape=N1)
            src = np.memmap(work_fine / f"field_1.{ph}.buf", dtype="<f4", mode="r", shape=(G["nx"], G["ny"], NZF))
            for i0 in range(0, G["nx"], 64):                      # the fine field, unchanged
                i1 = min(i0 + 64, G["nx"])
                a[i0:i1, :G["ny"], :NZF] = src[i0:i1]
            for k in range(NZF, N1[2]):
                zf = G["z0"] + k + 0.5                              # flattened depth of the cell centre
                zt = RE * (1 - np.exp(-zf / RE))                    # true depth (> 80 km here)
                v, w = dm.velocity(ph, X2, Y2, lon, lat, inside, zt)
                a[:G["nx"], :G["ny"], k] = (1.0 / (v * np.exp(zf / RE))).astype("<f4")
                stats[f"{ph}.{k}"] = [round(float(zt), 2), round(w, 3), round(float(np.median(v[~inside])), 4)
                                      if (~inside).any() else None, round(float(np.median(v[inside])), 4)]
            a[G["nx"]:, :G["ny"], :] = a[G["nx"] - 1:G["nx"], :G["ny"], :]     # edge copies (pad)
            a[:, G["ny"]:, :] = a[:, G["ny"] - 1:G["ny"], :]
            a.flush()
            del a
            part.rename(f1)
            Path(f"{roots[1]}.{ph}.mod.hdr").write_text(
                f"{N1[0]} {N1[1]} {N1[2]}  {G['x0']:f} {G['y0']:f} {G['z0']:f}  1.000000 1.000000 1.000000 "
                f"SLOW_LEN FLOAT\n{TRANS}\n")
            print(f"field 1 km {ph}: {N1}", flush=True)
        f2 = Path(f"{roots[2]}.{ph}.mod.buf")
        if not f2.exists():
            a = np.memmap(f1, dtype="<f4", mode="r", shape=N1)
            b = np.empty((NXW, NYW, NZW), "<f4")
            for i in range(0, NXW, 32):                             # SLOW_LEN at 2 km = 2 x mean slowness of 8 cells
                c = np.asarray(a[2 * i:2 * i + 64], dtype=np.float64)
                b[i:i + 32] = 2.0 * c.reshape(c.shape[0] // 2, 2, NYW, 2, NZW, 2).mean(axis=(1, 3, 5))
            part = work / f"mod_2.{ph}.partial"
            b.tofile(part)
            part.rename(f2)
            Path(f"{roots[2]}.{ph}.mod.hdr").write_text(
                f"{NXW} {NYW} {NZW}  {G['x0']:f} {G['y0']:f} {G['z0']:f}  {DXW:f} {DXW:f} {DXW:f} "
                f"SLOW_LEN FLOAT\n{TRANS}\n")
            print(f"field 2 km {ph}: {b.shape}", flush=True)
    if stats:
        (work / "deep_field_stats.json").write_text(json.dumps(stats))
    return roots


def g2t(wd, tag, label, src, ph, model_root):
    """Grid2Time (Podvin-Lecomte, no angles) on the shared model <model_root>.<ph>.mod, source at the grid-frame
    position src = (x, y, z') of the fine grid's header (GTSRCE XYZ: the fine build's own source); returns the root."""
    out = wd / f"t_{tag}"
    Path(f"{out}.{ph}.mod.buf").touch()               # Grid2Time saves a copy of the model unless this file exists
    ctl = wd / f"g2t_{tag}.in"
    ctl.write_text(f"CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\n"
                   f"GTFILES {model_root} {out} {ph}\nGTMODE GRID3D ANGLES_NO\n"
                   f"GTSRCE {label} XYZ {src[0]:.6f} {src[1]:.6f} {src[2]:.6f} 0.0\nGT_PLFD 1.0e-3 0\n")
    with open(wd / f"g2t_{tag}.log", "w") as f:
        subprocess.run([str(NLL_BIN / "Grid2Time"), ctl.name], cwd=wd, stdout=f, stderr=subprocess.STDOUT, check=True)
    Path(f"{out}.{ph}.mod.buf").unlink()
    return Path(f"{out}.{ph}.{label}.time")


def src_xyz(root):
    """x, y, z' of the source in a time-grid header (line 2)."""
    return [float(v) for v in Path(f"{root}.hdr").read_text().splitlines()[1].split()[1:4]]


def raw_wide(wd, label, src, ph, roots):
    """R = T1 + k (T1 - T2) on the 2 km nodes, T1 within R_T1 of the source, linear to R at R_COMB. Returns
    (R, T1 on the 2 km nodes, root of the 2 km time grid)."""
    r1 = g2t(wd, "1", label, src, ph, roots[1])
    r2 = g2t(wd, "2", label, src, ph, roots[2])
    for r in (r1, r2):
        assert np.allclose(src_xyz(r), src, atol=1e-5), (r, src_xyz(r), src)
    t1 = np.fromfile(f"{r1}.buf", "<f4").reshape(N1)[::2, ::2, ::2].copy()
    Path(f"{r1}.buf").unlink()                        # 0.7 GB
    t2 = np.fromfile(f"{r2}.buf", "<f4").reshape(NXW, NYW, NZW)
    tt = t1 + K[ph] * (t1 - t2)
    d = node_dist(src)
    w = np.clip((d - R_T1) / (R_COMB - R_T1), 0.0, 1.0)
    return (t1 + w * (tt - t1)).astype("<f4"), t1, r2


def node_axes():
    return (G["x0"] + DXW * np.arange(NXW), G["y0"] + DXW * np.arange(NYW), G["z0"] + DXW * np.arange(NZW))


def node_dist(src):
    xs, ys, zs = (a - s for a, s in zip(node_axes(), src))
    return np.sqrt(xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2).astype("<f4")


def fine_on_wide(fine_root, src):
    """The fine grid on the wide nodes it contains: (values, index slices of the fine region in the wide grid, values
    at the 1 km nodes halfway between consecutive wide nodes in depth, z' = -4, -2, ..., 78)."""
    h = read_nll_hdr(fine_root)
    assert h["dx"] == 1.0 and h["z0"] == G["z0"] and h["nz"] == NZF, h
    assert np.allclose(src_xyz(fine_root), src, atol=1e-4), (fine_root, src_xyz(fine_root), src)
    a = np.fromfile(f"{fine_root}.buf", "<f4").reshape(h["nx"], h["ny"], h["nz"])
    sl, fsl = [], []
    for o, n, g0 in ((h["x0"], h["nx"], G["x0"]), (h["y0"], h["ny"], G["y0"])):
        i0 = int(round(o - g0))                           # fine origin, in 1 km lattice nodes
        assert abs(o - g0 - i0) < 1e-6, (fine_root, o, g0)
        w0 = -(-i0 // 2)                                  # first wide node inside: 2 w0 >= i0
        w1 = (i0 + n - 1) // 2                            # last: 2 w1 <= i0 + n - 1
        sl.append(slice(w0, w1 + 1))
        fsl.append(slice(2 * w0 - i0, 2 * w1 - i0 + 1, 2))
    sl.append(slice(0, KF + 1))
    fodd = a[fsl[0], fsl[1], 1:2 * KF:2]
    fsl.append(slice(0, 2 * KF + 1, 2))
    return a[tuple(fsl)], tuple(sl), fodd


def rock_wide(field1_p, field05_p):
    """Rock nodes of the wide grid: the 1 km P cell below/after the node (the extended field) and the 0.5 km P cell
    centred 0.25 km above it (the fine build's field_0.5; z' <= 80) have Vp >= 1.6 km/s (build_field's water/air
    rule): the node is at least 0.25 km below the model's surface. Nodes below the 0.5 km field are rock."""
    a = np.memmap(field1_p, dtype="<f4", mode="r", shape=N1)
    out = np.empty((NXW, NYW, NZW), bool)
    for i in range(0, NXW, 64):
        out[i:i + 64] = 1.0 / np.asarray(a[2 * i:2 * i + 128:2, ::2, ::2]) >= 1.6
    n5 = (2 * (G["nx"] - 1) + 1, 2 * (G["ny"] - 1) + 1, 2 * (NZF - 1) + 1)
    f5 = np.memmap(field05_p, dtype="<f4", mode="r", shape=n5)
    up = np.ones_like(out)
    up[:, :, 0] = False                                   # z' = -5: nothing above in the model
    for i in range(0, NXW, 64):
        c = 1.0 / np.asarray(f5[4 * i:4 * i + 256:4, ::4, 3::4]) >= 1.6   # 0.5 km cell [z' - 0.5, z'] of node k: 4k - 1
        up[i:i + c.shape[0], :c.shape[1], 1:1 + c.shape[2]] = c
    up[:, (n5[1] - 1) // 4 + 1:, :KF + 1] = up[:, (n5[1] - 1) // 4:(n5[1] - 1) // 4 + 1, :KF + 1]   # padded row
    return out & up


def surface_rule(W, Fodd, sl, rock, src):
    """Wide nodes just above the solid (not rock, the node below rock; more than taper_km[1] from the source) hold
    the time of the rock below continued linearly upward instead of their water/air time, so that trilinear
    interpolation at an event between such a node and the rock node 2 km below is not pulled late by the slow cells
    above the surface (2 km nodes reach twice as far up as the fine grid's 1 km nodes). Inside the fine region the
    continuation goes through the fine grid's 1 km node between them, W(k) = 2 F(z'+1) - F(z'+2), so the wide grid
    equals the fine one on [z'+1, z'+2]; outside it W(k) = 2 W(k+1) - W(k+2). Events are never above the surface
    (D-B055/D-B056), so a node replaced here is never itself a hypocentre. Modifies W; returns the count."""
    m = ~rock[:, :, :-2] & rock[:, :, 1:-1] & (node_dist(src)[:, :, :-2] > R_COMB)
    k = np.nonzero(m)
    out = 2.0 * W[k[0], k[1], k[2] + 1].astype(np.float64) - W[k[0], k[1], k[2] + 2]
    ia, ib, ja, jb = sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1
    inn = (k[0] >= ia) & (k[0] <= ib) & (k[1] >= ja) & (k[1] <= jb) & (k[2] <= KF - 1)
    out[inn] = 2.0 * Fodd[k[0][inn] - ia, k[1][inn] - ja, k[2][inn]] - W[k[0][inn], k[1][inn], k[2][inn] + 1]
    W[k] = out.astype("<f4")
    return int(m.sum()), int(inn.sum())


def assemble(R, Fv, sl, rock, Fodd=None, src=None):
    """Wide grid W: the fine values Fv on the fine region sl (its +-250 km box, z' <= 79 on the wide nodes); outside,
    R + C. C is the continuity correction D = Fv - R of the nearest fine-region node (per axis: the node clamped into
    the region, so laterally the face node at the same depth, below z' = 79 the node above), with D taken on rock
    nodes only: a fine-region node that is water, air or within 0.25 km of the surface (rock_wide) carries D of the
    nearest rock node below it. Returns (W, diagnostics)."""
    D = (Fv.astype(np.float64) - R[sl]).astype("<f4")
    rk = rock[sl]
    Dc = D.copy()
    for k in range(D.shape[2] - 2, -1, -1):              # fill non-rock nodes from the rock node below (z' 79 is rock)
        Dc[:, :, k] = np.where(rk[:, :, k], D[:, :, k], Dc[:, :, k + 1])
    ia, ib, ja, jb, kb = sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1, sl[2].stop - 1
    C = Dc[np.ix_(np.clip(np.arange(NXW), ia, ib) - ia, np.clip(np.arange(NYW), ja, jb) - ja,
                  np.clip(np.arange(NZW), 0, kb))]
    W = (R + C).astype("<f4")
    W[sl] = Fv
    v = D[rk]
    diag = {"D_median": round(float(np.median(v)), 4), "D_p05": round(float(np.percentile(v, 5)), 4),
            "D_p95": round(float(np.percentile(v, 95)), 4), "D_absmax_rock": round(float(np.abs(v).max()), 4),
            "nonrock_filled": int((~rk).sum())}
    if Fodd is not None:
        diag["surface_nodes"], diag["surface_nodes_fine_region"] = surface_rule(W, Fodd, sl, rock, src)
    return W, diag


def job(a):
    label, ph, fine_dir, roots, rock_file, work, out = a
    dst = out / f"cascadia3d.{ph}.{label}.time"
    if Path(f"{dst}.hdr").exists():
        d = work / f"{label}.{ph}.diag.json"
        return label, ph, "exists", json.loads(d.read_text()) if d.exists() else None
    t0 = time.time()
    fine_root = fine_dir / f"cascadia3d.{ph}.{label}.time"
    src = src_xyz(fine_root)                          # the fine build's own source position (x, y, z')
    wd = work / f"{label}.{ph}"
    wd.mkdir(parents=True, exist_ok=True)
    R, _, r2 = raw_wide(wd, label, src, ph, roots)
    Fv, sl, Fodd = fine_on_wide(fine_root, src)
    rock = np.load(rock_file)
    W, diag = assemble(R, Fv, sl, rock, Fodd, src)
    # NLLoc rejects or drops negative predicted times without a message: stop here instead
    assert np.isfinite(W).all() and W.min() >= 0, (label, ph, float(np.nanmin(W)))
    keep = np.ones(W.shape, bool)                     # wide = fine on the fine region, but surface_rule's nodes
    keep[:, :, :-2] = ~(~rock[:, :, :-2] & rock[:, :, 1:-1] & (node_dist(src)[:, :, :-2] > R_COMB))
    assert np.array_equal(W[sl][keep[sl]], Fv[keep[sl]]), (label, ph)
    W.tofile(f"{dst}.buf")
    diag.update(fine_nodes=[s.stop - s.start for s in sl], wall_s=round(time.time() - t0, 1))
    (work / f"{label}.{ph}.diag.json").write_text(json.dumps(diag))
    shutil.copy(f"{r2}.hdr", f"{dst}.hdr")            # last: marks the grid complete (2 km header, same source)
    shutil.rmtree(wd)
    return label, ph, "ok", diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("time_name", help="the fine build (b4_time_grids.py TIME_NAME) these wide grids belong to")
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--stations", help="comma-separated STA or STA:PH (test; default: every station-phase of the fine "
                                       "build)")
    ap.add_argument("--phases", default="P,S")
    ap.add_argument("--out", help="output folder (test; default reloc_v5/wide_<TIME_NAME>)")
    a = ap.parse_args()
    ident, work_fine, version = fine_identity(a.time_name)
    fine_dir = RELOC / a.time_name
    out = Path(a.out) if a.out else RELOC / f"wide_{a.time_name}"
    prm = {"time_name": a.time_name, "fine": ident, "model_version": version, "wide": WP, "richardson_k": K,
           "lattice": {"x0": G["x0"], "y0": G["y0"], "z0": G["z0"], "nodes": [NXW, NYW, NZW], "node_km": DXW,
                       "field_cells_1km": list(N1)}, "eft": True, "centred": True,
           "code_md5": md5(Path(__file__))}               # a resume with edited code would mix grids: refused
    prm = json.loads(json.dumps(prm))
    # never mix builds: an existing folder must carry the same stamp (fine build, its fields, parameters)
    stamp = out / "params.json"
    old = [json.loads(p.read_text())["params"] for p in out.glob("b4_wide_grids_*.manifest.json")]
    old += [json.loads(stamp.read_text())] if stamp.exists() else []
    if any(o != prm for o in old) or (not old and any(out.glob("*.time.hdr"))):
        sys.exit(f"{out} holds wide grids of another fine build or other parameters (or of unknown origin): "
                 f"use a new folder. Stamp: {old[:1]}")
    out.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(prm, indent=1))
    work = out / "work"
    work.mkdir(exist_ok=True)
    roots = build_fields(work_fine, version, work)
    rock_file = work / "rock.npy"
    if not rock_file.exists():
        np.save(work / "rock.partial.npy", rock_wide(f"{roots[1]}.P.mod.buf", work_fine / "field_0.5.P.buf"))
        (work / "rock.partial.npy").rename(rock_file)
    sp = sorted(h.name[len("cascadia3d."):-len(".time.hdr")].split(".", 1) for h in fine_dir.glob("cascadia3d.*.time.hdr"))
    if a.stations:                                    # STA (both phases) or STA:PH items
        want = [w.split(":") for w in a.stations.split(",")]
        sp = [s for s in sp if any(w[0] == s[1] and (len(w) == 1 or w[1] == s[0]) for w in want)]
    sp = [s for s in sp if s[0] in a.phases.split(",")]
    jobs = [(lab, ph, fine_dir, roots, rock_file, work, out) for ph, lab in sp]
    print(f"{len(jobs)} station-phases -> {out}", flush=True)
    res = []
    with ProcessPoolExecutor(a.nproc) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=1)):
            res.append(r)
            if i % 25 == 0 or i == len(jobs) - 1:
                print(f"{time.strftime('%H:%M:%S')} {i + 1}/{len(jobs)} {r[:3]} {r[3]}", flush=True)
    fine = {h.name for h in fine_dir.glob("cascadia3d.*.time.hdr")}
    wide = {h.name for h in out.glob("cascadia3d.*.time.hdr")}
    diag = {f"{ph}.{lab}": d for lab, ph, _, d in res}
    walls = [d["wall_s"] for d in diag.values() if d and "wall_s" in d]
    dmed = {ph: [d["D_median"] for k, d in diag.items() if d and k[0] == ph] for ph in ("P", "S")}
    summ = {ph: {"D_median_p05_p50_p95": [round(float(np.percentile(v, q)), 4) for q in (5, 50, 95)]}
            for ph, v in dmed.items() if v}
    print(f"wide grids {len(wide)}; fine grids without a wide one: {len(fine - wide)}; D = fine - raw (median per "
          f"grid) {summ}", flush=True)
    write_manifest(f"b4_wide_grids_{a.time_name}", out,
                   inputs=[fine_dir / f"b4_time_grids_{a.time_name}.manifest.json",
                           *(work_fine / f"field_1.{ph}.buf" for ph in ("P", "S")),
                           Path(X["store"]) / X["name"] / f"v{version}.nc", NLL_MODEL / "cascadia3d.P.mod.buf",
                           NLL_MODEL / "cascadia3d.S.mod.buf", AK135],
                   params=prm,
                   counts={"station_phases": len(jobs), "ok": sum(r[2] == "ok" for r in res),
                           "exists": sum(r[2] == "exists" for r in res), "wide_grids": len(wide),
                           "fine_without_wide": len(fine - wide),
                           "wall_s_median": round(float(np.median(walls)), 1) if walls else None},
                   notes={"per_grid": diag, "summary": summ,
                          "about": "per grid: D = fine - raw wide time on the fine region's rock nodes (median, p05, "
                                   "p95, max |D|), non-rock fine-region nodes whose D comes from the rock node below, "
                                   "fine-region size in wide nodes, wall time of the job (s)"})


if __name__ == "__main__":
    main()
