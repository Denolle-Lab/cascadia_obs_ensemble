"""Track B: model v0.4.1 (D-B083) = the merged v0.4.1 inversion (v0.4.0 re-inverted with the D-B081 floor) exported
with the D-B076 product conventions (as b2_model_v032.py does for v0.3.2).

Input: the dry-run merge of the floored chain, <v041.out>/product/merged_v041.nc (b2_merge.py --product product_v041:
D-B049 gate with the v0.3.1 d200 masks, D-B050, D-B051), written by b2_v041_chain.py.
* Node files (v0.4.1.nc / .zarr, same tomoDD node grid):
  - medium (int8) from the D-B068 merged surface at the node: 0 rock, 1 water, 2 air (b2_model_v032.classify, the
    production field cells' test);
  - Vp: rock = the inversion's Vp; water 1.5 km/s; air NaN. Vs: rock = the inversion's Vs with the D-B073 node rule
    (offshore rock nodes within 2.5 km below the seafloor: Vs = min(Vs, Vp / 2.1)); water, air NaN. VpVs = Vp / Vs;
  - Vp_tomography, Vs_tomography: the inversion's values at every node (rock-continued above the surface: the
    inversion kept rock velocities there, D-B083; only event updates were floored);
  - merge variables (start, perturbations, DWS, resolution, tile weights, vs_reverted) as the merge wrote them;
  - surface_elevation, offshore, shallow_crust_node as v0.3.2.
* NonLinLoc grids (v0.4.1_nll/model.{P,S}.mod.*): b4_time_grids.build_field on the inversion's nodes (Vp, Vs =
  Vp_tomography, Vs_tomography), 1 km D-B005 Lambert lattice, merged mask, field-level D-B073 rule: the field rule,
  Earth-flattened, cell-centred SLOW_LEN, as the v5.1 production fields (time_b4_v0.3.1m_sc21) were built for v0.3.1.
  build_field's parameters are checked equal to that production build's (version and code md5 aside).
* v0.4.1_surface_lattice.nc: surface and offshore mask of the 1 km lattice (as v0.3.2's), for read_velocity_model.py.

  python b2_model_v041.py nodes      # write v0.4.1.nc / .zarr + surface lattice (not registered), read-back checks
  python b2_model_v041.py nll        # NLL grids by the field rule
  python b2_model_v041.py register   # registry row + provenance (b2_model_export.register: registry flock, union)
Run with the models env python.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_model_export as EX  # noqa: E402
import b2_model_v032 as V32  # noqa: E402
import b4_time_grids as TG  # noqa: E402
from manifest import load_params  # noqa: E402

P = load_params()
V = P["v041"]
VERSION = str(V["version"])
STORE = EX.STORE / EX.X["name"]
WORK = Path(V["out"]) / "product"
MERGED = WORK / "merged_v041.nc"
PROD = Path("/wd1/mdenolle_data/reloc_v5/work_time_b4_v0.3.1m_sc21")      # build_field parameters of v5.1 production
DESC = V["description"]
SC = TG.SC
# merged stages whose floor counts go into the attributes, the inversion binaries, the producing script (b2_model_v042.py
# sets these for v0.4.2; v0.4.1's values unchanged)
FLOOR_STAGES = {"M0": ("margin_m0_v4", "d200_s480"), "mendocino_gorda_c": ("mendocino_gorda_c_m0f", "d200_s480"),
                "blanco": ("blanco_m0f", "d200_s20"), "oregon": ("oregon_m0f", "d200_s480"),
                "washington": ("washington_m0f", "d200_s480")}
BINARIES = "/wd1/mdenolle_data/tomo_v1/src/frozen_2026-11-25_inv/bin"
FLOOR_DECISIONS = "D-B081 (tomoDD floor), D-B083 (floor in the inversion)"
SOURCE_NOTE = "b2_model_v041.py from the v0.4.1 merge (product_v041)"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lattice_cached():
    f = WORK / "lattice_1km.npz"
    if not f.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        np.savez(f, **V32.lattice(1.0))
    z = np.load(f)
    return {k: z[k] for k in z.files}


def floor_summary():
    """Floor settings and per-stage counts for the attributes (surface.dat headers, >>>Floor lines of each run)."""
    T = P["tomodd"]
    out = {}
    for t, (reg, tag) in FLOOR_STAGES.items():
        run = Path(T["runs"]) / reg / tag
        fj = Path(T["runs"]) / reg / "inputs" / "floor.json"
        lg = (run / "tomoDD-SE.log").read_text(errors="replace") if (run / "tomoDD-SE.log").exists() else ""
        hits = [ln.split()[1] for ln in lg.splitlines() if ln.startswith(">>>Floor: ")]
        out[t] = dict(run=f"{reg}/{tag}", spacing_s=json.loads(fj.read_text())["spacing_s"] if fj.exists() else None,
                      floor_hits=len(hits), events_hit=len(set(hits)),
                      not_met=sum(ln.startswith(">>>Floor not met") for ln in lg.splitlines()))
    return out


def build_nodes():
    base = xr.open_dataset(MERGED)
    L = lattice_cached()
    c = V32.classify(base, L)
    ds = base.copy(deep=True)
    dims3 = ("depth", "latitude", "longitude")
    f4 = lambda a: a.astype("float32")
    surf_src = TG.run_params(VERSION)[2]["surface_source"]
    sc_txt = (f"D-B073 at the nodes (2-km-node approximation of the field rule): rock nodes (medium 0) in offshore columns "
              f"(coast-scan mask of b4_time_grids.offshore_mask on the 1 km Lambert lattice, coast_run_km "
              f"{SC['coast_run_km']:g}, cuts {SC['cuts']}, at the lattice column containing the node), below sea level "
              f"and at most {SC['thickness_km']:g} km below the merged seafloor at the node (depth + surface_elevation <= "
              f"{SC['thickness_km']:g}): Vs = min(Vs_tomography, Vp_tomography / {SC['vpvs']:g}). Nodes above the "
              "seafloor are water (no rule). The NonLinLoc grids carry the exact field rule (1 km lattice cell centres)")
    med_txt = ("D-B068 merged solid surface (b4_surface.surface_m, bilinear on the first covering source: GMRT patches, "
               "earth_relief 01m elsewhere) sampled at the node; 0 rock (depth >= -surface_elevation), 1 water (above "
               "the surface, depth >= 0), 2 air (above the surface, depth < 0): the test of the production field cells")
    tomo_note = (f"v{VERSION} inversion value (tomoDD-SE with the D-B081 floor on event updates, D-B083), unchanged; the "
                 "inversion kept rock velocities above the solid surface (only hypocentres were floored), so nodes with "
                 "medium > 0 carry rock values: do not use them as velocities of water or air")
    ds["Vp"] = (dims3, f4(c["vp"]), dict(long_name="P-wave velocity", display_name="Vp (km/s)", units="km/s",
                                         source="rock: Vp_tomography; water: 1.5 km/s; air: NaN. " + med_txt))
    ds["Vs"] = (dims3, f4(c["vs"]), dict(long_name="S-wave velocity", display_name="Vs (km/s)", units="km/s",
                                         source="rock: Vs_tomography with the node rule; water and air: NaN. " + sc_txt))
    ds["VpVs"] = (dims3, f4(c["vp"] / c["vs"]), dict(long_name="Vp/Vs ratio", display_name="Vp/Vs", units="1",
                                                     source="Vp / Vs (NaN above the solid surface)"))
    ds["Vp_tomography"] = (dims3, base.Vp.values, dict(long_name=f"P-wave velocity of the v{VERSION} inversion",
                                                       display_name="Vp tomography (km/s)", units="km/s",
                                                       source=tomo_note + "; " + base.Vp.attrs.get("source", "")))
    ds["Vs_tomography"] = (dims3, base.Vs.values, dict(long_name=f"S-wave velocity of the v{VERSION} inversion",
                                                       display_name="Vs tomography (km/s)", units="km/s",
                                                       source=tomo_note + "; before the D-B073 rule; "
                                                       + base.Vs.attrs.get("source", "")))
    ds["medium"] = (dims3, c["medium"], dict(long_name="medium at the node against the merged solid surface",
                                             display_name="medium", units="1", flag_values=np.array([0, 1, 2], "int8"),
                                             flag_meanings="rock water air", source=med_txt))
    ds["shallow_crust_node"] = (dims3, c["changed"].astype("int8"), dict(
        long_name="Vs lowered by the D-B073 shallow oceanic crust rule at this node", display_name="D-B073 node",
        units="1", flag_values=np.array([0, 1], "int8"), flag_meanings="unchanged vs_lowered", source=sc_txt))
    ds["surface_elevation"] = (("latitude", "longitude"), f4(c["surf"]), dict(
        long_name="merged solid surface elevation at the node (seafloor or ground)", display_name="surface (km)",
        units="km", positive="up", source=f"b4_surface.surface_m (D-B068) at the node; finer: v{VERSION}_surface_lattice.nc"))
    ds["offshore"] = (("latitude", "longitude"), c["off"].astype("int8"), dict(
        long_name="offshore column of the D-B073 rule", display_name="offshore", units="1",
        flag_values=np.array([0, 1], "int8"), flag_meanings="onshore offshore",
        source="b4_time_grids.offshore_mask on the 1 km Lambert lattice, at the column containing the node"))
    for v in ("Vp_start", "Vs_start", "Vp_perturbation", "Vs_perturbation"):
        ds[v].attrs["source"] = ds[v].attrs.get("source", "") + f" ({tomo_note})"
    order = ["Vp", "Vs", "VpVs", "medium", "shallow_crust_node", "surface_elevation", "offshore", "Vp_tomography",
             "Vs_tomography"] + [v for v in base.data_vars if v not in ("Vp", "Vs", "VpVs")]
    ds = ds[order]
    for v in ds.variables:
        ds[v].encoding = {}
    srcs = TG.run_params(VERSION)[0]["surface_sources_md5"]
    prov = {k[len("provenance_"):]: v for k, v in base.attrs.items() if k.startswith("provenance_") and k != "provenance_git_commit"}
    fs = floor_summary()
    prov.update(merged_netcdf=str(MERGED), merged_netcdf_sha256=EX.sha256(MERGED),
                floor=json.dumps(dict(decisions=FLOOR_DECISIONS,
                                      binaries=BINARIES,
                                      settings=P["tomodd"]["floor"], stages=fs), default=str),
                nll_grids=("b4_time_grids.build_field(the inversion's nodes, 1 km, merged mask, field-level D-B073): "
                           "Earth-flattened cell-centred SLOW_LEN as the v5.1 production fields"))
    keep = {k: base.attrs[k] for k in base.attrs if not k.startswith("provenance_") and k not in (
        "title", "id", "version", "data_revision", "summary", "history", "year", "emc_check")}
    EX.add_attrs(ds, VERSION, DESC, prov, SOURCE_NOTE)
    ds.attrs.update(keep)
    ds.attrs.update(
        surface_rule="D-B068 merged solid surface: " + med_txt + "; water Vp 1.5 km/s (D-B005), Vs NaN; air NaN",
        surface_source=surf_src, surface_sources_md5=json.dumps(srcs),
        shallow_crust_rule=sc_txt, shallow_crust_vpvs=SC["vpvs"], shallow_crust_thickness_km=SC["thickness_km"],
        shallow_crust_coast_run_km=SC["coast_run_km"], shallow_crust_cuts=json.dumps(SC["cuts"]),
        shallow_crust_decisions="D-B068, D-B070, D-B073, D-B076; PI 2026-11-25",
        shallow_crust_node_note=("the node values carry a 2-km-node approximation of the rule; the NonLinLoc grids "
                                 f"(v{VERSION}_nll) carry the exact field rule; read_velocity_model.py evaluates the "
                                 f"field rule at any point from v{VERSION}_surface_lattice.nc"),
        floor_in_inversion=("D-B083: tomoDD-SE event updates stopped 1 m below the conservative floor of the D-B068 "
                            "merged surface (D-B081 floor, surface.dat per stage: M0 45 s, tiles 30 s cell minimum - 1 m); "
                            "events at the floor kept and flagged, Air_dep out of the way (-1000 km); velocities above the "
                            "surface rock-continued during the inversion"),
        medium_counts=json.dumps({k: c["info"][k] for k in ("rock", "water", "air")}),
        shallow_crust_node_count=c["info"]["rule_changed"],
        nll_frame=("Earth-flattened, cell-centred SLOW_LEN on the 1 km D-B005 Lambert lattice: depth axis z' = Re "
                   "ln(Re / (Re - z)), velocity v' = v Re / (Re - z), Re = 6371 km (as the v5.1 travel-time grids)"))
    return ds, c, L, base


def cmd_nodes(a):
    ds, c, L, base = build_nodes()
    print(json.dumps(c["info"], indent=1))
    issues = EX.emc_check(ds)
    ds.attrs["emc_check"] = "; ".join(issues) if issues else "passes the EMC inspector's required-attribute rules"
    hard = [i for i in issues if not i.startswith("pending")]
    if hard:
        raise SystemExit(f"EMC metadata problems: {hard}")
    EX.check_new(EX.read_registry(), VERSION)
    files = EX.write(ds, VERSION, nll=False)
    files["surface_lattice"] = V32.write_lattice_file(L, STORE / f"v{VERSION}_surface_lattice.nc")
    nc, zr = xr.open_dataset(files["netcdf"]), xr.open_zarr(files["zarr"], consolidated=True)
    chk = {}
    for v in ("Vp", "Vs", "VpVs", "medium", "shallow_crust_node", "Vp_tomography", "Vs_tomography"):
        chk[f"{v}_nc_eq_zarr"] = bool(np.array_equal(nc[v].values, zr[v].values, equal_nan=True))
    rock = nc.medium.values == 0
    chk["vp_rock_eq_inversion"] = bool(np.array_equal(nc.Vp.values[rock], base.Vp.values[rock]))
    chk["vs_rock_unchanged_off_rule"] = bool(np.array_equal(nc.Vs.values[rock & (nc.shallow_crust_node.values == 0)],
                                                            base.Vs.values[rock & (nc.shallow_crust_node.values == 0)]))
    chk["vp_water_1p5"] = bool(np.all(nc.Vp.values[nc.medium.values == 1] == 1.5))
    chk["vs_nan_above"] = bool(np.isnan(nc.Vs.values[~rock]).all())
    chk["vp_air_nan"] = bool(np.isnan(nc.Vp.values[nc.medium.values == 2]).all())
    chk["vpvs_min_rule_nodes"] = (float(np.nanmin(nc.VpVs.values[nc.shallow_crust_node.values == 1]))
                                  if c["info"]["rule_changed"] else None)
    chk["vpvs_rock_min_max"] = [float(np.nanmin(nc.VpVs.values[rock])), float(np.nanmax(nc.VpVs.values[rock]))]
    chk["tomography_eq_merge"] = bool(np.array_equal(nc.Vp_tomography.values, base.Vp.values) and
                                      np.array_equal(nc.Vs_tomography.values, base.Vs.values))
    chk["other_vars_eq_merge"] = {v: bool(np.array_equal(nc[v].values, base[v].values, equal_nan=True))
                                  for v in base.data_vars if v not in ("Vp", "Vs", "VpVs")}
    bad = [k for k, v in chk.items() if v is False or (isinstance(v, dict) and not all(v.values()))]
    out = dict(files={k: dict(path=str(p), sha256=EX.sha256(p)) for k, p in files.items()}, info=c["info"], checks=chk,
               failed_checks=bad)
    (WORK / "nodes_written.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    if bad:
        raise SystemExit(f"read-back checks failed: {bad} (files written, NOT registered)")


def cmd_nll(a):
    work = WORK / "field"
    work.mkdir(parents=True, exist_ok=True)
    assert TG.MASK == "merged" and TG.SC, "params b4.surface_mask must be merged and b4.shallow_crust set"
    prm = TG.run_params(VERSION)[0]
    prod_prm = json.loads((PROD / "params.json").read_text())
    same = {k: prm[k] == prod_prm.get(k) for k in prm if k not in ("code_md5", "version")}
    print("build_field parameters equal to the v5.1 production build (version, code md5 aside):", same)
    assert all(same.values()), same
    base = xr.open_dataset(MERGED)                      # the inversion's node values (rock everywhere)
    fields, shape = TG.build_field(base, 1.0, work)
    out = STORE / f"v{VERSION}_nll"
    out.mkdir(exist_ok=False)
    nx, ny, nz = shape
    res = {}
    for ph in ("P", "S"):
        shutil.copy(fields[ph], out / f"model.{ph}.mod.buf")
        res[ph] = dict(md5=md5(out / f"model.{ph}.mod.buf"), work_md5=md5(fields[ph]))
        assert res[ph]["md5"] == res[ph]["work_md5"]
        (out / f"model.{ph}.mod.hdr").write_text(
            f"{nx} {ny} {nz}  {TG.G['x0']:f} {TG.G['y0']:f} {TG.G['z0']:f}  1.000000 1.000000 1.000000 SLOW_LEN FLOAT\n"
            f"{TG.TRANS}\n")
        (out / f"model.{ph}.mod.README").write_text(
            f"cascadia_obs_tomo v{VERSION}, {ph}: Earth-flattened, cell-centred SLOW_LEN (depth axis z' = Re ln(Re / "
            f"(Re - z)), velocity v' = v Re / (Re - z), Re = 6371 km), merged-surface water/air cells, D-B073 field rule; "
            f"b4_time_grids.build_field from the v{VERSION} inversion nodes (Vp_tomography, Vs_tomography), 1 km lattice "
            f"(md5 {res[ph]['md5']}).\n")
        shutil.copy(work / f"field_1.{ph}.mask.json", out / f"field_1.{ph}.mask.json")
    (WORK / "nll_written.json").write_text(json.dumps(dict(out=str(out), md5=res, params=prm), indent=1, default=str))
    print("NLL grids:", out, json.dumps(res))


def cmd_register(a):
    ds = xr.open_dataset(STORE / f"v{VERSION}.nc")
    nw = json.loads((WORK / "nodes_written.json").read_text())
    if nw["failed_checks"]:
        raise SystemExit(f"node checks failed {nw['failed_checks']}: not registered")
    files = {"zarr": STORE / f"v{VERSION}.zarr", "netcdf": STORE / f"v{VERSION}.nc",
             "surface_lattice": STORE / f"v{VERSION}_surface_lattice.nc",
             "nll_P": STORE / f"v{VERSION}_nll" / "model.P.mod.buf", "nll_S": STORE / f"v{VERSION}_nll" / "model.S.mod.buf",
             "nll_P_hdr": STORE / f"v{VERSION}_nll" / "model.P.mod.hdr",
             "nll_S_hdr": STORE / f"v{VERSION}_nll" / "model.S.mod.hdr"}
    assert all(p.exists() for p in files.values()), files
    back = xr.open_zarr(files["zarr"], consolidated=True)           # read-back before the registration
    if not (np.array_equal(back.Vp.values, ds.Vp.values, equal_nan=True) and back.attrs.get("id") == ds.attrs.get("id")):
        raise SystemExit(f"read-back of {files['zarr']} differs from {files['netcdf']}: NOT registered")
    pj = EX.register(ds, VERSION, DESC, files, a.status or V["status"])
    print("registered", pj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nodes", "nll", "register"])
    ap.add_argument("--status", default=None)
    a = ap.parse_args()
    {"nodes": cmd_nodes, "nll": cmd_nll, "register": cmd_register}[a.cmd](a)


if __name__ == "__main__":
    main()
