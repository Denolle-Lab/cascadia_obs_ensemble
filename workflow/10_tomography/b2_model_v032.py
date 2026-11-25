"""Track B: model v0.3.2 = v0.3.1 with the solid surface (D-B068) and the D-B073 shallow oceanic crust rule.

PI 2026-11-25: "Register v0.3.1 + D-B073 shallow-crust rule as v0.3.2"; 05:50: "We needed a surface fix as there are
rock velocity above the sea floor". No new inversion: the tomography node values are v0.3.1's.

Two representations, one model:
* Node files (v0.3.2.nc / .zarr, the tomoDD node grid of v0.3.1, 0.125 x 0.1 deg, depth nodes -3, -1, 1, 3 ... km):
  - medium (int8) per node from the merged solid surface sampled at the node (b4_surface.surface_m: bilinear on its
    first covering source, GMRT patches then earth_relief 01m; D-B068): 0 rock (at or below the surface), 1 water (above
    the surface and at or below sea level, depth >= 0), 2 air (above the surface and above sea level). The same test as
    the production field cells (b4_time_grids.build_field, merged mask: fluid iff -z > surface; water iff z >= 0).
  - Vp: rock = v0.3.1 Vp; water = 1.5 km/s; air = NaN. Vs: rock = v0.3.1 Vs, then the node rule below; water, air = NaN.
    VpVs = Vp / Vs (NaN above the surface).
  - D-B073 node rule (shallow_crust_node = 1 where it lowered Vs): rock nodes (medium 0) in offshore columns, below sea
    level (depth > 0, as the field rule's zt > 0) and at most thickness_km below the seafloor at the node
    (depth + surface_km <= thickness_km): Vs = min(Vs, Vp / vpvs). Offshore = the coast-scan mask of b4_time_grids
    (offshore_mask on the 1 km Lambert lattice of the production field, cell centres, coast_run_km, cuts), taken at
    the 1 km column that contains the node. This is a 2-km-node approximation of the field rule.
  - Vp_tomography, Vs_tomography: v0.3.1's Vp and Vs unchanged at every node (the inversion had no surface: nodes
    above the solid surface carry rock values there). Vp_start, Vs_start, the perturbations, DWS, resolution and merge
    variables are v0.3.1's, unchanged (they describe the inversion).
  - surface_elevation (latitude, longitude): the merged solid surface at the nodes (km, positive up); offshore (int8).
* NonLinLoc grids (v0.3.2_nll/model.{P,S}.mod.buf|hdr): the exact v5.1 production fields, built here by
  b4_time_grids.build_field from the v0.3.1 nodes on the 1 km D-B005 Lambert lattice with the merged mask and the
  field-level D-B073 rule (params b4.surface_mask, b4.shallow_crust), never by interpolating the rule-applied nodes;
  their md5 is checked against the production work fields of time_b4_v0.3.1m_sc21. They are Earth-flattened, cell-
  centred SLOW_LEN (depth axis z' = Re ln(Re / (Re - z)), velocity v' = v Re / (Re - z)), exactly what Grid2Time ran
  on for the v5.1 relocation; header line 3 says so.
* v0.3.2_surface_lattice.nc: what a reader needs to evaluate the field rule at any point: the merged surface at the
  1 km lattice cell centres and the offshore mask of the lattice (int8), Lambert x / y cell-centre coordinates (km)
  and the projection. read_velocity_model.py uses it.

  python b2_model_v032.py nodes      # build and write v0.3.2.nc / .zarr (not registered), print the checks
  python b2_model_v032.py nll        # NLL grids from build_field (v0.3.1 nodes, 1 km), md5 against production
  python b2_model_v032.py register   # registry row + provenance (b2_model_export.register), all files
  python b2_model_v032.py check      # reader and node classification against the production field
  python b2_model_v032.py compare    # node approximation vs the field rule: profiles and travel times
Run with the models env python (zarr 3, netCDF4, pyproj).
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
import b4_time_grids as TG  # noqa: E402
from b2_fwd_common import RE, eft_depth  # noqa: E402

VERSION, BASE = "0.3.2", "0.3.1"
STORE = EX.STORE / EX.X["name"]
PROD = Path("/wd1/mdenolle_data/reloc_v5/work_time_b4_v0.3.1m_sc21")      # v5.1 production fields (1 km, 0.5 km)
WORK = Path("/wd1/mdenolle_data/tomo_v1/ensemble_v0.3.2/v032_build")
SC = TG.SC
DESC = ("v0.3.1 node velocities with the solid surface (medium: water Vp 1.5 km/s, air NaN above the D-B068 merged "
        "surface; v0.3.1 values kept as Vp_tomography, Vs_tomography) and the D-B073 shallow oceanic crust rule at the "
        "rock nodes (Vs <= Vp/2.1 within 2.5 km below the seafloor offshore); NonLinLoc grids = the exact v5.1 "
        "production fields (time_b4_v0.3.1m_sc21: v0.3.1 + field-level D-B073 on the 1 km lattice)")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lattice(dx=1.0):
    """The production field's columns (b4_time_grids.build_field, merged mask): cell-centre x, y (km), lon, lat, the
    merged surface (km, positive up) and the offshore mask (b4_time_grids.offshore_mask), computed as build_field does."""
    from b4_surface import surface_m
    nx, ny, nz = TG.field_shape(dx)
    xs, ys = (TG.G[k0] + dx * np.arange(n) + dx / 2 for k0, n in (("x0", nx), ("y0", ny)))
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    lon, lat = TG.proj()(X2 * 1e3, Y2 * 1e3, inverse=True)
    surf = surface_m(lon, lat) / 1e3
    assert np.isfinite(surf).all()
    off = TG.offshore_mask(surf, lon, lat, dx)
    return dict(xs=xs, ys=ys, lon=lon, lat=lat, surf=surf, off=off, dx=dx)


def lattice_cached():
    f = WORK / "lattice_1km.npz"
    if not f.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        L = lattice(1.0)
        np.savez(f, **{k: v for k, v in L.items()})
    z = np.load(f)
    return {k: z[k] for k in z.files}


def column_index(L, lon, lat):
    """Index (i, j) of the 1 km lattice column containing each point; -1 outside the lattice."""
    x, y = TG.proj()(np.asarray(lon), np.asarray(lat))
    i = np.floor((x / 1e3 - TG.G["x0"]) / L["dx"]).astype(int)
    j = np.floor((y / 1e3 - TG.G["y0"]) / L["dx"]).astype(int)
    ok = (i >= 0) & (i < len(L["xs"])) & (j >= 0) & (j < len(L["ys"]))
    return np.where(ok, i, -1), np.where(ok, j, -1)


def classify(base, L):
    """Node medium, surface, offshore and the D-B073 node rule on the v0.3.1 Dataset."""
    from b4_surface import surface_m
    lon, lat, z = base.longitude.values, base.latitude.values, base.depth.values
    LA, LO = np.meshgrid(lat, lon, indexing="ij")               # [lat, lon]
    surf = (surface_m(LO, LA) / 1e3).astype("f8")
    assert np.isfinite(surf).all(), "merged surface NaN at a node"
    ii, jj = column_index(L, LO, LA)
    off = np.where(ii >= 0, L["off"][np.maximum(ii, 0), np.maximum(jj, 0)], False)
    Z = z[:, None, None]
    above = -Z > surf[None]                                     # the field's test: fluid iff -z > surface
    medium = np.where(above, np.where(Z >= 0, 1, 2), 0).astype("int8")
    vp0, vs0 = base.Vp.values.astype("f8"), base.Vs.values.astype("f8")
    assert (vp0 >= 1.6).all(), "a v0.3.1 node below the rock threshold"
    rule = (medium == 0) & off[None] & (Z > 0) & (Z + surf[None] <= SC["thickness_km"])
    vs_rule = np.where(rule, np.minimum(vs0, vp0 / SC["vpvs"]), vs0)
    changed = rule & (vs_rule < vs0)
    vp = np.where(medium == 0, vp0, np.where(medium == 1, 1.5, np.nan))
    vs = np.where(medium == 0, vs_rule, np.nan)
    info = dict(nodes=int(medium.size), rock=int((medium == 0).sum()), water=int((medium == 1).sum()),
                air=int((medium == 2).sum()), offshore_columns=int(off.sum()),
                columns_outside_lattice=int((ii < 0).sum()),
                outside_lattice_below_sea_level=int(((ii < 0) & (surf < 0)).sum()),
                rule_nodes=int(rule.sum()), rule_changed=int(changed.sum()),
                water_dws_p_gt0=int(((medium == 1) & (base.dws_p.values > 0)).sum()),
                air_dws_p_gt0=int(((medium == 2) & (base.dws_p.values > 0)).sum()),
                offshore_above_surface=int((above & off[None]).sum()),
                onshore_above_surface=int((above & ~off[None]).sum()),
                above_surface_vp_median=float(np.median(vp0[above])),
                above_surface_vs_median=float(np.median(vs0[above])),
                by_depth={f"{zk:g}": dict(water=int((medium[k] == 1).sum()), air=int((medium[k] == 2).sum()),
                                           rule_changed=int(changed[k].sum()))
                          for k, zk in enumerate(z) if zk <= 9})
    if changed.any():
        d = (vs0 - vs_rule)[changed] / vs0[changed]
        info["rule_dvs_pct_median_max"] = [round(100 * float(np.median(d)), 2), round(100 * float(d.max()), 2)]
    return dict(surf=surf, off=off, medium=medium, vp=vp, vs=vs, changed=changed, info=info)


def build_nodes():
    base = xr.open_dataset(STORE / f"v{BASE}.nc")
    L = lattice_cached()
    c = classify(base, L)
    ds = base.copy(deep=True)
    dims3 = ("depth", "latitude", "longitude")
    f4 = lambda a: a.astype("float32")
    surf_src = TG.run_params(VERSION)[2]["surface_source"]
    sc_txt = (f"D-B073 at the nodes (2-km-node approximation of the field rule): rock nodes (medium 0) in offshore columns "
              f"(coast-scan mask of b4_time_grids.offshore_mask on the 1 km Lambert lattice of the production field, "
              f"coast_run_km {SC['coast_run_km']:g}, cuts {SC['cuts']}, taken at the lattice column containing the node), "
              f"below sea level and at most {SC['thickness_km']:g} km below the merged seafloor sampled at the node "
              f"(depth + surface_elevation <= {SC['thickness_km']:g}): Vs = min(Vs_tomography, Vp_tomography / "
              f"{SC['vpvs']:g}). Nodes above the seafloor are water (no rule). The NonLinLoc grids carry the exact "
              "field rule (cell centres of the 1 km lattice)")
    med_txt = ("D-B068 merged solid surface (b4_surface.surface_m, bilinear on the first covering source: GMRT patches, "
               "earth_relief 01m elsewhere) sampled at the node; 0 rock (depth >= -surface_elevation), 1 water (above "
               "the surface, depth >= 0), 2 air (above the surface, depth < 0): the test of the production field cells")
    tomo_note = ("v0.3.1 inversion value, unchanged; the inversion had no surface, so nodes above the solid surface "
                 "(medium > 0) carry rock values there: do not use them as velocities of water or air")
    ds["Vp"] = (dims3, f4(c["vp"]), dict(long_name="P-wave velocity", display_name="Vp (km/s)", units="km/s",
                                         source=f"rock: Vp_tomography (v{BASE}); water: 1.5 km/s; air: NaN. " + med_txt))
    ds["Vs"] = (dims3, f4(c["vs"]), dict(long_name="S-wave velocity", display_name="Vs (km/s)", units="km/s",
                                         source=f"rock: Vs_tomography (v{BASE}) with the node rule; water and air: NaN. "
                                                + sc_txt))
    ds["VpVs"] = (dims3, f4(c["vp"] / c["vs"]), dict(long_name="Vp/Vs ratio", display_name="Vp/Vs", units="1",
                                                     source="Vp / Vs (NaN above the solid surface)"))
    ds["Vp_tomography"] = (dims3, base.Vp.values, dict(long_name="P-wave velocity of the v0.3.1 inversion",
                                                       display_name="Vp tomography (km/s)", units="km/s", source=tomo_note))
    ds["Vs_tomography"] = (dims3, base.Vs.values, dict(long_name="S-wave velocity of the v0.3.1 inversion",
                                                       display_name="Vs tomography (km/s)", units="km/s",
                                                       source=tomo_note + "; before the D-B073 rule"))
    ds["medium"] = (dims3, c["medium"], dict(long_name="medium at the node against the merged solid surface",
                                             display_name="medium", units="1", flag_values=np.array([0, 1, 2], "int8"),
                                             flag_meanings="rock water air", source=med_txt))
    ds["shallow_crust_node"] = (dims3, c["changed"].astype("int8"), dict(
        long_name="Vs lowered by the D-B073 shallow oceanic crust rule at this node", display_name="D-B073 node",
        units="1", flag_values=np.array([0, 1], "int8"), flag_meanings="unchanged vs_lowered", source=sc_txt))
    ds["surface_elevation"] = (("latitude", "longitude"), f4(c["surf"]), dict(
        long_name="merged solid surface elevation at the node (seafloor or ground)", display_name="surface (km)",
        units="km", positive="up", source="b4_surface.surface_m (D-B068) at the node; finer: v0.3.2_surface_lattice.nc"))
    ds["offshore"] = (("latitude", "longitude"), c["off"].astype("int8"), dict(
        long_name="offshore column of the D-B073 rule", display_name="offshore", units="1",
        flag_values=np.array([0, 1], "int8"), flag_meanings="onshore offshore",
        source="b4_time_grids.offshore_mask on the 1 km Lambert lattice, at the column containing the node"))
    for v in ("Vp_start", "Vs_start", "Vp_perturbation", "Vs_perturbation"):
        ds[v].attrs["source"] = ds[v].attrs["source"] + f" ({tomo_note})"
    order = ["Vp", "Vs", "VpVs", "medium", "shallow_crust_node", "surface_elevation", "offshore", "Vp_tomography",
             "Vs_tomography"] + [v for v in base.data_vars if v not in ("Vp", "Vs", "VpVs")]
    ds = ds[order]
    for v in ds.variables:                                      # netCDF encodings of v0.3.1 must not reach the Zarr store
        ds[v].encoding = {}
    srcs = TG.run_params(VERSION)[0]["surface_sources_md5"]
    prov = {k[len("provenance_"):]: v for k, v in base.attrs.items() if k.startswith("provenance_") and k != "provenance_git_commit"}
    prov.update(base_version=f"v{BASE}", base_netcdf_sha256=EX.sha256(STORE / f"v{BASE}.nc"),
                nll_grids=f"exact v5.1 production fields ({PROD.name}): b4_time_grids.build_field(v{BASE} nodes, 1 km, "
                          "merged mask, field-level D-B073), md5 checked against the production work fields")
    keep = {k: base.attrs[k] for k in base.attrs if not k.startswith("provenance_") and k not in (
        "title", "id", "version", "data_revision", "summary", "history", "year", "emc_check")}
    EX.add_attrs(ds, VERSION, DESC, prov, f"b2_model_v032.py from v{BASE}")
    ds.attrs.update(keep)
    ds.attrs.update(
        surface_rule="D-B068 merged solid surface: " + med_txt + "; water Vp 1.5 km/s (D-B005), Vs NaN; air NaN",
        surface_source=surf_src, surface_sources_md5=json.dumps(srcs),
        shallow_crust_rule=sc_txt, shallow_crust_vpvs=SC["vpvs"], shallow_crust_thickness_km=SC["thickness_km"],
        shallow_crust_coast_run_km=SC["coast_run_km"], shallow_crust_cuts=json.dumps(SC["cuts"]),
        shallow_crust_decisions="D-B068, D-B070, D-B073; PI 2026-11-25",
        shallow_crust_node_note=("the node values carry a 2-km-node approximation of the rule; the NonLinLoc grids "
                                 "(v0.3.2_nll) carry the exact field rule that the v5.1 relocation used; "
                                 "read_velocity_model.py evaluates the field rule at any point from "
                                 "v0.3.2_surface_lattice.nc"),
        medium_counts=json.dumps({k: c["info"][k] for k in ("rock", "water", "air")}),
        shallow_crust_node_count=c["info"]["rule_changed"],
        nll_frame=("Earth-flattened, cell-centred SLOW_LEN on the 1 km D-B005 Lambert lattice: depth axis z' = Re "
                   "ln(Re / (Re - z)), velocity v' = v Re / (Re - z), Re = 6371 km (as the v5.1 travel-time grids)"))
    return ds, c, L


def write_lattice_file(L, path):
    a = xr.Dataset(
        dict(surface_km=(("x", "y"), L["surf"].astype("float32"), dict(
                 long_name="merged solid surface at the 1 km lattice cell centres", units="km", positive="up",
                 source="b4_surface.surface_m (D-B068) at the cell centres, as b4_time_grids.build_field")),
             offshore=(("x", "y"), L["off"].astype("int8"), dict(
                 long_name="offshore column of the D-B073 field rule", units="1", flag_values=np.array([0, 1], "int8"),
                 flag_meanings="onshore offshore", source="b4_time_grids.offshore_mask on this lattice"))),
        coords=dict(x=("x", L["xs"], dict(units="km", long_name="Lambert x of the cell centre")),
                    y=("y", L["ys"], dict(units="km", long_name="Lambert y of the cell centre"))))
    a.attrs = dict(title=f"cascadia_obs_tomo v{VERSION}: surface and offshore mask of the D-B073 field rule",
                   projection="LAMBERT conformal conic, WGS-84, lat_0 45.0, lon_0 -125.8, lat_1 41.0, lat_2 49.0 "
                              "(pyproj: proj=lcc lat_0=45 lon_0=-125.8 lat_1=41 lat_2=49 ellps=WGS84)",
                   lattice=f"cell i covers x in [x0 + i, x0 + i + 1] km, x0 {TG.G['x0']}, y0 {TG.G['y0']}, z0 "
                           f"{TG.G['z0']} (flattened), cell 1 km",
                   rule=("rock cell (centre at or below surface_km) with offshore = 1, true depth z > 0 and z + "
                         f"surface_km <= {SC['thickness_km']:g}: Vs = min(Vs, Vp / {SC['vpvs']:g}); fluid cells: water "
                         "(z >= 0) Vp 1.5, Vs 0.4 km/s in the NLL grids; air 0.3 / 0.3 km/s"),
                   shallow_crust=json.dumps(SC))
    enc = {k: dict(zlib=True, complevel=6, shuffle=True) for k in a.data_vars}
    a.to_netcdf(path, format="NETCDF4_CLASSIC", encoding=enc)
    return path


def cmd_nodes(a):
    ds, c, L = build_nodes()
    print(json.dumps(c["info"], indent=1))
    issues = EX.emc_check(ds)
    ds.attrs["emc_check"] = "; ".join(issues) if issues else "passes the EMC inspector's required-attribute rules"
    hard = [i for i in issues if not i.startswith("pending")]
    if hard:
        raise SystemExit(f"EMC metadata problems: {hard}")
    EX.check_new(EX.read_registry(), VERSION)
    files = EX.write(ds, VERSION, nll=False)
    files["surface_lattice"] = write_lattice_file(L, STORE / f"v{VERSION}_surface_lattice.nc")
    # read-back checks
    nc, zr = xr.open_dataset(files["netcdf"]), xr.open_zarr(files["zarr"], consolidated=True)
    base = xr.open_dataset(STORE / f"v{BASE}.nc")
    chk = {}
    for v in ("Vp", "Vs", "VpVs", "medium", "shallow_crust_node"):
        chk[f"{v}_nc_eq_zarr"] = bool(np.array_equal(nc[v].values, zr[v].values, equal_nan=True))
    rock = nc.medium.values == 0
    chk["vp_rock_eq_v031"] = bool(np.array_equal(nc.Vp.values[rock], base.Vp.values[rock]))
    chk["vs_rock_unchanged_off_rule"] = bool(np.array_equal(nc.Vs.values[rock & (nc.shallow_crust_node.values == 0)],
                                                            base.Vs.values[rock & (nc.shallow_crust_node.values == 0)]))
    chk["vp_water_1p5"] = bool(np.all(nc.Vp.values[nc.medium.values == 1] == 1.5))
    chk["vs_nan_above"] = bool(np.isnan(nc.Vs.values[~rock]).all())
    chk["vp_air_nan"] = bool(np.isnan(nc.Vp.values[nc.medium.values == 2]).all())
    chk["vpvs_min_rule_nodes"] = float(np.nanmin(nc.VpVs.values[nc.shallow_crust_node.values == 1])) if c["info"]["rule_changed"] else None
    chk["vpvs_rock_min_max"] = [float(np.nanmin(nc.VpVs.values[rock])), float(np.nanmax(nc.VpVs.values[rock]))]
    chk["tomography_eq_v031"] = bool(np.array_equal(nc.Vp_tomography.values, base.Vp.values) and
                                     np.array_equal(nc.Vs_tomography.values, base.Vs.values))
    chk["other_vars_eq_v031"] = {v: bool(np.array_equal(nc[v].values, base[v].values, equal_nan=True))
                                 for v in base.data_vars if v not in ("Vp", "Vs", "VpVs")}
    out = dict(files={k: dict(path=str(p), sha256=EX.sha256(p), size=sum(f.stat().st_size for f in Path(p).rglob('*'))
                              if Path(p).is_dir() else Path(p).stat().st_size) for k, p in files.items()},
               info=c["info"], checks=chk)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "nodes_written.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def cmd_nll(a):
    """NLL grids = build_field(v0.3.1 nodes, 1 km, merged mask, field rule): the production fields, md5-checked."""
    work = WORK / "field"
    work.mkdir(parents=True, exist_ok=True)
    assert TG.MASK == "merged" and TG.SC, "params b4.surface_mask must be merged and b4.shallow_crust set"
    prm = TG.run_params(BASE)[0]
    prod_prm = json.loads((PROD / "params.json").read_text())
    same = {k: prm[k] == prod_prm.get(k) for k in prm if k != "code_md5"}
    print("params equal to production (code md5 aside):", same)
    assert all(same.values()), same
    base = xr.open_dataset(STORE / f"v{BASE}.nc")
    fields, shape = TG.build_field(base, 1.0, work)
    res = {}
    for ph in ("P", "S"):
        m_new, m_prod = md5(fields[ph]), md5(PROD / f"field_1.{ph}.buf")
        res[ph] = dict(md5=m_new, production_md5=m_prod, identical=m_new == m_prod)
    print(json.dumps(res, indent=1))
    if not all(r["identical"] for r in res.values()):
        raise SystemExit("NLL fields differ from the production fields: not exported")
    out = STORE / f"v{VERSION}_nll"
    out.mkdir(exist_ok=False)
    nx, ny, nz = shape
    for ph in ("P", "S"):
        shutil.copy(fields[ph], out / f"model.{ph}.mod.buf")
        (out / f"model.{ph}.mod.hdr").write_text(
            f"{nx} {ny} {nz}  {TG.G['x0']:f} {TG.G['y0']:f} {TG.G['z0']:f}  1.000000 1.000000 1.000000 SLOW_LEN FLOAT\n"
            f"{TG.TRANS}\n")
        (out / f"model.{ph}.mod.README").write_text(
            f"cascadia_obs_tomo v{VERSION}, {ph}: Earth-flattened, cell-centred SLOW_LEN (depth axis z' = Re ln(Re / "
            f"(Re - z)), velocity v' = v Re / (Re - z), Re = 6371 km), merged-surface water/air cells, D-B073 field rule; "
            f"identical (md5 {res[ph]['md5']}) to {PROD}/field_1.{ph}.buf, the 1 km field of the v5.1 relocation.\n")
    shutil.copy(work / "field_1.S.mask.json", out / "field_1.S.mask.json")
    shutil.copy(work / "field_1.P.mask.json", out / "field_1.P.mask.json")
    (WORK / "nll_written.json").write_text(json.dumps(dict(out=str(out), md5=res), indent=1))
    print("NLL grids:", out)


def cmd_register(a):
    ds = xr.open_dataset(STORE / f"v{VERSION}.nc")
    files = {"zarr": STORE / f"v{VERSION}.zarr", "netcdf": STORE / f"v{VERSION}.nc",
             "surface_lattice": STORE / f"v{VERSION}_surface_lattice.nc",
             "nll_P": STORE / f"v{VERSION}_nll" / "model.P.mod.buf", "nll_S": STORE / f"v{VERSION}_nll" / "model.S.mod.buf",
             "nll_P_hdr": STORE / f"v{VERSION}_nll" / "model.P.mod.hdr",
             "nll_S_hdr": STORE / f"v{VERSION}_nll" / "model.S.mod.hdr"}
    assert all(p.exists() for p in files.values()), files
    pj = EX.register(ds, VERSION, DESC, files, a.status)
    print("registered", pj)


# ---------------------------------------------------------------- checks against the production field
def prod_field(ph, which="P"):
    nx, ny, nz = TG.field_shape(1.0)
    return np.memmap(PROD / f"field_1.{ph}.buf", dtype="<f4", mode="r", shape=(nx, ny, nz))


def cell_velocity(a, i, j, k):
    zf = TG.G["z0"] + k + 0.5
    return 1.0 / (np.asarray(a[i, j, k], "f8") * np.exp(zf / RE))


def cmd_check(a):
    import velocity_model_tools as RV
    L = lattice_cached()
    nc = xr.open_dataset(STORE / f"v{VERSION}.nc")
    ap_, as_ = prod_field("P"), prod_field("S")
    nx, ny, nz = ap_.shape
    g = np.random.default_rng(20261125)
    out = {}
    # (1) reader (exact mode) at random production cell centres inside the model volume
    n = 20000
    i, j, k = g.integers(0, nx, n), g.integers(0, ny, n), g.integers(0, nz, n)
    lon, lat = L["lon"][i, j], L["lat"][i, j]
    zf = TG.G["z0"] + k + 0.5
    zt = RE * (1 - np.exp(-zf / RE))
    inside = ((lon >= nc.longitude.values[0]) & (lon <= nc.longitude.values[-1]) & (lat >= nc.latitude.values[0]) &
              (lat <= nc.latitude.values[-1]) & (zt >= nc.depth.values[0]) & (zt <= nc.depth.values[-1]))
    i, j, k, lon, lat, zt = (x[inside] for x in (i, j, k, lon, lat, zt))
    vp_prod, vs_prod = cell_velocity(ap_, i, j, k), cell_velocity(as_, i, j, k)
    R = RV.velocity_at(STORE, VERSION, lat, lon, zt, mode="field")
    rock = R["medium"] == 0
    for ph, vprod, vr in (("P", vp_prod, R["Vp"]), ("S", vs_prod, R["Vs"])):
        d = np.abs(vr - vprod) / vprod
        out[f"reader_field_{ph}"] = dict(n=int(len(d)), rock=int(rock.sum()),
                                         rel_diff_rock_max=float(np.nanmax(d[rock])),
                                         rel_diff_rock_p99=float(np.nanpercentile(d[rock], 99)),
                                         n_rock_gt_1em4=int((d[rock] > 1e-4).sum()))
    fluid_prod = vp_prod < 1.6
    out["reader_medium_vs_production_fluid"] = dict(
        agree=int(((R["medium"] > 0) == fluid_prod).sum()), n=int(len(fluid_prod)),
        reader_rock_prod_fluid=int((rock & fluid_prod).sum()), reader_fluid_prod_rock=int((~rock & ~fluid_prod).sum()))
    # P unchanged from v0.3.1: production P at rock cells vs trilinear v0.3.1 Vp (= Vp_tomography)
    tri = RV.trilinear(nc, "Vp_tomography", lat, lon, zt)
    dp = np.abs(tri - vp_prod)[rock] / vp_prod[rock]
    out["P_production_vs_v031_trilinear"] = dict(rel_diff_max=float(dp.max()), n=int(rock.sum()))
    # the field rule changed S at how many of these sampled rock cells
    tri_s = RV.trilinear(nc, "Vs_tomography", lat, lon, zt)
    out["S_rule_cells_in_sample"] = int((rock & (vs_prod < tri_s * (1 - 1e-5))).sum())
    # (2) node medium vs the NLL grid's water/air cell containing the node
    LA, LO = np.meshgrid(nc.latitude.values, nc.longitude.values, indexing="ij")
    ii, jj = column_index(L, LO, LA)
    med = nc.medium.values
    rows = []
    for kk, z in enumerate(nc.depth.values):
        kz = int(np.floor(float(eft_depth(z)) - TG.G["z0"]))
        if kz < 0 or kz >= nz:
            continue
        okc = ii >= 0
        v = np.full(LO.shape, np.nan)
        v[okc] = cell_velocity(ap_, ii[okc], jj[okc], np.full(okc.sum(), kz))
        cls = np.where(np.isnan(v), -1, np.where(v >= 1.6, 0, np.where(z >= 0, 1, 2)))
        m = med[kk]
        rows.append(dict(depth=float(z), n=int(okc.sum()), agree=int(((cls == m) & okc).sum()),
                         node_rock_cell_fluid=int(((m == 0) & (cls > 0)).sum()),
                         node_fluid_cell_rock=int(((m > 0) & (cls == 0)).sum()),
                         node_water_cell_air=int(((m == 1) & (cls == 2)).sum()),
                         node_air_cell_water=int(((m == 2) & (cls == 1)).sum())))
        if rows[-1]["node_rock_cell_fluid"] + rows[-1]["node_fluid_cell_rock"]:
            bad = okc & (((m == 0) & (cls > 0)) | ((m > 0) & (cls == 0)))
            sf = nc.surface_elevation.values
            rows[-1]["mismatch_dist_to_surface_km_median_max"] = [float(np.median(np.abs(z + sf[bad]))),
                                                                  float(np.abs(z + sf[bad]).max())]
            surf_cell = L["surf"][ii[bad], jj[bad]]
            rows[-1]["mismatch_surface_node_minus_cell_km_median"] = float(np.median(np.abs(sf[bad] - surf_cell)))
    out["medium_vs_nll_cells"] = rows
    out["medium_vs_nll_total"] = {k: sum(r[k] for r in rows) for k in ("n", "agree", "node_rock_cell_fluid",
                                                                         "node_fluid_cell_rock")}
    (WORK / "check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nodes", "nll", "register", "check"])
    ap.add_argument("--status", default="candidate")
    a = ap.parse_args()
    {"nodes": cmd_nodes, "nll": cmd_nll, "register": cmd_register, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
