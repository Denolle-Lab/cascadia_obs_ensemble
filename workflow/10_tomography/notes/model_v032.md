# Model v0.3.2: v0.3.1 with the solid surface and the D-B073 shallow-crust rule

Agent note, 2026-11-25 (worktree ~/wt_ens, branch tomo/ensemble-v032). PI: "Register v0.3.1 + D-B073 shallow-crust
rule as v0.3.2 and start the ensembling for this configuration so that we get uncertainties for this velocity model";
05:50 "We needed a surface fix as there are rock velocity above the sea floor". Code: b2_model_v032.py (build, NLL,
register, checks), velocity_model_tools.py (point queries), b2_v032_compare.py (node rule against the field rule).
Outputs: /wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.3.2*, checks in
/wd1/mdenolle_data/tomo_v1/ensemble_v0.3.2/v032_build/ (nodes_written.json, check.json, compare/).

## 1. What v0.3.2 is

No new inversion: the tomography node values are v0.3.1's. Two representations of one model:

| file | what it holds | rule |
|---|---|---|
| v0.3.2.nc / .zarr (tomoDD nodes, 0.125 x 0.1 deg, depth -3, -1, 1, 3 ... 125 km) | `medium` (0 rock, 1 water, 2 air); `Vp` (rock: v0.3.1; water 1.5 km/s; air NaN); `Vs` (rock: v0.3.1 with the node rule; water and air NaN); `VpVs`; `shallow_crust_node` flag; `surface_elevation` and `offshore` (2D at the nodes); `Vp_tomography`, `Vs_tomography` (v0.3.1 everywhere, unchanged); every other v0.3.1 variable unchanged | node approximation of D-B073 |
| v0.3.2_nll/model.{P,S}.mod (1 km D-B005 Lambert lattice) | the exact v5.1 production fields: Earth-flattened, cell-centred SLOW_LEN, merged water/air cells, station rock patches, D-B005 continued outside the tomography volume | exact field rule (per 1 km cell) |
| v0.3.2_surface_lattice.nc | merged surface at the 1 km cell centres and the offshore mask of that lattice | lets readers apply the field rule at any point |

Node classification: the merged solid surface (D-B068, b4_surface.surface_m, bilinear on its first covering source)
at the node; fluid iff -z > surface (the production field's test), water iff z >= 0, else air.
Node rule: rock nodes (medium 0) in offshore columns (b4_time_grids.offshore_mask on the production 1 km lattice,
coast_run_km 5, cuts Strait of Juan de Fuca and Columbia mouth, taken at the column containing the node), z > 0 and
z + surface <= 2.5 km: Vs = min(Vs, Vp / 2.1).
NLL grids: b4_time_grids.build_field on the v0.3.1 nodes, 1 km, merged mask, field rule from params b4.shallow_crust
(never by interpolating the rule-applied nodes).

## 2. Numbers

* 232,932 nodes: rock 203,988, water 8,538 (z = 1 km 6,032; 3 km 2,506), air 20,406 (z = -3 km 11,091; -1 km 9,315).
  Above the solid surface 28,944 nodes (22,108 in offshore columns, 6,836 onshore) carried rock velocities in v0.3.1
  (Vp median 4.72, Vs 2.48 km/s); rays had reached 471 water and 1,497 air nodes (dws_p > 0).
* The node rule lowered Vs at 8,739 rock nodes (z = 1 km 765, 3 km 2,878, 5 km 4,882, 7 km 214), median 14.5 %,
  max 23.8 %; Vp/Vs there 2.100. 6 deep-ocean nodes at the SW corner (131.75 W, 38.9-39.4 N) are outside the D-B005
  lattice, so outside the offshore mask: no rule.
* Read-back: nc = zarr for Vp, Vs, VpVs, medium, flag; rock Vp = v0.3.1 exactly; rock Vs = v0.3.1 except the rule
  nodes; Vp_tomography, Vs_tomography and the 12 other variables = v0.3.1; EMC check: only the TO FILL placeholders.
* NLL grids: md5 P cc18e3e1d2044e5d05922705d741f000, S ac9ab92c4c8c52168549bcac04ea3116, identical to
  reloc_v5/work_time_b4_v0.3.1m_sc21/field_1.{P,S}.buf (build parameters equal to production's params.json).
* Reader (velocity_model_tools.velocity_at, mode "field") against the production cells: 15,788 random rock cells
  inside the model volume: max relative difference 5.9e-8 (P) and 8.0e-8 (S); 31,441 rock cells in the rule zone
  (offshore, -0.3 to 3.5 km below the seafloor; 22,141 lowered by the rule): 5.9e-8 (P), 1.1e-7 (S); medium agrees at
  16,587 / 16,587 random cells and 34,572 / 34,579 rule-zone cells. Production P = v0.3.1 trilinear to 5.9e-8 (P is
  unchanged from v0.3.1). Mode "nll" returns the grid cells exactly (difference 0).
* Node medium against the NLL cell holding the node: 206,080 of 208,696 nodes agree; the 2,616 others are nodes
  classed water or air whose cell is rock, none the other way: the cell centre sits up to 0.5 km below a node at
  -3/-1/1/3 km, and these nodes lie a median 0.22-0.28 km above the surface (max 0.92 km, station rock patches).

## 3. Node approximation against the field rule (b2_v032_compare.py)

Each node variant turned into a 1 km field by build_field with the merged mask and the field rule switched off, Grid2Time
at 1 km on both, 13 OBS (AXCC1, AXAS1, BB200, BB350, FS06B, FS08B, G03B, G19B, J25C, J49C, J73A, M08A, NC89), 3,024
paths to events 2, 4, 6, 10 km below the seafloor at 5-50 km. Variant minus production, seconds, median [p5, p95], max:

| node variant | P | S |
|---|---|---|
| v0.3.1 (no rule) | 0 | -0.244 [-0.521, -0.095], 0.94 |
| rule at every offshore node <= 2.5 km below the seafloor, above-seafloor nodes kept as rock | 0 | -0.038 [-0.193, +0.062], 0.55 |
| v0.3.2 rule (rock nodes only), above-surface nodes = unruled tomography values | 0 | -0.064 [-0.195, +0.052], 0.55 |
| v0.3.2 with water nodes as water (Vp 1.5, Vs 0.4) | +0.040 [0.000, +0.295], 0.36 | +0.056 [-0.193, +0.768], 0.98 |

Vs along depth profiles 0-4 km below the seafloor (Axial, Endeavour, JdF flank, Blanco, Gorda plate, Oregon slope,
Washington shelf, Mendocino slope), trilinear nodes minus field rule, median / max |difference|: no rule +9 to +27 % /
15-28 %; rule at every offshore node -1 to +2 % / 10-19 %; rock-only rule 0 to +3.5 % / 10-20 %; water nodes -8 to
+2 % / 10-58 %. On 2 km nodes a tomoDD model should carry rock-continued, rule-applied values above the seafloor, never
water; -18's S1 test (graded 0.5-1 km nodes) found the same (JdF S rms 88 ms with water and air, 8.6 ms rock-continued).

## 4. Proposed decisions (for the main session to record)

* P-V1: v0.3.2 = the v0.3.1 nodes with `medium` (water Vp 1.5 / Vs NaN, air NaN, D-B068 surface at the node), the
  D-B073 rule at the rock nodes, the inversion values kept as Vp_tomography / Vs_tomography; candidate.
* P-V2: v0.3.2's NonLinLoc grids are the Earth-flattened production fields (as Grid2Time used them for v5.1), not the
  true-depth grids of v0.1.0-v0.3.1; the README of each grid and the tutorial say so.
* P-V3: the exact rule for any point is the field rule (velocity_model_tools mode "field", or the NLL cells); the node
  values are an approximation and say so in their attributes.
* P-V4: a tomoDD MOD built from v0.3.2 (B5) fills medium > 0 nodes with rule-applied rock values continued upward,
  never water velocities.
