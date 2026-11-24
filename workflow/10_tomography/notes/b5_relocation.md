# B5: double-difference relocation of the whole catalog with tomoDD-SE (relocation only)

2026-11-24. Branch `tomo/b5` (worktree `~/wt_b5_full`, from `tomo/tiles` e0d46772 + merge of `tomo/b5-blanco-test`
310e59f7). Decisions: D-B053 (tomoDD-SE relocation only for B5; centroid back on B4; 50 m from the frame centre),
D-B054 (frozen binaries), D-B055 (no event above the surface, never moved), D-B060 (catalog differential times now,
waveform cross-correlation later; to be logged by the coordinator). The result becomes catalog v5.2 (v5.0 = B4 NLLoc
in model v0.3.1). Code `workflow/10_tomography/b5_relocate.py` (stages), `b5_relocate.sh` (detached driver),
`b5_check_test.py` (check against the D-B048 test); parameters `params.yaml` block `b5`; summaries
`data/tomo_v1/b5/<run>/`; large files `/wd1/mdenolle_data/tomo_v1/b5/<run>/`.

## 1. Short answer

Production run `b5_v0.3.1` (model v0.3.1, start = B4 run b4_iter1_v0.3.1) is done:
`/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/catalog_b5_b5_v0.3.1.csv` (172,339 events, `%.8g`), summary and per-cluster
table in `data/tomo_v1/b5/b5_v0.3.1/` (compare.md, clusters_summary.csv). It is **not publishable under D-B055**:
869 B5-relocated events are above the surface (all were below it at their B4 position), plus the 4,077 tier U events
that keep their v4 hypocentres (B4 kept: 0).

* Relocated: 97,619 of the 157,269 B4 3D events (62 %). Not relocated: 54,548 in no pair with >= 8 common
  station-phases (`unlinked`), 3,607 removed by tomoDD as air quakes (3,039 at Axial), 1,480 left without data after
  re-weighting, 15 in clusters tomoDD gave up on; they keep B4 (loc_source nll3d). Tier U: 15,070 (loc_source v4).
* Clusters: 2,125 components of the pair graph; the three above 13,000 events (Mendocino 21,964, Axial 21,008,
  Puget-Cascades 18,889) were each cut in two chunks with a 2-hop halo; 0 pairs lost; 2,128 tomoDD runs.
* Shift from B4 (relocated): 0.59 km horizontal (median; p90 2.1), 0.52 km vertical (|dz| median; p90 3.0), no net
  depth change (median -0.004 km). Centroid shift removed (D-B053): 0.085 km horizontal, 0.19 km vertical
  (event-weighted mean; max 0.38 / 0.59 km).
* Differential-time fit: tomoDD's own rms falls by about a quarter (e.g. Blanco 143 -> 111 ms, median cluster), but
  in the B4 time grids P is unchanged (0.147 -> 0.148 s) and S gets worse (0.181 -> 0.232 s): B5 fits a different S
  forward model than B4 (section 8, P-B5-6).
* Cluster shape: no thinning (PCA thickness sigma3 B5/B4 median 1.01; 45 % of 365 clusters thinner). Catalog
  differential times add little to B4, as the D-B048 test found on the Blanco swarms.
* Above the surface: the events that go up are those B4 pressed against the surface (36 % of the events < 0.1 km
  below it at B4 end above in B5; 0.08 % of those > 3 km deep), and tomoDD has no topography (its Air_dep level is
  1.6 km above the local surface, median). Cause = the open D-B056 upward pull, plus under-damping in large clusters
  (section 7). Nothing was moved.
* Run time: 20 min wall with 40 processes (2,128 runs, 2.1 h of tomoDD time; largest chunk 20 min, peak RSS 3.2 GB),
  plus 20 min to rerun one chunk that crashed (section 6); prep 20 s, split 5 s, collect 25 s.
* Check against the D-B048 test (v0.3.0 run): B5 and the test agree better with each other (0.13-0.34 km
  horizontal, 0.06-0.84 km vertical, median) than either with B4.

## 2. What the driver does

Stages (`b5_relocate.py STAGE --version V [--run NAME]`, each resumable; `b5_relocate.sh` runs wait, prep, split,
run, collect, commit with a state file `<run>/b5_relocate.state`).

**Start (prep).** The B4 NLLoc solutions of run `b4_iter1_v<V>` at full precision (`runs/<R1>/locations.csv`, status
LOCATED, 6 decimals), Earth-flattening undone (z = Re (1 - exp(-z'/Re)), as `b4_catalog.py`); IDs, tiers and
regions from `catalog_b4_<R1>.csv`, checked against it to its 5th digit (the v0.3.0 catalog file is still `%.5g`).
Tier U (no B4 location, v4 hypocentre kept) is not relocated. Event 172113 (mis-associated in v4.0, removed in v4.1) is
left out everywhere, catalog included (`b5.exclude_ids`).

**Picks.** Catalog v4.0 picks (the B4 input), travel time = pick - B4 origin time, kept where the event lies >= 5 km
inside the station's B4 time grid (x, y, z'); weight (0.05/sigma)^2 with the D-B008 sigma of the pick_prob class
(P 0.05/0.10/0.15 s, S twice). No repeated event-station-phase was found.

**Differential times.** The `tomodd.dt` rule: each event with its 10 nearest events within 10 km, pair kept with
>= 8 common station-phases, one dt per common station-phase, weight = the smaller of the two pick weights. Distance
is 3D at the B4 hypocentres in Earth-centred coordinates (sphere of radius Re - depth). The tiles and the D-B048
test used a flat projection about the mean latitude, which over the whole margin (40-50 N) distorts east-west
distances by up to 8 %; for one swarm the two agree.

**Clusters.** Connected components of the kept-pair graph. A component is one tomoDD run if it fits one of the frozen
binaries (`BUILD.md`; tried in the order e10k, e13k, o140, w, m): MAXEVE events, MAXDATA = dt + absolute rows, MAXSTA
stations, MAXOBS = distinct stations of one event (tomoDD's `eve_sta`, no bounds check in the optimised builds:
checked here), and RaySPDR.inc `maxnx` 90 / `maxny` 80 / `maxnz` 50 MOD nodes.

**Split rule** (components that fit no binary). Recursive bisection: the core events are cut at the median of their
projection on their principal horizontal axis; each chunk = core + halo, the halo being every event within 2 hops of
the core in the pair graph; a chunk that still fits no binary is cut again. Every event is core in exactly one chunk
and its B5 location comes from that chunk only. That is the chunk where it is most interior: there it is >= 3 hops
from the chunk edge, in any chunk where it is halo it is 1 or 2 hops from it. With a 1-hop halo every pair of a
core event is in its chunk; the second hop also keeps every pair of the core event's partners, so the events that
constrain a core event near the cut keep their own links. Counts: pairs lost (no chunk holds both events: 0 by
construction), halo-edge cuts (pairs a 2-hop halo event loses at the chunk edge; they only affect halo events), and
the split effect measured directly on the overlap: events relocated as core in one chunk and as halo in another.

**tomoDD (run).** tomoDD-SE relocation-only, variant `dd` of the D-B048 test: four sets of 3 iterations, WTCT 1,
WRCT -9/8/6/6 (MAD multiples), WDCT -9/-9/10/10 km, WTCD 0 (absolute times weightless), JOINT 0, LSQR damping 100,
OBSCT 0 (no tomoDD clustering). MOD = the netCDF nodes of model v<V> over the cluster's events and stations + 0.375
deg, edge nodes as the tiles (8 deg, -6 and 400 km), D-B028 P-2 ellipsoid rescaling about the frame centre. Frame
centre = `b5_common.frame_centre` (median epicentre + 0.0005 deg, moved until no event lies within 50 m; checked).
Air_dep = -(highest station elevation of the cluster + 0.05 km), as the test. Absolute times (weightless in dd) =
pick - B4 origin - B4 station term of `b4_iter0_v<V>`; no P-4 correction (it is per tile and the data have no
weight). Station codes S0000-S0693 (global). Binaries: `/wd1/mdenolle_data/tomo_v1/bin/frozen_2026-11-24/` (D-B054;
private, never copied). Runs in parallel (fork pool; the shared tables are inherited), largest first; a run starts
only while MemAvailable >= 30 GB + the largest RSS seen; a `running.pid` guard lets two drivers share a run folder.
Bootstrap (`--nboot K`, default 0): K more runs per cluster with the dt rows resampled with replacement (seed
48000 + k), each member's centroid put back on B4, per-event spread (rms horizontal, std depth) in the catalog
(`b5_boot_*`).

**Collect.** Per cluster, the mean shift of its relocated core events (latitude, longitude, depth, origin time)
is removed, so the centroid of the events it contributes sits on their B4 centroid (D-B053); the shift removed is
in the cluster summary. Origin time = B4 time + tomoDD's origin-time shift (10 ms steps: tomoDD writes centiseconds)
- the cluster's mean shift. Every other event keeps the B4 (loc_source nll3d) or v4 (tier U, loc_source v4)
hypocentre; `b5_status` says why (unlinked, airquake_by_tomodd, dropped_by_tomodd, tier_U, cluster_failed or
skipped). `surface_km` / `above_surface` are computed at every final position as `b4_catalog.py` does (GMT
earth_relief 01m, gridline, bilinear; depth < -surface_km), and at the start position (`b4_surface_km`,
`b4_above_surface`); nothing is moved or dropped (D-B055). `depth_quality` and `event_flag` are recomputed at the
final position (b4_catalog.flags). Catalog written with `%.8g`. dt residual rms in the B4 time grids before (B4
positions) and after (B5 positions) on the relocated core-core pairs, per-pair median removed (the metric of the
D-B048 test), by region and by cluster; tomoDD's own dt rms (first and last iteration, its forward model) and
condition number per cluster.

Catalog columns added to the B4 catalog's: `b4_time`, `b4_lat`, `b4_lon`, `b4_depth` (start), `b4_surface_km`,
`b4_above_surface`, `b4_depth_quality`, `b4_event_flag`, `b5_cluster`, `b5_component`, `b5_component_size`,
`b5_n_pairs`, `b5_ex`/`ey`/`ez` (LSQR, km), `b5_nctp`/`ncts` (dt used), `b5_rct` (tomoDD dt rms, s), `b5_dt_s`
(origin-time shift), `b5_status`, `b5_shift_h_km`, `b5_shift_z_km` (B5 - B4). `lat`, `lon`, `depth`, `time`,
`loc_source`, `surface_km`, `above_surface`, `depth_quality`, `event_flag`, `depth_frame` are the final values; the
other B4 columns (NLLoc quality: rms, gap, sig_z, h_min/h_max, shift_h_km/shift_z_km = B4 - v4) are kept as they were.

## 3. Data and clusters

| | v0.3.0 (test) | v0.3.1 (production) |
|---|---|---|
| B4 3D events (start) / tier U | 157,270 / 15,069 | 157,269 / 15,070 |
| picks kept (outside the 5 km grid margin) | 2,358,557 (37,188) | 2,358,487 (37,347) |
| candidate pairs / kept (>= 8 common) | 975,043 / 335,962 | 974,437 / 336,633 |
| differential times (P / S) | 4,323,353 (2.14 M / 2.18 M) | 4,339,354 |
| events linked / unlinked | 102,855 / 54,415 | 102,721 / 54,548 |
| components (2 / 3-9 / 10-99 / 100-999 / 1,000-9,999 / >= 10,000 events) | 2,206 (952 / 864 / 335 / 46 / 6 / 3) | 2,125 (913 / 843 / 310 / 51 / 5 / 3) |
| largest components | 22,275 Mendocino, 21,108 Axial, 18,739 Puget-Cascades | 21,964, 21,008, 18,889 |
| clusters (runs) / chunks of split components | 2,209 / 6 | 2,128 / 6 |
| binaries e10k / e13k / o140 / w / m | 2,196 / 4 / 7 / 1 / 1 | 2,115 / 4 / 7 / 1 / 1 |
| pairs in split components / lost / halo-edge cuts | 223,241 / 0 / 3,439 | 221,792 / 0 / 3,365 |
| halo events (all chunks) | 2,989 | 2,980 |
| largest run: events / stations / data / MOD nx x ny | 11,651 / 293 / 1.0 M / 65 x 70 | 11,518 / 293 / 1.0 M / 65 x 69 |

Only the three components above 13,000 events had to be split, each in two chunks (cores of 9,400-11,100
events, halos of 80-980 events; Axial has the widest halo because its events are densest). A third of the events
(35 %) are in no kept pair: most have fewer than 8 picks (the 8 common station-phase rule), as in the D-B048 test.
An event can have up to 127 stations (Puget-Cascades), above the e10k/e13k MAXOBS of 100, hence o140/w/m there.

## 4. Tests on v0.3.0 (before production)

* Small cluster c000080 (60 Blanco events): all relocated, CND 7 -> 6, tomoDD dt rms 131 -> 78 ms, shift from B4
  0.23 km horizontal, 0.20 km vertical (median), 5-9 s.
* Large cluster c000003 (4,513 Blanco events, 1,034 of the f0399 swarm): 7 min, RSS 1.0 GB, CND 135 -> 117, dt rms
  117 -> 89 ms, 4,458 relocated, 9 air quakes.
* Largest chunk c000000_p000 (11,651 Mendocino events, e13k): 22 min, RSS 2.6 GB.
* Check against the D-B048 test (`data/tomo_v1/b5/b5_v0.3.0/check_b5_test.md`; same code and model, different
  pair graph and cluster): on the events both relocated, B5 and the test differ by 0.28 / 0.19 km (f1803),
  0.34 / 0.84 km (f0399) and 0.13 / 0.06 km (f1051) horizontally / vertically (median, centroids removed), less than
  either differs from B4 (test - B4: 0.42 / 0.27, 0.67 / 1.23, 0.36 / 0.23 km). Strikes agree within 4 deg, depth
  percentiles within 0.7 km. The f0399 vertical difference (0.84 km, p90 3.2 km) is the unresolved swarm depth of
  the test (section 5 of notes/b5_blanco_test.md). For v0.3.1 the same table (`b5_v0.3.1/check_b5_test.md`) compares
  across models (the test is v0.3.0; B4 itself moved the f1803 median depth from 21 to 15 km), so only its B5 - B4
  rows are meaningful.
* Two tomoDD crashes found and handled (section 6).

The whole v0.3.0 catalog ran in 22 min with 5 processes (while B4 iter1 still ran), so it was run in full
(`data/tomo_v1/b5/b5_v0.3.0/compare.md`); it is a test, not a product.

## 5. Production results (b5_v0.3.1)

From `data/tomo_v1/b5/b5_v0.3.1/compare.md` (tables by region there).

| region | relocated (% of B4 3D) | dh med / p90 | abs dz med / p90 | S dt rms B4 -> B5 (s) | air quakes | B5 above surface |
|---|---|---|---|---|---|---|
| Axial | 17,683 (67 %) | 0.45 / 1.28 | 0.34 / 1.29 | 0.123 -> 0.158 | 3,039 | 37 |
| Blanco | 21,281 (59 %) | 0.45 / 1.64 | 0.36 / 2.42 | 0.207 -> 0.274 | 30 | 159 |
| Cascades | 9,490 (59 %) | 0.53 / 1.99 | 0.46 / 2.42 | 0.167 -> 0.202 | 145 | 102 |
| Gorda | 5,342 (55 %) | 0.66 / 2.51 | 0.83 / 4.73 | 0.223 -> 0.313 | 27 | 36 |
| JdF ridge/plate | 525 (24 %) | 0.19 / 3.49 | 0.09 / 0.67 | 0.193 -> 0.221 | 124 | 88 |
| Mendocino | 15,331 (66 %) | 0.92 / 2.84 | 0.74 / 3.23 | 0.182 -> 0.228 | 26 | 40 |
| Offshore forearc | 8,804 (62 %) | 0.73 / 2.36 | 0.88 / 4.02 | 0.202 -> 0.249 | 23 | 23 |
| Other onshore | 9,260 (65 %) | 0.73 / 2.64 | 0.66 / 3.83 | 0.186 -> 0.236 | 143 | 286 |
| Puget/Georgia | 9,447 (67 %) | 0.83 / 2.50 | 0.92 / 4.06 | 0.182 -> 0.234 | 50 | 98 |
| Explorer, Endeavour | 456 | 0.01-0.23 | | | 0 | 0 |
| all | 97,619 (62 %) | 0.59 / 2.15 | 0.52 / 2.96 | 0.181 -> 0.232 | 3,607 | 869 |

P dt rms in the B4 grids changes by less than 0.01 s in every region (0.147 -> 0.148 s overall). Condition numbers:
2 (two-event clusters) to 339 (Puget chunk); the large clusters are under-damped by hypoDD's guide (Axial 262-279,
Mendocino 268, Puget 278-339, Blanco 178). Split effect (events relocated as core in one chunk and as halo in the
other): 0.31 km horizontal, 0.25 km vertical (median), about half the shift from B4, so the halo events at the edge
of a chunk are clearly less constrained than in their core chunk. Frame centre: every start >= 50 m from it; in 44
clusters an event ended closer (min 0.4 m) without a crash. v0.3.0 (test, same code) gives the same picture:
97,647 relocated, S 0.178 -> 0.230 s, 829 above the surface (`data/tomo_v1/b5/b5_v0.3.0/compare.md`).

## 6. What failed, and the fixes

* **tomoDD segfaults in the ray tracer** (exit 139): v0.3.0 c000058 (94 events) and, in production, c000286 (14
  events) and the Puget chunk c000002_p000 (9,534 events, same iteration every time). The bounds-checked build
  (`tomoDD-SE_chk_e10k`, run on a copy in `<run>/diag/chk/`) stops at RaySPDR2.f line 1045 (intmap, called from vel3
  < ttmder < tomoDD-SE.f 951, the partial derivatives): index -2147483648 = a NaN ray coordinate. No event was near
  the frame centre (> 480 m) or under a station (> 10 km), so it is not the D-B053 case; it depends on the frame:
  the same cluster with the frame centre moved by ~0.5 km runs. The driver now retries a segfault up to twice with
  the centre moved by (+0.0037, -0.0043) deg x retry (`b5.tomodd.crash_retries`; crashed files kept in
  `crash<k>/`). The relocation then depends on the frame only through tomoDD's rounding.
* **"Cluster has less than 2 events"** (tomoDD exit 0, no output): small clusters where tomoDD removed all but one
  event (air quakes, or fewer data than unknowns after re-weighting). Status `skipped`; events keep B4.
* **Air quakes.** tomoDD removes an event from the relocation when an update would put it above Air_dep (= the
  cluster's highest station + 50 m; D-B048 rule). In v0.3.0 3,715 events (3,162 at Axial, 16 % and 13 % of the two
  Axial chunks), in v0.3.1 3,607 (3,039 at Axial). They keep their B4 position (`b5_status` airquake_by_tomodd).
  They are an above-the-surface symptom (section 7), not removed by this driver; nothing is clamped.
* **First run's state.** The v0.3.0 test clusters c000003, c000000_p000 were run by the first code version (no
  air-quake IDs in run.json); collect reads the IDs from their logs. Partial-catalog summaries of the first test
  were removed from git (7e8fe29c).
* A second (helper) run process on the same run folder, to use more cores once B4 iter1 had ended, was refused by
  the session's permission check; not needed (v0.3.0 finished in 22 min with 5 processes).

## 7. Above the surface (D-B055): counts and cause

Counts (production, compare.md): see section 1. B4 v0.3.1 has no located event above the surface (D-B056 floor);
tier U keeps its v4 hypocentres, 4,077 of which are above (v4 depths, D-B057; not relocated here). Every B5 event
above the surface is therefore made by B5. Nothing was moved, clamped or dropped by this driver; tomoDD's own
air-quake rule removed events from the relocation (they keep B4, `b5_status` airquake_by_tomodd).

What the numbers say about the cause:
1. **The events that go up are the events B4 pressed on the surface.** Of the relocated events whose B4 depth was
   < 0.1 km below the surface, 36 % (479 of 1,334) end above it in B5; 0.1-0.5 km: 13 % (152 of 1,194); 0.5-1 km:
   4 % (63 of 1,602); 1-3 km: 0.6 % (118 of 18,185); deeper: 0.08 % (57 of 75,304). Their B5 - B4 depth change is
   upward (median -0.56 km); the median B4 start of the 869 is 76 m below the surface (12.5 km for all events). This
   is the "upward pull" of D-B056: for ~4 % of B4 events the travel-time likelihood peaks above the surface and the
   NLLoc floor stops them. Catalog differential times carry the same pull, and tomoDD has no floor.
2. **tomoDD has no topography.** Its only limit is Air_dep, one flat level per cluster at the highest station
   (+ 50 m). For the above-surface events that level is 1.6 km above the local surface (median; p90 3.8 km): onshore
   clusters include mountain stations, offshore clusters include shallower OBS or land stations. Events rise up to 5.2 km
   above the surface (128 are more than 1 km above; median of the 869: 0.25 km).
3. **Air quakes are the same population.** At Axial (v0.3.1 chunk c000001_p000), with Air_dep moved to 10 km
   above sea level (`b5_diag.py`, `<run>/diag/airdep_m10/`), 752 of 10,388 core events end above the seafloor (7 %;
   median 0.57 km above, max 5.9 km); of the 1,488 events tomoDD removed as air quakes in the run, 681 end above
   the seafloor and the rest just below it (median 70 m): an air quake is an update that overshoots Air_dep at
   some iteration. Same test on v0.3.0: 793 of 10,379 (7.6 %); MSH (v0.3.0 c000007): 9 of 1,460.
4. **Damping matters but is not the cause.** One damping (100) gives condition numbers from 2 (pairs) to 339
   (Puget chunk), 262-279 at Axial, 268 Mendocino, 178 Blanco: under-damped by hypoDD's guide (40-80) for the large
   clusters. Axial with damping 400 (CND 40; `diag/airdep_m10_d400/`): 412 of 10,489 above the seafloor (4 %,
   median 0.25 km, max 2.8 km) instead of 752 (7 %); 2 air quakes instead of 19. Small well-conditioned clusters
   also go up (JdF ridge c000035: 66 of 142 above, CND 10).
5. **v0.3.0's unphysical Vp/Vs at Axial (1.35-1.40 at 3-5 km, fixed by D-B051 in v0.3.1) is not the cause**: Axial
   air quakes 3,162 (v0.3.0) vs 3,039 (v0.3.1).

So the B5 above-surface events are the open D-B056 upward pull (stations placed 50 m below the surface, near-surface
velocities, Vp/Vs, station terms: round-2 investigation) expressed by a code that has no surface, made larger by
under-damping in the large clusters. Fixing the upward pull at its root should remove most of them; a damping per
cluster would reduce the rest. Neither is done here (PI decisions, section 8).

## 8. Proposed decisions (for the coordinator to log)

* **P-B5-1 Split rule.** Components that fit no frozen binary are cut by recursive bisection on the principal
  horizontal axis with a 2-hop pair-graph halo; each event is taken from the chunk where it is core (most interior).
  0 pairs lost. Split effect measured on the overlap: 0.31 km horizontal, 0.25 km vertical (median, v0.3.1) between
  an event's core and halo solutions.
* **P-B5-2 Origin time.** B5 origin time = B4 time + tomoDD's shift - the cluster's mean shift (consistent with the
  D-B053 centroid rule; tomoDD writes 10 ms steps).
* **P-B5-3 Crash retry.** A tomoDD segfault in the ray tracer (NaN path point) is rerun with the frame centre moved
  ~0.5 km, up to twice (3 clusters: v0.3.0 c000058, v0.3.1 c000286 and the
  9,534-event Puget chunk c000002_p000; all ran on the first retry). Same kind as the D-B053 50 m rule.
* **P-B5-4 Air quakes.** Decide what tomoDD's air-quake removal means for the catalog: now the events keep B4
  (3,607 in v0.3.1, 84 % at Axial) and are flagged. They are not B5 relocations.
* **P-B5-5 Not publishable as is (D-B055).** Production has 869 B5 events above the surface (section 1). Options for the PI:
  (a) fix the upward pull at its root (round 2), rerun B4 and then B5; (b) damping per cluster to bring the
  condition number into 40-80 (hypoDD practice) instead of one damping of 100 (halves the Axial above-surface count;
  does not remove it); (c) both. Clamping or moving is excluded (D-B055).
* **P-B5-6 S misfit in the B4 grids.** B5 fits the differential times in tomoDD's forward model (rms decreases by a
  quarter) but the S differential residuals in the B4 grids grow by 28 % (0.181 -> 0.232 s; P unchanged 0.147 ->
  0.148 s): tomoDD's S times (pseudo-bending through its trilinear 0.1 x 0.125 deg node model) differ from the
  B4 grids' (finite differences on 1 km cells of the same model, Earth-flattened); not examined further here. The B5 catalog mixes two forward models (B4 grid frame for the centroid, tomoDD for relative positions).
  Before v5.2, either check tomoDD's S against the B4 grids on a few clusters (P-4-type corrections, D-B028) or
  relocate with the B4 grids (only GrowClust3D reads them; D-B053 rejected it on accuracy).
* **P-B5-7 Bootstrap.** Not run (nboot 0). Cost per member = one full run: 1.8-2.1 h of CPU, about 20 min of wall
  time with 40-48 processes (the largest chunk alone takes 13-22 min). 30 members ~ 1-1.5 h of wall time at 48
  processes. The D-B048 test found tomoDD's bootstrap 2-11 times smaller than its true error, so it is not
  recommended as the B5 uncertainty.
* **P-B5-8 Clusters too small to matter.** 913 two-event components and 843 of 3-9 events are relocated like the
  rest (centroid fixed on B4, so a pair only changes its separation). Keep, or require a minimum cluster size.

## 9. Files and how to rerun

* Code: `workflow/10_tomography/b5_relocate.py`, `b5_relocate.sh`, `b5_check_test.py`, `b5_diag.py`; parameters
  `params.yaml` block `b5` (`b5_test` unchanged). Every script writes a manifest (`write_manifest`; per-run checksums
  in `clusters/<cid>/files.json`, one manifest for the run stage).
* Summaries (git): `data/tomo_v1/b5/<run>/compare.md`, `clusters_summary.csv` (per cluster: binary, events, core /
  halo, stations, dt, run time, RSS, CND first / last, tomoDD dt rms first / last, centroid shift removed, LSQR
  errors, PCA before / after, dt rms in the B4 grids before / after), `collect.json`, `check_b5_test.md`, `diag.md`.
* Large files: `/wd1/mdenolle_data/tomo_v1/b5/<run>/` (`prep/` events, picks, dt, pairs; `split/` clusters, members;
  `clusters/<cid>/` tomoDD runs; `catalog_b5_<run>.csv`; `clusters_summary.csv`; `diag/`). 2.3 GB per run.
  Logs `/wd1/mdenolle_data/tomo_v1/b5/b5_relocate_v0.3.1.log`, `..._v0.3.0.log`.
* Rerun (resumable; production waits for the B4 catalog of the same version):

      cd ~/wt_b5_full
      VERSION=0.3.1 NPROC=48 setsid nohup bash workflow/10_tomography/b5_relocate.sh 9>&- \
          >> /wd1/mdenolle_data/tomo_v1/b5/b5_relocate_v0.3.1.log 2>&1 < /dev/null &

  A new B4 run or model: `VERSION=x.y.z` (and `RUN=`, `R1=` to name them). To redo one cluster, delete
  `clusters/<cid>/` and run `python b5_relocate.py run --version V --only <cid>`, then `collect`.
* Production history: the run stage (10:46-11:06, 40 processes) ran before the crash retry existed; c000286 and
  c000002_p000 were rerun with it (11:07-11:31) and `collect` was run again (11:31). The commit made by the driver
  at 11:06 holds the first collect (c000002_p000 missing); the following commit replaces it.
