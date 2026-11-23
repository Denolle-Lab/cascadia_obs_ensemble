# Track B decision log

Every choice that changes the result gets an entry: what was decided, the alternatives, why, and the
evidence (file, figure, number). Entries are never edited after the fact; a change of mind is a new
entry that says which one it supersedes. Parameter values live in `params.yaml` and point back here.
Who: PI = Marine Denolle; Agent = Claude (session or subagent named).

Status: **adopted**, **proposed** (waiting on evidence), **superseded by D-Bxxx**.

---

### D-B001 Mount Hood out of scope (adopted, 2026-11-22, PI)
Mount Hood depths are left for future work with PNSN picks ("PNSN territory"). Hood events stay in
the catalog with their Track A tiers (all C/D). No special treatment in the tomography: they enter
the inversion set only if they pass the generic criteria (D-B003).
Evidence: Track A Hood test, three models give median depths -1.0 / 8.1 / 16.7 km at equal RMS
(workflow/08_relocation/README.md).

### D-B002 Sequence (adopted, 2026-11-22, PI)
B1 tomography event and pick set -> B2 joint Vp/Vs tomography with relocation, starting from the
Track A model -> B3 checkerboard and spike tests -> B4 NonLinLoc relocation of all 172,340 events in
the new model, new station terms, new tiers -> B5 GrowClust relative relocation (last, on the B4
locations). GrowClust is not run on Track A locations.
Alternative: GrowClust now on Track A (rejected: would be redone after B4).

### D-B003 Tomography event set (adopted, 2026-11-22, PI approved the recommendation)
From the Track A iter1 catalog: tiers A-C, nphs >= 12, gap <= 200 deg, RMS <= 0.5 s; decluster in
3 km cubic cells keeping the 3 most-picked events per cell. Expected ~23k events, ~0.70 M picks.
Alternatives (counts from iter1, 3 km x 3): A+B only, 20k events / 0.58 M picks but drops most of
Blanco and Gorda (tier C there: no station within 1-2 focal depths, epicentres fine);
A-D, 65k events / 1.1 M picks, adds depth-velocity trade-off from unconstrained events and is
dominated by Axial and Mendocino. Tiers are a poor filter for tomography because the tier depth
test depends on the Track A model; pick count, gap and ray coverage are what matter.

### D-B004 Starting hypocentres (adopted, 2026-11-22)
Track A run iter1_vpvs176 maximum-likelihood locations. They are relocated jointly in B2.

### D-B005 Starting model (adopted, 2026-11-22)
Track A 3D model model_vpvs176: CRESCENT Gen0 Vs + Brocher (2005) Vp with Vp/Vs >= 1.76; outside
Gen0 a young-ocean 1D plus Axial (Wilcock 2016), Endeavour (Weekly 2013), Blanco (Kuna 2019).

### D-B006 Static station terms (adopted in principle, 2026-11-22; OBS handling proposed)
Track A static station terms are not applied in the tomography: they absorb the shallow structure
the inversion should resolve. Exception to be decided: OBS S delays (up to ~±2 s, sediment below
grid resolution) need either a sediment layer in the starting model or a per-station S correction
(see notes/dataset.md, notes/design.md).

### D-B007 Tomography code (proposed, 2026-11-22)
TomoATT (adjoint-state eikonal traveltime tomography, Chen et al. 2023) is the primary code:
spherical grid, topography and seafloor stations, scales to ~10^6 arrivals, joint relocation.
tomoDD is not used in this pass (Cartesian, fixed array sizes, weaker on a ~1,400 x 1,000 km domain).
Adopted once the build and the code's own examples pass (notes/tomoatt_build.md).

### D-B008 Pick uncertainties (adopted, 2026-11-22)
Same classes as Track A (a2_make_obs.py): P sigma 0.05 / 0.10 / 0.15 s for pick_prob >= 0.6 /
0.3-0.6 or missing / < 0.3; S twice P. Mapping to TomoATT weights: see notes/dataset.md.

### D-B009 Pick QC (adopted, 2026-11-22, coordinator; proposal in notes/dataset.md)
Keep picks NLLoc used in iter1 (weight > 0); drop |iter1 residual - median| > 4 sigma_MAD with median and
sigma_MAD from iter0 (P 0.83 s, S 1.80 s); drop S picks with tS/tP outside [1.4, 2.5] or station-corrected
S-P residual beyond 5 sigma_MAD (1.23 s); no minimum S count. Testing iter1 (with station terms) rather than
iter0 residuals avoids cutting OBS preferentially (4.8% of OBS P vs 2.5% land at k = 4 on iter0).
Effect: -9.4% picks (weight 0), -0.9% (residual + S-P). Params: params.yaml `pick_qc`.
Amended by D-B012 for the weight-0 picks.

### D-B010 OBS S correction, exception to D-B006 (adopted, 2026-11-22, coordinator)
For B2 iteration 1, subtract the Track A iter1 S station term from OBS S travel times only; land stations
and all P picks are uncorrected. Move to jointly solved, damped station statics if TomoATT supports them.
Drop the 92 OBS S picks at stations without an S term. Before B2: waveform check of the deep-water OBS with
the most negative S terms (W01.7A, J35A.7D, M03C.7D, W02.7A) to tell a too-slow shallow model from Sp
picked as S. Evidence: shelf OBS median S term +0.55 s vs P +0.06 s; 41/205 deep-water OBS with S term
< -0.5 s, which a sediment layer cannot produce. Alternatives: sediment layer (not resolvable on the
inversion grid, does not explain negative terms); drop OBS S (-31% of S picks).

### D-B011 Events below D-B003 after pick QC (adopted, 2026-11-22, coordinator)
Keep an event with >= 10 kept picks and gap <= 200 deg (284, flagged in events.csv); drop the other 121
(listed in events_below_threshold.csv); no replacement within the cell.

### D-B012 Re-admit far-station picks in B2 (adopted, 2026-11-22, coordinator)
The 72,868 weight-0 picks (median distance 216 km) were rejected by NLLoc only because Track A searched
each 25 km tile in a box of tile +-75 km, so stations beyond ~150-175 km had no time grid there: a Track A
geometry artefact, not a pick-quality flag. They are the longest rays and carry the deep (slab, mantle wedge)
constraint. B2 re-admits them (from picks_rejected.csv) after a forward calculation in the starting model,
with the same 4 sigma_MAD residual test as D-B009 computed on the forward residuals. Report the count
re-admitted and the depth coverage gained.

### D-B013 Full-precision Track A coordinates (adopted, 2026-11-22)
catalog_v5_trackA_*.csv was written with float_format %.5g (lon to ~+-400 m). Fixed in
workflow/08_relocation/a2_build_catalog.py (%.8g) and the catalog rebuilt. B1 already read
full precision from runs/iter1_vpvs176/locations.csv, so the B1 dataset is unaffected.

### D-B014 Code facts that reshape D-B002/D-B007 (adopted, 2026-11-22; source notes/design.md, TomoATT commit 8b72f69)
TomoATT inverts one velocity field per run, so "joint Vp/Vs" = separate P and S runs, and in-run
relocation sees only one phase. It does not solve station statics for absolute local times
(`use_sta_correction` only with common-source differential data). Consequences: B2 is an outer loop of
[P run, S run with hypocentres fixed] -> [external relocation with P+S in the updated models], up to 3
loops; OBS corrections (D-B010 / D-B019) are applied to the picks before inversion. To verify on our build
(notes/tomoatt_build.md, pending PI approval of the build).

### D-B015 Forward grid (proposed, 2026-11-22)
Spherical 0.02 deg (lat) x 0.025 deg (lon) x 1.0 km, z -3 to 80 km (20.2 M nodes, ~130 GB at 32
simultaneous sources). Adopted only if it passes an accuracy test against a 0.5 km reference:
P median error <= 0.02 s, 95th percentile <= 0.05 s, including slope and shelf OBS. Fallback: 3 x 3 x 1 km.

### D-B016 Inversion grid and optimisation (adopted, 2026-11-22)
Five staggered multigrids, 0.30 x 0.40 deg, then 0.20 x 0.25 deg; depth nodes -3 to 80 km (notes/design.md
section 2.2), horizontal spacing x1.5 below 30 km. Gradient descent, max update 1 %/iteration,
Kdensity_coe 0.3, no Laplacian smoothing. Stop at chi^2 <= 1.1, or < 1 % misfit gain over 5 iterations,
or 20 iterations per run. Isotropic (no anisotropy in the first model).

### D-B017 Data types and weights (adopted, 2026-11-22)
Absolute P and S throughout; common-source differential times (weight 0.5, separation <= 60 km) added in
stage 2; no common-receiver differentials in B2 (GrowClust in B5 does that). Weights: pick (0.05/sigma)^2;
residual down-weighting P [0.5, 1.5] s, S [1, 3] s; distance taper 150-250 km (applies to the D-B012
re-admitted picks); event-density weights. Test the Bai et al. (2026) S-P correction as a variant, not
the baseline.

### D-B018 Resolution and robustness (adopted, 2026-11-22)
Before any interpretation: checkerboards 40 / 60 / 100 km by depth at +-8 %, two phases of the pattern,
noise per pick sigma, perturbed hypocentres run through the full outer loop; 6 spike tests; restoring
test. "Resolved" = local correlation >= 0.7 and amplitude recovery >= 0.5. Robustness: inversions from a
1D start and from Delph et al. (2018) onshore, 5 x 70 % jackknife, a no-OBS run. Interpret only features
stable to 2 % (Vp) / 3 % (Vs) across these.

### D-B019 OBS corrections: open conflict (proposed, 2026-11-22)
notes/design.md proposes a priori sediment delays from the literature (Rychert 2018, Ruan 2014, Bell 2016,
own receiver functions, GlobSed) instead of D-B010's Track A S terms, on the claim that Track A OBS S
terms are "centred near zero". notes/dataset.md shows the median is near zero only because shelf OBS
(+0.55 s median) and 41 deep-water OBS (< -0.5 s, to -2.35 s) cancel. Negative delays cannot be sediment,
so a sediment-only correction would leave them in the data. Resolution waits on the OBS waveform check
(notes/obs_check.md). D-B010 stays in force until then.

### D-B020 OBS S statics from near events only (adopted, 2026-11-22, coordinator; supersedes D-B010, resolves D-B019)
Waveform check (notes/obs_check.md; data/tomo_v1/obs_check/): at 16 deep-water OBS with negative Track A
S terms, 8 shelf OBS, 3 control OBS and 3 land stations (<= 25 events each, pnwstore archive), every S pick
checked is the true S onset (S-Ppred 10-18 s, moveout slope 0.83-0.96 as controls, horizontal onset at the
pick, H/Z 1.7-15), not Ps or Sp. The negative terms are a distance-dependent path misfit (S term vs median
pick distance r = -0.47; most negative stations have median pick distance 125-190 km; controls show the same
trend beyond 100 km), i.e. a starting-model error for B2 to fix, not a station constant. Shelf delays are
station constants (near-event term = Track A term within 0.05 s; P terms ~0), consistent with 0.2-0.9 km
of Vs 0.3-0.6 km/s sediment.
Rule: keep all OBS S picks. Static = median no-term S residual of picks within 60 km (n >= 10); else
within 100 km (n >= 10); else no static (pick kept uncorrected). B1 classes: 95,804 / 8,336 / 2,879 picks.
13 OBS keep a static < -0.5 s (was 45), none below -0.91 s. P: no OBS statics (terms ~0).
Literature sediment delays (D-B019 proposal) are used only as an independent check of the shelf statics,
not applied. Follow-up for B2: check whether the starting Vp/Vs of 1.76 is too high in the oceanic plate.

### D-B021 tomoDD instead of TomoATT (adopted, 2026-11-22, PI; supersedes D-B007, and D-B014 where it assumed TomoATT)
The tomography uses tomoDD (Zhang & Thurber) with the catalog v4 picks as absolute travel times, plus
catalog differential times built from the same picks. tomoDD solves Vp, Vp/Vs and hypocentres in one run, so
the separate P/S runs and outer loop of D-B014 are not needed. Consequences to work out (notes, next steps):
Cartesian grid and compile-time array sizes, so the margin is split into overlapping sub-regions of up to
~400 km (curvature ~38 km over 700 km); source and build need PI approval. Uncertainty from a Bayesian
ensemble over station terms, pick noise and the starting velocity model (design to come, D-B023).

### D-B022 B1b balanced dataset of well-recorded stations (adopted, 2026-11-22, PI approved the plan)
Core stations: per phase >= 100 picks (land) / 30 (OBS), >= 3 (land) / 2 (OBS) back-azimuth sectors of
45 deg and >= 2 of 4 distance bins with >= 5 picks each; P or S may pass. Weights instead of a cap:
w_bin = min(1, 40/n) per station, phase, 25 km x 30 deg bin; region weights to equal shares over 7 regions
(clip 0.33-3); events keep >= 10 core picks and gap <= 200. Result: 424 stations (174 OBS), 22,976 events,
677,235 picks; station Gini 0.65 -> 0.39, top-50 share 0.37 -> 0.29, region shares 7.8-24.2% -> 13.8-14.4%,
Kish n_eff 462k. Alternatives: hard cap per bin (K = 40 loses a third of the picks and 7.5k events with >= 10
picks; region shares unchanged), P-only coverage test (drops S-rich Mendocino/Gorda OBS), 3 sectors for
OBS (drops the Blanco arrays, linear geometry). Evidence: notes/balanced.md, data/tomo_v1/b1b_balanced/.
Params: params.yaml `balance`.

### D-B023 Ridge regions back in with softer criteria (adopted, 2026-11-22, PI request; amends D-B003, D-B022)
Axial, Explorer and JdF ridge/plate had dropped to 109 events in B1 (Axial: 3 km declustering of a caldera a
few km across; Explorer/JdF: stations 30-55 km away, nphs < 12) and to ~0 in B1b (station test). New rule for
these regions only: tiers A-D, nphs >= 8, gap <= 270, RMS <= 0.5, 1 km cells x 3; after QC >= 6 picks, gap <= 300;
stations with >= 10 ridge-event picks kept for those picks only, no coverage test; the three regions are one
weight group ("Ridges", weight clipped at 3, 7.5% of w_bal). Pick-QC spreads come from main-rule events only, so
the main set is unchanged (23,319 events; 2 of 690,809 picks differ). Result: +1,294 events (Axial 428, Explorer
403, JdF 463), +12,663 picks, 80 stations (57 ridge-only). Evidence: notes/balanced.md section "Ridge regions";
params `selection.ridge`, `balance.ridge`, `balance.region_groups`.

### D-B024 "Track A" renamed v4-NLL3D (adopted, 2026-11-22, PI)
The catalog v4 relocated with NonLinLoc in the 3D starting model (D-B004/D-B005; Track A run iter1_vpvs176) is
called **v4-NLL3D** in text, figures and the atlas ("v4-NLL3D (preliminary)" until B4 replaces it). File and
variable names stay (params `trackA_*`, /wd1/.../reloc_v5/catalog/catalog_v5_trackA_iter1_vpvs176.csv); alias
symlink catalog_v4-NLL3D_iter1.csv in the same folder. Earlier decision entries keep the old name (log not edited);
notes and script messages updated.

### D-B025 Overlapping regional inversions (adopted, 2026-11-22, PI)
tomoDD runs on overlapping sub-regions of up to ~400 km (flat-earth/Cartesian grid and compile-time array sizes;
earth curvature ~38 km over 700 km). First region Mendocino-Gorda. Region boundaries, overlaps and how the overlap
zones are compared or merged: proposed with the tomoDD audit (notes/tomodd_build.md) and the regional design note.

### D-B026 tomoDD version, build and run strategy (proposed, 2026-11-22, agent; source notes/tomodd_build.md)
Code: tomoDD-SE (Zhang & Thurber; spherical pseudo-bending) from the USTC 2020 training package (archive sha256
e618c691...275fe1; same source as the GitHub mirror travisalongi/tomoDD-SE-training2020; the 2017 USTC package is
the same code without the 2020 fixes, incl. the longitude wrap that negative longitudes need). No author repository
exists; ask Haijiang Zhang to confirm the version. Built with gfortran 8.5 (gfortran 13 stops at a malformed FORMAT),
shipped flags minus -ffloat-store (1.72x faster, outputs byte-identical); not -march=native (changes hypocentres by
0.1-0.3 km median, up to 5-8 km: compiler, flags and machine are pinned for all ensemble members). Shipped example
reproduced byte for byte. Mendocino benchmark (9,334 events, 828k data, 40x40x23 nodes, 5 iterations): 9.9 min on one
core, 2.5 GB; ensemble members run as parallel serial jobs (64 at once = 34x throughput). Licence (WARF academic):
use allowed; no modification beyond the array sizes the manual asks users to set, no public posting, so source and
binaries stay out of git and Zenodo. Input rules: weights rescaled into (0, 3]; numeric station codes (7-character
limit); OBS statics applied to the picks; top grid edge at or above -4 km; air-quake depth about -2.2 km.
Amends D-B025: ray tracing is spherical, so the sub-region size is set by memory and run time, not by curvature.

### D-B027 D-B026 adopted (adopted, 2026-11-22, PI)
The PI approved D-B026 as written (tomoDD-SE USTC 2020 package, gfortran 8.5, shipped flags minus -ffloat-store,
array sizes, parallel serial jobs, input rules). Note: the D-B026 header had been edited in place to "adopted";
restored to its original wording so the log stays append-only, and the adoption recorded here.

### D-B028 Forward-model consistency fixes (adopted, 2026-11-22, PI; source notes/forward_accuracy.md)
Finding: in the same nominal model, the production NLL time grids (v4-NLL3D, reloc_v5/time_vpvs176) are too slow by
+0.21 s P / +0.37 s S median (1.5-2% of travel time): half-cell shift of node-sampled velocities in Podvin-Lecomte
cells (+0.08/+0.15), 1 km FD error (+0.10/+0.16), no Earth-flattening (+0.03/+0.05). tomoDD-SE in its MOD is close
to exact (+0.016/+0.033 s). Reference: Grid2Time at 0.5 and 0.25 km extrapolated to zero cell size, checked against
TauP in 1D (median 0.000 s). Adopted, the agent's proposals P-1..P-7:
P-1 NLL grids for B4 (and any v4-NLL3D rerun): cell-centred velocities, Earth-flattening, T = T0.5 - 1.33 (T1 - T0.5)
on 1 km nodes (+0.014 / +0.022 s). P-2 tomoDD coordinates rescaled so sphere distances equal WGS-84 distances
(inputs only), outputs transformed back. P-3 MOD from the nearest NLL rock column, depth nodes 1 km to 9 km then 2 km
to 49 km (42 interior depth nodes), bounds-checked run before use. P-4 per-station, per-phase tomoDD forward
correction (median tomoDD - reference in the starting model) subtracted from absolute times. P-5 final model to NLL
by sampling tomoDD's field at cell centres, grids per P-1. P-6 recompute the D-B020 OBS S statics with the P-1
forward (current ones carry ~0.26 s of the NLL error). P-7 relocate one or two tiles with P-1 grids to measure the
v4-NLL3D depth and origin-time bias.
Consequences: the v4-NLL3D catalog (atlas layer "preliminary") was located with times 1.5-2% too slow; the
Mendocino-Gorda L-curve (summary_runs.csv, damping 100-3000) used the old MOD and statics and is superseded
except as a guide for damping (~1000, CND ~60). The Bayesian ensemble design becomes D-B029.

### D-B029 Depth nodes kept; smoothing checked by spike test (adopted, 2026-11-22, PI; amends D-B028 P-3)
Spike test of tomoDD smoothing against uneven node spacing (notes/smoothing_test.md, parallel session; proposed
there under the label "D-B028", renumbered here): at smooth 20/20/10, spikes in the resolved crust (12-34 km) are
not broadened (factor 1.0-1.1 laterally and vertically); above 12 km almost nothing is resolved at node scale
(1 of 25 spikes at 3 and 7 km), so the upper crust is coverage-limited. PI: choose the option grounded in that
physics. Decision: keep the current node_z (21 interior depth nodes); from D-B028 P-3 keep only the nearest-NLL-
rock-column sampling (forward sampling error +0.018 / +0.031 s instead of +0.012 / +0.017 s with 42 levels; the
difference is far below pick sigma and partly absorbed by the P-4 station corrections). Finer upper-crust nodes
would add unresolved parameters, double derivatives per ray and worsen thin-layer smoothing. Repeat the spike test
at the smoothing and damping finally selected and on every new region grid; if broadening in the resolved crust
exceeds ~1.5, move to more even spacing in 0-40 km. Report upper-crust structure only where spike/checkerboard
tests show recovery. The Bayesian ensemble design becomes D-B030.

### D-B031 Forward fixes implemented for Mendocino-Gorda v2; P-4 sign corrected (adopted, 2026-11-22, coordinator; amends D-B028)
(D-B030 is reserved for the Bayesian ensemble design.) Implementation in notes/forward_accuracy.md section 7.
* P-4 sign: D-B028 said "subtracted"; the correction c = median(tomoDD time - P-1 reference) per station and phase is
  ADDED to the absolute times (tomoDD residual obs - t_tdd = (obs + c) - t_tdd for t_tdd = t_ref + c). The end-to-end
  check caught the wrong sign (P bias doubled to -0.095 s). With +c: median tomoDD - reference residual within
  +-0.02 s (P) and +-0.035 s (S) in every station-type and distance bin; p95 0.067 (land) / 0.119 (OBS) s for P,
  0.111 / 0.234 s for S (path-dependent node-scale error).
* P-4 corrections kept uncapped, including +0.54 to +0.82 s (S) at five shelf/slope OBS (G22D, FS07D, FS02B, BB660,
  BB200): they hold sub-node seafloor structure the 10 km grid cannot represent, the same physics as the D-B020
  statics; a cap would push the remainder into neighbouring nodes. Revisit if those stations stand out in residuals.
* P-6 statics recomputed with the P-1 forward for the 110 Mendocino-Gorda OBS (108 with a static):
  data/tomo_v1/obs_check/obs_near_station_S_terms_p1.csv (old file kept). Every static increases (median +0.27 s,
  r = 0.94 with the old NLL error); only 1 static below -0.5 s (was 8): with a correct forward, near-station OBS S is
  ~0.3 s late, consistent with sediment. Other regions: run b2_fwd_p1.py and b2_fwd_statics.py per region.
* P-2 coordinate rescaling and nearest-column MOD (D-B029 node_z) in inputs <runs>/mendocino_gorda_v2/inputs; the
  builder defaults reproduce the first build byte for byte. Bounds-checked run clean through rays, G and LSQR
  (36.2 M derivatives, 44 per datum of MAXND 100); the debug binary then stops on a malformed FORMAT in a log write
  (debug build only). Production 1-iteration run: 284 s, 2.65 GB. P-1 1 km region grids kept (4.8 GB); they are
  region-limited and not the B4 grids.

### D-B030 Bayesian uncertainty by randomize-then-optimize ensemble (adopted, 2026-11-22, PI; design notes/bayes_design.md)
Ensemble of full tomoDD inversions, each with its own draws of: starting model (mixture of the v4-NLL3D model with
Vp/Vs 1.76 / 1.73 / 1.80, plus Delph et al. 2018 where it covers the region, each with a 3% Vp and Vs random field,
correlation 30 km laterally and 8 km in depth, tapered below 60 km), station statics (OBS S around the D-B031
statics, others around 0), forward-error offsets, pick noise (class factors by empirical Bayes), starting hypocentres
(v4-NLL3D covariance), damping (700-1500) and smoothing. Members: 100 for the synthetic calibration, 200 for real
data per region. Gate: synthetic calibration (68/95% interval coverage on resolved nodes and hypocentres) must pass
before any real-data ensemble. B4 relocates all events in 20 ensemble models for location uncertainty. Damping
baseline 1000 (CND ~60, Mendocino-Gorda v2 scan); smoothing chosen from spike/checkerboard tests (D-B029 repeat).

### D-B032 Ensemble size 100 members (adopted, 2026-11-22, PI; amends D-B030)
100 members per region for both the synthetic calibration and the real-data ensembles (D-B030 had 200 for real data).


### D-B033 Four more tiles for tomoDD (adopted, 2026-11-23, PI request "set up the other regions"; implements D-B025)
Tiles beside Mendocino-Gorda (events box; interior nodes 0.125 deg x 0.1 deg as Mendocino-Gorda, 0.75 deg / 0.5 deg
beyond the box; node_z unchanged). Counts are B1b events inside the box and stations within the box + 0.5 / 0.7 deg:

| tile | events lat | events lon | events | stations | max stations per event | B1b regions |
|---|---|---|---|---|---|---|
| mendocino_gorda (existing) | 39.5-43.1 | -127.5 to -122.5 | 9,334 | 153 | 57 | Mendocino, Gorda, south forearc and onshore |
| blanco | 42.6-45.2 | -130.6 to -125.6 | 4,294 | 81 | 28 | Blanco |
| oregon | 42.6-46.4 | -126.2 to -121.0 | 3,061 | 156 | 88 | Oregon onshore, Cascades, forearc |
| washington | 45.8-49.8 | -126.4 to -121.0 | 7,968 | 189 | 109 | Puget/Georgia, Washington onshore, Cascades, forearc |
| ridges_north | 44.4-50.0 | -131.0 to -126.6 | 1,489 | 78 | 14 | JdF ridge/plate, Axial, Explorer, north Blanco |

Overlaps (for the D-B025 comparison): 1,980 events fall in two tiles; 83 B1b events fall in none (57 Gorda west of
127.5 W, 16 Mendocino, 10 onshore). Oregon's eastern node column stops at 120.375 W, where the v4-NLL3D model ends
(x = +457 km at 42 N). ridges_north is 620 km north-south, above the ~400 km guide of D-B025; with tomoDD-SE's
spherical geometry the limit is memory and run time (D-B026), and this tile is the smallest in data.
Washington needs MAXOBS 140 (109 stations per event + 20 %, notes/tomodd_build.md): builds O3nofs_o140, syn_o140 and
chk_o140 (sources copied from O3nofs_big, syn_big, chk_big, MAXOBS only changed), selected per region by
`tomodd.regions.<tile>.bins` (manifest.tomodd_bin).
Each tile is built as Mendocino-Gorda: <tile> first build, P-1 reference from its picks, per-tile OBS S statics
(obs_near_station_S_terms_p1_<tile>.csv), <tile>_v2 with the D-B028/D-B031 forward fixes, check01 bounds run.
Resolution tests per tile (D-B018) at damping 1000: the Mendocino-Gorda v2 set (spikes at 7 depths and
checkerboards 40/60/100 km at smoothing 10/20/40; pick noise, both pattern phases and perturbed locations at
smoothing 20) on a spike lattice 8 nodes apart inside each events box, plus the real-data damping/smoothing scan
(d500/1000/2000 at s20, s10/s40 at d1000). Driver: b2_res_pipeline.sh (detached, resumable, publishes to GitHub).
Damping and smoothing per tile are chosen after these runs, as for Mendocino-Gorda.

### D-B034 FORMAT-fixed tomoDD builds for the new tiles (adopted, 2026-11-23, agent; bug fix)
tomoDD-SE.f line 2324 (and tomoDD-SE_syn.f) writes the "Location summary" with `f7.3")"`, a missing comma that
gfortran's runtime rejects ("Missing comma between descriptors") in some runs: always in the -fcheck=bounds build
(notes/forward_accuracy.md 7.4), never in the Mendocino-Gorda production runs, and in every ridges_north run after the
first LSQR (exit 2, no model written). Fix: `f7.3,")"` in both writes; nothing else changed. Builds O3nofs_fmt,
chk_fmt, O3nofs_o140_fmt, chk_o140_fmt (binaries *_fmt in tomo_v1/bin), selected per tile with `bins`. A ridges_north
test (d1000, s20) with the fixed build ran to the end (exit 0, 12 location summaries). Mendocino-Gorda v2 and v2s keep
the original builds, so their results stay reproducible; the forward (syn) builds are unchanged.

### D-B035 Offshore forearc kept: soft selection rule and OBS keep all picks (adopted, 2026-11-23, PI)
PI: offshore Washington is hard to image but should be in the model, poorly constrained if need be, with the
bootstrap ensemble grading it; balancing should not remove regions from the map. Before this change the D-B003
selection kept 371 of 2,196 v4-NLL3D events off Washington (west of 124.6 W) and 123 of 761 off Oregon (west of
124.3 W), mostly losing events to nphs >= 12 and gap <= 200; B1b then dropped all 223 OBS picks of Washington-tile
events, because the 17 OBS there passed only the ridge bar and kept only ridge-event picks.
Changes: (1) the region `Offshore forearc` uses the D-B023 soft rule (tiers A-D, nphs >= 8, gap <= 270, rms <= 0.5,
1 km declustering; after B1b >= 6 picks, gap <= 300); (2) `balance.ridge.obs_keep_all`: an OBS admitted by the soft
station bar keeps all its picks, not only soft-rule ones (88 OBS, +3,346 picks). Weights unchanged (w_bin, region
targets), so dense regions stay down-weighted. Outputs versioned: b1_dataset_c, b1b_balanced_c (B1 inputs identical,
picks file = catalog v4.0, md5 17b7d0bced600f8c83e4e1a8c5881e38); b1_dataset / b1b_balanced are kept and
mendocino_gorda_v2 / v2s (ensemble work) stay on them.
Result: 28,986 events (was 24,249), 754,710 picks (689,602), 521 stations (475); weighted region shares 13.2-13.3 %
each (ridges group 7.1 %, weight clipped at 3).

| tile (events box as D-B033) | events old -> c | stations | OBS stations | OBS picks | offshore forearc events |
|---|---|---|---|---|---|
| mendocino_gorda (-> mendocino_gorda_c) | 9,334 -> 12,191 | 153 -> 161 | 111 -> 113 | 97,823 -> 102,789 | 1,471 -> 4,296 |
| blanco | 4,294 -> 4,378 | 81 -> 87 | 81 -> 87 | 83,640 -> 85,293 | 25 -> 87 |
| oregon | 3,061 -> 3,390 | 156 -> 181 | 21 -> 39 | 771 -> 2,005 | 138 -> 470 |
| washington | 7,968 -> 9,332 | 189 -> 225 | 0 -> 22 | 0 -> 297 | 556 -> 1,921 |
| ridges_north | 1,489 -> 1,592 | 78 -> 82 | 73 -> 76 | 14,153 -> 14,537 | 0 -> 91 |

Offshore Washington stays recorded mostly by land stations (297 OBS picks over 22 OBS), so expect poor resolution
there; the checkerboards and the ensemble will say how poor. New builds for the sizes (FORMAT-fixed, D-B034):
O3nofs_e13k_fmt / chk_e13k_fmt (MAXEVE 13000, MAXDATA 2e6) for mendocino_gorda_c, O3nofs_w_fmt / syn_w_fmt /
chk_w_fmt (MAXSTA 260, MAXOBS 140) for washington. The tile runs started under D-B033 on the old dataset were stopped
and deleted before any result (logs in b2_tomodd/_superseded_pre_DB035/).
Not done here: CASIE21 (2021 active-source survey off Washington and Oregon) as a constraint on the shallow offshore
starting model / prior; to be taken up with the starting-model items (D-B005, D-B030).

### D-B036 Novelty wording for the tomography (adopted, 2026-11-23, PI; P-F1 of notes/science_framing.md)
Claim: the first local-earthquake Vp and Vs model of the crust and seismogenic volumes across the whole Juan de
Fuca plate system, from the ridges to the arc, onshore and offshore, with ensemble uncertainty and in the same frame
as the earthquake locations, magnitudes and fault surfaces. Never "first plate-wide local-earthquake tomography":
VanderBeek & Toomey 2019 (JGR, doi:10.1029/2019JB017707) inverted Pn times from local earthquakes for 3D P velocity of
the uppermost mantle across the whole Juan de Fuca and Gorda plates. Plate-wide teleseismic P and S (Byrnes et al.
2017; Bodmer et al. 2020) and Rayleigh-wave Vs (Bell et al. 2016) models also exist. The manuscript placeholder cited
"VanderBeek et al. 2020 G-cubed", which does not exist; it is Bodmer et al. 2020. References in
paper_sciadv/references.bib since PR #34 (merged cbaf5efc).

### D-B037 Fault-zone properties as the main tomography product for interpretation (adopted, 2026-11-23, PI; P-F2)
For the joint interpretation of sources and structure, report Vp, Vs and Vp/Vs averaged in a volume around each
fault surface (workflow/07_faults), with the posterior over the ensemble members (D-B030, D-B032). Node maps are shown
only inside the resolution masks. Volume width from the spike half-widths of each tile (8-9 km laterally where the
Oregon spikes are resolved). Reason: Vs at single nodes is resolved over 16-26 % of the Vp area at the Blanco and
67-78 % in Oregon and Mendocino-Gorda (15 km, damping 1000), so volume averages are what the data support. Width and
averaging set in params.yaml when the step is implemented.

### D-B038 ridges_north used for relocation only, with one exception (adopted, 2026-11-23, PI; P-F3)
ridges_north resolves nothing on the checkerboard lattice at any damping (1000, 300, 200, 100). It is used to relocate
events, not interpreted as structure (answers open question 3 of the STATUS handoff). Exception written into P-F3
(cluster spikes at damping 100 recovering >= 50 % somewhere) is met: at damping 100, smoothing 20, 3 P and 2 S
cluster spikes with DWS > 1000 recover 50-64 %, all at 12 km between the northern Blanco and Axial (44.4-45.4 N,
130.0-130.3 W; data/tomo_v1/b2_res_tests/ridges_north_v2/d100/). That volume may be reported from the damping-100
model inside its mask, and compared with the blanco tile where they overlap (D-B025). The ridges_north damping itself
(open question 1) stays open.

### D-B039 Location uncertainty from 10-20 ensemble members (adopted, 2026-11-23, PI; P-F4)
The relocation of all 172,342 events in the final model (B4, B5) is repeated in 10 to 20 members of the real-data
ensemble, not all 100, to give location uncertainty. Reason: 100 x 172,342 relocations is heavy, and 10-20 members
sample the model spread for locations. Exact number and member choice set in params.yaml when B4 is implemented.

### D-B040 Waveform cross-correlation for clustered families only (adopted, 2026-11-23, PI; P-F5)
Before the final double-difference relocation (B5), compute waveform cross-correlation differential times within
the clustered families of the catalog (catalog v4 nearest-neighbour clustering: 177 swarms and 822 multiplets among
1,944 families; redone on the relocated catalog), and add them to the catalog differential times for those events.
Reason: catalog times from picks cannot separate the Blanco strand pairs about 2 km apart or confirm repeating
earthquakes; correlation for every event pair of 172,342 events is not needed.

### D-B041 b-value comparisons between settings only after an ML-Mw check by region (adopted, 2026-11-23, PI; P-F6)
After the magnitudes are recomputed with the new hypocentres (STATUS step 9), compare ML with Mw by region using the
112 matched ComCat moment tensors. b-values are compared between settings (v4: 0.77 Axial to 1.33 Puget Lowland) only
where that check shows no regional ML bias, or after correcting it. Reason: attenuation differs between hot ridge and
transform crust and the forearc, and ComCat anchors are sparse offshore.

### D-B042 Zhang & Niksejel catalog in the comparison registry (adopted, 2026-11-23, PI; P-F7)
When the offshore deep-learning catalog of Zhang & Niksejel (SSA 2025 poster; about 24,000 events, Mc 1.5; no paper
or data release yet) is published, add it to the catalog comparison registry (workflow/11_compare, catalogs.yaml) and
compare event counts, completeness and locations offshore. It is the closest competing catalog; we do not claim the
first deep-learning offshore catalog. Listed in paper_sciadv/references_no_doi.md until it has a DOI.

### D-B043 ridges_north damping 200 (adopted, 2026-11-23, PI)
ridges_north is run at damping 200 (answers STATUS open question 1, "100 or 200"); smoothing not yet chosen (tests
at 20). Numbers at smoothing 20, real data: first-iteration condition number 119, last-iteration 33 (Mendocino-Gorda
v2 at its damping 1000: 60 and 19; like-for-like comparison in notes/merge_design.md section 2); weighted RMS 0.060
(0.054 at damping 100, 0.087 at 1000). Cluster spikes with DWS >= 100 (104 P and S): median recovery 16 % (17 % at
damping 100); one spike >= 50 % (21 km, 49.2 N, 128.0 W, DWS 271). None of the five spikes at 12 km between the
northern Blanco and Axial that pass at damping 100 passes at 200, so the D-B038 exception volume stays as written
there (reported from the damping-100 model inside its mask) unless the PI drops it. Set in params
tomodd.regions.ridges_north_v2.damp.

### D-B044 Common starting model from a coarse whole-margin inversion, M0 (adopted, 2026-11-23, PI; P-M2 (a))
One coarse tomoDD inversion of the whole margin (region margin_m0_v2) becomes the starting model of every tile, as
in the California statewide model (Lin et al. 2010; notes/merge_design.md M0). Set-up: events of the five tile boxes
declustered to the two best-recorded events (most picks, then most S picks) per 10 x 10 x 10 km cell (6,943 of
28,816 events, 229,308 picks, 518 stations); interior nodes 0.375 x 0.3 deg (32 x 40) on the shared depth nodes;
start = the D-B005 model averaged over a 31 km box at each node (the node spacing; a single nearest column would
alias at 30 km spacing); forward fixes as the tiles (D-B028 P-1, P-2, P-4, P-6); weights on the dataset scale
(D-B045); damping 200, with a damping-1000 run for the condition-number record; smoothing 20. New tomoDD build for
518 stations (MAXSTA 560, MAXOBS 130; array sizes only, tomoDD_opt/O3nofs_m_fmt and chk_m_fmt). After review of M0,
each tile's damping run and resolution tests are rerun from it (about one day of the machine). Fallback if M0 fails
review: the D-B005 model sampled once (M1).

### D-B045 One damping for all tiles: 200, on one weight scale (adopted, 2026-11-23, PI; P-M1)
The PI asked for one small damping in every tile. Chosen: damping 200, smoothing 20 in every tile and in M0 (the
value already set for ridges_north, D-B043). At 200 against 300 the resolved area is the same (Blanco Vp at 15 km:
19,558 against 19,783 km2) and Oregon's weighted RMS is the same (0.1505 against 0.1511), with more air quakes
(Oregon: 10 against 4 repeats, 24 events of 3,382 lost against 10); Mendocino-Gorda and Washington have not been run
below 500. Condition numbers (first iteration) will be higher than the Mendocino-Gorda benchmark (Blanco 284, Oregon
347, ridges_north 119 at 200) and are recorded, not used to choose.
Weight scale: pick weights were scaled per tile (w = 3 q / max q of the tile's picks), so one pick had a different
weight in each tile (largest q: Blanco and ridges_north 4.941, Mendocino-Gorda 2.788, Oregon and Washington 1.646).
tomoDD-SE adds the smoothing rows before it scales the G columns to unit norm (tomoDD-SE.f lines 1694-2064), so the
weight scale sets the balance of data against smoothing and damping: the same damping number was a different
regularisation in each tile. From now on every build uses one scale, the largest q of the whole dataset (4.941;
region key weights_scale: dataset). Blanco and ridges_north inputs are unchanged by this; Oregon and Washington
weights drop to a third, Mendocino-Gorda to 0.56. The column scaling also divides by the number of data rows, so
the same damping still weighs less in a tile with more data; that part is left as is (it is what the condition
number shows). Applies to M0, the T6 test and the tile reruns from M0.
