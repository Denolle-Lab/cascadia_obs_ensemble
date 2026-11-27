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

### D-B046 M0 = margin_m0_v2 d200_s480; every tile restarts from it (adopted, 2026-11-23, PI; P-M11)
M0 is the whole-margin run at damping 200 and smoothing 480/480/240, the smoothest of the scan 20-1920 whose weighted
RMS stays within 1.5 % of the best (0.2012 against 0.1984; notes/merge_design.md, M0 paragraph). Each tile
(mendocino_gorda_c, blanco, oregon, washington, ridges_north; regions <tile>_m0) restarts from it at damping 200 on
the dataset weight scale, scans smoothing 20-480 (params merge.tile_smooth_scan) and takes the largest smoothing
whose weighted RMS is within 1.5 % of the scan's best (merge.smooth_rule_tol), then runs its resolution tests at that
smoothing. Implementation choices made while building it (Claude, 2026-11-23; tell the PI):
* Start = the D-B005 model at the tile's own nodes (nearest column, as <tile>_v2) multiplied by (1 + M0's change
  from its own start), not M0's velocities as they are. M0 started from a 31 km box average, so taking its values
  directly would replace the fine near-surface structure that the OBS S statics and the P-4 corrections were derived
  in; M0's change carries what it learned (the long wavelengths).
* P-4 station corrections reused from <tile>_v2. Recomputed in the new start they differ by +0.10 s (P) and -0.15 s
  (S) median, up to 1 s, at ridges_north: that is M0's change, which P-4 would turn into per-station constants.
  P-4 is meant to correct tomoDD's forward error, which the start model barely changes.
* Vp/Vs floor 1.65 on the start (params start_vpvs_min; Vs lowered): M0 has isolated nodes with Vp/Vs down to 1.49
  where Vp and Vs moved apart (48 nodes < 1.6); in the ridges_north start the floor changes 1.0 % of the nodes.
Iteration: tomoDD solves locations and velocities together in every run (12 iterations, rays retraced each time);
there is no separate outer tomography-location loop. After the merge, B4 relocates all 172,340 events in the merged
model; a second tomography pass (M7) only if T4/T7 show seams or misfit the tiles cannot explain.

### D-B047 Velocity models are versioned and exported cloud-native (adopted, 2026-11-23, PI request)
PI: "version control all of the velocity model ... reusable for relocation and dissemination as a standard earth
model, maybe on EarthScope EMC"; "being cloud native is good"; "upload to Zenodo and add a tutorial on how to read
it". Every model that is adopted or handed on (M0, tiles, merged, ensemble) gets a semantic version, written by
b2_model_export.py (pixi environment `models`: Python 3.12, zarr 3, xarray, netCDF4) as
* Zarr v3 with consolidated metadata (canonical, cloud-native) and netCDF-4 Classic (compressed) holding the same
  Dataset: Vp, Vs, VpVs, starting model, perturbations in %, DWS (0 = no ray); interior tomoDD nodes, depth km below
  sea level positive down; EMC header attributes (checked against the EMC inspector's required list; authors, paper
  reference and DOIs stay "TO FILL" until release) and CRESCENT CVM variable names;
* NonLinLoc SLOW_LEN grids on the D-B005 1 km Lambert grid for relocation (--nll; model inside its volume, D-B005
  outside and in water/air cells);
* registered in data/tomo_v1/models/registry.csv (append-only, sha256 of every file, git commit of the code) with a
  provenance JSON; versions are immutable; netCDF < 5 MB also kept in git; stores in /wd1/mdenolle_data/tomo_v1/models.
Distribution: one bundle `velocity_models.zip` (README, tutorial read_velocity_model.py, registry, all versions) in the
private Zenodo draft 23084158, replaced at each new version (the draft is at 99 of 100 files). Public release
(EMC submission, CRESCENT CVM, a DOI) only after the peer-reviewed paper and only when the PI says so.
v0.1.0 = M0 (margin_m0_v2 d200_s480).

### D-B048 Relative-relocation method tested before B5: tomoDD (fixed velocity) vs GrowClust3D on the Blanco swarms (adopted, 2026-11-23, PI)
PI doubts that GrowClust is more accurate than HypoDD; HypoDD does not scale to this catalog and uses a 1D layered
model. In the 3D model the alternatives are GrowClust3D (Trugman et al. 2023; 3D travel-time tables, hierarchical
cluster merging, L1 misfit, bootstrap uncertainty; Trugman & Shearer 2017 found relative accuracy comparable to
HypoDD in synthetic tests, with fewer pathological solutions) and tomoDD-SE in relocation-only mode (HypoDD-type
double-difference least squares in the 3D model, velocity frozen; 10,000-13,000 events per tile). Test, once B4
(NonLinLoc of all events in v0.3.0) has landed and before the B5 GrowClust3D run:
- Events: the Blanco swarms, starting with the June 2015 West Blanco swarm (755 events over 64 days, family 1803),
  from their B4 locations.
- Same model (v0.3.0), same stations, same differential times for both codes (catalog differential times, plus
  waveform cross-correlation times where D-B040 gives them).
- Compare: relative positions (event-pair distances), cluster spread and orientation, depth distribution, shift of
  the cluster centroid from B4, bootstrap uncertainty; one synthetic case with a known geometry (planar swarm)
  through both codes.
- Outcome: if GrowClust3D matches tomoDD within the uncertainties, B5 uses GrowClust3D for the whole catalog;
  otherwise report where and by how much they differ before choosing.

### D-B049 Resolution-gated merge, for lightly smoothed tiles only (adopted, 2026-11-24, PI; amended the same day)
Context: the literature comparison of depths (data/tomo_v1/depth_literature/depth_literature.html) found B4 too deep at
Axial, Endeavour, Explorer and the JdF ridge (the tomography moved them away from published depths) and on the
shallow Blanco, from unresolved tile structure entering the merged model v0.3.0. Rule: per phase, V = R V_tiles +
(1 - R) V_ref, with R = max over the covering tiles of the resolution window without its floor (1 on the tile's
checkerboard-resolved nodes, cosine taper to 0 over taper_km), so unresolved nodes fall back to the whole-margin
reference (M0). As first implemented it sent 98.7 % (P) and 99.5 % (S) of covered nodes to M0, because checkerboards
at smoothing 480 (mendocino_gorda_c, oregon, washington) barely recover (median amplitude 0.005-0.13 against 0.5)
although those tiles' changes are small and smooth (< 1 % rms Vp) and are what moved Gorda depths toward the
literature. Amended (PI): only tiles merged at smoothing W1 <= tomodd.merge.gate_max_smooth = 100 are gated (blanco,
s20); s480 tiles count fully in their box. Dry run: 16.0 % of covered nodes on the reference (unresolved Blanco).
Variables tile_weight_p / tile_weight_s (R) and attributes gated_tiles, gate_max_smooth record it.

### D-B050 ridges_north left out of the merge (adopted, 2026-11-24, PI)
D-B038 approved the ridges tile for relocation only. Its velocities (smoothing 60, Vp/Vs 1.35-1.40 under the Axial
caldera, 1.47 at Endeavour) put Axial events 2-2.5 km too deep against Waldhauser et al. (2020), while v4-NLL3D
agreed. tomodd.merge.exclude_tiles: [ridges_north]; the ridges sit in M0 (dry run: Axial Vp/Vs 1.70-1.71 at 3-5 km).

### D-B051 Vs reverted where Vp/Vs < 1.6 or S coverage is low (adopted, 2026-11-24, PI; replaces the drafted floor 1.5)
Where the merged Vp/Vs < tomodd.merge.vpvs_min = 1.6 or the S-ray DWS of the covering tiles < dws_s_min = 10, Vs is set to
the reference Vs; if Vp/Vs is still below 1.6 (Vp changed), Vs = Vp / 1.6. Flag variable vs_reverted (0 unchanged,
1 reverted, 2 reverted and capped; int8, netCDF-4 Classic has no unsigned type) and node counts in attributes. v0.3.0
had 3,228 nodes below 1.65 and 168 below 1.5 (minimum 1.275), inherited from poorly resolved tile nodes. Dry run:
minimum 1.60; the 87,565 low-coverage reverts change Vs by a median 0.01 % (no S rays there), the 258 capped nodes by
a median 1.5 %.

### D-B052 Near-station travel times; no whole-grid fallback (adopted, 2026-11-24, PI)
PR #35 review finding 14: the Richardson combination T = T0.5 - k (T1 - T0.5) is wrong next to the station, where
the 1 km grid's error comes from the source-cell start-up and does not scale like h^p (101 of 1,392 v0.3.0 grids had
negative times next to the station, down to -0.16 s; 2,581 B4 events within 2 km of a station, Axial's cabled array
among them). Fix in b4_time_grids.py: T0.5 within 2 km of the station, linear taper to the combination at 5 km, times
asserted >= 0. Beyond 5 km the start-up error is carried along the rays (median 0.014 s P / 0.024 s S, up to 0.08 /
0.17 s at a few 7D OBS); it is recorded per grid in the manifest. A whole-grid T0.5 fallback for the outliers (median
+ 3 x 1.4826 MAD: P 0.044, S 0.075 s) was tested and rejected: against co-located stations it fixed NED.CC, J17D.7D,
BS58.XG, CL60.XG and HYS11.OO but worsened FN08C.7D, G30B.7D and J25C.7D, and T0.5 is itself late with distance. The
remaining offset is close to a constant shift per station, which the B4 station terms absorb (params
b4.grid_fallback.enabled: false).

### D-B053 B5 relative relocation with tomoDD-SE in relocation-only mode (adopted, 2026-11-24, PI)
D-B048 test (branch tomo/b5-blanco-test, notes/b5_blanco_test.md): on three Blanco swarms and three synthetic planar
swarms, with catalog differential times only, GrowClust3D was not more accurate than tomoDD: neither improved the B4
locations or resolved swarm depth (a 5 km tall plane spread over 6-29 km), GrowClust3D made good starts worse and
changed with small input changes (90 m shift, rounded start), and its bootstrap spread was 4-7 times tomoDD's.
tomoDD relocation-only (JOINT = 0, differential times, velocity frozen) was stable. B5 uses it, with each cluster's
centroid put back on its B4 location (tomoDD's centroid is free: 0.6 km down in f0399) and events kept at least 50 m
from the frame centre (tomoDD segfaults otherwise). Waveform cross-correlation times (D-B040) remain to be tested.

### D-B054 tomoDD-SE FORMAT fixes approved; patched version frozen for future use (adopted, 2026-11-24, PI)
The relocation-only path stops at a malformed FORMAT string in lsfitHFDD_lsqr_lm5.f (line ~504, a missing comma
before a string literal), the same kind as D-B034 (tomoDD-SE.f / tomoDD-SE_syn.f line ~2324). PI approved the fix (a
comma only) and asked to keep this tomoDD version. Frozen, private on /wd1 only (WARF licence: no posting, never in
git or on Zenodo): /wd1/mdenolle_data/tomo_v1/src/tomoDD-SE_patched_2026-11-24 (pristine USTC 2020 source with every
malformed FORMAT string fixed by a comma, PATCH.diff, build_variants.sh, README), binaries for every size class in
/wd1/mdenolle_data/tomo_v1/bin/frozen_2026-11-24 (main, syn, chk), archive in /wd1/mdenolle_data/tomo_v1/archive.
Licence caveat: the WARF licence forbids modification; the request to Haijiang Zhang for permission is still open.
Freeze audit (11-24): six commas in total. D-B034 (tomoDD-SE.f 2318, 2320; tomoDD-SE_syn.f 2283, 2285), the relocation
fix (lsfitHFDD_lsqr_lm5.f 504), and one more found by the full scan, getinpSPDR.f 215 ('": DAMP= "f7.1' -> '": DAMP=
",f7.1'; no output change; PI acknowledgement needed). Three string-string joins without a comma are left unchanged
(cluster1.f 182-183, tomoDD-SE.f 2651-2652, tomoDD-SE_syn.f 2598-2599; accepted by the gfortran 8.5 and 15.2 runtimes).
51 test runs (joint, relocation-only, synthetics, every size class, pixi 15.2 and system 8.5 runtimes): all result
files byte-identical to the binaries in use. Corrections to the frozen README (read-only): it cites D-B054 for the B5
relocation-only mode (that is D-B053), and says the permission request was sent; no record of the email was found.
PI (2026-11-24): fix 6 (getinpSPDR.f 215) is kept. PI: do not write to Haijiang Zhang; no permission request has been
or will be sent, so the frozen README's statement that one was sent is wrong (the README is read-only and archived;
this entry is the correction). The WARF licence terms (no modification, no public posting) are unchanged: the patched
version stays private on /wd1, never in git or on Zenodo.

### D-B055 No event above the ground or seafloor; fix the computation, never move events (adopted, 2026-11-24, PI)
PI: an event above the solid surface is the product of a wrong computation or workflow; a catalog with any such
event is not published, and moving events to the surface is rejected (a first version of this entry did that in
b4_catalog.py and the atlas; both reverted the same day). b4_catalog.py computes surface_km (GMT earth_relief 01m,
gridline, bilinear) at every final epicentre and above_surface on the true depth, never changes a depth, and writes
the counts to compare.md and "publishable" to the manifest. Counts on 11-24: B4 v0.3.0 1,079 of 157,270 located
events above (median 0.03 km, max 0.41 km; Track A 594), tier U (v4 position kept) 4,080 of 15,070, catalog v4
40,553 of 172,340 (23.5 %, median 2.1 km above; 71 % of Axial). Root causes under investigation (workflow
above-surface-root-cause, 11-24): v4 GENIE/GraphDD depth datum, velocity model top and search domain; NLLoc
LOCTOPO_SURFACE read by node lookup (velmod.c get_surface_z) instead of the bilinear surface; why tier U events are not
located. The atlas shows catalog positions as they are and flags the events above the surface.

### D-B056 NLLoc topography floor that NLLoc's nearest-node lookup cannot get above (adopted 2026-11-24, implementation of D-B055)
Root cause (above-surface root-cause workflow, 11-24, confirmed by an independent verifier): NLLoc reads
LOCTOPO_SURFACE by nearest node (velmod.c get_surface_z, pix_shift 0.5), while the surface is defined by bilinear
interpolation; reproducing the lookup explains 1,079 of 1,079 B4 v0.3.0 and 594 of 594 Track A located events above
the surface (plus, in B4, a flattened-vs-true term of at most 1.5 m, 4 events). Fix, no code change: each node of a
15 s grid holds the minimum of the bilinear earth_relief 01m surface over the area NLLoc assigns to it, minus 1 m,
Earth-flattened (h' = Re ln(1 + h/Re), never above h, so one file serves both frames):
/wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_eft.bin, built by workflow/08_relocation/a2_make_topo_surface_cons.py
(reloc/track-a 85cc3dae; md5 identical to the tested files), used by nlloc_template.in from v0.3.1 iter0 on. Tests:
0 of 500 sampled, 0 of 1,078 B4 and 0 of 594 Track A affected events above; 197 of 200 random controls unchanged
(3 moved one oct-tree cell); +50 MB per NLLoc process. Not a root-cause fix for everything: for about 4 % of B4 events
(29.5 % of JdF ridge, 18.9 % Endeavour, 7 % Axial) the travel-time likelihood peaks 0.3-1.7 km above the surface
(verifier's no-topography reruns), so their depth is set by the floor, not the data. That upward pull is the open
computation error (candidates: stations placed 50 m below the surface by a1_build_model.py line 253, near-surface
velocities, Vp/Vs, station terms); under investigation.

### D-B057 Paper hypocentres from the final NLLoc relocation; v4 depths not published (adopted, 2026-11-24, PI)
Catalog v4/v4.1 depths come from GENIE location and GraphDD relocation without any surface (round-1 root-cause report,
data/tomo_v1/above_surface/above_surface_round1.html: 23.5 % of events above the ground or seafloor, 71 % at Axial;
no datum change fixes it), so under D-B055 they cannot be published. PI: the paper's hypocentres are the NLLoc 3D
relocation in the final Track B model, once the open workflow fixes are in (round 2: upward pull, station reach, 80 km
floor, tier U without v4 hypocentres). Catalog v4 stays the association and pick catalog (event IDs, picks, times,
magnitudes). Asking Ian for a corrected GENIE/GraphDD run is not needed for the paper.

### D-B058 Catalog versions v5.x (adopted, 2026-11-24, PI)
Catalog v5.0 = the catalog v4 events and picks (v4.1 event list) with hypocentres from the B4 NonLinLoc relocation in
merged model v0.3.1 (run b4_iter1_v0.3.1, D-B056 topography floor). It is an interim version: v5.1 = the NLLoc
relocation rerun after the round-2 workflow fixes (upward pull, station reach, 80 km floor, tier U; D-B057), v5.2 =
the B5 tomoDD relative relocation (D-B053) of v5.1. Each version goes into the private Zenodo draft 23084158 as one
file, catalog_v5.zip (all versions, replaced at each version, as velocity_models.zip), and onto the atlas. Magnitudes
in v5.0 are the v4.1 Route A values (computed with v4 hypocentres); recomputation with the new hypocentres is STATUS
step 9. Note: the pick sets are also numbered ver5/ver6; "catalog v5.x" always means this event catalog.

### D-B059 Tier U keeps its v4 hypocentre in v5.0, flagged (adopted, 2026-11-24, PI; exception to D-B055 for v5.0)
Events NonLinLoc cannot locate (tier U, about 15,000, mostly <= 5 picks) keep their v4 hypocentre in v5.0 with
loc_source = v4, tier U and the above_surface flag (about 4,000 of them are above the ground or seafloor). The PI chose
this over blank hypocentres or dropping the rows, knowing it is an exception to D-B055; v5.0 is therefore flagged
"publishable: false" by b4_catalog.py, and stays in the private Zenodo draft and on the atlas with the flag shown.
Round 2 (D-B057) is to remove the copy of v4 hypocentres for v5.1.

### D-B060 B5 uses catalog differential times first; waveform cross-correlation times later (adopted, 2026-11-24, PI)
The whole-catalog B5 run (tomoDD-SE relocation-only, D-B053; frozen binaries D-B054; model frozen at the B4 model)
uses catalog differential times (tomodd.dt rules) now, as in the D-B048 test, where they moved the B4 locations
little. Waveform cross-correlation times (D-B040) are a second pass once waveforms are fetched. Driver on branch
tomo/b5 (worktree ~/wt_b5_full), notes/b5_relocation.md.

### D-B061 B5 weights S differential times like P (S weight 1.0) (adopted, 2026-11-24, PI; P-W1)
Root cause of the S misfit growth in the B5 run b5_v0.3.1 (notes/b5_s_mismatch.md): not a tomoDD vs NLLoc forward
difference (the two agree on differential times to a MAD of 2-5 ms) but the weights: dt.ct S weights carried the
D-B008 factor 1/4 (sigma S = 2 x P) and B5 fitted P at S's cost. Weighting test (notes/b5_weighting_test.md; 26
clusters in every region, 18 variants, station cross-validation and synthetics): S weight equal to P of the same
pick class gives held-out dt rms -6 % (P) and -11 % (S) against the 1/4 weighting (both phases better in 24 of 26
clusters, every region) and ~10 % smaller synthetic relative errors; 1.5 and 2.0 are equal within 1 %; 1.0 is the
rule. Implementation: params b5.tomodd.s_weight = 1.0 (WTCTS = s_weight / 0.25 with the present dt.ct weights).
D-B008 is unchanged for absolute locations (B4).

### D-B062 B5 keeps the production MAD re-weighting (adopted, 2026-11-24, PI; P-W2)
WRCT 8 then 6 x MAD, WDCT 10 km in sets 3-4 (params b5.tomodd.sets). Without the cuts held-out dt rms is 2 % worse
at every S weight; the looser 12/10 cut is 0.6 % worse (notes/b5_weighting_test.md).

### D-B063 S weights from the measured S/P residual ratio are not used (adopted, 2026-11-24, PI; P-W3)
Inverse-variance weights from the S/P MAD ratio of B4 dt residuals (1.02-1.69 by region) were worse than equal weights
on held-out data (+2.5 %) and synthetics (+1.8 % horizontal): the measured ratio includes B4 location error, which
moves S times about Vp/Vs times more than P.

### D-B064 B5 acceptance by station cross-validation (adopted, 2026-11-24, PI; P-W4; replaces the proposed P-S2)
Every B5 run is checked by station cross-validation on sample clusters of every region (b5_wtest.py accept RUN):
held-out dt rms of the adopted weighting must not exceed that of the D-B008 weighting (S weight 0.25) in either phase,
by region (median over the region's clusters). Misfit against B4 in-sample is not a criterion (B4 fitted the same
picks). Events above the surface are reported, never moved (D-B055). B5 for the catalog (v5.2) runs on v5.1.

### D-B065 Two checks before the v5.1 NLLoc relocation (adopted, 2026-11-24, PI)
Round 2 (data/tomo_v1/above_surface/above_surface_round2.html) found that the template's LOCGAU2 0.02 0.05 2.0
(model error growing with travel time) inside EDT_OT_WT acts as a pull toward short travel times (7 of 50 perfect-data
events above the surface with it, 1 without), and that the S arrivals of floor-held events are late relative to P, more
so with distance. Before v5.1: (1) rerun floor-held events, controls and the Hirao/Lange/Axial reference events with
LOCGAU2 off and with a constant model error, and choose the setting by those scores; (2) check the S picks of a few
pulled events on the waveforms (wrong picks or associations vs a Vp/Vs bias).

### D-B066 Five-pick events located with a 5-phase minimum, published as tier D (adopted, 2026-11-24, PI)
The 12,108 events with exactly 5 picks (the v4 minimum) were skipped by LOCMETH's 6-phase minimum and kept v4 positions.
From v5.1 the template's minimum is 5: round 2 located 12,044 of 12,089 tested, none above the surface, nearly all
tier D (one degree of freedom, sigma_z ~13 km, horizontal error ~27 km); events with 6 or more phases are unchanged.

### D-B067 Deep pass below 80 km with ak135 outside the tomography box (adopted, 2026-11-24, PI)
The model, time grids and LOCGRID stop at z' = 80 km, so well-recorded intraslab events at 66-107 km are rejected or
pinned at the floor. From v5.1, wide grids extend to z' ~121 km and events rejected at the bottom face or located at
z' >= 78 km get a deep second pass (round 2: 16 of 17 located at 78-107 km, controls within 0.35 km). Below 80 km and
outside the tomography box the velocities come from ak135, blended under the deepest model layer; deep-pass locations
are flagged. Events seen only by distant stations, which the same trigger would select, are flagged separately.

### D-B068 Solid surface from GMRT where available, earth_relief 01m elsewhere (adopted, 2026-11-24, PI)
Under 725 v5.0 events the GMRT multibeam seafloor is lower than earth_relief 01m (median 41 m, up to 0.5 km). From
v5.1 one merged surface (GMRT patches where available, earth_relief 01m elsewhere) defines the NLLoc topography floor
(conservative node minimum as in D-B056), the D-B055 above-surface test, and the water/rock boundary of the model.

### D-B069 NLLoc model error: constant 0.2 s, no LOCGAU2 (adopted, 2026-11-24, PI; result of the D-B065 test)
Test on 2,397 events of b4_iter1_v0.3.1 (/wd1/mdenolle_data/tomo_v1/above_surface/check_misfit/, README and
out/s06_table.md): with LOCGAU 0.2 0.0 and no LOCGAU2 (variant C02), Blanco minus Lange 2026 goes from +1.80 to +1.08 km
(bootstrap change -0.73 [-0.99, -0.48]; West Blanco +4.00 -> +1.74; Cascadia Depression -1.32 -> -1.98), Mount St Helens
is unchanged (+0.45 -> +0.47), 28 % of floor-held events leave the floor, Axial stays at 28-31 % within 2 km of the
seafloor. Costs: tier A-B 1,094 -> 1,045 in the samples; sigma_z at Axial 1.25 -> 1.63 km and at MSH 0.97 -> 1.29 km
(production used 0.05 s on short paths); 42 events (all tier C/D) more than 10 km deeper. NLLoc's EDT_OT_WT RMS is
weighted by the EDT probabilities, so compare fits with the unweighted RMS. v5.1 template: LOCGAU 0.2 0.0, LOCGAU2
removed; station terms recomputed with it in iter0.

### D-B070 Test a slower offshore S before v5.1 (adopted, 2026-11-24, PI; from the D-B065 waveform check)
Waveform check (/wd1/mdenolle_data/tomo_v1/above_surface/check_spicks/report.html): the S picks of floor-held events are
on the S onsets (bad picks explain 8 %, at most 22 %, of the S excess); confirmed S-P is a median 0.51 s longer than the
model's; implied path Vp/Vs offshore beyond 20 km 1.79-1.83 against 1.73-1.74 in the model; 84 % of the checked S nodes
are reverted to the start model (D-B051); the excess is mostly constant per event (+0.2 s), pointing to slow S near the
source (young ridge crust) or to epicentres on one-sided networks. Before v5.1: rebuild S grids for test stations with
offshore S about 4 % slower and with a high Vp/Vs shallow crust, and score as in D-B065; if it works, the fix goes into
the model as a new version before the v5.1 relocation.

### D-B071 Flag ANSS explosions and quarry blasts (adopted, 2026-11-24, PI)
326 of 1,700 onshore floor-held events (19 %, against 3.1 % of other onshore events) match ANSS explosions or quarry
blasts, for which a surface depth is right. v5.1 sets event_flag for events matched to an ANSS event of those types
(time and distance match), next to the Texada quarry flag; floor_limited is computed as before.

Also 2026-11-24 (PI): a separate check of the UW short-period stations TDL, ELK, LO2, SOS, whose P and S residuals drop
together by 0.1-0.3 s from November 2013 (timing or filter delay), before the v5.1 station terms.

### D-B072 Re-time the drifting analog picks from short reads around each pick; v5.1 uses them (adopted, 2026-11-24, PI)
UW timing check (/wd1/mdenolle_data/tomo_v1/checks/uw_timing/report.html): from 2013-10-22 14:48:30 UTC, picks on 34
UW analog short-period stations (+KOS to 2013-11-18; UO IRO, HUO, FRIS to ~2015-01) are early by up to 0.52 s (UO
~0.79 s) by the end of each UTC day: the channels run at 99.99939 Hz labelled 100 Hz, and elep_picker.py run_detection
reads a whole pnwstore day, filters, resample(100) and merge(fill_value='interpolate'), so a pick's time is the day
trace's start plus samples/100. PI: first verify on our own pnwstore data; then re-time every affected pick by reading
the waveform with ObsPy only around the pick (a short read keeps each MiniSEED record's header time; a whole-day read
accumulates the lag), not by a rate model. Corrected picks go into a new pick file with a flag column, used by the
v5.1 relocation (station terms re-estimated). Other products (pick tables on Zenodo, Track B tomography, the picker
code) are not changed for now.

### D-B073 Shallow oceanic crust: Vp/Vs >= 2.1 in the top 2.5 km below the seafloor offshore (adopted, 2026-11-24, under the PI's "go continuously 1 to 5"; PI may revise before v5.1 is published)
D-B070 test (/wd1/mdenolle_data/tomo_v1/above_surface/check_slow_s/, out/s16_runs_all_table.md; 2,397 events; constant
0.2 s model error; "w" = station terms re-estimated in the new model, as in production): V4 (offshore rock cells whose
centre lies within 2.5 km below the seafloor: Vs = min(Vs, Vp / 2.1)) raises Axial A-B events within 2 km of the seafloor
from 31 % to 86 % (92 % without re-estimated terms; median 2.47 -> 1.56 km below the seafloor: meets the >= 85 % target),
lowers floor-held events from 72 % to 46 % of the floor sample (13.1 % -> 8.5 % of all samples), raises tier A-B from
1,045 to 1,106; controls: median depth change 0, p90 1.15 km, 17 of 300 move > 2 km, RMS unchanged or lower; Mount St
Helens unchanged. Cost: Blanco minus Lange 2026 +1.08 -> +2.14 km if Lange's depths are below sea level (West Blanco
+1.74 -> +3.88, Blanco Ridge +2.06 -> +3.19, Cascadia Depression -1.98 -> -0.69); Lange's datum is not stated: if his
depths are below the seafloor (~3 km of water), V4 is the closest variant. At Blanco V4 also lowers floor-held events
(3.9 -> 1.0 %) and the unweighted RMS (0.219 -> 0.203 s). V2 (Vp/Vs 1.95) is milder (Axial 56 %, Blanco +1.53 km);
V1/V3 (S 4 % slower everywhere offshore) push Blanco ~5 km deeper: rejected. Implementation: params b4.shallow_crust
{vpvs 2.1, thickness_km 2.5} in b4_time_grids.py (field-level rule on the merged surface, D-B068; coast-scan offshore
mask of the test), checked on a full field build (v51_build/sc_test: P unchanged; every changed S cell within 2.5 km
below the merged seafloor at exactly the set Vp/Vs). The fields are model v0.3.1 plus this rule.

### D-B074 v5.1 wide-grid choices: no distance cap; near-seafloor S interpolation accepted (adopted, 2026-11-24, under the PI's "go continuously 1 to 5")
Wide verifier (/wd1/mdenolle_data/tomo_v1/v51_build/verify_wide/): wide-served S residuals trend with distance
(-0.41 s per 100 km; -0.8 to -1.1 s beyond 300 km), the same for well-recorded events, so from the model or the picks,
not the grids (wide minus fine and wide minus a whole-lattice reference show no trend). v5.1 keeps every arrival (no
distance cap): the EDT likelihood is robust to outlying arrivals, events located by distant stations only are flagged
far_only, and capping would leave many offshore events unlocated; the trend is a known limit to examine (far S picks,
upper-mantle S) before the next model version. Near-seafloor S at OBS stations: wide minus fine at real event positions
within 2 km of the seafloor has p95 0.046-0.061 s (max 0.23 s), from interpolating on 2 km nodes; far below the 0.2 s
model error of D-B069: accepted (1 km vertical wide grids would cost 246 GB).

### D-B075 Re-time every pick whose day trace drifts from its record headers, if the co-located test confirms the headers (adopted, 2026-11-24, under the PI's "go continuously 1 to 5"; implements D-B072)
Re-timing (/wd1/mdenolle_data/tomo_v1/checks/pick_retiming/, RESUME.md): verified on pnwstore (exact sample matches;
TDL 2014-07-13: 0.010 s at 00:30, 0.521 s at 23:30; data gaps reset the count). The listed epoch (D-B072) is re-timed:
87,301 picks (AIC onsets: catalog - onset -0.310 -> -0.010 s). The header scan found the same mechanism beyond the list:
UW analog EHZ channels before 2013-10-22 (opposite sign, typically -0.01 to -0.07 s/day; daily maximum error median
0.03 s, up to ~0.24 s in 2010), more UW analog stations (RER RVC LMW STW CPW GHW GMW MPO OSR PGO WPO SSO; OBC OCP OOW OSD
OTR +0.05-0.07 s/day), CC SEP SUG NED EHZ (+0.02 to +0.4 s/day), UO FRIS and HUO already in 2010 (~1.1 s/day), and
small sporadic cases. The PI's rule (time each pick from a short read, because a whole-day read accumulates the lag)
applies to every such channel-day, so v5.1 uses every correction (rule: a station-day is corrected when its record
headers and the sample count differ by >= 5 ms anywhere in the day; each pick gets its own record's header time),
PROVIDED the co-located test (tables/validation_colocated_slope.csv: pick differences against GPS-timed digital stations
LON, STD.CC, B202 regressed on the correction; slope ~ 0 after re-timing) confirms that the headers are right. If it
does not, v5.1 uses only the corrections of the listed epoch (epoch_class == 'listed' in corrections.csv). Not
re-timable: NC/BK (not in pnwstore), channels resampled from another rate (flag rate_not_100), off-grid picks (kept).
Result (2026-11-24 23:40, finish.sh): co-located slope test (tables/validation_colocated_slope.csv, pooled with one
intercept per pair-phase) listed: catalog -0.992 +- 0.007 -> re-timed +0.008 +- 0.006 (n 16,859); found: -1.000 +- 0.107
-> -0.000 +- 0.114 (n 6,293): the headers are right for both, so v5.1 uses ALL corrections. 218,809 picks corrected on
75 stations (listed 86,711, found 189,931 incl. in_fill; corrections.csv.gz, column applied); AIC onsets after the
onset: catalog - onset -0.264 -> -0.010 s (MAD 0.104 -> 0.020, n 673); 2,284,818 uncorrected rows byte-identical to
v4.1. Not re-timed, kept unchanged and flagged: 45,879 picks (off_grid 23,331, mostly OBS Z5/7D with small drift, and CC
SEP/SUG EHZ from 2014-01-18 on, whose picks are no longer on the 100 Hz grid; rate_not_100 22,541; error 7). v5.1 pick
file: /wd1/mdenolle_data/tomo_v1/checks/pick_retiming/Cascadia_picks_catalog_v4.1t.csv (md5 762f18e8f181eaaaa247d367c2d2c302; a2_make_obs.py
reads it: 2,503,381 picks of 172,342 events in 1,206 tiles; spot check HUO.UO event 41 P 0.772 s later).

### D-B076 Model v0.3.2 = v0.3.1 + the D-B073 rule, with no rock above the solid surface (adopted, 2026-11-25, PI)
PI 05:30: "Register v0.3.1 + D-B073 shallow-crust rule as v0.3.2"; PI 05:34: "We needed a surface fix as there are rock
velocity above the sea floor" (v0.3.1.nc: 22,500 offshore nodes above the merged seafloor with rock values, Vp median 4.78
km/s, 726 with P rays; 6,444 onshore above the ground). v0.3.2 (candidate; branch tomo/ensemble-v032 2609b7be,
b2_model_v032.py): same grid; int8 `medium` (0 rock, 1 water: Vp 1.5, Vs/VpVs NaN; 2 air: NaN) from the merged surface
(D-B068) at each node; D-B073 at offshore rock nodes within 2.5 km below the seafloor (8,739 nodes, VpVs 2.100);
Vp_tomography/Vs_tomography = v0.3.1; NLL grids md5-identical to the v5.1 production fields (field rule on the 1 km
lattice). nc md5 0487b11d; bundle velocity_models.zip md5 044f5c81 (to session -18 for the private draft). tomoDD users:
water/air nodes must not enter a tomoDD MOD (node vs field test and -18's S1: water nodes add up to 0.8 s S error);
rock-continued values match NLLoc to a few ms, as NLLoc's own time grids (rock_continued_1km); the surface acts as a floor
(D-B081 for tomoDD).

### D-B077 Re-invert v0.3.1 with the re-timed picks (v0.4.0); floor in the inversion after its results; ensemble on the final configuration (adopted, 2026-11-25, PI)
PI 05:36: "re-run v0.3.1 with the correct re-timed picks": v0.4.0 = the v0.3.1 configuration (B1/B1b selection,
weights, M0 d200 s480, four merged tiles, D-B049/D-B050/D-B051) with only the pick times from
Cascadia_picks_catalog_v4.1t.csv (D-B072/D-B075), no surface treatment; branch tomo/v040-retimed. PI 06:12: decide the
D-B081 floor in the inversion (v0.4.1, inversion binaries copied from the D-B081 freeze) after the v0.4.0 results. The
D-B030/D-B032 ensemble (PI 05:30 "start the ensembling") runs on the final configuration; its driver is built and tested
now (branch tomo/ensemble-v032) with dataset, M0 and tiles as parameters.

### D-B078 The B5 relative-relocation code is chosen on waveform cross-correlation data (adopted, 2026-11-25, PI)
PI 06:07 asked whether to reconsider GrowClust; 06:12 chose the comparison. Catalog differential times did not improve the
locations in any test (D-B048; b5_v0.3.1: cluster thickness ratio 1.00). Test on 3-4 clusters (Axial, Blanco, one
well-recorded onshore cluster) with D-B040 cross-correlation times: tomoDD relocation-only with the D-B081 floor vs
GrowClust3D on the NonLinLoc grids (no medium problem) vs NLL-SSST-coherence (Lomax & Savvaidis 2022, JGR 127,
e2021JB023190, doi:10.1029/2021JB023190); scored on synthetic planes, held-out stations and above-surface counts.

### D-B079 Catalog v6.0 = the relocation in the re-inverted model, with NonLinLoc's SSST upgrades; amends D-B058 (adopted, 2026-11-25, PI)
PI 06:12: "New version V6.0 and consider the NLL with the recent upgrades from 2022 with SSST corrections". The NonLinLoc
relocation in the re-inverted model (v0.4.x) is catalog v6.0, not v5.2; v5.2 stays reserved for B5 on v5.1 (D-B058). v6.0
considers NLL-SSST (source-specific station terms, smooth in source position) and the coherence stage of Lomax &
Savvaidis (2022) in place of the D-B006 static station terms. Not to be confused with the pick dataset "ver6".

### D-B080 Catalog bundle carries the pick file it was located with; atlas tests the second catalog against its own surface (adopted, 2026-11-25, session -18; implements D-B058 for v5.1; PI may revise)
* catalog_v5.zip: a version located with picks that are not in the Zenodo record carries them in its folder: v5.1 has
  Cascadia_picks_catalog_v4.1t.csv.gz (re-timed picks, D-B072/D-B075, md5 762f18e8 uncompressed, checked at build) and
  the re-timing corrections, README and report; the draft holds 100 files, so they go inside catalog_v5.zip.
* catalog_version.py pins the solid surface per version (surface_source: v5.0 earth_relief_01m, v5.1 merged), so a
  rebuild of v5.0 still reproduces its published CSV (md5 23781b52).
* Atlas (atlas/model-layer 3f661608): events without a hypocentre (loc_source none) are drawn in the v4 view only;
  the above-surface flag of the second catalog uses the catalog's surface_km (the atlas terrain has gaps and differs by
  ~30 m), and its depths are stored exactly (float32), so the atlas reports the catalog's own count (v5.1: 0).
* b4_acceptance.py bug fixed: above_surface.astype(bool) turned empty values (no hypocentre) into True, so v5.1's
  acceptance.md listed 623 tier U events "at the v4 position" above the surface; it now reports 0, and 623 without a
  hypocentre. Located counts were unaffected.

### D-B081 A local floor inside tomoDD for B5: NonLinLoc's floor, an update stops at it (adopted, 2026-11-25, PI: "I approve that you change the tomoDD source keeping track of all the changes we had to make")
Cause found by the S1 test (tomo/b5 b5_s1_fieldmod.py, run b5_v5.1_s1/diag/s1; 4 v5.1 clusters): tomoDD's calculated
differential times match the NonLinLoc v5.1 grids to a few ms with rock velocities above the surface (the node MOD, or
the 1 km field rock-continued, which is what NonLinLoc's own grids use at water and air nodes: FLUID_NODES =
rock_continued_1km), and get worse with water or air in tomoDD's model (JdF S MAD 15 -> 88 ms; the other session's
builder found the same on 2 km nodes). So the B5 events above the surface come from the missing floor (tomoDD's only
limit was one flat Air_dep per cluster, a median 1.6 km above the local surface), not from the velocity medium.
* Source change (new private freeze `/wd1/mdenolle_data/tomo_v1/src/tomoDD-SE_patched_2026-11-25`, binaries
  `bin/frozen_2026-11-25`, archive `archive/tomoDD-SE_patched_2026-11-25.tar.gz`; WARF: never in git or Zenodo): in
  `tomoDD-SE.f`, when a file `surface.dat` is in the run directory, an update that would put an event above the floor
  depth of the nearest floor node stops 1 m below it and is logged (`>>>Floor:`); the event stays in the relocation.
  Without `surface.dat` the program is unchanged (byte-identical results on 3 clusters; synthetic binaries same md5).
  Every change ever made to the package (the six FORMAT commas of D-B034/D-B054 and the four floor additions) is
  listed with reasons and checks in `CHANGES.md` of the freeze; `PATCH.diff` rebuilds the patched source from the
  pristine one.
* Floor = NonLinLoc's floor of the v5.1 runs (true frame, `reloc_v5/topo/topo_15s_cons_merged.bin`: D-B068 merged
  surface, exact 15 s cell minimum - 1 m), same nearest node, in tomoDD's rescaled frame (b5_relocate.write_floor);
  Air_dep moved to -1000 km (tomoDD tests it on the raw update and drops the event).
* Collect removes each cluster's centroid shift (D-B053), which can lift events: the floor is written offset by that
  shift (up to 3 runs per cluster until it holds in the final frame); where the runs have no fixed point (most of the
  cluster on the floor), the vertical shift is limited to the largest that keeps every event at or below the floor
  (cluster-level datum, recorded as anchor_z_limited_km; no event moved on its own).
* New catalog columns: b5_floor_hit (the update hit the floor in the final run), b5_floor_limited (ends within 5 m of
  the floor), as v5.1's floor_limited.
* Test (4 v5.1 clusters, catalog dt, node MOD v0.3.1; tomo/b5 3f910802): above the surface 76 -> 0, events dropped as
  air quakes 922 -> 0, relocated 12,291 -> 13,202, dt misfit in the NonLinLoc grids unchanged by the floor. Axial
  settled in 3 runs (469 events end on the floor); JdF ridge c000032 needed the limited anchor (44 m, 59 on the floor).
* Scope: B5 relocation-only runs. Whether the floor is used in the tomography inversion waits for v0.4.0 (D-B077).
  B5 production waits for the D-B078 comparison, in which tomoDD + this floor is one of the three codes.

### D-B082 Model v0.4.0 (v0.3.1 re-inverted with the re-timed picks) is the base model (adopted, 2026-11-25 07:59, PI)
v0.4.0 (D-B077; branch tomo/v040-retimed; registered intermediate; report /wd1/mdenolle_data/tomo_v1/v040_retimed/report/):
the v0.3.1 configuration with only the pick times of Cascadia_picks_catalog_v4.1t.csv (65,562 of 754,710 B1b picks re-timed,
59 stations; 33,976 by > 0.02 s, 11,975 by > 0.3 s). At the re-timed land stations P rms falls 0.246 -> 0.222 s (M0), 0.222
-> 0.187 (Oregon), 0.225 -> 0.192 (Washington), and their time-of-day residual slopes go from -0.15..-0.31 s/day to about
+0.02..+0.08. Resolved shallow onshore Vp (1-9 km) changes by 0.34-0.77 % rms against a numerical noise floor of 0.01 %
(real); resolved nodes at 15-29 km (mostly the Blanco tile at smoothing 20) change by 0.86-1.0 %, inside their noise floor
(0.76-1.06 %; a 1e-6 s perturbation also moves 5.7 % of Blanco's events by > 1 km). Smoothing rule choices unchanged (480).
The B1 pick QC stays as in v0.3.1 (812 of 854,465 picks would change status with the re-timed times; not redone).

### D-B083 v0.4.1 = v0.4.0 with the D-B081 surface floor in the tomography inversion (adopted, 2026-11-25 07:59, PI; follows D-B077)
M0 and the four merged tiles are re-inverted from the v0.4.0 inputs with tomoDD's event updates floored at the local solid
surface (D-B081: surface.dat from the merged surface D-B068, an update stops 1 m under it, the event is flagged, not removed),
using inversion binaries built from the D-B081 freeze after a byte-identity check without surface.dat. Velocities above the
surface stay rock-continued during the inversion (water/air nodes harm tomoDD: v0.3.2 node-vs-field test, -18's S1). The
product follows D-B076 (medium, D-B073 node rule, Vp/Vs_tomography, NLL grids with the field rule). The ensemble's real
phase (D-B077) runs on the v0.4.1 configuration.

### D-B090 Correct station clock drift in the picks for catalog v6.0, with a reusable pick-timing toolkit (adopted, 2026-11-25 08:44, PI)
PI: "we might as well do this correction: Correcting the MH07/MG07 picks for v6.0, and we will probably have to do that for
other west coast picks that we had in another project, so generate a product that can be reused by human and agents".
The CC-dt work for D-B078 (/wd1/mdenolle_data/tomo_v1/cc_dt/qc/clock_epochs.csv, tomo/cc-dt 2d033ec6) found two free-running
clocks in spring 2016: MH07.XD drifting +8.35 +- 0.28 ms/day from 2016-02-06 (+- 2.4 d) to about +0.80 s at its data restart
2016-05-12T05:09:09, and MG07.XD -4.40 +- 0.59 ms/day from 2016-02-20 (+- 9.3 d) to about -0.34 s at 2016-05-08T07:57:28
(record headers carry no timing-quality information). Header errors cannot be fixed by the D-B072/D-B075 re-timing. v6.0 uses a
pick file with these drifts removed (picks corrected by the fitted clock model, flagged and kept, never dropped), after a
scan of every station for similar clock problems, each candidate confirmed by a person before it is applied. The detection,
modelling, correction, re-timing (D-B072/D-B075) and reporting steps become one documented toolkit with a command line, an
HTML review report and an agent instruction file, so the same procedure can be run on other pick datasets. Numbering: this
session uses D-B090..D-B099 from here on; session -18 continues from D-B084.

### D-B091 Run the Route A magnitude station-term inversion to convergence and recompute the v4.1 and v5.1 magnitudes (adopted, 2026-11-25 10:24, PI)
Found in the v5.1 magnitude fix round (branch analysis/v51-mag cd0334d6, workflow/05_magnitude/v51_phase19_convergence.py):
phase19's LSQR (damp 1e-3) stops at iter_lim 3000 without converging (istop 7) in v4.1 (the paper's magnitudes) and v5.1.
Run to convergence (4,863 iterations, istop 2), final ML moves by Blanco -0.68, JdF ridge/plate -0.51, Gorda, Mendocino and
offshore forearc -0.28, Explorer -0.08, Axial -0.04, onshore +0.01, with the residual norm almost unchanged (268.37 -> 267.86):
the picks only weakly tie the offshore level to the onshore network. The converged solution agrees better with both outside
references: ComCat ML anchor gaps Blanco +0.59 -> +0.10, Gorda +0.33 -> +0.02, Mendocino +0.54 -> +0.26, anchor std 0.230 ->
0.176; offshore ML - tensor Mw (Mw 3-4.5) +0.26 -> -0.01. Both the v4.1 (paper) and v5.1 magnitudes are recomputed with a
converged solver that checks istop, together with a map of how strongly each offshore deployment is tied to the onshore
network; Mc, b and every paper figure or number that uses offshore ML are updated (manuscript: bullets only).

### D-B092 Prepare the catalog v6.0 relocation now: v0.4.1 travel-time grids and NonLinLoc SSST (adopted, 2026-11-25 10:24, PI; implements D-B079)
Build the travel-time grids of model v0.4.1 (D-B083; field rule, merged surface, wide and deep grids as for v5.1) and implement
and test NLL-SSST (source-specific station terms, Lomax & Savvaidis 2022) against the D-B014/D-B069 static station terms on a
few clusters. Picks: Cascadia_picks_catalog_v4.1tc.csv (D-B090; MH07/MG07 corrected). The full relocation runs after the PI's
review of the other clock-candidate epochs; the location-review flags are applied when the catalog is built.

### D-B093 Catalog v6.0 station terms: static terms plus the NLL-coherence stage, with waveform cross-correlation for all families; use the whole machine (adopted, 2026-11-25 13:12, PI)
SSST test on the four D-B078 sets (v5.1 grids, /wd1/mdenolle_data/reloc_v6/ssst_test/g51/; fit of the CC differential times,
median per pair, ms): static / best SSST step / static + coherence = Axial 12.3 / 13.5 / 11.2; Mount St Helens 17.2 / 16.9 /
10.6; Blanco 24.3 / 23.2 / 24.3; JdF 39.0 / 35.7 (16-8 km) / not applicable (too few compact pdfs). SSST pushed Mount St
Helens deeper against Hirao et al. 2025 (+0.91 -> +1.16 to +1.59 km) and degraded Axial at every step; static + coherence
kept the depths (+0.95 km) and brought repeater pairs to 74 m (Axial, MSH). No event above the surface in any arm. v6.0
therefore keeps the static station terms of v5.1 (D-B014/D-B069) and adds the coherence stage (NonLinLoc 7.1.07,
LOCPOSTERIOR OCT_TREE stacking; Lomax & Savvaidis 2022) for events with waveform-similar neighbours, which needs CC
differential times for every clustered family (D-B040), not only the test sets. PI: "use all resources you can while our
jobs are running - we are the only one using the machine today": the per-session process caps of today are lifted, subject
to keeping free memory (grid chain raised from 12 to 36 processes at 13:13).

### D-B094 Real-data ensemble on v0.4.1 with velocity uncertainties inflated by the calibration, rounded up (adopted, 2026-11-25 15:37, PI; amends D-B030, D-B077)
Synthetic calibration (ensemble_v0.3.2/synth/coverage.md, 102 members merged): hypocentres calibrated in every tile and
coordinate (68 % coverage 0.70-0.73, 95 % 0.95-0.96); Vp/Vs calibrated (0.75 / 0.96); Vp and Vs too narrow on resolved
nodes (Vp 0.565 / 0.849, rms z 1.28; Vs 0.581 / 0.863, rms z 1.22), as the f0 diagnostic predicted (each member misses the
same 3-5 % of v0.3.1's structure at Blanco and Mendocino; the 3 % prior does not cover it). The D-B030 all-items rule failed
(exit 2). PI: run the real phase on v0.4.1 (D-B083) and calibrate by inflation, "but round up because that's still a bit too
narrow": the Vp and Vs posterior spreads (std and percentile half-widths about the median) are multiplied by factors taken
from the synthetic members as the larger of the factors that bring the 68 % and the 95 % coverage of the resolved nodes to
nominal, rounded up to the next 0.05; hypocentres and Vp/Vs are not inflated. The values and their derivation are recorded
in the ensemble note and the product attributes. The v0.3.2-configuration calibration is accepted for v0.4.1 (P-E11).

### D-B095 Catalog v6.0 station terms: NLL-SSST + coherence, stopping distance set by a confirmatory check (adopted, 2026-11-25 15:51, PI; amends D-B093)
Independent-cluster test (branch reloc/v6-termtest 07b45252; /wd1/mdenolle_data/reloc_v6/termtest/; decision rule written before
scoring): on three clusters sharing no event with the D-B078 sets (Puget 1,000, Mendocino 2015 sequence 600, western Blanco 2012
swarm 1,200), SSST stopped at 2 km + coherence (B) beat static + coherence (A) on held-out stations (session -18's fold scorer,
no leak) in all three, P and S: MAD P 85 -> 59, 67 -> 56, 149 -> 107 ms; S 141 -> 90, 118 -> 88, 264 -> 164 ms (95 % bootstrap
intervals exclude 0); CC fit better in 3/3; depth closer to Lange 2026 at western Blanco (median |dz| 2.11 -> 1.52 km), ties
elsewhere; 0 events above the surface in every arm. Agrees in direction with -18's D-B078 four-set result. v6.0 uses NLL-SSST
(Loc2ssst, CharDist schedule 9999 -> 16 -> 8 -> 4 [-> 2] km) followed by the coherence stage, instead of D-B093's static terms. A
post-hoc hint (4 km stop better than 2 km at Puget and Mendocino; 16 km at western Blanco) is checked on two further independent
clusters with the same protocol before the full run; the stopping distance with the better held-out score is used.

### D-B096 Catalog v6.0 SSST schedule stops at 4 km (9999 -> 16 -> 8 -> 4 km), then coherence (adopted, 2026-11-25, by the D-B095 rule written before scoring)
Confirmatory test (branch reloc/v6-termtest 8489f8cc; /wd1/mdenolle_data/reloc_v6/termtest/score_stop/; rule written into
RESUME.md at 15:55 before any run): two further independent clusters (Strait of Georgia 700 events; 2014 M6.8 Gorda sequence 800
events). Held-out MAD P + S with coherence, mean of the two: 16 km 81.0, 8 km 78.4, 4 km 76.8, 2 km 80.2 ms; paired bootstrap 8 km
- 4 km +1.6 [0.3, 2.7] ms excludes 0, so 4 km is kept. Georgia lowest at 4 km on P and S; Gorda flat between 8 and 4 km. 0 events
above the surface in every arm; depths indistinguishable between stops. The coherence stage helped where it relocates many events
(Puget, Georgia: P -15 ms), was neutral where it relocates few (Gorda, western Blanco) and hurt at the Mendocino 2015 sequence;
v6.0 QC reports the held-out effect of coherence by region. Catalog scale: Loc2ssst's LSSTATIONS line is limited to 1,000
characters, so station lists are split across instances.

### D-B097 Run catalog v6.0 now; the Mount St Helens station issues and the other clock candidates wait for a later version (adopted, 2026-11-25 16:58, PI)
PI: "Can we just keep going and leave the Mount Saint Helens stations for later?". The full v6.0 relocation (model v0.4.1, D-B083;
picks Cascadia_picks_catalog_v4.1tc.csv with only the MH07.XD/MG07.XD clock corrections, D-B090; D-B096 SSST schedule + coherence)
starts as soon as the v0.4.1 wide grids and the family CC times are complete. Deferred to a later version (v6.1): the weekly
artificial signal on the analog UW channels near Mount St Helens (HSR, EDM, STD.UW; likely false deep events, waveform review P1),
the 129 other clock-candidate epochs (review_candidates.csv) and the location-review flags; v6.0 carries the automatic review
indicators as columns but applies no reviewer flag it does not have.

### D-B098 Catalog v6.0: no coherence relocation in the Mendocino region (adopted, 2026-11-25 20:35, PI; amends D-B095, D-B096)
PI answer "Skip at Mendocino" to: the held-out tests show the coherence stage helps at Axial (P -10 ms), Mount St Helens (-21),
Puget (-7) and Georgia (-15), does nothing at Blanco, JdF and Gorda, and hurts at the 2015 Mendocino sequence (P +8 ms, S +20 ms).
In v6.0, compact events (Len3 <= 5 km) whose catalog v5.1 region is "Mendocino" (a box on the v4 epicentre, 39.5-40.5 N,
128.1-123.5 W; 24,644 events in v5.1) are not relocated by the coherence stage and keep their SSST 4 km (it4) location. They can
still be stacking partners of events outside the region. Code: reloc/v6-prep 5596e49e (b4_v6_reloc.py cmd_coh, override
V6_COH_SKIP_REGIONS; catalog column coherence_skipped_region; coh/coherence_skipped_region.csv). Provisional: it rests on one
Mendocino test cluster; the acceptance report states it, and v6.1 can revisit it with a second Mendocino test.

### D-B099 Model v0.4.2 = v0.4.1 without self-paired differential times, with the patched tomoDD; ensemble and synthetic calibration rerun before registration (adopted, 2026-11-26 06:27, PI; amends D-B083, D-B094)
Diagnosis of the four failed real-ensemble members (2026-11-25 20:55-22:19; ensemble_v0.4.1/diag_failures/):
(1) tomoDD exit 139 in about 3 % of members (real r002, r049; synthetic s034, s060, s089): sph2car in tomoDD's
transform_r_gfortran.c takes sqrt(d2^2 - z^2) with z rounded to float, which goes negative for ray nodes within a few metres
of the vertical axis through the tile centre; the NaN node then indexes ixloc(int(NaN)) in vel3. Deterministic per member; no
silent cases. Patch: the algebraically identical stable form (d3 sin del; atan2 in car2sph) only where the old expression
leaves its domain; same bits elsewhere (r002, r049 first iterations byte-identical, then all 12 iterations run).
(2) Self-pairs in dt.ct: b2_tomodd_inputs.diff_times dropped column 0 of the cKDTree query as "self"; with two events at
identical hypocentres the twin can come first, so an event is paired with itself. tomoDD reads C1 == C2 as an absolute time
with travel time 0 s (weight x10), which pulls the event 10-31 km away. 8 events: washington 6 (28514, 40700, 53148, 55170,
80094, 94266), mendocino_gorda_c 1 (156844), blanco 1 (81324). Present in the deterministic v0.4.1, all ensemble members and
the v0.3.2 synthetic calibration; r017 failed on it (linearised residual -30 s overflowing tomoDD.res). Without them: Vs at
29 km under the Vancouver Island cluster 3.51 -> 4.06 km/s, 1 Vp and 11 Vs nodes change by > 0.05 km/s.
PI chose: fix the builder (tomo/ensemble-fix 53a3b868) and adopt the patched tomoDD (new freeze outside git); re-invert the
three affected tiles as v0.4.2; rerun the ensemble on v0.4.2; then redo the synthetic calibration on the fixed configuration
and register the ensemble only once the inflation factors are confirmed (updated if they move by more than 0.05). Catalog
v6.0 stays on v0.4.1 (local difference; stated in its README). B5 code (b5_common.diff_times) has the same bug (session -18
told 2026-11-25 22:20).

### D-B100 Register catalog v6.0 as built; its Axial depth bias from the SSST passes is stated and fixed in v6.1 (adopted, 2026-11-26 08:29, PI)
v6.0 (catalog_b4_v6.0.csv md5 ae77ad3c66aa6fbaa48528660a1c09ec, finished 2026-11-26 08:05; acceptance
reloc_v6/catalog/acceptance_v6.0.html): 172,182 located, 0 above the surface, floor-limited 3.3 % (v5.1 5.1 %), D tier 32,935
(v5.1 43,032), median rms 0.076 s (v5.1 0.122). It fails the Axial depth target: tier A-B within 2 km of the seafloor 66.1 %
(v5.1 81.0 %, target 85 %). Cause, by pass: it0 (no terms) 82.7 %, it1 76.3, it2 72.7, it3 68.8, it4 65.9 % (median below
seafloor 1.63 -> 1.85 km); the SSST terms trade off with depth. Median depth shift it4 - it0 (tiers A-C): Axial +0.22 km,
Puget/Georgia +0.29, Mendocino +0.17, Blanco +0.15, Gorda -0.22, forearc -0.15. The D-B095/D-B096 tests scored held-out CC
differential times (relative fit), which cannot see a depth shift common to a cluster. PI: register v6.0 as is in the private
Zenodo draft with this bias stated in its README; for v6.1, test station-term recipes on absolute depth as well (Axial vs
Waldhauser et al. 2020, Blanco vs Lange et al. 2026, Mount St Helens vs Hirao et al. 2025), together with the D-B097 deferrals.
Numbering: this session takes D-B100..D-B109 (D-B084..D-B089 stay reserved for session -18).

### D-B101 Clean up the private Zenodo draft: slim velocity models with v0.4.2, fault surfaces v1 instead of v0, amplitude and PickBlue parts packed into zips (adopted, 2026-11-26 08:58, PI)
Draft 23084158 was full (100/100 files, 6.7 GB). PI: "consider cleaning the zenodo especially for files that won't be reused
(since our reproducibility statements with the reports can reproduce them)"; chose all three steps:
(1) velocity_models.zip (672 MB; 958 MB of its content were the v0.3.2 NonLinLoc grids, rebuilt from the model by
b4_time_grids) replaced by the registered model netCDFs v0.1.0 to v0.4.2 with provenance, registry and a README on rebuilding
the grids (no zarr copies, no grids); (2) the eight fault surfaces v0 files removed (superseded by fault surfaces v1, one zip);
(3) the ver5/ver6 amplitude parquet parts and the PickBlue pick parts packed into one stored zip per set (data unchanged:
recomputing them takes days of waveform processing). Picks, catalogs and READMEs stay. Every replacement is uploaded and
md5-checked before the files it replaces are deleted; local copies are kept. The draft stays private (no publish).

### D-B084 B5 relative relocation = NLL-SSST-coherence (adopted, 2026-11-25, PI, in session -18; from the D-B078 comparison)
Fair held-out test on 4 sets (axial, blanco, msh, jdf; every code started from NonLinLoc relocated without the held-out
stations; tomo/b5 notes/b5_code_comparison.md, data/tomo_v1/b5/cmp/report.md): NLL-SSST-coherence has the smallest
tails and is best where data are sparse (Blanco, offshore); tomoDD + the D-B081 floor has the sharpest cluster cores
(Axial, Mount St Helens, JdF); GrowClust3D is out (largest tails, events above the surface). The first round's tomoDD
lead came from starting at the v5.1 positions, which used the held-out stations. The PI chose NLL-SSST-coherence. Its
schedule was then fixed on independent clusters by the other session: SSST 9999 -> 16 -> 8 -> 4 km and coherence
(D-B095, D-B096; Mendocino without coherence, D-B098), which is also the v6.0 recipe, so B5 and v6.0 share one code.
tomoDD + floor (D-B081) stays available as a cross-check for dense clusters; it gives no catalog column. Open for the
PI: whether v5.2 (B5 on v5.1, D-B058/D-B079) is still wanted now that v6.0 applies the same code in the re-inverted
model.

### D-B102 Each catalog carries magnitudes computed on its own hypocentres: v5.1 takes the converged v5.1-hypocentre table (adopted, 2026-11-26 13:33, PI; amends D-B091)
Independent review of PR #49 (catalog v6.0 bundle and converged v5.x magnitudes): the v5.1 bundle took M from the converged
v4.1 table (v4 hypocentres), while v6.0 takes it from the converged v5.1 table (v5.1 hypocentres), so 126,808 of 130,769
events differed between the two bundles (median |dM| 0.023, p95 0.26, max 2.14). PI: v5.1 switches to the v5.1-hypocentre
converged magnitudes (as v6.0); catalog_v5.zip is rebuilt once more (session -18) together with the six minor review fixes
(checks in magnitudes: catalog mode, 900003 label, docstring, v6.0 tier pointer, final zip md5 in summaries, upload note).
PR #49 merged as is (f7a87b11); the fixes come as a small follow-up PR.

### D-B103 Catalog v6.1 station terms: NLL-SSST stopped at 16 km, then coherence; near-surface Vp/Vs to be corrected in the model next (adopted, 2026-11-26 14:58, PI; amends D-B096)
Pre-registered depth test (rule written 13:09, before any run; reloc/v61-depth-test b5bc25eb; report
/wd1/mdenolle_data/reloc_v61/depth_test/report_depth_test.html): Axial (Waldhauser 2020), Mount St Helens (Hirao 2025),
western Blanco and Gorda Depression (Lange 2026); ten arms (no terms, static, SSST 16 km, SSST 4 km, null-space-free SSST,
each with and without coherence); out if held-out CC MAD is worse than the best by > 5 % (95 % paired bootstrap), then the
smallest mean |depth offset| wins. Only ssst16_coh passed the held-out gate everywhere (ssst4_coh, the v6.0 recipe, fails at
the Gorda Depression: P +23 ms [+8, +39], S +49 ms [+17, +79]); depth score D = 2.02 km (v6.0 recipe 2.35). Root cause of the
v6.0 depth drift: the station-term/hypocentre null space (Pujol 1988) - the term common to an event's S arrivals is nearly
the same unknown as depth (r = -0.94); Loc2ssst absorbs the S-P misfit of the first pass (Axial caldera S terms -40 to -62 ms
relative to P) and events drift deeper with each pass (Axial +0.22, Puget +0.50, Blanco +0.44 km medians). Removing that
common term outright over-corrects (21 % of Axial events on the floor), so part of it is a real near-surface Vp/Vs error of
the model. PI: v6.1 uses SSST 9999 -> 16 km + coherence on the whole margin (D-B098 unchanged); next model task: correct
near-surface Vp/Vs where the terms need large common S corrections (Axial about -60 ms, Puget about -150 ms), so a later
version can meet the Axial depth target (the 16 km stop gives about 73 % within 2 km at catalog scale; target 85 %).

### D-B104 Catalog v6.1 is located in model v0.4.2; paper figures and the atlas show the latest model only, in colourblind-safe colour maps; the atlas Mc map follows each catalog (adopted, 2026-11-26 15:30, PI)
PI: "We need to show the latest model only. The color map should be very colorblind-friendly but very beautiful ... we'll
submit this to Science Advances ... Make the atlas only show the latest MC or adjust the MC for each of the catalogs that are
shown. For the v6.1 we'll use the latest v042." (1) v6.1 travel-time grids are built from v0.4.2 (amends the v6.1 plan's open
choice); recipe D-B103. (2) Paper model figures show v0.4.2 only (published models may still appear as comparison); velocity
colour maps change from matplotlib "seismic" (the earlier atlas choice) to perceptually uniform, colourblind-safe scientific
colour maps (Crameri), checked under simulated colour-vision deficiency; the atlas model layer takes the same maps.
(3) Mc grids are computed on each catalog's own hypocentres and magnitudes (v5.1, v6.0; D-B102), so the atlas Mc layer follows
the catalog shown.

### D-B105 Magnitudes on the catalog's own hypocentres are computed for v6.1 (v6.0 keeps the v5.1 values); Mc maps exclude non-earthquakes for every catalog (adopted, 2026-11-26 15:42, PI; amends D-B102)
The Mc maps on each catalog's own hypocentres (D-B104; analysis/mc-maps-per-catalog ba9fa538) showed that v6.0 carries the
converged magnitudes of the v5.1 hypocentres, so D-B102's "own hypocentres" holds for v4.1 and v5.1 only. PI: compute the
converged Route A magnitudes (exact phase19 solver) on the v6.1 hypocentres as part of the v6.1 build; v6.0 keeps the v5.1
values, as its README states. The v4.1 Mc maps (and so the paper's Mc figures) kept blasts and explosions; removing them
changes 368 cells by >= 0.15 (range -1.0 to +0.7) and drops 170 cells in v6.0. PI: Mc maps exclude events flagged as not
earthquakes (event_flag) for v4.1, v5.1 and v6.0; the paper's Mc figures and the numbers they report are regenerated.

### D-B106 The paper clustering also leaves out the events flagged as not earthquakes (adopted, 2026-11-26 16:01, PI; extends D-B105)
The Mc maps leave out the 3,298 events flagged as quarry blasts or explosions (D-B105), and the v6.0 clustering already left
them out; the paper's v4.1 clustering (Table S5, Text S6, Figs. 5-6) still counted them. PI: same rule everywhere - rerun the
v4.1 clustering without the flagged events, regenerate Table S5 and Figs. 5-6, and list the changed numbers as manuscript
bullets, in the same PR as the Mc change (#59).

### D-B107 Paper framing: fault-zone structure and seismicity style at the plate boundaries; no full-size velocity-model maps (adopted, 2026-11-26 16:05, PI)
PI: "Figures in the paper will focus on the velocity model that is close to the fault ... just a fault zone structure and
seismicity type of paper. We don't need full-size velocity model plots. We will focus on analyzing the difference in the
seismic structure and seismicity style at the various plate boundaries." The model figure candidates of D-B104 (M1 overview
maps, M2 margin-normal sections) are not paper figures; the gallery and the atlas stay for exploration. Paper model figures
show velocity structure only near the faults where it is resolved, set against the seismicity style (clustering, b and Mc,
depth extent, fault surfaces) of each plate boundary; a feasibility check of which boundaries the model resolves comes first.
