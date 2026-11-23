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

