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
