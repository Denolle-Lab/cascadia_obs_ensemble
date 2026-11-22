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
