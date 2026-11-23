# Track B science framing: what one dataset for sources and structure can show

Note, 2026-11-23 (PI question: "what are we learning about sources and structure in this entire region that we
could not learn before?"). Written for the PI and for whoever writes the tomography and relocation parts of the
paper. It places Track B against published work, states the claim the data can carry, lists the questions that
need sources and structure from the same dataset, and orders the steps after the tomography. Numbers come from
the files named in each paragraph; literature is listed at the end with DOIs (or URLs where no DOI exists).
Nothing here is adopted: the proposals at the end become D-B entries only when the PI accepts them.

## 1. The claim

Every published model of this region was built from one region, one deployment or one wave type, with its own
picks, reference model and regularisation. Track B uses one pick set (ensemble deep-learning picks on 801 land
and ocean-bottom stations, 2010-2016), one catalog (172,342 events), one inversion code (tomoDD-SE), one set of
tiles with overlaps (D-B025, D-B033), one Bayesian ensemble for uncertainty (D-B030, D-B032), and then one
relocation, one magnitude scale and one clustering on top of that model. What is new is that settings can be
compared with the same yardstick: a Vp/Vs difference of 0.05 between the Blanco and the triple junction, or a
2 km difference in the depth of the deepest earthquakes, becomes a measured difference with an interval, not a
difference between two papers.

The claim has two limits. **It is not the first plate-wide local-earthquake tomography.** VanderBeek & Toomey
(2019) inverted Pn travel times from local earthquakes recorded by the Cascadia Initiative for 3D isotropic and
anisotropic P velocity of the uppermost mantle across the whole Juan de Fuca and Gorda plates. Plate-wide P and S
models also exist from teleseismic body waves (Byrnes et al. 2017; Bodmer et al. 2020) and Vs from Rayleigh waves
(Bell et al. 2016). **And it is not a continuous image of the plate.** The resolution tests so far resolve the
volumes where earthquakes are dense, mostly at 15-18 km and, in two tiles, at 3-7 km or 29-34 km (section 4).
Wording that the data support:

> the first local-earthquake Vp and Vs model of the crust and seismogenic volumes across the whole Juan de Fuca
> plate system, from the ridges to the arc, onshore and offshore, with ensemble uncertainty and in the same
> frame as the earthquake locations, magnitudes and fault surfaces.

The paper's tomography placeholder (commit acc7b8a1, now on Overleaf) cites "VanderBeek et al. 2020 G-cubed";
that paper does not exist. The onshore-offshore P and S paper is Bodmer et al. 2020 (G-cubed), and the
local-earthquake Pn paper is VanderBeek & Toomey 2019 (JGR). Both are added to `paper_sciadv/references.bib`
(keys `Bodmer2020`, `VanderBeek2019`).

## 2. What earlier work measured, setting by setting

| Setting | Earlier work, same scale (local earthquakes or OBS) | What Track B adds | Resolution so far (damping 1000) |
|---|---|---|---|
| Blanco transform | Kuna et al. 2019: one OBS deployment, P-only locations, Vp/Vs 1.85 crust and 1.73 mantle. Lange et al. 2026: ML catalog, repeaters, 2D tomography, Vp/Vs > 2 in most of the fault zone. Zhang & Niksejel 2025 (SSA poster): about 19,000 Blanco events | The whole transform in one model with the ridges, the Gorda plate and the margin; 3D | Vp at 15-18 km only (9,500-14,800 km2 at 15 km); Vs 1,550-3,800 km2. Damping 200 and 300 running |
| Gorda plate and Mendocino triple junction | Verdonck & Zandt 1994 (P). Guo et al. 2021 (tomoTD, land and CI OBS): slab-crust Vp/Vs falls from the locked to the transition zone. Guo 2025 (SCEC report 25290, post-2016 data and a nodal array): 1.80-1.86 locked, 1.72-1.76 transition, > 2.0 in the wedge, 1.8-1.95 Franciscan, 1.64-1.78 Klamath, a sharp Vp/Vs contrast across the Mendocino fault. Materna et al. 2018: aseismic transform slip from repeaters. Gong et al. 2018, 2021 | An independent reproduction with different picks and a different code family, tied to the rest of the plate system. Not new on its own | Best tile: Vp at 15 km 32,000 km2, 18 km 52,000 km2, 29 km 46,000 km2 |
| Offshore forearc, Oregon and Washington | Seismicity only: Stone et al. 2018; Morton et al. 2018, 2023 (subducted seamounts, deformed upper plate). Structure from CASIE21 reflection (Carbotte et al. 2024). No local-earthquake Vp/Vs offshore | First local-earthquake Vp and Vs of the offshore forearc where events are (D-B035 kept this region) | Oregon: Vp 10,000-23,000 km2 at 3-7 km, 17,500-26,000 km2 at 15-18 km; Vs 60-80 % of that. Washington running |
| Onshore forearc (Puget Lowland, Olympics, Vancouver Island) | Van Wagoner et al. 2002; Merrill et al. 2020 (tomoDD, Washington forearc, Siletzia); Savard et al. 2018 (tomoDD Vp/Vs, northern Cascadia forearc); Merrill et al. 2022 (Nootka); Delph et al. 2018, 2021 (Vs, Vp/Vs from noise and receiver functions) | Not new; value is the common yardstick with the offshore settings | Washington tile running |
| Ridges (Axial, Juan de Fuca, Explorer) | Cabled-array catalogs (Wilcock et al. 2016; Waldhauser et al. 2020); Baillard et al. 2019 (joint 3D P, S and locations at Axial) | Locations only | Nothing resolved on the lattice; spikes at event clusters recover a median 22 % (damping 100) |
| Plate interior and uppermost mantle | VanderBeek & Toomey 2019 (Pn, local earthquakes, plate-wide); Byrnes et al. 2017; Bodmer et al. 2020; Bell et al. 2016. Jian & Canales 2025 (CRESCENT workshop talk, active-source converted phases, ridge to trench): Vp/Vs mostly 1.70-1.82, crustal porosity 0.2-2 % upper and < 0.2 % lower crust, no significant serpentinization required overall, "the Juan de Fuca plate is dry"; local hydration at a propagator wake and local mantle serpentinization on one line | Overlap only where seismicity reaches the mantle (Gorda plate, possibly Blanco) | Gorda at 29-34 km |

Earlier offshore catalogs: Stone et al. 2018 (CI OBS) and Zhang & Niksejel 2025. The latter is an SSA 2025 poster
(no paper yet): about 24,000 offshore earthquakes, M -2.81 to 5.21, Mc 1.5, OBSTransformer and EQTransformer
picks, events kept only when REAL, GaMMA and PyOcto agree, template matching, double-difference relocation;
about 19,000 on the Blanco, 3,400 in the Gorda plate, 650 in the Juan de Fuca plate and 700 on the margin. We
cannot claim the first deep-learning catalog offshore. Our catalog differs by being amphibious (offshore and
onshore in one catalog, 172,342 events), by the fault surfaces and clustering, and by the tomography and
relocation with uncertainty. When that catalog is released it goes into the comparison registry
(`workflow/11_compare`, branch compare/catalogs).

Published Vp/Vs values between settings differ by 0.05-0.15. Method differences (data, reference model,
regularisation, resolution) are of the same size, so today one cannot say whether the Blanco fault zone is wetter
than the triple junction or the offshore forearc. That comparison is what Track B can make.

## 3. Questions that need sources and structure from the same data

Each question names the test, what would count against the hypothesis, and the step that delivers it.

**Q1. Do fluids or temperature set how faults slip?** On catalog v4 (`data/catalog_v4/clusters_zbz/`, branch
cluster/zbz), the share of background events is 50-61 % at Axial, the Blanco and the Gorda ridge and plate, 58 % at
the triple junction and 68-84 % in the forearc and onshore. Among families of five or more events, swarms are
61-66 % at Axial, Blanco and Gorda and 31 % at the triple junction, but also 66 % in the "onshore and other"
region (41 families, which include the arc volcanoes). The paper draft reads the pattern as the temperature
ordering of Zaliapin & Ben-Zion (2013, 2016). Fluids are the other candidate, and Vp/Vs measures them. Test:
compute Vp, Vs and Vp/Vs averaged in a volume around each fault surface (width from the spike half-widths of each
tile; 8-9 km laterally where Oregon spikes are resolved), in every ensemble member, and compare the swarm share per fault with fault-zone Vp/Vs inside
one setting. The Blanco gives the sharpest test: 33-45 % of events above Mc are in swarms on the West Blanco,
Surveyor and Cascadia Depression strands, fewer on the Blanco Ridge strand. If swarm share follows Vp/Vs at equal
setting, fluids matter beyond temperature; if it does not, temperature alone is enough. Volume averages are the
quantity the data can support: at 15 km, Vs is resolved over 16-26 % of the Vp area at the Blanco and 67-78 %
in Oregon and Mendocino-Gorda. Whether
the Blanco can answer depends on the damping 200 and 300 runs; at 1000, only 15-18 km is resolved, which may lie
below most Blanco seismicity. Steps: tomography ensemble (B2/B3), relocation (B4/B5), fault surfaces and
clustering recomputed.

**Q2. Does one temperature set the depth limit of seismicity across the plate system?** Catalog v4 depths cannot
answer this: 23.5 % of events plot above the seafloor or ground. After relocation in the new model, with station
terms and OBS statics, the deepest well-located events of the ridges, the transforms, the Gorda plate interior,
the triple junction, the megathrust and the slab are in one frame. Compare their depth limits with plate-cooling
and transform thermal models. Published depth limits from different regions differ in reference model by several
kilometres, so they cannot be compared directly. Counter-evidence: depth limits that cut across isotherms by more
than the location uncertainty from the ensemble. Steps: B4/B5 with location uncertainty from the ensemble.

**Q3. Where are the offshore forearc earthquakes relative to the plate interface?** Upper plate, interface or
slab, offshore Oregon and Washington, and whether upper-plate events sit in high-Vp/Vs volumes near the subducted
seamounts of Morton et al. (2018). The interface geometry to test against comes from CASIE21 (Carbotte et al.
2024) and from slab models. This needs the Oregon and Washington tiles, the relocation, and the OBS S statics
(D-B020).

**Q4. What are the fault geometries in depth?** Dips of the 141 surfaces, the Blanco strand pairs about 2 km apart
at equal depth (West Blanco Depression, north wall of the Gorda Depression), the 20 surfaces with no published
counterpart, and the choice of plane at the Gorda source near 41.85 N (NW-WNW or NE). The paper defers all of
these to the relocation. Catalog differential times from picks will improve dips; separating strands 2 km apart
and confirming repeating earthquakes needs waveform cross-correlation within families (proposal P-F5).

**Q5. Do b-values differ between settings?** On v4, b is 0.77 at Axial, 0.84 Endeavour-Explorer, 1.27 Blanco,
1.17 Gorda, 1.06 triple junction, 1.33 Puget Lowland (`regions_zbz.csv`). This means something only if the local
magnitude is unbiased between settings. Attenuation in hot ridge and transform crust differs from the forearc, and
ComCat anchors are sparse offshore. Check ML against Mw by region with the 112 matched moment tensors before
reading b-value differences as source differences. Step: magnitudes recomputed (STATUS step 9).

## 4. What the resolution tests allow, today

From `data/tomo_v1/b2_res_tests/<tile>/summary_tables.md` (damping 1000, noisy checkerboards, both phases):
Mendocino-Gorda resolves Vp over 32,000-52,000 km2 at 15-18 km and 46,000 km2 at 29 km, and 1,300-10,500 km2 at
1-7 km. Oregon resolves 10,000-23,000 km2 at 3-7 km and 17,500-26,000 km2 at 15-18 km. Blanco resolves Vp at
15-18 km only. Ridges resolve nothing on the lattice. In every tile the checkerboard mask is empty at 12, 21 and
25 km while neighbouring depths are resolved; this probably comes from where the vertical checker sign changes
fall relative to the depth nodes, not from coverage, and must be checked before the masks enter the ensemble.
Damping 1000 is too strong for the smaller tiles (real-data condition number 11-14 at Blanco and Oregon against
about 60 for Mendocino-Gorda v2); damping 200 and 300 are running for Blanco and Oregon. The ensemble calibration
(notes/ensemble_calibration.md) has no results yet.

What we cannot claim: structure at the ridges; Vs or Vp/Vs at single nodes outside the volumes resolved for both
phases; rates over years offshore (the array moved every year); anything at the triple junction that Guo's
group, with more data since 2016, has not already shown, except as an independent reproduction.

## 5. Steps after the tomography

1. **Finish the tomography.** Damping per tile (running), synthetic calibration of the ensemble, 100-member
   real-data ensembles per tile (D-B032), comparison and merge in the overlaps.
2. **Relocate all 172,342 events in the final model (B4).** Absolute locations with new station terms and tiers,
   then double-difference relocation with catalog differential times from the picks (GrowClust3D with 3D travel
   time tables, or tomoDD with the velocity fixed per tile). Location uncertainty from relocating in a subset of
   ensemble members (P-F4).
3. **Cross-correlate waveforms within families (P-F5)**: swarms and multiplets of catalog v4 (177 swarms and 822
   multiplets among 1,944 families), then the final DD relocation of those events with the correlation times
   added (B5).
4. **Recompute magnitudes** with the new hypocentral distances (STATUS step 9) and check ML against Mw by region.
5. **Recompute the fault surfaces** (`workflow/07_faults`; dips will move more than strikes) and **re-cluster**
   (nearest-neighbour method as on v4, now per fault, with depth and migration speed of each family).
6. **Join sources and structure**: fault-zone Vp, Vs and Vp/Vs per surface (P-F2) against swarm share,
   migration speed, depth limit and b-value (Q1, Q2, Q5).
7. **Focal mechanisms: future work.** Polarity extraction is paused until the deep-learning models are fixed
   (workflow/09_polarity/STATUS.md); until then the 112 matched ComCat moment tensors are the reference.

## Proposed decisions

* **P-F1 Novelty wording.** Use the wording in section 1. Never "first plate-wide local-earthquake tomography"
  (VanderBeek & Toomey 2019). Fix the placeholder citation on Overleaf: "VanderBeek et al. 2020 G-cubed" ->
  `Bodmer2020`, and add `VanderBeek2019`.
* **P-F2 Fault-zone properties as the main product for interpretation.** Vp, Vs and Vp/Vs averaged in a volume
  around each fault surface, with the posterior over ensemble members; node maps shown only inside the masks.
  Volume width from the spike half-widths of each tile.
* **P-F3 Ridges tiles for relocation only** (answers open question 3 of the STATUS handoff), unless damping 100
  with cluster spikes shows recovery of at least 50 % somewhere.
* **P-F4 Location uncertainty from 10-20 ensemble members**, not all 100 (172,342 events per member).
* **P-F5 Waveform cross-correlation for clustered families only** before the final DD relocation.
* **P-F6 b-value comparisons between settings only after the ML-Mw check by region.**
* **P-F7 Add the Zhang & Niksejel catalog to the comparison registry** when it is released.

## References

In `paper_sciadv/references.bib` (key in brackets; the last seven added 2026-11-23 on branch paper/framing-refs):
Verdonck & Zandt 1994 [Verdonck1994], Van Wagoner et al. 2002 [VanWagoner2002], Wilcock et al. 2016
[Wilcock2016], Stone et al. 2018 [Stone2018], Morton et al. 2018 [Morton2018], Morton et al. 2023 [Morton2023],
Materna et al. 2018 [Materna2018], Gong et al. 2018 [Gong2018], Gong et al. 2021 [Gong2021], Kuna et al. 2019
[Kuna2019], Baillard et al. 2019 [Baillard2019], Delph et al. 2018 [Delph2018], Waldhauser et al. 2020
[Waldhauser2020], Guo et al. 2021 [Guo2021], Merrill et al. 2022 [Merrill2022], Carbotte et al. 2024
[Carbotte2024], Lange et al. 2026 [Lange2026], Zaliapin & Ben-Zion 2013, 2016 [Zaliapin2013, Zaliapin2016];
added: Bell, Ruan & Forsyth 2016 [Bell2016] doi:10.1002/2016JB012990; Byrnes et al. 2017 [Byrnes2017]
doi:10.1002/2017GC006980; Savard, Bostock & Christensen 2018 [Savard2018] doi:10.1029/2017GC007417; VanderBeek &
Toomey 2019 [VanderBeek2019] doi:10.1029/2019JB017707; Bodmer et al. 2020 [Bodmer2020] doi:10.1029/2020GC009316;
Merrill et al. 2020 [Merrill2020] doi:10.1029/2020JB019750; Delph, Thomas & Levander 2021 [Delph2021]
doi:10.1016/j.epsl.2020.116724.

No DOI, so not in `references.bib` (the bibliography takes DOI-resolved entries only; see
`paper_sciadv/references_no_doi.md`):

* Zhang, M., & Niksejel, A. (2025). Deep Learning-driven Seismicity Catalog of the Cascadia Region. SSA Annual
  Meeting, poster 78, 17 April 2025. https://seismosoc.secure-platform.com/a/gallery/rounds/43/details/12405
  (author order as on the SSA page, Zhang presenting; a JpGU 2025 talk, SCG45-27, could not be read).
* Jian, H., & Canales, J. P. (2025). VP/VS Structure of the Juan de Fuca Plate from Converted-Phase Tomography of
  the Ridge-to-Trench Dataset. CRESCENT Offshore Observations Topical Workshop, Seattle, 27 October 2025.
  https://cascadiaquakes.org/wp-content/uploads/2025/11/CRESCENT_OO_Workshop_JIAN_web.pdf
* Guo, H. (2025). Incorporating Abundant Earthquake Data from Densified Permanent Seismic Networks and Temporary
  Nodal Array into Local Earthquake Tomography to Improve 3-D Seismic Velocity Structure in the Mendocino Triple
  Junction Region. SCEC Final Report, project 25290, University of Wisconsin-Madison.
  https://files.scec.org/s3fs-public/reports/2025/25290_report.pdf
