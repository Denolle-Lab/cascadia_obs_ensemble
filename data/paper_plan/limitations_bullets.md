# Limitations of the study: bullets for the Science Advances manuscript

- Written 2026-11-27 09:18 PST (time from `date`), branch paper/limitations. Draft for the PI.
- PI, 2026-11-27: "All caveats can be written in terms of limitations in the paper, keep next updates for v6.2."
  The fixes are listed in `workflow/10_tomography/notes/v6_2_backlog.md`; nothing is changed now.
- Bullets only (PI rule of 2026-11-23); the PI writes the text on Overleaf. The manuscript and `paper_sciadv/` are not edited.
- Each bullet: the limitation and its size, its effect on the results ("Effect"), and what the paper does ("Paper").
  The three responses first marked "proposed" were adopted by the PI on 2026-11-27 09:36 (D-B111).
- Paper catalog: v6.1 (registered 2026-11-27, D-B085), model v0.4.2, ensemble v0.4.2+ens.100.
- Numbers marked "(v6.0)" come from analyses run on catalog v6.0 as a stand-in (plate-boundary plan, clustering).
  D-B109 reruns them on v6.1, so they will move.
- Each bullet ends with an HTML comment naming its sources; the keys are listed at the end of this file.

## Catalog and locations

- Axial depth bias: 72.7 % of the 12,584 tier A-B Axial events lie within 2 km of the seafloor (target 85 %;
  Waldhauser et al. 2020: 90 %), a systematic deep offset of about 0.2-0.4 km. Effect: Axial depths and depth extent
  are too deep by that amount; epicentres and the 2-D clustering are not affected. Paper: offset stated; fix in v6.2.
  <!-- src: ACC61 (Axial table); DEC D-B085, D-B103 -->
- Station terms trade with depth: the source-specific terms absorb a common S-P misfit, so clusters drift deeper with
  each pass (v6.0 tests: Puget +0.50, Blanco +0.44, Axial +0.22 km). v6.1 stops at 16 km and is still 0.22 km deeper than
  v5.1 at Puget and Mendocino. Effect: depths there may be 0.2-0.5 km too deep. Paper: recipe from a pre-registered test.
  <!-- src: DEPTH (section 1 tile table); ACC61 (change from v5.1 by region, dz medians); DEC D-B103 -->
- Mount St Helens depths: v6.1 is 0.85 km deeper than Hirao et al. (2025) (median, 2,369 paired events; 1.04 km for
  tiers A-B). Every station-term recipe gave +1.1 to +1.5 km, so the offset comes from the model and the datums.
  Effect: depth extent under the volcano about 1 km too deep. Paper: offset stated.
  <!-- src: ACC61 (Hirao table); DEPTH (scores and "offsets common to all arms") -->
- Blanco depths: against Lange et al. (2026), v6.1 is 1.01 km deeper over 11,298 paired events (90th percentile +11.2 km),
  3.17 km for tiers A-B and 4.85 km in the Gorda Depression (503 events). Effect: Blanco depth extent and seismogenic
  bottom carry km-scale errors. Paper: depth metrics use depth_quality ok events only; offsets stated.
  <!-- src: ACC61 (Lange absolute table); PBPLAN (section 4, depth-good events) -->
- Depth quality: 24,967 depths are unconstrained (ridges; Blanco outside the 2012-2013 OBS deployment), 9,231 biased
  deep (Gorda, nearest station > 15 km), 7,076 at the floor (35 % of JdF events). Effect: few depths where stations are
  sparse. Paper: depth metrics use tier A-C, depth_quality ok events off the floor.
  <!-- src: README61 (counts, depth_quality column); ACC61 (floor-limited table); PBPLAN (section 4 depth rule) -->
- Weakly constrained events: 33,412 events are tier D, among them 12,036 located with five phases (one degree of
  freedom); 345 have no hypocentre. Effect: tier D positions, depths above all, are poor. Paper: analyses use tiers A-C
  (135,954 located earthquakes in the plan, v6.0).
  <!-- src: ACC61 (tiers, located events); README61 (five_phase); DEC D-B066; PBPLAN (section 1) -->
- Mendocino without coherence relocation: 10,500 compact Mendocino events keep their station-term location because the
  coherence stage worsened the held-out fit there (S +13.1 ms in v6.1); the rule rests on one test cluster. Elsewhere
  coherence cut the held-out misfit by 7-29 ms. Effect: Mendocino relative locations are less sharp. Paper: stated.
  <!-- src: ACC61 (held-out coherence tables, R3 columns); DEC D-B098 -->
- Relative fit against v6.0: the held-out fit of waveform differential times is more than 5 % worse than v6.0 at Puget
  (P +17.9 %, S +15.2 %), Mendocino (S +7.2 %) and Gorda (S +7.7 %). Effect: relative locations there are less sharp than
  in v6.0. v6.1 changed both the term stop and the model; the cause is not separated. Paper: stated; test in v6.2.
  <!-- src: ACC61 (comparison with reloc_v6/heldout); HANDOFF 2026-11-27 07:00 (PR #60 review) -->
- No waveform double-difference catalog: the only relative relocation is the coherence stage (51,415 events).
  Effect: fault-plane and swarm-migration results are provisional. Paper: stated.
  <!-- src: README61 (counts, coherence); CLU60 (caveats); DEC D-B084 -->
- Station clocks: only MH07.XD and MG07.XD (spring 2016) are corrected; 129 candidate epochs are not, the largest at
  Axial (AXEC2.OO -0.67 s, Dec 2014 - Jan 2015) and G37B.7D (0.65 s); 111 stations were not screened. Effect: picks there
  can be off by up to these amounts. Paper: stated; review and correction in v6.2.
  <!-- src: CLOCK (candidate counts, clock-like list, open items); DEC D-B090, D-B097, D-B108 -->
- Location review not applied: reviewer verdicts on 1,340 events (1,512 event-problem rows of the v5.1 review) are not
  in v6.1; only the automatic indicators are carried. Effect: known problem events (possible non-earthquakes,
  unreliable depths) are not marked. Paper: stated; verdicts in v6.2.
  <!-- src: README61 (known limits, review columns); ACC61 (flags line); REVIEW (row count); DEC D-B097, D-B108 -->
- Mount St Helens test pulse: a weekly pulse on 25 UW analog channels (2011-05 to 2013-08) made 131 false events, now
  flagged; 5 probable and 68 waveform-only cases are not, and the flagged events fed the station terms. Effect: some
  false events remain under the volcano. Paper: flagged events left out of Mc, b and clustering (D-B105/D-B106 rule).
  <!-- src: MSH (findings, section 7b); HANDOFF 2026-11-26 23:22; README61 (known limits); DEC D-B108 -->
- Deep events under Mount St Helens: 275 unflagged events deeper than 30 km (v6.0 depths) remain in the volcano box;
  none of 68 with M >= 2.5 is in ComCat (35 of 41 shallow ones are). Effect: this deep population holds other non-earthquakes.
  Paper (D-B111): deep events under the volcano are not interpreted.
  <!-- src: MSH (section 0 ComCat check; section 5 depth table) -->
- Non-earthquake flags: the quarry-blast and explosion flags were carried from v6.0; 8 events keep a flag that v6.1's
  rules would not give and 6 lack one they would; v5.1 and v6.0 disagree on 27 ComCat matches near the 3 s / 20 km
  limits. Effect: negligible (tens of 172,342 events). Paper: none needed.
  <!-- src: HANDOFF 2026-11-27 07:00 (PR #60 review); MCCMP (Notes, 27 disagreements); README61 (counts) -->

## Velocity model and resolution

- Resolution: the finest checkerboard cell that passes near every fault is 40 km, on a node grid of about 10 x 11 km.
  Effect: the model shows plate-boundary-zone structure, not fault damage zones. Paper: stated in text and captions;
  sections show resolved nodes only.
  <!-- src: PBPLAN (sections 1, 3, 7); DEC D-B109 -->
- Nodes kept on ray count: most near-fault nodes pass the mask on ray sampling alone (DWS >= 100). The checkerboard
  share within 10 km is 28 % (P) at Blanco, 57 % at Mount St Helens, 13 % in Puget and below 10 % at the Mendocino fault
  zone, Gorda plate, triple junction and Gorda slab. Effect: DWS measures sampling, not resolution. Paper: stated.
  <!-- src: PBPLAN (section 1 "Scale", section 7) -->
- Model coverage: of 203,988 rock nodes, 5,559 are resolved for P (888 by checkerboard) and 1,084 for S (320); 16,833
  (8 %) move more than 0.5 % from the starting model. Effect: away from seismicity the model is the starting model.
  Paper: no full-size model maps (D-B107).
  <!-- src: MODELB (Results); DEC D-B107 -->
- No model at the northern ridges or the JdF plate: no tomography tile covers Axial, Endeavour or Explorer, and the JdF
  plate interior lies at tile edges (largest P DWS 84). Axial events are located in the starting model. Effect: no
  structure comparison there. Paper: seismicity-only panels for these segments (supplement S9).
  <!-- src: PBPLAN (sections 1, 5); DEC D-B085, D-B109 -->
- Circularity: ray coverage follows the seismicity, so velocity errors can sit where the events are. Effect: a contrast
  near seismicity can be an artefact. Paper: synthetic control on v0.4.2 before contrasts are read (D-B109; not yet run).
  <!-- src: PBPLAN (section 7 "Resolution"); DEC D-B109 -->
- Ensemble spread: Vp and Vs spreads are inflated by 1.50 and 1.35 to reach nominal coverage on resolved nodes, but the
  needed factor varies by tile (Oregon resolved Vp 2.10 on 29 nodes, 68 % coverage 0.48 after inflation); the spread
  leaves out smoothing bias. Effect: 1σ too narrow in Oregon. Paper: margin-wide factors and their derivation stated.
  <!-- src: INFL (per-tile rows); ENS (calibration section); PBPLAN (section 7 "Uncertainty"); DEC D-B094, D-B099 -->
- Vs calibration from one tile: all 320 resolved Vs nodes of the synthetic calibration lie in the Blanco tile.
  Effect: the Vs factor (1.35) is not tested elsewhere. Paper: stated.
  <!-- src: INFL (resolved_Vs rows by tile) -->
- Vp/Vs uncertainty: Vp/Vs spreads are not inflated; coverage is nominal overall (0.68 and 0.96 on 255 resolved nodes)
  but not calibrated per node (the plan assumed raw x 1.50). Effect: Vp/Vs contrasts lack a tested error bar.
  Paper: calibrated Vp/Vs uncertainty before use (D-B109; not yet run).
  <!-- src: ENS (synthetic table, resolved_VpVs); PBPLAN (sections 1, 7); DEC D-B109 -->
- Shallow Vp/Vs: station terms need large common S corrections at Axial (about -60 ms relative to P) and Puget
  (about -150 ms), a sign of a shallow Vp/Vs error; offshore, Vs in the top 2.5 km below the seafloor is capped at Vp/2.1
  by rule (8,739 nodes). Effect: it biases depths (Axial above). Paper: stated; model fix in v6.2.
  <!-- src: DEC D-B103, D-B085; DEPTH; MODEL (product info, rule_changed); MODELB (Methods) -->

## Magnitudes, Mc and b

- Offshore magnitude level: the offshore scale is tied to the onshore network through few shared picks. The converged
  solver moved ML by -0.68 at Blanco and -0.51 at the JdF ridge and plate; ComCat ML anchor gaps stay +0.26 at Mendocino
  and +0.10 at Blanco; anchor scatter 0.17 (3,412 anchors). Effect: regional levels can be off by 0.1-0.3. Paper: stated.
  <!-- src: DEC D-B091; MAGPROV61 (n_anchors, anchor_std) -->
- Axial magnitude tie: only 66 of the 22,053 events of the 2014 Axial cabled deployment have land picks; its anchor gap
  of +0.39 is open, and thinly tied levels move by 0.11-0.15 with a 100 km instead of a 150 km pick limit (v4.1 run).
  Effect: Axial magnitudes, Mc and b share this offset. Paper: stated.
  <!-- src: MAGREADME (Gauge section) -->
- Far-only magnitudes: 3,529 events with picks only beyond 150 km get an ML mapped onto the near-event scale with a
  scatter of 0.65 units. Effect: their magnitudes are rough. Paper: scatter stated.
  <!-- src: MAGPROV61 (ml_method mapped, mapped_fit std); MSB (Materials and Methods, mapped ML) -->
- Events without magnitude: 131,095 of the 172,342 events carry a magnitude, 537 of them from the v4.1 table
  (v4 hypocentres). Effect: Mc, b and the clustering use only events with a magnitude. Paper: counts stated.
  <!-- src: README61 (counts); ACC61 (flags line, magnitude sources) -->
- Mc rule: the clustering Mc (largest yearly maximum curvature + 0.2) sits where the two largest bins differ by 2-19 %,
  so a small magnitude change moves it by one or two bins (offshore Washington +0.20 cut its set from 706 to 428 events,
  v6.0). Effect: regional analysis sets shift. Paper: Mc stated; Mc + 0.3 and yearly-Mc variants in Text S6.
  <!-- src: CLU60 (caveats); MSB (Text S6 sensitivity) -->
- Axial frequency-magnitude distribution: two slopes (2014-2015 cabled array; an M 2-3 bulge in 2015), so the pooled
  Mc ranges from 0.3 to 1.5 by method (goodness-of-fit sigma 0.63). Effect: Axial Mc and b depend on the method.
  Paper: maximum curvature (0.7) or node values (0.4-1.2) at Axial (Text S7).
  <!-- src: MCREADME (Caveats; regional table, AX row); MSB (Text S7) -->
- b-value intervals: segment b-values range from 0.76 to 1.44 (v6.0); their 95 % bootstrap intervals ignore
  correlation in time. Effect: intervals can be too narrow where events cluster in time. Paper: stated.
  <!-- src: PBPLAN (sections 1, 4, 7) -->

## Clustering and swarms

- Epicentral distance: the nearest-neighbour clustering links events by 2-D epicentral distance. Effect: the depth biases
  above do not change the families; events stacked at different depths can join one family. Paper: 3-D variants (d3, d3q)
  in the sensitivity tests.
  <!-- src: CLU60 ("Unchanged", caveats, sensitivity variants) -->
- Family stability: of 2,097 v5.1 families, 1,550 are kept in v6.0 and 354 dissolve (magnitude change alone: 204 of
  the 547 not kept; location change alone: 197), but 144 of the 174 with 20 or more events are kept. Effect: small
  families are unstable. Paper (D-B111): claims about single families for large sequences only.
  <!-- src: CLU60 ("What changed against v5.1") -->
- Offshore Oregon: with 293 events above Mc, the bimodality of the nearest-neighbour distance is not stable (two of three
  time-shuffled catalogs fit a low-eta mode, Ashman D 3.1 and 3.6). Effect: the Oregon background and clustered split
  is uncertain. Paper: Text S6 says so (wording by the PI).
  <!-- src: MSB (D-B106, Text S6 bimodality) -->
- Swarm migration: 11 of 433 families with 10 or more events migrate by the permutation test, on tier A-C epicentres.
  Effect: provisional until a waveform double-difference catalog exists. Paper: stated.
  <!-- src: CLU60 ("Families: migration and depth", caveats) -->
- Brookings swarm: the 2012 swarm off Brookings (family 346 and surface GO020 in the draft) has to be re-identified after
  two family renumberings; its depth moved from about 20 km (v4) to 42-65 km (v5.1), where S-P is not monotonic with
  depth in the grids. Effect: its depth is unreliable. Paper: the [TO CHECK] stays until the v6.1 depth is checked.
  <!-- src: MSB (D-B091 and D-B106 table bullets); FAULT (Known limits) -->
- Network changes: rates and burstiness depend on the moving OBS array (2011-2015), and Axial enters tiers A-C only from
  September 2013. Effect: temporal metrics compare segments only roughly. Paper: stated.
  <!-- src: PBPLAN (section 4 "Temporal character", section 7) -->

## Fault surfaces

- Fitted on older locations: fault surfaces v1 were fitted on v5.1 positions; 7,557 of the 65,709 events on a surface
  moved more than 2 km between v5.1 and v6.0. Effect: membership and geometry lag the paper catalog. Paper: refit on
  v6.1 (D-B109; not yet run).
  <!-- src: CLU60 ("Fault surfaces v1"); FAULT; DEC D-B109 -->
- Detection stability: 42 of 130 non-strand surfaces are stable over detection reruns (strike range 6.5° against 103°
  for the others); 38 survive an 80 % subsample and 9 also survive moving every event by its error. Effect: the
  0.59 km position error holds for stable surfaces only. Paper: dips and widths are descriptive.
  <!-- src: FAULT ("Uncertainty scope"); PBPLAN (section 7) -->
- No null test: the detection has not been run on a spatially randomised catalog of the same density. Effect: the number
  of surfaces expected by chance is unknown. Paper: stated.
  <!-- src: FAULT (Known limits) -->
- Dip rule: faults flatter than 30° cannot be represented; the 1992 Cape Mendocino low-angle thrust zone (774 events) is
  split over three surfaces and the Olympia structure is lost. Effect: low-angle faults are missing from the set.
  Paper: stated.
  <!-- src: FAULT ("Dip rule cost") -->
- Depth control: 37 surfaces rest mostly on unconstrained or biased-deep events, and the bottom of the Blanco surfaces
  (30-50 km) is set by the v5.1 deep tail (13-17 km deeper than Lange et al. 2026 at the 90th percentile). Effect: those
  surfaces may reach too deep. Paper: dips and depth ranges from depth_quality ok events.
  <!-- src: FAULT (Known limits); PBPLAN (section 4, seismogenic width from depth-ok members) -->

## Structure versus seismicity

- Few contrasts above the uncertainty: of 105 depth-matched velocity contrasts near the plate boundaries, 34 exceed 2σ of
  the ensemble members and 27 the node 1σ; none at Blanco (largest 1.8σ); Vp/Vs reaches 2σ only on the Cascade arc
  (v6.0). Effect: structure separates boundary types less clearly than seismicity style. Paper: style leads (D-B109).
  <!-- src: PBPLAN (section 1 "Contrasts" and "Seismicity style", section 3); DEC D-B109 -->
- Contrasts that follow geography: the largest contrasts (0.4-0.6 km/s along strike at the triple junction and in the
  Gorda slab) follow the onshore-offshore transition. Effect: they are regional, not fault-zone, signals. Paper: not read
  as fault-zone structure.
  <!-- src: PBPLAN (section 1) -->
- Seismicity depth against velocity: the model rarely reaches 8 km/s offshore, so the 7.5 km/s level is used; the bottom
  of Gorda-slab seismicity follows it (ρ = 0.52, p = 0.001), Blanco weakly (ρ = 0.12, p = 0.046), the Mendocino fault zone
  not at all (v6.0). Effect: one segment supports a link. Paper (D-B111): the link is claimed for the Gorda slab only.
  <!-- src: PBPLAN (section 1, section 6 table) -->
- Swarm against mainshock-aftershock structure: only Blanco (9 swarm against 32 mainshock-aftershock families, Vp/Vs)
  and Puget (7 against 13) have enough resolved nodes; no difference (p >= 0.1). Effect: a null result with little power.
  Paper: reported as a null result.
  <!-- src: PBPLAN (section 1, section 5 figure 5D, section 6 table) -->
- Segment boundaries: PB2002 traces are approximate (Mendocino fault zone seismicity 8 km north of the trace, Blanco
  5 km south); the triple-junction segment holds 6,745 to 14,134 events for radii of 30 to 50 km. Effect: segment counts
  and shares depend on the thresholds. Paper: threshold sensitivity in supplement S1.
  <!-- src: PBPLAN (section 7, threshold table, section 5 supplement list) -->

## Data and network

- Moving OBS array: without the OBS (2010, 2016) the offshore catalog is incomplete at M 2.5-3 (Mc 1.6-1.9 with them);
  Blanco keeps 28-29 % of its 2013-2015 rate of M >= 2.5 events. Effect: offshore rates outside 2011-2015 are not
  comparable with the OBS years. Paper: offshore Mc compared within the OBS years; rate ratios stated.
  <!-- src: MCREADME ("Effect of the moving OBS array"); MSB (D-B091, Results bullets) -->
- Northern California pick gap: there are no NC or BK picks in 2016; the Mendocino rate of M >= 2.5 events falls to 11 %
  of its 2013-2015 rate, and v4 missed the 2016 Mw 6.6 sequence (added in v4.1). Effect: the 2016 drop there matches the
  pick gap. Paper: stated with the 2016 drop.
  <!-- src: M66 ("Why v4 missed them"); MCREADME ("2016 onshore drop"); MSB (Mendocino 11 %); README61 (v4.1 additions) -->
- Automatic association: events come from automatic picks and GENIE association; one spurious event built from Pn
  arrivals of the 2016 Mw 6.6 at 708-797 km was found and removed in v4.1. Effect: other spurious associations can
  remain. Paper: stated; the location review (v6.2) is the check.
  <!-- src: M66; README61 ("What it is", event 172113) -->

## Not a limitation (for the PI)

- Delph et al. (2018) comparison model: the IRIS EMC netCDF names "Ward & Lin (2018)" in its reference field, a
  metadata error in the file. Cite Delph, Levander & Niu (2018), GRL 45(20), doi:10.1029/2018GL079518; model
  doi:10.17611/dp/cascadiaantrfd2018. Resolved 2026-11-26 13:27.
  <!-- src: HANDOFF 2026-11-26 13:10 block (13:27 line); MODELB (References) -->

## Source keys

| key | file |
|---|---|
| DEC | `workflow/10_tomography/DECISIONS.md` on origin/tomo/tiles (decision IDs as given) |
| HANDOFF | `/wd1/mdenolle_data/tomo_v1/handoff_2026-11-24/README.md` (entry time as given) |
| ACC60 | `/wd1/mdenolle_data/reloc_v6/catalog/acceptance_v6.0.html` |
| ACC61 | `/wd1/mdenolle_data/reloc_v61/run/catalog/acceptance_v6.1.html` |
| README61 | `/wd1/mdenolle_data/catalog_v6/v6.1/README_v6.1.md` |
| CLOCK | `/wd1/mdenolle_data/catalog_v6/v6.1/pick_timing_README.md` |
| DEPTH | `/wd1/mdenolle_data/reloc_v61/depth_test/report_depth_test.html` and `diag_explanation.md` |
| PBPLAN | `/wd1/mdenolle_data/paper_plan/plate_boundaries_plan.html` (built on catalog v6.0) |
| ENS | `/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2/products/report_real.html` |
| INFL | `/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2/calibration/inflation.md` |
| MODEL | `/wd1/mdenolle_data/tomo_v1/models/staging_v0.4.2/report_v0.4.2.html` |
| MODELB | `/wd1/mdenolle_data/tomo_v1/models/figures_v0.4.2/manuscript_bullets_model.md` |
| MSH | `/wd1/mdenolle_data/reloc_v61/flags/msh_analog_flags.html` |
| CLU60 | `/wd1/mdenolle_data/catalog_v6/v6.0_analysis/clusters/report_v60_clusters.html` |
| MCCMP | `/wd1/mdenolle_data/magnitude_converged/mc_maps_v60/compare.html` |
| MAGPROV61 | `/wd1/mdenolle_data/magnitude_converged/v6.1/cascadia_catalog_M_v61_converged.provenance.json` |
| MSB | `data/magnitude_converged/manuscript_bullets.md` (origin/main) |
| MAGREADME | `data/magnitude_converged/README.md` (origin/main) |
| MCREADME | `data/catalog_v4/mc_maps/README.md` (origin/main) |
| FAULT | `data/catalog_v5/v5.1_analysis/faults/README.md` (origin/main) |
| REVIEW | `data/review_v51/README_review.md` (origin/main) |
| M66 | `data/catalog_v4/fixes/m66_2016/README.md` (origin/main) |
