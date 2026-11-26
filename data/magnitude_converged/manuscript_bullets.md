# Manuscript changes for the converged Route A magnitudes (D-B091)

- Scope: every number and sentence of the Science Advances draft that changes when the paper uses the converged Route A magnitudes (exact phase19 solver, PR #45; switch PR #46).
- Bullets only, for the authors to apply on Overleaf. The manuscript and Overleaf are not edited.
- Line numbers: "OL" = Overleaf `sciadv/main.tex` / `sciadv/supplementary.tex` as mirrored in Denolle-Lab/Offshore-Cascadia-Ensemble-Catalog (commit 8def2ea4, 2026-11-23); "repo" = `paper_sciadv/*.tex` on main (5d6658bf). Search for the quoted words if lines have moved.
- Values: `data/magnitude_converged/paper_numbers.csv` (stored and converged, recomputed with the paper's own scripts: `workflow/05_magnitude/mag_converged_paper.py`). Every stored value below reproduces the number now in the text, except where marked.

## Main text, Results: "A plate-wide catalog from a moving amphibious array"

- OL 78, repo 78, "(3,416 common events, scatter 0.21 magnitude units)": scatter 0.21 -> **0.17**. "130,769 events" unchanged.
- OL 80-81, repo 80-81, "lowest at Mount St. Helens and in the Axial caldera (0.0--0.3)": no change (node minima: Mount St. Helens -0.2, Axial 0.3; stored -0.2 and 0.2).
- OL 81-82, repo 81-82, "Puget Lowland and offshore Washington (node medians 1.0--1.1)": 1.0--1.1 -> **1.1--1.2** (Puget 1.2, Washington 1.1).
- OL 82-83, repo 82-83, "highest on the western Blanco transform and the northern Gorda Ridge (2.9--3.2)":
  - -> **highest on the western Blanco transform (2.6--2.8; 43.6--43.9°N, 129.3--129.6°W)**.
  - Drop "and the northern Gorda Ridge": the Gorda nodes reach 2.4 at most (stored 2.8); no node is at or above 2.9 (stored 17).
- OL 83-85, repo 83-85, "Pooled by region, Mc is 1.2--1.7 onshore and in the forearc and 1.8--2.2 on the offshore plate boundaries":
  - onshore and forearc 1.2--1.7 -> **1.3--1.5** (WA 1.3, PU 1.3, ON 1.3, OR 1.5, MT 1.5);
  - offshore plate boundaries 1.8--2.2 -> **1.6--1.9** (Endeavour 1.6, Blanco 1.6, Gorda 1.9).
- OL 86-88, repo 86-88, "in 2010 and 2016 no offshore node has 150 events within 50 km": no change (Endeavour and Blanco no node, Gorda 5 nodes near the triple junction in 2010, as before).
- OL 88-89, repo 88-89, "the rate of M >= 2.5 events on the Blanco transform drops to 5% (2010) and 8% (2016) of its 2013--2015 rate":
  - 5% -> **28%** (2010); 8% -> **29%** (2016).
  - At M >= 3.0: 39% and 47% (stored 20% and 19%).
- OL 89-91, repo 89-91, "Offshore Mc also improved by 0.2--0.4 between the first two deployment years and 2013--2015 (Blanco 2.6 to 2.2)":
  - no longer true: delete, or -> **offshore Mc is about the same in 2011--2012 and 2013--2015 (pooled Blanco 1.5 and 1.6)**.
  - Pooled 2011--2012 -> 2013--2015: Endeavour 1.7 -> 1.6, Blanco 1.5 -> 1.6, Gorda 1.9 -> 1.9 (stored 1.9 -> 1.8, 2.6 -> 2.2, 2.1 -> 2.1).
  - Node medians: Endeavour 1.9 -> 1.8, Blanco 1.6 -> 1.5, Gorda 1.9 -> 1.8.
  - Why: the stored Blanco level differed between deployments (near-event median ML 2012 / 2013 / 2014 / 2015: stored 2.23 / 2.07 / 2.12 / 1.61, converged 1.55 / 1.39 / 1.62 / 1.56).
- OL 91-92, repo 91-92, "median change 0.0 to 0.1": no change (-0.1 from 2010 to 2013--2015, 0.0 from 2013--2015 to 2016).
- OL 92-94, repo 92-94, "The 2016 drop near Mendocino (M >= 2.5 rate 10% of 2013--2015)": 10% -> **11%**.

## Main text, Results: "Seismic velocity structure" (Overleaf bullets only)

- OL 126, "Zhang & Niksejel 2025, SSA abstract, about 24,000 offshore events, Mc 1.5": no number of ours in this bullet. If compared: our pooled offshore plate-boundary Mc is now **1.6--1.9** (stored 1.8--2.2); all offshore boxes pooled 1.7 (stored 2.2).

## Main text, Results: "How faults slip follows the tectonic setting"

- OL 230-232, repo 208-210, "All six large earthquakes ... mainshock--aftershock sequences with the mainshock first": no change (all six; no event above Mc before the mainshock).
- OL 232, repo 210, "the 2015 Axial eruption as one 632-event swarm": 632 -> **601** (events above Mc; 762 with the attached events below Mc; stored 805).
- OL 234-236, repo 212-214, "The ridges and transforms (Axial, Blanco, Gorda) are about half background": holds for Axial 51.6% and Blanco 49.8%; Gorda **62.5%** (stored 50.6, 50.2, 60.8%).
- OL 236-237, repo 214-215, "two thirds of their families of five or more events are swarms":
  - -> **55--63%** (Axial 60.7% of 61 families, Blanco 62.8% of 145, Gorda 55.0% of 20; stored 60.6, 66.0, 65.4%).
  - "two thirds" no longer fits Gorda; "more than half" fits all three.
- OL 237, repo 215, "The Mendocino triple junction has fewer swarms (31%)": 31% -> **28%** (8 of 29 families; stored 15 of 49).
- OL 238-239, repo 216-217, "76% of its events 1--2 magnitude units above Mc have no linked aftershock above Mc": 76% -> **74%** (of 417 events; stored 592).
- OL 240-241, repo 218-219, "The forearc, Puget Lowland and onshore regions are 68--84% background": -> **67--82%** (WA 82.5, OR 66.7, PU 78.7, ON 70.0).
- OL 245-246, repo 223-224, "swarms are short (median 4 hours) and compact (4 km)":
  - "4 hours" unchanged (4.0 h);
  - 4 km -> **5 km** (median extent 4.6 km; stored 3.9 km);
  - 91 Blanco swarms (stored 62).
- OL 246-248, repo 224-226, "the 2012 Mw 5.8 and 6.0 events share one swarm, and three M 5.4--5.9 events": no change (tensor Mw; the 2012 family is still one swarm, 115 events above Mc, stored 77).
- OL 248-250, repo 226-228, "Swarm shares are highest on the West Blanco, Surveyor and Cascadia Depression strands (33--45% of events above Mc)":
  - 33--45% -> **28--45%** (BL004a 44.8, BL015a 42.2, BL006a 28.1, BL012 27.8; stored 45.0, 34.5, 34.6, 32.9).
  - The same strands stay the highest (the fifth is BL018a, West Blanco north wall, 28.3%).

## Main text, Results: "Short-lived processes caught by the moving array"

- OL 260-262, repo 238-240, "rose from about 60 in November 2014 to about 300 in mid-March 2015" (events above Mc per week, Fig. 6A):
  - about 60 -> **about 55** (highest November 7-day bin 54, stored 59; November mean 36, stored 39);
  - about 300 -> **about 290** (mid-March mean 285, stored 314), or keep "about 300".
- OL 262-263, repo 240-241, "swarm families from 1 per month to 16 in March and 13 in April": 16 -> **15**; 1 and 13 unchanged.
- OL 263, repo 241, "The eruption week of 24 April holds 1,021 events":
  - state the count and its definition:
    - calendar week Monday 20 -- Sunday 26 April 2015: **998** events above Mc (stored 1,024);
    - the 7-day bin of Fig. 6A that holds 24 April: **975** (stored 999).
  - 1,021 is not reproduced by either definition with the stored clustering on main.
  - Suggested: "998 events above Mc in the week of 20--26 April".
- OL 264, repo 242, "within four weeks the rate fell below 15 per week": no change (7-day bins after the peak: 41, 5, 4, 1; stored 45, 11, 4, 2).
- OL 265, repo 243, "then to almost none above Mc 0.75": 0.75 -> **0.85** (Axial clustering Mc).
- OL 270-273, repo 248-251, West Blanco swarm: "M 5.8 at 06:52 UTC", "M 5.4 four hours later", "M 5.9 thirteen hours after": no change (tensor Mw).
- OL 273-274, repo 251-252, "Within six hours, events spanned 23 km of the transform; within one day, 41 km": no change (5--95% range along 111°: 23 and 41 km).
- OL 274-275, repo 252-253, "the length the family kept for the next 64 days (755 events)":
  - -> **about the length the family kept for the next 100 days (850 events)**;
  - the family now runs 1 June -- 9 September 2015 (stored 1 June -- 4 August), with 277 events above Mc (stored 77);
  - its 5--95% extent is 46 km (stored 43 km).
- OL 275-276, repo 253-254, "Seismicity spreading to 40 km in a day": no change.
- OL 291-294, repo 269-272, Fig. 6D, "only events near the coast or above M 4 remain when they were not":
  - not supported with either magnitude set: in weeks with no ocean-bottom station within 120 km, 58 well-recorded events remain west of 125.5°W (1,486 in weeks with stations);
  - in 2012--2015 these 56 events are M **1.5--3.2** (stored 1.7--3.4);
  - only in 2016 are they all above M 4 (the Mw 6.6 and its M 4.7 aftershock);
  - suggested: "only a few events remain (56 of M 1.5--3.2 in 2012--2015; in 2016 only the Mw 6.6 and its aftershock)".

## Main text, Discussion

- OL 341-343, repo 319-321, "(33--45% of events above Mc on the West Blanco, Surveyor and Cascadia Depression strands)": -> **28--45%**.
- OL 352-354, repo 330-332, "releases half of its earthquakes in clusters, two thirds of its families as swarms":
  - clustered share: Axial 48.4%, Blanco 50.2%, Gorda 37.5% (stored 49.4, 49.8, 39.2): "half" holds for Axial and Blanco;
  - "two thirds of its families as swarms" -> **more than half (55--63%)**.
- OL 360-362, repo 338-340, "where the network, the magnitude scale and the classification are the same everywhere": keep. The converged solution supports it better: ComCat ML anchor gap offshore and on the coast +0.12 -> +0.01.
- OL 383-385, repo 361-363, Limits, "The per-region Mc keeps 27,555 events for the clustering": 27,555 -> **27,423**.

## Materials and Methods

- OL 429 (bullet; not in the repo), "using the 112 matched moment tensors": 112 -> **147** (catalog v4.1; 112 was the catalog v3 count).
- OL 435-438, repo 405-408, "Wood--Anderson amplitudes were measured for every pick ..., inverted for event size, distance attenuation and station terms, and anchored to 3,416 ComCat local magnitudes":
  - wrong for 127,245 of the 130,769 events (picks within 150 km): there the Hutton--Boore distance term is fixed and is not inverted;
  - for those events: event terms and station-phase terms (per station, phase and deployment epoch) are inverted together;
  - the solver is exact least squares: event terms eliminated, the 1,337 x 1,337 station-phase system solved directly, two refinement steps on the full system, one gauge row (P terms sum to zero), convergence and gauge checked; one offset to the 3,416 ComCat ML anchors (scatter 0.17);
  - 3,521 events with picks only beyond 150 km: ML from the distance-attenuation inversion, mapped to the near-event scale by a Theil--Sen fit;
  - moment magnitudes above 4.5: unchanged;
  - cite Hutton & Boore 1987 (`\citep{Hutton1987}`, already in references.bib, DOI 10.1785/BSSA0770062074).
- OL 439-450, repo 409-420, Mc methods: no change.
- OL 486-501, repo 456-471, clustering method: no change ("Mc is the maximum curvature plus 0.2 per region and year" still holds).

## Supplementary text

- Text S5, OL/repo 177-181, "Station magnitudes with the Hutton--Boore distance term are inverted for event and station terms (separate for each deployment epoch) within 150 km, anchored to 3,416 ComCat local magnitudes (scatter 0.21) ... 130,767 events have a magnitude":
  - add the citation after "Hutton--Boore distance term": `\citep{Hutton1987}` (named, never cited);
  - "inverted for event and station terms" -> add **"by exact least squares with one gauge row (D-B091)"**;
  - scatter 0.21 -> **0.17**;
  - 130,767 -> **130,769** (catalog v4.1; 130,767 was catalog v4.0; the main text already says 130,769).
- Text S6, OL/repo 224, "measured epicentral D2 0.8--1.4": no change.
- Text S6, OL/repo 228-229, "bimodality index drops from 1.6--3.4 to 0.3--1.6": -> **1.7--2.9 to 0.4--1.3**.
- Text S6, OL/repo 229-231, "only 3--17% of shuffled events pass the real threshold (34% at Axial ...)": -> **3--19% (33% at Axial)**.
- Text S6, OL/repo 231-233, validation sequences: no change (the six earthquakes are mainshock--aftershock; the eruption is one swarm).
- Text S6, OL/repo 234-236, "84--99% of events keep their background or clustered label under df 1.3/2.0/estimated, 3D distance, Mc + 0.3 and yearly Mc; a conservative threshold raises background by 1--32 points":
  - 84--99% -> **83--99%**;
  - 1--32 points -> **2--29 points** (most at Axial: 51.6% -> 81.1%).
- Text S6, OL/repo 236-238, "the per-region Mc keeps 27,555 events; Mc differs between regions (0.75--2.45)": -> **27,423; 0.85--2.25** (per region: AX 0.85, EN 2.25, BL 1.75, GO 2.25, MT 1.85, WA 1.65, OR 1.85, PU 1.35, ON 1.65).
- Text S6, OL/repo 238-239, "the swarm rule's magnitude gap (0.5) is close to the ML uncertainty (about 0.2)": no change.
- Text S7, OL/repo 250-252, "Maximum curvature + 0.2 is the most conservative at 84% of nodes; b-value stability averages 0.19 lower and the best estimate 0.14 lower": -> **83%; 0.19; 0.13**.
- Text S7, OL/repo 252-254, QC-only catalog "median difference 0.0, interquartile range +-0.1": no change.
- Text S7, OL/repo 254-256, "The per-region Mc of the clustering is at or above all three methods everywhere except the Blanco b-value-stability value (2.5 against 2.45)":
  - -> **at or above maximum curvature and the best estimate everywhere; below the b-value-stability value at Axial (0.85 against 1.5) and by half a bin at Blanco (1.75 against 1.8) and Puget (1.35 against 1.4)**;
  - the map methods round M to 0.1, so 1.8 keeps M >= 1.75;
  - Puget was already 1.35 against 1.4 with the stored magnitudes, but the text did not list it.
- Text S7, OL/repo 256-259, "pooled goodness-of-fit value is unstable (sigma 0.94); use maximum curvature (0.7) or the node values (0.4--1.0)":
  - sigma 0.94 -> **0.63**;
  - maximum curvature 0.7 unchanged;
  - node values 0.4--1.0 -> **0.4--1.2**;
  - the pooled Axial b-value-stability Mc is now 1.5 (stored 0.4), also on the second slope.

## Supplementary tables and figures (files; same paths, replaced in the figures PR)

- Table S4 (`data/faults/v0/fault_surfaces_v0_literature.csv` and `.html`, column lit_evidence; not changed in the PR):
  - GO004, "catalog maximum M5.6 on 2015-04-24" -> **"largest Route A magnitude Mw 5.45 on 2015-04-24 (event 144373, the ComCat M 5.4 tensor)"** (same in the stored file);
  - GO011, "Catalog maximum M4.1 on 2014-10-03" -> **"largest Route A magnitude ML 3.7 on 2014-10-03 (event 103612; stored 3.85)"**;
  - both quoted the catalog's own magnitude column, not Route A.
- Table S5 (`data/catalog_v4/clusters_zbz/families_zbz.csv`, `regions_zbz.csv`): replaced. Family IDs renumber:
  - Axial eruption family 1758 -> **1750**;
  - June 2015 West Blanco family 1803 -> **1797**;
  - 2012 Blanco family 270 -> **259**.
- OL 311-312, repo 289-290, "[TO CHECK ...] a swarm off Brookings, Oregon, July--September 2012 (family 346, 28 events, GO020)": family IDs changed; re-identify it in the new `families_zbz.csv` before the text uses it.
- Table S6 (`data/catalog_v4/mc_maps/mc_regional.csv`): replaced; README numbers in `data/catalog_v4/mc_maps/README.md` updated.
- Figs. S3--S7 (`data/catalog_v4/mc_maps/fig1`--`fig5`): replaced.
- Text S6 report (`data/catalog_v4/clusters_zbz/catalog_v4_clusters_report.html`): tables and figures recomputed; its hand-written prose still quotes the stored run and says so at the top.

## Figure captions

- Fig. 1, 2, 3, 5, 6 and the Mendocino perspective view: new files with the same names in `paper_sciadv/figures/`; captions unchanged.
- Fig. 4: unchanged (it does not use M; pixel-identical).
- Fig. 6A axis label now reads "events >= Mc (0.85) per week" (was 0.75).
- Fig. 6B in-panel note now reads "503 events in 12 days (850 in the family, 100 days)" (was "501 ... (755 ..., 64 days)").
