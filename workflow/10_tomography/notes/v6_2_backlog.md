# Catalog v6.2 backlog

- Written 2026-11-27 09:18 PST (time from `date`), branch paper/limitations.
- PI, 2026-11-27: "All caveats can be written in terms of limitations in the paper, keep next updates for v6.2."
  Nothing in this list is done now. The paper states each caveat as a limitation
  (`data/paper_plan/limitations_bullets.md`).
- One entry per item: what, why (evidence), proposed fix, decision, effort.
- Effort: S = under a day of work; M = one to three days, including one relocation run (about 6 h, D-B108);
  L = more than three days or a model re-inversion.
- Sources: HTML comment after each entry; keys at the end of this file (same keys as the limitations file).
- Section G lists the open D-B109 steps for the paper. They are not v6.2 items unless the PI says so.

## A. Velocity model and station terms

### A1. Near-surface Vp/Vs at Axial and Puget
- What: correct the shallow Vp/Vs of the model where the station terms need large common S corrections, then relocate.
- Why: 72.7 % of the 12,584 tier A-B Axial events lie within 2 km of the seafloor against the 85 % target (v6.0 66.1 %,
  v5.1 81.0 %). The common S-P term is about -60 ms at Axial and about -150 ms at Puget. Removing the common term from the
  station terms instead (arm cm4) puts 21 % of the Axial events on the floor, so part of it is a real model error.
- Fix: a local shallow Vp/Vs model at Axial (no tomography tile covers it; the model there is the starting model) and a
  shallow correction in the Washington tile; score with the D-B103 protocol (held-out fit gate, then literature depths);
  relocate.
- Decision: D-B103, D-B085.
- Effort: L.
<!-- src: ACC61 (Axial table); DEC D-B085, D-B103; DEPTH (implications); PBPLAN (section 1, no tile at Axial) -->

### A2. Station terms without the common cluster shift, every event weighted equally
- What: test a tile-wise removal of the part of the terms that a common shift of the cluster would absorb, with every
  event weighted equally.
- Why: the registered arm nf4 (Hessian-weighted) gave the best Axial depth (92.9 % within 2 km) but over-corrected the
  sparse offshore clusters (western Blanco -2.4 km against Lange) and failed the fit gate at the Gorda Depression.
  The depth test names this as follow-up (2).
- Fix: pre-register the rule, run the ten-arm protocol on the four literature clusters, adopt only if it passes the gate.
- Decision: D-B103 (depth-test follow-up).
- Effort: M.
<!-- src: DEPTH (scores, nf4 paragraph, implications) -->

### A3. Ensemble inflation factors by tile
- What: decide whether the Vp and Vs spreads get one factor per tile instead of one for the margin.
- Why: the margin-wide Vp factor 1.50 leaves the Oregon resolved nodes at 68 % coverage 0.48 (needed factor 2.10,
  29 nodes); sampled Vp at Blanco needs 1.70. All 320 resolved Vs nodes of the calibration are in the Blanco tile.
  Vp/Vs has no per-node calibration.
- Fix: per-tile factors from the existing synthetic coverage (no new runs), or state the margin-wide factor as a floor.
  PI decides.
- Decision: D-B094, D-B099 (amend if adopted).
- Effort: S.
<!-- src: INFL (per-tile rows); ENS (calibration, synthetic table) -->

## B. Picks and event flags

### B1. Clock-candidate epochs
- What: review and apply the 129 candidate clock epochs.
- Why: only MH07.XD and MG07.XD are corrected. 25 of the 131 candidates look clock-like, among them AXEC2.OO at Axial
  (-0.67 s, 2014-12-13 to 2015-01-15) and G37B.7D (0.65 s over its deployment). Record headers were scanned only for
  MH07 and MG07; 111 stations were not screened.
- Fix: PI fills `detect_full/review_candidates.csv` (accept / reject); `scan-headers` for every accepted candidate;
  `ingest-review`, `model-clock`, `apply`; new pick file; relocate.
- Decision: D-B090, D-B097, D-B108, D-B085.
- Effort: M (person review, then one relocation run).
<!-- src: CLOCK (candidates, clock-like list, open items); DEC D-B090, D-B097, D-B108, D-B085 -->

### B2. Location-review verdicts
- What: apply the reviewer verdicts of the v5.1 location review.
- Why: 1,512 event-problem rows on 1,340 events; v6.1 carries the automatic indicators, and the verdict columns
  (review_not_earthquake, review_depth_unreliable, review_relocate, review_depth_km) are empty.
- Fix: PI completes the review page and returns the CSV; the catalog stage ingests it (the v6.1 builder already reads
  reviewer verdicts when given).
- Decision: D-B097, D-B108.
- Effort: M (person review).
<!-- src: README61 (columns, known limits); ACC61 (flags line); REVIEW; PR60 (overview: reviewer-verdict ingestion) -->

### B3. Mount St Helens review lists
- What: decide on the Mount St Helens events the weekly-pulse rule did not flag.
- Why: 5 probable (timing only) and 68 waveform-only events (pulses outside the weekly windows) are listed, not flagged.
  275 unflagged events deeper than 30 km remain in the volcano box; none of 68 with M >= 2.5 is in ComCat, against
  35 of 41 shallow ones. Event 8194 (2010-08-04) sits on an irregular calibration the scan did not cover.
- Fix: PI review of `msh_analog_candidates.csv` and `msh_analog_waveform_only.csv`; a pre-registered rule for the deep
  box population (ComCat, waveform and moveout checks); flags only, no deletion.
- Decision: D-B108.
- Effort: M.
<!-- src: MSH (findings, controls, section 7b); DEC D-B108 -->

### B4. Station terms without flagged events
- What: leave the events with an event_flag out of the station-term and coherence inputs.
- Why: the 131 analog-signal events and the quarry-blast and explosion events fed the Loc2ssst, coherence and static
  terms of v6.1.
- Fix: drop flagged events from the Loc2ssst input and from the coherence partner lists; relocate every event with the
  new terms; report the change in held-out fit.
- Decision: D-B108 (v6.2 candidate).
- Effort: S for the code; the run goes with the v6.2 relocation.
<!-- src: HANDOFF 2026-11-26 23:22; README61 (known limits) -->

### B5. Non-earthquake flags from the v6.1 rules
- What: compute the quarry-blast and explosion flags from the rules on each version's own hypocentres.
- Why: v6.0's flags went into v6.1 through the flag file; 8 events keep a flag that v6.1's rules would not give and 6
  lack one they would. v5.1 and v6.0 also disagree on 27 ComCat matches near the 3 s / 20 km association limits.
- Fix: run the D-B071 rules at the catalog stage, then add the analog-signal flags; list the changes against the
  previous version.
- Decision: D-B071, D-B108 (PR #60 review).
- Effort: S.
<!-- src: HANDOFF 2026-11-27 07:00; MCCMP (Notes) -->

## C. Relocation recipe

### C1. Held-out fit worse than v6.0
- What: find why v6.1 fits held-out waveform differential times worse than v6.0 in four cluster-phases.
- Why: Puget P +17.9 %, Puget S +15.2 %, Mendocino S +7.2 %, Gorda S +7.7 % (4 of 18 cluster-phases worse by more than
  5 %). v6.1 differs from v6.0 in the station-term stop (16 against 4 km) and the model (v0.4.2 against v0.4.1). The
  pair counts differ only in the Blanco and JdF sets (for example JdF S 1,180 against 1,216).
- Fix: rerun the held-out protocol on puget, mtj and gorda with the two changes separated (v0.4.2 with a 4 km stop;
  v0.4.1 with a 16 km stop), on the same pairs.
- Decision: D-B103 (PR #60 review).
- Effort: M.
<!-- src: ACC61 (comparison table); ACC60 (held-out table); HANDOFF 2026-11-27 07:00; DEC D-B103 -->

### C2. Mendocino coherence skip: second test
- What: test the coherence stage on a second Mendocino cluster before keeping the skip.
- Why: the skip rests on the 2015 sequence alone (v6.0: P +8 ms, S +20 ms worse). In v6.1 the same cluster gives
  S +13.1 ms [+7.4, +18.5] and P no clear effect. The stage also hurts Gorda S slightly (+1.1 ms [0.0, +2.0]), which is
  not skipped. 10,500 compact events keep their station-term location.
- Fix: pre-register a test on an independent Mendocino cluster (and the Gorda set), then set the skip regions by rule.
- Decision: D-B098.
- Effort: M.
<!-- src: DEC D-B098; ACC61 (held-out coherence table, skipped_region counts) -->

### C3. Negative corrected travel times and the arrival cap in Loc2ssst
- What: check two warnings in the Loc2ssst table of the v6.1 acceptance report.
- Why: at the 16 km pass, 5,186 grid nodes in 386 tiles have a corrected travel time below 0 (minimum -1.65 s; NonLinLoc
  does not use a negative predicted time), so at least one station-phase term is near or below -1.65 s. 244 of 1,227
  tiles had partial input from the 20,000-arrivals-per-station-phase limit.
- Fix: list the station-phases with the largest negative terms and trace their cause; measure how the arrival cap
  changes the terms on one busy tile.
- Decision: none yet (found in the v6.1 acceptance report).
- Effort: S.
<!-- src: ACC61 (Loc2ssst table) -->

## D. Pipeline and reports

### D1. Catalog-stage rerun undoes the magnitudes stage
- What: make the chain keep the v6.1-hypocentre magnitudes when only the catalog stage is rerun.
- Why: PR #60 review: rerunning stage catalog alone replaces the magnitudes stage output.
- Fix: the catalog stage reads the magnitude table named in the run state, or the chain reruns magnitudes after it;
  a check that the magnitude_source counts match the magnitude table.
- Decision: D-B105 (PR #60 review).
- Effort: S.
<!-- src: HANDOFF 2026-11-27 07:00 -->

### D2. Partner-rule table in the acceptance report
- What: add the partner-rule check table to every acceptance report.
- Why: v6.0's report has "Rule check before the coherence stage" (R3 against the test rule, puget and mtj); v6.1's does
  not.
- Fix: emit the table from `b4_v6_acceptance.py` for every version.
- Decision: partner rule R3 (notes/v6_plan.md), PR #60 review.
- Effort: S.
<!-- src: ACC60 (rule-check table); ACC61; HANDOFF 2026-11-27 07:00 -->

### D3. Copilot items on PR #60
- What: two small code items.
- Why: `--mag-tables` accepts entries without "=" that later crash on unpacking (`b4_v6_catalog.py`); the `pool_map`
  docstring says completion order but one branch returns input order (`b4_time_grids.py`).
- Fix: validate the argument and fail with a clear message; correct the docstring.
- Decision: none (code review).
- Effort: S.
<!-- src: PR60 -->

## E. Magnitudes

### E1. Axial magnitude level
- What: tie the 2014 Axial cabled deployment to the onshore scale or state its offset.
- Why: 66 of its 22,053 events have land picks; its ComCat anchor gap of +0.39 is open; thinly tied levels move by
  0.11-0.15 with a 100 km instead of a 150 km pick limit (v4.1 run).
- Fix: check the gap with the v6.1 table; if it holds, a deployment offset from shared events or a stated uncertainty.
- Decision: D-B091.
- Effort: M.
<!-- src: MAGREADME (Gauge section); DEC D-B091 -->

## F. Products downstream of the catalog

### F1. Brookings swarm depth
- What: check the depth of the 2012 swarm off Brookings in v6.1 and re-identify its family.
- Why: in v5.1 its events sat at 42-65 km (about 20 km in v4), where S-P is not monotonic with depth in the grids and
  NonLinLoc took the deep minimum (review set P2). The draft cites family 346 and surface GO020; family IDs were
  renumbered twice since (D-B091, D-B106).
- Fix: compare its v6.1 depths with v4; check S-P against depth in the v0.4.2 grids at the swarm; re-identify the family.
- Decision: D-B106 (manuscript bullets), review P2.
- Effort: S.
<!-- src: FAULT (Known limits); MSB (D-B091 and D-B106 table bullets) -->

### F2. Fault-surface detection: null test and stability
- What: run the detection on a spatially randomised catalog of the same density and repeat the stability tests.
- Why: 42 of 130 non-strand surfaces are stable over detection reruns; 9 pass both the subsample and the perturbation
  tests; no null test exists.
- Fix: the null test, then the subsample and perturbation tests on the v6.1 refit.
- Decision: none yet (fault surfaces v1 README, Known limits); follows the D-B109 refit.
- Effort: M.
<!-- src: FAULT ("Uncertainty scope", Known limits) -->

### F3. Waveform double-difference catalog (optional)
- What: decide whether a relative relocation from waveform differential times is made as a separate product.
- Why: fault-plane and swarm-migration results stay provisional until one exists. tomoDD with the surface floor is
  kept as a cross-check for dense clusters but gives no catalog column.
- Fix: PI decides; if yes, tomoDD plus floor on the clustered families, compared on held-out fit and literature depths.
- Decision: D-B078, D-B084.
- Effort: L.
<!-- src: CLU60 (caveats); DEC D-B084 -->

## G. Not v6.2: open D-B109 steps for the paper on catalog v6.1

Listed so nothing is lost. D-B109 makes them paper work on v6.1; whether they wait for v6.2 is the PI's call.

- Fault surfaces refit on v6.1 (v1 is fitted on v5.1; 7,557 of 65,709 on-surface events moved more than 2 km by v6.0).
- Clustering on v6.1, with the Mc-map change of D-B105 (mc_best variant: OR 1.5 to 1.6, PU 1.3 to 1.2).
- Mc maps on v6.1 (maps exist for v4.1, v5.1 and v6.0 only).
- Plate-boundary numbers recomputed on v6.1 (the plan used v6.0 as a stand-in).
- Synthetic circularity control on v0.4.2 before any contrast is read.
- Calibrated Vp/Vs uncertainty.
<!-- src: DEC D-B109; CLU60 (fault section); MAGRES (2026-11-26 16:31 entry); MCCMP; PBPLAN (header, section 7) -->

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
| MSH | `/wd1/mdenolle_data/reloc_v61/flags/msh_analog_flags.html` |
| CLU60 | `/wd1/mdenolle_data/catalog_v6/v6.0_analysis/clusters/report_v60_clusters.html` |
| MCCMP | `/wd1/mdenolle_data/magnitude_converged/mc_maps_v60/compare.html` |
| MAGRES | `/wd1/mdenolle_data/magnitude_converged/RESUME.md` |
| MSB | `data/magnitude_converged/manuscript_bullets.md` (origin/main) |
| MAGREADME | `data/magnitude_converged/README.md` (origin/main) |
| FAULT | `data/catalog_v5/v5.1_analysis/faults/README.md` (origin/main) |
| REVIEW | `data/review_v51/README_review.md` (origin/main) |
| PR60 | GitHub PR #60, Copilot review comments (head ea048c25) |
