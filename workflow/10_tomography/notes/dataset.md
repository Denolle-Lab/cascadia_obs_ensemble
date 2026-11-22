# B1 tomography dataset (events, picks, stations)

Agent: Claude (B1 subagent), 2026-11-22. Status: dataset built; pick QC, post-QC event rule and OBS handling
are **proposed** (D-B009, D-B010 to be written by the coordinating session).

Scripts: `workflow/10_tomography/b1_hyp_phases.py` (reads PHASE lines of every per-event NLLoc hyp file of a
Track A run into a parquet cache) and `workflow/10_tomography/b1_select_events.py` (everything else). Parameters:
`params.yaml` blocks `selection` (D-B003, unchanged), `pick_errors` (D-B008), and the new `pick_qc` and `obs`
blocks (marked "proposed"). Run manifest: `/wd1/mdenolle_data/tomo_v1/b1_dataset/b1_select_events.manifest.json`.
All numbers below come from `data/tomo_v1/b1_dataset/b1_tables.md`, which the script writes.

```
LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/10_tomography/b1_hyp_phases.py iter1_vpvs176
LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/10_tomography/b1_hyp_phases.py iter0_vpvs176
LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/10_tomography/b1_select_events.py
```

## Result

| | count |
|---|---|
| events | 23,428 (A 7,060, B 7,729, C 8,639) |
| picks | 692,246 (P 343,544, S 348,702) |
| stations with picks | 646 (286 OBS, 360 land) |
| OBS picks | P 79,530, S 107,019 (31% of all S) |
| events with at least one OBS pick | 11,955 (51%) |
| picks rejected by QC (kept in picks_rejected.csv with a reason) | 82,664 |

Outputs in `/wd1/mdenolle_data/tomo_v1/b1_dataset/`:

* `events.csv`: ID, origin_time (Track A iter1 maximum likelihood), lat, lon, depth_bsl_km, tier, nphs, gap, rms,
  h_max, sig_z, region, nP, nS (after QC), gap_qc (recomputed from kept picks), below_DB003_after_qc, dist_min,
  x_km, y_km (Lambert frame), cell_n (events in its 3 km cell before declustering).
* `picks.csv`: event_ID, station, network, phase, pick_time (UTC, absolute), travel_time_s (pick minus Track A
  origin time), sigma_s, pick_prob, iter1_res_s, iter1_weight, iter1_station_term_s, iter0_res_s, epi_dist_km,
  sta_lat, sta_lon, sta_elev_km (+up), is_obs.
* `stations.csv`: station, network, lat, lon, elevation_km (+up, from the station table), zsrce_km (GTSRCE depth,
  +down, as used for the Track A time grids), is_obs, n_P, n_S.
* `picks_rejected.csv` (same columns + `qc` reason), `events_below_threshold.csv` (with decision),
  `station_terms_analysis.csv`, `residual_threshold_alternatives.csv`, `cache/phases_iter{0,1}_vpvs176.parquet`.

Figures in `data/tomo_v1/b1_dataset/`: `b1_map_events.png` (events by depth, stations), `b1_depth_by_region.png`,
`b1_ray_coverage.png` (straight event-station paths sampled every 2 km, number of distinct paths per 0.25 deg cell,
P and S), `b1_picks_per_station.png`, `b1_station_terms_obs_land.png`.

## 1. Event selection (D-B003, exactly as adopted)

Hypocentres: the catalog CSV `catalog_v5_trackA_iter1_vpvs176.csv` is written with 5 significant digits
(longitude -121.71 means +-0.005 deg, ~400 m). The script therefore takes time, lat, lon and depth of the same
NLLoc solutions at full precision from `runs/iter1_vpvs176/locations.csv`. Tier, nphs, gap, RMS come from the
catalog. (Worth fixing in `a2_build_catalog.py`: drop `float_format="%.5g"` for coordinates.)

Declustering frame: NLL Lambert conformal conic (WGS-84, lat_0 45, lon_0 -125.8, standard parallels 41 and 49),
x and y in km, and depth below sea level in km. Cell index = floor(x/3), floor(y/3), floor(z/3). In each cell the
events are ranked by nphs (NLLoc phases used, descending), then RMS (ascending), then ID (ascending); the first 3
are kept. This is deterministic.

| step | events |
|---|---|
| catalog v5 Track A | 172,340 |
| tier A-C | 129,589 |
| + nphs >= 12 | 60,588 |
| + gap <= 200 | 47,668 |
| + rms <= 0.5 | 47,605 |
| decluster 3 km x 3 (14,831 occupied cells) | 23,549 |
| after pick QC and the post-QC rule (section 2e) | 23,428 |

Events by region and tier (final):

| region | A | B | C | all | P picks | S picks |
|---|---|---|---|---|---|---|
| Puget/Georgia | 1,444 | 1,719 | 1,092 | 4,255 | 84,751 | 82,852 |
| Blanco | 158 | 778 | 3,280 | 4,216 | 41,524 | 43,377 |
| Other onshore | 1,599 | 1,370 | 1,144 | 4,113 | 69,736 | 63,079 |
| Mendocino | 2,137 | 1,250 | 664 | 4,051 | 53,383 | 55,858 |
| Gorda | 94 | 1,101 | 1,358 | 2,553 | 22,895 | 31,215 |
| Offshore forearc | 675 | 1,039 | 599 | 2,313 | 28,739 | 29,298 |
| Cascades | 943 | 454 | 421 | 1,818 | 41,740 | 42,362 |
| Explorer | 0 | 0 | 71 | 71 | 508 | 443 |
| Axial | 10 | 18 | 1 | 29 | 212 | 174 |
| JdF ridge/plate | 0 | 0 | 9 | 9 | 56 | 44 |
| all | 7,060 | 7,729 | 8,639 | 23,428 | 343,544 | 348,702 |

Depth: 5/50/95th percentiles 4.1 / 20.2 / 38.0 km bsl. Things to keep in mind for B2:

* Axial (29 events) and the JdF plate interior (9) are almost absent: Axial events are mostly tier D or have gaps
  > 200 deg. The Axial/Endeavour 1D models of the starting model will not be updated by data there.
* Blanco is 78% tier C (no station within 2 focal depths) and its depth histogram reaches 30 km, which is deeper
  than expected for an oceanic transform. Those depths come from the Track A model; B2 relocates them, but they
  add depth-velocity trade-off offshore. A test inversion without Blanco tier C (3,280 events) would show how much.

## 2. Picks and pick QC

Source: `data/catalog_v4/Cascadia_picks_catalog_v4.csv`, 774,910 picks for the 23,549 declustered events. All 696
Track A stations are in the GTSRCE list, so no pick is lost to the station filter. Each pick is matched to its
iter1 and iter0 NLLoc PHASE line on (event, station, phase, pick time rounded to 0.1 ms).

Sigma (D-B008): P 0.05 / 0.10 / 0.15 s for pick_prob >= 0.6 / 0.3-0.6 or missing / < 0.3; S twice P.
Kept picks per class: P 126,281 / 145,137 / 72,126; S 77,468 / 177,332 / 93,902. pick_prob is missing for 26.6%
of kept picks (2011-2015), which all go to the middle class.

Filters, applied in this order (first reason wins; each pick removed once):

| filter | picks | P | S | of which OBS S | events lost |
|---|---|---|---|---|---|
| no_nll: pick absent from the iter1 hyp file | 344 | 136 | 208 | 166 | 0 |
| w0: NLLoc EDT weight 0 in iter1 | 72,868 | 42,349 | 30,519 | 12,097 | 0 |
| duplicate event/station/phase | 0 | 0 | 0 | 0 | 0 |
| res: residual outlier, P | 4,917 | 4,917 | 0 | 0 | 0 |
| res: residual outlier, S | 1,448 | 0 | 1,448 | 847 | 0 |
| sp: S-P inconsistent (S pick dropped) | 713 | 0 | 713 | 253 | 0 |
| event below D-B003 after QC, dropped (section 2e) | 2,374 | 1,170 | 1,204 | 277 | 121 |
| total rejected | 82,664 | | | | 121 |

### 2a. Weight 0 and residual outliers

**Weight 0.** In Track A each 25 km tile is located in its own box (tile +- 75 km); stations whose ±250 km time
grid does not cover that box get weight 0 and no residual (distance column 0). They are far stations: epicentral
distance 5/25/50/75/95th percentiles 167 / 188 / 216 / 268 / 534 km, 31% beyond the 250 km Track A cap, 66 picks
closer than 150 km. They are 9.4% of the picks. They are dropped here because nothing has checked them. They are
kept in `picks_rejected.csv` (qc = w0): B2 can re-admit the ones within ~250 km after checking them against the
TomoATT forward times of the starting model, which would add long, deep rays (the current kept set has 95% of
picks within 152 km).

**Residual outliers.** Reference distribution: iter0 residuals (no station terms) of the used picks (w > 0) of the
selected events. Robust centre and spread per phase: P median 0.014 s, sigma_MAD = 1.4826 MAD = 0.207 s;
S median -0.101 s, sigma_MAD = 0.450 s. Rule: drop when |iter1 residual - median| > 4 sigma_MAD, that is
0.83 s for P and 1.80 s for S. The tested residual is the iter1 one, which includes the Track A station terms and
the iter1 location.

Why test the iter1 residual (with terms) against the iter0 spread: the tomography sees no station terms
(D-B006), so its data spread is the iter0 spread, and that sets the width. But a pick should be called an outlier
only if it disagrees with the other picks of its own station; a station with a systematic +1.5 s S delay
(sediment) is structure, not an outlier. The iter1 residual has the station's systematic delay removed, so it
isolates bad picks; testing the iter0 residual instead cuts preferentially at OBS with large delays.

Alternatives (fraction of used picks dropped; full table in `residual_threshold_alternatives.csv`):

| phase | sigma from | residual tested | k | threshold (s) | drop % | OBS % | land % |
|---|---|---|---|---|---|---|---|
| P | iter0 | iter1 (chosen) | 4 | 0.83 | 1.41 | 2.74 | 1.00 |
| P | iter0 | iter1 | 5 | 1.03 | 0.58 | 1.44 | 0.32 |
| P | iter0 | iter0 | 4 | 0.83 | 3.04 | 4.81 | 2.49 |
| P | iter0 | iter0 | 5 | 1.03 | 1.38 | 2.66 | 0.98 |
| P | iter1 | iter1 | 4 | 0.68 | 2.75 | 4.64 | 2.17 |
| S | iter0 | iter1 (chosen) | 4 | 1.80 | 0.41 | 0.78 | 0.25 |
| S | iter0 | iter1 | 5 | 2.25 | 0.12 | 0.23 | 0.07 |
| S | iter0 | iter0 | 4 | 1.80 | 1.94 | 3.31 | 1.34 |
| S | iter0 | iter0 | 5 | 2.25 | 0.66 | 1.29 | 0.39 |
| S | iter1 | iter1 | 4 | 1.26 | 1.93 | 3.06 | 1.42 |

k = 4 instead of 5: TomoATT minimises a least-squares misfit, so a handful of 1 s P outliers matter more than a few
percent fewer data; at k = 4 the cut still removes only 1.4% of P and 0.4% of S. OBS picks are cut about twice as
often as land picks under every option (higher noise, fewer picks per station for the terms).

### 2b. Duplicates

None: catalog v4 has no event with two picks of the same phase at the same station among the selected events.
The check (keep highest pick_prob, then earliest) stays in the script.

### 2c. S-P consistency

For each event-station pair with both P and S still kept (258,699 pairs) two tests; when either fails the S pick is
dropped (the P pick is the more reliable one):

* tS/tP outside [1.4, 2.5] with tS, tP measured from the Track A origin: 104 pairs. The 0.1/50/99.9th percentiles
  of tS/tP are 1.52 / 1.74 / 2.19, so the window only catches mislabelled phases. No pair has S at or before P.
* S-P differential residual (iter1 location, no terms: res_S - res_P) minus the station median of that quantity,
  beyond 5 sigma_MAD (sigma_MAD = 0.246 s, threshold 1.23 s). The station median absorbs a station's sediment
  delay; the difference does not depend on origin time, so it catches S picks that the location step absorbed.

Total 713 S picks (0.2% of S), 253 of them OBS.

### 2d. S picks per event (after QC)

| S picks | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | >= 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| events | 30 | 40 | 66 | 141 | 345 | 787 | 1,628 | 2,085 | 2,123 | 1,875 | 14,308 |

No minimum is required (`min_S_per_event: 0`). Events with 0-2 S picks are 136 (0.6%); they still carry P
information and are relocated. Requiring >= 3 S would remove 136 events (mostly onshore: Other onshore 74,
Cascades 36), 1,846 P and 172 S picks. Their P picks are in the best-covered part of the model, so either choice
changes little; keeping them avoids a rule that has no clear benefit.

### 2e. Events below D-B003 after pick QC

Pick QC moves 405 events below nphs >= 12 or gap <= 200 deg (gap recomputed from the kept picks' stations;
it agrees with NLLoc's gap to a median 0.06 deg, std 5 deg). Proposed rule: keep the event if it still has
>= 10 picks (the Track A tier A minimum) and gap <= 200 deg; otherwise drop it.

| reason | decision | events |
|---|---|---|
| npick < 12, but >= 10 and gap <= 200 | keep | 284 |
| npick < 10 | drop | 6 |
| gap > 200 | drop | 93 |
| npick < 12 and gap > 200 | drop | 22 |

Kept events are flagged `below_DB003_after_qc = True` in events.csv; all 405 are listed in
`events_below_threshold.csv` with their decision. Alternatives (events dropped): strict 12 picks / 200 deg, 405
(123 of them in Gorda, which loses its edge events); 10 / 220 deg, 72; 8 / 250 deg, 37. Dropped events are not
replaced by the 4th-ranked event of their cell (would change D-B003's rule; the loss is 0.5%).

## 3. Stations and OBS (D-B006)

OBS are identified by the `seafloor` flag of `picks_share/stations/stations_ver5_v6.csv`, the table
`a1_build_model.py` built the GTSRCE file from. The GTSRCE elevation column is 0.0 for all stations; the depth
column zSrce (+down) is -elevation, or 50 m below the grid surface where the 1-arcmin surface is lower than the
station. zSrce > 0.1 km gives the same set except 2 coastal stations. 333 OBS among the 696 Track A stations,
286 of them with picks in this dataset.

Data: iter0 residuals of used picks of well-located events (LOCATED, nphs >= 8, gap <= 270, RMS <= 0.6, the
a2_station_terms.py criteria), and the Track A cumulative terms after iter1 (`runs/iter1_vpvs176/station_terms.csv`,
delay = iter0 term + iter1 median residual; n >= 10).

| group | stations | with S term | median P term | median S term | P term p5/p95 | S term p5/p95 | abs(S term) > 0.5 | > 1.0 | corr(P, S terms) | slope S on P |
|---|---|---|---|---|---|---|---|---|---|---|
| land | 363 | 333 | +0.04 | -0.13 | -0.24 / +0.42 | -0.73 / +0.47 | 56 | 8 | 0.65 | 1.41 |
| OBS | 333 | 255 | +0.01 | +0.01 | -0.28 / +0.46 | -1.26 / +0.99 | 85 | 29 | 0.53 | 1.23 |

The 141 stations with |S term| > 0.5 s are listed in `b1_tables.md` (85 OBS, 56 land). Largest: W01.7A -2.35,
J35A.7D -2.15, M03C.7D -2.14 (P -1.01), W02.7A -1.99, FN13C.7D -1.97, J45C.7D +1.87, FC03D.7D +1.83,
NC89.NV +1.70 (P +0.10), D04D.TA / D04E.TA +1.63 (land, coastal), FS15D.7D +1.61.

OBS S terms by water depth (GTSRCE zSrce):

| water depth | OBS | S picks kept | median S term | median P term | S term > +0.5 | S term < -0.5 | no S term |
|---|---|---|---|---|---|---|---|
| < 0.5 km (shelf) | 46 | 7,223 | +0.55 | +0.06 | 16 | 1 | 15 |
| 0.5-1.5 km | 35 | 15,514 | +0.07 | -0.01 | 7 | 3 | 6 |
| 1.5-2.5 km | 41 | 14,774 | -0.05 | +0.03 | 7 | 10 | 5 |
| > 2.5 km | 164 | 69,508 | -0.07 | -0.01 | 5 | 31 | 17 |

(OBS with picks in the dataset, 286.)

What this shows:

* Sediment signature on the shelf: shelf OBS have S delays of +0.5 to +1.6 s with P delays near zero. Among the
  stations with |S term| > 0.5 s whose P and S terms have the same sign, the median S/P term ratio is 5.5 for OBS
  and 3.5 for land, and 81% of the OBS (62% of land) have |S term| > 3 |P term|. A delay ratio of 5 or more needs
  material with a high Vp/Vs, i.e. unconsolidated sediment that the model (CRESCENT Gen0, no sediment layer
  offshore) does not have.
* Deep-water OBS (> 1.5 km, mostly J-, G- and GB- sites on the Juan de Fuca plate) have the opposite: 41 of 205
  have S terms below -0.5 s, down to -2.3 s, i.e. S arrives earlier than predicted. A sediment layer cannot
  produce that. Candidates: Vs in the top kilometres of the model is too low under the abyssal plain (the Gen0
  columns were extended upward from their shallowest solid value; the young-ocean 1D west of 128.5 W shows no such
  bias, median S term +0.01 for the 54 OBS there), or the picker picks the Sp conversion at the sediment-basement
  contact instead of S on the OBS. This needs a look at waveforms of a few stations (W01.7A, J35A.7D, M03C.7D)
  before B2.
* OBS S picks at stations with |S term| > 0.5 s: 36,700 (34% of OBS S); > 1.0 s: 8,432. 92 kept OBS S picks are
  at stations without an S term (fewer than 10 residuals). Land: 34,185 S picks (14%) at stations with
  |S term| > 0.5 s.
* Most of the OBS S delay is a station constant that the inversion grid (several km) cannot put anywhere except
  in the nodes around the station, where it would also bend P/S structure of the forearc.

Options:

| option | effect | numbers |
|---|---|---|
| (i-a) fixed per-station S correction from Track A, OBS only: subtract the iter1 cumulative S term from OBS S travel times; land and all P uncorrected | removes the sub-grid sediment/shallow signal; keeps all OBS S | 243 OBS in the dataset with a term, 106,927 S picks corrected; 92 S picks at stations without a term (drop or leave uncorrected) |
| (i-b) per-station S static solved jointly in B2 (damped, started at the Track A value) | lets the correction follow the model update; trades off with shallow Vs under the station | requires station-static support in TomoATT (to check in notes/tomoatt_build.md); 255 extra unknowns |
| (ii) sediment layer in the starting model | physically right for the shelf, but needs a thickness map (e.g. GlobSed) and Vs/Vp of sediments, and a grid of ~100 m near the seafloor; does not explain the negative deep-water terms (41 stations) | 0.1-2 km layer below a several-km grid: not resolvable |
| (iii) drop OBS S | loses 107,019 S picks (31% of S) and nearly all Vs information offshore, where half of the events are | 51% of events have OBS picks |

## 4. Mapping of sigma to TomoATT weights (D-B008)

Proposal, to be checked against TomoATT's misfit definition when the format converter is written: datum weight
w = (sigma_ref / sigma)^2 with sigma_ref = 0.05 s, i.e. P 1.0 / 0.25 / 0.11 and S 0.25 / 0.0625 / 0.028. If
TomoATT already normalises weights per phase or per event, use the relative values only.

## Proposed decisions

### D-B009 (proposed) Pick QC for the tomography dataset
**Recommendation.** Keep picks NLLoc used in iter1 (weight > 0); drop |iter1 residual - median| > 4 sigma_MAD
with median and sigma_MAD from the iter0 residuals (P 0.83 s, S 1.80 s); drop S picks with tS/tP outside
[1.4, 2.5] or a station-corrected S-P differential residual beyond 5 sigma_MAD (1.23 s); no duplicates found;
no minimum number of S picks per event.
**Alternatives considered.** Thresholds on the iter0 residual (cuts OBS preferentially: 4.8% of OBS P vs 2.5% of
land at k = 4); k = 5 (keeps 0.8% more P outliers above 0.83 s); requiring >= 3 S per event (-136 events).
Re-admitting weight-0 far picks (72,868, median 216 km) after a forward-time check in B2 is left open.
**Evidence.** Section 2; `b1_tables.md`; `residual_threshold_alternatives.csv`. Effect: -9.4% picks (w0),
-0.9% (res + sp), 0 events from pick filters themselves.

### D-B011 (proposed) Events falling below D-B003 after pick QC
**Recommendation.** Keep an event with >= 10 kept picks and gap <= 200 deg (284 kept, flagged); drop the others
(121), recorded in `events_below_threshold.csv`; no replacement within the cell.
**Alternatives considered.** Strict 12 / 200 (-405 events, 123 in Gorda); 10 / 220 (-72); replacement by the next
event of the cell (changes the D-B003 rule).
**Evidence.** Section 2e.

### D-B010 (proposed) OBS S correction (exception to D-B006)
**Recommendation.** Option (i-a) for B2 iteration 1: subtract the Track A iter1 S term from OBS S travel times
(OBS only, S only); land stations and all P picks stay uncorrected as D-B006 says. Move to (i-b), joint damped
station statics starting at these values, if TomoATT supports them. Drop the 92 OBS S picks at stations without
an S term. Before B2, look at waveforms of the strongest negative-term deep-water OBS (W01.7A, J35A.7D, M03C.7D,
W02.7A) to tell a too-slow shallow model from Sp picked as S.
**Alternatives considered.** (ii) sediment layer: cannot be represented on the inversion grid and cannot explain
the 41 deep-water stations with S terms below -0.5 s; (iii) drop OBS S: -107,019 S picks (31%) and the offshore
Vs constraint.
**Evidence.** Section 3; `station_terms_analysis.csv`; `b1_station_terms_obs_land.png`: shelf OBS median S term
+0.55 s with P +0.06 s; OBS |S|/|P| term ratio median 5.5 (land 3.5); 85 OBS with |S term| > 0.5 s, 29 > 1 s.
