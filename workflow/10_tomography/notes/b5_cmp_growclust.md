# B5 code comparison (D-B078): the GrowClust3D leg

2026-11-25. Branch `tomo/cmp-growclust` (from `origin/tomo/b5` cc09cb37). Driver `b5_cmp_growclust.py`, on the common
frame of `b5_cmp_common.py` (not edited) and D-B048's `b5_growclust_run.jl` (not edited). Runs, inputs and logs:
`/wd1/mdenolle_data/tomo_v1/b5/cmp/growclust/<set>/<run>/` (`reloc.csv`, `run.json`, `detail.csv`, `in/`, `out/`);
scores `scores.json` (score_set), `scores_same_events.json`, `diag.json` in the same folder.

## 1. Short answer

GrowClust3D on the CC times fits the training stations almost exactly (median absolute residual 1-5 ms, against 4-37 ms
at the v5.1 positions) but predicts the held-out stations worse than v5.1 in rms on all four sets. Only at Mount St.
Helens (88 stations) does it beat v5.1 on the bulk of the held-out data: MAD 12 ms against 24 ms for P, 8 ms against 32
ms for S. There the rms is still worse (107 against 80 ms for P) because of a tail. The tail comes from the events
GrowClust3D moves by more than 1 km: their held-out misfit grows with the shift on every set, while the events it moves
less than 1 km fit the held-out stations better than v5.1 does on the same rows (lower MAD). On the three offshore sets
(7-17 OBS, 4-14 left per fold) the held-out rms is 1.7-3.3 times the v5.1 value on the same events, and only jdf P
has a better MAD. GrowClust3D also thickens every cluster (smallest principal axis 0.36 against 0.20 km at Axial, 1.00
against 0.60 km at Blanco) and puts 88 Axial, 29 Blanco, 40 jdf and 1 MSH events above the solid surface (v5.1: 0).

The damage comes from weakly linked events. The only parameter changed from D-B048 is ngoodmin (8 -> 0), because with
8 only 17-40 % of the events have a usable pair. A run with D-B048's 8 (section 4.5) relocates 208 Axial, 214 Blanco,
576 MSH and 41 jdf events; on those it beats v5.1 at the held-out stations (MSH P MAD 6 against 18 ms, S 5 against
28 ms; rms lower too), puts none above the surface outside jdf (4), and thins the MSH cloud. The main run places those
same events equally well; what it adds, the events without any pair of 8 or more CC dt, it places worse than v5.1.

## 2. What was run

* Code: GrowClust3D.jl 0.1.0 (git 592f6179, git-tree-sha1 2f35f37e), Julia 1.13.4, the D-B048 environment and depot
  (params.yaml `b5_test.growclust`), D-B048's driver `b5_growclust_run.jl` (the package's example driver with the
  algorithm constants read from `aux.txt`), 16 threads, one run at a time, nice 10.
* Travel times: the NonLinLoc grids of catalog v5.1, `/wd1/mdenolle_data/reloc_v5/time_b4_v0.3.1m_sc21`
  (`cascadia3d.<P|S>.<STA>.time`; LAMBERT WGS-84, lat0 45, lon0 -125.8, parallels 41/49; GrowClust3D checks the frame
  against every header). Grid depth axis z' (Earth-flattened), so start depths go in as z' = 6371 ln(6371 / (6371 - z))
  and come out as z = 6371 (1 - exp(-z'/6371)). Table box: event box +- 40 km, z' from -5 to 80 km (the grids' range);
  linear interpolation, flat extrapolation above the top. Every station of the four sets has both grids (no row
  dropped; the check is in the code).
* Start: the v5.1 hypocentres and origin times of `events(s)`. Magnitude 0 (not used).
* Data: `cc_rows(s)`, written as GrowClust's xcordata with tdif = dt_cc = tt1 - tt2 (tdif_fmt 12), rxcor = cc_max,
  origin-time correction 0 in every pair header. Sign checked: at the v5.1 positions dt_cc correlates positively with
  the predicted tt1 - tt2 on all four sets (0.25 to 0.71). No catalog dt (section 4.1: CC-only links 92-100 % of the
  events in the full runs).
* Folds: run foldK drops every CC row at `held_out(s, "foldK")`. Events with no row left are singletons of that run.
* Output: `reloc.csv` holds the events in a cluster of at least 2 events (nbranch_min); GrowClust3D leaves singletons at
  their start, so they are left out and listed in `run.json` (`unrelocated_ids`). out.cat precision: 1e-5 deg, 1 m.
* Scoring: `score_set("growclust", s)` from `b5_cmp_common.py`, saved as `scores.json`. Its v5.1 reference uses every
  event of the set, while the GrowClust3D score uses only the events it relocated; `scores_same_events.json` scores the
  v5.1 positions on the same events per fold.

## 3. Parameters

D-B048 variant `default` (the package example values for cross-correlation data) and the D-B048 `inp` block, with one
change.

| parameter | value | note |
|---|---|---|
| distmax / distmax2 | 8 / 6 km | catalog distance / relocated distance to join clusters |
| hshiftmax / vshiftmax | 2 / 2 km | cluster shift limits per merge |
| rmsmax / rmedmax | 0.2 / 0.1 s | misfit limits per merge |
| maxlink / nbeststa | 12 / 24 | event pairs per merge / CC values in the merge-order factor |
| boxwid / nit / irelonorm | 3 km / 15 / 1 (L1) | grid search |
| torgdifmax / tdifmax | 10 / 30 s | |
| rmin / rmincut / rpsavgmin | 0 / 0 / 0 | every cc_max is >= 0.70, so these cut nothing |
| delmax | 400 km | every station is inside |
| **ngoodmin** | **0 (D-B048: 8)** | changed, see below |
| nbranch_min | 2 | |
| nboot | 100 | bootstrap MADs in `detail.csv`; not used by the scores |

Why ngoodmin changed: D-B048 took 8 from its catalog-dt rule (8 common station-phases per pair). The CC pairs have a
median of 2 dt (90th percentile 4 to 9) because the CC session keeps only rows above its threshold. With 8, only 208 of
1,200 (Axial), 214 of 1,044 (Blanco), 578 of 1,436 (MSH) and 41 of 162 (jdf) events have a usable pair. 0 is the value
in the package's own example control files (`examples/example.*.inp`). GrowClust3D still orders the merges by the mean
of the best 24 cc_max values of a pair padded with zeros, so pairs with many good dt merge first. Section 4.5 shows the
run with 8.

## 4. Results

### 4.1 Events per run

Rows and pairs after the fold cut; clusters = clusters of >= 2 events (largest); singletons (with a usable pair /
without); GrowClust3D's own dt rms (P / S, ms, pairs inside one cluster, with its origin-time shifts); wall time.

| set | run | stations out | CC rows | pairs | events | relocated | clusters | singletons | rms P / S | s |
|---|---|---|---|---|---|---|---|---|---|---|
| axial | full | 0 | 22,346 | 10,377 | 1,200 | 1,169 | 5 (1,158) | 31 (25 / 6) | 76 / 107 | 63 |
| axial | fold0 | 3 | 9,563 | 6,201 | 1,200 | 969 | 22 (925) | 231 (219 / 12) | 140 / 216 | 48 |
| axial | fold1 | 3 | 17,826 | 9,433 | 1,200 | 1,124 | 6 (1,114) | 76 (69 / 7) | 84 / 129 | 63 |
| axial | fold2 | 3 | 17,303 | 9,699 | 1,200 | 1,136 | 5 (1,128) | 64 (57 / 7) | 86 / 132 | 63 |
| blanco | full | 0 | 22,703 | 9,367 | 1,044 | 990 | 7 (969) | 54 (54 / 0) | 65 / 77 | 60 |
| blanco | fold0 | 9 | 10,513 | 6,993 | 1,044 | 790 | 33 (406) | 254 (247 / 7) | 84 / 130 | 48 |
| blanco | fold1 | 8 | 22,176 | 9,315 | 1,044 | 979 | 9 (944) | 65 (65 / 0) | 61 / 78 | 60 |
| blanco | fold2 | 8 | 12,718 | 7,094 | 1,044 | 833 | 28 (418) | 211 (208 / 3) | 66 / 105 | 57 |
| msh | full | 0 | 50,389 | 12,627 | 1,436 | 1,328 | 2 (1,326) | 108 (104 / 4) | 69 / 36 | 89 |
| msh | fold0 | 36 | 29,162 | 10,146 | 1,436 | 1,207 | 6 (1,196) | 229 (217 / 12) | 70 / 40 | 82 |
| msh | fold1 | 36 | 36,251 | 11,164 | 1,436 | 1,229 | 11 (1,195) | 207 (195 / 12) | 62 / 48 | 77 |
| msh | fold2 | 36 | 35,365 | 11,394 | 1,436 | 1,284 | 14 (1,231) | 152 (144 / 8) | 73 / 38 | 73 |
| jdf | full | 0 | 7,008 | 1,981 | 162 | 162 | 1 (162) | 0 | 57 / 54 | 37 |
| jdf | fold0 | 3 | 6,400 | 1,978 | 162 | 162 | 1 (162) | 0 | 50 / 58 | 44 |
| jdf | fold1 | 3 | 5,469 | 1,944 | 162 | 160 | 1 (160) | 2 (2 / 0) | 62 / 33 | 46 |
| jdf | fold2 | 3 | 2,147 | 1,297 | 162 | 148 | 5 (136) | 14 (14 / 0) | 60 / 71 | 44 |

Most singletons have a usable pair but every merge they were offered failed a limit (rms, median residual, shift or
distance). Each set ends in one dominant cluster in the full run.

### 4.2 Held-out CC residuals (score_set, pooled over the 3 folds)

Residual of each held-out row minus the median training residual of its pair, in ms: rms / MAD (n). "v5.1" is the
reference of score_set (all events); "v5.1 same" is the v5.1 positions on the events GrowClust3D relocated.

| set | phase | v5.1 | v5.1 same | GrowClust3D |
|---|---|---|---|---|
| axial | P | 72.6 / 28.2 (10,157) | 69.4 / 26.6 (9,273) | 155.3 / 40.6 (9,273) |
| axial | S | 62.4 / 28.5 (4,999) | 60.2 / 27.9 (4,784) | 148.5 / 39.7 (4,784) |
| blanco | P | 94.8 / 43.8 (7,764) | 88.8 / 41.6 (6,516) | 227.9 / 76.0 (6,516) |
| blanco | S | 95.3 / 52.7 (8,859) | 90.9 / 51.0 (7,643) | 301.0 / 112.2 (7,643) |
| msh | P | 80.1 / 23.5 (31,042) | 75.8 / 23.1 (30,187) | 106.7 / 12.1 (30,187) |
| msh | S | 63.8 / 31.6 (13,043) | 60.9 / 31.4 (12,785) | 86.0 / 7.9 (12,785) |
| jdf | P | 121.3 / 70.1 (2,829) | 119.4 / 69.6 (2,757) | 202.6 / 48.6 (2,757) |
| jdf | S | 162.2 / 95.4 (2,779) | 159.0 / 93.4 (2,690) | 306.5 / 99.7 (2,690) |

90th percentiles of |residual| (ms), v5.1 / GrowClust3D: axial P 118 / 239, S 101 / 225; blanco P 141 / 381, S 143 /
503; msh P 95 / 134, S 92 / 71; jdf P 192 / 348, S 260 / 513.

### 4.3 Full runs

| set | relocated | above surface (v5.1: 0) | thickness GrowClust3D / v5.1 (km) | median shift h / z (km) |
|---|---|---|---|---|
| axial | 1,169 / 1,200 | 88 | 0.361 / 0.195 | 0.488 / +0.111 |
| blanco | 990 / 1,044 | 29 | 1.003 / 0.603 | 0.971 / -0.119 |
| msh | 1,328 / 1,436 | 1 | 0.359 / 0.300 | 0.299 / -0.010 |
| jdf | 162 / 162 | 40 | 0.208 / 0.078 | 0.565 / +0.017 |

Thickness = smallest principal axis on the relocated events (v5.1 on the same events); z shift positive down. The
above-surface events sit a median 0.47 km (Axial), 0.57 km (Blanco), 0.18 km (jdf) above the surface, at most 2.7 km
(Axial). At jdf the 40 events were 0.02 km below the surface in v5.1 (median): GrowClust3D spreads a cluster that v5.1
holds on its floor.

### 4.4 Where the held-out misfit comes from

Held-out rows binned by the larger 3D shift (from v5.1) of the pair's two events, pooled over folds
(`diag.json`). Each cell: v5.1 -> GrowClust3D on the same rows, rms / MAD in ms, (rows). Blanco also has 23 rows
with a shift above 5 km (GrowClust3D rms 940 / 593 ms).

| set | phase | <= 0.5 km | 0.5-1 km | 1-2 km | 2-3 km | 3-5 km |
|---|---|---|---|---|---|---|
| axial | P | 44/19 -> 45/12 (517) | 58/20 -> 73/18 (3,088) | 73/30 -> 155/58 (3,969) | 81/36 -> 233/107 (1,435) | 104/51 -> 354/153 (264) |
| axial | S | 45/22 -> 48/19 (387) | 52/24 -> 79/25 (2,197) | 68/33 -> 170/72 (1,689) | 75/42 -> 265/118 (434) | 66/36 -> 387/172 (77) |
| blanco | P | 42/25 -> 25/10 (60) | 85/44 -> 95/23 (583) | 90/42 -> 183/69 (2,453) | 87/41 -> 238/94 (2,594) | 95/42 -> 343/197 (817) |
| blanco | S | 60/39 -> 39/17 (77) | 84/53 -> 109/30 (843) | 88/51 -> 222/94 (3,379) | 97/51 -> 331/156 (2,562) | 92/52 -> 548/366 (768) |
| msh | P | 65/19 -> 64/11 (11,266) | 76/25 -> 82/12 (12,134) | 84/27 -> 147/15 (5,981) | 132/45 -> 305/145 (764) | 80/45 -> 586/314 (42) |
| msh | S | 54/27 -> 44/8 (3,745) | 61/33 -> 50/8 (5,794) | 66/35 -> 112/8 (3,102) | 101/63 -> 414/198 (141) | 110/10 -> 1794/108 (3) |
| jdf | P | 83/45 -> 59/18 (304) | 115/69 -> 89/25 (877) | 130/75 -> 217/74 (984) | 121/74 -> 308/164 (540) | 130/84 -> 380/279 (52) |
| jdf | S | 107/65 -> 75/23 (216) | 166/99 -> 150/46 (593) | 154/90 -> 262/112 (1,195) | 166/105 -> 432/221 (611) | 230/111 -> 751/454 (75) |

Where GrowClust3D moves both events by less than 1 km, its held-out MAD is lower than v5.1's on the same rows in every
set and phase, and its rms is similar (lower at jdf, higher at Axial). Beyond 1 km its held-out misfit is worse than
v5.1's and grows with the shift, while v5.1's stays flat: the large moves are not supported by the stations left out.
MSH is the exception between 1 and 2 km, where the MAD still drops (P 27 -> 15, S 35 -> 8 ms) but the rms doubles.
At the training stations the L1 grid search fits most dt almost exactly (MAD P / S: axial 2.2 / 4.6 ms, blanco
1.9 / 1.4, msh 2.8 / 2.7, jdf 2.6 / 2.2; v5.1 on the same rows 4.3 / 9.8, 11.6 / 16.7, 10.4 / 21.7, 19.3 / 37.0),
while the training rms changes little or rises (axial S from 31 to 54 ms). With 2 dt per pair and 4 to 14 stations
left in an offshore fold, a merge can move a cluster along a direction those stations do not resolve, and the
held-out stations see the error. MSH keeps 58-59 stations per fold and moves fewer events beyond 2 km (806 of 30,187
held-out P rows), so it gains on the bulk of the data.

### 4.5 D-B048 unchanged (ngoodmin 8), for reference

Same driver with ngoodmin 8 (`--sens-ngoodmin 8`, folder `growclust/_sens_ngoodmin8/`, scored the same way). Not the
leg's result: it was run after the main result to document the change of section 3, and choosing it now would select
on the test.

| set | relocated full / fold0 / fold1 / fold2 | above surface | thickness (v5.1) km | held-out P rms / MAD, v5.1 same -> run (n) | held-out S |
|---|---|---|---|---|---|
| axial | 208 / 11 / 69 / 34 | 0 | 0.168 (0.170) | 25 / 10 -> 22 / 6 (177) | 25 / 13 -> 20 / 7 (132) |
| blanco | 214 / 4 / 170 / 9 | 0 | 0.569 (0.614) | 61 / 34 -> 49 / 7 (66) | 65 / 49 -> 20 / 6 (57) |
| msh | 576 / 300 / 408 / 415 | 0 | 0.180 (0.299) | 57 / 18 -> 53 / 6 (14,323) | 52 / 28 -> 30 / 5 (8,230) |
| jdf | 41 / 20 / 0 / 0 | 4 | 0.048 (0.076) | 2 rows | 13 rows |

jdf fold1 and fold2 have no pair with 8 dt; GrowClust3D stops on such input ("0 unique stations"), so the driver
writes an empty reloc.csv for them. With ngoodmin 8 GrowClust3D relocates 17 % (Axial) to 40 % (MSH) of the events in
the full runs, and on those events it beats v5.1 at the held-out stations in rms and MAD on every set; only the MSH
sample is large.

The main run (ngoodmin 0) places these same well-linked events as well (`diag.json`, `_events_of_ngoodmin8_run`;
MSH held-out P 51.6 / 5.7 ms, S 28.2 / 5.2 ms, against 52.8 / 5.9 and 30.1 / 5.3 with ngoodmin 8 and 57.3 / 18.3 and
51.6 / 28.2 at v5.1). So lowering ngoodmin does not spoil the well-linked events: the held-out damage of section 4.2
comes from the events that only have pairs with fewer than 8 CC dt, which ngoodmin 0 lets GrowClust3D merge and move.

## 5. Caveats

* The CC files were rewritten by the CC session between 08:11 and 08:20 PST (run_final.sh with the clock-epoch step,
  whose `straddles_clock_epoch` column removes 117 MSH rows at MH07.XD and MG07.XD). The first pass of MSH fold2 read
  the file in between, without that column (35,426 input rows instead of 35,365). All 16 runs were redone after 08:20;
  `run.json` holds the md5 of the events and CC files used (cc_axial 6e8e4195, cc_blanco 9430ffda, cc_msh 63c9f2d9,
  cc_jdf 287e1f6c). The other 14 runs, whose inputs had not changed, came out byte-identical: GrowClust3D is
  deterministic on the same input.
* `run.json` records git_commit cc09cb37 (the base): the driver was committed after the runs. Its relocation code did
  not change after the runs; later edits only added the diagnostics and the empty-output case of section 4.5.
* The scorer and GrowClust3D use the same frame: with GrowClust3D's own origin-time shifts, the scorer's grid lookup at
  the GrowClust3D positions gives its logged dt rms exactly for MSH (69 / 36 ms) and jdf (57 / 54 ms); Blanco differs by
  5 ms because the log counts only pairs inside one cluster.
* GrowClust3D keeps each cluster's centroid at the mean of its start positions; it cannot move a cluster as a whole.
  The scorer removes each pair's median training residual, so common shifts do not count anyway.
* GrowClust3D's misfit is an unweighted L1 norm; cc_max only orders the merges. No data weights reach the misfit.
* No surface constraint (none was added). GrowClust3D only refuses a merge whose relocated cluster centroid would lie
  shallower than min(0, shallowest start z'), which does not hold single events below the seafloor.
* Singletons are not in reloc.csv, so the held-out sample is smaller than the v5.1 reference's (`scores_same_events`
  gives the like-for-like v5.1 number; it does not change the conclusions).
* Parameters were not tuned on the held-out scores; ngoodmin 0 was set before any score was seen. Values between 1 and
  7, or tighter shift and rmedmax limits, were not tried: choosing them on these scores would select on the test. If
  the PI wants GrowClust3D with a stricter link rule, it should be set on other clusters first.
* The bootstrap (100 resamples) is run as in D-B048 but not used here.

## 6. Rerun

    cd workflow/10_tomography
    PY="env LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
    $PY b5_cmp_growclust.py --force --score              # 16 runs (~17 min at 16 threads), scores.json, scores_same_events.json
    $PY b5_cmp_growclust.py --sens-ngoodmin 8 --score    # section 4.5, under growclust/_sens_ngoodmin8/ (~10 min)
    $PY b5_cmp_growclust.py --diag                       # diag.json (sections 4.4, 4.5)
