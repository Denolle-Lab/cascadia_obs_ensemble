# B5 code comparison (D-B078): the NLL-SSST-coherence leg

2026-11-25. Branch `tomo/cmp-nll` (from `origin/tomo/b5` cc09cb37). Code `workflow/10_tomography/b5_cmp_nll.py`
on the common frame `b5_cmp_common.py` (unchanged). Outputs `/wd1/mdenolle_data/tomo_v1/b5/cmp/{nllbase,nllssst,
nllcoh}/<set>/<run>/{reloc.csv,run.json}`, scores `/wd1/mdenolle_data/tomo_v1/b5/cmp/nllcoh/scores.json`, scratch
`/wd1/mdenolle_data/tomo_v1/b5/cmp/nll_work/<set>/<run>/` (per-pass hyps, SSST grids, oct-trees, logs).
Method: Lomax & Savvaidis (2022, JGR 127, e2021JB023190) as implemented in NonLinLoc v7.1.07 (the reloc_v5 build)
and in Lomax's Parkfield example (Zenodo 10.5281/zenodo.18833124, v9.0: `run_ssst_relocations.bash`,
`run_coherence.bash`), which is where the run parameters below come from when the paper is not cited.

## 1. Short answer

All 16 set-runs ran (4 sets x full, fold0-2). Every event is located in the full runs and none is above the merged
surface (the LOCTOPO_SURFACE floor holds in all three legs). Held-out CC scores, P / S MAD in ms, pooled over the
three folds (full table in section 3):

| set | v5.1 on the same events | nllbase | nllssst | nllcoh |
|---|---|---|---|---|
| axial | 28 / 28 | 51 / 78 | 56 / 91 | 33 / 41 |
| blanco | 43 / 54 | 149 / 191 | 151 / 221 | 60 / 79 |
| msh | 23 / 31 | 44 / 54 | 49 / 60 | **18 / 19** |
| jdf | 68 / 93 | 124 / 160 | 121 / 200 | **65 / 79** |

* The coherence step does the work. On the same training picks it cuts the held-out MAD of the NLLoc baseline by
  35-60 % (P) and 47-65 % (S) in every set. In msh and jdf its MAD is below that of v5.1, whose positions used the
  held-out stations; its rms is not (msh P 86 vs 79 ms, jdf P 139 vs 119 ms): a tail of events stays off.
* SSST alone makes the held-out fit worse. The static pass (D = 9999 km) helps a little (P MAD -1 to -24 ms), D = 2 km
  is about neutral, and D = 1 and 0.5 km degrade it in every set while the misfit at the training stations keeps
  falling (median event rms in the full runs, it0 -> it4: axial 77 -> 46 ms, blanco 102 -> 41, msh 116 -> 78, jdf
  130 -> 53; section 4). nllssst ends worse than nllbase, except P in blanco and jdf (within 3 ms).
* Diagnostic, found on the held-out scores (so not a fair entry): the coherence step on top of SSST stopped at
  D = 2 km does better than nllcoh in every set (P / S MAD axial 32 / 38, blanco 50 / 64, msh 17 / 17, jdf 63 / 79;
  msh rms 79 / 56 ms, at or below v5.1 on the same events) and thins axial, blanco and msh further (section 4).
  Coherence with no SSST at all is close behind.
* Axial and blanco stay well behind v5.1 (axial P 33 vs 28 ms, S 41 vs 28 ms; blanco 60 vs 43 / 79 vs 54). This is
  mostly the test design: an absolute code relocating without the held-out stations starts far behind (nllbase vs
  v5.1), while v5.1 and anything that starts from it keep what the held-out stations told v5.1 (section 5).
* Fold runs lose the events left with fewer than 5 training picks (blanco fold0: 477 of 1,044; jdf fold2: 122 of
  162); their CC rows drop out of the NLL scores.
* Full runs: coherence thins msh (smallest-axis thickness 0.32 -> 0.23 km), leaves axial and jdf as v5.1 (0.21 vs
  0.20 km, 0.08 vs 0.08 km) and blanco slightly thicker (0.68 vs 0.62 km); SSST alone thickens every set.
* Cost: 16 processes, 1.9 h wall for the 16 set-runs (msh 12-17 min per run, axial 6-7, blanco 5-10, jdf 1);
  Loc2ssst takes two thirds of it or more, the coherence pass about 8-16 %.

## 2. What was run

Every set (axial 1,200 events, blanco 1,044, msh 1,436, jdf 162) and every run of `runs()` (full, fold0-2) goes
through the same chain. A fold run drops the picks at the fold's held-out stations (`held_out`) and the CC rows at
those stations; nothing else changes.

**Picks and obs files.** `picks(s)` of the run, written as NLLOC_OBS with absolute times = v5.1 origin + tt and a
Gaussian error = `sigma` (the D-B008 classes: P 0.05/0.10/0.15 s, S twice; the same classes as `a2_make_obs.py`).
One block per event, `PUBLIC_ID ev<ID>`; NLLoc names its output files by public ID (`FILENAME_PUBLIC_ID`), so two
events in the same second cannot overwrite each other.

**Time grids.** For each station-phase of the set, the v5.1 grid (`time_b4_v0.3.1m_sc21`, 1 km) if its extent holds
the set's search box, else the wide 2 km grid (`wide_time_b4_v0.3.1m_sc21`); this is the rule NLLoc applies to the
two LOCFILES roots of production. All station-phases come from the fine grids (axial 16, jdf 14, msh 203) except 2
of the 39 in blanco, which come from the wide grids. The grids are
cropped to the box plus one node; node values are unchanged, so NLLoc's trilinear times inside the box are those of
the v5.1 grids. Depths are Earth-flattened (z'); outputs are true depths, z = 6371 (1 - exp(-z'/6371)).

**Search box.** The set's epicentres +- 10 km, z' -5 to 25 km, 1 km LOCGRID nodes (axial 23 x 23 x 31 nodes,
blanco 33 x 39 x 31, msh 27 x 25 x 31, jdf 27 x 31 x 31). Production used the 25 km tile +- 75 km and z' -5 to 80 km;
the smaller box lets the SSST grids be fine (0.25 km) and keeps the oct-tree start cells near the production count.

**Baseline (`nllbase`, pass it0).** The v5.1 control file (`b4_v51_iter1/<tile>/nlloc.in`) with the static terms of
`b4_v51_iter0/station_terms.in`:

```
LOCMETH EDT_OT_WT 9999.0 5 -1 -1 -1 -1 -1.0 1     # production: 20 grids in memory (no effect on the result)
LOCGAU 0.2 0.0                                    # no LOCGAU2 (D-B069)
LOCPHASEID P P p Pg Pn
LOCPHASEID S S s Sg Sn
LOCQUAL2ERR 0.05 0.1 0.15 0.3 99999.9
LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9
LOCANGLES ANGLES_YES 5
LOCTOPO_SURFACE /wd1/mdenolle_data/reloc_v5/topo/topo_15s_cons_merged_eft.bin 0
LOCGRID <box> 1.0 1.0 1.0 PROB_DENSITY SAVE
LOCSEARCH OCT <~2 km start cells, e.g. axial 11 11 15> 0.02 40000 2000 0 1
LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 FILENAME_PUBLIC_ID
INCLUDE /wd1/mdenolle_data/reloc_v5/runs/b4_v51_iter0/station_terms.in
```

Events are run in 16 chunks in parallel. An event needs 5 picks (production rule); in fold runs the events left with
fewer than 5 are skipped by NLLoc and listed in run.json.

**SSST (`nllssst`, passes it1-it4).** Four Loc2ssst passes, each followed by an NLLoc pass with the new grids and no
static terms (Lomax's cumulative scheme: pass k reads the hyps of pass k-1 and the grids T(k-1) and writes
T(k) = T(k-1) + the smoothed mean of residual + delay; the delay of it0 is the b4_v51_iter0 static term, so the first
pass re-estimates the static terms on the set). Characteristic distance D per pass:

| pass | D (km) | role |
|---|---|---|
| Loc2ssst 0 -> it1 | 9999 | static terms of the set (Lomax: CHAR_DIST_INIT = 9999) |
| Loc2ssst 1 -> it2 | 2 | about the extent of the set cores (Lomax: start at 1/2 to 1 x the size of the seismicity) |
| Loc2ssst 2 -> it3 | 1 | halving (Lomax: CHAR_DIST_DIVISOR = 2) |
| Loc2ssst 3 -> it4 | 0.5 | last; it4 saves the oct-tree PDFs |

D stops at 0.5 km because below that the sparse sets have too few events inside D: the median effective number of
other events within D (sum of exp(-d^2/D^2) at v5.1 positions) at D = 0.5 km is 344 for axial, 125 msh, 8 jdf, 6
blanco, and at 0.25 km 82, 20, 2, 1. Lomax's Parkfield example goes 9999, 16, 8, 4 (1 km floor) for a ~30 km
cluster; the ratio D_min / D_start = 1/4 here, 1/16 there. Loc2ssst control (one instance per group of stations,
LSSTATIONS, at most 16 at once):

```
LSGRID    <box at 0.25 km, e.g. axial 89 89 121  -335.00 102.00 -5.00  0.25 0.25 0.25> SSST_TIMECORR FLOAT
LSOUTGRID <same> TIME FLOAT
LSPHSTAT  1.0 0 360.0 1.0 2.0 9999.0     # RMS <= 1 s, |res| <= 1 s P / 2 s S, no Len3 cut
LSPARAMS  <D> 1e-07 0                    # WeightFloor 1e-7 as run_ssst_relocations.bash, REJECTED events not used
LSMODE    ANGLES_NO
LOCFILES  <unused obs> NLLOC_OBS <T(k-1) root> <unused out> 0
LOCMETH   EDT_OT_WT 9999.0 5 -1 -1 -1 20 -1.0 1      # Loc2ssst only reads VpVs = -1 (S grids exist)
```

LSGRID = LSOUTGRID with nodes on the 1 km (and 2 km) nodes of the input grids, so the base times are not altered by
the resampling; 0.25 km = D_min / 2. Lomax's example cuts Len3 > 5 km (LSPHSTAT) and MaxSE3 = 5 km (coherence);
that cut belongs to his LOCGAU 0.05 s. With LOCGAU 0.2 s the 68 % ellipsoids of these events have Len3 2-15 km
(median 2.9 km axial, 7.3 km jdf), so the cut would keep 45 of 162 jdf events; both are off here (Loc2ssst and NLLoc
defaults). A first axial run with the 5 km cuts is kept for comparison (section 4).

**Coherence (`nllcoh`).** Event-pair coherence = mean `cc_max` over the pair's accepted CC rows (`cc_rows(s)`) at the
run's training stations; pairs without rows have no coherence (Lomax takes the maximum over channels of a P+S window
correlation; the mean of per-phase maxima is the closest the CC table allows, and is lower than the maximum). Accepted
rows have cc_max >= 0.7, so pair coherences lie in 0.7-1.0. One file per event:

```
STREAM_COHERENCES
  1.000000 <it4 hyp root of the event>                  # self, coherence 1 (written first, as get_coherence.py)
  0.916851 <it4 hyp root of neighbour 1>                # root = .../c10.20150407.050258_ev138945.grid0.loc
  ...                                                   # NLLoc opens <root>.octree and <root>.hyp
```

Only neighbours LOCATED in it4 are listed (others have no oct-tree). One NLLoc run per event, with the event's own
picks (they give the origin time and residuals; the location PDF is the stack only), the it4 time grids and:

```
LOCPOSTERIOR OCT_TREE <ev>.stream_coherences 1e-30 0.5 -1.0 -1.0 -1 -1.0 -999
LOCSEARCH OCT <~12,000 start cells, e.g. axial 21 21 28> 0.02 50000 2000 0 0
LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 FILENAME_PUBLIC_ID
```

The fields are DefaultValue 1e-30, CoherenceMin 0.5 (`run_coherence.bash` COHERENCE_MIN_NLL), MaxOtherWeight off,
MaxSE3 off (see above), MaxNother off, MaxMagDiff off, MinMag off. NLLoc (`NLLocLib.c` GetNLLoc_PdfGrid,
getLogPdfValue, STACK_POSTERIOR) maps coherence c to weight w = 0 at c <= 0.5, 1 at c >= 0.9, w = 0.5 cos(pi (1 -
(c - 0.5) / 0.4)) + 0.5 between; the posterior at a point is log(sum_i w_i pdf_i) x sum_i w_i over the self and the
neighbour oct-trees, and it replaces the EDT likelihood in the oct-tree search (MFmin/MFmax are -1e30 in the hyps,
the sign the posterior was used). With pair coherences of 0.7-1.0 every neighbour has w >= 0.5. The search follows
`run_coherence.bash` (>= 10^4 start cells so that a broad stacked PDF does not trap the search; no stop at the
minimum node, 50000 nodes); it reaches 16 m leaves.

## 3. Results

Held-out CC scores (`score_set`: residual of each CC row at a held-out station minus the pair's median residual at
the training stations, pooled over fold0-2; rms / MAD in ms, n rows). v5.1 is the reference positions (no
relocation). Rows differ between v5.1 and the NLL legs because events with fewer than 5 training picks are not
located in the fold runs (column "located per fold").

| set | positions | located per fold | P rms / MAD ms (n) | S rms / MAD ms (n) |
|---|---|---|---|---|
| axial | v5.1 (reference) | all | 73 / 28 (10157) | 62 / 29 (4999) |
| axial | v5.1 on the nllcoh events | 1093, 1191, 1188 | 72 / 28 (9523) | 62 / 28 (4802) |
| axial | nllbase | 1093, 1191, 1188 | 116 / 51 (9523) | 181 / 78 (4802) |
| axial | nllssst | 1093, 1191, 1188 | 123 / 56 (9523) | 224 / 91 (4802) |
| axial | nllcoh | 1093, 1191, 1188 | 80 / 33 (9523) | 101 / 41 (4802) |
| blanco | v5.1 (reference) | all | 95 / 44 (7764) | 95 / 53 (8859) |
| blanco | v5.1 on the nllcoh events | 556, 1044, 920 | 93 / 43 (4453) | 99 / 54 (5222) |
| blanco | nllbase | 482, 1044, 915 | 463 / 149 (3933) | 656 / 191 (4771) |
| blanco | nllssst | 557, 1044, 920 | 451 / 151 (4455) | 673 / 221 (5222) |
| blanco | nllcoh | 556, 1044, 920 | 155 / 60 (4453) | 202 / 79 (5222) |
| msh | v5.1 (reference) | all | 80 / 24 (31042) | 64 / 32 (13043) |
| msh | v5.1 on the nllcoh events | 1389, 1408, 1432 | 79 / 23 (30505) | 63 / 31 (12692) |
| msh | nllbase | 1387, 1409, 1432 | 143 / 44 (30507) | 162 / 54 (12692) |
| msh | nllssst | 1389, 1408, 1432 | 172 / 49 (30505) | 199 / 60 (12692) |
| msh | nllcoh | 1389, 1408, 1432 | 86 / 18 (30505) | 66 / 19 (12692) |
| jdf | v5.1 (reference) | all | 121 / 70 (2829) | 162 / 95 (2779) |
| jdf | v5.1 on the nllcoh events | 162, 162, 40 | 119 / 68 (1444) | 162 / 93 (992) |
| jdf | nllbase | 162, 162, 40 | 280 / 124 (1444) | 331 / 160 (992) |
| jdf | nllssst | 162, 162, 40 | 307 / 121 (1444) | 400 / 200 (992) |
| jdf | nllcoh | 162, 162, 40 | 139 / 65 (1444) | 198 / 79 (992) |

Full runs (`score_set` "full"): relocated events, events above the merged surface, cluster thickness (smallest
principal axis), median horizontal and vertical shift from v5.1, and the wall time of the leg (sum of its passes,
16 processes; nllssst includes nllbase, nllcoh includes nllssst).

| set | positions | relocated | above surface | thickness km (v5.1) | shift h / z km | wall s (full; folds) |
|---|---|---|---|---|---|---|
| axial | nllbase | 1200/1200 | 0 | 0.198 (0.198) | 0.040 / -0.002 | 15; 10, 12, 12 |
| axial | nllssst | 1200/1200 | 0 | 0.268 (0.198) | 0.186 / -0.020 | 344; 300, 312, 305 |
| axial | nllcoh | 1200/1200 | 0 | 0.205 (0.198) | 0.090 / +0.068 | 398; 343, 365, 359 |
| blanco | nllbase | 1044/1044 | 0 | 0.888 (0.617) | 0.049 / -0.003 | 13; 6, 12, 8 |
| blanco | nllssst | 1044/1044 | 0 | 1.030 (0.617) | 0.282 / +0.117 | 574; 257, 560, 495 |
| blanco | nllcoh | 1044/1044 | 0 | 0.682 (0.617) | 0.372 / -0.018 | 626; 279, 606, 536 |
| msh | nllbase | 1436/1436 | 0 | 0.323 (0.319) | 0.024 / -0.002 | 51; 30, 33, 31 |
| msh | nllssst | 1436/1436 | 0 | 0.417 (0.319) | 0.205 / -0.066 | 909; 661, 666, 693 |
| msh | nllcoh | 1436/1436 | 0 | 0.230 (0.319) | 0.182 / -0.146 | 999; 729, 740, 768 |
| jdf | nllbase | 162/162 | 0 | 0.075 (0.078) | 0.069 / -0.003 | 3; 3, 3, 1 |
| jdf | nllssst | 162/162 | 0 | 0.318 (0.078) | 0.279 / +0.214 | 65; 66, 57, 13 |
| jdf | nllcoh | 162/162 | 0 | 0.079 (0.078) | 0.433 / +0.053 | 77; 76, 66, 15 |

## 4. The SSST schedule, pass by pass

`--pass-scores` scores every NLLoc pass the same way (scratch only, `nll_work/<set>/pass_scores.json`); it0 is
nllbase, it4 nllssst, coh nllcoh. `var_coh0` and `var_coh2` are the coherence pass run on top of it0 (no SSST,
v5.1 static terms) and on top of it2 (SSST stopped at D = 2 km) instead of it4 (`--variant 0`, `--variant 2`).
These are diagnostics, not leg outputs; the schedule of the leg was fixed before any score was seen.

| set | pass | P rms / MAD ms | S rms / MAD ms |
|---|---|---|---|
| axial | v5.1 | 73 / 28 (10157) | 62 / 29 (4999) |
| axial | v5.1, nllcoh events | 72 / 28 (9523) | 62 / 28 (4802) |
| axial | it0 | 116 / 51 (9523) | 181 / 78 (4802) |
| axial | it1 | 112 / 50 (9523) | 185 / 76 (4802) |
| axial | it2 | 111 / 51 (9523) | 187 / 79 (4802) |
| axial | it3 | 114 / 52 (9523) | 193 / 83 (4802) |
| axial | it4 | 123 / 56 (9523) | 224 / 91 (4802) |
| axial | coh | 80 / 33 (9523) | 101 / 41 (4802) |
| axial | var_coh0 | 76 / 32 (9523) | 85 / 37 (4802) |
| axial | var_coh2 | 77 / 32 (9523) | 87 / 38 (4802) |
| blanco | v5.1 | 95 / 44 (7764) | 95 / 53 (8859) |
| blanco | v5.1, nllcoh events | 93 / 43 (4453) | 99 / 54 (5222) |
| blanco | it0 | 463 / 149 (3933) | 656 / 191 (4771) |
| blanco | it1 | 394 / 134 (4155) | 557 / 177 (4946) |
| blanco | it2 | 405 / 125 (4390) | 600 / 178 (5178) |
| blanco | it3 | 443 / 139 (4413) | 659 / 199 (5177) |
| blanco | it4 | 451 / 151 (4455) | 673 / 221 (5222) |
| blanco | coh | 155 / 60 (4453) | 202 / 79 (5222) |
| blanco | var_coh0 | 167 / 53 (3929) | 210 / 69 (4761) |
| blanco | var_coh2 | 122 / 50 (4390) | 141 / 64 (5178) |
| msh | v5.1 | 80 / 24 (31042) | 64 / 32 (13043) |
| msh | v5.1, nllcoh events | 79 / 23 (30505) | 63 / 31 (12692) |
| msh | it0 | 143 / 44 (30507) | 162 / 54 (12692) |
| msh | it1 | 134 / 39 (30498) | 152 / 49 (12691) |
| msh | it2 | 142 / 39 (30505) | 158 / 47 (12692) |
| msh | it3 | 153 / 42 (30505) | 178 / 50 (12692) |
| msh | it4 | 172 / 49 (30505) | 199 / 60 (12692) |
| msh | coh | 86 / 18 (30505) | 66 / 19 (12692) |
| msh | var_coh0 | 81 / 18 (30507) | 57 / 19 (12692) |
| msh | var_coh2 | 79 / 17 (30505) | 56 / 17 (12692) |
| jdf | v5.1 | 121 / 70 (2829) | 162 / 95 (2779) |
| jdf | v5.1, nllcoh events | 119 / 68 (1444) | 162 / 93 (992) |
| jdf | it0 | 280 / 124 (1444) | 331 / 160 (992) |
| jdf | it1 | 272 / 100 (1442) | 345 / 145 (992) |
| jdf | it2 | 282 / 102 (1442) | 361 / 146 (992) |
| jdf | it3 | 298 / 113 (1444) | 376 / 162 (992) |
| jdf | it4 | 307 / 121 (1444) | 400 / 200 (992) |
| jdf | coh | 139 / 65 (1444) | 198 / 79 (992) |
| jdf | var_coh0 | 149 / 72 (1444) | 189 / 86 (992) |
| jdf | var_coh2 | 130 / 63 (1442) | 179 / 79 (992) |

What the passes show:

* Static terms re-estimated on the set (it1) lower the held-out MAD in every set; D = 2 km (it2) holds that level;
  D = 1 and 0.5 km lose it and more, while the training-station misfit keeps falling. The small-scale SSST field
  absorbs mislocation along with path effects, in all four sets (6 to 72 training stations).
* The coherence step is robust to what sits under it: on the it0 baseline (no SSST at all) it already gives most of
  the gain. Its rms is lower than nllcoh's in axial and msh (S: 85 vs 101 ms, 57 vs 66 ms) and its MAD lower in
  axial and blanco; its MAD is higher than nllcoh's in jdf (P 72 vs 65 ms, S 86 vs 79 ms). (blanco var_coh0 is
  scored on fewer rows: 85 fold0 events are REJECTED at the bottom face in it0.)
* Coherence on it2 (SSST stopped at D = 2 km) is the best of the three in blanco, msh and jdf and within 2 ms of the
  best in axial: P / S MAD 32 / 38 (axial), 50 / 64 (blanco), 17 / 17 (msh), 63 / 79 (jdf), and rms 79 / 56 ms in
  msh, i.e. at or below v5.1 on the same events in all four msh numbers. In the full runs it is the thinnest of the
  three coherence versions in axial, blanco and msh (smallest-axis thickness 0.16 km axial, 0.51 blanco, 0.19 msh,
  0.08 jdf; v5.1 0.20, 0.62, 0.32, 0.08), with no event above the surface.
* This ranking was found on the held-out scores themselves, so it is not a fair entry for the comparison; it says
  the schedule chosen a priori (down to 0.5 km) is too fine for these station counts. A rerun with the schedule
  static + 2 km (D not below about the set extent) would give exactly the var_coh2 positions; they are in
  `nll_work/<set>/<run>/var_coh2/locs.csv` (LOCATED rows; `depth` = true depth km bsl).
* var_coh0 collapses jdf to a near-plane in the full run (thickness 0.015 km against 0.08 km for v5.1): stacking
  broad PDFs (Len3 median 7 km) without SSST can pull a whole cluster onto one surface; whether that is the fault or
  an artefact is not tested here.

## 5. Caveats and doubts

* **The fold test is harder for an absolute code.** In a fold run NLLoc locates every event again from the training
  picks alone. The v5.1 reference positions were located with every station, held-out ones included, and a relative
  code that starts from v5.1 keeps part of that information in its starting geometry. nllbase reproduces v5.1 in the
  full runs (median horizontal difference 24 m msh, 40 m axial, 49 m blanco, 69 m jdf; vertical median within 3 m;
  the differences come from the re-timed v4.1t picks, the smaller box and the oct-tree), yet in the
  fold runs it scores far below v5.1 (table above). That gap is what the held-out stations gave v5.1, not a model or
  code difference. The NLL legs should be compared with each other, and a relative code by its gain over its own
  start, rather than NLL against v5.1.
* **Events lost in fold runs.** An event needs 5 picks (production rule). Events left with fewer training picks
  (fold0 / fold1 / fold2): axial 107 / 9 / 12, msh 47 / 27 / 4, blanco 477 / 0 / 123, jdf 0 / 0 / 122 (jdf fold2
  holds out J54C, J61C and KEMF, three of the five stations that record most jdf events). These events and their CC
  pairs drop out of the NLL scores, so the NLL legs are scored on fewer rows than v5.1 (n in the table); the row
  "v5.1 on the nllcoh events" scores v5.1 on the same rows as nllcoh and differs from the full v5.1 score by 0-5 ms.
  Every event is located in the full runs.
* **SSST at small D does not help here.** Each pass lowers the misfit at the training stations (axial full: median
  rms 77 ms at it0, 46 ms at it4) while the held-out fit gets worse below D = 2 km (section 4), in all four sets
  (6 to 72 training stations). A correction field that varies over the cluster trades off with the event positions;
  Loc2ssst cannot tell a path effect from a mislocation, and the cluster drifts and spreads (axial full: median 3D
  shift from v5.1 58 m at it0, 175 m at it1, 365 m at it4; thickness 0.20 km at it0, 0.27 km at it4). The static pass
  (D = 9999) alone already moves events by ~0.2 km because the set-wide station terms differ from the catalog-wide
  b4_v51_iter0 terms by tens of ms at stations 1-3 km from the events.
* **Oct-tree resolution.** With the production search (40,000 nodes, 0.02 km minimum never reached) the
  maximum-likelihood point of nllbase and nllssst sits on 62-125 m leaves (v5.1 itself: 37-148 m, `ns_z` of
  b4_v51_iter1). A test with 100,000 and 200,000 nodes on 75 axial events (38 s and 157 s instead of 14 s) only
  reached 31 m leaves, so the production search was kept. The coherence search reaches 16 m leaves.
* **Coherence measure.** Lomax takes the maximum over channels of a P+S waveform correlation; here it is the mean
  per-phase cc_max of the accepted rows. Rows below the acceptance threshold (0.7) and failed rows are absent, so a
  pair with one accepted row at 0.71 counts as coherent (weight 0.6) and nothing counts as evidence against
  coherence. All CC pairs are within 2.9 km (epicentral, v5.1), inside Lomax's 5 km pair distance.
* **What the coherence pass does.** LOCPOSTERIOR replaces the likelihood: the location is the maximum of the
  weighted stack of the SSST PDFs of the event (weight 1) and its CC neighbours (weights 0.5-1, median 18 neighbours
  in the full runs), raised to the power of the summed weight. The event's own picks act only through its own PDF.
  This pulls events onto the common PDF of their coherent group and thins clusters (msh 0.32 -> 0.23 km), which is
  the purpose, but an event whose neighbours were mislocated by SSST inherits their error. In jdf (PDF semi-axes
  median 7 km) the full-run coherence positions move 0.43 km (median) from v5.1, with an event-to-event spread of
  0.5-0.6 km horizontally, at unchanged thickness.
* **Filters.** The Len3 / MaxSE3 cuts of Lomax's example (5 km) are off (section 2). The first axial run had them on;
  its scores are within 3 ms of the final ones (`nll_work/axial/pass_scores_len3cut5km.json`: coh P 80.0 / 32.9,
  S 98.2 / 40.2 ms rms / MAD; it4 P 125.0 / 56.5, S 226.0 / 92.1 ms). For blanco (Len3 median 10 km) and jdf (7 km)
  the cut would have removed most events from SSST and from the stacks.
* **Box.** The search box (+- 10 km, z' to 25 km) is smaller than production's (tile +- 75 km, z' to 80 km, then the
  D-B067 deep pass). NLLoc rejects an event whose best point lies on the box face: blanco fold0 85 events in nllbase,
  10 in nllssst, 11 in nllcoh (bottom face; once BB640, BB660, BB670, BB710... are held out these events have no
  nearby station and sink), blanco fold2 6 / 1 / 1, msh fold0 2 / 0 / 0, msh fold1 0 / 1 / 1. Production would have
  located them deep. Listed in run.json as REJECTED (or "no oct-tree in the last SSST pass" for nllcoh).
* **Thickness is a standard deviation.** blanco nllbase is 0.89 km thick against 0.62 km for v5.1 although its
  positions differ from v5.1 by 70 m (median 3D) because two events moved more than 2 km (one 13 km).
* **Static terms.** b4_v51_iter0 terms were computed by production from the whole catalog; in a fold run only the
  training stations' terms are used, so held-out data enter only indirectly (the terms are mean residuals at
  locations that used every station). The SSST legs replace them in their first pass.
* **Loc2ssst LSSTATIONS** matches station codes by substring (`strstr`); stations whose code is contained in another
  are put in the same Loc2ssst instance, so no grid is written twice.

## 6. Reproduce

```
cd workflow/10_tomography
PY="env LD_LIBRARY_PATH=$PWD/../../.pixi/envs/default/lib $PWD/../../.pixi/envs/default/bin/python"
$PY b5_cmp_nll.py axial jdf msh blanco          # all runs; resumes from the scratch folders
$PY b5_cmp_nll.py --score                       # -> cmp/nllcoh/scores.json
$PY b5_cmp_nll.py axial jdf msh blanco --pass-scores
$PY b5_cmp_nll.py axial jdf blanco msh --variant 0   # and --variant 2 (diagnostics)
```

Scratch kept (22 GB): per-pass hyps and locs.csv, the SSST grids T2 and T4 (inputs of var_coh2 and nllcoh), the
it4 / var_it oct-trees, the stream_coherences files and logs; the grid buffers of T1 and T3 were deleted (their
done.json is kept, so a rerun still resumes). Tables of this note: `pass_scores.json` per set and `scores.json`.
