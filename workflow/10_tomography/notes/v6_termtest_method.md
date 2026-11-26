### Why this test

On 2026-11-25 at 14:29 the PI chose a test over a decision between two station-term recipes for the v6.0 NonLinLoc relocation. Recipe A is the v5.1 static station terms followed by the NLL-coherence stage. Recipe B is NLL-SSST with the CharDist schedule 9999, 16, 8, 4, 2 km (it stops at 2 km) followed by the same coherence stage. The two earlier tests (the v6-prep test and -18's D-B078 comparison) used the same four clusters (axial, blanco, msh, jdf), so this test uses three clusters that share no event with them.

### Clusters

Clusters come from the v5.1 clustering (events_zbz_v51.csv.gz, families_zbz_v51.csv) and are cut like -18's sets: the located events nearest the median epicentre. Puget Lowland has no ZBZ family above 205 events, so the onshore set is spatial: the densest spot of v5.1 epicentres in the Puget/Georgia region (Kitsap Peninsula, about 17 km deep), 1,000 events. The Mendocino set is family 1536, the 2015-01-28 M5.7 offshore Cape Mendocino sequence, recorded by NC, BK, PB and TA land stations and by the 7D and Z5 ocean-bottom seismometers of Cascadia Initiative year 4 (600 events). The offshore set is family 427, the October 2012 swarm on the western Blanco transform, recorded by the Blanco OBS experiment (X9) and 7D (1,200 events).

### Relocation

Every arm runs the v6-prep code (b4_v6_ssst.py) unchanged in what it computes: the clock-corrected picks v4.1tc (md5 checked), a2_make_obs.py, the v5.1 travel-time grids (the 1 km grid, or the 2 km wide grid for a station whose 1 km grid does not cover the search box), a search box of the set's epicentres plus 30 km and z' from -5 to 80 km, NLLoc with the v5.1 settings (LOCGAU 0.2 s, at least 5 phases, the merged-surface floor), Loc2ssst with the v6-prep LSPHSTAT and WeightFloor, and the coherence stage of b4_v6_ssst.coherence_run. That stage relocates only events whose 68 % ellipsoid is at most 5 km long (Len3), by stacking the oct-tree pdfs of the event and of its partners with pair coherence of at least 0.5; the other events keep their previous location. Pair coherence is the median cc_max of the pair's measured CC rows (at least 3 rows). The static arm is the static_oct run (static terms, oct-trees saved): in the v6-prep axial run its locations are identical to the run without oct-trees.

### CC data

The CC differential times come from the tested D-B078 code (tomo/cc-dt 2d033ec6), with its parameters and the 3-15 Hz band that D-B093 uses for every block except Mount St Helens. Only the output folder and the set names change. Snippets that the D-B093 all-family job had already read were copied from its cache (read only); the others were read from pnwstore with the same reader. NC and BK snippets (Mendocino) come from the D-B093 NCEDC cache. Rows that straddle a confirmed D-B090 clock epoch (MH07.XD, MG07.XD) are flagged as in D-B078.

### Criterion 1: held-out stations

This is -18's protocol and scorer (b5_cmp_common.fold_score from tomo/b5 b0dffcdd). The stations with picks in a set are dealt into 3 folds with a fixed seed. For each fold every arm is rerun from the beginning without the picks of that fold's stations, and the coherence stage uses no CC row at those stations, so no position or pair weight has seen the held-out stations. The held-out CC differential times are then predicted from each arm's positions in the v5.1 grids; each residual has the pair's median training residual removed, which cancels the origin-time offset. MAD and rms are pooled over the folds. The one change from -18's scorer: the prediction uses the grid NLLoc used for that station (1 km or 2 km wide grid) instead of always the 1 km grid, because in the western Blanco set 91 of 143 station-phases have no 1 km grid over the box and -18's lookup would be clipped at the grid edge. Arms are compared on the rows whose two events are located in every arm of the fold.

### Criterion 2: fit of the CC times

On the full runs (all stations), the median per-pair residual of b4_v6_ssst_metrics.cc_score: for each accepted CC row, the observed minus the predicted differential arrival time with each arm's own corrections, minus the pair's median, then the median absolute value. This criterion is not independent of recipe B, whose SSST corrections are fitted to the same stations, and it does not test the coherence stage against data it has not seen. The repeater check takes the CC repeaters (median CC at least 0.95 over at least 6 station-phases), whose true separations are about 0.1 km, and reports the median 3-D distance each arm gives them.

### Criterion 3: depths

Full runs only, against published catalogs: Lange et al. (2026) absolute OBS locations for western Blanco (the pair table of b4_lit_paired.py), PNSN (ANSS ComCat net uw) for Puget, NCSN (net nc) for Mendocino. ComCat events are paired one-to-one with the v5.1 events, greedy by origin-time difference then distance (Puget 2 s and 10 km, Mendocino 3 s and 20 km). The same events are used for every arm. Also counted: events above the merged solid surface (must be 0) and events at the NLLoc floor.

### Statistics and decision rule

Intervals are 95 % percentile intervals from 2,000 paired bootstrap resamples (event pairs for criteria 1 and 2, with all of a pair's rows kept together; events for criterion 3). A difference is material when the interval of B - A excludes zero and the difference is at least 5 % of A (0.2 km for depths). The rule, written before any score of these runs: B is recommended only if it is materially better on held-out stations in at least 2 of the 3 clusters, materially worse in none, and puts no event above the surface; A on the mirror condition; otherwise "no material difference", and then A because it is simpler. Criteria 2 and 3 can only veto: a winner of criterion 1 that loses materially on both of them in 2 clusters is not recommended.
