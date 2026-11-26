**Recommendation: B** (NLL-SSST stopped at 2 km, then the coherence stage). The rule was written before the scores: B had to win the held-out-station test in at least 2 of 3 clusters and lose in none. It won in all 3, on P and on S, and no arm put an event above the surface.

### What was tested

Three clusters that share no event with the four earlier sets: 1,000 events in the Puget Lowland (onshore, about 17 km deep), 600 events of the 2015 M5.7 sequence off Cape Mendocino (NC and PB land stations plus 7D and Z5 ocean-bottom seismometers) and 1,200 events of the October 2012 swarm on the western Blanco transform (X9 and 7D ocean-bottom seismometers). Every arm used the same picks (v4.1tc), the same v5.1 grids and the v6-prep code. CC differential times were measured for these clusters with the tested D-B078 code: 42,400 accepted rows in Puget, 22,100 at Mendocino and 38,700 at western Blanco.

### Held-out stations (the leak-free test)

Each arm was rerun three times, each time without one third of the stations, and scored on the CC times at the stations it never saw (-18's scorer). Median misfit (MAD) on the same rows, A then B, with the 95 % interval of B - A:

- Puget: P 85 vs 59 ms (B - A -26 ms, [-32, -20]); S 141 vs 90 ms (-51, [-58, -44]).
- Mendocino: P 67 vs 56 ms (-12, [-15, -8]); S 118 vs 88 ms (-30, [-36, -24]).
- western Blanco: P 149 vs 107 ms (-41, [-46, -37]); S 264 vs 164 ms (-100, [-107, -92]).

### CC fit and depths (full runs)

The median per-pair residual of the CC times favours B in all three clusters: Puget 16.9 vs 14.8 ms, Mendocino 47.1 vs 41.9 ms, western Blanco 41.5 vs 38.6 ms (all intervals exclude 0). The repeaters (true separation about 0.1 km) are mixed: no difference in Puget (0.46 vs 0.53 km), B tighter at Mendocino (1.06 vs 0.38 km), A tighter at western Blanco (0.71 vs 1.77 km; the 2 km SSST pass spreads them there, they sit at 0.81-0.95 km after the 8 and 4 km passes). Depths: against Lange et al. (2026) at western Blanco, B is closer (median |dz| 2.11 vs 1.52 km, 891 events); against PNSN in Puget (345 events, A 1.35, B 1.22 km) and NCSN at Mendocino (39 events, both about 6 km deeper than NCSN, the known offset) there is no material difference. No event is above the surface in any arm.

### Why the gap is large, and how this fits -18's result

The coherence stage of the v6-prep code relocates only events whose error ellipsoid is at most 5 km long. In the held-out runs that is 16-32 % of the located events in Puget, 42-56 % at Mendocino and 0.4-2 % at western Blanco. For every other event A keeps its static-term location and B its SSST location, so most of the difference is static terms against SSST. On the rows where the coherence stage relocated both events in both recipes, B is still better, but by less: Puget P -3.5 ms, S -4.5 ms; Mendocino P -4.5 ms, S -13 ms (intervals exclude 0). Part of B's gain comes from re-estimating station terms on the cluster itself: the first SSST pass (one term per station-phase, CharDist 9999 km) already beats the catalog-wide v5.1 terms (P 91 vs 101 ms in Puget, 69 vs 79 ms at Mendocino, 113 vs 149 ms at western Blanco).

This agrees with -18's four-set result in direction. -18's coherence on SSST stopped at 2 km beat coherence on the static terms by 0 to 9 ms (equal at Axial within 1 ms), without intervals; their coherence stage relocated every event, which is close to the "both relocated" rows here, where the differences are of the same size. It also agrees with -18 that fine SSST passes hurt: here the 2 km pass is already slightly worse than the 4 km pass in all three clusters (held-out P MAD after the 4 km and after the 2 km pass: Puget 64 and 68 ms, Mendocino 43 and 48 ms; at western Blanco the 16 km pass is best, 95 ms against 107 ms at 2 km). That comparison was made after seeing the scores, so it is a lead for a later check, not a result.

### Two things the PI should know

- At Mendocino, SSST alone does better than B on the held-out stations (P 48 vs 56 ms, S 68 vs 88 ms): there the coherence stage moves SSST locations away from what the unseen stations say. In Puget it helps (68 to 59 ms P). B still beats A everywhere, so the choice between A and B does not change.
- The verdict is about the recipe as the v6-prep code runs it (coherence on compact events only, v5.1 grids, cluster-sized search boxes). The v6.0 run will compute SSST on the whole catalog; its first pass then gives catalog-wide terms again, and the local gain comes from the 16 to 2 km passes.
