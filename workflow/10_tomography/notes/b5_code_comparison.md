# D-B078: choice of the B5 relative-relocation code on waveform cross-correlation data

Session -18, 2026-11-25. Code: `b5_cmp_common.py` (frame and scorer), `b5_cmp_tomodd.py` (tomoDD leg),
`b5_cmp_report.py` (report), on tomo/b5; GrowClust3D leg on tomo/cmp-growclust (941d9084,
`notes/b5_cmp_growclust.md`); NLL legs on tomo/cmp-nll (47fd5e2b, `notes/b5_cmp_nll.md`). Full tables:
`data/tomo_v1/b5/cmp/report.md` / `report.json`. Large outputs: `/wd1/mdenolle_data/tomo_v1/b5/cmp/`.

## Test

- Sets (catalog v5.1; `b5_cc_sets.py`): axial (1,200 events of chunk c000001_p001), blanco (c000010, 1,044), msh
  (Mount St Helens c000007, 1,436), jdf (JdF ridge c000032, 162; most events on the floor).
- Data, the same for every code: re-timed catalog picks (v4.1t) and waveform cross-correlation differential times
  (CC dt; other session, `/wd1/mdenolle_data/tomo_v1/cc_dt`, short reads on the header time base; cuts: threshold,
  triangle closure, pairs straddling the MH07/MG07 clock epochs).
- Score: 3 station folds per set; each code relocates without the held-out stations; the held-out CC dt are predicted
  in the v5.1 NonLinLoc grids from the relocated positions, each pair's median training residual removed (origin time
  and common shifts cancel); MAD and rms pooled over folds. Events a code leaves unrelocated are scored at their start
  position (`_filled`, the operational score). Full runs: events above the solid surface (D-B055), cluster thickness.

## The leak, and the fair test

The first round started the relative codes (tomoDD, GrowClust3D) from the v5.1 positions, which NonLinLoc located
with every station, the held-out ones included; damping keeps events near their start, so held-out information leaked
into those codes (and the v5.1 row itself is in-sample). The NLL legs relocate from the training picks only. In the
fair test every code starts from NonLinLoc relocated without the held-out stations (`nllbase`, the NLL leg's
baseline: v5.1 settings, training stations only); tomoDD was rerun that way (`--start nllbase`, legs `tomodd_snll`,
`tomodd_d300_o0_snll`). The leak was large: e.g. blanco tomoDD (damping 300) MAD P 37 ms from v5.1 vs 112 ms from
the fair start. GrowClust3D was not rerun from the fair start: it was already the worst with the leak in its favour.

## Results, fair test (held-out CC dt, MAD / rms in ms)

| set, phase | NLLoc baseline (nllbase) | NLL-SSST-coherence (nllcoh) | tomoDD + D-B081 floor, damping 100 | tomoDD, damping 300 |
|---|---|---|---|---|
| axial P | 51 / 115 | 34 / **81** | **29** / 90 | 32 / 92 |
| axial S | 78 / 179 | 42 / **101** | **35** / 129 | 41 / 144 |
| blanco P | 138 / 472 | **67 / 168** | 101 / 355 | 112 / 376 |
| blanco S | 172 / 667 | **84 / 219** | 128 / 479 | 143 / 514 |
| msh P | 44 / 143 | 19 / **87** | **14** / 112 | 19 / 108 |
| msh S | 54 / 163 | 20 / **68** | **9** / 114 | 16 / 117 |
| jdf P | 136 / 411 | 99 / **348** | **81** / 373 | 110 / 385 |
| jdf S | 183 / 495 | 143 / **474** | **139** / 480 | 160 / 482 |
| above the surface (4 sets) | 0 | 0 | 0 | 0 |

GrowClust3D (from v5.1, leak in its favour): MAD / rms axial 46 / 167, 43 / 153; blanco 87 / 231, 123 / 301; msh 13 /
112, 8 / 90; jdf 51 / 205, 103 / 309; 158 events above the surface (no floor in GrowClust).
Axial without the CC rows that straddle unconfirmed candidate clock epochs of the Axial array: same ranking.

## Reading

- Both NLL-SSST-coherence and tomoDD + floor improve strongly on NonLinLoc alone, on every set.
- tomoDD + floor (damping 100) places the core of a cluster best (lowest MAD at axial, msh, jdf); NLL-SSST-coherence
  has the smallest tails (lowest rms at axial, blanco, msh, jdf) and is clearly better at Blanco (sparse, linear OBS
  arrays, few CC dt per pair), where tomoDD's tails are large. Neither puts an event above the surface.
- GrowClust3D is out: the largest tails, events above the surface, and singletons left unrelocated.
- Damping 300 helps tomoDD only when the start is already good (from v5.1); from the fair start it is worse than 100.
- NLL leg details: SSST below a 2 km smoothing distance made the held-out fit worse in all sets; a variant chosen after
  seeing the scores (coherence on SSST stopped at 2 km, `var_coh2`) beats `nllcoh` everywhere (MAD P/S: axial 32/38,
  blanco 50/64, msh 17/17, jdf 63/79) but is not a fair entry until confirmed on other clusters.

## Options for the PI

1. NLL-SSST-coherence for B5: smallest tails, best where data are sparse (Blanco and offshore), the NonLinLoc floor
   and grids natively (no tomoDD source change), and the same code as the v6.0 relocation (D-B079, D-B092). Its SSST
   schedule (stop at about 2 km) to be confirmed on independent clusters first.
2. tomoDD + D-B081 floor: sharpest cluster cores where links are dense (Axial, Mount St Helens), larger tails.
3. Both: NLL-SSST-coherence as the catalog product, tomoDD + floor as a cross-check for dense clusters (no catalog
   column from it).

Recommendation of session -18: option 1 (with the SSST schedule confirmed), keeping tomoDD + floor as the
cross-check of option 3.
