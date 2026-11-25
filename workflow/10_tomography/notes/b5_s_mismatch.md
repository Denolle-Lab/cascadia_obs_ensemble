# B5: why the S differential-time misfit grows (root cause)

2026-11-24, coordinator session (PI asked to find the root cause before anything else). Production run b5_v0.3.1
(notes/b5_relocation.md): dt residual rms in the B4 grids P 0.147 -> 0.148 s, S 0.181 -> 0.232 s (worse in every
region), while tomoDD reported its own misfit falling by a quarter. The first reading (P-B5-6) was that tomoDD's S
travel times differ from the B4 NLLoc grids. **That is wrong.** The cause is the relative weight of S.

Scripts (this folder): b5_sdiag_run_fwd.py, b5_sdiag_compare.py, b5_sdiag_variants.py, b5_sdiag_var_table.py.
Outputs: /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/diag/s_mismatch/<cluster>/. Clusters: c000009 (Blanco OBS swarm
cluster, 850 events, 63 stations, 35,401 dt) and c000023 (onshore Oregon, 306 events, 109 stations, 17,417 dt).
All runs copy the production inputs (MOD, dt.ct, event.dat, station.dat, frozen binary) and change only the lines
named below.

## 1. The two forward models agree

One-iteration tomoDD runs (NSET 1, NITER 1) at the B4 start and at the B5 final positions (tomoDD.reloc) give
tomoDD's calculated differential time for every dt row (DT - RES in tomoDD-SE.res.ini). The same rows computed in
the B4 time grids (time_b4_v0.3.1, trilinear, EFT depth) at the same positions:

| cluster | positions | phase | tomoDD minus grids, median / MAD (ms) | residual rms tomoDD / grids (s) |
|---|---|---|---|---|
| c000009 | B4 | P | +0.2 / 3.8 | 0.197 / 0.196 |
| c000009 | B4 | S | +0.0 / 4.0 | 0.271 / 0.269 |
| c000009 | B5 | P | +0.2 / 5.2 | 0.179 / 0.178 |
| c000009 | B5 | S | -0.1 / 5.3 | 0.330 / 0.327 |
| c000023 | B4 | P | -0.1 / 1.5 | 0.152 / 0.150 |
| c000023 | B4 | S | -0.1 / 2.3 | 0.167 / 0.166 |
| c000023 | B5 | P | +0.1 / 2.4 | 0.146 / 0.146 |
| c000023 | B5 | S | +0.0 / 3.8 | 0.199 / 0.204 |

(Residuals with the per-pair median removed, as compare.md.) The S misfit rises by the same amount in both models.
The only large differences (c000023 at B5: 2,028 of 17,401 rows, 143 events) involve events B5 moved above the
ground (e.g. 90241 at -0.70 km near Eugene), where tomoDD's rock-filled nodes and the grids differ: that is the
D-B055 problem, not this one.

## 2. The weighting makes B5 trade S for P

Full B5 relocations of the same clusters (4 sets x 3 iterations, as production) with only the weights changed;
each result evaluated by a one-iteration run at its final positions, on the dt rows common to all runs:

```
== c000009: dt residual rms (s) on 34,992 common rows, per-pair median removed (tomoDD forward)
  B4 start     P 0.196  S 0.269   P MAD 0.092  S MAD 0.139
  prod         P 0.179  S 0.328   P MAD 0.074  S MAD 0.153
  s_eq         P 0.201  S 0.254   P MAD 0.086  S MAD 0.105
  norew        P 0.174  S 0.298   P MAD 0.078  S MAD 0.153
  s_eq_norew   P 0.197  S 0.234   P MAD 0.089  S MAD 0.113
  p_only       P 0.192  S 0.382   P MAD 0.074  S MAD 0.191
  s_only       P 0.230  S 0.233   P MAD 0.105  S MAD 0.096
== c000023: dt residual rms (s) on 17,401 common rows, per-pair median removed (tomoDD forward)
  B4 start     P 0.152  S 0.166   P MAD 0.058  S MAD 0.067
  prod         P 0.146  S 0.199   P MAD 0.044  S MAD 0.081
  s_eq         P 0.151  S 0.165   P MAD 0.051  S MAD 0.053
  norew        P 0.140  S 0.191   P MAD 0.045  S MAD 0.082
  s_eq_norew   P 0.148  S 0.154   P MAD 0.052  S MAD 0.054
  p_only       P 0.147  S 0.206   P MAD 0.045  S MAD 0.088
  s_only       P 0.163  S 0.145   P MAD 0.066  S MAD 0.044
```

prod = production (dt.ct weight (0.05/sigma)^2 with the D-B008 sigma, S = 2 x P, so S carries 1/4 of P's weight;
WTCTP = WTCTS = 1; WRCT 8 then 6 MAD cut, WDCT 10 km). s_eq = WTCTS 4 (S weight = P weight). norew = WRCT and
WDCT -9 in all sets. p_only / s_only = WTCTS 0 / WTCTP 0.

* Production behaves almost like P only: P improves a little, S gets much worse.
* With S weighted like P, S improves and P returns to its B4 value; without the MAD cuts as well, S improves most
  (Blanco 0.269 -> 0.234 s, onshore 0.166 -> 0.154 s) and P is unchanged (Blanco) or better (onshore).
* P-only and S-only solutions each degrade the other phase: the catalog P and S differential times disagree on the
  relative positions, so fitting one alone overfits it. tomoDD's falling rmsct is weighted (S at 1/4) and computed on
  the rows the MAD cut keeps (70 % of dt in c000009 at the last iteration), so it hid this.
* The D-B008 ratio S = 2 x P is not what the data show: S/P residual MAD at B4 is 1.51 (c000009) and 1.16
  (c000023); the catalog-wide dt rms ratio is 0.181 / 0.147 = 1.23. Inverse-variance weights would give S about
  0.45-0.75 of P's weight, not 0.25.

## 3. Proposed decisions (for the PI; not in DECISIONS.md)

* P-S1: B5 weights S differential times from the measured S/P residual ratio (per region, from the B4 residuals),
  or equal to P, instead of the D-B008 factor 4; MAD re-weighting off or looser (to test on more clusters).
* P-S2: acceptance rule for any B5 run: the dt residual rms in the B4 grids must not grow for either phase, by region
  (compare.md already computes it).
* P-S3: correct P-B5-6 (notes/b5_relocation.md): the B5 catalog does not mix two forward models; tomoDD and the B4
  grids agree to a few ms except above the surface.
Not tested here: the effect of these weights on the above-surface count, other regions, damping.
