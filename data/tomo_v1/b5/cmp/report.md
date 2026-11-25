# D-B078: B5 code comparison on waveform cross-correlation data

Held-out CC dt in the v5.1 NonLinLoc grids (3 station folds, each pair's training median removed): MAD / rms (ms), P and S. `filled` = the code's events, with events it left unrelocated kept at v5.1 (the operational score). Full runs: relocated / events, above the solid surface, cluster thickness (smallest principal axis, km; v5.1 in brackets).

**Reading the v5.1 row:** v5.1 was located with every station, the held-out ones included, so its held-out score is in-sample and flattered. The fair NonLinLoc baseline is `nllbase` (NonLinLoc with the v5.1 settings, relocated without the held-out stations, like every other code). `axial_no_candidates`: axial scored without the CC rows that straddle unconfirmed candidate clock epochs of the Axial array (D-B090 screen).

## axial

| code | P MAD | P rms | S MAD | S rms | relocated | above surface | thickness km |
|---|---|---|---|---|---|---|---|
| v5.1 (NLLoc) | 28.23 | 72.57 | 28.53 | 62.36 | - | 0 | - |
| tomodd | 22.79 | 78.99 | 26.79 | 85.47 | 1197/1200 | 0 | 0.316 (0.198) |
| tomodd_d300_o0 | 22.15 | 67.18 | 23.46 | 62.83 | 1198/1200 | 0 | 0.28 (0.198) |
| tomodd_snll | 28.63 | 89.55 | 34.59 | 128.54 | 1197/1200 | 0 | 0.315 (0.198) |
| tomodd_d300_o0_snll | 32.16 | 92.38 | 40.87 | 144.0 | 1198/1200 | 0 | 0.277 (0.198) |
| growclust | 45.93 | 167.27 | 42.59 | 152.78 | 1169/1200 | 88 | 0.361 (0.195) |
| nllbase | 51.23 | 115.1 | 77.63 | 179.31 | 1200/1200 | 0 | 0.198 (0.198) |
| nllssst | 55.81 | 122.33 | 91.17 | 222.47 | 1200/1200 | 0 | 0.268 (0.198) |
| nllcoh | 33.85 | 81.1 | 42.03 | 101.37 | 1200/1200 | 0 | 0.205 (0.198) |

## blanco

| code | P MAD | P rms | S MAD | S rms | relocated | above surface | thickness km |
|---|---|---|---|---|---|---|---|
| v5.1 (NLLoc) | 43.83 | 94.79 | 52.69 | 95.34 | - | 0 | - |
| tomodd | 41.09 | 107.26 | 54.33 | 141.01 | 1044/1044 | 0 | 0.547 (0.617) |
| tomodd_d300_o0 | 37.12 | 93.94 | 51.91 | 108.25 | 1044/1044 | 0 | 0.582 (0.617) |
| tomodd_snll | 101.27 | 354.89 | 128.47 | 479.26 | 1043/1044 | 0 | 0.576 (0.615) |
| tomodd_d300_o0_snll | 112.16 | 376.2 | 143.01 | 514.3 | 1042/1044 | 0 | 0.585 (0.614) |
| growclust | 87.2 | 230.82 | 123.26 | 300.57 | 990/1044 | 29 | 1.003 (0.603) |
| nllbase | 137.97 | 472.2 | 171.88 | 667.13 | 1044/1044 | 0 | 0.888 (0.617) |
| nllssst | 140.49 | 456.73 | 198.18 | 682.1 | 1044/1044 | 0 | 1.03 (0.617) |
| nllcoh | 67.35 | 167.73 | 84.39 | 218.92 | 1044/1044 | 0 | 0.682 (0.617) |

## msh

| code | P MAD | P rms | S MAD | S rms | relocated | above surface | thickness km |
|---|---|---|---|---|---|---|---|
| v5.1 (NLLoc) | 23.51 | 80.09 | 31.62 | 63.77 | - | 0 | - |
| tomodd | 13.23 | 99.93 | 8.57 | 80.01 | 1436/1436 | 0 | 0.399 (0.319) |
| tomodd_d300_o0 | 15.82 | 82.34 | 13.57 | 56.97 | 1436/1436 | 0 | 0.384 (0.319) |
| tomodd_d100_o4 | 13.23 | 99.4 | 8.6 | 77.8 | 1436/1436 | 0 | 0.434 (0.319) |
| tomodd_d300_o4 | 15.81 | 82.29 | 13.57 | 56.8 | 1436/1436 | 0 | 0.388 (0.319) |
| tomodd_snll | 13.87 | 111.8 | 8.74 | 113.64 | 1436/1436 | 0 | 0.397 (0.319) |
| tomodd_d300_o0_snll | 18.61 | 107.82 | 15.63 | 117.21 | 1436/1436 | 0 | 0.38 (0.319) |
| growclust | 12.62 | 111.82 | 8.16 | 90.04 | 1328/1436 | 1 | 0.359 (0.3) |
| nllbase | 43.92 | 142.97 | 53.97 | 163.27 | 1436/1436 | 0 | 0.323 (0.319) |
| nllssst | 49.74 | 172.1 | 61.54 | 199.49 | 1436/1436 | 0 | 0.417 (0.319) |
| nllcoh | 18.71 | 86.53 | 19.52 | 67.59 | 1436/1436 | 0 | 0.23 (0.319) |

## jdf

| code | P MAD | P rms | S MAD | S rms | relocated | above surface | thickness km |
|---|---|---|---|---|---|---|---|
| v5.1 (NLLoc) | 70.14 | 121.28 | 95.41 | 162.24 | - | 0 | - |
| tomodd | 41.29 | 122.08 | 64.77 | 179.04 | 162/162 | 0 | 0.085 (0.078) |
| tomodd_d300_o0 | 55.93 | 109.91 | 78.71 | 148.88 | 162/162 | 0 | 0.094 (0.078) |
| tomodd_d100_o4 | 41.45 | 121.34 | 63.82 | 167.93 | 162/162 | 0 | 0.085 (0.078) |
| tomodd_d300_o4 | 56.55 | 106.9 | 79.72 | 146.16 | 162/162 | 0 | 0.094 (0.078) |
| tomodd_snll | 81.43 | 373.42 | 138.56 | 480.02 | 162/162 | 0 | 0.087 (0.078) |
| tomodd_d300_o0_snll | 109.86 | 384.97 | 160.03 | 482.35 | 162/162 | 0 | 0.093 (0.078) |
| growclust | 50.53 | 205.4 | 102.53 | 308.89 | 162/162 | 40 | 0.208 (0.078) |
| nllbase | 136.1 | 410.62 | 183.08 | 494.63 | 162/162 | 0 | 0.075 (0.078) |
| nllssst | 138.51 | 465.03 | 205.38 | 636.29 | 162/162 | 0 | 0.318 (0.078) |
| nllcoh | 99.11 | 348.1 | 142.97 | 474.19 | 162/162 | 0 | 0.079 (0.078) |

## axial_no_candidates

| code | P MAD | P rms | S MAD | S rms | relocated | above surface | thickness km |
|---|---|---|---|---|---|---|---|
| v5.1 (NLLoc) | 29.19 | 71.21 | 28.72 | 62.18 | - | 0 | - |
| tomodd | 23.2 | 76.65 | 26.54 | 80.93 | 1197/1200 | 0 | 0.316 (0.198) |
| tomodd_d300_o0 | 22.57 | 64.46 | 23.32 | 58.24 | 1198/1200 | 0 | 0.28 (0.198) |
| tomodd_snll | 29.31 | 87.64 | 34.54 | 122.58 | 1197/1200 | 0 | 0.315 (0.198) |
| tomodd_d300_o0_snll | 33.42 | 90.18 | 40.88 | 140.03 | 1198/1200 | 0 | 0.277 (0.198) |
| growclust | 47.61 | 168.87 | 43.83 | 149.8 | 1169/1200 | 88 | 0.361 (0.195) |
| nllbase | 53.33 | 114.18 | 78.91 | 175.46 | 1200/1200 | 0 | 0.198 (0.198) |
| nllssst | 57.46 | 121.25 | 91.56 | 219.28 | 1200/1200 | 0 | 0.268 (0.198) |
| nllcoh | 35.03 | 80.12 | 43.28 | 102.17 | 1200/1200 | 0 | 0.205 (0.198) |

