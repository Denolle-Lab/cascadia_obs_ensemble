# T6 joint against stitched: blanco_oregon_v2_d200_s20

Joint blanco_oregon_v2 d200_s20; stitched = blanco_v2 d200_s20 + oregon_v2g d200_s20 with box taper x resolution window (taper 15 km, floor 0.1); overlap width 48 km. dlnV pooled over 1-34 km, in %. Yardsticks: diff rms of damp200_vs_300_A = blanco_v2 d200_s20 vs blanco_v2 d300_s20; damp200_vs_300_B = oregon_v2 d200_s20 vs oregon_v2 d300_s20; weightscale_B = oregon_v2 d200_s20 vs oregon_v2g d200_s20.
Masks: ['/home/mdenolle/wt_tomo_res/data/tomo_v1/b2_res_tests/blanco_v2/d200/resolution_mask.npz', '/home/mdenolle/wt_tomo_res/data/tomo_v1/b2_res_tests/oregon_v2/d200/resolution_mask.npz'].

| phase | zone | class | n | rms joint | rms stitched | diff mean | diff rms | corr | amp S/J | damp200_vs_300_A diff rms | damp200_vs_300_B diff rms | weightscale_B diff rms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | blanco only | resolved | 383 | 5.91 | 5.50 | -0.04 | 2.10 | 0.93 | 0.87 | 2.13 |  |  |
| P | blanco only | unresolved | 9337 | 5.05 | 4.75 | -0.08 | 1.71 | 0.94 | 0.89 | 1.65 |  |  |
| P | blanco only | all | 9720 | 5.09 | 4.79 | -0.08 | 1.73 | 0.94 | 0.88 | 1.68 |  |  |
| P | near seam blanco | resolved | 27 | 8.06 | 8.41 | -0.75 | 2.96 | 0.94 | 0.98 | 1.65 |  |  |
| P | near seam blanco | unresolved | 1593 | 4.73 | 4.42 | +0.06 | 1.58 | 0.94 | 0.88 | 1.31 |  |  |
| P | near seam blanco | all | 1620 | 4.80 | 4.52 | +0.05 | 1.61 | 0.94 | 0.89 | 1.31 |  |  |
| P | overlap | unresolved | 1620 | 2.94 | 2.15 | +0.13 | 1.88 | 0.76 | 0.56 | 0.71 | 1.07 | 2.00 |
| P | overlap | all | 1620 | 2.94 | 2.15 | +0.13 | 1.88 | 0.76 | 0.56 | 0.71 | 1.07 | 2.00 |
| P | near seam oregon | unresolved | 1860 | 2.55 | 2.20 | +0.01 | 1.26 | 0.87 | 0.75 |  | 1.27 | 1.98 |
| P | near seam oregon | all | 1860 | 2.55 | 2.20 | +0.01 | 1.26 | 0.87 | 0.75 |  | 1.27 | 1.98 |
| P | oregon only | resolved | 507 | 5.35 | 4.62 | +0.00 | 2.35 | 0.89 | 0.78 |  | 2.02 | 2.72 |
| P | oregon only | unresolved | 15165 | 4.01 | 3.57 | +0.03 | 2.01 | 0.86 | 0.77 |  | 2.32 | 2.99 |
| P | oregon only | all | 15672 | 4.06 | 3.61 | +0.03 | 2.02 | 0.86 | 0.77 |  | 2.31 | 2.98 |
| S | blanco only | resolved | 336 | 5.39 | 5.22 | +0.27 | 1.14 | 0.98 | 0.95 | 1.20 |  |  |
| S | blanco only | unresolved | 9384 | 3.49 | 3.13 | -0.06 | 0.81 | 0.98 | 0.88 | 0.80 |  |  |
| S | blanco only | all | 9720 | 3.57 | 3.23 | -0.05 | 0.83 | 0.98 | 0.88 | 0.81 |  |  |
| S | near seam blanco | resolved | 3 | 4.82 | 4.98 | -0.67 | 1.68 | 0.93 | 0.97 | 0.77 |  |  |
| S | near seam blanco | unresolved | 1617 | 2.15 | 2.03 | -0.10 | 0.89 | 0.91 | 0.86 | 0.63 |  |  |
| S | near seam blanco | all | 1620 | 2.15 | 2.03 | -0.10 | 0.89 | 0.91 | 0.86 | 0.63 |  |  |
| S | overlap | unresolved | 1620 | 2.13 | 1.45 | -0.01 | 1.36 | 0.77 | 0.53 | 0.36 | 0.58 | 1.26 |
| S | overlap | all | 1620 | 2.13 | 1.45 | -0.01 | 1.36 | 0.77 | 0.53 | 0.36 | 0.58 | 1.26 |
| S | near seam oregon | unresolved | 1860 | 2.46 | 1.94 | -0.15 | 1.08 | 0.91 | 0.72 |  | 0.67 | 1.45 |
| S | near seam oregon | all | 1860 | 2.46 | 1.94 | -0.15 | 1.08 | 0.91 | 0.72 |  | 0.67 | 1.45 |
| S | oregon only | resolved | 1367 | 5.47 | 4.89 | -0.02 | 1.40 | 0.97 | 0.87 |  | 1.36 | 3.06 |
| S | oregon only | unresolved | 14305 | 2.42 | 2.13 | +0.02 | 0.74 | 0.95 | 0.84 |  | 0.92 | 1.94 |
| S | oregon only | all | 15672 | 2.82 | 2.49 | +0.02 | 0.82 | 0.96 | 0.85 |  | 0.97 | 2.06 |

## Same events: joint against each tile

Events in a tile run but not relocated in the joint run: {'blanco': 3, 'oregon': 13}.

| tile | zone | n | horiz med / p90 km | depth med km | abs depth med / p90 km | tile's own shift h / abs z med km |
|---|---|---|---|---|---|---|
| blanco | blanco only | 4229 | 0.24 / 0.67 | +0.02 | 0.46 / 2.06 | 0.94 / 1.75 |
| blanco | near seam blanco | 66 | 0.48 / 1.00 | +0.28 | 0.95 / 3.67 | 1.03 / 1.69 |
| blanco | overlap | 76 | 0.84 / 3.24 | +0.08 | 1.12 / 4.01 | 1.49 / 1.71 |
| oregon | near seam oregon | 9 | 0.68 / 1.52 | -0.40 | 1.27 / 6.83 | 2.39 / 1.21 |
| oregon | oregon only | 3264 | 0.27 / 0.81 | +0.02 | 0.34 / 1.38 | 0.95 / 1.11 |
| oregon | overlap | 76 | 1.09 / 2.76 | -0.57 | 1.72 / 6.23 | 1.46 / 1.59 |
