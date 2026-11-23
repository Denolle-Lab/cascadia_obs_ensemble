# Tiles compared in their overlaps (b2_tile_overlap.py; notes/merge_design.md T1, T2)

Pooled over 1-34 km. dlnV = change from each tile's own starting model. 'both' = resolved in both tiles' checkerboard masks at their damping (smoothing 20); 'one' = in one only; 'no mask' = a mask is missing for that damping. diff = dlnV_A - dlnV_B (mean and rms, %); amp = B/A amplitude ratio (least squares through 0); corr = Pearson; start = |V0_B / V0_A - 1| of the two starting models (median / max, %).

| A | B | class | n | rms A % | rms B % | diff mean % | diff rms % | corr | amp B/A | start med / max % |
|---|---|---|---|---|---|---|---|---|---|---|
| blanco_v2 d1000_s20 | oregon_v2 d1000_s20 | neither | 1620 | 1.36 | 2.24 | +0.13 | 2.03 | 0.40 | 0.74 | 0.01 / 1.05 |
| blanco_v2 d500_s20 | oregon_v2 d500_s20 | no mask | 1620 | 1.74 | 2.72 | +0.06 | 2.61 | 0.33 | 0.59 | 0.01 / 1.05 |
| blanco_v2 d300_s20 | oregon_v2 d300_s20 | neither | 1620 | 2.03 | 2.94 | -0.02 | 2.95 | 0.29 | 0.49 | 0.01 / 1.05 |
| blanco_v2 d300_s20 | oregon_v2 d1000_s20 | neither | 1620 | 2.03 | 2.24 | -0.12 | 2.38 | 0.32 | 0.42 | 0.01 / 1.05 |
| blanco_v2 d1000_s20 | oregon_v2 d300_s20 | neither | 1620 | 1.36 | 2.94 | +0.23 | 2.72 | 0.34 | 0.84 | 0.01 / 1.05 |
| mendocino_gorda_c_v2 d1000_s20 | blanco_v2 d1000_s20 | one | 44 | 8.29 | 4.03 | -1.56 | 6.68 | 0.48 | 0.29 | 0.01 / 0.18 |
| mendocino_gorda_c_v2 d1000_s20 | blanco_v2 d1000_s20 | neither | 1107 | 5.05 | 2.54 | -0.48 | 4.52 | 0.42 | 0.23 | 0.02 / 7.59 |
| mendocino_gorda_c_v2 d500_s20 | blanco_v2 d500_s20 | no mask | 1152 | 6.23 | 3.44 | -0.52 | 5.59 | 0.42 | 0.25 | 0.02 / 7.59 |
| mendocino_gorda_c_v2 d1000_s20 | blanco_v2 d300_s20 | one | 66 | 7.67 | 5.48 | -1.80 | 6.62 | 0.40 | 0.38 | 0.01 / 0.18 |
| mendocino_gorda_c_v2 d1000_s20 | blanco_v2 d300_s20 | neither | 1084 | 5.02 | 3.84 | -0.22 | 4.87 | 0.39 | 0.32 | 0.03 / 7.59 |
| mendocino_gorda_c_v2 d1000_s20 | oregon_v2 d1000_s20 | one | 6 | 7.04 | 4.22 | -3.78 | 6.33 | 0.30 | 0.28 | 0.02 / 0.37 |
| mendocino_gorda_c_v2 d1000_s20 | oregon_v2 d1000_s20 | neither | 2154 | 2.89 | 2.42 | -0.30 | 3.12 | 0.29 | 0.27 | 0.05 / 4.03 |
| mendocino_gorda_c_v2 d500_s20 | oregon_v2 d500_s20 | no mask | 2160 | 3.56 | 3.09 | -0.35 | 3.85 | 0.31 | 0.29 | 0.05 / 4.03 |
| mendocino_gorda_c_v2 d1000_s20 | oregon_v2 d300_s20 | one | 6 | 7.04 | 6.23 | -4.23 | 7.48 | 0.37 | 0.33 | 0.02 / 0.37 |
| mendocino_gorda_c_v2 d1000_s20 | oregon_v2 d300_s20 | neither | 2154 | 2.89 | 3.58 | -0.25 | 3.83 | 0.29 | 0.39 | 0.05 / 4.03 |
| blanco_v2 d1000_s20 | ridges_north_v2 d1000_s20 | one | 14 | 4.14 | 3.75 | +1.30 | 2.44 | 0.87 | 0.74 | 0.01 / 0.06 |
| blanco_v2 d1000_s20 | ridges_north_v2 d1000_s20 | neither | 3442 | 2.82 | 2.27 | -0.09 | 2.44 | 0.52 | 0.45 | 0.01 / 29.99 |
| blanco_v2 d500_s20 | ridges_north_v2 d500_s20 | no mask | 3456 | 3.38 | 2.78 | -0.11 | 3.10 | 0.47 | 0.42 | 0.01 / 29.99 |
| blanco_v2 d300_s20 | ridges_north_v2 d200_s20 | one | 21 | 6.87 | 6.29 | -1.86 | 9.69 | -0.10 | -0.08 | 0.01 / 0.03 |
| blanco_v2 d300_s20 | ridges_north_v2 d200_s20 | neither | 3435 | 3.71 | 3.21 | -0.10 | 3.46 | 0.48 | 0.44 | 0.01 / 29.99 |
| blanco_v2 d1000_s20 | ridges_north_v2 d200_s20 | one | 14 | 4.14 | 7.88 | +2.28 | 5.53 | 0.72 | 1.42 | 0.01 / 0.06 |
| blanco_v2 d1000_s20 | ridges_north_v2 d200_s20 | neither | 3442 | 2.82 | 3.21 | +0.00 | 2.97 | 0.49 | 0.59 | 0.01 / 29.99 |
| oregon_v2 d1000_s20 | washington_v2 d1000_s20 | no mask | 3528 | 3.60 | 4.97 | -0.19 | 4.28 | 0.53 | 0.75 | 0.05 / 4.15 |
| oregon_v2 d500_s20 | washington_v2 d500_s20 | no mask | 3528 | 4.42 | 6.04 | -0.13 | 5.59 | 0.46 | 0.64 | 0.05 / 4.15 |
| oregon_v2 d300_s20 | washington_v2 d1000_s20 | no mask | 3528 | 5.03 | 4.97 | -0.15 | 5.20 | 0.45 | 0.45 | 0.05 / 4.15 |

## Same events relocated in two tiles (events inside the overlap box)

| A | B | n | horiz A-B med / p90 km | depth A-B med km | abs depth A-B med / p90 km | horiz shift A / B med km | abs depth shift A / B med km |
|---|---|---|---|---|---|---|---|
| blanco_v2 d1000_s20 | oregon_v2 d1000_s20 | 77 | 0.63 / 1.84 | -0.07 | 0.36 / 2.45 | 0.73 / 0.51 | 0.57 / 0.38 |
| blanco_v2 d500_s20 | oregon_v2 d500_s20 | 77 | 0.80 / 2.53 | +0.02 | 0.86 / 3.05 | 0.91 / 0.75 | 0.85 / 0.54 |
| blanco_v2 d300_s20 | oregon_v2 d300_s20 | 77 | 1.00 / 2.61 | +0.08 | 1.37 / 3.99 | 1.26 / 1.01 | 1.31 / 0.85 |
| blanco_v2 d300_s20 | oregon_v2 d1000_s20 | 77 | 1.04 / 2.43 | -0.09 | 1.38 / 4.13 | 1.26 / 0.51 | 1.31 / 0.38 |
| blanco_v2 d1000_s20 | oregon_v2 d300_s20 | 77 | 1.04 / 2.35 | -0.07 | 0.79 / 3.50 | 0.73 / 1.01 | 0.57 / 0.85 |
| mendocino_gorda_c_v2 d1000_s20 | blanco_v2 d1000_s20 | 103 | 0.36 / 0.91 | -0.20 | 0.38 / 1.62 | 0.64 / 0.56 | 0.58 / 0.49 |
| mendocino_gorda_c_v2 d500_s20 | blanco_v2 d500_s20 | 103 | 0.55 / 1.20 | -0.33 | 0.83 / 3.06 | 0.87 / 0.81 | 0.93 / 0.81 |
| mendocino_gorda_c_v2 d1000_s20 | blanco_v2 d300_s20 | 103 | 0.66 / 1.10 | -0.04 | 0.96 / 3.33 | 0.64 / 0.94 | 0.58 / 1.09 |
| mendocino_gorda_c_v2 d1000_s20 | oregon_v2 d1000_s20 | 32 | 0.37 / 0.83 | +0.02 | 0.58 / 2.00 | 0.47 / 0.34 | 0.58 / 0.41 |
| mendocino_gorda_c_v2 d500_s20 | oregon_v2 d500_s20 | 32 | 0.47 / 1.54 | +0.28 | 0.85 / 2.42 | 0.60 / 0.43 | 0.96 / 0.62 |
| mendocino_gorda_c_v2 d1000_s20 | oregon_v2 d300_s20 | 32 | 0.53 / 1.40 | +0.09 | 0.96 / 3.66 | 0.47 / 0.49 | 0.58 / 0.67 |
| blanco_v2 d1000_s20 | ridges_north_v2 d1000_s20 | 281 | 0.27 / 0.75 | +0.12 | 0.21 / 0.70 | 0.49 / 0.32 | 0.37 / 0.25 |
| blanco_v2 d500_s20 | ridges_north_v2 d500_s20 | 281 | 0.35 / 1.12 | +0.16 | 0.34 / 1.27 | 0.62 / 0.48 | 0.60 / 0.37 |
| blanco_v2 d300_s20 | ridges_north_v2 d200_s20 | 280 | 0.33 / 0.90 | +0.22 | 0.48 / 1.34 | 0.78 / 0.63 | 0.84 / 0.58 |
| blanco_v2 d1000_s20 | ridges_north_v2 d200_s20 | 281 | 0.33 / 1.06 | +0.03 | 0.33 / 1.17 | 0.49 / 0.63 | 0.37 / 0.58 |
| oregon_v2 d1000_s20 | washington_v2 d1000_s20 | 1568 | 0.21 / 0.67 | -0.11 | 0.20 / 0.69 | 0.43 / 0.54 | 0.33 / 0.50 |
| oregon_v2 d500_s20 | washington_v2 d500_s20 | 1563 | 0.27 / 0.85 | -0.16 | 0.29 / 1.02 | 0.60 / 0.69 | 0.59 / 0.83 |
| oregon_v2 d300_s20 | washington_v2 d1000_s20 | 1562 | 0.33 / 1.06 | +0.18 | 0.40 / 1.33 | 0.70 / 0.54 | 0.83 / 0.50 |
