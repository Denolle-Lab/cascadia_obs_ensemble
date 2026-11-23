# B1b balanced dataset: tables

Written by workflow/10_tomography/b1b_balance.py.

## Metrics

| set | picks | kish_n_eff | stations | events | gini_station | top50_station_share | obs_share | S_share | pct_Axial | pct_Blanco | pct_Cascades | pct_Explorer | pct_Gorda | pct_JdF ridge/plate | pct_Mendocino | pct_Offshore forearc | pct_Other onshore | pct_Puget/Georgia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 692246 | 692246 | 646 | 23428 | 0.653 | 0.369 | 0.269 | 0.504 | 0.1 | 12.3 | 12.1 | 0.1 | 7.8 | 0.0 | 15.8 | 8.4 | 19.2 | 24.2 |
| B1b core, unweighted | 677235 | 677235 | 424 | 22976 | 0.496 | 0.375 | 0.269 | 0.504 |  | 12.2 | 12.2 | 0.0 | 7.9 | 0.0 | 16.0 | 8.1 | 19.2 | 24.4 |
| B1b w_bin | 677235 | 546111 | 424 | 22976 | 0.416 | 0.296 | 0.253 | 0.501 |  | 8.9 | 14.3 | 0.0 | 8.0 | 0.0 | 11.7 | 8.2 | 23.2 | 25.8 |
| B1b w_bal | 677235 | 461804 | 424 | 22976 | 0.389 | 0.286 | 0.393 | 0.512 |  | 14.4 | 14.3 | 0.0 | 14.4 | 0.0 | 14.4 | 13.8 | 14.3 | 14.4 |

## Region weights

| region | w_region |
|---|---|
| Blanco | 1.62 |
| Cascades | 0.995 |
| Explorer | 1.0 |
| Gorda | 1.799 |
| JdF ridge/plate | 1.0 |
| Mendocino | 1.23 |
| Offshore forearc | 1.684 |
| Other onshore | 0.619 |
| Puget/Georgia | 0.558 |

## Stations admitted only by the OBS bar (main region of their picks)

| main_region | stations | P | S |
|---|---|---|---|
| Blanco | 1 | 7 | 83 |
| Explorer | 6 | 228 | 207 |
| Gorda | 7 | 335 | 486 |
| Mendocino | 1 | 71 | 10 |
| Offshore forearc | 6 | 178 | 333 |

## Dropped stations by reason

| is_obs | fail | stations |
|---|---|---|
| False | nP | 35 |
| False | nP azimuth | 34 |
| False | nP azimuth distance | 38 |
| False | nP distance | 2 |
| True | azimuth | 1 |
| True | distance | 4 |
| True | nP | 2 |
| True | nP azimuth | 4 |
| True | nP azimuth distance | 91 |
| True | nP distance | 10 |

## Dropped stations by main region

| main_region | stations | P | S |
|---|---|---|---|
| Axial | 8 | 212 | 174 |
| Blanco | 8 | 673 | 1148 |
| Cascades | 14 | 305 | 287 |
| Explorer | 16 | 338 | 274 |
| Gorda | 9 | 67 | 146 |
| JdF ridge/plate | 11 | 18 | 19 |
| Mendocino | 7 | 116 | 42 |
| Offshore forearc | 64 | 712 | 750 |
| Other onshore | 37 | 894 | 644 |
| Puget/Georgia | 47 | 1293 | 917 |

## Events by region

| region | B1 | kept |
|---|---|---|
| Axial | 29 | 0 |
| Blanco | 4216 | 4192 |
| Cascades | 1818 | 1758 |
| Explorer | 71 | 1 |
| Gorda | 2553 | 2526 |
| JdF ridge/plate | 9 | 2 |
| Mendocino | 4051 | 4032 |
| Offshore forearc | 2313 | 2179 |
| Other onshore | 4113 | 4056 |
| Puget/Georgia | 4255 | 4230 |

## Log

```
B1 input: 692246 picks, 23428 events, 646 stations
core stations: 425 of 646 (OBS 174, land 251); admitted only by the OBS bar: 21; only by the S test: 15
region weights: {'Blanco': 1.62, 'Cascades': 0.995, 'Explorer': 1.0, 'Gorda': 1.799, 'JdF ridge/plate': 1.0, 'Mendocino': 1.23, 'Offshore forearc': 1.684, 'Other onshore': 0.619, 'Puget/Georgia': 0.558}
events kept: 22976 of 23428; picks kept 677235; dropped 15011 (station 9029, event 5982)
| set | picks | kish_n_eff | stations | events | gini_station | top50_station_share | obs_share | S_share | pct_Axial | pct_Blanco | pct_Cascades | pct_Explorer | pct_Gorda | pct_JdF ridge/plate | pct_Mendocino | pct_Offshore forearc | pct_Other onshore | pct_Puget/Georgia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 692246 | 692246 | 646 | 23428 | 0.653 | 0.369 | 0.269 | 0.504 | 0.1 | 12.3 | 12.1 | 0.1 | 7.8 | 0.0 | 15.8 | 8.4 | 19.2 | 24.2 |
| B1b core, unweighted | 677235 | 677235 | 424 | 22976 | 0.496 | 0.375 | 0.269 | 0.504 |  | 12.2 | 12.2 | 0.0 | 7.9 | 0.0 | 16.0 | 8.1 | 19.2 | 24.4 |
| B1b w_bin | 677235 | 546111 | 424 | 22976 | 0.416 | 0.296 | 0.253 | 0.501 |  | 8.9 | 14.3 | 0.0 | 8.0 | 0.0 | 11.7 | 8.2 | 23.2 | 25.8 |
| B1b w_bal | 677235 | 461804 | 424 | 22976 | 0.389 | 0.286 | 0.393 | 0.512 |  | 14.4 | 14.3 | 0.0 | 14.4 | 0.0 | 14.4 | 13.8 | 14.3 | 14.4 |
```