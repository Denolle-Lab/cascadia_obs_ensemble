# B1b balanced dataset: tables

Written by workflow/10_tomography/b1b_balance.py.

## Metrics

| set | picks | kish_n_eff | stations | events | gini_station | top50_station_share | obs_share | S_share | pct_Blanco | pct_Cascades | pct_Gorda | pct_Mendocino | pct_Offshore forearc | pct_Other onshore | pct_Puget/Georgia | pct_Ridges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 703559 | 703559 | 658 | 24621 | 0.645 | 0.363 | 0.28 | 0.503 | 12.1 | 12.0 | 7.7 | 15.5 | 8.2 | 18.9 | 23.8 | 1.8 |
| B1b core, unweighted | 689602 | 689602 | 475 | 24249 | 0.529 | 0.368 | 0.281 | 0.503 | 12.0 | 12.0 | 7.8 | 15.7 | 7.9 | 18.9 | 23.9 | 1.8 |
| B1b w_bin | 689602 | 557761 | 475 | 24249 | 0.453 | 0.289 | 0.268 | 0.5 | 8.7 | 14.0 | 7.8 | 11.4 | 7.9 | 22.6 | 25.2 | 2.4 |
| B1b w_bal | 689602 | 431050 | 475 | 24249 | 0.401 | 0.267 | 0.431 | 0.508 | 13.3 | 13.2 | 13.3 | 13.4 | 12.7 | 13.3 | 13.3 | 7.5 |

## Region weights

| group | w_region |
|---|---|
| Blanco | 1.45 |
| Cascades | 0.89 |
| Gorda | 1.61 |
| Mendocino | 1.101 |
| Offshore forearc | 1.51 |
| Other onshore | 0.554 |
| Puget/Georgia | 0.499 |
| Ridges | 3.0 |

## Stations admitted only by the OBS bar (main region of their picks)

| main_region | stations | P | S |
|---|---|---|---|
| Blanco | 1 | 7 | 83 |
| Gorda | 7 | 335 | 486 |
| Mendocino | 1 | 71 | 10 |
| Offshore forearc | 5 | 173 | 302 |

## Dropped stations by reason

| is_obs | fail | stations |
|---|---|---|
| False | nP | 33 |
| False | nP azimuth | 33 |
| False | nP azimuth distance | 41 |
| False | nP distance | 2 |
| True | azimuth | 1 |
| True | distance | 4 |
| True | nP | 1 |
| True | nP azimuth | 3 |
| True | nP azimuth distance | 119 |
| True | nP distance | 3 |

## Dropped stations by main region

| main_region | stations | P | S |
|---|---|---|---|
| Axial | 8 | 0 | 0 |
| Blanco | 7 | 657 | 1145 |
| Cascades | 14 | 305 | 287 |
| Explorer | 29 | 131 | 128 |
| Gorda | 9 | 67 | 146 |
| JdF ridge/plate | 28 | 16 | 13 |
| Mendocino | 7 | 116 | 42 |
| Offshore forearc | 54 | 650 | 674 |
| Other onshore | 37 | 894 | 644 |
| Puget/Georgia | 47 | 1293 | 917 |

## Events by region

| region | rule | B1 | kept |
|---|---|---|---|
| Axial | ridge | 428 | 428 |
| Blanco | main | 4216 | 4192 |
| Cascades | main | 1818 | 1758 |
| Explorer | ridge | 403 | 403 |
| Gorda | main | 2553 | 2526 |
| JdF ridge/plate | ridge | 471 | 463 |
| Mendocino | main | 4051 | 4032 |
| Offshore forearc | main | 2313 | 2161 |
| Other onshore | main | 4113 | 4056 |
| Puget/Georgia | main | 4255 | 4230 |

## Log

```
B1 input: 703559 picks, 24621 events, 658 stations
core stations: 418 of 658 (OBS 167, land 251); admitted only by the OBS bar: 14; only by the S test: 13
ridge-only stations (>= 10 ridge picks, no coverage test): 57
region weights: {'Blanco': 1.45, 'Cascades': 0.89, 'Gorda': 1.61, 'Mendocino': 1.101, 'Offshore forearc': 1.51, 'Other onshore': 0.554, 'Puget/Georgia': 0.499, 'Ridges': 3.0}
events kept: 24249 of 24621; picks kept 689602; dropped 13957 (station 8180, event 5777)
| set | picks | kish_n_eff | stations | events | gini_station | top50_station_share | obs_share | S_share | pct_Blanco | pct_Cascades | pct_Gorda | pct_Mendocino | pct_Offshore forearc | pct_Other onshore | pct_Puget/Georgia | pct_Ridges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 703559 | 703559 | 658 | 24621 | 0.645 | 0.363 | 0.28 | 0.503 | 12.1 | 12.0 | 7.7 | 15.5 | 8.2 | 18.9 | 23.8 | 1.8 |
| B1b core, unweighted | 689602 | 689602 | 475 | 24249 | 0.529 | 0.368 | 0.281 | 0.503 | 12.0 | 12.0 | 7.8 | 15.7 | 7.9 | 18.9 | 23.9 | 1.8 |
| B1b w_bin | 689602 | 557761 | 475 | 24249 | 0.453 | 0.289 | 0.268 | 0.5 | 8.7 | 14.0 | 7.8 | 11.4 | 7.9 | 22.6 | 25.2 | 2.4 |
| B1b w_bal | 689602 | 431050 | 475 | 24249 | 0.401 | 0.267 | 0.431 | 0.508 | 13.3 | 13.2 | 13.3 | 13.4 | 12.7 | 13.3 | 13.3 | 7.5 |
```