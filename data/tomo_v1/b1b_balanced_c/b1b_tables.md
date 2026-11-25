# B1b balanced dataset: tables

Written by workflow/10_tomography/b1b_balance.py.

## Metrics

| set | picks | kish_n_eff | stations | events | gini_station | top50_station_share | obs_share | S_share | pct_Blanco | pct_Cascades | pct_Gorda | pct_Mendocino | pct_Offshore forearc | pct_Other onshore | pct_Puget/Georgia | pct_Ridges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 765486 | 765486 | 666 | 29205 | 0.655 | 0.378 | 0.265 | 0.501 | 11.1 | 11.0 | 7.1 | 14.4 | 15.6 | 17.4 | 21.9 | 1.7 |
| B1b core, unweighted | 754710 | 754710 | 521 | 28986 | 0.573 | 0.382 | 0.267 | 0.501 | 11.1 | 10.9 | 7.1 | 14.5 | 15.8 | 17.1 | 21.8 | 1.7 |
| B1b w_bin | 754710 | 586862 | 521 | 28986 | 0.486 | 0.283 | 0.265 | 0.499 | 8.5 | 13.4 | 7.4 | 10.7 | 12.6 | 21.3 | 24.0 | 2.2 |
| B1b w_bal | 754710 | 468522 | 521 | 28986 | 0.438 | 0.256 | 0.423 | 0.506 | 13.3 | 13.2 | 13.2 | 13.3 | 13.3 | 13.2 | 13.3 | 7.1 |

## Region weights

| group | w_region |
|---|---|
| Blanco | 1.486 |
| Cascades | 0.928 |
| Gorda | 1.693 |
| Mendocino | 1.175 |
| Offshore forearc | 0.999 |
| Other onshore | 0.584 |
| Puget/Georgia | 0.522 |
| Ridges | 3.0 |

## Stations admitted only by the OBS bar (main region of their picks)

| main_region | stations | P | S |
|---|---|---|---|
| Gorda | 8 | 365 | 555 |
| Mendocino | 2 | 142 | 106 |
| Offshore forearc | 1 | 49 | 72 |

## Dropped stations by reason

| is_obs | fail | stations |
|---|---|---|
| False | azimuth | 1 |
| False | distance | 1 |
| False | nP | 34 |
| False | nP azimuth | 30 |
| False | nP azimuth distance | 52 |
| False | nP distance | 4 |
| True | azimuth distance | 1 |
| True | distance | 7 |
| True | nP | 1 |
| True | nP azimuth | 2 |
| True | nP azimuth distance | 139 |

## Dropped stations by main region

| main_region | stations | P | S |
|---|---|---|---|
| Axial | 8 | 0 | 0 |
| Blanco | 12 | 1603 | 2389 |
| Cascades | 14 | 305 | 287 |
| Explorer | 27 | 0 | 4 |
| Gorda | 8 | 64 | 132 |
| JdF ridge/plate | 25 | 10 | 0 |
| Mendocino | 7 | 103 | 32 |
| Offshore forearc | 89 | 1022 | 830 |
| Other onshore | 36 | 814 | 590 |
| Puget/Georgia | 46 | 1267 | 895 |

## Events by region

| region | rule | B1 | kept |
|---|---|---|---|
| Axial | ridge | 428 | 428 |
| Blanco | main | 4216 | 4214 |
| Cascades | main | 1818 | 1758 |
| Explorer | ridge | 403 | 403 |
| Gorda | main | 2554 | 2525 |
| JdF ridge/plate | ridge | 471 | 463 |
| Mendocino | main | 4082 | 4063 |
| Offshore forearc | ridge | 6856 | 6847 |
| Other onshore | main | 4122 | 4055 |
| Puget/Georgia | main | 4255 | 4230 |

## Log

```
B1 input: 765486 picks, 29205 events, 666 stations
core stations: 394 of 666 (OBS 156, land 238); admitted only by the OBS bar: 11; only by the S test: 10
D-B035: 88 soft-bar OBS keep all their picks (+3346)
ridge-only stations (>= 10 ridge picks, no coverage test): 127
region weights: {'Blanco': 1.486, 'Cascades': 0.928, 'Gorda': 1.693, 'Mendocino': 1.175, 'Offshore forearc': 0.999, 'Other onshore': 0.584, 'Puget/Georgia': 0.522, 'Ridges': 3.0}
events kept: 28986 of 29205; picks kept 754710; dropped 10776 (station 7271, event 3505)
| set | picks | kish_n_eff | stations | events | gini_station | top50_station_share | obs_share | S_share | pct_Blanco | pct_Cascades | pct_Gorda | pct_Mendocino | pct_Offshore forearc | pct_Other onshore | pct_Puget/Georgia | pct_Ridges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 765486 | 765486 | 666 | 29205 | 0.655 | 0.378 | 0.265 | 0.501 | 11.1 | 11.0 | 7.1 | 14.4 | 15.6 | 17.4 | 21.9 | 1.7 |
| B1b core, unweighted | 754710 | 754710 | 521 | 28986 | 0.573 | 0.382 | 0.267 | 0.501 | 11.1 | 10.9 | 7.1 | 14.5 | 15.8 | 17.1 | 21.8 | 1.7 |
| B1b w_bin | 754710 | 586862 | 521 | 28986 | 0.486 | 0.283 | 0.265 | 0.499 | 8.5 | 13.4 | 7.4 | 10.7 | 12.6 | 21.3 | 24.0 | 2.2 |
| B1b w_bal | 754710 | 468522 | 521 | 28986 | 0.438 | 0.256 | 0.423 | 0.506 | 13.3 | 13.2 | 13.2 | 13.3 | 13.3 | 13.2 | 13.3 | 7.1 |
```