# B5 run b5_v0.3.0 against the D-B048 test (tomoDD dd) on the Blanco cases

Catalog /wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.0/catalog_b5_b5_v0.3.0_partial.csv. Common set = events relocated by both. Differences in km; 'c.r.' = each solution's centroid removed.

| case | test relocated | B5 relocated (of the case) | common | B5 clusters | pair | h diff c.r. med (p90) | z diff c.r. med (p90) | pair-distance diff med | centroid shift h / z |
|---|---|---|---|---|---|---|---|---|---|
| f0399 | 1088 | 1010 of 1455 | 1004 | 1 | B5 - test | 0.334 (1.101) | 0.836 (3.137) | 1.019 | 0.012 / +0.313 |
| | | | | | B5 - B4 | 0.766 (2.090) | 1.875 (5.976) | 2.090 | 0.090 / +0.924 |
| | | | | | test - B4 | 0.657 (1.762) | 1.245 (4.084) | 1.377 | 0.102 / +0.610 |

Shape of the common set:

| case | solution | PCA sigma1 / 2 / 3 | strike / dip | depth p10 / med / p90 |
|---|---|---|---|---|
| f0399 | B4 | 4.88 / 2.08 / 1.09 | 130 / 90 | 7.5 / 8.9 / 19.0 |
| f0399 | test dd | 5.13 / 2.24 / 1.18 | 132 / 89 | 6.7 / 9.5 / 19.8 |
| f0399 | B5 | 5.34 / 2.36 / 1.29 | 135 / 89 | 6.4 / 10.2 / 20.1 |
