**Choice: stop the SSST schedule at 4 km** (9999, 16, 8, 4 km), by the rule written in RESUME.md before any run of this test. With the coherence stage, the held-out score (MAD P + MAD S, mean of the two clusters) is 81.0 ms at 16 km, 78.4 at 8 km, 76.8 at 4 km and 80.2 at 2 km. The next coarser stop, 8 km, is worse than 4 km by 1.6 ms (95 % interval 0.3 to 2.7 ms), so the rule keeps 4 km. Stopping at 2 km costs 3.4 ms (1.8 to 4.6) against 4 km. The gaps are small, 2 to 5 % of the score, but they point the same way as the post-hoc hint from the first three clusters.

### Clusters

Two new clusters that share no event with the seven earlier ones: 700 events in the Strait of Georgia (onshore, about 15 km deep, CN, PB, UW and TA stations) and 800 events of the 2014 M6.8 Gorda plate sequence (ZBZ family 1162, Z5 and 7D ocean-bottom seismometers with NC, PB, BK and TA land stations). CC data came from the same tested code: 42,000 accepted rows at Georgia and 23,400 at Gorda. Each run used one SSST chain, with oct-trees saved after the 16, 8, 4 and 2 km passes and the coherence stage run on each.

### Held-out stations, per cluster

- Georgia, with coherence, P / S MAD: 20.1 / 31.5 ms at 16 km, 20.2 / 30.8 at 8 km, 19.2 / 29.6 at 4 km, 20.4 / 31.1 at 2 km. The 4 km stop is lowest on both phases.
- Gorda, with coherence: 48.1 / 62.2 ms at 16 km, 46.7 / 59.2 at 8 km, 47.1 / 57.7 at 4 km, 49.2 / 59.7 at 2 km. The curve is flat between 8 and 4 km, and 2 and 16 km are both worse.

### Does the coherence stage help?

At Georgia it helps a lot, at every stop: P drops by about 15 ms (34.0 to 19.2 ms at 4 km) and S by about 21 ms, to values below those of the v5.1 catalog, which saw the held-out stations (26.5 / 43.4 ms). At Gorda it changes nothing that matters (-0.4 to +1.2 ms). There the stage relocates few events, because most Gorda events have error ellipsoids longer than 5 km: 0.5 to 22 % of the located events in the fold runs, against 41 to 53 % at Georgia. Over the five independent clusters tested today, coherence helped where it relocates many events (Puget, Georgia), hurt at Mendocino, and was neutral where it relocates few (Gorda, western Blanco).

### Other criteria (full runs)

- CC fit (median per-pair residual): Georgia lowest at 4 km with coherence (13.8 ms; 14.0 at 8 km, 14.3 at 2 km). At Gorda the 16 and 8 km stops fit best (18.8 and 18.6 ms), 4 km is at 20.1 ms and 2 km at 25.9 ms, so the 2 km stop is clearly worse.
- Repeaters: Georgia 168 pairs, median separation 0.24 km at 4 km with coherence (0.26 to 0.29 km at the other stops, 0.58 km in v5.1). Gorda has only 15 pairs; its intervals overlap at every stop.
- Depths: no stop is distinguishable from another. All arms sit about 2.7 km deeper than PNSN at Georgia (138 events; v5.1 2.4 km) and about 6 km deeper than NCSN at Gorda (27 events, the known Mendocino-area offset).
- Above the surface: 0 events in every arm. Floor-limited: 9 to 16 events at Georgia, 0 or 1 at Gorda.
