# Magnitude of completeness maps, catalog v4

Built by `workflow/06_analysis/v4_mc_maps.py` on 2026-11-26 with the converged Route A magnitudes (decision D-B091:
phase19 solved exactly, PR #45; `/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv`,
md5 fb49ecc28b75432f49cf1a6e399cd90f, recorded in `run_params.json`). Inputs: `Cascadia_relocated_catalog_v4.csv`
(catalog v4.1, 172,342 events) and those magnitudes (130,769 events with M; 99.9% ML).

Events flagged as not earthquakes are left out (decision D-B105, PI 2026-11-26): 3,298 of the 130,769, leaving 127,471.
The v4 catalog has no flag column, so an event is left out when catalog v5.1 or catalog v6.0 flags it (`event_flag`:
weekday daytime events in the Texada Island quarry box, or a ComCat explosion, quarry blast or other event within 3 s
and 20 km; `workflow/10_tomography/b4_catalog.py`, D-B071). v5.1 alone flags 3,280 of them and v6.0 alone 3,289; the
two disagree on 27 ComCat explosion / quarry-blast matches near the 3 s / 20 km limits. The flag files and their md5s
are in `run_params.json`. The maps that kept these events are on main up to 08d9dbe4 (and in
`/wd1/mdenolle_data/magnitude_converged/_superseded_2026-11-26_mc_with_blasts/mc_maps_v41`).

Rerun with `python v4_mc_maps.py --yearly --exclude-flagged --flags <catalog_v5/v5.1/cascadia_catalog_v5.1.csv>
<catalog_v6/v6.0/cascadia_catalog_v6.0.csv>` (files under `/wd1/mdenolle_data/`; about 20 s of compute on 8 cores, plus
1 min of plotting). Without `--exclude-flagged` the script gives the maps with the flagged events; `--mag stored` gives
the maps of 2026-11-22 (stored magnitudes, LSQR stopped at 3,000 iterations; on main up to 5d6658bf).

## Method

Magnitudes are rounded to 0.1. Three estimators run on each sample:

- **MAXC + 0.2**: centre of the most populated bin of the non-cumulative FMD (Wiemer & Wyss 2000), plus the +0.2 correction of Woessner & Wiemer (2005).
- **MBS**, b-value stability (Cao & Gao 2002) in the form of Woessner & Wiemer (2005): the lowest cutoff where the mean of b over the next five bins (a 0.5 window) differs from b(Mc) by less than the Shi & Bolt (1982) error of b(Mc). b is the Aki (1965) / Utsu (1965) estimate with the bin-edge correction.
- **GFT**, goodness of fit (Wiemer & Wyss 2000): the lowest cutoff where a GR law fitted above it reproduces the observed cumulative counts with R >= 90 % (GFT90) or R >= 95 % (GFT95).
- **best**: the rule of Woessner & Wiemer (2005): GFT95 if reached, else GFT90, else MAXC+0.2. Called "GFT (best)" in the figures.

Every b fit needs 30 events above the cutoff. Uncertainty comes from 200 bootstrap resamples of each sample; `sigma_boot` is the standard deviation of the bootstrap Mc_best (columns `sigma_maxc`, `sigma_mbs`, `sigma_gft` give the same for each method). `b` and `b_sigma` are fitted above Mc_best. The MAXC, Aki-Utsu, Shi-Bolt and b-stability code is adapted from AgentSeismo (Z. Zhang), `studies/gibraltar_2010_2011/02_assessment/mc_extended.py`; the MBS test uses the Shi-Bolt error instead of that script's fixed 0.03 tolerance, and GFT was added.

Choices:

- Grid every 0.1 deg over lon -131 to -119.5, lat 39 to 51. A node takes its nearest 250 events with M (not flagged as non-earthquakes) if all of them lie within 50 km (epicentral distance); otherwise it stays empty. For the period maps the sample is 150 events (same 50 km cap), because a single year has too few events for 250 almost everywhere offshore. The cost shows in `sigma_boot`.
- Events of any QC are used. The variant with at least 4 P and 4 S picks is in `mc_grid_qc.csv`, `mc_best_qc` in `mc_grid.nc`, and Figure 5.
- Regions are the boxes of `v4_clusters_zbz.py`, most specific first. OFF pools the offshore boxes EN, BL, GO, WA and OR. The regional Mc comes from the pooled regional FMD and mixes areas with different Mc; the node percentiles in `mc_nodes_by_region.csv` show the spread inside each box.
- Rate check: for each region and period, `rr20`, `rr25`, `rr30` give the annual rate of events with M >= 2.0, 2.5, 3.0 divided by the 2013-2015 rate. It needs no GR fit, so it still works where a period has too few events for a node Mc. Swarms and aftershock sequences make it noisy.

## Products

| File | Content |
|---|---|
| `mc_grid.csv`, `mc_grid.nc` | all years, N = 250: lon, lat, n, radius_km, mc_maxc, mc_mbs, mc_gft90, mc_gft95, mc_best, sigma_boot, b, b_sigma (+ per-method sigma, n_above); the NetCDF adds `mc_best_qc` |
| `mc_grid_qc.csv` | same, QC events only (4 P + 4 S) |
| `mc_grid_<period>.csv`, `mc_grid_periods.nc` | periods 2010, 2011-2012, 2013-2015, 2016, N = 150 |
| `mc_grid_<year>.csv` | each year 2010-2016, N = 150 |
| `mc_grid_period_range.csv` | per node, worst minus best period Mc_best (nodes with >= 2 periods) |
| `mc_regional.csv` | pooled Mc per region and period, three methods, bootstrap sigma, b, rate ratios; `variant` = all / qc |
| `mc_nodes_by_region.csv` | node Mc_best percentiles per region and period |
| `fig1_mc_three_methods` | Mc by MAXC+0.2, MBS, GFT (best rule), with stations (white inverted triangles = seafloor, dark triangles = land) |
| `fig2_sigma_b` | bootstrap sigma of Mc_best, and b above Mc_best |
| `fig3_mc_periods` | Mc_best per period with the stations open in that period, and worst minus best period |
| `fig4_regional_fmd` | FMD per region with the three Mc and the clusters_zbz Mc |
| `fig5_qc_minus_all` | Mc_best(QC) minus Mc_best(all) |
| `references.bib` | method references, DOIs checked on Crossref / DataCite |

## Main numbers

**Where Mc is lowest and highest.** 4,408 nodes have 250 events within 50 km (median radius 31 km). Node Mc_best, median (10th-90th percentile): Axial 0.8 (0.4-1.2), offshore Washington 1.1 (0.8-1.4), Puget 1.1 (0.9-1.4), onshore and other 1.3 (0.9-1.7), Mendocino 1.5 (1.1-1.8), offshore Oregon 1.5 (1.2-1.9), Blanco 1.6 (1.3-2.2), Endeavour 1.7 (1.4-2.0), Gorda 1.9 (1.7-2.1). The lowest values (-0.2 to 0.3) sit at Mount St. Helens and inside the Axial caldera, both under dense instrumentation. The highest (2.6-2.8) are on the western Blanco transform (43.6-43.9 N, 129.3-129.6 W); the Gorda ridge reaches 2.4 at most. With the stored magnitudes the highest nodes were 2.9-3.2 on the western Blanco and the northern Gorda ridge.

**Which method is most conservative.** At the nodes MAXC+0.2 gives the highest Mc at 83 % of nodes; on average MBS is 0.18 lower and Mc_best 0.14 lower. With 250 events, GFT95 is reached at only 26 % of nodes and GFT90 at 85 %, so Mc_best is mostly GFT90 there; MBS finds a stable b at 73 % of nodes. For the pooled regional FMDs, with thousands of events, MBS is the highest in Axial (1.5), Blanco (1.8), Mendocino (1.6), Washington (1.4) and Puget (1.3), and GFT90 is always the lowest. The clusters_zbz Mc is at or above MAXC+0.2 and Mc_best in every region. It is below the MBS value at Axial (0.85 against 1.5; the pooled Axial FMD has two slopes, see Caveats) and by half a bin at Blanco (1.75 against 1.8); the map methods work on magnitudes rounded to 0.1, so a map Mc of 1.8 keeps M >= 1.75.

**Effect of the moving OBS array.** Where a node can be computed in more than one period, Mc changes little: the median change is -0.1 from 2010 to 2013-2015 and 0.0 from 2013-2015 to 2016, and the worst-minus-best period range has a median of 0.3 (75th percentile 0.4). Those common nodes are mostly onshore, where the land network did not change much. Offshore, the array decides whether Mc can be measured at all. With the OBS out (2010, 2016), Blanco and Endeavour have no node with 150 events within 50 km and Gorda 5 (2010) or none (2016); with the OBS in (2013-2015), they have 295 (Endeavour), 500 (Blanco) and 648 (Gorda) nodes. The rate ratios say the same: offshore (OFF) the rate of M >= 2.5 events is 0.31 of the 2013-2015 rate in 2010, 0.75 in 2011-2012 and 0.28 in 2016; for Blanco alone it is 0.28, 0.79 and 0.29, and at M >= 3.0 it is 0.39 and 0.47 in 2010 and 2016. Without the OBS the offshore catalog is therefore incomplete at M 2.5-3, against Mc 1.6-1.9 with the OBS. Between the two OBS periods offshore Mc hardly changes (pooled Endeavour 1.7 to 1.6, Blanco 1.5 to 1.6, Gorda 1.9 to 1.9; node medians Endeavour 1.9 to 1.8, Blanco 1.6 to 1.5). With the stored magnitudes 2013-2015 was 0.2-0.4 lower than 2011-2012 (pooled Blanco 2.6 to 2.2): that difference came from the stored solution, whose Blanco level differed between deployments.

**2016 onshore drop.** In 2016, the M >= 2.5 rate falls to 0.11 of the 2013-2015 rate in Mendocino and 0.30 in the onshore box, and Mendocino has no node. This matches the NC/BK pick gap south of 42 N, not the OBS. Puget keeps its rate (0.98 at M >= 2.5) and its Mc (node 10-90% 0.7-1.4).

**QC variant.** Keeping only events with 4 P and 4 S picks removes nodes (3,234 instead of 4,408) but hardly moves Mc: the node difference has a median of 0.0 and an interquartile range of -0.1 to +0.1; pooled regional Mc_best rises by 0.2 in Endeavour and 0.1 in Blanco, falls by 0.1 in Oregon and the onshore box, and does not change elsewhere.

## Caveats

- The pooled Axial FMD has two slopes (2014-2015 OOI cabled array and an M 2-3 bulge in 2015), so its pooled GFT value of 0.3 is unstable (bootstrap sigma 0.63) and its MBS value (1.5) sits on the second slope. Use MAXC+0.2 (0.7) or the node values (0.4-1.2) for Axial.
- Pooled Mc for 2010 offshore (OFF, 1.8) and 2016 Blanco (2.1) comes with b near 0.6-0.7, a sign that the sample is incomplete above the estimated Mc. Read the rate ratios for those cases.
- The rate ratio assumes a steady rate of larger events. The Blanco and Gorda rates vary on their own, so a ratio of 0.6-0.8 in 2011-2012 mixes detection and real changes.
- Mc is in Route A M, mostly ML. Mc values below 0 at Mount St. Helens follow the local ML scale.

## Mc per region and period (pooled FMD, all QC, flagged events left out)

Mc_best = GFT95, else GFT90, else MAXC+0.2. sigma = bootstrap sigma of Mc_best. Empty cells: fewer than 100 events, or the method found no Mc. Rate ratio = annual rate of M >= 2.5 events over the 2013-2015 rate.

| Region | period | N | MAXC+0.2 | MBS | GFT90 | GFT95 | best | sigma | b | rate ratio M>=2.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| AX | all | 13913 | 0.7 | 1.5 | 0.3 |  | 0.3 | 0.63 | 0.72 | 0.43 |
| EN | all | 1823 | 2.0 | 1.8 | 1.6 |  | 1.6 | 0.09 | 0.75 | 0.71 |
| BL | all | 31563 | 1.7 | 1.8 | 1.4 | 1.6 | 1.6 | 0.02 | 1.10 | 0.73 |
| GO | all | 6957 | 2.0 | 1.9 | 1.7 | 1.9 | 1.9 | 0.08 | 1.22 | 0.69 |
| MT | all | 28750 | 1.4 | 1.6 | 1.2 | 1.5 | 1.5 | 0.17 | 1.00 | 0.82 |
| WA | all | 2804 | 1.2 | 1.4 | 1.0 | 1.3 | 1.3 | 0.08 | 1.05 | 1.13 |
| OR | all | 1185 | 1.7 | 1.6 | 1.4 | 1.6 | 1.6 | 0.08 | 1.11 | 0.78 |
| PU | all | 12947 | 1.2 | 1.3 | 1.0 | 1.2 | 1.2 | 0.07 | 1.23 | 0.91 |
| ON | all | 27529 | 1.5 | 1.5 | 1.1 | 1.3 | 1.3 | 0.00 | 0.93 | 0.80 |
| OFF | all | 44332 | 1.7 | 1.9 | 1.5 | 1.7 | 1.7 | 0.03 | 1.09 | 0.73 |
| AX | 2010 | 0 |  |  |  |  |  |  |  | 0.00 |
| EN | 2010 | 52 |  |  |  |  |  |  |  | 0.16 |
| BL | 2010 | 78 |  |  |  |  |  |  |  | 0.28 |
| GO | 2010 | 185 | 2.2 | 1.9 | 1.8 |  | 1.8 | 0.14 | 0.79 | 0.38 |
| MT | 2010 | 3852 | 1.8 | 1.6 | 1.3 | 1.6 | 1.6 | 0.13 | 0.99 | 1.18 |
| WA | 2010 | 387 | 1.6 | 1.3 | 1.2 | 1.4 | 1.4 | 0.12 | 1.12 | 1.36 |
| OR | 2010 | 103 | 1.8 | 1.4 |  |  | 1.8 | 0.21 | 0.90 | 0.62 |
| PU | 2010 | 2329 | 1.3 | 1.5 | 1.2 | 1.5 | 1.5 | 0.15 | 1.66 | 0.39 |
| ON | 2010 | 3296 | 1.5 | 1.3 | 1.1 | 1.3 | 1.3 | 0.09 | 0.99 | 0.56 |
| OFF | 2010 | 805 | 1.7 | 1.9 | 1.8 |  | 1.8 | 0.16 | 0.67 | 0.31 |
| AX | 2011-2012 | 0 |  |  |  |  |  |  |  | 0.00 |
| EN | 2011-2012 | 599 | 1.9 | 1.8 | 1.7 |  | 1.7 | 0.14 | 0.86 | 0.76 |
| BL | 2011-2012 | 5051 | 1.7 | 2.3 | 1.5 |  | 1.5 | 0.23 | 0.86 | 0.79 |
| GO | 2011-2012 | 1207 | 2.1 | 2.0 | 1.8 | 1.9 | 1.9 | 0.11 | 1.16 | 0.62 |
| MT | 2011-2012 | 7286 | 1.5 | 1.5 | 1.2 | 1.4 | 1.4 | 0.08 | 0.98 | 0.73 |
| WA | 2011-2012 | 840 | 1.2 | 1.4 | 1.0 | 1.2 | 1.2 | 0.14 | 1.00 | 1.57 |
| OR | 2011-2012 | 335 | 1.7 | 1.5 | 1.4 | 1.5 | 1.5 | 0.12 | 1.14 | 0.67 |
| PU | 2011-2012 | 3357 | 1.1 | 1.3 | 1.0 | 1.3 | 1.3 | 0.21 | 1.27 | 1.01 |
| ON | 2011-2012 | 6504 | 1.5 | 1.5 | 1.2 | 1.4 | 1.4 | 0.08 | 0.91 | 0.85 |
| OFF | 2011-2012 | 8032 | 1.7 | 2.2 | 1.7 |  | 1.7 | 0.13 | 0.94 | 0.75 |
| AX | 2013-2015 | 13640 | 0.7 | 1.5 | 0.3 |  | 0.3 | 0.46 | 0.72 | 1.00 |
| EN | 2013-2015 | 1140 | 2.0 | 1.8 | 1.6 |  | 1.6 | 0.09 | 0.77 | 1.00 |
| BL | 2013-2015 | 26293 | 1.7 | 1.8 | 1.4 | 1.6 | 1.6 | 0.02 | 1.22 | 1.00 |
| GO | 2013-2015 | 5519 | 2.0 | 1.9 | 1.7 | 1.9 | 1.9 | 0.07 | 1.32 | 1.00 |
| MT | 2013-2015 | 17554 | 1.4 | 1.6 | 1.2 | 1.5 | 1.5 | 0.36 | 1.03 | 1.00 |
| WA | 2013-2015 | 1544 | 1.3 | 1.3 | 0.9 | 1.1 | 1.1 | 0.15 | 1.04 | 1.00 |
| OR | 2013-2015 | 619 | 1.9 | 1.6 | 1.4 | 1.6 | 1.6 | 0.07 | 1.15 | 1.00 |
| PU | 2013-2015 | 5903 | 1.2 | 1.2 | 0.9 | 1.1 | 1.1 | 0.25 | 1.20 | 1.00 |
| ON | 2013-2015 | 13799 | 1.5 | 1.4 | 1.1 | 1.3 | 1.3 | 0.00 | 0.93 | 1.00 |
| OFF | 2013-2015 | 35115 | 1.7 | 1.8 | 1.5 | 1.7 | 1.7 | 0.02 | 1.20 | 1.00 |
| AX | 2016 | 273 | 0.4 | 0.4 | 0.4 |  | 0.4 | 0.18 | 1.28 | 0.00 |
| EN | 2016 | 32 |  |  |  |  |  |  |  | 0.32 |
| BL | 2016 | 141 | 2.3 | 2.2 | 2.1 |  | 2.1 | 0.26 | 0.71 | 0.29 |
| GO | 2016 | 46 |  |  |  |  |  |  |  | 0.20 |
| MT | 2016 | 58 |  |  |  |  |  |  |  | 0.11 |
| WA | 2016 | 33 |  |  |  |  |  |  |  | 0.41 |
| OR | 2016 | 128 | 2.3 |  |  |  | 2.3 | 0.28 |  | 0.52 |
| PU | 2016 | 1358 | 1.2 | 1.4 | 1.1 | 1.3 | 1.3 | 0.15 | 1.10 | 0.98 |
| ON | 2016 | 3930 | 0.5 | 1.2 | 0.8 | 1.1 | 1.1 | 0.10 | 0.97 | 0.30 |
| OFF | 2016 | 380 | 2.3 | 2.0 | 1.9 |  | 1.9 | 0.19 | 0.60 | 0.28 |

clusters_zbz Mc (from `../clusters_zbz/completeness_b_df.csv`, selected years): AX 0.85, EN 2.25, BL 1.75, GO 2.25, MT 1.85, WA 1.65, OR 1.85, PU 1.35, ON 1.65.

## References

- Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits. Bull. Earthq. Res. Inst. Univ. Tokyo, 43, 237-239. No DOI.
- Utsu, T. (1965). A method for determining the value of b in a formula log n = a - bM showing the magnitude-frequency relation for earthquakes. Geophys. Bull. Hokkaido Univ., 13, 99-103. No DOI.
- Shi, Y., & Bolt, B. A. (1982). The standard error of the magnitude-frequency b value. BSSA, 72(5), 1677-1687. https://doi.org/10.1785/BSSA0720051677
- Wiemer, S., & Wyss, M. (2000). Minimum magnitude of completeness in earthquake catalogs: Examples from Alaska, the western United States, and Japan. BSSA, 90(4), 859-869. https://doi.org/10.1785/0119990114
- Cao, A., & Gao, S. S. (2002). Temporal variation of seismic b-values beneath northeastern Japan island arc. GRL, 29(9), 1334. https://doi.org/10.1029/2001GL013775
- Woessner, J., & Wiemer, S. (2005). Assessing the quality of earthquake catalogues: Estimating the magnitude of completeness and its uncertainty. BSSA, 95(2), 684-698. https://doi.org/10.1785/0120040007
- Mignan, A., & Woessner, J. (2012). Estimating the magnitude of completeness for earthquake catalogs. CORSSA. https://doi.org/10.5078/corssa-00180805 (DataCite DOI; not in Crossref)
