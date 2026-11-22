# Magnitude of completeness maps, catalog v4

Built by `workflow/06_analysis/v4_mc_maps.py` (branch `cluster/zbz`) on 2026-11-22.
Inputs: `Cascadia_relocated_catalog_v4.csv` (172,340 events) and the Route A magnitudes
`cascadia_catalog_M_routeA.csv` (130,767 events with M; 99.9% ML). Rerun with
`python v4_mc_maps.py --yearly` (about 15 s of compute on 32 cores, plus 1 min of plotting).

## Method

Magnitudes are rounded to 0.1. Three estimators run on each sample:

- **MAXC + 0.2**: centre of the most populated bin of the non-cumulative FMD (Wiemer & Wyss 2000), plus the +0.2 correction of Woessner & Wiemer (2005).
- **MBS**, b-value stability (Cao & Gao 2002) in the form of Woessner & Wiemer (2005): the lowest cutoff where the mean of b over the next five bins (a 0.5 window) differs from b(Mc) by less than the Shi & Bolt (1982) error of b(Mc). b is the Aki (1965) / Utsu (1965) estimate with the bin-edge correction.
- **GFT**, goodness of fit (Wiemer & Wyss 2000): the lowest cutoff where a GR law fitted above it reproduces the observed cumulative counts with R >= 90 % (GFT90) or R >= 95 % (GFT95).
- **best**: the rule of Woessner & Wiemer (2005): GFT95 if reached, else GFT90, else MAXC+0.2. Called "GFT (best)" in the figures.

Every b fit needs 30 events above the cutoff. Uncertainty comes from 200 bootstrap resamples of each sample; `sigma_boot` is the standard deviation of the bootstrap Mc_best (columns `sigma_maxc`, `sigma_mbs`, `sigma_gft` give the same for each method). `b` and `b_sigma` are fitted above Mc_best. The MAXC, Aki-Utsu, Shi-Bolt and b-stability code is adapted from AgentSeismo (Z. Zhang), `studies/gibraltar_2010_2011/02_assessment/mc_extended.py`; the MBS test uses the Shi-Bolt error instead of that script's fixed 0.03 tolerance, and GFT was added.

Choices:

- Grid every 0.1 deg over lon -131 to -119.5, lat 39 to 51. A node takes its nearest 250 events with M if all of them lie within 50 km (epicentral distance); otherwise it stays empty. For the period maps the sample is 150 events (same 50 km cap), because a single year has too few events for 250 almost everywhere offshore. The cost shows in `sigma_boot`.
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

**Where Mc is lowest and highest.** 4,578 nodes have 250 events within 50 km (median radius 31 km). Node Mc_best, median (10th-90th percentile): Axial 0.7 (0.4-1.0), offshore Washington 1.0 (0.8-1.4), Puget 1.1 (0.8-1.5), onshore and other 1.4 (0.9-1.9), Mendocino 1.6 (1.3-2.0), offshore Oregon 1.7 (1.4-1.9), Endeavour 1.8 (1.6-2.2), Gorda 2.1 (1.8-2.3), Blanco 2.1 (1.9-2.5). The lowest values (0.0-0.3) sit at Mount St. Helens and inside the Axial caldera, both under dense instrumentation. The highest (2.9-3.2) are on the western Blanco transform and the northern Gorda ridge (43-44 N, 128-130.4 W), and at the southern edge of the grid near 39.3 N, 123.2 W.

**Which method is most conservative.** At the nodes MAXC+0.2 gives the highest Mc at 84 % of nodes; on average MBS is 0.19 lower and Mc_best 0.14 lower. With 250 events, GFT95 is reached at only 25 % of nodes and GFT90 at 84 %, so Mc_best is mostly GFT90 there; MBS finds a stable b at 68 % of nodes. For the pooled regional FMDs, with thousands of events, MBS is the highest in Blanco (2.5), Mendocino (1.8) and Puget (1.4), and GFT90 is always the lowest. The clusters_zbz Mc is at or above all three methods in every region (by 0.05 to 0.4), except the Blanco MBS (2.5 against 2.45), so the cluster analysis sits on the safe side.

**Effect of the moving OBS array.** Where a node can be computed in more than one period, Mc changes little: the median change is -0.1 from 2010 to 2013-2015 and 0.0 from 2013-2015 to 2016, and the worst-minus-best period range has a median of 0.3 (75th percentile 0.5). Those common nodes are mostly onshore, where the land network did not change much. Offshore, the array decides whether Mc can be measured at all. With the OBS out (2010, 2016), Blanco, Gorda and Endeavour have no node with 150 events within 50 km; with the OBS in, they have 400-650 nodes each. The rate ratios say the same: offshore (OFF) the rate of M >= 2.5 events is 0.11 of the 2013-2015 rate in 2010, 0.64 in 2011-2012 and 0.10 in 2016; for Blanco alone it is 0.05, 0.62 and 0.08, and even at M >= 3.0 it is 0.20 and 0.19 in 2010 and 2016. Without the OBS the offshore catalog is therefore incomplete above M 3, against Mc 2.1-2.2 with the OBS. Between the two OBS periods, 2013-2015 is about 0.2-0.4 lower than 2011-2012 offshore (pooled Blanco 2.6 to 2.2; node medians Endeavour 2.2 to 1.8, Blanco 2.3 to 2.1).

**2016 onshore drop.** In 2016, the M >= 2.5 rate falls to 0.10 of the 2013-2015 rate in Mendocino and 0.21 in the onshore box, and Mendocino has no node. This matches the NC/BK pick gap south of 42 N, not the OBS. Puget keeps its rate (0.89 at M >= 2.5) and its Mc (1.0-1.5).

**QC variant.** Keeping only events with 4 P and 4 S picks removes nodes (3,395 instead of 4,578) but hardly moves Mc: the node difference has a median of 0.0 and an interquartile range of -0.1 to +0.1; pooled regional Mc_best rises by 0.2 in Endeavour, 0.1 in Blanco and Washington, and does not change elsewhere.

## Caveats

- The pooled Axial FMD has two slopes (2014-2015 OOI cabled array and an M 2-3 bulge in 2015), so its pooled GFT value of 0.2 is unstable (bootstrap sigma 0.94). Use MAXC+0.2 (0.7) or the node values (0.4-1.0) for Axial.
- Pooled Mc for 2010 offshore (OFF, 2.0) and 2016 Blanco (2.4) comes with b near 0.6, a sign that the sample is incomplete above the estimated Mc. Read the rate ratios for those cases.
- The rate ratio assumes a steady rate of larger events. The Blanco and Gorda rates vary on their own, so a ratio of 0.6 in 2011-2012 mixes detection and real changes.
- Mc is in Route A M, mostly ML. Mc values below 0 at Mount St. Helens follow the local ML scale.

## Mc per region and period (pooled FMD, all QC)

Mc_best = GFT95, else GFT90, else MAXC+0.2. sigma = bootstrap sigma of Mc_best. Empty cells: fewer than 100 events, or the method found no Mc. Rate ratio = annual rate of M >= 2.5 events over the 2013-2015 rate.

| Region | period | N | MAXC+0.2 | MBS | GFT90 | GFT95 | best | sigma | b | rate ratio M>=2.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| AX | all | 13913 | 0.7 | 0.4 | 0.2 |  | 0.2 | 0.94 | 0.70 | 0.43 |
| EN | all | 1823 | 2.2 | 1.9 | 1.8 |  | 1.8 | 0.05 | 0.81 | 0.75 |
| BL | all | 31563 | 2.3 | 2.5 | 2.0 | 2.2 | 2.2 | 0.03 | 1.17 | 0.62 |
| GO | all | 6958 | 2.3 | 2.2 | 1.9 | 2.1 | 2.1 | 0.02 | 1.11 | 0.69 |
| MT | all | 28750 | 1.6 | 1.8 | 1.4 | 1.7 | 1.7 | 0.17 | 1.01 | 0.81 |
| WA | all | 2828 | 1.3 | 1.4 | 1.0 | 1.2 | 1.2 | 0.09 | 0.99 | 1.14 |
| OR | all | 1393 | 1.7 | 1.6 | 1.4 | 1.6 | 1.6 | 0.08 | 1.03 | 0.85 |
| PU | all | 13348 | 1.2 | 1.4 | 1.0 | 1.2 | 1.2 | 0.16 | 1.22 | 0.91 |
| ON | all | 30191 | 1.5 | 1.4 | 1.1 | 1.3 | 1.3 | 0.00 | 0.92 | 0.73 |
| OFF | all | 44565 | 2.3 | 2.4 | 2.0 | 2.2 | 2.2 | 0.01 | 1.15 | 0.64 |
| AX | 2010 | 0 |  |  |  |  |  |  |  | 0.00 |
| EN | 2010 | 52 |  |  |  |  |  |  |  | 0.25 |
| BL | 2010 | 78 |  |  |  |  |  |  |  | 0.05 |
| GO | 2010 | 185 | 2.5 | 2.1 | 2.0 | 2.2 | 2.2 | 0.14 | 0.84 | 0.33 |
| MT | 2010 | 3852 | 2.0 | 1.8 | 1.5 | 1.8 | 1.8 | 0.24 | 1.00 | 1.14 |
| WA | 2010 | 388 | 1.6 | 1.3 | 1.2 | 1.4 | 1.4 | 0.12 | 1.11 | 1.43 |
| OR | 2010 | 123 | 1.9 | 1.4 |  |  | 1.9 | 0.20 | 0.73 | 0.87 |
| PU | 2010 | 2392 | 1.3 | 1.6 | 1.2 | 1.5 | 1.5 | 0.15 | 1.69 | 0.39 |
| ON | 2010 | 3583 | 1.5 | 1.4 | 1.2 | 1.3 | 1.3 | 0.08 | 0.97 | 0.53 |
| OFF | 2010 | 826 | 1.6 | 2.1 | 2.0 |  | 2.0 | 0.25 | 0.59 | 0.11 |
| AX | 2011-2012 | 0 |  |  |  |  |  |  |  | 0.00 |
| EN | 2011-2012 | 599 | 2.2 | 1.9 | 1.9 |  | 1.9 | 0.16 | 0.97 | 0.76 |
| BL | 2011-2012 | 5051 | 2.4 | 2.6 | 2.3 | 2.6 | 2.6 | 0.13 | 1.26 | 0.62 |
| GO | 2011-2012 | 1210 | 2.4 | 2.8 | 2.1 |  | 2.1 | 0.16 | 1.02 | 0.67 |
| MT | 2011-2012 | 7286 | 1.6 | 1.7 | 1.4 | 1.6 | 1.6 | 0.12 | 0.99 | 0.71 |
| WA | 2011-2012 | 850 | 1.1 | 1.3 | 1.0 | 1.2 | 1.2 | 0.17 | 0.98 | 1.50 |
| OR | 2011-2012 | 382 | 1.7 | 1.6 | 1.4 | 1.6 | 1.6 | 0.11 | 1.08 | 0.80 |
| PU | 2011-2012 | 3453 | 1.1 | 1.3 | 1.0 |  | 1.0 | 0.18 | 1.04 | 1.03 |
| ON | 2011-2012 | 7172 | 1.6 | 1.5 | 1.2 | 1.4 | 1.4 | 0.04 | 0.96 | 0.67 |
| OFF | 2011-2012 | 8092 | 2.4 | 2.6 | 2.2 | 2.5 | 2.5 | 0.17 | 1.19 | 0.64 |
| AX | 2013-2015 | 13640 | 0.7 | 0.4 | 0.2 |  | 0.2 | 0.94 | 0.70 | 1.00 |
| EN | 2013-2015 | 1140 | 2.2 | 1.9 | 1.8 |  | 1.8 | 0.08 | 0.85 | 1.00 |
| BL | 2013-2015 | 26293 | 2.3 | 2.4 | 2.0 | 2.2 | 2.2 | 0.01 | 1.28 | 1.00 |
| GO | 2013-2015 | 5520 | 2.2 | 2.2 | 1.9 | 2.1 | 2.1 | 0.05 | 1.21 | 1.00 |
| MT | 2013-2015 | 17554 | 1.5 | 1.8 | 1.4 | 1.7 | 1.7 | 0.33 | 1.03 | 1.00 |
| WA | 2013-2015 | 1556 | 1.3 | 1.3 | 0.9 | 1.1 | 1.1 | 0.15 | 1.04 | 1.00 |
| OR | 2013-2015 | 737 | 1.9 | 1.5 | 1.4 | 1.6 | 1.6 | 0.09 | 1.08 | 1.00 |
| PU | 2013-2015 | 6108 | 1.1 | 1.2 | 0.9 | 1.2 | 1.2 | 0.30 | 1.24 | 1.00 |
| ON | 2013-2015 | 15350 | 1.5 | 1.3 | 1.1 | 1.2 | 1.2 | 0.02 | 0.84 | 1.00 |
| OFF | 2013-2015 | 35246 | 2.2 | 2.3 | 2.0 | 2.2 | 2.2 | 0.03 | 1.25 | 1.00 |
| AX | 2016 | 273 | 0.7 | 0.3 | 0.2 |  | 0.2 | 0.20 | 1.11 | 0.00 |
| EN | 2016 | 32 |  |  |  |  |  |  |  | 0.45 |
| BL | 2016 | 141 | 2.9 | 2.6 | 2.4 |  | 2.4 | 0.26 | 0.65 | 0.08 |
| GO | 2016 | 43 |  |  |  |  |  |  |  | 0.13 |
| MT | 2016 | 58 |  |  |  |  |  |  |  | 0.10 |
| WA | 2016 | 34 |  |  |  |  |  |  |  | 0.57 |
| OR | 2016 | 151 | 1.7 |  |  |  | 1.7 | 0.21 | 0.94 | 0.47 |
| PU | 2016 | 1395 | 1.2 | 1.5 | 1.2 | 1.5 | 1.5 | 0.20 | 1.21 | 0.89 |
| ON | 2016 | 4086 | 0.5 | 1.2 | 1.0 | 1.1 | 1.1 | 0.08 | 0.96 | 0.21 |
| OFF | 2016 | 401 | 2.7 | 2.5 | 2.3 | 2.5 | 2.5 | 0.18 | 0.66 | 0.10 |

clusters_zbz Mc (from `../clusters_zbz/completeness_b_df.csv`, selected years): AX 0.75, EN 2.25, BL 2.45, GO 2.45, MT 1.85, WA 1.65, OR 1.85, PU 1.35, ON 1.75.

## References

- Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits. Bull. Earthq. Res. Inst. Univ. Tokyo, 43, 237-239. No DOI.
- Utsu, T. (1965). A method for determining the value of b in a formula log n = a - bM showing the magnitude-frequency relation for earthquakes. Geophys. Bull. Hokkaido Univ., 13, 99-103. No DOI.
- Shi, Y., & Bolt, B. A. (1982). The standard error of the magnitude-frequency b value. BSSA, 72(5), 1677-1687. https://doi.org/10.1785/BSSA0720051677
- Wiemer, S., & Wyss, M. (2000). Minimum magnitude of completeness in earthquake catalogs: Examples from Alaska, the western United States, and Japan. BSSA, 90(4), 859-869. https://doi.org/10.1785/0119990114
- Cao, A., & Gao, S. S. (2002). Temporal variation of seismic b-values beneath northeastern Japan island arc. GRL, 29(9), 1334. https://doi.org/10.1029/2001GL013775
- Woessner, J., & Wiemer, S. (2005). Assessing the quality of earthquake catalogues: Estimating the magnitude of completeness and its uncertainty. BSSA, 95(2), 684-698. https://doi.org/10.1785/0120040007
- Mignan, A., & Woessner, J. (2012). Estimating the magnitude of completeness for earthquake catalogs. CORSSA. https://doi.org/10.5078/corssa-00180805 (DataCite DOI; not in Crossref)
