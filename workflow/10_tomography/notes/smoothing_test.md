# Smoothing vs uneven node spacing: spike test on the Mendocino-Gorda grid

Date: 2026-11-22. Scripts: `b2_smooth_spike_test.py`, `b2_smooth_spike_analyse.py`; params `tomodd.smooth_test`.
Runs: `/wd1/mdenolle_data/tomo_v1/b2_tomodd/mendocino_gorda/smooth_test/` (driver `run_all.sh`).
Results: `data/tomo_v1/b2_smooth_test/{spikes.csv, by_depth.csv, layers.csv, psf.png}`.

## Question

tomoDD-SE builds smoothing rows as `weight * (m_i - m_j)` between neighbouring nodes and never divides by the
spacing (`dx, dy, dz` are computed and unused, tomoDD-SE.f ~l.1726). Our depth nodes are 2 km apart near the
surface and 3-8 km apart in the resolved crust, laterally ~10.4 x 11.1 km. Per km of gradient, the vertical
penalty relative to the lateral one, (w3 dz)/(w1 dx), goes from 0.10 (dz = 2 km) to 0.38 (dz = 8 km at 47 km).
Thin layers also collect less sensitivity per node (DWS roughly proportional to layer thickness), so for the
same weight they lean more on their neighbours. Does this distort the recovered model at our settings?

Other facts from the code that bear on it:
* G is column-scaled to unit norm before LSQR. That is a change of variables: it does not change the
  smoothing/data balance, but it makes the damping node-dependent (stronger where data are dense).
* Smoothing and damping act on each iteration's update (rows with d = 0), not on the total model, so after
  12 iterations the final model is less smooth than one step with the same weights.
* Nodes with DWS below THRES (S: THRES/3) are fixed at the starting value; below ~50 km almost all are
  (8% of nodes active at 55 km, 2% at 65 km, none below), so the coarse deep spacing does not matter.

## Test

One run per depth (-1, 3, 7, 12, 21, 34, 55 km): +5% Vp at single nodes (Vp/Vs kept, so Vs also +5%) on a
5 x 5 lateral lattice 8 nodes (~85-89 km) apart inside the event box. Noise-free synthetics from
tomoDD-SE_syn on the real geometry (9,334 events, 153 stations), the real absolute pick list (228,454) and
catalogue pairs (599,440) with their real weights; inversion with the production settings (4 sets x 3
iterations, smooth 20/20/10, damp 100, thres 0.2) from the background model and true locations. A null run
(synthetics from the background model) is subtracted; it returned the starting model exactly. 26-27 min and
2.2 GB per run, 8 runs in parallel.

A single-node spike is a trilinear pyramid, so its true half-width at half maximum is half the node spacing
in each direction. Broadening factor = recovered / true half-width.

## Results

Recovery is all or nothing: where rays reach a node the spike comes back at 60-100%, elsewhere at ~0 (node
fixed below the DWS threshold, or no rays). Medians over all 25 spikes are therefore meaningless; the
summary is over resolved spikes (peak >= 50%).

| depth (km) | Vp resolved / 25 | Vp median peak | lateral broadening | vertical broadening (up / down) | Vs resolved / 25 |
|---|---|---|---|---|---|
| -1 | 0 | - | - | - | 0 |
| 3 | 1 | 69% | 1.97 / 1.00 (x / y) | 1.00 / 2.54 | 1 |
| 7 | 1 | 96% | 1.04 / 1.03 | 1.04 / 1.02 | 1 |
| 12 | 5 | 56% | 1.24 / 1.01 | 1.33 / 1.03 | 1 |
| 21 | 10 | 90% | 1.06 / 1.04 | 1.02 / 1.02 | 10 |
| 34 | 10 | 84% | 1.09 / 1.06 | 1.01 / 1.03 | 6 |
| 55 | 0 | - | - | - | 0 |

Vs is similar where resolved (broadening 1.04-1.12 at 12-34 km).

* **At 12-34 km, smoothing at these weights does not broaden a spike measurably** (factors 1.0-1.1 laterally
  and vertically, also on 3-4 km vertical spacing next to 5-6 km). The uneven spacing is not a practical
  problem there.
* **Above 12 km almost nothing is resolved at node scale** (1 of 25 at 3 and 7 km, none at -1 km). The
  binding limit in the upper crust is coverage, not smoothing. The two resolved shallow spikes are too few
  to measure broadening; the one at 3 km is twice as wide in x and 2.5 times below, consistent with the
  thin-layer argument, but it is one spike.
* **The southern (39.7 N) and northern (42.9 N) rows are mostly unresolved at all depths**; resolution is
  concentrated in 40.5-42.1 N, -127.2 to -124.2 E.

## Caveats

* Noise-free data and a node-scale target. With real noise the L-curve will likely pick stronger smoothing,
  and then the spacing effect grows in proportion. Re-run this test at the L-curve weights.
* One region, one lattice (25 positions per depth). Resolved counts per depth are small (1-10).
* Same geometry and starting model for synthetics and inversion: no ray-tracer or starting-model error.
  The synthetic calibration (STATUS step 5) covers those.

## Proposed decision (D-B028)

Keep the current node_z for the Mendocino-Gorda grid: at smooth 20/20/10 the spike broadening in the
resolved crust (12-34 km) is <= 1.1, so uneven spacing does not need a fix now. Repeat this spike test
(same scripts) at the smoothing and damping the L-curve selects and on every new region grid; if broadening
in the resolved crust then exceeds ~1.5, move to more even spacing in 0-40 km (licence-compatible; a
spacing-scaled smoothing row would be a source change). Report upper-crust (< 12 km) structure only where
the spike/checkerboard tests show recovery; node-scale features there are mostly unresolved with this
event set.
