# Plate boundaries v2 (D-B116): pre-registered rules

Written Fri Nov 27 14:01:52 PST 2026, before any seismicity axis is fitted or any bias test is scored. Includes the PI
geometry constraint relayed by the coordinator at about 13:55 PST (few straight segments; Blanco 3 segments with two
step-overs; JdF one straight line from Axial to Endeavour; keep the MFZ and GR offsets; triple junction where the
straight traces meet). The rules below are not changed after the results are seen; any other choice is reported as a
variant only. Code: workflow/11_paper_plan/pbv2_*.py; outputs /wd1/mdenolle_data/paper_plan/plate_boundaries_v2/.

## 1. Events

* Catalog v6.1 (/wd1/mdenolle_data/catalog_v6/v6.1/cascadia_catalog_v6.1.csv), loc_source nll3d, earthquakes only
  (event_flag empty), tier A or B, well-located epicentre: h_max <= 3.0 km and ell_len3 <= 10 km (selection S0).
* Variants (reported, not drawn): S1 strict (h_max <= 1.5 km, ell_len3 <= 6 km); S2 tier A-C with the S0 location
  limits (used only to state whether a sparse section could be resolved).
* Events are never moved or clamped.

## 2. Sections and frames

Each section has a straight reference frame (s along strike from its first end, km; u across strike, km, positive to
the north for transforms and to the east for ridges), an along-strike range and an across-strike window:
MFZ (Mendocino transform, Gorda Ridge corner to 125.0 W, u in [-20, +25]); GR (Gorda Ridge, MFZ corner to Blanco
corner, u in [-20, +30]); BL (Blanco transform, Gorda Ridge corner to Cleft, u in [-30, +20]); JRS (JdF, Cleft to
Axial), JRN (JdF, Axial to Endeavour), EN (Endeavour and West Valley): u in [-25, +25]. Explorer, Sovanco and Nootka:
no line is fitted (Section 6).

## 3. Seismicity axis picks

* Bins of 5 km along strike (non-overlapping). In each bin with >= 10 events: the peak of a Gaussian kernel density
  of u (bandwidth 1.5 km, grid 0.1 km) = the pick.
* Bootstrap: 300 resamples of whole UTC days (day-block bootstrap, keeps swarms together); per-bin pick uncertainty
  sigma_boot = half the 16-84 % range of the bootstrap picks; band = 16-84 % of the bootstrap fitted lines.

## 4. Straight-segment fit (parsimony, BIC)

* Data: picks (s_i, u_i) with sigma_i^2 = sigma_boot,i^2 + sigma_loc^2; sigma_loc = median h_max of the section's
  S0 events (wiggles below the location uncertainty are not resolved).
* Model with K breaks: K+1 straight segments u = a_k + b_k s; each segment may be offset from its neighbour (step-over);
  each segment spans >= L_min = 30 km (pick span + one bin) and has >= 4 picks.
* BIC_K = chi2_K + p_K ln n + lambda K, with p_K = 3K + 2 and an extra per-break penalty lambda = ln n (n = number of
  picks). K = argmin BIC_K over K = 0 .. min(5, floor(L / L_min) - 1). Exact minimum by dynamic programming.
* Drawing: at a break the step-over offset is Delta = u_{k+1}(s_b) - u_k(s_b). |Delta| < 2 sigma_loc: a bend (the two
  segments meet at the break); otherwise a step-over (both segments end at s_b, joined by a connector marked as a
  step-over).
* PI constraints (drawn model): Blanco K = 2 (three segments, two step-overs); the BIC choice is reported as a check.
  JdF Axial-Endeavour: K = 0 (one straight line). Break positions, step-over offsets and their 68 / 95 % intervals
  come from refitting every bootstrap sample with the same K.
* Bathymetric axis lines: within each segment's s-range a straight line is fitted to the bathymetric picks
  (D-B113 picks of pb_axes.py decimated to one per 10 km, the D-B113 median-filter length; sigma = their RMS
  residual), or the MAPRIDGES v2602 traces (Blanco: no D-B113 pick). Bathymetric references: GR axial valley
  (pb_axes 'valley'), MFZ base of the north-facing slope (pb_axes 'base'), JdF axial high (pb_axes 'crest'), Blanco
  MAPRIDGES TF traces.
* Sections without enough picks (< 8 picks or < 50 % of the section length covered): the segments come from the BIC
  fit of the bathymetric picks (same rule, sigma = 1 km), source "bathymetry".

## 5. Combination (step 3)

* Per segment: the seismicity line and the bathymetric line are compared at the segment's two ends and middle. They
  agree if max |u_seis - u_bathy| <= 2 sqrt(sigma_seis^2 + sigma_bathy^2) (sigma_seis: bootstrap 68 % half-width of
  the seismicity line; sigma_bathy: standard error of the bathymetric line fit). Agree: the proposed trace is the
  bathymetric line, source "both". Disagree: both are kept, the section is flagged; the proposed trace is the
  seismicity line (source "seismicity") with the bathymetric line as an alternative feature.
* Corners: ridge-transform corners = intersection of the end segments of adjacent sections (extended);
  triple junction = intersection of the easternmost MFZ segment with the PB2002 trench trace. Uncertainty: the
  bootstrap scatter of the intersection (68 % semi-axes, km).

## 6. Deformation zones

* Gorda plate interior and Explorer plate edges (Nootka, Sovanco): polygon = the 80 % highest-density region of the
  epicentre density (Gaussian kernel 5 km, grid 1 km) of the events between the boundaries, events within 10 km of a
  proposed line left out; simplified at 2 km, pieces < 100 km^2 dropped. Gorda: S0. Explorer: S0 has too few events, so
  S2 (tier A-C) and the zone is flagged low confidence.

## 7. Location bias tests (step 1)

Offsets: per 5 km bin, pick minus PB2002 trace (and minus the bathymetric axis), in u; section offset O = median over
bins, 95 % interval from the day-block bootstrap.
* B_pair: across-strike shift (ours minus theirs) of common events (|dt| <= 3 s, <= 50 km) against the OBS-based
  published catalogs (Blanco: Lange et al. 2026 absolute and relative, Zhou et al. 2024; all sections: Niksejel &
  Zhang 2026); ComCat/ANSS shifts are reported but not used in B (land-network locations offshore carry their own
  bias, shown by their horizontalError).
* B_ver: max - min of O over catalogs v4.1, v5.1, v6.0, v6.1 (same events, the v6.1 S0 selection).
* B_geo: max of |O(gap <= 180 deg and dist_min <= 15 km) - O(all)| and |O(Cascadia Initiative year) - O(all)| over
  the years with >= 5 bins.
* B = max(B_pair, B_ver, B_geo). Verdict: "offset is not location bias" if |O| >= 2 B and an independent catalog's
  own offset from PB2002 has the same sign; "partly" if |O| >= B; else "bias cannot be excluded".
