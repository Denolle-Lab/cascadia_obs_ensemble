# Local magnitude estimation from ocean-bottom and land amplitudes (Method B)

*Methods section for the technical report accompanying the Cascadia OBS ensemble catalog.*

## Overview

We estimate a local magnitude, $M_\mathrm{L}$, for every event in the relocated
catalog from the peak ground-motion amplitudes measured at each phase pick. Because
the amplitudes are recorded in raw digital counts (the instrument response is not
removed; Section 1), we do not attempt to convert individual measurements to physical
ground motion. Instead we pose a single joint linear inverse problem that solves
simultaneously for a *relative* per-event magnitude, a per-station calibration term
that absorbs the (in-band) instrument response and site amplification, and a
distance-dependent attenuation function. The relative magnitudes are placed on the
absolute $M_\mathrm{L}$ scale by regression against independent, catalog-reported
local magnitudes from the U.S. Geological Survey Comprehensive Catalog (ComCat). This
"station-term" or relative-magnitude approach follows the logic of network
$M_\mathrm{L}$ calibration [Richter, 1935; Hutton and Boore, 1987], generalized to a
joint inversion over the entire event–station data set.

## 1. Amplitude measurements

For each P and S arrival in the association/relocation output, a waveform window of
30 s before to 150 s after the pick was retrieved for the corresponding station.
Records were resampled to a common 100 Hz, the mean was removed, a 5% cosine taper
was applied, and a 2 Hz high-pass filter was applied to suppress oceanographic and
long-period noise (particularly important for ocean-bottom seismometers, OBS). The
amplitude $A$ assigned to a pick is the maximum absolute sample value, in counts,
within a short window bracketing the arrival,

$$A = \max_{t\in[t_p-0.5\,\mathrm{s},\; t_p+2\,\mathrm{s}]}\; |u(t)|,$$

taken over the available components at that station (vertical for stations returning
only a vertical channel; vertical and horizontals otherwise). Amplitudes are therefore
**phase-specific** (a station contributes independent P and S measurements) and are
expressed in instrument counts. This yields $\sim 1.0\times10^{6}$ amplitude
measurements across 14 networks and 441 stations; measurements failing the window or
data checks (0.05%) are recorded as missing.

## 2. Amplitude–distance data set and quality control

Each measurement was joined to its relocated hypocenter (latitude, longitude, depth)
and station location. The epicentral distance was computed by the haversine formula
and combined with the event depth and station elevation to give the hypocentral
distance,

$$r_{ij} = \sqrt{\Delta_{ij}^2 + \left(h_i + e_j\right)^2},$$

where $\Delta_{ij}$ is the epicentral distance, $h_i$ the event depth, and $e_j$ the
station elevation (both in km, positive down for depth and up for elevation). We
formed the logarithmic amplitude $y_{ijp}=\log_{10}A_{ijp}$ for event $i$, station
$j$, and phase $p\in\{P,S\}$.

Before inversion we removed non-positive amplitudes, required each station to retain
at least 8 measurements and each event at least 3 measurements (iterated to a fixed
point so that neither threshold is violated after the other is applied), and, after a
first inversion, rejected observations with residuals exceeding four scaled
median-absolute-deviations (MAD) and re-inverted (a single robust pass). The final
data set comprises $\sim 9.9\times10^{5}$ observations, 63{,}798 events, and 430
stations, with hypocentral distances of 0.8–990 km (median 82 km).

## 3. Joint inversion for relative magnitude, station terms, and attenuation

We model the log-amplitude of each observation as the sum of a source, path, and
receiver contribution,

$$\log_{10} A_{ijp} \;=\; M_i \;+\; C_{j,p} \;-\; D_p(r_{ij}) \;+\; \varepsilon_{ijp},$$

where $M_i$ is the (relative) magnitude of event $i$; $C_{j,p}$ is a per-station,
per-phase term; $D_p(r)$ is the phase-dependent attenuation; and
$\varepsilon_{ijp}$ is a residual. The station term $C_{j,p}$ absorbs the
approximately time-invariant, in-band instrument response gain (the counts-to-ground-
motion factor, treated as a scalar over the fixed 2 Hz–Nyquist passband) together
with local site amplification. It is this term that makes measurements from
instruments of different gain (e.g. high-gain borehole seismometers versus OBS)
mutually comparable; consequently the raw counts require no response deconvolution,
provided the response is stable in time and an external absolute calibration is
applied (Section 5). Terms $C_{j,p}$ are created only for station–phase pairs that are
actually observed.

The attenuation is parameterized as a geometric-spreading term plus a linear
(anelastic) term,

$$D_p(r) \;=\; n_p\,\log_{10}\!\left(\frac{r}{r_\mathrm{ref}}\right) \;+\; k_p\,r,
\qquad r_\mathrm{ref}=100\ \mathrm{km},$$

following the functional form used for regional $M_\mathrm{L}$ distance corrections
[Hutton and Boore, 1987]. The problem is linear in the unknowns
$\{M_i,\,C_{j,p},\,n_p,\,k_p\}$ and was assembled as a large sparse system and solved
by damped least squares (LSQR; [Paige and Saunders, 1982]) with light Tikhonov
damping ($10^{-3}$).

**Gauge.** The magnitudes and station terms are non-identifiable up to an additive
constant within each phase: adding a constant to all $M_i$ and subtracting it from all
$C_{j,p}$ leaves the residuals unchanged. We remove this null space with the zero-mean
gauge $\sum_j C_{j,p}=0$ for each phase $p$, imposed as heavily weighted constraint
equations. The magnitudes $M_i$ are thereby determined on an internally consistent
*relative* scale whose absolute level is fixed subsequently by calibration.

## 4. Physical attenuation constraint

With $n_p$ free, the inversion returns a steep geometric term ($n_p\!\approx\!1.5$)
and a **negative** anelastic coefficient $k_p$. A negative $k$ is unphysical (it
implies amplitude growth with distance). It arises from two effects: (i) over the
sampled distance range $\log_{10} r$ and $r$ are strongly collinear, so a steep
geometric term can be traded against a negative linear term with little change in fit;
and (ii) the magnitude–distance selection of the catalog (small events are recorded
only at short distances, larger events reach greater distances) flattens the apparent
amplitude decay at large distance, which the free linear term absorbs as negative $k$.

To obtain a physically meaningful, low-attenuation model we fix the geometric-spreading
exponent to the body-wave value $n_p = 1$ and solve for $k_p$ (achieved by moving the
known geometric term to the data vector). Because the pinned geometric term is
shallower than the free solution, the remaining decay is carried by the anelastic
term, which is then positive: $k_P = 3.5\times10^{-4}$ and
$k_S = 1.1\times10^{-3}\ \mathrm{km^{-1}}$, with $k_S>k_P$ as expected for the more
strongly attenuated S wave. Interpreting $k$ through
$k = \pi f\,/\,(Q\,v\,\ln 10)$ with a representative dominant frequency $f\approx4$ Hz
and crustal velocities $v_P=6.0$, $v_S=3.5\ \mathrm{km\,s^{-1}}$ gives apparent quality
factors $Q_P\approx2.6\times10^{3}$ and $Q_S\approx1.5\times10^{3}$
($Q_P/Q_S\approx1.8$). These values scale linearly with the assumed frequency and are
best regarded as apparent, weakly resolved upper bounds, because most of the distance
decay is carried by the fixed geometric term. Adopting the constrained model changes
individual magnitudes negligibly while improving agreement with independent catalog
magnitudes (Section 5).

## 5. Absolute calibration to local magnitude

The relative magnitudes were tied to the absolute $M_\mathrm{L}$ scale using events
that appear in ComCat with a reported local magnitude. We queried the USGS FDSN event
service over the catalog's spatial bounding box and time span (2010–2015), retaining
only origins carrying an ML-type magnitude (`ML`/`MLv`; duration and other magnitude
types were excluded). Each ComCat–ML event was matched to the nearest-in-time
relocated event within $|\Delta t|\le 15$ s and epicentral distance $\le 50$ km,
yielding 2{,}383 calibration anchors spanning $M_\mathrm{L}\,1.5$–$4.3$.

We fit a linear calibration

$$M_\mathrm{L} = a\,M_i^{\mathrm{rel}} + b$$

by the robust Theil–Sen estimator [Theil, 1950; Sen, 1968], which is insensitive to
the outliers expected from occasional mis-associations and from ComCat magnitude
rounding. The calibration is $a=1.03$ (95% CI $1.01$–$1.05$), $b=-1.07$, with a linear
correlation of $0.91$ and a residual scatter of $0.20$ magnitude units (MAD). The
slope near unity indicates that the relative scale already tracks $M_\mathrm{L}$
closely; the calibration is applied to every event to yield the final $M_\mathrm{L}$.

## 6. Uncertainty quantification

For each event we report a magnitude uncertainty combining the within-event scatter of
single-station magnitudes and the calibration residual,

$$\sigma_{M_i} = \sqrt{\left(a\,\sigma_i^{\mathrm{rel}}\right)^2 + \sigma_\mathrm{cal}^2},
\qquad \sigma_i^{\mathrm{rel}} = \frac{\mathrm{std}_j\!\left(m_{ij}\right)}{\sqrt{N_i}},$$

where $m_{ij}=\log_{10}A_{ij}+D_p(r_{ij})-C_{j,p}$ is the single-station magnitude,
$N_i$ the number of observations for the event, and
$\sigma_\mathrm{cal}=0.20$ the calibration MAD. The median within-event
single-station-magnitude scatter is 0.30 magnitude units, comparable to well-calibrated
regional $M_\mathrm{L}$ networks.

## 7. Catalog quality assessment

The final catalog contains $M_\mathrm{L}$ for 63{,}798 of 63{,}885 events (99.9%; the
remainder are recorded on too few stations to constrain a magnitude), with
$M_\mathrm{L}$ from $-1.5$ to $4.6$ and a median of $1.36$. Post-fit residuals are flat
with hypocentral distance, confirming that the attenuation term removes the distance
dependence. The frequency–magnitude distribution was characterized by the magnitude of
completeness $M_c$, estimated by the maximum-curvature method [Wiemer and Wyss, 2000],
and the Gutenberg–Richter $b$-value, estimated by maximum likelihood [Aki, 1965] with
the uncertainty of [Shi and Bolt, 1982]. We obtain $M_c=1.35$ and $b=0.93\pm0.004$,
consistent with tectonic seismicity; the steeper apparent slope of the cumulative
distribution at large magnitude reflects the finite catalog duration. Magnitudes below
$M_\mathrm{L}\approx1.5$ (below the calibration anchor range) are linear extrapolations
and should be treated with corresponding caution.

## Data and software

Amplitudes were measured with the project amplitude routine
(`4_relocation/calculate_amplitudes.py`). The inversion, calibration, quality control,
and figures were produced by `4_relocation/magnitude/phase1`–`phase6`. Sparse least
squares used SciPy; ComCat access used ObsPy's FDSN client; maps used PyGMT. The
per-station calibration terms are provided in
`4_relocation/magnitude/station_corrections_kpos.csv`.

## References

- Aki, K. (1965). Maximum likelihood estimate of *b* in the formula log *N* = *a* − *bM*. *Bull. Earthq. Res. Inst.*, 43, 237–239.
- Hutton, L. K., and D. M. Boore (1987). The *M*<sub>L</sub> scale in southern California. *BSSA*, 77, 2074–2094.
- Morton, E. A., et al. (2023). A test of the Cascadia offshore seismicity... *J. Geophys. Res. Solid Earth*, 128, e2023JB026607.
- Paige, C. C., and M. A. Saunders (1982). LSQR: An algorithm for sparse linear equations and least squares. *ACM Trans. Math. Softw.*, 8, 43–71.
- Richter, C. F. (1935). An instrumental earthquake magnitude scale. *BSSA*, 25, 1–32.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *J. Am. Stat. Assoc.*, 63, 1379–1389.
- Shi, Y., and B. A. Bolt (1982). The standard error of the magnitude–frequency *b* value. *BSSA*, 72, 1677–1687.
- Theil, H. (1950). A rank-invariant method of linear and polynomial regression analysis. *Proc. K. Ned. Akad. Wet.*, 53, 386–392.
- Wiemer, S., and M. Wyss (2000). Minimum magnitude of completeness in earthquake catalogs. *BSSA*, 90, 859–869.
- U.S. Geological Survey, Advanced National Seismic System (ANSS) Comprehensive Catalog (ComCat), https://earthquake.usgs.gov/.
