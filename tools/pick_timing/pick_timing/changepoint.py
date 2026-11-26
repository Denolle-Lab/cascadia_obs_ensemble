"""Change models for a station's network-relative residual time series, fitted on binned medians.

Data model. The picks of one station (P and S pooled after removing each phase's median) are grouped in time
order into bins of at most ``bin_picks`` picks spanning at most ``bin_max_days`` days. Each bin gives a median
y_i at time t_i (days) with sampling variance s2_i = (1.2533 sigma / sqrt(n_i))^2, sigma being the station's
robust (MAD) scale. Bin-to-bin scatter that is not sampling noise (changes in the mix of events and paths) is a
variance tau^2 added to every bin; tau^2 is estimated by the Paule-Mandel method on the residuals of the best
model (started from first differences, which a slow drift or a single step hardly affects).

Models of the clock error c(t) (b = the station's reference level, not part of the clock):
  constant       y = b                                                     k = 1
  step           y = b + a H(t - t1)                                       k = 3
  offset_epoch   y = b + a 1[t1 <= t < t2]                                 k = 4
  drift_open     y = b + r (t - t1) H(t - t1)                              k = 3
  drift_reset    y = b + r (t - t1) 1[t1 <= t < t2]   (free-running clock re-synchronised at t2)   k = 4
Fit: weighted least squares (weights 1 / (s2_i + tau^2)) profiled over the change times, which run over a
grid of bin boundaries (at most ``grid_max``) and, for resets, the header restarts when given; the best grid
point is refined at full bin resolution (and a fine time grid for drift starts). Model choice by BIC:
BIC = chi2 + k ln(n); dBIC = BIC(constant) - BIC(model) > 0 favours the model. A reset time is only known to lie
between the last bin inside the epoch and the first bin after it (t2_lo, t2_hi).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = {"constant": 1, "step": 3, "offset_epoch": 4, "drift_open": 3, "drift_reset": 4}
RAMP = {"drift_open", "drift_reset"}
CLOSED = {"offset_epoch", "drift_reset"}
MEDIAN_EFF = 1.2533  # SE of a median = 1.2533 sigma / sqrt(n) for Gaussian data


# --------------------------------------------------------------------------------------------- binning
def bin_assign(t_days: np.ndarray, bin_picks=20, bin_max_days=7.0, bin_min_picks=3, breaks=None) -> np.ndarray:
    """Bin id per pick (t sorted ascending). A bin closes at bin_picks picks, or when the next pick is more than
    bin_max_days after the bin's first pick and the bin already holds bin_min_picks, and always at a break time
    (header restarts: a bin never straddles a data restart)."""
    ids = np.empty(len(t_days), int)
    b, n, t0 = 0, 0, None
    seg = np.searchsorted(np.sort(np.asarray(breaks, float)), t_days, side="right") if breaks is not None and len(breaks) \
        else np.zeros(len(t_days), int)
    for i, t in enumerate(t_days):
        if n > 0 and (n >= bin_picks or (t - t0 > bin_max_days and n >= bin_min_picks) or seg[i] != seg[i - 1]):
            b += 1
            n = 0
        if n == 0:
            t0 = t
        ids[i] = b
        n += 1
    return ids


def robust_sigma(y) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return np.nan
    return float(1.4826 * np.median(np.abs(y - np.median(y))))


def make_bins(t_days, y, ids, sigma=None, sigma_floor=0.01) -> pd.DataFrame:
    """Bin table: t (median time), t_start, t_end, n, y (median), s2 (sampling variance of the median)."""
    d = pd.DataFrame({"t": t_days, "y": y, "b": ids})
    g = d.groupby("b")
    out = pd.DataFrame({"t": g.t.median(), "t_start": g.t.min(), "t_end": g.t.max(), "n": g.size(),
                        "y": g.y.median()})
    s = robust_sigma(y) if sigma is None else sigma
    s = max(s if np.isfinite(s) else sigma_floor, sigma_floor)
    out["s2"] = (MEDIAN_EFF * s) ** 2 / out.n
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- core WLS
class _Sums:
    def __init__(self, t, y, w):
        self.tc = float(np.mean(t))
        tt = t - self.tc
        z = lambda a: np.r_[0.0, np.cumsum(a)]  # noqa: E731
        self.W, self.WT, self.WTT = z(w), z(w * tt), z(w * tt * tt)
        self.WY, self.WTY, self.WYY = z(w * y), z(w * tt * y), z(w * y * y)
        self.n = len(t)
        self.S, self.Sy, self.Syy = self.W[-1], self.WY[-1], self.WYY[-1]

    def chi2_const(self):
        return self.Syy - self.Sy ** 2 / self.S, self.Sy / self.S

    def eval(self, ramp: bool, i1, c1, i2):
        """chi2, b, a for g = 1[i1 <= i < i2] (indicator) or (t - c1) 1[i1 <= i < i2] (ramp); arrays broadcast."""
        R = lambda P: P[i2] - P[i1]  # noqa: E731
        Wr, WYr = R(self.W), R(self.WY)
        if ramp:
            c = c1 - self.tc
            WTr = R(self.WT)
            G = WTr - c * Wr
            GG = R(self.WTT) - 2 * c * WTr + c * c * Wr
            GY = R(self.WTY) - c * WYr
        else:
            G, GG, GY = Wr, Wr, WYr
        det = self.S * GG - G * G
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = self.Syy - (GG * self.Sy ** 2 - 2 * G * self.Sy * GY + self.S * GY ** 2) / det
            b = (GG * self.Sy - G * GY) / det
            a = (self.S * GY - G * self.Sy) / det
        bad = ~(det > 1e-12 * self.S * np.maximum(GG, 1e-300))
        chi2 = np.where(bad, np.inf, chi2)
        return chi2, b, a


def g_of(model: str, t, t1, t2=np.inf):
    t = np.asarray(t, float)
    if model == "constant":
        return np.zeros_like(t)
    if model == "step":
        return (t >= t1).astype(float)
    if model == "offset_epoch":
        return ((t >= t1) & (t < t2)).astype(float)
    if model == "drift_open":
        return np.where(t >= t1, t - t1, 0.0)
    if model == "drift_reset":
        return np.where((t >= t1) & (t < t2), t - t1, 0.0)
    raise ValueError(model)


def amplitude(model, a, t1, t2, t_last):
    """Largest |c(t)| of the fitted clock term (s)."""
    if model in ("step", "offset_epoch"):
        return abs(a)
    if model == "drift_open":
        return abs(a) * max(t_last - t1, 0.0)
    if model == "drift_reset":
        return abs(a) * max(t2 - t1, 0.0)
    return 0.0


# --------------------------------------------------------------------------------------------- fitting
def _bounds(t_start, t_end):
    """Candidate change times between bins j-1 and j (j = 1..n-1): midpoint of the gap."""
    return 0.5 * (t_end[:-1] + t_start[1:])


def _fit_model(Sm, model, bins, J, cJ, p, restarts_days=None):
    """Grid search over change indices for one model. Returns dict or None."""
    n = len(bins)
    t = bins.t.values
    nb_out = 3
    ramp = model in RAMP
    if model in ("step", "drift_open"):
        i1, c1 = J, cJ
        i2 = np.full_like(J, n)
        ok = (i1 >= nb_out) & (n - i1 >= max(p["min_bins_in_epoch"], nb_out))
        ok &= (t[-1] - c1) >= p["min_epoch_days"]
        if not ok.any():
            return None
        chi2, b, a = Sm.eval(ramp, i1, c1, i2)
        chi2 = np.where(ok, chi2, np.inf)
        k = int(np.argmin(chi2))
        if not np.isfinite(chi2[k]):
            return None
        return dict(model=model, chi2=float(chi2[k]), i1=int(i1[k]), c1=float(c1[k]), i2=n, c2=np.inf,
                    b=float(b[k]), a=float(a[k]))
    # two change times: i1 from J, i2 from J (+ restarts as extra end candidates)
    J2, c2 = J, cJ
    if restarts_days is not None and len(restarts_days):
        rr = np.asarray(restarts_days, float)
        rr = rr[(rr > t[0]) & (rr < t[-1])]
        if len(rr):
            ri = np.searchsorted(t, rr)  # first bin at or after the restart
            J2 = np.r_[ri, J2]           # restarts first: on a tie (no pick between) the reset is the restart
            c2 = np.r_[rr, c2]
    I1, C1 = J[:, None], cJ[:, None]
    I2, C2 = J2[None, :], c2[None, :]
    ok = (I1 >= nb_out) & (I2 - I1 >= p["min_bins_in_epoch"]) & (n - I2 >= 1) & (C2 - C1 >= p["min_epoch_days"])
    if not ok.any():
        return None
    chi2, b, a = Sm.eval(ramp, I1, C1, I2)
    chi2 = np.where(ok, chi2, np.inf)
    k = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    if not np.isfinite(chi2[k]):
        return None
    return dict(model=model, chi2=float(chi2[k]), i1=int(J[k[0]]), c1=float(cJ[k[0]]), i2=int(J2[k[1]]),
                c2=float(c2[k[1]]), b=float(b[k]), a=float(a[k]))


def _refine(Sm, f, bins, bounds, p, restarts_days=None):
    """Refine the change indices at full bin resolution near the grid optimum (and the ramp start on a fine time
    grid inside the neighbouring gaps)."""
    n = len(bins)
    t = bins.t.values
    model = f["model"]
    ramp = model in RAMP
    span = max(2, int(np.ceil(n / max(p["grid_max"], 1))) + 1)
    lo1, hi1 = max(3, f["i1"] - span), min(n - 1, f["i1"] + span)
    i1s = np.arange(lo1, hi1 + 1)
    c1s = bounds[i1s - 1]
    if ramp:
        # fine grid of start times across each candidate gap and the bins themselves
        grid = np.linspace(bounds[max(lo1 - 2, 0)], bounds[min(hi1, n - 2)], 400)
        c1s = np.r_[c1s, grid]
        i1s = np.searchsorted(t, c1s)
    if model in ("step", "drift_open"):
        i2s, c2s = np.array([n]), np.array([np.inf])
    else:
        lo2, hi2 = max(f["i1"] + 1, f["i2"] - span), min(n - 1, f["i2"] + span)
        i2s = np.arange(lo2, hi2 + 1)
        c2s = bounds[i2s - 1]
        if restarts_days is not None and len(restarts_days) and np.isfinite(f["c2"]) \
                and np.any(np.isclose(np.asarray(restarts_days, float), f["c2"], rtol=0, atol=1e-9)):
            i2s, c2s = np.array([f["i2"]]), np.array([f["c2"]])   # keep a reset that sits on a header restart
    I1, C1, I2, C2 = i1s[:, None], c1s[:, None], i2s[None, :], c2s[None, :]
    ok = (I1 >= 3) & (I2 - I1 >= p["min_bins_in_epoch"]) & ((I2 == n) | (n - I2 >= 1))
    ok &= np.where(np.isfinite(C2), C2 - C1, t[-1] - C1) >= p["min_epoch_days"]
    if model in ("step", "drift_open"):
        ok &= (n - I1) >= 3
    chi2, b, a = Sm.eval(ramp, I1, C1, I2)
    chi2 = np.where(ok, chi2, np.inf)
    k = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    if np.isfinite(chi2[k]) and chi2[k] <= f["chi2"] + 1e-9:
        f = dict(f, chi2=float(chi2[k]), i1=int(I1[k[0], 0]), c1=float(C1[k[0], 0]), i2=int(I2[0, k[1]]),
                 c2=float(C2[0, k[1]]), b=float(b[k]), a=float(a[k]))
    return f


def paule_mandel(r, s2, dof):
    """tau^2 >= 0 with sum r^2 / (s2 + tau^2) = dof (bisection); 0 if already below."""
    r2 = r * r
    if dof <= 0 or np.sum(r2 / s2) <= dof:
        return 0.0
    lo, hi = 0.0, max(np.var(r) * 10, 1e-6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sum(r2 / (s2 + mid)) > dof:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tau2_from_diffs(bins):
    y, s2 = bins.y.values, bins.s2.values
    if len(y) < 4:
        return 0.0
    d = np.diff(y)
    v = (1.4826 * np.median(np.abs(d - np.median(d)))) ** 2
    return max(0.0, 0.5 * (v - np.median(s2[:-1] + s2[1:])))


def fit_bins(bins: pd.DataFrame, p: dict, models=None, restarts_days=None, tau2=None, n_iter=2) -> dict:
    """Fit every model to a bin table. Returns dict with per-model fits, dBIC, the best model and tau."""
    models = list(models or p["models"])
    bins = bins.sort_values("t").reset_index(drop=True)
    n = len(bins)
    t = bins.t.values
    bounds = _bounds(bins.t_start.values, bins.t_end.values)
    J = np.unique(np.linspace(1, n - 1, min(p["grid_max"], n - 1)).round().astype(int)) if n > 2 else np.array([1])
    cJ = bounds[J - 1]
    fixed = tau2 is not None
    tau2 = tau2_from_diffs(bins) if tau2 is None else tau2
    fits, best = {}, None
    for it in range(1 if fixed else n_iter + 1):
        w = 1.0 / (bins.s2.values + tau2)
        Sm = _Sums(t, bins.y.values, w)
        c0, b0 = Sm.chi2_const()
        fits = {"constant": dict(model="constant", chi2=float(c0), b=float(b0), a=0.0, c1=np.nan, c2=np.nan,
                                 i1=0, i2=n)}
        for m in models:
            f = _fit_model(Sm, m, bins, J, cJ, p, restarts_days)
            if f is not None:
                fits[m] = _refine(Sm, f, bins, bounds, p, restarts_days)
        lnn = np.log(n)
        for f in fits.values():
            f["k"] = K[f["model"]]
            f["bic"] = f["chi2"] + f["k"] * lnn
        best = min(fits.values(), key=lambda f: f["bic"])
        if fixed or it == n_iter:
            break
        pred = best["b"] + best["a"] * g_of(best["model"], t, best["c1"], best["c2"])
        tau2 = paule_mandel(bins.y.values - pred, bins.s2.values, n - best["k"])
    b0 = fits["constant"]["bic"]
    for f in fits.values():
        f["dbic"] = b0 - f["bic"]
        f["amp"] = amplitude(f["model"], f["a"], f["c1"], f["c2"], t[-1])
        if f["model"] in CLOSED:
            f["c2_lo"] = float(bins.t_end.values[f["i2"] - 1])
            f["c2_hi"] = float(bins.t_start.values[f["i2"]]) if f["i2"] < n else np.inf
        else:
            f["c2_lo"] = f["c2_hi"] = np.nan
        f["c1_lo"] = float(bins.t_end.values[f["i1"] - 1]) if f["i1"] >= 1 and f["model"] != "constant" else np.nan
        f["c1_hi"] = float(bins.t_start.values[f["i1"]]) if f["i1"] < n and f["model"] != "constant" else np.nan
        f["n_in"] = int(f["i2"] - f["i1"]) if f["model"] != "constant" else 0
    return dict(fits=fits, best=best["model"], tau=float(np.sqrt(tau2)), n_bins=n)


def predict(f: dict, t):
    return f["b"] + f["a"] * g_of(f["model"], t, f["c1"], f["c2"])


def wild_bootstrap(bins: pd.DataFrame, fit: dict, p: dict, n_boot=200, seed=0, restarts_days=None) -> np.ndarray:
    """Wild (Rademacher) residual bootstrap of the best model: rows (c1, c2, a, b, amp)."""
    rng = np.random.default_rng(seed)
    bins = bins.sort_values("t").reset_index(drop=True)
    f = fit["fits"][fit["best"]]
    if f["model"] == "constant":
        return np.zeros((0, 5))
    pred = predict(f, bins.t.values)
    r = bins.y.values - pred
    out = []
    for _ in range(n_boot):
        bb = bins.copy()
        bb["y"] = pred + r * rng.choice([-1.0, 1.0], len(r))
        ff = fit_bins(bb, p, models=[f["model"]], restarts_days=restarts_days, tau2=fit["tau"] ** 2)
        g = ff["fits"].get(f["model"])
        if g is None:
            continue
        out.append((g["c1"], g["c2"], g["a"], g["b"], g["amp"]))
    return np.array(out)
