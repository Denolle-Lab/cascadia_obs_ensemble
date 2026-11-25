"""detect-clock: screen every station for clock offsets and drifts in its network-relative residuals.

Per station (picks with a network-relative residual rd; P and S pooled after removing each phase's median):
 1. bin, fit the change models (changepoint.fit_bins), keep the model with the largest dBIC among those whose
    clock term reaches ``amp_min_s`` (statistic T);
 2. pre-screen: T >= ``dbic_min``;
 3. permutation test: the residuals are shuffled among the station's picks (times kept) ``n_perm`` times and
    refitted; p = (1 + #{T_perm >= T}) / (1 + n_perm); candidate if p <= ``p_max``;
 4. up to ``max_epochs`` epochs: the fitted clock term is removed and the series screened again.
Evidence per candidate epoch: wild-bootstrap uncertainties, P and S fitted separately (a clock moves both by
the same amount), a co-located-event control (pairs of events within ``colocated_event_km`` of each other, one
inside and one outside the epoch: a clock error survives the pair difference, a change in the event mix does
not), coincidence of the epoch end with a MiniSEED header restart, and CC differential times when given.

False alarms are estimated by running the same detector (with its permutation test) on (a) synthetic clean
stations with each real station's pick times and noise level (Student-t picks plus bin-level scatter tau) and
(b) each real station with its residuals shuffled in time; detection power by injecting drifts into (b).
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import changepoint as cp
from .tables import posix_to_iso, write_json, write_table

DAY = 86400.0


# --------------------------------------------------------------------------------------------- statistic
def _stat(fit, p):
    """Best non-constant model whose clock term reaches amp_min: (T, model). T = -inf if none."""
    best, T = None, -np.inf
    for m, f in fit["fits"].items():
        if m == "constant" or f["amp"] < p["amp_min_s"]:
            continue
        if f["dbic"] > T:
            T, best = f["dbic"], m
    return T, best


def _fit(t, y, ids, sigma, p, restarts):
    bins = cp.make_bins(t, y, ids, sigma)
    return bins, cp.fit_bins(bins, p, restarts_days=restarts)


def permutation_p(t, y, ids, sigma, p, T_obs, restarts, rng, n_perm, stop_after=10):
    """Sequential permutation p-value (stops once `stop_after` permutations reach T_obs)."""
    ge, k = 0, 0
    Ts = []
    for k in range(1, n_perm + 1):
        _, f = _fit(t, rng.permutation(y), ids, sigma, p, restarts)
        T, _ = _stat(f, p)
        Ts.append(T)
        if T >= T_obs:
            ge += 1
            if ge >= stop_after:
                break
    return (1 + ge) / (1 + k), k, float(np.nanmax(np.where(np.isfinite(Ts), Ts, np.nan))) if Ts else np.nan


def detect_series(t, y, p, restarts=None, seed=0, max_epochs=2, n_perm=None, keep_series=False):
    """Screen one station. t in days (any origin), y = network-relative residual (s). Returns (screen, epochs)."""
    rng = np.random.default_rng(seed)
    n_perm = p["n_perm"] if n_perm is None else n_perm
    o = np.argsort(t, kind="stable")
    t, y = np.asarray(t, float)[o], np.asarray(y, float)[o]
    screen = dict(n_picks=len(t), status="screened")
    if len(t) < p["min_picks"]:
        screen["status"] = "insufficient_picks"
        return screen, []
    ids = cp.bin_assign(t, p["bin_picks"], p["bin_max_days"], p["bin_min_picks"], breaks=restarts)
    if ids[-1] + 1 < p["min_bins"]:
        screen.update(status="insufficient_bins", n_bins=int(ids[-1] + 1))
        return screen, []
    sigma = cp.robust_sigma(y)
    yw = y.copy()
    epochs = []
    for e in range(max_epochs):
        bins, fit = _fit(t, yw, ids, sigma, p, restarts)
        T, m = _stat(fit, p)
        if e == 0:
            screen.update(n_bins=fit["n_bins"], sigma=sigma, tau=fit["tau"], best_bic_model=fit["best"],
                          T=T, T_model=m, **{f"dbic_{k}": v["dbic"] for k, v in fit["fits"].items() if k != "constant"},
                          **{f"amp_{k}": v["amp"] for k, v in fit["fits"].items() if k != "constant"})
        if m is None or T < p["dbic_min"]:
            break
        pval, nperm, Tperm_max = permutation_p(t, yw, ids, sigma, p, T, restarts, rng, n_perm)
        if e == 0:
            screen.update(p_perm=pval, n_perm_run=nperm, T_perm_max=Tperm_max)
        if pval > p["p_max"]:
            break
        f = fit["fits"][m]
        ep = dict(epoch_index=e, model=m, T=T, p_perm=pval, n_perm_run=nperm, fit=f, tau=fit["tau"],
                  sigma=sigma, n_bins=fit["n_bins"], dbic_all={k: v["dbic"] for k, v in fit["fits"].items()})
        if keep_series:
            ep["bins"], ep["fit_all"], ep["y"] = bins, fit, yw.copy()
        epochs.append(ep)
        yw = yw - f["a"] * cp.g_of(m, t, f["c1"], f["c2"])
    screen["n_epochs"] = len(epochs)
    screen["candidate"] = len(epochs) > 0
    return screen, epochs


# --------------------------------------------------------------------------------------------- evidence
def huber_slope(x, y, scale=0.05, intercept=False, n_iter=10):
    """Huber IRLS regression of y on x (through the origin unless intercept). Returns (slope, se, n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.all(x == 0):
        return np.nan, np.nan, int(len(x))
    A = np.c_[x, np.ones_like(x)] if intercept else x[:, None]
    w = np.ones(len(x))
    for _ in range(n_iter):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ beta
        w = 1.0 / np.maximum(1.0, np.abs(r) / scale)
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    try:
        cov = np.linalg.inv((A * w[:, None]).T @ A) * s * s
        se = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        se = np.nan
    return float(beta[0]), se, int(len(x))


def ps_consistency(x: pd.DataFrame, f: dict):
    """Clock amplitude fitted on P and on S separately (same change times)."""
    out = {}
    for ph in ("P", "S"):
        z = x[x.phase1 == ph]
        g = cp.g_of(f["model"], z.t.values, f["c1"], f["c2"])
        if len(z) < 10 or np.count_nonzero(g) < 5:
            out[f"a_{ph}"], out[f"a_{ph}_se"], out[f"n_{ph}"] = np.nan, np.nan, len(z)
            continue
        a, se, n = huber_slope(g, z.y.values, scale=0.1, intercept=True)
        out[f"a_{ph}"], out[f"a_{ph}_se"], out[f"n_{ph}"] = a, se, n
    out["ratio_S_P"] = out["a_S"] / out["a_P"] if np.isfinite(out.get("a_P", np.nan)) and out["a_P"] != 0 else np.nan
    return out


def colocated_control(x: pd.DataFrame, f: dict, radius_km=5.0, k=3, seed=0, n_boot=200):
    """Pairs (event inside the epoch, event outside it) within radius_km, same phase: slope of the pair difference
    of rd against the pair difference of g. A clock gives the fitted a; a change of event mix gives ~0."""
    from scipy.spatial import cKDTree
    if not {"ev_x_km", "ev_y_km", "ev_z_km"} <= set(x.columns):
        return dict(coloc_a=np.nan, coloc_se=np.nan, coloc_n=0, coloc_ratio=np.nan, coloc_note="no event coordinates")
    g = cp.g_of(f["model"], x.t.values, f["c1"], f["c2"])
    dy, dg = [], []
    for ph in ("P", "S"):
        m = (x.phase1 == ph).values
        inn = m & (g != 0)
        out = m & (g == 0)
        if inn.sum() < 3 or out.sum() < 3:
            continue
        P_out = x.loc[out, ["ev_x_km", "ev_y_km", "ev_z_km"]].values
        tree = cKDTree(P_out)
        d, j = tree.query(x.loc[inn, ["ev_x_km", "ev_y_km", "ev_z_km"]].values, k=min(k, out.sum()),
                          distance_upper_bound=radius_km)
        d, j = np.atleast_2d(d.T).T if d.ndim == 1 else d, np.atleast_2d(j.T).T if j.ndim == 1 else j
        yi, gi = x.y.values[inn], g[inn]
        yo, go = x.y.values[out], g[out]
        for c in range(j.shape[1]):
            ok = np.isfinite(d[:, c])
            dy.append(yi[ok] - yo[j[ok, c]])
            dg.append(gi[ok] - go[j[ok, c]])
    if not dy:
        return dict(coloc_a=np.nan, coloc_se=np.nan, coloc_n=0, coloc_ratio=np.nan, coloc_note="no pairs")
    dy, dg = np.concatenate(dy), np.concatenate(dg)
    if len(dy) < 10:
        return dict(coloc_a=np.nan, coloc_se=np.nan, coloc_n=int(len(dy)), coloc_ratio=np.nan,
                    coloc_note="fewer than 10 pairs")
    a, _, n = huber_slope(dg, dy, scale=0.1)
    rng = np.random.default_rng(seed)
    bs = [huber_slope(dg[i], dy[i], scale=0.1)[0] for i in (rng.integers(0, len(dy), len(dy)) for _ in range(n_boot))]
    se = float(np.nanstd(bs))
    return dict(coloc_a=a, coloc_se=se, coloc_n=n, coloc_ratio=a / f["a"] if f["a"] else np.nan, coloc_note="")


def cc_refine(cc: pd.DataFrame, station: str, f: dict, exclude: set, q: dict, boots=None, seed=0, n_boot=200):
    """Size of the clock term from CC differential times: y = dt_cc(station) - median dt_cc of the other stations
    of the same pair (>= min_consensus rows, candidate stations excluded) = a (g(t1) - g(t2)). Huber IRLS through
    the origin on rows with g(t1) != g(t2); pair bootstrap combined with the residual bootstrap of the times."""
    if cc is None or not len(cc):
        return dict(cc_n=0)
    d = cc
    if q.get("accepted_only", True) and "accepted" in d.columns:
        d = d[d.accepted.astype(bool)]
    if "pair" not in d.columns:
        d = d.assign(pair=d.event1.astype(str) + "-" + d.event2.astype(str))
    others = d[~d.station.isin(exclude)]
    cons = others.groupby("pair").dt_cc.agg(["median", "size"])
    x = d[d.station == station].copy()
    if not len(x):
        return dict(cc_n=0)
    x["cons"] = x.pair.map(cons["median"])
    x["ncons"] = x.pair.map(cons["size"]).fillna(0)
    x = x[x.ncons >= q["min_consensus"]]
    t1, t2 = x.time1_s.values / DAY, x.time2_s.values / DAY
    y = (x.dt_cc - x.cons).values

    def one(c1, c2, idx):
        g = cp.g_of(f["model"], t1[idx], c1, c2) - cp.g_of(f["model"], t2[idx], c1, c2)
        m = g != 0
        if m.sum() < q["min_rows"]:
            return np.nan, int(m.sum())
        a, _, n = huber_slope(g[m], y[idx][m], scale=q["huber_s"])
        return a, n
    a, n = one(f["c1"], f["c2"], np.arange(len(x)))
    if not np.isfinite(a):
        return dict(cc_n=n, cc_rows=len(x))
    rng = np.random.default_rng(seed)
    bs = []
    for k in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        c1, c2 = (boots[k % len(boots), 0], boots[k % len(boots), 1]) if boots is not None and len(boots) else (f["c1"], f["c2"])
        bs.append(one(c1, c2, idx)[0])
    g = cp.g_of(f["model"], t1, f["c1"], f["c2"]) - cp.g_of(f["model"], t2, f["c1"], f["c2"])
    resid_before = y[g != 0]
    resid_after = (y - a * g)[g != 0]
    return dict(cc_a=a, cc_a_sd=float(np.nanstd(bs)), cc_n=n, cc_rows=len(x),
                cc_amp=cp.amplitude(f["model"], a, f["c1"], f["c2"], t1.max() if len(t1) else np.nan),
                cc_med_abs_before=float(np.median(np.abs(resid_before))) if len(resid_before) else np.nan,
                cc_med_abs_after=float(np.median(np.abs(resid_after))) if len(resid_after) else np.nan)


def restart_check(restarts_days, f, tol_days):
    if restarts_days is None or not len(restarts_days):
        return dict(reset_at_restart=None, restart_utc="", restarts_in_epoch=0)
    r = np.asarray(restarts_days, float)
    out = dict(restarts_in_epoch=int(((r > f["c1"]) & (r < (f["c2"] if np.isfinite(f["c2"]) else np.inf))).sum()))
    if f["model"] in cp.CLOSED:
        lo, hi = f["c2_lo"] - tol_days, (f["c2_hi"] if np.isfinite(f["c2_hi"]) else f["c2_lo"]) + tol_days
        hit = r[(r >= lo) & (r <= hi)]
        out.update(reset_at_restart=bool(len(hit)), restart_utc=posix_to_iso(hit[0] * DAY, 3) if len(hit) else "")
    else:
        out.update(reset_at_restart=None, restart_utc="")
    return out


def classify(row: dict) -> str:
    """Evidence summary used to order the review (a person decides)."""
    sup, against = [], []
    if np.isfinite(row.get("coloc_ratio", np.nan)) and row.get("coloc_n", 0) >= 10:
        r = row["coloc_ratio"]
        if 0.5 <= r <= 1.5 and abs(row["coloc_a"]) > 2 * row.get("coloc_se", np.inf):
            sup.append("colocated")
        elif r < 0.3:
            against.append("colocated~0")
    ap, as_ = row.get("a_P", np.nan), row.get("a_S", np.nan)
    if np.isfinite(ap) and np.isfinite(as_):
        if np.sign(ap) == np.sign(as_) == np.sign(row["a"]) and 0.5 <= as_ / ap <= 2.0:
            sup.append("P=S")
        elif np.sign(ap) != np.sign(as_):
            against.append("P/S opposite")
    if row.get("reset_at_restart") is True:
        sup.append("reset@restart")
    if np.isfinite(row.get("cc_a", np.nan)) and row.get("cc_n", 0) >= 10:
        if np.sign(row["cc_a"]) == np.sign(row["a"]) and abs(row["cc_a"]) > 3 * row.get("cc_a_sd", np.inf):
            sup.append("CC")
        elif abs(row["cc_a"]) < 2 * row.get("cc_a_sd", 0):
            against.append("CC~0")
    if "colocated" in sup and len(sup) >= 2 and not against:
        cls = "clock-like"
    elif against and not sup:
        cls = "event-mix-like"
    else:
        cls = "unclear"
    return cls + (" [" + ",".join(sup) + "]" if sup else "") + (" {" + ",".join(against) + "}" if against else "")


# --------------------------------------------------------------------------------------------- per station job
def _station_job(args):
    st, x, p, restarts, seed, mode = args
    t = x.t.values
    y = x.y.values
    if mode == "real":
        screen, epochs = detect_series(t, y, p, restarts, seed=seed, max_epochs=p.get("max_epochs", 2),
                                       keep_series=True)
        return st, mode, screen, epochs
    rng = np.random.default_rng(seed)
    if mode.startswith("synthetic"):
        sig = cp.robust_sigma(y)
        ids = cp.bin_assign(np.sort(t), p["bin_picks"], p["bin_max_days"], p["bin_min_picks"])
        tau = p.get("_tau", {}).get(st, 0.0) or 0.0
        ys = sig / 1.134 * rng.standard_t(3, len(t))
        ys = ys + rng.normal(0, tau, ids[-1] + 1)[ids[np.argsort(np.argsort(t))]] if tau > 0 else ys
        screen, epochs = detect_series(t, ys, p, None, seed=seed, max_epochs=1)
    elif mode.startswith("permuted"):
        screen, epochs = detect_series(t, rng.permutation(y), p, None, seed=seed, max_epochs=1)
    elif mode.startswith("inject"):
        _, rate, days = mode.split(":")
        rate, days = float(rate), float(days)
        yp = rng.permutation(y)
        ts = np.sort(t)
        if ts[-1] - ts[0] < days + 30:
            return st, mode, dict(status="too_short"), []
        te = rng.uniform(ts[0] + days + 10, ts[-1] - 5)
        t1 = te - days
        yp = yp + rate * np.where((t >= t1) & (t < te), t - t1, 0.0)
        screen, epochs = detect_series(t, yp, p, None, seed=seed, max_epochs=1)
        screen.update(inj_t1=t1, inj_te=te, inj_rate=rate)
        if epochs:
            f = epochs[0]["fit"]
            screen.update(rec_model=epochs[0]["model"], rec_t1=f["c1"], rec_a=f["a"],
                          rec_ok=bool(epochs[0]["model"] in cp.RAMP and abs(f["c1"] - t1) <= 0.25 * days
                                      and 0.5 <= f["a"] / rate <= 1.5))
        else:
            screen.update(rec_ok=False)
    return st, mode, screen, epochs


def _pool_map(fn, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            return list(pool.imap_unordered(fn, jobs, chunksize=1))
    return [fn(j) for j in jobs]


# --------------------------------------------------------------------------------------------- driver
def station_frames(rd: pd.DataFrame) -> dict:
    """{station: DataFrame(t days, y, phase1, event_id, ev_*)} with y = rd minus the phase median."""
    d = rd[np.isfinite(rd.rd)].copy()
    d["t"] = d.time_s / DAY
    d["y"] = d.rd - d.groupby(["station", "phase1"]).rd.transform("median")
    cols = ["t", "y", "phase1", "event_id", "time_s"] + [c for c in ("ev_x_km", "ev_y_km", "ev_z_km") if c in d.columns]
    return {s: g[cols].sort_values("t").reset_index(drop=True) for s, g in d.groupby("station")}


def load_restarts(path) -> dict:
    """scan-headers events CSV -> {station: restart times in days}. Rows with kind 'restart' (time tears)."""
    if not path:
        return {}
    e = pd.read_csv(path)
    if "kind" in e.columns:
        e = e[e.kind.isin(["restart", "tear"])]
    tcol = "time_s" if "time_s" in e.columns else "time"
    ts = e[tcol].values if tcol == "time_s" else pd.to_datetime(e[tcol].str.rstrip("Z"), format="ISO8601").astype("int64") / 1e9
    e = e.assign(_t=np.asarray(ts, float) / DAY)
    return {s: np.sort(g._t.unique()) for s, g in e.groupby("station")}


def run_detect(rd: pd.DataFrame, p: dict, workers=4, restarts=None, cc=None, cc_q=None, false_alarm=True,
               out_dir=None, log=print):
    if 1.0 / (p["n_perm"] + 1) > p["p_max"]:
        raise ValueError(f"n_perm = {p['n_perm']} cannot reach p_max = {p['p_max']}: need n_perm >= "
                         f"{int(np.ceil(1 / p['p_max'] - 1))}")
    restarts = restarts or {}
    frames = station_frames(rd)
    log(f"detect-clock: {len(frames)} stations, {sum(len(v) for v in frames.values())} picks with rd")
    jobs = [(s, x, p, restarts.get(s), p["seed"] + k, "real") for k, (s, x) in enumerate(frames.items())]
    jobs.sort(key=lambda j: -len(j[1]))
    t0 = time.time()
    res = _pool_map(_station_job, jobs, workers)
    log(f"  real screen done in {time.time() - t0:.0f} s")
    screen_rows, cand_rows, series = [], [], {}
    cand_stations = {st for st, _, _, ep in res if ep}
    for st, _, sc, epochs in res:
        x = frames[st]
        row = dict(station=st, net=st.split(".")[-1] if "." in st else "", **sc,
                   t_first=posix_to_iso(x.t.min() * DAY, 0), t_last=posix_to_iso(x.t.max() * DAY, 0),
                   n_P=int((x.phase1 == "P").sum()), n_S=int((x.phase1 == "S").sum()))
        screen_rows.append(row)
        for ep in epochs:
            f = ep["fit"]
            xe = x.assign(y=ep["y"])            # the series this epoch was found in (earlier epochs removed)
            boots = cp.wild_bootstrap(ep["bins"], dict(ep["fit_all"], best=ep["model"]), p, n_boot=p["n_boot"],
                                      seed=p["seed"], restarts_days=restarts.get(st))
            cid = f"{st}#{ep['epoch_index'] + 1}"
            r = dict(candidate_id=cid, station=st, net=row["net"], model=ep["model"], dbic=ep["T"],
                     p_perm=ep["p_perm"], n_perm_run=ep["n_perm_run"], n_picks=len(x), n_bins=ep["n_bins"],
                     n_bins_in=f["n_in"], tau_s=ep["tau"], sigma_s=ep["sigma"],
                     t_start_utc=posix_to_iso(f["c1"] * DAY, 0),
                     t_start_lo_utc=posix_to_iso(f["c1_lo"] * DAY, 0) if np.isfinite(f["c1_lo"]) else "",
                     t_start_hi_utc=posix_to_iso(f["c1_hi"] * DAY, 0) if np.isfinite(f["c1_hi"]) else "",
                     t_end_utc=posix_to_iso(f["c2"] * DAY, 0) if np.isfinite(f["c2"]) else "",
                     t_end_lo_utc=posix_to_iso(f["c2_lo"] * DAY, 0) if np.isfinite(f.get("c2_lo", np.nan)) else "",
                     t_end_hi_utc=posix_to_iso(f["c2_hi"] * DAY, 0) if np.isfinite(f.get("c2_hi", np.nan)) else "",
                     t_start_days=f["c1"], t_end_days=f["c2"], a=f["a"], b=f["b"], amp_s=f["amp"],
                     a_units="s/day" if ep["model"] in cp.RAMP else "s",
                     sd_t_start_days=float(np.nanstd(boots[:, 0])) if len(boots) else np.nan,
                     sd_t_end_days=float(np.nanstd(np.where(np.isfinite(boots[:, 1]), boots[:, 1], np.nan)))
                     if len(boots) else np.nan,
                     sd_a=float(np.nanstd(boots[:, 2])) if len(boots) else np.nan,
                     sd_amp_s=float(np.nanstd(boots[:, 4])) if len(boots) else np.nan,
                     **{f"dbic_{k}": v for k, v in ep["dbic_all"].items() if k != "constant"})
            r.update(ps_consistency(xe, f))
            r.update(colocated_control(xe, f, p["colocated_event_km"], seed=p["seed"]))
            r.update(restart_check(restarts.get(st), f, p["reset_tolerance_days"]))
            if cc is not None:
                r.update(cc_refine(cc, st, f, cand_stations, cc_q, boots=boots, seed=p["seed"]))
            r["evidence"] = classify(r)
            cand_rows.append(r)
            series[cid] = dict(boots=boots, bins=ep["bins"], fit=f, y=ep["y"])
    screen = pd.DataFrame(screen_rows).sort_values(["candidate", "T"], ascending=[False, False], na_position="last")
    cands = pd.DataFrame(cand_rows)
    if len(cands):
        cands = cands.sort_values(["dbic"], ascending=False).reset_index(drop=True)
    fa = None
    if false_alarm:
        fa = run_false_alarm(frames, screen, p, workers, log)
    return screen, cands, fa, frames, series


def run_false_alarm(frames, screen, p, workers, log=print):
    fa = p["false_alarm"]
    scr = screen[screen.status == "screened"]
    p2 = dict(p, _tau=dict(zip(scr.station, scr.tau)))
    jobs = []
    k = 10_000
    for st in scr.station:
        x = frames[st]
        for i in range(fa.get("n_synthetic", 1)):
            jobs.append((st, x, p2, None, p["seed"] + k, f"synthetic{i}"))
            k += 1
        for i in range(fa.get("n_permuted", 1)):
            jobs.append((st, x, p2, None, p["seed"] + k, f"permuted{i}"))
            k += 1
        for rate, days in fa.get("inject", []):
            jobs.append((st, x, p2, None, p["seed"] + k, f"inject:{rate:g}:{days:g}"))
            k += 1
    jobs.sort(key=lambda j: -len(j[1]))
    t0 = time.time()
    res = _pool_map(_station_job, jobs, workers)
    log(f"  false-alarm / injection runs ({len(jobs)}) done in {time.time() - t0:.0f} s")
    rows = []
    for st, mode, sc, epochs in res:
        rows.append(dict(station=st, net=st.split(".")[-1], mode=mode, flagged=bool(epochs),
                         T=sc.get("T", np.nan), p_perm=sc.get("p_perm", np.nan), status=sc.get("status"),
                         **{k: v for k, v in sc.items() if k.startswith(("inj_", "rec_"))}))
    runs = pd.DataFrame(rows)
    summ = {}
    family = np.where(runs["mode"].str.startswith("inject"), runs["mode"],
                      runs["mode"].str.replace(r"\d+$", "", regex=True))
    for mode, g in runs.groupby(family):
        g = g[g.status == "screened"]
        if not len(g):
            continue
        d = dict(n=int(len(g)), flagged=int(g.flagged.sum()), rate=float(g.flagged.mean()))
        if mode.startswith(("synthetic", "permuted")):
            Tf = g["T"].replace(-np.inf, np.nan)
            d.update(T_p95=float(np.nanpercentile(Tf, 95)) if Tf.notna().any() else None,
                     T_max=float(Tf.max()) if Tf.notna().any() else None,
                     prescreen_rate=float((Tf >= p["dbic_min"]).mean()))
            # binomial 95 % upper limit of the per-station false-alarm rate
            n, kf = len(g), int(g.flagged.sum())
            d["rate_upper95"] = float(_binom_upper(kf, n))
        if mode.startswith("inject"):
            d["recovered"] = int(g["rec_ok"].map(lambda v: v is not None and v == v and bool(v)).sum()) \
                if "rec_ok" in g else 0
            d["recovery_rate"] = d["recovered"] / d["n"]
        summ[mode] = d
    return dict(summary=summ, runs=runs)


def _binom_upper(k, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if n == 0 else (1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k)))


# --------------------------------------------------------------------------------------------- command line
def cli_detect(cfg, a):
    from .config import out_path
    from .residuals import event_xyz, network_relative, station_xy
    from .tables import load, provenance
    p = dict(cfg["detect"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res_spec = dict(cfg["residuals"] or {})
    if a.residuals:
        res_spec["path"] = a.residuals
    cfg2 = dict(cfg, residuals=res_spec)
    print(f"reading residuals {res_spec['path']}", flush=True)
    res = load(cfg2, "residuals")
    if a.start:
        res = res[res.time_s >= pd.Timestamp(a.start).timestamp()]
    if a.end:
        res = res[res.time_s < pd.Timestamp(a.end).timestamp()]
    stations = load(cfg, "stations") if cfg.get("stations") else None
    origins = load(cfg, "origins") if cfg.get("origins") else None
    res = station_xy(res, stations)
    res = event_xyz(res, origins)
    print(f"{len(res)} residual rows, {res.station.nunique()} stations; network-relative residuals ...", flush=True)
    rd = network_relative(res, p, workers=cfg["workers"])
    keep = [c for c in ("event_id", "station", "phase", "phase1", "time_s", "residual", "weight", "rd", "n_ref",
                        "ref_kind", "ev_x_km", "ev_y_km", "ev_z_km", "x_km", "y_km") if c in rd.columns]
    write_table(rd[keep], out / "rd.parquet")
    if a.stations_only:
        want = set(a.stations_only.split(","))
        rd = rd[rd.station.isin(want)]
    restarts = load_restarts(a.restarts)
    cc = None
    cc_spec = dict(cfg.get("cc") or {})
    if a.cc:
        cc_spec["path"] = a.cc
    if cc_spec.get("path"):
        cc = load(dict(cfg, cc=cc_spec), "cc")
        print(f"CC table: {len(cc)} rows", flush=True)
    screen, cands, fa, frames, series = run_detect(rd, p, cfg["workers"], restarts, cc, cfg["cc_refine"],
                                      false_alarm=not a.no_false_alarm, log=lambda s: print(s, flush=True))
    write_table(screen, out / "screen_stations.csv")
    write_table(cands, out / "candidates.csv")
    sd = out / "series"
    sd.mkdir(exist_ok=True)
    for cid, s in series.items():
        tag = cid.replace("#", "_")
        s["bins"].to_csv(sd / f"{tag}_bins.csv", index=False)
        pd.DataFrame(s["boots"], columns=["t_start_days", "t_end_days", "a", "b", "amp_s"]).to_csv(
            sd / f"{tag}_boot.csv", index=False)
    if fa is not None:
        write_table(fa["runs"], out / "false_alarm_runs.csv")
        write_json(fa["summary"], out / "false_alarm.json")
    write_json(provenance({"residuals": res_spec.get("path"), "restarts": a.restarts, "cc": cc_spec.get("path")},
                          {"detect_params": p, "n_stations": int(screen.station.nunique()),
                           "n_screened": int((screen.status == "screened").sum()),
                           "n_candidates": int(len(cands)), "false_alarm": fa["summary"] if fa else None,
                           "config": cfg.get("_config_path")}),
               out / "detect.provenance.json")
    print(f"screened {int((screen.status == 'screened').sum())} of {len(screen)} stations; "
          f"{len(cands)} candidate epochs at {cands.station.nunique() if len(cands) else 0} stations", flush=True)
    if fa:
        print(json.dumps(fa["summary"], indent=1, default=float), flush=True)
    return 0
