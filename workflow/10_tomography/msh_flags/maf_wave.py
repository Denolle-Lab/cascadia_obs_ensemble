"""Waveform signature of the pulse at one pick (used by maf_score.py).

Two edge templates cut from the empirical template (maf_template.py): ON = onset edge (-10..+20 s around the onset;
the negative box 0-1.8 s masked), OFF = plateau end (-10..+20 s around the end of the plateau, 27.5 s after the
onset). A standard pulse has both edges 27.5 s apart; on some Wednesdays the plateau lasts until the second pulse
(~20 min), so the edges are matched separately.
check_pick(sta, t_pick): reads t_pick - 45 s .. t_pick + 45 s (10-Hz block means of the raw counts), finds edges
with |cc| >= CC_MIN and |amplitude| / noise >= SNR_MIN (either polarity: some channels carry the pulse inverted)
and calls the pick matched when an edge lies at most EDGE_LEAD (3) s after or EDGE_LAG (10) s before the pick,
i.e. edge - EDGE_LEAD <= t_pick <= edge + EDGE_LAG (in_edge(); the wording of RULE.md).
"""
import numpy as np
import pandas as pd

import maf_common as C

CC_MIN = 0.9
SNR_MIN = 5.0
EDGE_LEAD = 3.0      # the pick may precede the edge by up to 3 s
EDGE_LAG = 10.0      # ... or follow it by up to 10 s
T_END = 27.5         # plateau end after the onset (s), from the template
READ = 45.0
_T = {}


def edge_templates():
    if not _T:
        tt, y, w = C.template()
        for name, t0 in (("on", 0.0), ("off", T_END)):
            m = (tt >= t0 - 10.0) & (tt < t0 + 20.0)
            _T[name] = (tt[m] - t0, y[m], w[m])
    return _T


def in_edge(t_pick, t_edge):
    return (t_edge - EDGE_LEAD) <= t_pick <= (t_edge + EDGE_LAG)


def edges(x, s0):
    """Edges in one 10-Hz segment (x starting at POSIX s0): list of dict(kind, t, cc, amp, snr)."""
    out = []
    sig = C.noise_sigma(x)
    for name, (tt, ty, tw) in edge_templates().items():
        cc, amp = C.cc_series(x, tt, ty, tw)
        if len(cc) == 0:
            continue
        lead = -tt[0]
        ok = np.where((np.abs(cc) >= CC_MIN) & (np.abs(amp) / sig >= SNR_MIN))[0]
        # keep local maxima of |cc| (one per 5 s)
        last = -10 ** 9
        for i in ok[np.argsort(-np.abs(cc[ok]))]:
            if any(abs(i - j) < 5 * C.FS for j in [e["_i"] for e in out if e["kind"] == name]):
                continue
            out.append(dict(kind=name, _i=int(i), t=s0 + i / C.FS + lead, cc=float(cc[i]), amp=float(amp[i]),
                            snr=float(abs(amp[i]) / sig)))
    return out


def check_pick(sta, t_pick):
    """t_pick POSIX float. Returns dict(status, matched, edge kind / cc / snr / dt of the best edge near the pick)."""
    a = pd.Timestamp(t_pick - READ, unit="s", tz="UTC")
    b = pd.Timestamp(t_pick + READ, unit="s", tz="UTC")
    segs, st = C.read10(sta, a, b)
    if not segs:
        return dict(status=st, matched=False)
    best = None
    best_any = None
    for s0, x in segs:
        for e in edges(x, s0):
            d = t_pick - e["t"]
            if best_any is None or abs(e["cc"]) > abs(best_any["cc"]):
                best_any = dict(e, dt=d)
            if in_edge(t_pick, e["t"]) and (best is None or abs(e["cc"]) > abs(best["cc"])):
                best = dict(e, dt=d)
    if best is not None:
        return dict(status="ok", matched=True, edge=best["kind"], cc=best["cc"], snr=best["snr"], dt=best["dt"],
                    amp=best["amp"])
    r = dict(status="ok", matched=False)
    if best_any is not None:
        r.update(edge_far=best_any["kind"], cc_far=best_any["cc"], dt_far=best_any["dt"])
    return r
