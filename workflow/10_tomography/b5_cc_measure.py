"""Step 3 of the CC differential times: cross-correlate every pair-station-phase of a set.

For each row of <set>/obs.parquet (pair, common station-phase) the two snippets (b5_cc_snippets.py cache) are
band-passed (zero-phase 4-pole Butterworth on the 16 s snippet, 1 s cosine taper) and a window around each
pick is correlated:
* P: vertical; window [tP - pre, tP + post], post cut to (S-P) - sp_margin of the pair (no S energy inside);
* S: the two horizontals as one vector (vector CC: sum of the component products / sqrt of the total
  energies, invariant to the unknown OBS horizontal orientation) when both events have both horizontals,
  else the vertical (comps = Z);
* event 1 window fixed, event 2 window slid over +-maxlag; normalized CC; parabolic sub-sample peak.
Same channel for both events (else resampled to the lower rate and flagged chan_mismatch).

Definitions (all times on the record-header time base of the snippets):
  dt_cat = (t1 - T0_1) - (t2 - T0_2)  (v5.1 origin times; t = pick times of v4.1t)
  dt_cc  = (w1 - T0_1) - (w2 - T0_2) - tau,  w = window start (pick - pre on the sample grid), tau = CC lag
           (tau > 0: event 2's phase is later relative to its window than event 1's)
  lag_s  = dt_cat - dt_cc  (the correction CC applies to the catalog dt)
  snr    = rms(window) / rms(noise window); P: noise [t-3.5, t-0.5] s (pre-event), S: [t-0.2-Ln, t-0.2] s
           with Ln = clip(S-P - 0.3, 0.3, 3) (S over P coda).
  cc_2nd = highest CC outside +-0.6 period (centre frequency) of the main peak (cycle-skip risk);
           margin = cc_max - cc_2nd.
  max lag: min(p_maxlag or s_maxlag, lag_sp_frac * (S-P)), at least min_lag (S-P of the pair, smaller event).

  python b5_cc_measure.py --set msh --out <file.parquet> [--cfg '{"band":[2,12]}'] [--sample-pairs N]
          [--workers 12] [--source given|added|all]
"""
import argparse
import json
import sys
import time
import traceback
from collections import OrderedDict
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import obspy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402

DEFAULT = dict(band=[2.0, 12.0], p_pre=0.2, p_post=1.0, s_pre=0.3, s_post=1.5, p_maxlag=0.3, s_maxlag=0.5,
               sp_margin=0.1, min_win=0.3, taper=1.0, lag_sp_frac=0.5, min_lag=0.05)


class EventCache:
    def __init__(self, name, n=400):
        self.name = name
        self.d = OrderedDict()
        self.n = n
        self.pre = OrderedDict()

    def get(self, eid):
        if eid in self.d:
            self.d.move_to_end(eid)
            return self.d[eid]
        base = C.SNIP / self.name
        ev = {}
        for ph in ("P", "S"):
            f = base / f"{eid}.{ph}.mseed"
            if f.exists():
                st = obspy.read(str(f))
                for tr in st:
                    key = (f"{tr.stats.station}.{tr.stats.network}", ph)
                    ev.setdefault(key, {})[tr.stats.channel] = tr
        mf = base / f"{eid}.meta.csv"
        meta = pd.read_csv(mf) if mf.exists() else None
        self.d[eid] = (ev, meta)
        if len(self.d) > self.n:
            self.d.popitem(last=False)
        return self.d[eid]


def _resample(x, sr, target):
    if sr == target:
        return x
    from scipy.signal import resample_poly
    fr = Fraction(target / sr).limit_denominator(200)
    return resample_poly(x, fr.numerator, fr.denominator)


def prepped(cache, eid, sta, ph, cha, tr, sr_target, cfg):
    key = (eid, sta, ph, cha, sr_target, tuple(cfg["band"]))
    if key in cache.pre:
        return cache.pre[key]
    sr = tr.stats.sampling_rate
    x = tr.data.astype(np.float64)
    if sr != sr_target:
        x = _resample(x - x.mean(), sr, sr_target)
    y = C.prep(x, sr_target, cfg["band"], cfg["taper"])
    out = (y, tr.stats.starttime.timestamp)
    cache.pre[key] = out
    if len(cache.pre) > 4000:
        cache.pre.popitem(last=False)
    return out


def comps_for(ev, sta, ph, n_h_needed=2):
    tr = ev.get((sta, ph), {})
    z = [c for c in tr if c[2] in "Z3"]
    h = sorted(c for c in tr if c[2] in "EN12")
    return z, h


def measure(row, cache, cfg):
    ph = row.phase
    sta = row.station
    out = dict(pair=row.pair, station=sta, phase=ph)
    ev1, _ = cache.get(int(row.ID1))
    ev2, _ = cache.get(int(row.ID2))
    z1, h1 = comps_for(ev1, sta, ph)
    z2, h2 = comps_for(ev2, sta, ph)
    if ph == "S" and len(h1) == 2 and len(h2) == 2:
        c1, c2, comps = h1, h2, "H"
    elif z1 and z2:
        c1, c2, comps = z1[:1], z2[:1], "Z"
    else:
        out["status"] = "no_waveform"
        out["no_wf_event"] = ("1" if not (z1 or h1) else "") + ("2" if not (z2 or h2) else "")
        return out
    t1s = ev1[(sta, ph)]
    t2s = ev2[(sta, ph)]
    sr1 = t1s[c1[0]].stats.sampling_rate
    sr2 = t2s[c2[0]].stats.sampling_rate
    sr = min(sr1, sr2)
    out.update(comps=comps, channel1=",".join(c1), channel2=",".join(c2), rate1=sr1, rate2=sr2, rate=sr,
               chan_mismatch=bool(c1 != c2 or sr1 != sr2))
    band = cfg["band"]
    pre = cfg["p_pre"] if ph == "P" else cfg["s_pre"]
    post = cfg["p_post"] if ph == "P" else cfg["s_post"]
    maxlag = cfg["p_maxlag"] if ph == "P" else cfg["s_maxlag"]
    sp = np.nanmin([row.sp_est1, row.sp_est2])
    if ph == "P" and np.isfinite(sp):
        post = min(post, sp - cfg["sp_margin"])
    if np.isfinite(sp):
        # the slid window must not reach the other phase: max lag <= lag_sp_frac * (S-P)
        maxlag = max(min(maxlag, cfg["lag_sp_frac"] * sp), cfg["min_lag"])
    out.update(win_pre=pre, win_post=post, maxlag_s=maxlag, band_lo=band[0], band_hi=min(band[1], 0.45 * sr))
    if post + pre < cfg["min_win"]:
        out["status"] = "short_window"
        return out
    L = int(round(maxlag * sr))
    n = int(round((pre + post) * sr)) + 1
    A, B = [], []
    noise1, noise2 = [], []
    if ph == "P":
        nw = (3.5, 0.5)
    else:
        ln = float(np.clip((sp if np.isfinite(sp) else 1.0) - 0.3, 0.3, 3.0))
        nw = (0.2 + ln, 0.2)
    w1 = w2 = None
    for k, (a, b) in enumerate(zip(c1, c2)):
        y1, s1 = prepped(cache, int(row.ID1), sta, ph, a, t1s[a], sr, cfg)
        y2, s2 = prepped(cache, int(row.ID2), sta, ph, b, t2s[b], sr, cfg)
        i1 = int(round((row.t_pick1 - pre - s1) * sr))
        i2 = int(round((row.t_pick2 - pre - s2) * sr))
        m = int(cfg["taper"] * sr)
        if i1 < m or i1 + n > len(y1) - m or i2 - L < m or i2 + n + L > len(y2) - m:
            out["status"] = "short_data"
            return out
        w1 = s1 + i1 / sr
        w2 = s2 + i2 / sr
        A.append(y1[i1:i1 + n])
        B.append(y2[i2 - L:i2 + n + L])
        j1a = int(round((row.t_pick1 - nw[0] - s1) * sr))
        j1b = int(round((row.t_pick1 - nw[1] - s1) * sr))
        j2a = int(round((row.t_pick2 - nw[0] - s2) * sr))
        j2b = int(round((row.t_pick2 - nw[1] - s2) * sr))
        noise1.append(y1[max(j1a, m):max(j1b, m + 1)])
        noise2.append(y2[max(j2a, m):max(j2b, m + 1)])
    A = np.array(A)
    B = np.array(B)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)) or (A == 0).all() or (B == 0).all():
        out["status"] = "bad_data"
        return out
    cc = C.norm_cc(A, B, L)
    k = int(np.argmax(cc))
    frac, cmax = C.subsample_peak(cc, k)
    tau = (k - L + frac) / sr
    fc = np.sqrt(band[0] * min(band[1], 0.45 * sr))
    ex = max(1, int(round(0.6 * sr / fc)))
    mask = np.ones(len(cc), bool)
    mask[max(0, k - ex):k + ex + 1] = False
    cc2 = float(cc[mask].max()) if mask.any() else np.nan
    dt_cc = (w1 - row.T01) - (w2 - row.T02) - tau
    rms = lambda x: float(np.sqrt(np.mean(np.concatenate([np.ravel(v) for v in x]) ** 2))) if len(x) else np.nan
    sig1 = float(np.sqrt(np.mean(A ** 2)))
    sig2 = float(np.sqrt(np.mean(B[:, L:L + n] ** 2)))
    n1, n2 = rms(noise1), rms(noise2)
    out.update(status="ok", maxlag_s=maxlag, cc_max=float(cmax), cc_int=float(cc[k]), lag_s=float(row.dt_cat - dt_cc),
               tau_s=float(tau), dt_cc=float(dt_cc), cc_2nd=cc2, edge=bool(k == 0 or k == len(cc) - 1),
               cc_neg=float(-cc.min()), snr1=sig1 / n1 if n1 > 0 else np.nan, snr2=sig2 / n2 if n2 > 0 else np.nan)
    return out


def run_chunk(args):
    name, rows, cfg = args
    cache = EventCache(name)
    out = []
    for r in rows.itertuples(index=False):
        try:
            out.append(measure(r, cache, cfg))
        except Exception as e:
            traceback.print_exc()
            out.append(dict(pair=r.pair, station=r.station, phase=r.phase, status=f"error:{type(e).__name__}"))
    return pd.DataFrame(out)


def load_obs(name, source="all", sample_pairs=0, seed=0, pairs=None):
    out = C.OUT / name
    obs = pd.read_parquet(out / "obs.parquet")
    ev = C.load_events(name)[["ID", "T0"]]
    obs = obs.merge(ev.rename(columns={"ID": "ID1", "T0": "T01"}), on="ID1")
    obs = obs.merge(ev.rename(columns={"ID": "ID2", "T0": "T02"}), on="ID2")
    if source != "all":
        obs = obs[obs.source == source]
    if pairs is not None:
        obs = obs[obs.pair.isin(pairs)]
    elif sample_pairs:
        pr = obs.drop_duplicates("pair")
        pr = pr[pr.source == "given"].sample(min(sample_pairs, len(pr)), random_state=seed)
        obs = obs[obs.pair.isin(pr.pair)]
    return obs.sort_values(["ID1", "ID2", "station", "phase"]).reset_index(drop=True)


def run(name, obs, cfg, workers=12, chunk=2000):
    chunks = []
    # chunks follow the ID1 order so a worker's event cache is reused
    for k in range(0, len(obs), chunk):
        chunks.append((name, obs.iloc[k:k + chunk], cfg))
    res = []
    if workers <= 1:
        res = [run_chunk(c) for c in chunks]
    else:
        with Pool(workers) as pool:
            for r in pool.imap(run_chunk, chunks):
                res.append(r)
    r = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    return obs.merge(r, on=["pair", "station", "phase"], how="left")


def load_params(path, name):
    if not path:
        return {}
    import yaml
    P = yaml.safe_load(Path(path).read_text())
    return {**P.get("common", {}), **P.get("sets", {}).get(name, {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--params", default="", help="b5_cc_params.yaml: common + sets.<set> override DEFAULT")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="all")
    a = ap.parse_args()
    cfg = {**DEFAULT, **load_params(a.params, a.set), **json.loads(a.cfg)}
    obs = load_obs(a.set, a.source, a.sample_pairs, a.seed)
    t = time.time()
    r = run(a.set, obs, cfg, a.workers)
    r.attrs["cfg"] = json.dumps(cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_parquet(a.out, index=False)
    print(f"{a.set}: {len(r)} rows, {time.time() - t:.0f} s, status {r.status.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
