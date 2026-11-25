"""Checks of the CC code (b5_cc_measure.py) before the full run.

1. Synthetic: a real snippet against a copy delayed by a known fraction of a sample (Fourier shift) and with a
   known pick offset: recovered dt must equal the imposed dt (sign convention dt_cc = dt_cat - lag_s) and the
   parabolic sub-sample bias is measured per sampling rate and band.
2. 20-pair test: 5 random given pairs per set (seed 1), all common station-phases, wide max lag (1 s), plots of
   the waveform pairs (filtered windows, event 2 shifted by the CC lag) -> <out>/check/.

  python b5_cc_check.py [--synthetic] [--pairs]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402
import b5_cc_measure as M  # noqa: E402

OUTC = C.OUT / "check"


def fshift(x, d):
    """Delay x by d samples (Fourier shift)."""
    n = len(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * d), n)


def synthetic():
    rows = []
    for name in C.SETS:
        obs = M.load_obs(name, sample_pairs=5, seed=1)
        cache = M.EventCache(name)
        for r in obs.drop_duplicates(["ID1", "station", "phase"]).head(12).itertuples(index=False):
            ev, _ = cache.get(int(r.ID1))
            trs = ev.get((r.station, r.phase))
            if not trs:
                continue
            for d_samp in (-7.3, -2.5, -0.5, -0.25, 0.0, 0.1, 0.33, 0.5, 1.7, 4.9):
                for band in ([2, 12], [3, 20], [1, 8]):
                    cfg = {**M.DEFAULT, "band": band, "p_maxlag": 0.5, "s_maxlag": 0.5}
                    # event 2 = event 1 delayed by d samples, same origin time and pick shifted by 0.137 s
                    ev2 = {}
                    for cha, tr in trs.items():
                        t2 = tr.copy()
                        t2.data = fshift(tr.data.astype(np.float64), d_samp)
                        ev2[cha] = t2
                    sr = trs[next(iter(trs))].stats.sampling_rate
                    c2 = M.EventCache(name)
                    c2.d[int(r.ID1)] = cache.get(int(r.ID1))
                    c2.d[-1] = ({(r.station, r.phase): ev2}, None)
                    true_dt = -d_samp / sr                    # (a1 - T0) - (a2 - T0), a2 = a1 + d/sr
                    row = r._replace(ID2=-1, t_pick2=r.t_pick1 + 0.137, T02=r.T01,
                                     dt_cat=(r.t_pick1 - r.T01) - (r.t_pick1 + 0.137 - r.T01))
                    res = M.measure(row, c2, cfg)
                    if res.get("status") != "ok":
                        continue
                    rows.append(dict(set=name, station=r.station, phase=r.phase, rate=sr, band=str(band),
                                     d_samp=d_samp, true_dt=true_dt, dt_cc=res["dt_cc"], cc=res["cc_max"],
                                     err_ms=(res["dt_cc"] - true_dt) * 1e3))
    df = pd.DataFrame(rows)
    OUTC.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTC / "synthetic_shift_test.csv", index=False)
    df["frac"] = (df.d_samp - np.round(df.d_samp)).abs()
    s = df.groupby(["rate", "band"]).err_ms.agg(["count", "median", lambda x: np.abs(x).max()]).round(3)
    s.columns = ["n", "median_err_ms", "max_abs_err_ms"]
    print(s.to_string())
    print("sign check: max |err| over all", round(float(df.err_ms.abs().max()), 3), "ms; min cc", round(float(df.cc.min()), 4))
    return df


def plot_pair(r, cache, cfg, ax):
    ev1, _ = cache.get(int(r.ID1))
    ev2, _ = cache.get(int(r.ID2))
    trs1, trs2 = ev1.get((r.station, r.phase), {}), ev2.get((r.station, r.phase), {})
    if not trs1 or not trs2 or r.status != "ok":
        ax.text(0.5, 0.5, f"{r.station} {r.phase}: {r.status}", transform=ax.transAxes, ha="center")
        return
    chans = r.channel1.split(",")
    chans2 = r.channel2.split(",")
    sr = r.rate
    for j, (a, b) in enumerate(zip(chans, chans2)):
        y1, s1 = M.prepped(cache, int(r.ID1), r.station, r.phase, a, trs1[a], sr, cfg)
        y2, s2 = M.prepped(cache, int(r.ID2), r.station, r.phase, b, trs2[b], sr, cfg)
        t1 = s1 + np.arange(len(y1)) / sr - r.t_pick1
        # event 2 on event 1's pick-relative axis after the CC correction: shift by (t_pick2 - t_pick1) + lag
        t2 = s2 + np.arange(len(y2)) / sr - r.t_pick2 - r.tau_s
        m1 = (t1 > -1.5) & (t1 < 3)
        m2 = (t2 > -1.5) & (t2 < 3)
        sc1 = np.abs(y1[m1 & (t1 > -r.win_pre) & (t1 < r.win_post)]).max() or 1
        sc2 = np.abs(y2[m2 & (t2 > -r.win_pre) & (t2 < r.win_post)]).max() or 1
        off = -2.2 * j
        ax.plot(t1[m1], y1[m1] / sc1 + off, "k", lw=0.7)
        ax.plot(t2[m2], y2[m2] / sc2 + off, "r", lw=0.7, alpha=0.8)
        # event 2 at its catalog pick (no CC shift), grey
        t2c = s2 + np.arange(len(y2)) / sr - r.t_pick2
        m2c = (t2c > -1.5) & (t2c < 3)
        ax.plot(t2c[m2c], y2[m2c] / sc2 + off - 1.0, color="0.6", lw=0.6)
    ax.axvspan(-r.win_pre, r.win_post, color="y", alpha=0.15)
    ax.axvline(0, color="b", lw=0.5)
    ax.set_title(f"{r.ID1}-{r.ID2} {r.station} {r.phase} [{r.comps}] cc={r.cc_max:.2f} "
                 f"lag={r.lag_s * 1e3:+.0f} ms sep={r.sep_km * 1e3:.0f} m {r.timing1[:5]}/{r.timing2[:5]}",
                 fontsize=6)
    ax.set_yticks([])
    ax.tick_params(labelsize=6)


def pairs_test(cfg_over=None):
    cfg = {**M.DEFAULT, "p_maxlag": 1.0, "s_maxlag": 1.0, **(cfg_over or {})}
    allr = []
    for name in C.SETS:
        obs = M.load_obs(name, sample_pairs=5, seed=1)
        r = M.run(name, obs, cfg, workers=1)
        r["set"] = name
        allr.append(r)
        cache = M.EventCache(name)
        for pid, g in r.groupby("pair"):
            n = len(g)
            nc = 3
            nr = int(np.ceil(n / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(10, 1.9 * nr), squeeze=False)
            for ax, rr in zip(axs.ravel(), g.itertuples(index=False)):
                plot_pair(rr, cache, cfg, ax)
            for ax in axs.ravel()[n:]:
                ax.axis("off")
            fig.suptitle(f"{name} pair {g.ID1.iloc[0]}-{g.ID2.iloc[0]} band {cfg['band']} Hz: black ev1, red ev2 "
                         f"shifted by CC lag, grey ev2 at catalog pick; yellow = CC window", fontsize=8)
            fig.tight_layout()
            (OUTC / "pairs20").mkdir(parents=True, exist_ok=True)
            fig.savefig(OUTC / "pairs20" / f"{name}_{g.ID1.iloc[0]}_{g.ID2.iloc[0]}.png", dpi=110)
            plt.close(fig)
    df = pd.concat(allr, ignore_index=True)
    df.to_csv(OUTC / "pairs20_results.csv", index=False)
    cols = ["set", "ID1", "ID2", "station", "phase", "comps", "rate", "timing1", "timing2", "dt_cat", "dt_cc",
            "lag_s", "cc_max", "cc_2nd", "snr1", "snr2", "status"]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df[cols].round(3).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    OUTC.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        synthetic()
    if a.pairs:
        pairs_test(json.loads(a.cfg))


if __name__ == "__main__":
    main()
