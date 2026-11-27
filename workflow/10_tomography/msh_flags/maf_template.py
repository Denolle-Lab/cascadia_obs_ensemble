"""Step 1: empirical template of the weekly pulse (before any scoring).

Pulses of two Wednesdays with catalogued window events (2011-11-23 PST, 2012-05-02 PDT), both pulses of each day
(~11:58 and ~12:18 local), at HSR, EDM, STD and SHW.UW: 16 traces. Onset = first 10-Hz sample below half of the
negative box minimum. Each trace is cut T_PRE before to T_POST after its onset, divided by its plateau (median of
20-26 s after the onset) and the traces are averaged. The negative box (0-1.8 s) is masked in the correlation
(its depth is set by clipping and differs between stations).
Output: WORK/template.npz (t, y, w), WORK/template_members.csv, FIG/template.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import maf_common as C  # noqa: E402


def onset_in(x, t0):
    i = int(np.argmin(x))
    lo = x[i]
    j = i
    while j > 0 and x[j - 1] < 0.5 * lo:
        j -= 1
    return t0 + j / C.FS - 0.05


def main():
    C.WORK.mkdir(parents=True, exist_ok=True)
    C.FIG.mkdir(parents=True, exist_ok=True)
    tt = np.arange(-int(C.T_PRE * C.FS), int(C.T_POST * C.FS)) / C.FS
    rows, stack = [], []
    for day in ("2011-11-23", "2012-05-02"):
        for hm in ("11:50", "12:10"):
            a = pd.Timestamp(f"{day} {hm}", tz=C.TZ).tz_convert("UTC")
            b = a + pd.Timedelta(minutes=12)
            for sta in ("HSR.UW", "EDM.UW", "STD.UW", "SHW.UW"):
                segs, st = C.read10(sta, a, b)
                if not segs:
                    print(day, hm, sta, st)
                    continue
                s0, x = max(segs, key=lambda s: len(s[1]))
                on = onset_in(x, s0)
                tx = s0 + np.arange(len(x)) / C.FS
                y = np.interp(on + tt, tx, x)
                base = np.median(y[tt < -1])
                y = y - base
                plat = np.median(y[(tt >= 20) & (tt <= 26)])
                rows.append(dict(day=day, sta=sta, onset_utc=pd.Timestamp(on, unit="s", tz="UTC").isoformat(),
                                 onset_local=C.local(pd.Timestamp(on, unit="s", tz="UTC")).isoformat(),
                                 plateau=plat, box_min=float(y[(tt >= 0) & (tt < 1.8)].min())))
                stack.append(y / plat)
    S = np.array(stack)
    y = np.median(S, axis=0)
    w = np.ones_like(y)
    w[(tt >= C.MASK[0]) & (tt < C.MASK[1])] = 0.0
    np.savez(C.WORK / "template.npz", t=tt, y=y, w=w)
    pd.DataFrame(rows).to_csv(C.WORK / "template_members.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for s in S:
        ax.plot(tt, s, color="#9b9a96", lw=0.6)
    ax.plot(tt, np.where(w > 0, y, np.nan), color="#0b0b0b", lw=1.6, label="template (median, mask removed)")
    ax.axvspan(*C.MASK, color="#eb6834", alpha=0.2, label="masked: negative box")
    ax.set_xlabel("s after the pulse onset")
    ax.set_ylabel("raw counts / plateau")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(color="#d9d8d4", lw=0.4)
    fig.tight_layout()
    fig.savefig(C.FIG / "template.png", dpi=110)


if __name__ == "__main__":
    main()
