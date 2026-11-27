"""Step 0: when are the picks on the analog channels? (pick timing only; window definition, before any scoring)

Local time = America/Los_Angeles (PST/PDT by date). Figures in FIG/, numbers in WORK/hist.json:
  hist_week.png     picks per weekday x hour (local) on HSR, EDM, STD.UW against all other stations (ratio of shares)
  hist_wed_fine.png Wednesday 11:00-13:30 local, 1-min bins, picks on HSR, EDM, STD.UW per year
  station_excess.csv picks per station in the Wednesday noon window (11:50-12:25 local) against the same clock
                    window on the other six weekdays (mean per weekday)
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import maf_common as C  # noqa: E402

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main():
    C.FIG.mkdir(parents=True, exist_ok=True)
    p = C.picks()
    lo = p.t.dt.tz_convert(C.TZ)
    p["dow"] = lo.dt.dayofweek.values
    p["hh"] = lo.dt.hour.values
    p["mod"] = (lo.dt.hour * 60 + lo.dt.minute + lo.dt.second / 60).values
    p["yr"] = lo.dt.year.values
    a = p.station.isin(C.NAMED)
    ta = pd.crosstab(p[a].dow, p[a].hh).reindex(index=range(7), columns=range(24), fill_value=0)
    to = pd.crosstab(p[~a].dow, p[~a].hh).reindex(index=range(7), columns=range(24), fill_value=0)
    ratio = (ta / ta.values.sum()) / (to / to.values.sum())
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.3), gridspec_kw=dict(width_ratios=[1, 1]))
    im = axs[0].imshow(ta.values, aspect="auto", cmap="viridis")
    axs[0].set_title("picks on HSR, EDM, STD.UW (2010-2016) per weekday and local hour", fontsize=9, loc="left")
    fig.colorbar(im, ax=axs[0], fraction=0.03)
    im = axs[1].imshow(ratio.values, aspect="auto", cmap="cividis", vmin=0.5, vmax=2.0)
    axs[1].set_title("share of HSR/EDM/STD picks / share of all other picks", fontsize=9, loc="left")
    fig.colorbar(im, ax=axs[1], fraction=0.03)
    for ax in axs:
        ax.set_yticks(range(7))
        ax.set_yticklabels(DOW)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("hour of day, local (PST/PDT)")
    fig.tight_layout()
    fig.savefig(C.FIG / "hist_week.png", dpi=110)
    plt.close(fig)

    w = a & (p.dow == 2) & (p["mod"] >= 660) & (p["mod"] < 810)
    o = a & (p.dow != 2) & (p["mod"] >= 660) & (p["mod"] < 810)
    yrs = sorted(y for y in p[w].yr.unique() if y >= 2010)
    fig, axs = plt.subplots(len(yrs), 1, figsize=(10, 1.15 * len(yrs) + 0.6), sharex=True)
    bins = np.arange(660, 811, 1)
    fine = {}
    for ax, y in zip(axs, yrs):
        hw, _ = np.histogram(p[w & (p.yr == y)]["mod"], bins)
        ho, _ = np.histogram(p[o & (p.yr == y)]["mod"], bins)
        ax.bar(bins[:-1], hw, width=1, align="edge", color="#2a78d6", label="Wednesday")
        ax.step(bins[:-1], ho / 6, where="post", color="#eb6834", lw=0.9, label="other weekdays (mean per day)")
        ax.text(0.005, 0.75, str(y), transform=ax.transAxes, fontsize=8)
        ax.grid(color="#d9d8d4", lw=0.4)
        fine[int(y)] = {f"{int(b) // 60:02d}:{int(b) % 60:02d}": int(c) for b, c in zip(bins[:-1], hw) if c >= 5}
    axs[0].legend(fontsize=7, loc="upper right")
    axs[0].set_title("picks on HSR, EDM, STD.UW, 11:00-13:30 local, 1-min bins", fontsize=9, loc="left")
    tk = np.arange(660, 811, 15)
    axs[-1].set_xticks(tk)
    axs[-1].set_xticklabels([f"{t // 60:02d}:{t % 60:02d}" for t in tk])
    axs[-1].set_xlabel("local time (PST/PDT)")
    fig.tight_layout()
    fig.savefig(C.FIG / "hist_wed_fine.png", dpi=110)
    plt.close(fig)

    win = (p["mod"] >= 11 * 60 + 50) & (p["mod"] < 12 * 60 + 25)
    g = pd.DataFrame({"wed": p[win & (p.dow == 2)].groupby("station").size(),
                      "other_per_day": p[win & (p.dow != 2)].groupby("station").size() / 6}).fillna(0)
    g["excess"] = g.wed - g.other_per_day
    g = g.sort_values("excess", ascending=False)
    g.round(2).to_csv(C.WORK / "station_excess.csv")
    json.dump(dict(n_picks=int(len(p)), n_named=int(a.sum()),
                   named_wed_11_13=int((a & (p.dow == 2) & (p.hh.isin([11, 12]))).sum()),
                   named_other_11_13_mean=float((a & (p.dow != 2) & (p.hh.isin([11, 12]))).sum() / 6),
                   ratio_wed_11=float(ratio.loc[2, 11]), ratio_wed_12=float(ratio.loc[2, 12]),
                   fine_bins_ge5=fine, top_excess=g.head(20).round(1).reset_index().to_dict("records")),
              open(C.WORK / "hist.json", "w"), indent=1)
    print(g.head(20).round(1).to_string())


if __name__ == "__main__":
    main()
