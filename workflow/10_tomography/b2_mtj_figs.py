"""MTJ shallow-anomaly diagnosis (notes/mtj_shallow.md): figures in data/tomo_v1/b2_mtj/.

  fig1_baseline_maps.png    dlnVp of the baseline (d1000_s20) at -1..12 km; stations coloured by their
                            location-only static (median absolute residual after relocation in the start model,
                            network median removed); nodes never inverted in grey
  fig2_evolution.png        Vp per iteration at the reference nodes; weighting sets shaded; 0.4 km/s clip steps
  fig3_station_vs_node.png  station static vs mean dlnVp of the nodes just below the station (P and S)
  fig4_tests.png            dlnVp at 1 and 5 km for each test run (args: tags), titles with n(>25%) and abs RMS

  python b2_mtj_figs.py [TAG ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_mtj_common import FIG, TESTS, grid, read_res, read_vel, stations  # noqa: E402
from b2_mtj_analyse import REF_NODES  # noqa: E402

BASE_TAG = "d1000_s20"
CMAP = "RdBu"            # red = slow, blue = fast (diverging, neutral midpoint)
EXT = (-128.3, -121.7, 39.0, 43.6)


def statics():
    r = read_res(TESTS / "loconly")
    a = r[r.absolute & (r.wt > 0)]
    c = a.groupby(["code", "phase"]).res_ms.agg(["median", "size"]).reset_index()
    c["rel_s"] = (c["median"] - c.groupby("phase")["median"].transform("median")) / 1000
    st = stations()
    return c.merge(st, on="code")


def dln(tag):
    V = read_vel(tag)
    g = grid(tag)
    its = sorted(V)
    d = V[its[-1]]["vp"] / g["vp0"] - 1
    active = (V[its[0]]["dwsp"] >= 0.2)
    return np.where(active, d, np.nan), g, V


def map_panel(ax, g, d, k, title, sta=None):
    LON, LAT = np.meshgrid(g["lon"][1:-1], g["lat"][1:-1], indexing="ij")
    ax.set_facecolor("0.85")
    pc = ax.pcolormesh(LON, LAT, 100 * d[1:-1, 1:-1, k], cmap=CMAP + "_r" if False else CMAP, vmin=-40, vmax=40,
                       shading="nearest")
    if sta is not None:
        ax.scatter(sta.sta_lon, sta.sta_lat, c=sta.rel_s, cmap="PuOr_r", vmin=-0.8, vmax=0.8, s=14, marker="^",
                   edgecolors="k", linewidths=0.3, zorder=3)
    ax.plot([-125.2, -124.0, -124.0, -125.2, -125.2], [40.0, 40.0, 40.6, 40.6, 40.0], "k--", lw=0.7)
    ax.set_xlim(EXT[:2])
    ax.set_ylim(EXT[2:])
    ax.set_aspect(1 / np.cos(np.radians(41.3)))
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    return pc


def fig1(sta):
    d, g, _ = dln(BASE_TAG)
    sP = sta[sta.phase == "P"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6), layout="constrained")
    for ax, z in zip(axs.flat, [-1, 1, 3, 5, 9, 12]):
        k = int(np.argmin(abs(g["z"] - z)))
        pc = map_panel(ax, g, d, k, f"baseline d1000_s20: dlnVp at {z} km", sP)
    cb = fig.colorbar(pc, ax=axs, shrink=0.6, location="right", pad=0.02)
    cb.set_label("dlnVp (%)  (red slower, blue faster; grey = node never inverted)", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="PuOr_r", norm=plt.Normalize(-0.8, 0.8))
    cb2 = fig.colorbar(sm, ax=axs, shrink=0.6, location="left", pad=0.04)
    cb2.set_label("station P static (s, location-only run, network median removed)", fontsize=7)
    fig.savefig(FIG / "fig1_baseline_maps.png", dpi=160)
    plt.close(fig)


def fig2():
    _, g, V = dln(BASE_TAG)
    its = sorted(V)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    for s in range(4):
        ax.axvspan(3 * s + 0.5, 3 * s + 3.5, color=["0.80", "0.88", "0.94", "1.0"][s], zorder=0)
        ax.text(3 * s + 2, 1.2, ["set 1\nWTCD 10", "set 2\nWTCD 10", "set 3\nWTCD 1", "set 4\nWTCD 0.1"][s],
                ha="center", fontsize=7)
    for lon, lat, z in REF_NODES:
        i, j, k = (int(np.argmin(abs(g["lon"] - lon))), int(np.argmin(abs(g["lat"] - lat))),
                   int(np.argmin(abs(g["z"] - z))))
        v = [g["vp0"][i, j, k]] + [V[t]["vp"][i, j, k] for t in its]
        ax.plot(range(13), v, "o-", ms=3, lw=1.5, label=f"{lat:.1f}N {-lon:.3f}W {z:g} km")
    ax.set_xlabel("iteration (0 = start)")
    ax.set_ylabel("Vp (km/s)")
    ax.set_ylim(1, 9)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.set_title("baseline: Vp at the reference nodes per iteration", fontsize=9)
    ax = axs[1]
    vp = np.stack([g["vp0"]] + [V[t]["vp"] for t in its])
    st = np.abs(np.diff(vp, axis=0))[:, 1:-1, 1:-1, 1:-1]
    ax.bar(range(1, 13), (st > 0.39).reshape(12, -1).sum(1), color="0.35")
    ax.set_xlabel("iteration")
    ax.set_ylabel("nodes with a Vp step at the 0.4 km/s clip")
    ax.set_title("baseline: clipped updates per iteration", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_evolution.png", dpi=160)
    plt.close(fig)


def below_station(g, d, s):
    ii = [np.searchsorted(g["lon"], s.sta_lon) - 1, np.searchsorted(g["lon"], s.sta_lon)]
    jj = [np.searchsorted(g["lat"], s.sta_lat) - 1, np.searchsorted(g["lat"], s.sta_lat)]
    zs = -s.sta_elev_km
    ks = [k for k in range(1, len(g["z"]) - 1) if zs - 2 <= g["z"][k] <= zs + 5]
    return float(np.nanmean(d[np.ix_(ii, jj, ks)]))


def fig3(sta):
    V = read_vel(BASE_TAG)
    g = grid()
    out = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ph, key in ((axs[0], "P", "vp"), (axs[1], "S", "vs")):
        d = V[max(V)][key] / g[key + "0"] - 1
        x = sta[(sta.phase == ph) & (sta["size"] >= 50)].copy()
        x["dln_below"] = [below_station(g, d, s) for s in x.itertuples()]
        rho = spearmanr(x.rel_s, x.dln_below).statistic
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(0, color="0.7", lw=0.8)
        for obs, mk, lab in ((False, "s", "land"), (True, "^", "OBS")):
            y = x[x.is_obs == obs]
            ax.scatter(y.rel_s, 100 * y.dln_below, marker=mk, s=18, label=lab, alpha=0.8)
        for s in x[(x.rel_s.abs() > 0.45) | (x.dln_below.abs() > 0.13)].itertuples():
            ax.annotate(s.station.split(".")[0], (s.rel_s, 100 * s.dln_below), fontsize=6, xytext=(2, 2),
                        textcoords="offset points")
        ax.set_xlabel(f"station {ph} static (s, location-only run, network median removed)")
        ax.set_ylabel(f"mean dln{ph.replace('P', 'Vp').replace('S', 'Vs')} of nodes below station (%)")
        ax.set_title(f"baseline, {ph}: Spearman rho = {rho:.2f} (n = {len(x)})", fontsize=9)
        ax.legend(fontsize=7)
        out.append(x.assign(rho=rho))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_station_vs_node.png", dpi=160)
    plt.close(fig)
    pd.concat(out)[["station", "phase", "median", "size", "rel_s", "dln_below", "is_obs", "sta_lat", "sta_lon"]] \
        .round(4).to_csv(FIG / "station_static_vs_node.csv", index=False)


def fig4(tags):
    cmp_ = pd.read_csv(FIG / "runs_compare.csv").set_index("tag")
    fig, axs = plt.subplots(2, len(tags), figsize=(2.7 * len(tags), 5.4), squeeze=False, layout="constrained")
    for c, tag in enumerate(tags):
        d, g, _ = dln(tag)
        for r, z in enumerate([1, 5]):
            k = int(np.argmin(abs(g["z"] - z)))
            t = f"{tag}\n{z} km" if r == 0 else f"{z} km"
            if r == 0 and tag in cmp_.index:
                t += f"\nn>25%: {int(cmp_.loc[tag, 'n_gt25'])}, absP {cmp_.loc[tag, 'abs_P_common_rms_ms']:.0f} ms"
            pc = map_panel(axs[r, c], g, d, k, t)
            axs[r, c].set_xlim(-126.0, -123.3)
            axs[r, c].set_ylim(39.7, 41.0)
    cb = fig.colorbar(pc, ax=axs, shrink=0.6, pad=0.01)
    cb.set_label("dlnVp (%)", fontsize=7)
    fig.savefig(FIG / "fig4_tests.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sta = statics()
    fig1(sta)
    fig2()
    fig3(sta)
    if len(sys.argv) > 1:
        fig4(sys.argv[1:])


if __name__ == "__main__":
    main()
