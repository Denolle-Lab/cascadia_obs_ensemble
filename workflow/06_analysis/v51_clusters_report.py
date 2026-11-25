"""Figures and self-contained HTML report for the catalog v5.1 clustering (v51_clusters_zbz.py, v51_swarms.py).

Reads data/catalog_v5/v5.1_analysis/clusters/ (and the event table on /wd1), the v4 results in
data/catalog_v4/, and writes fig_*.png and catalog_v51_clusters_report.html (figures embedded) in the same
folder.  Run after v51_clusters_zbz.py and v51_swarms.py:
    python v51_clusters_report.py [--no-map]
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "utils"))
from paper_style import use, FULL, INK, MUTED  # noqa: E402
import v51_clusters_zbz as Z  # noqa: E402
from v51_clusters_zbz import REGIONS, REGION_NAMES, REG_ORDER, KM, YR  # noqa: E402

OUT = Z.OUT
V4 = Z.V4
V4S = Z.ROOT / "data/catalog_v4/swarms"
TYPE_COLORS = {"mainshock-aftershock": "#0072B2", "swarm": "#D55E00", "multiplet": "#009E73",
               "background": "#9A9A9A"}
REFS = [
    ("Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits. "
     "Bull. Earthq. Res. Inst. 43, 237-239.", None),
    ("Baiesi, M., & Paczuski, M. (2004). Scale-free networks of earthquakes and aftershocks. Phys. Rev. E 69, "
     "066106.", "10.1103/PhysRevE.69.066106"),
    ("Boettcher, M. S., & McGuire, J. J. (2009). Scaling relations for seismic cycles on mid-ocean ridge "
     "transform faults. GRL 36, L21301.", "10.1029/2009GL040115"),
    ("Hsu, Y.-F., Martínez-Garzón, P., Trugman, D. T., & Ben-Zion, Y. (2026). Earthquake clustering and stress "
     "inversion in the Mendocino Triple Junction region. GJI 245(3), ggag089. (first author's initials not "
     "checked)", "10.1093/gji/ggag089"),
    ("Kuna, V. M., Nábělek, J. L., & Braunmiller, J. (2019). Mode of slip and crust-mantle interaction at "
     "oceanic transform faults. Nat. Geosci. 12, 138-142.", "10.1038/s41561-018-0287-1"),
    ("McGuire, J. J., Boettcher, M. S., & Jordan, T. H. (2005). Foreshock sequences and short-term earthquake "
     "predictability on East Pacific Rise transform faults. Nature 434, 457-461.", "10.1038/nature03377"),
    ("Roland, E., & McGuire, J. J. (2009). Earthquake swarms on transform faults. GJI 178(3), 1677-1690.",
     "10.1111/j.1365-246X.2009.04214.x"),
    ("Shapiro, S. A., Huenges, E., & Borm, G. (1997). Estimating the crust permeability from fluid-injection-"
     "induced seismic emission at the KTB site. GJI 131(2), F15-F18.", "10.1111/j.1365-246X.1997.tb01215.x"),
    ("Vidale, J. E., & Shearer, P. M. (2006). A survey of 71 earthquake bursts across southern California. "
     "JGR 111, B05312.", "10.1029/2005JB004034"),
    ("Wells, D. L., & Coppersmith, K. J. (1994). New empirical relationships among magnitude, rupture length, "
     "rupture width, rupture area, and surface displacement. BSSA 84, 974-1002.", None),
    ("Wilcock, W. S. D., et al. (2016). Seismic constraints on caldera dynamics from the 2015 Axial Seamount "
     "eruption. Science 354, 1395-1399.", "10.1126/science.aah5563"),
    ("Woessner, J., & Wiemer, S. (2005). Assessing the quality of earthquake catalogues: estimating the "
     "magnitude of completeness and its uncertainty. BSSA 95, 684-698.", None),
    ("Zaliapin, I., Gabrielov, A., Keilis-Borok, V., & Wong, H. (2008). Clustering analysis of seismicity and "
     "aftershock identification. PRL 101, 018501.", "10.1103/PhysRevLett.101.018501"),
    ("Zaliapin, I., & Ben-Zion, Y. (2013a). Earthquake clusters in southern California I: Identification and "
     "stability. JGR 118, 2847-2864.", "10.1002/jgrb.50179"),
    ("Zaliapin, I., & Ben-Zion, Y. (2013b). Earthquake clusters in southern California II: Classification and "
     "relation to physical properties of the crust. JGR 118, 2865-2877.", "10.1002/jgrb.50178"),
    ("Zaliapin, I., & Ben-Zion, Y. (2016). A global classification and characterization of earthquake clusters. "
     "GJI 207(1), 608-634.", "10.1093/gji/ggw300"),
]


def tb(x: pd.Series) -> pd.Series:
    """True where a CSV bool column is True (NaN and False are False)."""
    return x.astype(str).str.strip().str.lower().eq("true")


def km(lon, lat, lon0, lat0):
    return np.hypot((lon - lon0) * KM * np.cos(np.radians(lat0)), (lat - lat0) * KM)


def load():
    ev = pd.read_csv(Z.BIG / "events_zbz_v51.csv.gz", low_memory=False)
    ev["time"] = pd.to_datetime(ev.time, format="ISO8601", utc=True)
    ev["excluded"] = ev.excluded.fillna("")
    fam = pd.read_csv(OUT / "families_zbz_v51.csv", parse_dates=["start", "end"])
    d = dict(ev=ev, fam=fam)
    for k in ["regions_zbz", "thresholds_zbz", "completeness_b_df", "sensitivity_zbz", "shuffle_test",
              "validation_sequences", "volcano_families", "volcano_local_zbz", "setting_stats",
              "faults_v1_cluster_counts", "v4_families_status", "v51_families_status", "dbscan_clusters_v51",
              "dbscan_v4_status", "dbscan_v51_status", "faults_v1_daytime_surfaces_events",
              "v4_dissolved_members_fate"]:
        d[k] = pd.read_csv(OUT / f"{k}.csv")
    d["classes"] = pd.read_csv(OUT / "v4_vs_v51_event_classes.csv", index_col=0)
    d["ev4"] = pd.read_csv(V4 / "events_zbz.csv.gz")
    d["fam4"] = pd.read_csv(V4 / "families_zbz.csv")
    for k in ["regions_zbz", "thresholds_zbz", "completeness_b_df", "validation_sequences", "volcano_families",
              "volcano_local_zbz", "setting_stats", "shuffle_test"]:
        d[k + "_v4"] = pd.read_csv(V4 / f"{k}.csv")
    d["dbscan_v4"] = pd.read_csv(V4S / "v4_clusters.csv")
    return d


# ---------------------------------------------------------------- figures
def fig_tr(d, out):
    ev, thr = d["ev"], d["thresholds_zbz"].set_index("region")
    sh = np.load(OUT / "shuffle_eta_seed0.npz")
    from scipy.ndimage import gaussian_filter
    fig, axs = plt.subplots(3, 3, figsize=(FULL, FULL * 0.95), sharex=True, sharey=True)
    for ax, r in zip(axs.flat, REG_ORDER):
        e = ev[ev.in_analysis & (ev.region == r)].dropna(subset=["log_T", "log_R"])
        h, xe, ye = np.histogram2d(e.log_T, e.log_R, bins=[np.arange(-10, 1.01, 0.2), np.arange(-7, 4.01, 0.2)])
        h = h / h.sum() / 0.04
        ax.pcolormesh(xe, ye, np.ma.masked_equal(h.T, 0), cmap="Greys", shading="auto",
                      vmax=np.quantile(h[h > 0], 0.98))
        k = sh["region"] == r
        hs, _, _ = np.histogram2d(sh["log_T"][k], sh["log_R"][k], bins=[xe, ye])
        if hs.sum():
            hs = gaussian_filter(hs / hs.sum() / 0.04, 1.5)
            ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), hs.T,
                       levels=np.quantile(hs[hs > 0], [0.8, 0.95]), colors="#0072B2", linewidths=0.6)
        x = np.array([-10, 1])
        ax.plot(x, thr.loc[r, "log_eta0"] - x, color="#D55E00", lw=0.9)
        ax.set_title(f"{r}  {REGION_NAMES[r]}  (n={len(e)})", fontsize=7)
        ax.set_xlim(-10, 1)
        ax.set_ylim(-7, 4)
    for ax in axs[-1]:
        ax.set_xlabel("log$_{10}$ rescaled time T")
    for ax in axs[:, 0]:
        ax.set_ylabel("log$_{10}$ rescaled distance R")
    fig.text(0.01, 0.005, "Grey: events above Mc. Blue contours: one time-shuffled catalog. "
             "Red line: log T + log R = log eta0.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "fig_TR_density.png", dpi=170)
    plt.close(fig)


def fig_eta(d, out):
    ev, t = d["ev"], d["thresholds_zbz"].set_index("region")
    t4 = d["thresholds_zbz_v4"].set_index("region")
    sh = np.load(OUT / "shuffle_eta_seed0.npz")
    fig, axs = plt.subplots(3, 3, figsize=(FULL, FULL * 0.7), sharex=True)
    bins = np.arange(-11, 1.01, 0.2)
    e4 = d["ev4"]
    for ax, r in zip(axs.flat, REG_ORDER):
        le = ev[ev.in_analysis & (ev.region == r)].log_eta.dropna().values
        ax.hist(le, bins=bins, density=True, color="#BBBBBB", label="v5.1")
        le4 = e4[e4.in_analysis & (e4.region == r)].log_eta.dropna().values
        ax.hist(le4, bins=bins, density=True, histtype="step", color=INK, lw=0.6, label="v4")
        k = sh["region"] == r
        ax.hist(sh["log_eta"][k][np.isfinite(sh["log_eta"][k])], bins=bins, density=True, histtype="step",
                color="#0072B2", lw=0.8, label="v5.1 time-shuffled")
        g = t.loc[r]
        ax.axvline(g.log_eta0, color="#D55E00", lw=0.9, label="eta0 v5.1 (mixture)")
        ax.axvline(t4.loc[r, "log_eta0"], color=INK, lw=0.7, ls="--", label="eta0 v4 (mixture)")
        ax.axvline(g.log_eta0_shuf5, color="#D55E00", lw=0.7, ls=":", label="eta0 v5.1 (5% shuffled)")
        ax.set_title(f"{r}  D={g.ashman_D:.1f} (v4 {t4.loc[r, 'ashman_D']:.1f})", fontsize=7)
    axs[0, 0].legend(loc="upper left", fontsize=5)
    for ax in axs[-1]:
        ax.set_xlabel("log$_{10}$ eta")
    fig.tight_layout()
    fig.savefig(out / "fig_eta_hist.png", dpi=170)
    plt.close(fig)


def fig_map(d, out):
    import pygmt
    ev, fam = d["ev"], d["fam"]
    fig = pygmt.Figure()
    region = [-131.0, -120.5, 39.5, 51.5]
    pygmt.config(FONT_ANNOT_PRIMARY="7p", FONT_LABEL="7p", MAP_FRAME_TYPE="plain")
    fig.coast(region=region, projection="M12c", land="#EEEEEE", water="white", shorelines="0.25p,#888888",
              borders="1/0.25p,#BBBBBB", frame=["WSne", "af"])
    for key, _, (a, b, c, e) in REGIONS:
        fig.plot(x=[a, b, b, a, a], y=[c, c, e, e, c], pen="0.3p,#666666,--")
        fig.text(x=a + 0.05, y=e - 0.08, text=key, font="6p,Helvetica,#444444", justify="TL")
    bg = ev[ev.in_analysis & ev.background]
    fig.plot(x=bg.lon, y=bg.lat, style="c0.03c", fill="#AAAAAA")
    f = fam[fam.n >= 5].sort_values("n", ascending=False)
    for typ in ["mainshock-aftershock", "swarm"]:
        g = f[f.type == typ]
        fig.plot(x=g.lon_main, y=g.lat_main, size=0.08 + 0.09 * np.log10(g.n.values) ** 1.3 * 1.3,
                 style="cc", pen=f"0.5p,{TYPE_COLORS[typ]}", label=f"{typ} (n>=5)+S0.25c")
    fig.legend(position="JBL+jBL+o0.15c", box="+gwhite+p0.3p,#999999")
    fig.savefig(out / "fig_family_map.png", dpi=200)


def fig_compare(d, out):
    """v4 vs v5.1 locations with v4.1 magnitudes vs v5.1 main, per region."""
    r4 = d["regions_zbz_v4"].set_index("region").reindex(REG_ORDER)
    s = d["sensitivity_zbz"]
    vm = s[s.variant == "v41M"].set_index("region").reindex(REG_ORDER)
    r5 = d["regions_zbz"].set_index("region").reindex(REG_ORDER)
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.5))
    x = np.arange(len(REG_ORDER))
    w = 0.27
    cols = [("#BBBBBB", "v4 (v4 locations, v4 M)"), ("#88AACC", "v5.1 locations, v4.1 M"),
            ("#D55E00", "v5.1 (v5.1 locations, v5.1 M)")]
    for k, (tab, (c, lab)) in enumerate(zip([r4, vm, r5], cols)):
        axs[0].bar(x + (k - 1) * w, tab.pct_background, w, color=c, label=lab)
        axs[1].bar(x + (k - 1) * w, tab.n_families, w, color=c)
        axs[2].bar(x + (k - 1) * w, tab.n_swarm, w, color=c)
    for a, yl in zip(axs, ["% background (events above Mc)", "families", "swarm families"]):
        a.set_xticks(x, REG_ORDER)
        a.set_ylabel(yl)
    axs[0].legend(fontsize=5.5, loc="upper left")
    for a, l in zip(axs, "abc"):
        a.set_title(l, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_v4_compare.png", dpi=170)
    plt.close(fig)


def seq_panel(axm, axd, ev, fam, F, title):
    e = ev[ev.family_id == F].sort_values("time")
    f = fam.set_index("family_id").loc[F]
    t0 = e.time.min()
    td = (e.time - t0).dt.total_seconds() / 86400
    mainrow = e[e.event_id == f.mainshock_id].iloc[0]
    dist = km(e.lon.values, e.lat.values, mainrow.lon, mainrow.lat)
    col = np.where(e.in_analysis, TYPE_COLORS[f.type], "#BBBBBB")
    axm.scatter(td, e.M.fillna(-0.5), s=3, c=col, lw=0)
    axm.set_title(f"{title}\n{f.type}, n={int(f.n)} (+{len(e) - int(f.n)} <Mc), M1-M2={f.dM:.2f}", fontsize=5.8)
    axm.set_ylabel("M (missing: -0.5)", fontsize=6.5)
    axd.scatter(td, dist, s=3, c=col, lw=0)
    axd.set_ylabel("km from largest", fontsize=6.5)
    axd.set_xlabel("days from first event", fontsize=6.5, labelpad=1)
    if td.max() > 30:
        for a in (axm, axd):
            a.set_xscale("symlog", linthresh=1.0)


def fig_examples(d, out):
    ev, fam, val = d["ev"], d["fam"], d["validation_sequences"]
    pick = [(int(r.family_id), r.label) for _, r in val.iterrows() if r.family_id > 0]
    for r in ["GO", "PU", "ON", "BL"]:
        f = fam[(fam.region == r) & (fam.type == "swarm")].sort_values("n", ascending=False)
        if len(f) and int(f.family_id.iloc[0]) not in [p[0] for p in pick]:
            pick.append((int(f.family_id.iloc[0]), f"largest {r} swarm, {pd.Timestamp(f.start.iloc[0]):%Y-%m-%d}"))
    pick = pick[:12]
    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axs = plt.subplots(2 * nrow, ncol, figsize=(FULL, 2.6 * nrow + 0.3), gridspec_kw=dict(hspace=1.0, wspace=0.35))
    axs = np.atleast_2d(axs)
    for k, (F, lab) in enumerate(pick):
        i, j = divmod(k, ncol)
        seq_panel(axs[2 * i, j], axs[2 * i + 1, j], ev, fam, F, lab)
    for k in range(len(pick), nrow * ncol):
        i, j = divmod(k, ncol)
        axs[2 * i, j].axis("off")
        axs[2 * i + 1, j].axis("off")
    fig.savefig(out / "fig_examples.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def mig_frame(ev, F, df):
    """Tier A-C members of family F, time-sorted, with t (days) and r (km from the first-10 % centroid)."""
    e = ev[ev.family_id == F][["event_id"]].merge(df, on="event_id").sort_values("t_yr")
    e = e[e.tierABC]
    la, lo = e.lat.values, e.lon.values
    x = (lo - lo.mean()) * KM * np.cos(np.radians(la.mean()))
    y = (la - la.mean()) * KM
    k0 = max(3, len(e) // 10)
    r = np.hypot(x - x[:k0].mean(), y - y[:k0].mean())
    t = (e.t_yr.values - e.t_yr.values[0]) * YR
    return t, r, k0


def fig_migration(d, df, out):
    ev, fam = d["ev"], d["fam"]
    f = fam[(fam.mig_n >= 20)].copy()
    sel = pd.concat([f[tb(f.mig_migrating)],
                     f[f.type == "swarm"].sort_values("n_all", ascending=False).head(6)]).drop_duplicates("family_id")
    sel = sel.head(9)
    n = len(sel)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(FULL, 2.2 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, (_, row) in zip(axs, sel.iterrows()):
        t, r, k0 = mig_frame(ev, row.family_id, df)
        ax.scatter(np.maximum(t, 1e-4), r, s=3, c=np.where(np.arange(len(t)) < k0, "#BBBBBB", INK), lw=0)
        tt = np.logspace(-4, np.log10(max(t.max(), 1e-3)), 100)
        if np.isfinite(row.D90_m2s):
            Dk = row.D90_m2s * 86400 / 1e6
            ax.plot(tt, np.sqrt(4 * np.pi * Dk * tt), color="#D55E00", lw=0.8,
                    label=f"sqrt(4 pi D t), D90={row.D90_m2s:.3g} m$^2$/s")
        ax.axhspan(0, 2 * row.h_err_med_km, color="#0072B2", alpha=0.12, lw=0, label="2 x median h error")
        ax.set_xscale("log")
        ax.set_ylim(0, max(np.quantile(r, 0.99) * 1.2, 2 * row.h_err_med_km * 1.2))
        ax.set_title(f"F{int(row.family_id)} {row.region} {pd.Timestamp(row.start):%Y-%m-%d} {row.type}\n"
                     f"n={int(row.mig_n)} tier A-C, rho={row.mig_rho:.2f}, p_env={row.env_p:.2f}"
                     + (" migrating" if str(row.mig_migrating) == "True" else ""), fontsize=5.8)
        ax.set_xlabel("days from first event", fontsize=6.5)
        ax.set_ylabel("km from first 10%", fontsize=6.5)
        ax.legend(fontsize=4.8, loc="upper left")
    for ax in axs[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_migration.png", dpi=170)
    plt.close(fig)


def fig_depth(d, out):
    ev = d["ev"]
    e = ev[(ev.excluded == "") & tb(ev.depth_good)].copy()
    e["c"] = np.where(e.family_id > 0, e.family_type, np.where(e.linked, "linked_to_single", "background"))
    regs = [r for r in REG_ORDER if r not in ("EN",)]
    fig, axs = plt.subplots(1, len(regs), figsize=(FULL, 2.6), sharey=True)
    cats = ["background", "mainshock-aftershock", "swarm", "multiplet"]
    for ax, r in zip(axs, regs):
        g = e[e.region == r]
        data = [g.depth[g.c == c].values for c in cats]
        bp = ax.boxplot([x if len(x) else [np.nan] for x in data], widths=0.6, showfliers=False, patch_artist=True)
        for p_, c in zip(bp["boxes"], cats):
            p_.set_facecolor(TYPE_COLORS[c])
            p_.set_alpha(0.7)
        ax.set_xticks(range(1, 5), ["bg", "MA", "sw", "mu"], fontsize=6)
        ax.set_title(f"{r}\n" + "/".join(str(len(x)) for x in data), fontsize=6)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("depth (km bsl), depth_good events")
    fig.tight_layout()
    fig.savefig(out / "fig_depth.png", dpi=170)
    plt.close(fig)


def fig_topology(d, out):
    fam, st = d["fam"], d["setting_stats"]
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.6))
    f = fam[fam.n >= 5]
    for typ in ["mainshock-aftershock", "swarm"]:
        g = f[f.type == typ]
        axs[0].scatter(g.n, g.mean_leaf_depth, s=4, color=TYPE_COLORS[typ], label=typ, lw=0, alpha=0.7)
        axs[1].scatter(g.frac_before_largest, g.dM, s=4, color=TYPE_COLORS[typ], lw=0, alpha=0.7)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("family size n (above Mc)")
    axs[0].set_ylabel("mean leaf depth")
    axs[0].legend(fontsize=6)
    axs[1].set_xlabel("fraction of events before the largest")
    axs[1].set_ylabel("M1 - M2")
    x = np.arange(len(REG_ORDER))
    st4 = d["setting_stats_v4"].set_index("region").reindex(REG_ORDER)
    st5 = st.set_index("region").reindex(REG_ORDER)
    axs[2].bar(x - 0.2, st4.pct_swarm_of_ge5, 0.4, color="#BBBBBB", label="v4")
    axs[2].bar(x + 0.2, st5.pct_swarm_of_ge5, 0.4, color=TYPE_COLORS["swarm"], label="v5.1")
    for xi, n in zip(x, st5.n_fam_ge5):
        axs[2].text(xi + 0.2, 1, str(int(n)), ha="center", fontsize=5, color="white", rotation=90)
    axs[2].set_xticks(x, REG_ORDER)
    axs[2].set_ylabel("% swarms among families n>=5")
    axs[2].legend(fontsize=6)
    for a, l in zip(axs, "abc"):
        a.set_title(l, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_topology.png", dpi=170)
    plt.close(fig)


def fig_sens(d, out):
    sens = d["sensitivity_zbz"]
    vs = list(dict.fromkeys(sens.variant))
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.6))
    x = np.arange(len(REG_ORDER))
    cmap = plt.get_cmap("tab20")
    for k, v in enumerate(vs):
        s = sens[sens.variant == v].set_index("region").reindex(REG_ORDER)
        kw = dict(color=INK, s=18, marker="D", zorder=5) if v == "main" else dict(color=cmap(k % 20), s=8)
        axs[0].scatter(x + (k - len(vs) / 2) * 0.05, s.pct_background, label=v, **kw)
        axs[1].scatter(x + (k - len(vs) / 2) * 0.05, s.n_swarm / s.n_families.clip(lower=1) * 100, **kw)
    for a in axs:
        a.set_xticks(x, REG_ORDER)
    axs[0].set_ylabel("% background (above Mc)")
    axs[1].set_ylabel("% swarms among all families")
    axs[0].legend(fontsize=5, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig_sensitivity.png", dpi=170)
    plt.close(fig)


def fig_faults(d, out):
    pf = d["faults_v1_cluster_counts"]
    p = pf[pf.n_analysis >= 30].copy()
    p["pct_MA"] = 100 * p["n_mainshock-aftershock"] / p.n_analysis
    p["pct_mult"] = 100 * p.n_multiplet / p.n_analysis
    p = p.sort_values(["region", "pct_background"])
    fig, ax = plt.subplots(figsize=(FULL, 2.6))
    x = np.arange(len(p))
    b0 = np.zeros(len(p))
    for col, typ in [("pct_background", "background"), ("pct_MA", "mainshock-aftershock"),
                     ("pct_swarm", "swarm"), ("pct_mult", "multiplet")]:
        ax.bar(x, p[col], bottom=b0, color=TYPE_COLORS[typ], width=0.85, label=typ)
        b0 += p[col].values
    ax.set_xticks(x, p.surface_id.str.replace("v1_", ""), rotation=90, fontsize=4.5)
    ax.set_ylabel("% of events above Mc")
    ax.set_xlim(-0.6, len(p) - 0.4)
    ax.legend(ncol=4, fontsize=6, loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    fig.savefig(out / "fig_faults.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- HTML helpers
CSS = """
:root{--bg:#ffffff;--ink:#222;--muted:#666;--rule:#ddd;--head:#f4f4f4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1b1b;--ink:#e6e6e6;--muted:#aaa;
 --rule:#444;--head:#2a2a2a} img{background:#fff}}
:root[data-theme="dark"]{--bg:#1b1b1b;--ink:#e6e6e6;--muted:#aaa;--rule:#444;--head:#2a2a2a}
body{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
 max-width:1080px;margin:0 auto;padding:16px}
h1{font-size:1.6em;margin-bottom:.2em} h2{margin-top:1.8em;border-bottom:1px solid var(--rule)} h3{margin-top:1.3em}
table{border-collapse:collapse;font-size:12.5px;margin:.6em 0;display:block;overflow-x:auto;max-width:100%}
th,td{border:1px solid var(--rule);padding:3px 6px;text-align:right;white-space:nowrap}
th{background:var(--head)} td:first-child,th:first-child{text-align:left}
img{max-width:100%;height:auto;border:1px solid var(--rule)}
.muted{color:var(--muted);font-size:.92em} code{font-size:.92em}
"""


def tab(df, cols=None, rename=None, fmt=None, index=False):
    t = df[cols] if cols else df
    if rename:
        t = t.rename(columns=rename)
    return t.to_html(index=index, border=0, na_rep="", float_format=fmt or (lambda x: f"{x:.2f}")).replace(
        "&amp;", "&")


def img(path: Path, alt: str) -> str:
    b = base64.b64encode(path.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{b}' alt='{html.escape(alt)}'>"


# ---------------------------------------------------------------- report
def write_html(d, out):
    ev, fam = d["ev"], d["fam"]
    ev4, fam4 = d["ev4"], d["fam4"]
    a = ev[ev.in_analysis]
    a4 = ev4[ev4.in_analysis]
    nrow = len(ev)
    nx = ev.excluded.value_counts()
    n_used = int((ev.excluded == "").sum())
    used = ev[ev.excluded == ""]
    reg = d["regions_zbz"].set_index("region").reindex(REG_ORDER)
    reg4 = d["regions_zbz_v4"].set_index("region").reindex(REG_ORDER)
    thr = d["thresholds_zbz"].set_index("region").reindex(REG_ORDER)
    thr4 = d["thresholds_zbz_v4"].set_index("region").reindex(REG_ORDER)
    sens = d["sensitivity_zbz"]
    piv_bg = sens.pivot(index="region", columns="variant", values="pct_background").reindex(REG_ORDER)
    piv_ag = sens.pivot(index="region", columns="variant", values="agree_bg_vs_clustered_with_main").reindex(REG_ORDER)
    sh = d["shuffle_test"].groupby("region").mean(numeric_only=True).reindex(REG_ORDER)
    sh4 = d["shuffle_test_v4"].groupby("region").mean(numeric_only=True).reindex(REG_ORDER)
    comp, comp4 = d["completeness_b_df"].set_index("region"), d["completeness_b_df_v4"].set_index("region")
    val, val4 = d["validation_sequences"], d["validation_sequences_v4"]
    st, st4 = d["setting_stats"].set_index("region"), d["setting_stats_v4"].set_index("region")
    s4, s5 = d["v4_families_status"], d["v51_families_status"]
    db, db4s, db5s, db4 = d["dbscan_clusters_v51"], d["dbscan_v4_status"], d["dbscan_v51_status"], d["dbscan_v4"]
    pf = d["faults_v1_cluster_counts"]
    tf = fam.type.value_counts()
    tf4 = fam4.type.value_counts()
    nfam_ev = int((ev.family_id > 0).sum())
    nfam_ev4 = int((ev4.family_id > 0).sum())
    # per-region nn distance medians
    nn5 = a.groupby("region").nn_dist_km.median()
    nn4 = a4.groupby("region").nn_dist_km.median()
    # tier effect at Axial
    ax_tier = a[a.region == "AX"].groupby("tier").background.mean()
    nax_d = int(((a.region == "AX") & (a.tier == "D")).sum())
    nax = int((a.region == "AX").sum())
    fam10 = fam[fam.mig_n >= 10]
    nmig = int(tb(fam10.mig_migrating).sum())
    nres = int(tb(fam10.mig_resolvable).sum())
    sw10 = fam10[fam10.type == "swarm"]
    day = d["faults_v1_daytime_surfaces_events"].iloc[0]
    cls_t = d["classes"]

    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Catalog v5.1 clusters</title><style>{CSS}</style>"
         f"</head><body>"]
    h.append("<h1>Earthquake clusters in catalog v5.1 (nearest-neighbour method)</h1>")
    h.append(f"<p class='muted'>Cascadia catalog v5.1 (NonLinLoc in model v0.3.1), 2010-2016, with the v5.1 Route A "
             f"magnitudes (piece <i>mag</i>). Method of Zaliapin &amp; Ben-Zion (2013a, 2016), same code and "
             f"parameters as the catalog v4 analysis. Code: <code>workflow/06_analysis/v51_clusters_zbz.py</code>, "
             f"<code>v51_swarms.py</code>, <code>v51_clusters_report.py</code>, branch "
             f"<code>analysis/v51-clusters</code>, 2026-11-25. Private working product; not peer reviewed.</p>")

    # ---------------- summary
    h.append("<h2>Summary</h2>")
    h.append(f"""<p>Of the {nrow:,} catalog rows, {n_used:,} are located earthquakes and go into the analysis. Left out:
{nx.get('no_hypocentre', 0):,} rows without a hypocentre (loc_source none) and {nx.get('not_earthquake', 0):,} events
flagged as explosions, quarry blasts or other non-earthquakes. Above the regional completeness magnitude Mc there are
{len(a):,} events (v4: {len(a4):,}); {100 * a.background.mean():.1f}% of them are background singles (v4:
{100 * (a4.role == 'single').mean():.1f}%). The rest fall in {len(fam):,} families (v4: {len(fam4):,}):
{tf.get('mainshock-aftershock', 0)} mainshock-aftershock families (v4: {tf4.get('mainshock-aftershock', 0)}),
{tf.get('swarm', 0)} swarms (v4: {tf4.get('swarm', 0)}) and {tf.get('multiplet', 0)} doublets or small multiplets
(v4: {tf4.get('multiplet', 0)}). With the events below Mc attached, {nfam_ev:,} events
({100 * nfam_ev / nrow:.1f}% of all rows) are in a family (v4: {nfam_ev4:,}).</p>
<ol>
<li><b>The v4 picture holds.</b> All nine regions are bimodal in the (T, R) plane and lose it in time-shuffled
catalogs (Ashman D {reg.ashman_D.min():.1f}-{reg.ashman_D.max():.1f}, shuffled
{sh.ashman_D.min():.1f}-{sh.ashman_D.max():.1f}). Background shares order by setting as in v4: about 50-62% at
Axial, Endeavour-Explorer, the Blanco and the Gorda plate, 58% at the Mendocino triple junction, 70-84% onshore, in
the Puget Lowland and offshore Washington and Oregon. Swarms make
{st.pct_swarm_of_ge5[['AX', 'BL', 'ON']].min():.0f}-{st.pct_swarm_of_ge5[['AX', 'BL', 'ON']].max():.0f}% of
the families with 5 or more events at Axial, the Blanco and in ON, against {st.pct_swarm_of_ge5['MT']:.0f}% at the
Mendocino triple junction, where trees are still the shallowest (median leaf depth
{st.median_leaf_depth_ge5['MT']:.2f}; v4 {st4.median_leaf_depth_ge5['MT']:.2f}).</li>
<li><b>Families are stable between versions.</b> Matching families by shared event IDs,
{(s4.status == 'kept').sum():,} of the {len(s4):,} v4 families are kept one-to-one,
{(s4.status == 'split').sum()} split, {(s4.status == 'merged').sum()} merged with another v4 family,
{(s4.status == 'grown').sum()} grew by more than a factor of two and {(s4.status == 'dissolved').sum()} dissolved
(fewer than half of their members are in any v5.1 family; {(s4[s4.status == 'dissolved'].n4_all <= 4).mean() * 100:.0f}%
of these are doublets or triplets). On the v5.1 side {(s5.status == 'new').sum()} families are new
({(s5[s5.status == 'new'].n5_all <= 4).mean() * 100:.0f}% of them with 2-4 members). Of the kept families,
{_kept_same_type(s4):.0f}% keep their type.</li>
<li><b>The largest change is at Axial,</b> where the background share rose from {reg4.pct_background['AX']:.1f}% to
{reg.pct_background['AX']:.1f}%. The change comes from the locations, not the magnitudes (v5.1 locations with the
v4.1 magnitudes give {piv_bg.loc['AX', 'v41M']:.1f}%). The v5.1 Axial epicentres are tighter (median
nearest-neighbour distance {nn5['AX']:.2f} km against {nn4['AX']:.2f} km), so every eta drops, but the mixture
threshold dropped further ({thr.log_eta0['AX']:.2f} against {thr4.log_eta0['AX']:.2f}) and now cuts more links.
With the conservative threshold (5% of shuffled links) the two versions agree
({piv_bg.loc['AX', 'eta0_shuf5']:.1f}% against 82.2%). The 2015 eruption is still one chain-like swarm, starting
2015-04-24 05:13 UTC, with {int(val[val.label.str.contains('Axial')].n.iloc[0])} events above Mc
(v4: 632).</li>
<li><b>Location quality matters most at Axial.</b> {nax_d:,} of the {nax:,} Axial events above Mc are tier D (most
outside the caldera network: median nearest station 4.3 km, azimuthal gap 327 deg); {100 * ax_tier.get('D', np.nan):.0f}%
of them are background, against {100 * ax_tier.get('A', np.nan):.0f}% of tier A events. Taking tier D out of the
analysis set lowers the Axial background share to {piv_bg.loc['AX', 'tierABC']:.1f}%. Elsewhere the tierABC variant
changes the background/clustered label of at most {100 * (1 - piv_ag.drop(index='AX')['tierABC'].min()):.1f}% of
events.</li>
<li><b>Migration is rarely resolved.</b> Of {len(fam10)} families with 10 or more tier A-C members, {nres} have a
spread (90% of distances from the first 10% of events) larger than twice their median horizontal error, and {nmig}
migrate by our test (Spearman rho &ge; 0.5, p &lt; 0.01, positive Theil-Sen slope, resolvable). Only
{int(tb(sw10.mig_resolvable).sum())} of the {len(sw10)} swarms of this size are resolvable, so the
sqrt(t) diffusion fits (D90) mostly describe location scatter, not a triggering front. The 90% distance envelope
grows with time as t^p with median p = {sw10.env_p.median():.2f} for swarms, i.e. no growth.</li>
<li><b>Swarm scan (DBSCAN).</b> {len(db)} short space-time clusters (v4: {len(db4)}); {(db4s.status == 'kept').sum()}
of the v4 clusters are kept, {(db4s.status == 'dissolved').sum()} dissolved, mostly into clusters now longer than 30
days or smaller than 15 events.</li>
</ol>""")

    # ---------------- what changed
    h.append("<h2>What changed from v4, and why</h2>")
    t = pd.DataFrame(dict(
        region=REG_ORDER, Mc_v4=comp4.Mc.reindex(REG_ORDER).values, Mc_v51=comp.Mc.reindex(REG_ORDER).values,
        b_v4=comp4.b.reindex(REG_ORDER).values, b_v51=comp.b.reindex(REG_ORDER).values,
        n_v4=reg4.n_analysis.values, n_v51=reg.n_analysis.values,
        nn_km_v4=nn4.reindex(REG_ORDER).values, nn_km_v51=nn5.reindex(REG_ORDER).values,
        eta0_v4=thr4.log_eta0.values, eta0_v51=thr.log_eta0.values,
        bg_v4=reg4.pct_background.values, bg_loc=piv_bg["v41M"].values, bg_v51=reg.pct_background.values,
        fam_v4=reg4.n_families.values, fam_v51=reg.n_families.values,
        sw_v4=reg4.n_swarm.values, sw_v51=reg.n_swarm.values))
    h.append(tab(t, rename={"nn_km_v4": "NN dist v4 (km)", "nn_km_v51": "NN dist v5.1 (km)",
                            "eta0_v4": "log eta0 v4", "eta0_v51": "log eta0 v5.1", "bg_v4": "% bg v4",
                            "bg_loc": "% bg v5.1 loc, v4.1 M", "bg_v51": "% bg v5.1", "fam_v4": "families v4",
                            "fam_v51": "families v5.1", "sw_v4": "swarms v4", "sw_v51": "swarms v5.1",
                            "n_v4": "events &ge; Mc v4", "n_v51": "events &ge; Mc v5.1"}))
    h.append(img(out / "fig_v4_compare.png", "v4 vs v5.1 by region"))
    h.append("<p class='muted'>NN dist: median distance from each event above Mc to its nearest-neighbour parent. "
             "The middle bars and the '% bg v5.1 loc, v4.1 M' column are the variant <code>v41M</code>: v5.1 "
             "locations with the catalog v4.1 magnitudes, its own Mc and b. v4 used catalog v4 and the v4 Route A "
             "magnitudes.</p>")
    h.append(f"""<p><b>Three things changed between the runs: the events, the locations and the magnitudes.</b></p>
<ul>
<li><b>Events.</b> v5.1 leaves out {nx.get('not_earthquake', 0):,} flagged non-earthquakes and
{nx.get('no_hypocentre', 0)} rows without a hypocentre; v4 kept all 172,340. Of the flagged events,
{int(cls_t.loc['family', 'excluded_not_earthquake'])} were in v4 families and
{int(cls_t.loc['background', 'excluded_not_earthquake']):,} were v4 background. Catalog v4.1 also removed event
172113 and added 900001-900003 (the 2016-12-08 Mw 6.6 event and two 2016-01-30 M 5 events).</li>
<li><b>Magnitudes.</b> The v5.1 ML is higher by a median 0.05 (offshore 0.06; Axial -0.11). With the v4 Mc rule
(maximum curvature + 0.2, largest yearly value) Mc changed by -0.10 to +0.10 in eight boxes and by -0.20 offshore
Washington; the analysis set grew from {len(a4):,} to {len(a):,} events. Comparing the v41M variant with the main run
isolates the magnitude effect: it moves the background share by
{np.abs(reg.pct_background - piv_bg['v41M']).min():.1f}-{np.abs(reg.pct_background - piv_bg['v41M']).max():.1f}
points, and the labels of events in both analysis sets agree for
{100 * piv_ag['v41M'].min():.1f}-{100 * piv_ag['v41M'].max():.1f}%.</li>
<li><b>Locations.</b> Comparing v4 with the v41M variant isolates the location effect (and the event list). It is
small everywhere except at Axial (+{piv_bg.loc['AX', 'v41M'] - reg4.pct_background['AX']:.1f} points) and offshore
Oregon (+{piv_bg.loc['OR', 'v41M'] - reg4.pct_background['OR']:.1f} points, on {int(reg.n_analysis['OR'])} events).
At Axial, the mixture threshold moved because the clustered and background modes of log eta moved by different
amounts ({thr4.mu1['AX']:.2f} to {thr.mu1['AX']:.2f} and {thr4.mu2['AX']:.2f} to {thr.mu2['AX']:.2f}) and the
clustered weight dropped from {thr4.w1['AX']:.2f} to {thr.w1['AX']:.2f}. The shuffled false-link rate at Axial
dropped from 34% to {100 * sh.frac_linked['AX']:.0f}%: the v5.1 split is cleaner, but at a stricter
threshold.</li>
</ul>""")
    # event classes
    h.append("<h3>Event labels, v4 against v5.1 (all rows)</h3>")
    h.append(tab(cls_t.reset_index(), fmt=lambda x: f"{x:,.0f}"))
    h.append(f"""<p class='muted'>Rows: v4 class; columns: v5.1 class. 'family': in a family (above or below Mc);
'linked_to_single': a below-Mc event linked to a background single; 'background': no link. {int(cls_t.loc['background',
'family']):,} v4 background events are now in families and {int(cls_t.loc['family', 'background']):,} v4 family
members are now background.</p>""")
    tr = pd.read_csv(OUT / "v4_vs_v51_event_classes_analysis_by_region.csv")
    h.append("<p>Events above Mc in both versions, background/family label v4 -&gt; v5.1, by region:</p>"
             + tab(tr, fmt=lambda x: f"{x:.0f}"))
    # family matching
    h.append("<h3>Families kept, split, merged, new</h3>")
    h.append("""<p>Families are matched by shared event IDs (all members, including those attached below Mc). A v4
family is <i>kept</i> when one v5.1 family holds at least half of its members and at least half of that family's
members come from it; <i>split</i> when two or more v5.1 families each hold at least 2 and at least 20% of its
members; <i>merged</i> when its main v5.1 family also takes at least 20% of its members from another v4 family;
<i>grown</i> when its main v5.1 family holds at least half of its members but is more than twice its size;
<i>dissolved</i> when fewer than half of its members are in any v5.1 family. A v5.1 family is <i>new</i> when no v4
family contributes 2 members and 20% of its size.</p>""")
    s4["size"] = pd.cut(s4.n4_all, [1, 4, 9, 49, 1e9], labels=["2-4", "5-9", "10-49", "50+"])
    s5["size"] = pd.cut(s5.n5_all, [1, 4, 9, 49, 1e9], labels=["2-4", "5-9", "10-49", "50+"])
    h.append("<p>v4 families by size (all members) and status:</p>"
             + tab(pd.crosstab(s4["size"], s4.status, margins=True).reset_index(), fmt=lambda x: f"{x:.0f}"))
    h.append("<p>v5.1 families by size and status:</p>"
             + tab(pd.crosstab(s5["size"], s5.status, margins=True).reset_index(), fmt=lambda x: f"{x:.0f}"))
    h.append("<p>v4 families by region (box of the v4 largest event) and status:</p>"
             + tab(pd.crosstab(s4.region, s4.status, margins=True).reset_index(), fmt=lambda x: f"{x:.0f}"))
    k = s4[s4.status == "kept"]
    h.append("<p>Type of kept families, v4 (rows) against v5.1 (columns):</p>"
             + tab(pd.crosstab(k.type, k.type_v51_main, margins=True).reset_index(), fmt=lambda x: f"{x:.0f}"))
    fate = d["v4_dissolved_members_fate"]
    h.append(f"""<p>Members of dissolved v4 families are now: {', '.join(f"{r.cls5} {r.n:,}" for r in
    fate.itertuples())}. Dissolved families are mostly small (doublets and triplets near the threshold); their
members moved by a few km and their links crossed eta0. Large families that split or merged are listed below.</p>""")
    big = s4[(s4.n4_all >= 50) & (s4.status != "kept")].sort_values("n4_all", ascending=False)
    if len(big):
        h.append(tab(big, ["f4", "region", "type", "n4_all", "start", "status", "f5_main", "shared_main", "n5_main",
                           "type_v51_main", "n_v51_parts"],
                     rename={"f4": "v4 family", "n4_all": "v4 members", "f5_main": "main v5.1 family",
                             "shared_main": "shared", "n5_main": "v5.1 members", "type_v51_main": "v5.1 type",
                             "n_v51_parts": "v5.1 parts"}, fmt=lambda x: f"{x:.0f}"))

    # ---------------- data and choices
    h.append("<h2>Data and choices</h2>")
    nfl = int(tb(used.floor_limited).sum())
    nfar = int(tb(used.far_only).sum())
    ndp = int(tb(used.deep_pass).sum())
    ndg = int(tb(used.depth_good).sum())
    h.append(f"""<p><b>Events.</b> Catalog v5.1 ({nrow:,} rows). Used: {n_used:,} located earthquakes (171,717 NonLinLoc
locations and the 2 ComCat hypocentres of the v4.1 additions, minus {nx.get('not_earthquake', 0):,} events with
<code>is_earthquake</code> False: anss_explosion 1,665, quarry_blast_likely 1,572, anss_quarry_blast 74, both
flags 1, anss_other_event 1). The {nx.get('no_hypocentre', 0)} rows without a hypocentre are in the event table with
role <code>excluded</code> and are not used anywhere.</p>
<p><b>Poorly constrained depths.</b> floor_limited ({nfl:,} used events), far_only ({nfar}) and deep_pass ({ndp})
events enter through their epicentres only: the main run uses epicentral distance, as in v4. Their depths are not
used: depth ranges, depth migration and the 3-D variant <code>d3q</code> use only <i>depth_good</i> events (tier
A-C, depth_quality ok, not floor_limited, far_only or deep_pass; {ndg:,} events). depth_quality
'unconstrained' (ridge regions, Blanco outside the 2012-13 OBS deployment) and 'biased_deep' (Gorda) are also left
out of depth statistics. Tier D events stay in the analysis set (as all events did in v4); variant
<code>tierABC</code> takes them out of it and attaches them afterwards.</p>
<p><b>Magnitudes.</b> The v5.1 Route A preferred magnitude M (ML calibrated to ComCat, tensor Mw above 4.5,
Mw_cal). {int(used.M.isna().sum()):,} used events have no M; they cannot be parents and are attached after the
clustering (variant <code>noM_floor</code> gives them M = Mc).</p>
<p><b>Regions, completeness, b.</b> The nine v4 boxes, applied to the v5.1 epicentres. Mc with the v4 rule:
maximum curvature + 0.2 for each region and year with at least 200 events, the largest yearly value for the whole
period. b by Aki (1965) above Mc. The mag piece's Mc_best (whole period, Woessner &amp; Wiemer 2005 best rule) is
lower by {(comp.Mc - comp.Mc_best).min():.2f}-{(comp.Mc - comp.Mc_best).max():.2f} because it does not take the worst
year; it is variant <code>mc_best</code>.</p>""")
    ct = comp.reset_index()[["region", "n", "n_M", "Mc_all", "Mc_bstab", "Mc", "Mc_best", "years_used", "n_above",
                             "b", "b_err", "df_est"]]
    ct["Mc_v4"] = ct.region.map(comp4.Mc)
    ct["b_v4"] = ct.region.map(comp4.b)
    h.append(tab(ct, rename={"n": "events", "n_M": "with M", "Mc_all": "Mc (all years)", "Mc_bstab": "Mc (b-stab.)",
                             "Mc": "Mc used", "Mc_best": "Mc_best (mag piece)", "years_used": "years setting Mc",
                             "n_above": "events &ge; Mc", "b_err": "b error", "df_est": "D2 epicentres"}))
    h.append("""<p><b>Proximity, threshold, families, classification.</b> Unchanged from v4: eta = t r^1.6
10^(-b m) (Baiesi &amp; Paczuski 2004; Zaliapin et al. 2008) with t in years and r epicentral in km floored at 0.5 km, q = 0.5; parents in another region only within
the rupture-length cap r &le; max(10 km, 3 L(m)) (Wells &amp; Coppersmith 1994); threshold eta0 where the two
components of a Gaussian mixture on log eta cross, per region; links below eta0 and within the cap form trees
(families); events below Mc or without M are attached to the nearest above-Mc parent if that link passes the same
tests. Families of 2-4 events above Mc are multiplets when M1-M2 &lt; 0.3, otherwise mainshock-aftershock;
families of 5 or more are swarms when at least two of S1 (M1-M2 &lt; 0.5), S2 (&ge; 20% of events before the
largest) and S3 (mean leaf depth &ge; 2 and fewer than half of the family direct offspring of the largest event)
hold (topology criteria of Zaliapin &amp; Ben-Zion 2013b, 2016; burst criteria of Vidale &amp; Shearer 2006).</p>
<p><b>Per-family migration and diffusion (new).</b> Computed on the tier A-C members (above and below Mc) of
families with at least 10 of them. Origin: centroid of the first 10% of events (at least 3); r = epicentral
distance from it, t = time since the first event. Spearman rho and Theil-Sen slope of r against t (radial speed);
Theil-Sen slopes of east and north position against t give a velocity vector and its azimuth; depth migration
from depth_good members. Diffusion: D90 = 90th percentile of r&sup2;/(4 pi t) over the events after the first 10%
(the sqrt(4 pi D t) front of Shapiro et al. 1997 that encloses 90% of them). Envelope exponent p: slope of log
r90 against log t over up to 8 log-time bins with at least 3 events each (p near 0.5 for diffusion, near 1 for
constant-speed migration, near 0 for no growth). <i>Resolvable</i>: r90 at least twice the median 68% horizontal
semi-axis (h_max) of the members. <i>Migrating</i>: rho &ge; 0.5, p &lt; 0.01, positive radial slope and
resolvable.</p>""")

    # ---------------- bimodality
    h.append("<h2>Bimodality and threshold</h2>")
    h.append(img(out / "fig_TR_density.png", "T-R density per region"))
    h.append("<p class='muted'>Density of rescaled time T and distance R of each event's nearest neighbour, events "
             "above Mc. Blue contours: one time-shuffled catalog (80% and 95% density levels). Red: threshold.</p>")
    h.append(img(out / "fig_eta_hist.png", "log eta histograms"))
    tt = thr.reset_index()[["region", "w1", "mu1", "sd1", "w2", "mu2", "sd2", "log_eta0", "log_eta0_kde",
                            "log_eta0_shuf5", "ashman_D", "n_below_eta0", "n_cut_by_cap"]]
    tt["log_eta0_v4"] = tt.region.map(thr4.log_eta0)
    tt["D_shuffled"] = tt.region.map(sh.ashman_D)
    tt["pct_linked_shuffled"] = tt.region.map(100 * sh.frac_linked)
    tt["pct_linked_shuffled_v4"] = tt.region.map(100 * sh4.frac_linked)
    h.append(tab(tt, rename={"w1": "w clust.", "mu1": "mu clust.", "sd1": "sd clust.", "w2": "w bg", "mu2": "mu bg",
                             "sd2": "sd bg", "log_eta0": "log eta0", "log_eta0_kde": "KDE valley",
                             "log_eta0_shuf5": "5% shuffled", "ashman_D": "D", "n_below_eta0": "links &lt; eta0",
                             "n_cut_by_cap": "cut by cap", "log_eta0_v4": "log eta0 v4",
                             "pct_linked_shuffled": "% linked, shuffled",
                             "pct_linked_shuffled_v4": "% linked, shuffled v4"}))
    h.append(f"""<p>The clustered mode is clear in EN, BL, GO, MT and ON (D {reg.ashman_D[['EN', 'BL', 'GO', 'MT',
'ON']].min():.1f}-{reg.ashman_D[['EN', 'BL', 'GO', 'MT', 'ON']].max():.1f}) and weaker offshore Washington, offshore
Oregon and in Puget (D {reg.ashman_D[['WA', 'OR', 'PU']].min():.1f}-{reg.ashman_D[['WA', 'OR', 'PU']].max():.1f}),
where background dominates. In three time-shuffled catalogs the share of events that pass the real threshold and cap
is {100 * sh.frac_linked.drop('AX').min():.0f}-{100 * sh.frac_linked.drop('AX').max():.0f}% outside Axial and
{100 * sh.frac_linked['AX']:.0f}% at Axial (v4: 34%), against real clustered shares of
{100 - reg.pct_background.max():.0f}-{100 - reg.pct_background.min():.0f}%. Axial remains the weakest test: its
events above Mc sit in a 5 km caldera and most occur during the months of rising rate before the April 2015
eruption (Wilcock et al. 2016), which shuffling keeps.</p>""")

    # ---------------- validation
    h.append("<h2>Validation: known sequences</h2>")
    v = val.merge(val4[["event_id", "type", "n", "n_all", "M2", "duration90_days", "mean_leaf_depth"]].rename(
        columns=lambda c: c if c == "event_id" else c + "_v4"), on="event_id", how="left")
    h.append(tab(v, ["label", "expected", "region", "M", "type", "type_v4", "n", "n_v4", "n_all", "n_all_v4", "M2",
                     "M2_v4", "duration90_days", "duration90_days_v4", "extent_km", "mean_leaf_depth",
                     "mean_leaf_depth_v4"],
                 rename={"n": "n &ge; Mc", "n_v4": "v4", "n_all": "n all", "n_all_v4": "v4 ", "M2_v4": "M2 v4",
                         "duration90_days": "t90 (d)", "duration90_days_v4": "t90 v4", "extent_km": "extent (km)",
                         "mean_leaf_depth": "leaf depth", "mean_leaf_depth_v4": "leaf depth v4"}))
    h.append("""<p>All six large tectonic mainshocks are again mainshock-aftershock families, and the two Blanco pairs and
triplets are again swarms with M1-M2 near 0.1. Two changes come from single links near the threshold. (1) The
2010-02-04 Mw 5.9 event, 44 km south of the 2010-01-10 Mw 6.5 Gorda event in v4, is 54 km from it in v5.1; its
link (log eta -5.27) now misses the MT threshold (-5.39) and it heads its own family, so the 2010 family has M2 4.95
instead of 5.88 and 579 instead of 824 events above Mc. (2) The 2013-08-04 M 5.5 event near the epicentre of the
2014-04-24 Mw 6.5 Vancouver Island event (1 km apart) is now linked to it as a foreshock 8.7 months earlier,
merging the two v4
families: t90 grows from 24 to 280 days and the mean leaf depth from 1.05 to 2.1. The 2016-12-08 Mw 6.6 event
(added in v4.1) has only 23 days of catalog after it and 4 events above Mc in its family.</p>""")
    h.append(img(out / "fig_examples.png", "example sequences"))
    h.append("<p class='muted'>Magnitude and distance from the largest event against time. Coloured: events above Mc; "
             "grey: attached events below Mc (missing M at -0.5).</p>")

    # ---------------- families and swarms
    h.append("<h2>Families: depth, migration, swarms</h2>")
    h.append(img(out / "fig_family_map.png", "map of families"))
    h.append("<p class='muted'>Families with 5 or more events above Mc at their largest event, sized by log n; grey "
             "dots: background events above Mc. Dashed boxes: regions.</p>")
    h.append(img(out / "fig_depth.png", "depth by class"))
    dg = used[tb(used.depth_good)].copy()
    dg["c"] = np.where(dg.family_id > 0, dg.family_type, np.where(dg.linked, "linked_to_single", "background"))
    dt = dg.groupby(["region", "c"]).depth.median().unstack().reindex(REG_ORDER)
    h.append("<p class='muted'>Depth of depth_good events by class (bg background, MA mainshock-aftershock, sw "
             "swarm, mu multiplet; all events, above and below Mc); counts above each panel. EN has too few "
             "depth_good events.</p>")
    h.append("<p>Median depth (km below sea level) of depth_good events by class:</p>"
             + tab(dt.reset_index(), fmt=lambda x: f"{x:.1f}"))
    pu = used[(used.region == "PU") & used.in_analysis & tb(used.depth_good)]
    pu_sh = pu[pu.depth < 30].background.mean() * 100
    pu_dp = pu[pu.depth >= 30].background.mean() * 100
    h.append(f"""<p>Swarm members are shallower than mainshock-aftershock members in every region with enough
depth_good events except Axial, where everything is within the upper 4 km: Gorda plate {dt.loc['GO', 'swarm']:.1f}
against {dt.loc['GO', 'mainshock-aftershock']:.1f} km (ridge and Escanaba Trough swarms against plate-interior
sequences; most Gorda depths are 'biased_deep' and left out), ON {dt.loc['ON', 'swarm']:.1f} against
{dt.loc['ON', 'mainshock-aftershock']:.1f} km (volcanic and upper-crustal swarms), Mendocino
{dt.loc['MT', 'swarm']:.1f} against {dt.loc['MT', 'mainshock-aftershock']:.1f} km. In the Puget box,
{pu_dp:.0f}% of the depth_good events above Mc at 30 km or deeper (intraslab) are background, against
{pu_sh:.0f}% above 30 km.</p>""")
    h.append("<h3>Migration and diffusion</h3>")
    h.append(img(out / "fig_migration.png", "migration examples"))
    h.append("<p class='muted'>Distance from the centroid of the first 10% of events (grey) against time for the "
             "migrating families and the largest swarms. Red: the sqrt(4 pi D90 t) front; blue band: twice the "
             "median horizontal error, below which distances are not resolved.</p>")
    mig = fam[tb(fam.mig_migrating)]
    h.append(f"<p>Families that migrate by our test ({len(mig)}):</p>")
    h.append(tab(mig, ["family_id", "region", "region_v51", "type", "start", "n", "n_all", "mig_n", "Mmax",
                       "duration_all_days", "mig_rho", "mig_v_radial_km_day", "mig_az", "mig_v_dir_km_day",
                       "mig_vz_km_day", "D90_m2s", "env_p", "h_err_med_km", "mig_r90_km", "mig_style", "v4_status"],
                 rename={"mig_n": "n A-C", "duration_all_days": "days", "mig_rho": "rho",
                         "mig_v_radial_km_day": "radial km/d", "mig_az": "azimuth", "mig_v_dir_km_day": "speed km/d",
                         "mig_vz_km_day": "vz km/d", "h_err_med_km": "h err", "mig_r90_km": "r90 km",
                         "mig_style": "style"}))
    h.append(f"""<p>The 2012-02-04 M 5.6 Explorer sequence (EN) is the clearest case, as in v4: its aftershock zone grows
north-northeast at about {mig[mig.region == 'EN'].mig_v_radial_km_day.max():.1f} km/day over four months, but its
envelope exponent p = {mig[mig.region == 'EN'].env_p.max():.2f} is far below 0.5, so the growth is fast early and
slows; neither a constant speed nor a sqrt(t) front describes it. The 2013-11-13 Gorda swarm moves about 40 km/day
over 9 hours (16 tier A-C events), a dike-like speed, but on few events. The v4 example of the 2013-05-20 Escanaba
Trough swarm (10 km/day in v4) is family {int(fam[(fam.region == 'GO') & (fam.start.dt.strftime('%Y-%m-%d') ==
'2013-05-20')].family_id.iloc[0]) if ((fam.region == 'GO') & (fam.start.dt.strftime('%Y-%m-%d') ==
'2013-05-20')).any() else -1} in v5.1 with rho 0.46: below our cut.</p>
<p>Most swarms are compact relative to their location errors: the median swarm with 10 or more tier A-C members has a
90% radius of {sw10.mig_r90_km.median():.1f} km and a median horizontal error of {sw10.h_err_med_km.median():.1f} km.
D90 values for those swarms ({sw10.D90_m2s.quantile(0.25):.0f}-{sw10.D90_m2s.quantile(0.75):.0f} m&sup2;/s
interquartile) are therefore upper bounds set by scatter divided by short durations, not hydraulic
diffusivities. Testing fluid-diffusion triggering needs relative relocation (double difference) of the swarms.</p>""")
    h.append("<h3>Swarms with 20 or more events</h3>")
    sw = fam[(fam.type == "swarm") & (fam.n_all >= 20)].sort_values("n_all", ascending=False).copy()
    sw["M_range"] = sw.Mmin_all.round(1).astype(str) + " - " + sw.Mmax.round(1).astype(str)
    sw["depth_q10_q90"] = np.where(sw.n_depth_good >= 3, sw.depth_good_q10.round(1).astype(str) + " - "
                                   + sw.depth_good_q90.round(1).astype(str), "")
    sw["D90"] = np.where(tb(sw.mig_resolvable), sw.D90_m2s.round(1).astype(str), "")
    h.append(tab(sw, ["family_id", "region", "region_v51", "start", "duration90_days", "duration_all_days", "n",
                      "n_all", "M_range", "n_depth_good", "depth_q10_q90", "extent_km", "axis_az", "mig_rho",
                      "mig_v_radial_km_day", "mig_az", "env_p", "mig_resolvable", "D90", "frac_on_faults",
                      "main_fault", "v4_status"],
                 rename={"duration90_days": "t90 &ge;Mc (d)", "duration_all_days": "duration all (d)", "n": "n &ge; Mc",
                         "n_depth_good": "n depth_good", "depth_q10_q90": "depth 10-90% (km)", "extent_km": "extent km",
                         "axis_az": "long axis az", "mig_rho": "rho", "mig_v_radial_km_day": "radial km/d",
                         "mig_az": "mig. az", "mig_resolvable": "resolvable", "D90": "D90 m2/s (resolvable only)",
                         "frac_on_faults": "on faults", "main_fault": "main surface"}))
    h.append("<p class='muted'>All families, with every column, are in <code>families_zbz_v51.csv</code>.</p>")

    # ---------------- DBSCAN
    h.append("<h2>Short space-time clusters (DBSCAN swarm scan)</h2>")
    h.append(f"""<p>The scan of <code>v4_swarms.py</code> with its parameters (3 km link in x, y and time at 2 km/day,
core events with 5 links, clusters of 15+ events lasting at most 30 days) on the {n_used:,} located earthquakes, all
magnitudes: {len(db)} clusters with {int(db.n.sum()):,} events (v4: {len(db4)} with {int(db4.n.sum()):,}),
{int((db.swarm_like == True).sum())} swarm-like by the scan's rule (v4: {int((db4.swarm_like == True).sum())}).</p>""")  # noqa: E712
    dbr = db.groupby("region").agg(clusters=("n", "size"), events=("n", "sum"),
                                   swarm_like=("swarm_like", lambda s: int((s == True).sum())),  # noqa: E712
                                   events_above_Mc=("n_above_Mc", "sum"),
                                   median_frac_in_NN_families=("frac_in_zbz_families", "median")).reset_index()
    dbr["clusters_v4"] = dbr.region.map(db4.groupby("region").size())
    h.append(tab(dbr, fmt=lambda x: f"{x:.2f}"))
    h.append("<p>v4 clusters by status:</p>" + tab(pd.crosstab(db4s.region, db4s.status, margins=True).reset_index(),
                                                     fmt=lambda x: f"{x:.0f}"))
    fates = db4s[db4s.status == "dissolved"].dissolved_member_fate.str.split(";").explode()
    fz = fates.str.rsplit(":", n=1).str[0].value_counts()
    h.append(f"""<p>v5.1 clusters: {', '.join(f'{k} {v}' for k, v in db5s.status.value_counts().items())}. The
{(db4s.status == 'dissolved').sum()} dissolved v4 clusters: their events are now mostly in clusters of fewer than 15
events or in no cluster, and four large 2013 Blanco clusters (145-263 events) are inside a v5.1 cluster that lasts
more than 30 days (the tighter v5.1 Blanco epicentres chain neighbouring bursts together, and the scan drops chains
longer than 30 days). Number of dissolved clusters with members in each fate: {html.escape(', '.join(f'{k} {v}' for k, v
in fz.items()))}.</p>""")
    h.append("<p>DBSCAN swarm-like flag against the type of the nearest-neighbour family that holds most of the "
             "cluster's members ('background' when none):</p>"
             + tab(pd.crosstab(db.swarm_like, db.main_zbz_type, margins=True).reset_index(), fmt=lambda x: f"{x:.0f}"))
    h.append("""<p>As in v4, the two methods agree where both see the events (Endeavour-Explorer, Gorda, Mendocino and
Puget clusters have medians of 68-100% of their events in nearest-neighbour families; the Blanco 30%) and differ
where the DBSCAN clusters are made of events below Mc (Axial and the Cascade volcanoes: median 0%). The DBSCAN swarm-like flag uses all magnitudes, so it is not the
same test as our swarm type.</p>""")

    # ---------------- volcanoes
    h.append("<h2>Cascade volcanoes</h2>")
    vl, vl4 = d["volcano_local_zbz"], d["volcano_local_zbz_v4"]
    vv = vl.merge(vl4[["volcano", "n_all", "Mc_local", "n_above", "pct_background", "ashman_D", "ashman_D_shuffled",
                       "n_families"]], on="volcano", suffixes=("", "_v4"))
    h.append(tab(vv, rename={"n_all": "events 20 km", "n_above": "&ge; local Mc", "ashman_D_shuffled": "D shuffled",
                             "n_all_v4": "events v4", "n_above_v4": "&ge; Mc v4",
                             "ashman_D_shuffled_v4": "D shuffled v4"}))
    h.append(f"""<p>Zoomed analysis within 20 km with a local Mc. At Mount St. Helens the shuffled catalog is now less
bimodal than the real one (D {vl.set_index('volcano').loc['Mount St. Helens', 'ashman_D_shuffled']:.2f} against
{vl.set_index('volcano').loc['Mount St. Helens', 'ashman_D']:.2f}; v4 1.91 against 2.23), so the test is a little
stronger than in v4, but none of the volcano families meets two swarm indicators: they are small bursts of
near-equal events. At Mount Hood the shuffled catalog is more bimodal than the real one (76 events), so the Hood result
is not meaningful. Within 15 km of the summits (regional Mc):</p>""")
    h.append(tab(d["volcano_families"].merge(d["volcano_families_v4"][["volcano", "n_events_15km", "pct_background",
                                                                      "n_families", "n_swarm"]],
                                             on="volcano", suffixes=("", "_v4"), how="left"), fmt=lambda x: f"{x:.1f}"))

    # ---------------- sensitivity
    h.append("<h2>Sensitivity</h2>")
    h.append(img(out / "fig_sensitivity.png", "sensitivity"))
    h.append("<p>% background above Mc by variant:</p>" + tab(piv_bg.reset_index(), fmt=lambda x: f"{x:.1f}"))
    h.append("<p>Share of events (in both analysis sets) with the same background/clustered label as the main "
             "run:</p>" + tab(piv_ag.drop(columns="main").reset_index(), fmt=lambda x: f"{x:.3f}"))
    piv3 = sens.pivot(index="region", columns="variant", values="n_swarm").reindex(REG_ORDER).reset_index()
    h.append("<p>Number of swarms by variant:</p>" + tab(piv3, fmt=lambda x: f"{x:.0f}"))
    h.append(f"""<p>Variants: <code>d3</code> 3-D distance with every depth; <code>d3q</code> 3-D only between two
depth_good events; <code>mc+0.3</code>; <code>mc_best</code> the mag piece's Mc_best with b above it;
<code>df_est</code>, <code>df1.3</code>, <code>df2.0</code> fractal dimension; <code>noM_floor</code> M = Mc for
events without M; <code>mc_year</code> Mc per region and year; <code>eta0_shuf5</code> threshold passing 5% of
shuffled links; <code>tierABC</code> tier D out of the analysis set; <code>v41M</code> v4.1 magnitudes.</p>
<p>The background/clustered label agrees with the main run for
{100 * piv_ag[['d3q', 'df1.3', 'df2.0', 'df_est', 'mc+0.3', 'mc_year', 'v41M']].min().min():.0f}-100% of events
under d3q, df, Mc and magnitude changes. 3-D distance with every depth (d3) raises the background share by up to
{(piv_bg['d3'] - piv_bg['main']).max():.1f} points; restricted to well-constrained depths (d3q) the change is at
most {np.abs(piv_bg['d3q'] - piv_bg['main']).max():.1f} points, so with v5.1 depths 3-D distance would be usable for
the depth_good events. The threshold is again the largest choice: the conservative eta0 raises the background share
by {(piv_bg['eta0_shuf5'] - piv_bg['main']).min():.1f}-{(piv_bg['eta0_shuf5'] - piv_bg['main']).max():.1f} points,
most in {', '.join(f"{r} (+{v:.0f})" for r, v in (piv_bg['eta0_shuf5'] - piv_bg['main']).sort_values(
ascending=False).head(3).items())}. Taking tier D out lowers the Axial background share by
{piv_bg.loc['AX', 'main'] - piv_bg.loc['AX', 'tierABC']:.0f} points and offshore Oregon by
{piv_bg.loc['OR', 'main'] - piv_bg.loc['OR', 'tierABC']:.0f} points, where tier D is
{100 * ((a.region == 'OR') & (a.tier == 'D')).sum() / max((a.region == 'OR').sum(), 1):.0f}% of the events above
Mc.</p>""")

    # ---------------- topology
    h.append("<h2>Topology and tectonic setting</h2>")
    h.append(img(out / "fig_topology.png", "topology"))
    stc = st.reset_index()
    for c in ["pct_swarm_of_ge5", "median_leaf_depth_ge5", "mean_offspring_Mc1to2", "pct_ge_Mc1_single"]:
        stc[c + "_v4"] = stc.region.map(st4[c])
    h.append(tab(stc, ["region", "n_fam_ge5", "pct_swarm_of_ge5", "pct_swarm_of_ge5_v4", "median_leaf_depth_ge5",
                       "median_leaf_depth_ge5_v4", "n_events_Mc1to2", "mean_offspring_Mc1to2",
                       "mean_offspring_Mc1to2_v4", "pct_Mc1to2_no_offspring", "pct_ge_Mc1_single",
                       "pct_ge_Mc1_single_v4", "pct_MA_with_foreshocks", "n_fam_all_ge10",
                       "pct_fam_all_ge10_migrating_v51"],
                 rename={"n_fam_ge5": "families n&ge;5", "pct_swarm_of_ge5": "% swarms", "pct_swarm_of_ge5_v4": "v4",
                         "median_leaf_depth_ge5": "median leaf depth", "median_leaf_depth_ge5_v4": "v4 ",
                         "n_events_Mc1to2": "events Mc+1 to +2", "mean_offspring_Mc1to2": "mean offspring",
                         "mean_offspring_Mc1to2_v4": "v4  ", "pct_Mc1to2_no_offspring": "% no offspring",
                         "pct_ge_Mc1_single": "% &ge;Mc+1 single", "pct_ge_Mc1_single_v4": "v4   ",
                         "pct_MA_with_foreshocks": "% MA with foreshocks", "n_fam_all_ge10": "families n_all&ge;10",
                         "pct_fam_all_ge10_migrating_v51": "% migrating"}, fmt=lambda x: f"{x:.2f}"))
    h.append(f"""<p>The setting contrasts of v4 remain. The Mendocino triple junction has the fewest swarms among
families of 5 or more ({st.pct_swarm_of_ge5['MT']:.0f}%), the shallowest trees (median leaf depth
{st.median_leaf_depth_ge5['MT']:.2f}) and low productivity (events 1-2 units above Mc have
{st.mean_offspring_Mc1to2['MT']:.2f} linked offspring on average, {st.pct_Mc1to2_no_offspring['MT']:.0f}% have none),
consistent with Hsu et al. (2026). The Blanco has the highest productivity ({st.mean_offspring_Mc1to2['BL']:.2f})
and {st.pct_swarm_of_ge5['BL']:.0f}% swarms, the oceanic-transform pattern of swarm-like sequences with foreshocks
(McGuire et al. 2005; Roland &amp; McGuire 2009; Boettcher &amp; McGuire 2009; Kuna et al. 2019). Axial
({st.pct_swarm_of_ge5['AX']:.0f}%) and ON ({st.pct_swarm_of_ge5['ON']:.0f}%, which holds the Juan de Fuca ridge
segments outside the boxes and the volcanic arc) have more swarms than in v4 (61% and 66%); the Gorda plate fewer
({st.pct_swarm_of_ge5['GO']:.0f}% against 65%, on 25 families). Those shifts are within what a handful of families
near the S1-S3 thresholds can produce. The heat-flow ordering of Zaliapin &amp; Ben-Zion (2013b, 2016) between ridge
and transform settings and the Mendocino triple junction holds; the forearc and Puget results rest on few families.</p>""")

    # ---------------- faults
    h.append("<h2>Families on the fault surfaces v1</h2>")
    h.append(f"""<p>Join with <code>fault_surfaces_v1_assignment.csv.gz</code> of the faults piece (142 surfaces;
fit members and uniquely assigned events; ambiguous events count as off-fault). Rerun with
<code>v51_clusters_zbz.py --steps faults,compare</code> if the surfaces change. Half of the families have at least
{100 * fam.frac_on_faults.median():.0f}% of their events on a surface (swarms
{100 * fam[fam.type == 'swarm'].frac_on_faults.median():.0f}%, mainshock-aftershock
{100 * fam[fam.type == 'mainshock-aftershock'].frac_on_faults.median():.0f}%).</p>""")
    h.append(img(out / "fig_faults.png", "faults"))
    pfs = pf[pf.n_analysis >= 30].sort_values("pct_swarm", ascending=False)
    pft = pfs.head(15).copy()
    pft["literature"] = pft.lit_match.fillna("").str.slice(0, 80)
    h.append(tab(pft, ["surface_id", "region", "kind", "n_analysis", "pct_background", "pct_swarm",
                       "n_families", "v0_surfaces", "literature"],
                 rename={"n_analysis": "events &ge; Mc", "v0_surfaces": "v0 surfaces",
                         "literature": "literature match (carried from v0, truncated)"}))
    ps = pfs.set_index("surface_id").pct_swarm
    h.append(f"""<p>The highest swarm shares (surfaces with 30 or more events above Mc) are on v1_ON016, the recurring
swarm source about 19 km north of Mount McLoughlin (v0 ON027; {ps.get('v1_ON016', np.nan):.0f}%), v1_ON010 on the
Mount Hood fault zone (v0 ON015; {ps.get('v1_ON010', np.nan):.0f}%), the two western Blanco strands v1_BL001 and
v1_BL002, which combine the v0 West Blanco Depression and Surveyor Depression strands BL004a-BL018a
({ps.get('v1_BL001', np.nan):.0f}% and {ps.get('v1_BL002', np.nan):.0f}%), the northern Escanaba Trough axis
(v1_GO004, v0 GO002; {ps.get('v1_GO004', np.nan):.0f}%) and the Gorda Ridge axis near the President Jackson
Seamounts (v1_GO002; {ps.get('v1_GO002', np.nan):.0f}%). These are the same places as in v4 (ON027, the West Blanco
Depression strands, GO002). The faults piece removed surfaces whose events occur mostly on weekday daytime (possible
unflagged blasts): {int(day.n):,} events, of which {int(day.in_analysis)} are above Mc and {int(day.in_families)}
are in families here; they are kept in this analysis because the catalog does not flag them.</p>""")

    # ---------------- limits
    h.append("<h2>Limits and open issues</h2><ul>")
    for li in [
        f"Completeness changes in space and time. The constant regional Mc keeps {len(a):,} of {n_used:,} events in "
        "the analysis set; volcano microswarms and the Axial events below M 0.65 are only attached afterwards.",
        "The main run uses epicentral distance, so slab and crustal events that share an epicentre can link. With "
        "v5.1 depths the d3q variant (3-D only between well-constrained depths) changes few labels, so a 3-D main run "
        "is possible in a next version.",
        "Tier D epicentres (median 68% semi-axis about 5 km at Axial, shift from v4 4-23 km by region) enter the "
        "analysis set. At Axial they are a third of the events above Mc and mostly background; the Axial clustered "
        "share depends on this choice (variant tierABC) and on the threshold (eta0_shuf5).",
        f"Location errors (median h_max {sw10.h_err_med_km.median():.1f} km for tier A-C swarm members) are as large "
        "as most swarms, so "
        "migration and sqrt(t) fits are unresolved for most families. Relative (double-difference) relocation of the "
        "swarms is needed before diffusivities can be interpreted.",
        "Magnitudes: offshore ML is 0.3-0.5 above ComCat ML at Mendocino, Blanco and Gorda (mag piece); M1-M2 gaps of "
        "a few tenths drive the swarm and multiplet rules and are close to the ML uncertainty (about 0.2). "
        f"{int(used.M.isna().sum()):,} located earthquakes have no M and cannot be parents.",
        "Swarm criteria are thresholds on continuous properties; families near them switch type between versions "
        f"({100 - _kept_same_type(s4):.0f}% of the kept families changed type).",
        "Rate changes from network changes (OBS deployments 2011-2015, NC/BK pick gap in 2016) are not modelled; Mc is "
        "the worst yearly value per region to limit their effect.",
        "1,137 events on surfaces the faults piece removed for weekday-daytime timing are kept; they may be unflagged "
        "blasts."]:
        h.append(f"<li>{li}</li>")
    h.append("</ul>")

    # ---------------- files
    h.append("<h2>Files</h2><ul>")
    for fn, desc in [
        ("events_zbz_v51.csv.gz (worktree, slim) and /wd1/.../v5.1_analysis/clusters/events_zbz_v51.csv.gz (full)",
         "every catalog row: region box, in_analysis, M, parent_id, log_eta, log_T, log_R, nn_dist_km, nn_dt_days, "
         "linked, family_id, family_type, role, background, cls, excluded (full file adds time, lat, lon, depth, tier, "
         "flags)"),
        ("families_zbz_v51.csv", "every family: type, size above Mc and in all, magnitude range, start/end, durations, "
         "extent, depth ranges (depth_good), swarm indicators, topology, v4-style migration, v5.1 migration "
         "(rho, radial speed, velocity vector and azimuth, vz), D90, D50, envelope exponent, resolvable, style, "
         "fault fractions, v4 status"),
        ("regions_zbz.csv, thresholds_zbz.csv, completeness_b_df.csv, completeness_by_year.csv",
         "per-region summary, mixture fits and thresholds, Mc and b"),
        ("sensitivity_zbz.csv; /wd1/.../clusters/variants/", "variant summaries; per-variant event and family tables"),
        ("shuffle_test.csv, shuffle_eta_seed0.npz", "time-shuffled catalogs"),
        ("v4_families_status.csv, v51_families_status.csv, v4_v51_family_overlap.csv, v4_vs_v51_event_classes*.csv, "
         "v4_vs_v51_analysis_membership.csv, v4_dissolved_members_fate.csv", "comparison with v4"),
        ("dbscan_clusters_v51.csv, dbscan_members_v51.csv, dbscan_v4_status.csv, dbscan_v51_status.csv",
         "DBSCAN swarm scan and its comparison with v4"),
        ("validation_sequences.csv, volcano_families.csv, volcano_local_zbz.csv, volcano_local_families.csv, "
         "setting_stats.csv", "validation and setting tables"),
        ("faults_v1_cluster_counts.csv, faults_v1_daytime_surfaces_events.csv", "fault surfaces v1 join")]:
        h.append(f"<li><code>{html.escape(fn)}</code>: {html.escape(desc)}</li>")
    h.append("</ul>")
    h.append("<h2>References</h2><p class='muted'>Reference list of the v4 clustering report (DOIs checked there "
             "against Crossref on 2026-11-22) plus Shapiro et al. (1997), whose DOI was not checked here.</p><ul>")
    for txt, doi in REFS:
        link = f" <a href='https://doi.org/{doi}'>doi:{doi}</a>" if doi else ""
        h.append(f"<li>{html.escape(txt)}{link}</li>")
    h.append("</ul></body></html>")
    (out / "catalog_v51_clusters_report.html").write_text("\n".join(h))


def _kept_same_type(s4):
    k = s4[s4.status == "kept"]
    return 100 * (k.type == k.type_v51_main).mean() if len(k) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-map", action="store_true")
    a = ap.parse_args()
    use()
    d = load()
    df = Z.prepare(Z.load_all())
    fig_tr(d, OUT)
    fig_eta(d, OUT)
    if not a.no_map:
        fig_map(d, OUT)
    fig_compare(d, OUT)
    fig_examples(d, OUT)
    fig_migration(d, df, OUT)
    fig_depth(d, OUT)
    fig_topology(d, OUT)
    fig_sens(d, OUT)
    fig_faults(d, OUT)
    write_html(d, OUT)
    print("report written:", OUT / "catalog_v51_clusters_report.html")


if __name__ == "__main__":
    main()
