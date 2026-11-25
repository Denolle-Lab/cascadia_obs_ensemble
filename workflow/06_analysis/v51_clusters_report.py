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
import json
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
              "v4_dissolved_members_fate", "null_test", "null_test_summary", "tierD_by_region", "mc_edge_check",
              "depth_swarm_vs_ma_families", "review_sets_in_clusters", "review_P1_families"]:
        d[k] = pd.read_csv(OUT / f"{k}.csv")
    d["input_commits"] = json.loads((OUT / "input_commits.json").read_text())
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
    nmig_sp = int(tb(fam10.mig_migrating_spearman_p).sum())
    day = d["faults_v1_daytime_surfaces_events"].iloc[0]
    cls_t = d["classes"]
    # null test (time-shuffled catalogs at the real thresholds)
    ns = d["null_test_summary"]
    nreg = ns[ns.null == "region"].set_index("region").reindex(REG_ORDER)
    nyr = ns[ns.null == "region_year"].set_index("region").reindex(REG_ORDER)
    nmo = ns[ns.null == "region_month"].set_index("region").reindex(REG_ORDER)
    lr = nreg.link_ratio_null_over_real
    low = [r for r in REG_ORDER if r not in ("AX", "OR", "MT", "BL")]
    valley = [r for r in REG_ORDER if bool(nreg.real_kde_valley[r])]
    novalley = [r for r in REG_ORDER if r not in valley]
    dmax = nreg.D_max
    # largest D over every shuffled catalog (null step: 3 kinds of shuffle; shuffle step: 3 seeds)
    nt = d["null_test"]
    dall = pd.concat([nt[nt.null != "real"][["region", "ashman_D"]],
                      d["shuffle_test"][["region", "ashman_D"]]]).groupby("region").ashman_D.max().reindex(REG_ORDER)
    near = [r for r in REG_ORDER if dall[r] >= 2]
    d_above_txt = ("with the v4 mixture fit no shuffled catalog reaches the real D, but shuffled D depends on the seed "
                   "and the kind of shuffle and reaches "
                   + ", ".join(f"{r} {dall[r]:.2f} (real {reg.ashman_D[r]:.2f})" for r in near)
                   + "; with a multi-start fit the reviewer found shuffled D of 5.35 at AX and 3.0 at OR")
    # swarm counts in shuffled catalogs
    swr = (nreg.n_swarm_mean / nreg.real_n_swarm.replace(0, np.nan)).reindex(REG_ORDER)
    sw_half = [r for r in REG_ORDER if swr.get(r, np.nan) >= 0.5]
    ry0, ry1 = 100 * nyr.link_ratio_null_over_real.min(), 100 * nyr.link_ratio_null_over_real.max()
    shs = d["shuffle_test"]
    sh_seed_max_en = float(shs[shs.region == "EN"].ashman_D.max())
    # tier D
    td = d["tierD_by_region"].set_index("region")
    tdo = td.drop(index=["AX", "MT", "all"]).h_max_med_tierD_km
    # Axial background range across the variants
    axr = piv_bg.loc["AX"].dropna()
    # review sets
    rsc = d["review_sets_in_clusters"].set_index(["problem", "role"])
    p1d = rsc.loc[("P1", "deep")]
    p1f = d["review_P1_families"]
    nwed = int(tb(p1f.wednesday_local_noon).sum())
    wt = p1f[tb(p1f.wednesday_local_noon)].time_local.sort_values()
    wed_t0, wed_t1 = (wt.iloc[0], wt.iloc[-1]) if len(wt) else ("", "")

    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Catalog v5.1 clusters</title><style>{CSS}</style>"
         f"</head><body>"]
    h.append("<h1>Earthquake clusters in catalog v5.1 (nearest-neighbour method)</h1>")
    h.append(f"<p class='muted'>Cascadia catalog v5.1 (NonLinLoc in model v0.3.1), 2010-2016, with the v5.1 Route A "
             f"magnitudes (piece <i>mag</i>). Method of Zaliapin &amp; Ben-Zion (2013a, 2016), same code and "
             f"parameters as the catalog v4 analysis. Code: <code>workflow/06_analysis/v51_clusters_zbz.py</code>, "
             f"<code>v51_swarms.py</code>, <code>v51_clusters_report.py</code>, branch "
             f"<code>analysis/v51-clusters</code>, 2026-11-25. Private working product; not peer reviewed. Revised "
             f"the same day after an independent review (null test, tier D errors, permutation migration test, "
             f"family-level depth test, fault surfaces after their own fixes).</p>")
    h.append("<p class='muted'><b>Context.</b> Catalog v5.1 is located in model v0.3.1. The PI has adopted model "
             "v0.4.0 (D-B082); v0.4.1 (surface floor in the inversion, D-B083) is being built; catalog v6.0 (D-B079, "
             "NonLinLoc with SSST station terms) will supersede v5.1 and these products will be redone on it. A "
             "waveform review of four v5.1 location problems exists: Mount St Helens deep events (P1, probably false "
             "events from a weekly artificial signal on the analog UW channels HSR, EDM and STD, 2011-05 to 2013-08), "
             "Brookings 2012 (P2, NonLinLoc took the deeper of two S-P depth minima), tensor events (P3, a median "
             "13.7 km deeper than ComCat) and far_only events (P4, moved from near the surface in v4 to 44-64 km). The "
             "main run uses epicentres only, so P2-P4 do not change links; P1 events are in the analysis set and "
             "their families are flagged.</p>")

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
<li><b>Background shares order by setting as in v4</b>: about 50-62% at Axial, Endeavour-Explorer, the Blanco and
the Gorda plate, 58% at the Mendocino triple junction, 70-84% onshore, in the Puget Lowland and offshore Washington
and Oregon. Swarms make
{st.pct_swarm_of_ge5[['AX', 'BL', 'ON']].min():.0f}-{st.pct_swarm_of_ge5[['AX', 'BL', 'ON']].max():.0f}% of
the families with 5 or more events at Axial, the Blanco and in ON, against {st.pct_swarm_of_ge5['MT']:.0f}% at the
Mendocino triple junction, where trees are still the shallowest (median leaf depth
{st.median_leaf_depth_ge5['MT']:.2f}; v4 {st4.median_leaf_depth_ge5['MT']:.2f}).</li>
<li><b>Time-shuffled catalogs reproduce many of the links, most of them at Axial.</b> At the real thresholds and
caps, catalogs whose event times are permuted within each region (mean of {int(nreg.n_seeds.iloc[0])} seeds) link
{100 * lr['AX']:.0f}% as many events as the real catalog at Axial, {100 * lr['OR']:.0f}% offshore Oregon,
{100 * lr['MT']:.0f}% at the Mendocino triple junction, {100 * lr['BL']:.0f}% on the Blanco and
{100 * lr[low].min():.0f}-{100 * lr[low].max():.0f}% in {', '.join(low)}. Most Axial links therefore come from the
concentration of events in the caldera, not from clustering in time. The Axial swarm count is reproduced less: a
shuffled catalog forms {nreg.n_swarm_mean['AX']:.1f} swarms on average against {int(nreg.real_n_swarm['AX'])} in the
real catalog.{(" Shuffled catalogs form at least half of the real swarm count in " + ", ".join(f"{r} ({nreg.n_swarm_mean[r]:.0f} against {int(nreg.real_n_swarm[r])})" for r in sw_half) + ".") if sw_half else ""}
Shuffles within the same calendar month reproduce {100 * nmo.link_ratio_null_over_real.min():.0f}-{100 * nmo.link_ratio_null_over_real.max():.0f}%
of the links. The mixture separation is not a test of clustering: Ashman D is below 2 in
{', '.join(r for r in REG_ORDER if nreg.real_ashman_D[r] < 2)}, the log eta density has a valley between its two
modes only in {', '.join(valley)}, and {d_above_txt}.</li>
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
({piv_bg.loc['AX', 'eta0_shuf5']:.1f}% against 82.2%). Across the sensitivity variants the Axial background share
ranges from {axr.min():.1f}% ({axr.idxmin()}) to {axr.max():.1f}% ({axr.idxmax()}): read the Axial clustered share as
a range, not a number. The 2015 eruption is still one chain-like swarm, starting 2015-04-24 05:13 UTC, with
{int(val[val.label.str.contains('Axial')].n.iloc[0])} events above Mc (v4: 632).</li>
<li><b>Tier D epicentres are poor outside Axial.</b> {nax_d:,} of the {nax:,} Axial events above Mc are tier D (most
outside the caldera network: median nearest station 4.3 km, azimuthal gap 327 deg, median 68% horizontal semi-axis
h_max {td.h_max_med_tierD_km['AX']:.1f} km); {100 * ax_tier.get('D', np.nan):.0f}% of them are background, against
{100 * ax_tier.get('A', np.nan):.0f}% of tier A events. Elsewhere tier D epicentres are much worse: median h_max
{td.h_max_med_tierD_km['MT']:.0f} km at the Mendocino triple junction and {tdo.min():.0f}-{tdo.max():.0f} km in the
other regions. {td.pct_links_tierD_end['all']:.0f}% of all links have a tier D end ({td.pct_links_tierD_end['ON']:.0f}%
in ON). Taking tier D out of the analysis set (variant tierABC) lowers the Axial background share to
{piv_bg.loc['AX', 'tierABC']:.1f}%; elsewhere it changes the background/clustered label of at most
{100 * (1 - piv_ag.drop(index='AX')['tierABC'].min()):.1f}% of events, and it removes the tier D foreshock that merges
the 2014 Vancouver Island family with a 2013 event (validation section).</li>
<li><b>Migration is rarely resolved.</b> Of {len(fam10)} families with 10 or more tier A-C members, {nres} have a
spread (90% of distances from the first 10% of events) larger than twice their median horizontal error, and {nmig}
migrate by our test (Spearman rho &ge; 0.5, permutation p &lt; 0.01, positive Theil-Sen slope, resolvable). By
construction about {0.01 * nres:.1f} of the {nres} resolvable families pass the permutation test by chance. The first
build used the Spearman p-value and found {nmig_sp}; that p-value is biased toward migration because the origin is
taken from the same events. Only {int(tb(sw10.mig_resolvable).sum())} of the {len(sw10)} swarms of this size are
resolvable, so the sqrt(t) diffusion fits (D90) mostly describe location scatter, not a triggering front. The 90%
distance envelope grows with time as t^p with median p = {sw10.env_p.median():.2f} for swarms, i.e. no growth.</li>
<li><b>Mount St Helens deep events (waveform review P1).</b> {int(p1d.in_analysis)} of the {int(p1d.n)} deep P1 events
are above Mc, and {int(p1d.in_families)} are in {int(p1d.n_families)} families. {nwed} of these families start on a
Wednesday between {wed_t0} and {wed_t1} Pacific time, i.e. at local noon in both standard and daylight time; the
review found a weekly artificial signal on Wednesdays on the analog UW channels HSR, EDM and STD. These families are
probably not earthquake sequences; they carry the flag <code>n_review_P1_msh_deep</code> until the human review is
done.</li>
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
clustered weight dropped from {thr4.w1['AX']:.2f} to {thr.w1['AX']:.2f}. The share of events linked in
time-shuffled catalogs at Axial dropped from 34% to {100 * sh.frac_linked['AX']:.0f}%, but the real linked share
dropped too: a shuffled catalog still links {100 * lr['AX']:.0f}% as many Axial events as the real one.</li>
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
    td_tab = tab(td.reset_index(), ["region", "n_analysis", "n_tierD", "pct_tierD", "h_max_med_tierD_km",
                                    "h_max_med_tierABC_km", "n_five_phase", "h_max_med_five_phase_km",
                                    "pct_background_tierD", "pct_background_tierABC", "n_links", "pct_links_tierD_end",
                                    "pct_links_hmax_ge10", "n_families_tierD_largest"],
                 rename={"n_analysis": "events &ge; Mc", "n_tierD": "tier D", "pct_tierD": "% tier D",
                         "h_max_med_tierD_km": "h_max tier D (km)", "h_max_med_tierABC_km": "h_max tier A-C (km)",
                         "n_five_phase": "five-phase", "h_max_med_five_phase_km": "h_max five-phase (km)",
                         "pct_background_tierD": "% bg tier D", "pct_background_tierABC": "% bg tier A-C",
                         "n_links": "links", "pct_links_tierD_end": "% links with a tier D end",
                         "pct_links_hmax_ge10": "% links with h_max &ge; 10 km",
                         "n_families_tierD_largest": "families led by a tier D event"}, fmt=lambda x: f"{x:.1f}")
    h.append(f"""<p><b>Events.</b> Catalog v5.1 ({nrow:,} rows). Used: {n_used:,} located earthquakes (171,717 NonLinLoc
locations and the 2 ComCat hypocentres of the v4.1 additions, minus {nx.get('not_earthquake', 0):,} events with
<code>is_earthquake</code> False: anss_explosion 1,665, quarry_blast_likely 1,572, anss_quarry_blast 74, both
flags 1, anss_other_event 1). The {nx.get('no_hypocentre', 0)} rows without a hypocentre are in the event table with
role <code>excluded</code> and are not used anywhere.</p>
<p><b>Poorly constrained depths.</b> floor_limited ({nfl:,} used events), far_only ({nfar}) and deep_pass ({ndp})
events enter through their epicentres only: the main run uses epicentral distance, as in v4. Their depths are not
used in the main run or in any depth statistic: depth ranges, depth migration and the 3-D variant <code>d3q</code>
use only <i>depth_good</i> events (tier A-C, depth_quality ok, not floor_limited, far_only or deep_pass; {ndg:,}
events). depth_quality 'unconstrained' (ridge regions, Blanco outside the 2012-13 OBS deployment) and 'biased_deep'
(Gorda) are also left out of depth statistics. Three outputs do use every depth and are labelled so: variant
<code>d3</code> (3-D distance with all depths, a sensitivity test), the family column
<code>depth_main_any_quality</code> (depth of the largest event; check <code>depth_main_good</code>) and the DBSCAN
columns <code>depth_any_quality_med</code> and <code>_iqr</code>. Because the main run uses epicentres only, the
depth problems found by the waveform review (P2 Brookings, P3 tensor events a median 13.7 km deeper than ComCat, P4
far_only events moved from near the surface to 44-64 km) do not change any link; they enter depth statistics only
where those events are depth_good.</p>
<p><b>Tier D.</b> Tier D events stay in the analysis set, as all events did in v4; variant <code>tierABC</code> takes
them out of it and attaches them afterwards. Their epicentres are reliable only at Axial. Median 68% horizontal
semi-axis (h_max) of the events above Mc, links with a tier D end, and background shares:</p>
{td_tab}
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
    mce = d["mc_edge_check"].set_index("region").reindex(REG_ORDER)
    mdiff = mce[np.abs(mce.Mc_v4_float - mce.Mc_exact) > 1e-6]
    mdiff_txt = "; ".join(f"{r} {row.Mc_v4_float:.2f} to {row.Mc_exact:.2f} ({int(row.n_above_v4_float):,} to "
                          f"{int(row.n_above_exact):,} events above Mc; the {int(row.year_v4_float)} histogram that "
                          f"sets Mc has two largest bins of {row.modal_counts_v4_float.replace(';', ' and ')} events "
                          f"with the v4 edges and {row.modal_counts_exact.replace(';', ' and ')} with exact edges)"
                          for r, row in mdiff.iterrows())
    h.append(f"""<p><b>The Mc rule depends on the bin edges.</b> The v4 code bins M with
<code>np.arange(-2, 8, 0.1)</code>, whose edges are not exact decimals, so the few magnitudes that fall exactly on a
0.1 value go to the bin below. With exact edges (-2.0, -1.9, ...) Mc changes in {len(mdiff)} regions:
{mdiff_txt}. Two near-equal modal bins decide Mc there. The run keeps the v4 edges so that v4 and v5.1 stay
comparable; an exact-edge rerun would have to redo v4 the same way (<code>mc_edge_check.csv</code>).</p>""")
    # b sample: the v4 function compares M with the unrounded Mc, the analysis set with the rounded one
    ev_an = d["ev"][d["ev"].in_analysis.astype(bool)]
    n_gap = (d["regions_zbz"].set_index("region").n_analysis - comp.n_above).reindex(REG_ORDER)
    b_full = pd.Series({r: 1.0 / (np.log(10) * (ev_an.M[ev_an.region == r].mean() - comp.Mc[r])) for r in REG_ORDER})
    b_gap = b_full - comp.b.reindex(REG_ORDER)
    h.append(f"""<p><b>b sample.</b> b and the events &ge; Mc of this table come from the v4 function, which compares
M with the unrounded Mc (a float error of a few 1e-15 above the nominal value), so the {int(n_gap.sum())} events with
M exactly at Mc ({', '.join(f'{r} {int(k)}' for r, k in n_gap.items() if k > 0)}) are left out of b; the analysis
set (<code>regions_zbz.csv</code>) includes them. With them b would be higher by at most {b_gap.max():.3f}, less
than the b error ({comp.b_err.min():.3f}-{comp.b_err.max():.3f}). v4 computed b the same way, so the run keeps it for
comparability.</p>""")
    ic = d["input_commits"]
    h.append("<p><b>Inputs from the other v5.1 pieces</b> are read from their pushed branches, not from their "
             "worktrees: " + "; ".join(f"<code>{html.escape(k)}</code> at {v['ref']} {v['commit'][:8]}"
                                        for k, v in sorted(ic.items())) + ".</p>")
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
    h.append("<h2>Threshold and null test</h2>")
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
    h.append(f"""<p><b>The mixture is not a test of clustering.</b> The threshold is where the two fitted Gaussian
components of log eta cross. Their separation (Ashman D) is above 2 in
{', '.join(r for r in REG_ORDER if reg.ashman_D[r] >= 2)} and below 2 in
{', '.join(r for r in REG_ORDER if reg.ashman_D[r] < 2)}. A density valley between the two modes (kernel density
minimum strictly between the two means) exists only in {', '.join(valley)}; in {', '.join(novalley)} the clustered
component is a shoulder of one peak (column 'KDE valley' equals 'mu clust.'). As for shuffled catalogs,
{d_above_txt}. The first build said that all nine regions are bimodal and lose it when
shuffled; that compared real D with the range of per-region means of three shuffles and does not hold seed by seed
(its own seed 1 gave EN D = {sh_seed_max_en:.2f}).</p>
<p><b>Null test of the links.</b> The test that holds compares the share of events above Mc that are linked (below
eta0 and within the cap) in the real catalog with the same share in time-shuffled catalogs at the real thresholds.
Shuffled catalogs link fewer events in every region, but the ratio varies (table below): a shuffle over the whole
period reproduces {100 * lr['AX']:.0f}% of the Axial links, {100 * lr['OR']:.0f}% offshore Oregon,
{100 * lr['MT']:.0f}% at the Mendocino triple junction, {100 * lr['BL']:.0f}% on the Blanco and
{100 * lr[low].min():.0f}-{100 * lr[low].max():.0f}% in {', '.join(low)}. Shuffled catalogs also form families and
swarms: at Axial {nreg.n_families_mean['AX']:.0f} families (real {int(nreg.real_n_families['AX'])}) and
{nreg.n_swarm_mean['AX']:.1f} swarms on average (real {int(nreg.real_n_swarm['AX'])}), on the Blanco
{nreg.n_swarm_mean['BL']:.1f} swarms (real {int(nreg.real_n_swarm['BL'])}), at the Mendocino triple junction
{nreg.n_swarm_mean['MT']:.1f} (real {int(nreg.real_n_swarm['MT'])}). Axial events above Mc sit in a 5 km caldera and
most occur in the months of rising rate before the April 2015 eruption (Wilcock et al. 2016); a whole-period shuffle
keeps the location and the event count of that period, so most Axial links and families are not evidence of
clustering in time; the Axial swarm count is less affected. Shuffles that keep events in the same calendar year or
month keep slow rate changes (network deployments, long sequences, the pre-eruption rise) and reproduce more:
{ry0:.0f}-{ry1:.0f}% of the links within the year and
{100 * nmo.link_ratio_null_over_real.min():.0f}-{100 * nmo.link_ratio_null_over_real.max():.0f}% within the month. At
these thresholds most links join events that are close in space and fall in the same month; the timing within the
month accounts for {100 * (1 - nmo.link_ratio_null_over_real.max()):.0f}-{100 * (1 - nmo.link_ratio_null_over_real.min()):.0f}%
of the links (most in {nmo.link_ratio_null_over_real.idxmin()}). The links measure space-time concentration at the
scale of weeks to months; on their own they are not evidence of short-term triggering.</p>""")
    nt_tab = pd.DataFrame(dict(
        region=REG_ORDER, real_pct_linked=100 * nreg.real_frac_linked.values,
        shuf_pct_linked=100 * nreg.frac_linked_mean.values, ratio_region=lr.values,
        ratio_year=nyr.link_ratio_null_over_real.values, ratio_month=nmo.link_ratio_null_over_real.values,
        real_D=nreg.real_ashman_D.values, shuf_D_min=nreg.D_min.values, shuf_D_max=nreg.D_max.values,
        real_valley=np.where(nreg.real_kde_valley.astype(bool), "yes", "no"),
        shuf_valley=100 * nreg.kde_valley_frac.values,
        real_fam=nreg.real_n_families.values, shuf_fam=nreg.n_families_mean.values,
        real_sw=nreg.real_n_swarm.values, shuf_sw=nreg.n_swarm_mean.values, shuf_sw_max=nreg.n_swarm_max.values))
    h.append(tab(nt_tab, rename={
        "real_pct_linked": "% linked, real", "shuf_pct_linked": "% linked, shuffled",
        "ratio_region": "shuffled / real (whole period)", "ratio_year": "shuffled / real (same year)",
        "ratio_month": "shuffled / real (same month)", "real_D": "D real", "shuf_D_min": "D shuffled min",
        "shuf_D_max": "D shuffled max", "real_valley": "KDE valley real", "shuf_valley": "% shuffles with valley",
        "real_fam": "families real", "shuf_fam": "families shuffled", "real_sw": "swarms real",
        "shuf_sw": "swarms shuffled (mean)", "shuf_sw_max": "swarms shuffled (max)"}))
    h.append(f"<p class='muted'>Shuffles at the real thresholds and caps: {int(nreg.n_seeds.iloc[0])} permutations of "
             f"the event times within each region over 2010-2016, {int(nyr.n_seeds.iloc[0])} within region and "
             f"calendar year, {int(nmo.n_seeds.iloc[0])} within region and month (step nulls, "
             "<code>null_test.csv</code>, <code>null_test_summary.csv</code>). Shuffled columns are means unless "
             "marked. The three shuffles of the shuffle step set the conservative threshold eta0_shuf5 and the figures "
             "above.</p>")

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
    vi = v.set_index("event_id")
    w = vi.loc[89617] if 89617 in vi.index else None
    vi_txt = ""
    if w is not None and str(w.get("other_tier", "")) == "D":
        vi_txt = (f"(2) The 2014-04-24 Mw 6.5 Vancouver Island family now starts with the 2013-08-04 M "
                  f"{w.other_M:.1f} event (ID {int(w.other_id)}), linked as a foreshock 263 days earlier: t90 grows "
                  f"from {w.duration90_days_v4:.0f} to {w.duration90_days:.0f} days and the mean leaf depth from "
                  f"{w.mean_leaf_depth_v4:.2f} to {w.mean_leaf_depth:.2f}. This link is a location artefact. Event "
                  f"{int(w.other_id)} is tier D (h_max {w.other_h_max_km:.0f} km, {int(w.other_n_sta)} stations, "
                  f"azimuthal gap {w.other_gap:.0f} deg). It sits {w.other_dist_v51_km:.1f} km from the mainshock "
                  f"epicentre in v5.1 but {w.other_dist_v4_km:.1f} km away in v4; the small v5.1 separation is far "
                  f"inside its location error. In variant tierABC the 2014 family is an ordinary "
                  f"{w.tierABC_type} sequence without it ({int(w.tierABC_n)} events above Mc, t90 "
                  f"{w.tierABC_duration90_days:.0f} days, mean leaf depth {w.tierABC_mean_leaf_depth:.2f}).")
    other = []
    for eid, row in vi.iterrows():
        if eid in (262, 89617) or not np.isfinite(row.get("duration90_days", np.nan)):
            continue
        ch = []
        t5, t4 = row.duration90_days, row.get("duration90_days_v4", np.nan)
        if np.isfinite(t4) and max(t5, t4) > 1 and (t5 > 2 * t4 or t5 < 0.5 * t4):
            ch.append(f"t90 {t4:.0f} to {t5:.0f} days")
        l5, l4 = row.mean_leaf_depth, row.get("mean_leaf_depth_v4", np.nan)
        if np.isfinite(l4) and abs(l5 - l4) >= 1:
            ch.append(f"mean leaf depth {l4:.2f} to {l5:.2f}")
        n5, n4 = row.n, row.get("n_v4", np.nan)
        if np.isfinite(n4) and (n5 > 1.5 * n4 or n5 < n4 / 1.5):
            ch.append(f"{n4:.0f} to {n5:.0f} events above Mc")
        if ch:
            # the same family without tier D (variant tierABC)
            ta = []
            tt, tl = row.get("tierABC_duration90_days", np.nan), row.get("tierABC_mean_leaf_depth", np.nan)
            if np.isfinite(tt) and np.isfinite(t4) and any("t90" in c for c in ch):
                ta.append(f"t90 {tt:.0f} days")
            if np.isfinite(tl) and np.isfinite(l4) and any("leaf" in c for c in ch):
                ta.append(f"leaf depth {tl:.2f}")
            other.append(f"{row.label}: {', '.join(ch)}" + (f" (without tier D: {', '.join(ta)})" if ta else ""))
    h.append(f"""<p>All six large tectonic mainshocks are again mainshock-aftershock families, and the two Blanco pairs and
triplets are again swarms with M1-M2 near 0.1. Two links changed. (1) The 2010-02-04 Mw 5.9 event, 44 km south of
the 2010-01-10 Mw 6.5 Gorda event in v4, is 54 km from it in v5.1; its link (log eta -5.27) now misses the MT
threshold (-5.39) and it heads its own family, so the 2010 family has M2 4.95 instead of 5.88 and 579 instead of 824
events above Mc. {vi_txt} The 2016-12-08 Mw 6.6 event (added in v4.1) has only 23 days of catalog after it and 4
events above Mc in its family.</p>
<p>Other known sequences changed shape without changing type: {'; '.join(other) if other else 'none'}. Where the
values without tier D (variant tierABC) are close to v4, tier D members drive the change; elsewhere it comes from
links of tier A-C events near the threshold. The table below gives the largest other member of each family, its
location quality, and the same family without tier D.</p>""")
    h.append(tab(v, ["label", "type", "n", "other_id", "other_M", "other_time", "other_tier", "other_h_max_km",
                     "other_n_sta", "other_gap", "other_dist_v51_km", "other_dist_v4_km", "tierABC_type", "tierABC_n",
                     "tierABC_duration90_days", "tierABC_mean_leaf_depth"],
                 rename={"n": "n &ge; Mc", "other_id": "largest other member", "other_M": "its M",
                         "other_time": "its time", "other_tier": "tier", "other_h_max_km": "h_max (km)",
                         "other_n_sta": "stations", "other_gap": "gap (deg)",
                         "other_dist_v51_km": "distance v5.1 (km)", "other_dist_v4_km": "distance v4 (km)",
                         "tierABC_type": "type, tierABC", "tierABC_n": "n, tierABC",
                         "tierABC_duration90_days": "t90, tierABC (d)",
                         "tierABC_mean_leaf_depth": "leaf depth, tierABC"}))
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
    dsm = d["depth_swarm_vs_ma_families"]
    da = dsm[dsm.events == "all"].set_index("region")
    dw = dsm[dsm.events == "without_P1_P3"].set_index("region")
    sig = [r for r in REG_ORDER if da.p_mannwhitney.get(r, np.nan) < 0.05]
    sigb = [r for r in REG_ORDER if da.p_bonferroni.get(r, np.nan) < 0.05]
    nsig = [r for r in REG_ORDER if np.isfinite(da.p_mannwhitney.get(r, np.nan)) and r not in sig]
    h.append(f"""<p><b>Swarm and mainshock-aftershock depths.</b> Events of one family are not independent, so the
comparison uses families as the unit: the median depth of each family's depth_good members (families with at least
3), compared by a Mann-Whitney test per region. Swarm families are shallower than mainshock-aftershock families in
{', '.join(f"{r} ({da.swarm_family_median_km[r]:.1f} against {da.MA_family_median_km[r]:.1f} km, "
           f"{int(da.n_swarm_families[r])} and {int(da.n_MA_families[r])} families, p = {da.p_mannwhitney[r]:.3f})"
           for r in sig)}. After a Bonferroni correction over the {int(da.n_tests.iloc[0])} regions tested only
{', '.join(sigb) if sigb else 'none'} stays below 0.05. {', '.join(nsig)} show no significant difference
(p {', '.join(f"{da.p_mannwhitney[r]:.2f}" for r in nsig)}). Without the review sets P1 (Mount St Helens deep events)
and P3 (tensor events) the p-values are {', '.join(f"{r} {dw.p_mannwhitney[r]:.3f}" for r in sig)}. The event-level
medians of the first build (GO {dt.loc['GO', 'swarm']:.1f} against {dt.loc['GO', 'mainshock-aftershock']:.1f} km,
ON {dt.loc['ON', 'swarm']:.1f} against {dt.loc['ON', 'mainshock-aftershock']:.1f}, MT {dt.loc['MT', 'swarm']:.1f}
against {dt.loc['MT', 'mainshock-aftershock']:.1f}) weight large families heavily and are not a test. In the Puget box,
{pu_dp:.0f}% of the depth_good events above Mc at 30 km or deeper (intraslab) are background, against
{pu_sh:.0f}% above 30 km.</p>""")
    h.append(tab(dsm, ["events", "region", "n_swarm_families", "n_MA_families", "swarm_family_median_km",
                       "MA_family_median_km", "swarm_event_median_km", "MA_event_median_km", "p_mannwhitney",
                       "p_bonferroni"],
                 rename={"events": "events used", "n_swarm_families": "swarm families", "n_MA_families": "MA families",
                         "swarm_family_median_km": "swarm (median of family medians, km)",
                         "MA_family_median_km": "MA (median of family medians, km)",
                         "swarm_event_median_km": "swarm events (km)", "MA_event_median_km": "MA events (km)",
                         "p_mannwhitney": "p", "p_bonferroni": "p Bonferroni"}, fmt=lambda x: f"{x:.3g}"))
    h.append("<h3>Migration and diffusion</h3>")
    h.append(img(out / "fig_migration.png", "migration examples"))
    h.append("<p class='muted'>Distance from the centroid of the first 10% of events (grey) against time for the "
             "migrating families and the largest swarms. Red: the sqrt(4 pi D90 t) front; blue band: twice the "
             "median horizontal error, below which distances are not resolved.</p>")
    mig = fam[tb(fam.mig_migrating)]
    lost = fam[tb(fam.mig_migrating_spearman_p) & ~tb(fam.mig_migrating)]
    h.append(f"""<p>Families that migrate by our test ({len(mig)}). The test uses a one-sided permutation p-value of
the radial Spearman rho (2,000 permutations of the positions against the times, the origin recomputed each time),
because taking the origin from the first 10% of the same events biases the ordinary Spearman p-value toward
migration. {' '.join(f"Family {int(r.family_id)} ({r.region}, {pd.Timestamp(r.start):%Y-%m-%d}, {r.type}, "
                     f"{int(r.mig_n)} tier A-C members) passes with the Spearman p-value ({r.mig_p:.3f}) but not "
                     f"with the permutation test (p = {r.mig_p_perm:.3f})." for r in lost.itertuples())}</p>""")
    h.append(tab(mig, ["family_id", "region", "region_v51", "type", "start", "n", "n_all", "mig_n", "Mmax",
                       "duration_all_days", "mig_rho", "mig_p", "mig_p_perm", "mig_v_radial_km_day", "mig_az",
                       "mig_v_dir_km_day", "mig_vz_km_day", "D90_m2s", "env_p", "h_err_med_km", "mig_r90_km",
                       "mig_style", "v4_status"],
                 rename={"mig_n": "n A-C", "duration_all_days": "days", "mig_rho": "rho", "mig_p": "Spearman p",
                         "mig_p_perm": "permutation p",
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
    vs = vl.set_index("volcano")
    ms = vs.loc["Mount St. Helens"] if "Mount St. Helens" in vs.index else None
    weak = [f"{v} ({int(r.n_above)} events above the local Mc; {int(r.n_shuffles_D_ge_real)} of {int(r.n_shuffles)} "
            f"shuffles reach the real D of {r.ashman_D:.2f}, up to {r.ashman_D_shuffled_max:.2f})"
            for v, r in vs.iterrows() if r.n_shuffles_D_ge_real > 0]
    ms_txt = ""
    if ms is not None:
        ms_txt = (f"At Mount St. Helens the real D is {ms.ashman_D:.2f}; over {int(ms.n_shuffles)} time shuffles D has "
                  f"a median of {ms.ashman_D_shuffled_med:.2f} and a maximum of {ms.ashman_D_shuffled_max:.2f} "
                  f"({int(ms.n_shuffles_D_ge_real)} of {int(ms.n_shuffles)} at or above the real value; the first "
                  f"build used one shuffle, D {ms.ashman_D_shuffled:.2f}). The zoom contains suspect events: "
                  f"{int(ms.n_review_P1_above)} of its {int(ms.n_above)} events above the local Mc are Mount St Helens "
                  f"deep events of review P1, {int(ms.n_review_P1_in_families)} of them in families. ")
    sw_txt = ("None of the volcano families meets two swarm indicators: they are small bursts of near-equal events."
              if vs.n_swarm.sum() == 0 else
              f"{int(vs.n_swarm.sum())} volcano families meet two swarm indicators "
              f"({', '.join(f'{v} {int(r.n_swarm)}' for v, r in vs.iterrows() if r.n_swarm > 0)}).")
    weak_txt = ("Shuffled catalogs are as bimodal as the real one at " + "; ".join(weak)
                + ", so the test is not meaningful there." if weak else "")
    # events of the v4 zoom with M exactly at its local Mc (the v4 zoom compared M with the unrounded value)
    at4 = {v: int(np.isclose(ev4.M[Z.km(ev4.lon.values, ev4.lat.values, *Z.VOLCANOES[v]) <= r.radius_km],
                             round(r.Mc_local, 2), rtol=0, atol=1e-9).sum())
           for v, r in vl4.set_index("volcano").iterrows()}
    h.append(f"""<p>Zoomed analysis within 20 km with a local Mc (maximum curvature + 0.2, rounded to 0.01 and applied
as in the main run, M &ge; Mc). The v4 zoom compared M with the unrounded value, a float error above the nominal Mc,
and so left out the events with M exactly at Mc ({', '.join(f'{v} {k}' for v, k in at4.items())}); the v5.1 zoom
keeps them. {ms_txt}{sw_txt} {weak_txt} Within 15 km of the summits (regional Mc):</p>""")
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
Mc.</p>
<p><b>Axial is the least robust region.</b> Its background share ranges over the variants from {axr.min():.1f}%
({axr.idxmin()}) to {axr.max():.1f}% ({axr.idxmax()}). The variants that move it most: noM_floor
{piv_bg.loc['AX', 'noM_floor']:.1f}% (M = Mc for the {int((used[used.region == 'AX'].M.isna()).sum()):,} Axial events
without M, label agreement with the main run {100 * piv_ag.loc['AX', 'noM_floor']:.1f}%), tierABC
{piv_bg.loc['AX', 'tierABC']:.1f}%, mc_best {piv_bg.loc['AX', 'mc_best']:.1f}% (Mc {comp.Mc_best['AX']:.2f}
instead of {comp.Mc['AX']:.2f}) and
eta0_shuf5 {piv_bg.loc['AX', 'eta0_shuf5']:.1f}%. With the null test above (a shuffled catalog reproduces
{100 * lr['AX']:.0f}% of the Axial links), the Axial clustered share should be read as a range of about
{100 - axr.max():.0f}-{100 - axr.min():.0f}%, most of it spatial concentration.</p>""")

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
    ic = d["input_commits"]
    fc = next((v["commit"][:8] for k, v in ic.items() if "faults" in k), "?")
    fam_fit = fam[fam.frac_on_faults.notna()]
    n_ax_only = int(fam.frac_on_faults.isna().sum())
    h.append(f"""<p>Join with <code>fault_surfaces_v1_assignment.csv.gz</code> of the faults piece after its review
fixes (142 surfaces, commit {fc} of analysis/v51-faults). An event is on a surface when the faults piece marks it
<code>on_surface_strict</code>: a fit member, or a tier A-C event compatible with exactly one surface and within 2 km
of it, without floor_limited, far_only or deep_pass. The first build also counted every 'assigned' event, of which
45% are tier D and 4,899 lie more than 5 km from their surface; that looser fraction is kept as
<code>frac_on_faults_loose</code>. Axial Seamount was not fitted, so its events are neither on nor off a surface:
fractions leave them out, and the {n_ax_only} families with no member in a fitted region have no value. Half of the
other {len(fam_fit):,} families have at least {100 * fam_fit.frac_on_faults.median():.0f}% of their events on a
surface (swarms {100 * fam_fit[fam_fit.type == 'swarm'].frac_on_faults.median():.0f}%, mainshock-aftershock
{100 * fam_fit[fam_fit.type == 'mainshock-aftershock'].frac_on_faults.median():.0f}%; loose join
{100 * fam_fit.frac_on_faults_loose.median():.0f}%). Rerun with <code>v51_clusters_zbz.py --steps faults</code> and
this report if the surfaces change.</p>""")
    h.append(img(out / "fig_faults.png", "faults"))

    def short_flags(s):
        s = "" if pd.isna(s) else str(s)
        out_ = []
        if "Mount St Helens deep" in s:
            out_.append("review P1: members may be false events")
        if "Brookings" in s:
            out_.append("review P2: depth probably wrong")
        if "unstable" in s:
            out_.append("unstable in detection reruns")
        if "moved a median" in s:
            out_.append("members moved > 10 km in depth from v4")
        if "depth poorly constrained" in s:
            out_.append("depth poorly constrained")
        return "; ".join(out_)

    pfs = pf[pf.n_analysis >= 30].sort_values("pct_swarm", ascending=False).copy()
    pfs["flags_short"] = pfs["flags"].map(short_flags)
    pft = pfs.head(15).copy()
    pft["literature"] = pft.lit_match.fillna("").str.slice(0, 60)
    h.append(tab(pft, ["surface_id", "region", "kind", "stable", "n_analysis", "pct_background", "pct_swarm",
                       "n_families", "n_swarm_families", "swarm_share_top1", "swarm_share_top2", "v0_surfaces",
                       "flags_short", "literature"],
                 rename={"n_analysis": "events &ge; Mc", "v0_surfaces": "v0 surfaces",
                         "n_swarm_families": "swarm families", "swarm_share_top1": "share of swarm events, largest family",
                         "swarm_share_top2": "share, two largest", "flags_short": "flags (faults piece, shortened)",
                         "literature": "literature match (carried from v0, truncated)"}))
    top = pfs[pfs.pct_swarm > 0].head(8)
    conc = pfs[pfs.n_swarm > 0]
    top_txt = "; ".join(
        f"{r.surface_id} ({r.region}, {r.pct_swarm:.0f}% of {int(r.n_analysis)} events, {int(r.n_swarm_families)} "
        f"swarm famil{'y' if r.n_swarm_families == 1 else 'ies'}, the largest giving {100 * r.swarm_share_top1:.0f}% "
        f"of the swarm events{'' if bool(r.stable) else ', unstable surface'})" for r in top.itertuples())
    h.append(f"""<p>Highest swarm shares among the surfaces with 30 or more events above Mc (ranked from the table):
{top_txt}. The swarm events of a surface
are concentrated in few families: over the {len(conc)} surfaces with swarm events, the largest swarm family gives a
median of {100 * conc.swarm_share_top1.median():.0f}% of them and the two largest
{100 * conc.swarm_share_top2.median():.0f}%. The surfaces were also fitted to these same events, so a surface's swarm
share is not an independent property of a fault: it mostly says which one or two swarms built the surface.
{int((~top.stable.astype(bool)).sum())} of these {len(top)} surfaces are unstable in the faults piece's detection
reruns.</p>""")
    h.append(f"""<p><b>Flagged events.</b> The faults piece removed surfaces whose events occur mostly on weekday daytime
(possible unflagged blasts). After its fix, the weekday-daytime members of those surfaces form their own class
({int(day.n):,} events, the first build had 1,137 because it counted every member of the removed surfaces); here
{int(day.in_analysis)} of them are above Mc and {int(day.in_families)} are in {int(day.n_families)} families. They
stay in the analysis because the catalog does not flag them; families holding them carry
<code>n_daytime_screened</code> &gt; 0 ({int((fam.n_daytime_screened > 0).sum())} families). Families with events
of the waveform-review sets carry <code>n_review_P1_msh_deep</code> ({int((fam.n_review_P1_msh_deep > 0).sum())}
families) and <code>n_review_P2_brookings</code> ({int((fam.n_review_P2_brookings > 0).sum())} families); the faults
piece marks v1_ON028 (Mount St Helens deep population) and v1_GO022 (Brookings 2012) as not usable until that review
is done.</p>""")

    # ---------------- proposed decisions
    h.append("<h2>Proposed decisions for the PI</h2><ol>")
    for li in [
        "Run the main analysis on tier A-C events and attach tier D afterwards, as the PI chose for the fault "
        f"surfaces; keep the all-tier run as a variant. Outside Axial, tier D epicentres have median errors of "
        f"{td.h_max_med_tierD_km['MT']:.0f}-{tdo.max():.0f} km, {td.pct_links_tierD_end['all']:.0f}% of the links "
        "have a tier D end, and the tierABC variant changes at most "
        f"{100 * (1 - piv_ag.drop(index='AX')['tierABC'].min()):.1f}% of the labels outside Axial while it removes the "
        "2014 Vancouver Island artefact. (Not applied here: the main run is unchanged from the first build.)",
        "Report the null test (shuffled-to-real link ratio per region) with every clustered share, and give the Axial "
        f"clustered share as a range ({100 - axr.max():.0f}-{100 - axr.min():.0f}%) rather than one number.",
        "Leave the Mount St Helens deep events (review P1) out of the next run if the human review confirms that they "
        "are false events; until then their families are flagged.",
        "Do these on catalog v6.0 (D-B079, NonLinLoc with SSST station terms), which supersedes v5.1, with 3-D distance "
        "restricted to well-constrained depths (d3q) as the main run, and relocate the swarms relatively (double "
        "difference) before interpreting migration or diffusivity."]:
        h.append(f"<li>{li}</li>")
    h.append("</ol>")

    # ---------------- limits
    h.append("<h2>Limits and open issues</h2><ul>")
    for li in [
        f"Completeness changes in space and time. The constant regional Mc keeps {len(a):,} of {n_used:,} events in "
        "the analysis set; volcano microswarms and the Axial events below M 0.65 are only attached afterwards. The Mc "
        f"rule depends on floating-point bin edges in {len(mdiff)} regions ({', '.join(mdiff.index)}).",
        "The main run uses epicentral distance, so slab and crustal events that share an epicentre can link. With "
        "v5.1 depths the d3q variant (3-D only between well-constrained depths) changes few labels, so a 3-D main run "
        "is possible in a next version.",
        f"Tier D epicentres enter the analysis set. Their median h_max is {td.h_max_med_tierD_km['AX']:.1f} km at "
        f"Axial, {td.h_max_med_tierD_km['MT']:.0f} km at the Mendocino triple junction and "
        f"{tdo.min():.0f}-{tdo.max():.0f} km elsewhere ({', '.join(f'{r} {v:.0f}' for r, v in tdo.items())}); the "
        f"five-phase events have {td.h_max_med_five_phase_km['all']:.0f} km. {td.pct_links_tierD_end['all']:.0f}% of "
        f"the links have a tier D end ({td.pct_links_tierD_end['ON']:.0f}% in ON). Links between tier D events "
        "tens of km apart can be artefacts, as the 2014 Vancouver Island foreshock shows.",
        f"Time-shuffled catalogs reproduce {100 * lr.min():.0f}-{100 * lr.max():.0f}% of the links ({100 * lr['AX']:.0f}% "
        "at Axial), and within-month shuffles more. The clustered shares mix triggering with spatial concentration "
        "and slow rate changes.",
        f"Location errors (median h_max {sw10.h_err_med_km.median():.1f} km for tier A-C swarm members) are as large "
        "as most swarms, so "
        "migration and sqrt(t) fits are unresolved for most families. Relative (double-difference) relocation of the "
        "swarms is needed before diffusivities can be interpreted.",
        "Magnitudes: in its own review the mag piece found that its phase19 station-term inversion stops at 3,000 "
        "iterations without converging; converged, ML would move by -0.68 on the Blanco, -0.51 in its Juan de Fuca "
        "region and -0.28 at Gorda, Mendocino and the offshore forearc (onshore +0.01) and agree better with ComCat "
        "offshore. Production M is unchanged (PI decision), so offshore Mc and M1-M2 here inherit that offset. "
        "Offshore ML is 0.3-0.5 above ComCat ML at Mendocino, Blanco and Gorda (mag piece); M1-M2 gaps of "
        "a few tenths drive the swarm and multiplet rules and are close to the ML uncertainty (about 0.2). "
        f"{int(used.M.isna().sum()):,} located earthquakes have no M and cannot be parents.",
        "Swarm criteria are thresholds on continuous properties; families near them switch type between versions "
        f"({100 - _kept_same_type(s4):.0f}% of the kept families changed type).",
        "Rate changes from network changes (OBS deployments 2011-2015, NC/BK pick gap in 2016) are not modelled; Mc is "
        "the worst yearly value per region to limit their effect.",
        f"{int(day.n):,} weekday-daytime members of surfaces the faults piece removed for their timing are kept; they "
        f"may be unflagged blasts ({int(day.in_families)} in {int(day.n_families)} families, flag "
        "n_daytime_screened).",
        f"Mount St Helens deep events (review P1): {int(p1d.in_analysis)} above Mc, {int(p1d.in_families)} in "
        f"{int(p1d.n_families)} families, {nwed} of which start on a Wednesday near local noon. They are probably false "
        "events from the weekly artificial signal on the analog UW channels; pending human review.",
        "Catalog v5.1 is located in model v0.3.1. The PI has adopted model v0.4.0 (v0.3.1 re-inverted with the "
        "re-timed picks, D-B082), v0.4.1 (with a surface floor in the inversion, D-B083) is being built, and catalog "
        "v6.0 (D-B079, NonLinLoc with SSST station terms) will supersede v5.1. These products will be redone on "
        "v6.0."]:
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
         "extent, depth ranges (depth_good), depth_main_any_quality (any depth quality; see depth_main_good), swarm "
         "indicators, topology, v4-style migration, v5.1 migration (rho, Spearman and permutation p, radial speed, "
         "velocity vector and azimuth, vz), D90, D50, envelope exponent, resolvable, style, fault fractions (strict "
         "and loose, Axial left out), flags n_daytime_screened, n_review_P1_msh_deep, n_review_P2_brookings, v4 "
         "status"),
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
        ("faults_v1_cluster_counts.csv, faults_v1_daytime_surfaces_events.csv", "fault surfaces v1 join"),
        ("null_test.csv, null_test_summary.csv", "null test: time-shuffled catalogs at the real thresholds (links, D, "
         "families, swarms)"),
        ("tierD_by_region.csv, mc_edge_check.csv, depth_swarm_vs_ma_families.csv",
         "tier D errors and links; Mc with float and exact bin edges; family-level depth test"),
        ("review_sets_in_clusters.csv, review_P1_families.csv", "waveform-review sets P1-P4 in the clustering; families "
         "with Mount St Helens deep events"),
        ("input_commits.json", "commits of the other pieces' products read by the run")]:
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
