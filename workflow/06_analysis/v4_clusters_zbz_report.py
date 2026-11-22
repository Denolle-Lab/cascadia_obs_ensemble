"""Figures, validation tables and HTML report for the catalog v4 clustering.

Reads the outputs of v4_clusters_zbz.py in data/catalog_v4/clusters_zbz/ and writes
there:
  validation_sequences.csv   known mainshocks and swarms -> family, type, role
  validation_v4_swarms.csv   each v4_swarms.py cluster -> ZBZ family types of its members
  volcano_families.csv       families within 15 km of Cascade volcano summits
  setting_stats.csv          per region: swarm share, topology, productivity, foreshocks
  fig_*.png                  figures
  catalog_v4_clusters_report.html

Run after v4_clusters_zbz.py:
    python workflow/06_analysis/v4_clusters_zbz_report.py
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "utils"))
sys.path.append(str(Path(__file__).resolve().parent))
from paper_style import use, FULL, INK, MUTED  # noqa: E402
from v4_clusters_zbz import (REGIONS, REGION_NAMES, REG_ORDER, gmm1d, load as load_catalog,  # noqa: E402
                             mc_maxc, b_aki, xyz, nn_search, tree_metrics, classify, rupture_cap)
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

OUT = ROOT / "data/catalog_v4/clusters_zbz"
SWARMS = ROOT / "data/catalog_v4/swarms"
TYPE_COLORS = {"mainshock-aftershock": "#0072B2", "swarm": "#D55E00", "multiplet": "#009E73",
               "background": "#9A9A9A"}
VOLCANOES = {"Mount Baker": (-121.813, 48.777), "Glacier Peak": (-121.113, 48.112),
             "Mount Rainier": (-121.760, 46.853), "Mount St. Helens": (-122.180, 46.200),
             "Mount Adams": (-121.490, 46.206), "Mount Hood": (-121.695, 45.374),
             "Mount Jefferson": (-121.800, 44.674), "Three Sisters": (-121.768, 44.103),
             "Newberry": (-121.229, 43.722), "Crater Lake": (-122.110, 42.940),
             "Medicine Lake": (-121.550, 41.590), "Mount Shasta": (-122.193, 41.409),
             "Lassen Peak": (-121.505, 40.492)}
# known events for validation (catalog v4 event_id; ComCat magnitudes)
KNOWN = [
    (262, "2010-01-10 Mw 6.5 Gorda", "mainshock-aftershock"),
    (83695, "2014-03-10 Mw 6.8 Gorda", "mainshock-aftershock"),
    (19960, "2011-09-09 Mw 6.4 Vancouver Island", "mainshock-aftershock"),
    (89617, "2014-04-24 Mw 6.5 Vancouver Island", "mainshock-aftershock"),
    (40500, "2012-11-08 Mw 6.1 Explorer plate", "mainshock-aftershock"),
    (117222, "2015-01-28 Mw 5.7 Mendocino fault zone", "mainshock-aftershock"),
    (150165, "2015-06-01 Mw 5.8, 5.4, 5.9 Blanco triplet", "swarm or multiplet"),
    (27442, "2012-02/04 Mw 5.8 and 6.0 Blanco", "swarm or multiplet"),
]


def km(lon, lat, lon0, lat0):
    return np.hypot((lon - lon0) * 111.195 * np.cos(np.radians(lat0)), (lat - lat0) * 111.195)


def load():
    ev = pd.read_csv(OUT / "events_zbz.csv.gz", parse_dates=["time"])
    fam = pd.read_csv(OUT / "families_zbz.csv", parse_dates=["start", "end"])
    reg = pd.read_csv(OUT / "regions_zbz.csv")
    thr = pd.read_csv(OUT / "thresholds_zbz.csv")
    comp = pd.read_csv(OUT / "completeness_b_df.csv")
    sens = pd.read_csv(OUT / "sensitivity_zbz.csv")
    sh = pd.read_csv(OUT / "shuffle_test.csv")
    return ev, fam, reg, thr, comp, sens, sh


# ---------------------------------------------------------------- validation
def validate(ev, fam):
    rows = []
    fi = fam.set_index("family_id")
    for eid, label, expect in KNOWN:
        e = ev[ev.event_id == eid].iloc[0]
        d = dict(event_id=eid, label=label, expected=expect, region=e.region, M=e.M, role=e.role,
                 family_id=int(e.family_id))
        if e.family_id > 0:
            f = fi.loc[e.family_id]
            nall = int((ev.family_id == e.family_id).sum())
            d.update(type=f.type, n=int(f.n), n_all=nall, Mmax=f.Mmax, M2=f.M2,
                     duration90_days=round(f.duration90_days, 1), extent_km=round(f.extent_km, 1),
                     mean_leaf_depth=round(f.mean_leaf_depth, 2), regions=f.regions)
        rows.append(d)
    # Axial 2015 eruption: families with events between 2015-04-23 and 2015-05-31 in AX
    ax = ev[(ev.region == "AX") & ev.in_analysis & (ev.time >= "2015-04-23") & (ev.time < "2015-05-31")]
    big = ax[ax.family_id > 0].family_id.value_counts().head(1)
    for F, nn in big.items():
        f = fi.loc[F]
        rows.append(dict(event_id=int(f.mainshock_id), label="2015-04-24 Axial Seamount eruption onset",
                         expected="swarm", region="AX", M=f.Mmax, role="largest event", family_id=int(F),
                         type=f.type, n=int(f.n), n_all=int((ev.family_id == F).sum()), Mmax=f.Mmax, M2=f.M2,
                         duration90_days=round(f.duration90_days, 1), extent_km=round(f.extent_km, 1),
                         mean_leaf_depth=round(f.mean_leaf_depth, 2), regions=f.regions,
                         note=f"start {f.start}; {100 * nn / len(ax):.0f}% of AX analysis events "
                              f"2015-04-23 to 05-31 in this family"))
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "validation_sequences.csv", index=False)

    # comparison with v4_swarms (DBSCAN) clusters
    sc = pd.read_csv(SWARMS / "v4_clusters.csv")
    sm = pd.read_csv(SWARMS / "v4_cluster_members.csv")
    e = sm.merge(ev[["event_id", "family_id", "role", "in_analysis"]], on="event_id", how="left")
    e = e.merge(fam[["family_id", "type"]], on="family_id", how="left")
    e["type"] = e["type"].fillna("background")
    out = []
    for c, g in e.groupby("cluster"):
        vc = g.type.value_counts()
        fc = g.family_id[g.family_id > 0].value_counts()
        out.append(dict(cluster=c, n=len(g), frac_in_families=round((g.family_id > 0).mean(), 3),
                        dominant_type=vc.index[0], n_families=len(fc),
                        frac_in_main_family=round(fc.iloc[0] / len(g), 3) if len(fc) else 0.0,
                        main_family=int(fc.index[0]) if len(fc) else -1))
    vs = sc.merge(pd.DataFrame(out), on=["cluster"], suffixes=("_dbscan", ""))
    # type of the main ZBZ family
    vs["main_family_type"] = vs.main_family.map(fam.set_index("family_id").type).fillna("background")
    vs.to_csv(OUT / "validation_v4_swarms.csv", index=False)

    # volcano families
    vrows = []
    for v, (lo, la) in VOLCANOES.items():
        f = fam[km(fam.lon, fam.lat, lo, la) <= 15]
        e2 = ev[(km(ev.lon, ev.lat, lo, la) <= 15) & ev.in_analysis]
        if len(e2) == 0:
            continue
        vrows.append(dict(volcano=v, n_events_15km=len(e2), pct_background=round(100 * (e2.role == "single").mean(), 1),
                          n_families=len(f), n_swarm=int((f.type == "swarm").sum()),
                          n_MA=int((f.type == "mainshock-aftershock").sum()),
                          n_multiplet=int((f.type == "multiplet").sum()),
                          largest_family_n=int(f.n.max()) if len(f) else 0,
                          largest_family_type=f.sort_values("n").type.iloc[-1] if len(f) else ""))
    vo = pd.DataFrame(vrows)
    vo.to_csv(OUT / "volcano_families.csv", index=False)
    return val, vs, vo


def volcano_local(radius=20.0, names=("Mount St. Helens", "Mount Hood", "Mount Rainier"), workers=8):
    """Zoomed NN analysis around volcanoes with a local Mc (the ON-region Mc of 1.75
    removes almost all volcano microseismicity). Same method as the main run:
    local Mc (max curvature + 0.2), local b, df 1.6, mixture threshold, distance cap,
    classification; one time-shuffled catalog for the bimodality check."""
    cat = load_catalog()
    rows, frows = [], []
    for v in names:
        lo, la = VOLCANOES[v]
        sub = cat[km(cat.lon.values, cat.lat.values, lo, la) <= radius]
        mc = mc_maxc(sub.M.values)
        b, _, _ = b_aki(sub.M.values, mc)
        A = sub[sub.M >= mc].sort_values("t_yr").reset_index(drop=True)
        n = len(A)
        if n < 50:
            continue
        u = xyz(A.lat.values, A.lon.values)
        t, z, m = A.t_yr.values, A.depth.values, A.M.values
        zero = np.zeros(n, int)
        cap = rupture_cap(m)
        res = {}
        for tag, tt in [("real", t), ("shuffled", np.sort(np.random.default_rng(5).permutation(t)))]:
            uu = u if tag == "real" else u[np.random.default_rng(6).permutation(n)]
            res[tag] = nn_search(tt, uu, z, np.full(n, 1.6), tt, uu, z, b * m, 0.5, False, True, workers,
                                 zero, zero, cap)
        par, o = res["real"]
        g = gmm1d(o[:, 0])
        gs = gmm1d(res["shuffled"][1][:, 0])
        linked = np.isfinite(o[:, 0]) & (o[:, 0] < g["log_eta0"]) & (o[:, 3] <= cap[np.maximum(par, 0)])
        ii = np.where(linked)[0]
        _, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, par[ii])), shape=(n, n)), directed=False)
        sz = np.bincount(lab)
        types = []
        for c in np.unique(lab[sz[lab] >= 2]):
            mem = np.where(lab == c)[0]
            tm = tree_metrics(mem, par, linked, m)
            ms = np.sort(m[mem])[::-1]
            typ, _ = classify(len(mem), ms[0] - ms[1], tm["imax_local"] / len(mem), tm)
            types.append(typ)
            frows.append(dict(volcano=v, n=len(mem), Mmax=ms[0], M2=ms[1], type=typ,
                              start=A.time.values[mem[0]],
                              duration_days=(t[mem[-1]] - t[mem[0]]) * 365.25,
                              mean_leaf_depth=tm["mean_leaf_depth"]))
        ty = pd.Series(types, dtype=object)
        rows.append(dict(volcano=v, radius_km=radius, n_all=len(sub), Mc_local=mc, b_local=round(b, 2), n_above=n,
                         log_eta0=round(g["log_eta0"], 2), ashman_D=round(g["ashman_D"], 2),
                         ashman_D_shuffled=round(gs["ashman_D"], 2),
                         pct_background=round(100 * (sz[lab] == 1).mean(), 1),
                         n_families=len(types), n_swarm=int((ty == "swarm").sum()),
                         n_MA=int((ty == "mainshock-aftershock").sum()), n_multiplet=int((ty == "multiplet").sum())))
    vl = pd.DataFrame(rows)
    vl.to_csv(OUT / "volcano_local_zbz.csv", index=False)
    pd.DataFrame(frows).to_csv(OUT / "volcano_local_families.csv", index=False, float_format="%.3f")
    return vl, pd.DataFrame(frows)


def setting_stats(ev, fam, comp):
    rows = []
    mc = comp.set_index("region").Mc
    for r in REG_ORDER:
        f = fam[fam.region == r]
        f5 = f[f.n >= 5]
        ma = f[f.type == "mainshock-aftershock"]
        # productivity: direct linked offspring (above Mc, any region) of events 1-2 units above Mc
        ea = ev[ev.in_analysis]
        off = ea[ea.linked].groupby("parent_id").size()
        band = ea[(ea.region == r) & (ea.M - mc[r] >= 1.0) & (ea.M - mc[r] < 2.0)]
        e_as = band.event_id.map(off).fillna(0)
        e_fs = ev[ev.in_analysis & (ev.role == "foreshock")].groupby("family_id").size()
        # all events >= Mc+1 in region: fraction that are singles (no fore/aftershocks above Mc)
        e = ev[ev.in_analysis & (ev.region == r)]
        big = e[e.M >= mc[r] + 1.0]
        rows.append(dict(
            region=r, n_fam_ge5=len(f5), pct_swarm_of_ge5=round(100 * (f5.type == "swarm").mean(), 1) if len(f5) else np.nan,
            median_leaf_depth_ge5=round(f5.mean_leaf_depth.median(), 2) if len(f5) else np.nan,
            n_events_Mc1to2=len(band),
            mean_offspring_Mc1to2=round(e_as.mean(), 2) if len(band) else np.nan,
            pct_Mc1to2_no_offspring=round(100 * (e_as == 0).mean(), 1) if len(band) else np.nan,
            n_events_ge_Mc1=len(big), pct_ge_Mc1_single=round(100 * (big.role == "single").mean(), 1) if len(big) else np.nan,
            pct_MA_with_foreshocks=round(100 * (ma.family_id.map(e_fs).fillna(0) > 0).mean(), 1) if len(ma) else np.nan,
            pct_fam_migrating_ge10=round(100 * f[f.n >= 10].migrating.mean(), 1) if (f.n >= 10).any() else np.nan,
            n_fam_ge10=int((f.n >= 10).sum())))
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "setting_stats.csv", index=False)
    return st


# ---------------------------------------------------------------- figures
def fig_tr(ev, thr, out):
    sh = np.load(OUT / "shuffle_eta_seed0.npz")
    fig, axs = plt.subplots(3, 3, figsize=(FULL, FULL * 0.95), sharex=True, sharey=True)
    t = thr.set_index("region")
    for ax, r in zip(axs.flat, REG_ORDER):
        e = ev[ev.in_analysis & (ev.region == r)].dropna(subset=["log_T", "log_R"])
        h, xe, ye = np.histogram2d(e.log_T, e.log_R, bins=[np.arange(-10, 1.01, 0.2), np.arange(-7, 4.01, 0.2)])
        h = h / h.sum() / 0.04
        ax.pcolormesh(xe, ye, np.ma.masked_equal(h.T, 0), cmap="Greys", shading="auto",
                      vmax=np.quantile(h[h > 0], 0.98))
        k = sh["region"] == r
        hs, _, _ = np.histogram2d(sh["log_T"][k], sh["log_R"][k], bins=[xe, ye])
        if hs.sum():
            from scipy.ndimage import gaussian_filter
            hs = gaussian_filter(hs / hs.sum() / 0.04, 1.5)
            ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), hs.T,
                       levels=np.quantile(hs[hs > 0], [0.8, 0.95]), colors="#0072B2", linewidths=0.6)
        x = np.array([-10, 1])
        ax.plot(x, t.loc[r, "log_eta0"] - x, color="#D55E00", lw=0.9)
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
    fig.savefig(out / "fig_TR_density.png", dpi=200)
    plt.close(fig)


def fig_eta(ev, thr, out):
    sh = np.load(OUT / "shuffle_eta_seed0.npz")
    t = thr.set_index("region")
    fig, axs = plt.subplots(3, 3, figsize=(FULL, FULL * 0.7), sharex=True)
    bins = np.arange(-11, 1.01, 0.2)
    for ax, r in zip(axs.flat, REG_ORDER):
        le = ev[ev.in_analysis & (ev.region == r)].log_eta.dropna().values
        ax.hist(le, bins=bins, density=True, color="#BBBBBB", label="catalog")
        k = sh["region"] == r
        ax.hist(sh["log_eta"][k][np.isfinite(sh["log_eta"][k])], bins=bins, density=True, histtype="step",
                color="#0072B2", lw=0.8, label="time-shuffled")
        g = t.loc[r]
        x = np.linspace(-11, 1, 400)
        for w, mu, sd in [(g.w1, g.mu1, g.sd1), (g.w2, g.mu2, g.sd2)]:
            ax.plot(x, w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sd) ** 2), color=INK, lw=0.6)
        ax.axvline(g.log_eta0, color="#D55E00", lw=0.9, label="eta0 (mixture)")
        ax.axvline(g.log_eta0_shuf5, color="#D55E00", lw=0.7, ls=":", label="eta0 (5% shuffled)")
        ax.set_title(f"{r}  D={g.ashman_D:.1f}", fontsize=7)
    axs[0, 0].legend(loc="upper left", fontsize=5.5)
    for ax in axs[-1]:
        ax.set_xlabel("log$_{10}$ eta")
    fig.tight_layout()
    fig.savefig(out / "fig_eta_hist.png", dpi=200)
    plt.close(fig)


def fig_map(ev, fam, out):
    import pygmt
    fig = pygmt.Figure()
    region = [-131.0, -120.5, 39.5, 51.5]
    pygmt.config(FONT_ANNOT_PRIMARY="7p", FONT_LABEL="7p", MAP_FRAME_TYPE="plain")
    fig.coast(region=region, projection="M12c", land="#EEEEEE", water="white", shorelines="0.25p,#888888",
              borders="1/0.25p,#BBBBBB", frame=["WSne", "af"])
    for key, _, (a, b, c, d) in REGIONS:
        fig.plot(x=[a, b, b, a, a], y=[c, c, d, d, c], pen="0.3p,#666666,--")
        fig.text(x=a + 0.05, y=d - 0.08, text=key, font="6p,Helvetica,#444444", justify="TL")
    bg = ev[ev.in_analysis & (ev.role == "single")]
    fig.plot(x=bg.lon, y=bg.lat, style="c0.03c", fill="#AAAAAA")
    f = fam[fam.n >= 5].sort_values("n", ascending=False)
    for typ in ["mainshock-aftershock", "swarm"]:
        g = f[f.type == typ]
        fig.plot(x=g.lon_main, y=g.lat_main, size=0.08 + 0.09 * np.log10(g.n.values) ** 1.3 * 1.3,
                 style="cc", pen=f"0.5p,{TYPE_COLORS[typ]}", label=f"{typ} (n>=5)+S0.25c")
    fig.legend(position="JBL+jBL+o0.15c", box="+gwhite+p0.3p,#999999")
    fig.savefig(out / "fig_family_map.png", dpi=250)


def seq_panel(axm, axd, ev, fam, F, title):
    e = ev[ev.family_id == F].sort_values("time")
    f = fam.set_index("family_id").loc[F]
    t0 = e.time.min()
    td = (e.time - t0).dt.total_seconds() / 86400
    mainrow = e[e.event_id == f.mainshock_id].iloc[0]
    d = km(e.lon.values, e.lat.values, mainrow.lon, mainrow.lat)
    col = np.where(e.in_analysis, TYPE_COLORS[f.type], "#BBBBBB")
    axm.scatter(td, e.M.fillna(-0.5), s=3, c=col, lw=0)
    axm.set_title(f"{title}\n{f.type}, n={int(f.n)} (+{len(e) - int(f.n)} <Mc), "
                  f"M1-M2={f.dM:.2f}", fontsize=5.8)
    axm.set_ylabel("M (missing: -0.5)", fontsize=6.5)
    axd.scatter(td, d, s=3, c=col, lw=0)
    axd.set_ylabel("km from largest", fontsize=6.5)
    axd.set_xlabel("days from first event", fontsize=6.5, labelpad=1)
    if td.max() > 30:
        for a in (axm, axd):
            a.set_xscale("symlog", linthresh=1.0)


def fig_examples(ev, fam, val, out):
    pick = []
    for _, r in val.iterrows():
        if r.family_id > 0:
            pick.append((int(r.family_id), r.label))
    # add the largest swarm-type families in GO, PU, ON (if any)
    for r in ["GO", "PU", "ON", "BL"]:
        f = fam[(fam.region == r) & (fam.type == "swarm")].sort_values("n", ascending=False)
        if len(f) and int(f.family_id.iloc[0]) not in [p[0] for p in pick]:
            pick.append((int(f.family_id.iloc[0]), f"largest {r} swarm, {pd.Timestamp(f.start.iloc[0]):%Y-%m-%d}"))
    pick = pick[:12]
    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axs = plt.subplots(2 * nrow, ncol, figsize=(FULL, 2.6 * nrow + 0.3),
                            gridspec_kw=dict(hspace=1.0, wspace=0.35))
    axs = np.atleast_2d(axs)
    for k, (F, lab) in enumerate(pick):
        i, j = divmod(k, ncol)
        seq_panel(axs[2 * i, j], axs[2 * i + 1, j], ev, fam, F, lab)
    for k in range(len(pick), nrow * ncol):
        i, j = divmod(k, ncol)
        axs[2 * i, j].axis("off")
        axs[2 * i + 1, j].axis("off")
    fig.savefig(out / "fig_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_topology(fam, st, out):
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
    axs[2].bar(x, st.pct_swarm_of_ge5, color=TYPE_COLORS["swarm"])
    for xi, n in zip(x, st.n_fam_ge5):
        axs[2].text(xi, 1, str(n), ha="center", fontsize=5.5, color="white")
    axs[2].set_xticks(x, REG_ORDER)
    axs[2].set_ylabel("% swarms among families n>=5")
    for a, l in zip(axs, "abc"):
        a.set_title(l, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_topology.png", dpi=200)
    plt.close(fig)


def fig_sens(sens, out):
    vs = list(dict.fromkeys(sens.variant))
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.5))
    x = np.arange(len(REG_ORDER))
    cmap = plt.get_cmap("tab10")
    for k, v in enumerate(vs):
        s = sens[sens.variant == v].set_index("region").reindex(REG_ORDER)
        kw = dict(color=INK, s=18, marker="D", zorder=5) if v == "main" else dict(color=cmap(k % 10), s=8)
        axs[0].scatter(x + (k - len(vs) / 2) * 0.06, s.pct_background, label=v, **kw)
        axs[1].scatter(x + (k - len(vs) / 2) * 0.06, s.n_swarm / s.n_families.clip(lower=1) * 100, **kw)
    for a in axs:
        a.set_xticks(x, REG_ORDER)
    axs[0].set_ylabel("% background (above Mc)")
    axs[1].set_ylabel("% swarms among all families")
    axs[0].legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_sensitivity.png", dpi=200)
    plt.close(fig)


def fig_faults(pf, out):
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
    ax.set_xticks(x, p.surface_id, rotation=90, fontsize=4.5)
    ax.set_ylabel("% of events above Mc")
    ax.set_xlim(-0.6, len(p) - 0.4)
    ax.legend(ncol=4, fontsize=6, loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    fig.savefig(out / "fig_faults.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-map", action="store_true")
    a = ap.parse_args()
    use()
    ev, fam, reg, thr, comp, sens, sh = load()
    val, vs, vo = validate(ev, fam)
    st = setting_stats(ev, fam, comp)
    pf = pd.read_csv(OUT / "faults_v0_cluster_counts.csv")
    vl, vlf = volcano_local()
    print(vl.to_string())
    print(val.to_string())
    print(vo.to_string())
    print(st.to_string())
    fig_tr(ev, thr, OUT)
    fig_eta(ev, thr, OUT)
    if not a.no_map:
        fig_map(ev, fam, OUT)
    fig_examples(ev, fam, val, OUT)
    fig_topology(fam, st, OUT)
    fig_sens(sens, OUT)
    fig_faults(pf, OUT)
    write_html(OUT, ev, fam, reg, thr, comp, sens, sh, val, vs, vo, st, pf, vl, vlf)




# ---------------------------------------------------------------- HTML
CSS = """
:root{--bg:#ffffff;--ink:#222;--muted:#666;--rule:#ddd;--head:#f4f4f4}
@media (prefers-color-scheme: dark){:root{--bg:#1b1b1b;--ink:#e6e6e6;--muted:#aaa;--rule:#444;--head:#2a2a2a}
 img{background:#fff}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
 max-width:1050px;margin:0 auto;padding:16px}
h1{font-size:1.6em;margin-bottom:.2em} h2{margin-top:1.8em;border-bottom:1px solid var(--rule)} h3{margin-top:1.3em}
table{border-collapse:collapse;font-size:12.5px;margin:.6em 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--rule);padding:3px 6px;text-align:right;white-space:nowrap}
th{background:var(--head)} td:first-child,th:first-child{text-align:left}
img{max-width:100%;height:auto;border:1px solid var(--rule)}
.muted{color:var(--muted);font-size:.92em} code{font-size:.92em}
"""


def tab(df, cols=None, rename=None, fmt=None):
    d = df[cols] if cols else df
    if rename:
        d = d.rename(columns=rename)
    return d.to_html(index=False, border=0, na_rep="", float_format=fmt or (lambda x: f"{x:.2f}"))


REFS = [
    ("Aki 1965", "Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits. Bull. Earthq. Res. Inst. 43, 237-239.", None),
    ("Baiesi & Paczuski 2004", "Baiesi, M., & Paczuski, M. (2004). Scale-free networks of earthquakes and aftershocks. Phys. Rev. E 69, 066106.", "10.1103/PhysRevE.69.066106"),
    ("Boettcher & McGuire 2009", "Boettcher, M. S., & McGuire, J. J. (2009). Scaling relations for seismic cycles on mid-ocean ridge transform faults. GRL 36, L21301.", "10.1029/2009GL040115"),
    ("Bohnenstiehl et al. 2002", "Bohnenstiehl, D. R., Tolstoy, M., Dziak, R. P., Fox, C. G., & Smith, D. K. (2002). Aftershock sequences in the mid-ocean ridge environment: an analysis using hydroacoustic data. Tectonophysics 354, 49-70.", "10.1016/S0040-1951(02)00289-5"),
    ("Hill 1977", "Hill, D. P. (1977). A model for earthquake swarms. JGR 82, 1347-1352.", "10.1029/JB082i008p01347"),
    ("Hooft et al. 2010", "Hooft, E. E. E., et al. (2010). A seismic swarm and regional hydrothermal and hydrologic perturbations: The northern Endeavour segment, February 2005. G3 11, Q12015.", "10.1029/2010GC003264"),
    ("Hsu et al. 2026", "Hsu, Y.-F., Martínez-Garzón, P., Trugman, D. T., & Ben-Zion, Y. (2026). Earthquake clustering and stress inversion in the Mendocino Triple Junction region. GJI 245(3), ggag089. (first author's initials not checked)", "10.1093/gji/ggag089"),
    ("Jones & Malone 2005", "Jones, J., & Malone, S. D. (2005). Mount Hood earthquake activity: Volcanic or tectonic origins? BSSA 95(3), 818-832.", "10.1785/0120040019"),
    ("Kuna et al. 2019", "Kuna, V. M., Nábělek, J. L., & Braunmiller, J. (2019). Mode of slip and crust-mantle interaction at oceanic transform faults. Nat. Geosci. 12, 138-142.", "10.1038/s41561-018-0287-1"),
    ("Lange et al. 2026", "Lange, D., Ren, Y., & Grevemeyer, I. (2026). Seismicity, repeating earthquakes, and tomographic imaging of the Blanco Transform Fault System, Northeast Pacific. JGR Solid Earth 131(5), e2025JB031584.", "10.1029/2025JB031584"),
    ("Martínez-Garzón et al. 2018", "Martínez-Garzón, P., Zaliapin, I., Ben-Zion, Y., Kwiatek, G., & Bohnhoff, M. (2018). Comparative study of earthquake clustering in relation to hydraulic activities at geothermal fields in California. JGR 123, 4041-4062.", "10.1029/2017JB014972"),
    ("McGuire et al. 2005", "McGuire, J. J., Boettcher, M. S., & Jordan, T. H. (2005). Foreshock sequences and short-term earthquake predictability on East Pacific Rise transform faults. Nature 434, 457-461.", "10.1038/nature03377"),
    ("Morton et al. 2023", "Morton, E. A., Bilek, S. L., & Rowe, C. A. (2023). Cascadia subduction zone fault heterogeneities from newly detected small magnitude earthquakes. JGR 128, e2023JB026607.", "10.1029/2023JB026607"),
    ("Ren et al. 2023", "Ren, Y., Lange, D., & Grevemeyer, I. (2023). Seismotectonics of the Blanco Transform Fault System, Northeast Pacific: Evidence for an immature plate boundary. JGR 128, e2022JB026045.", "10.1029/2022JB026045"),
    ("Roland & McGuire 2009", "Roland, E., & McGuire, J. J. (2009). Earthquake swarms on transform faults. GJI 178(3), 1677-1690.", "10.1111/j.1365-246X.2009.04214.x"),
    ("Shearer 2012", "Shearer, P. M. (2012). Self-similar earthquake triggering, Båth's law, and foreshock/aftershock magnitudes. JGR 117, B06310.", "10.1029/2011JB008957"),
    ("Skoumal et al. 2016", "Skoumal, R. J., Brudzinski, M. R., & Currie, B. S. (2016). An efficient repeating signal detector to investigate earthquake swarms. JGR 121, 5880-5897.", "10.1002/2016JB012981"),
    ("Stone et al. 2018", "Stone, I., Vidale, J. E., Han, S., & Roland, E. (2018). Catalog of offshore seismicity in Cascadia. JGR 123, 641-652.", "10.1002/2017JB014966"),
    ("Tolstoy et al. 2006", "Tolstoy, M., et al. (2006). A sea-floor spreading event captured by seismometers. Science 314, 1920-1922.", "10.1126/science.1133950"),
    ("Vidale & Shearer 2006", "Vidale, J. E., & Shearer, P. M. (2006). A survey of 71 earthquake bursts across southern California: Exploring the role of pore fluid pressure fluctuations and aseismic slip as drivers. JGR 111, B05312.", "10.1029/2005JB004034"),
    ("Wells & Coppersmith 1994", "Wells, D. L., & Coppersmith, K. J. (1994). New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement. BSSA 84, 974-1002.", None),
    ("Wilcock et al. 2016", "Wilcock, W. S. D., et al. (2016). Seismic constraints on caldera dynamics from the 2015 Axial Seamount eruption. Science 354, 1395-1399.", "10.1126/science.aah5563"),
    ("Woessner & Wiemer 2005", "Woessner, J., & Wiemer, S. (2005). Assessing the quality of earthquake catalogues: estimating the magnitude of completeness and its uncertainty. BSSA 95, 684-698.", None),
    ("Zaliapin et al. 2008", "Zaliapin, I., Gabrielov, A., Keilis-Borok, V., & Wong, H. (2008). Clustering analysis of seismicity and aftershock identification. PRL 101, 018501.", "10.1103/PhysRevLett.101.018501"),
    ("Zaliapin & Ben-Zion 2013a", "Zaliapin, I., & Ben-Zion, Y. (2013). Earthquake clusters in southern California I: Identification and stability. JGR 118, 2847-2864.", "10.1002/jgrb.50179"),
    ("Zaliapin & Ben-Zion 2013b", "Zaliapin, I., & Ben-Zion, Y. (2013). Earthquake clusters in southern California II: Classification and relation to physical properties of the crust. JGR 118, 2865-2877.", "10.1002/jgrb.50178"),
    ("Zaliapin & Ben-Zion 2016", "Zaliapin, I., & Ben-Zion, Y. (2016). A global classification and characterization of earthquake clusters. GJI 207(1), 608-634.", "10.1093/gji/ggw300"),
    ("Zaliapin & Ben-Zion 2016b", "Zaliapin, I., & Ben-Zion, Y. (2016). Discriminating characteristics of tectonic and human-induced seismicity. BSSA 106(3), 846-859.", "10.1785/0120150211"),
    ("Zaliapin & Ben-Zion 2020", "Zaliapin, I., & Ben-Zion, Y. (2020). Earthquake declustering using the nearest-neighbor approach in space-time-magnitude domain. JGR 125, e2018JB017120.", "10.1029/2018JB017120"),
]


def write_html(out, ev, fam, reg, thr, comp, sens, sh, val, vs, vo, st, pf, vl, vlf):
    a = ev[ev.in_analysis]
    shm = sh.groupby("region").mean(numeric_only=True).reindex(REG_ORDER)
    summ = reg.copy()
    summ["shuffled_linked_pct"] = summ.region.map(100 * shm.frac_linked).round(1)
    summ["ashman_D_shuffled"] = summ.region.map(shm.ashman_D).round(2)
    summ["name"] = summ.region.map(REGION_NAMES)
    roles_all = ev.role.value_counts()
    vsum = vs.copy()
    above = ev.set_index("event_id").in_analysis
    sm = pd.read_csv(SWARMS / "v4_cluster_members.csv")
    sm["above"] = sm.event_id.map(above)
    sm["fam"] = sm.event_id.map(ev.set_index("event_id").family_id)
    sm = sm.merge(vs[["cluster", "region", "volcano"]], on="cluster")
    sm["group"] = sm.volcano.fillna(sm.region)
    vcomp = sm.groupby("group").agg(members=("event_id", "size"), pct_above_Mc=("above", "mean"))
    vcomp["pct_above_Mc"] = (100 * vcomp.pct_above_Mc).round(1)
    vcomp["pct_of_those_clustered_by_ZBZ"] = sm[sm.above.astype(bool)].groupby("group").fam.apply(
        lambda x: round(100 * (x > 0).mean(), 1))
    vcomp = vcomp.reset_index()
    pfs = pf[pf.n_analysis >= 30].sort_values("pct_swarm", ascending=False)

    h = []
    h.append(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>Catalog v4 clusters</title><style>{CSS}</style>"
             f"</head><body>")
    h.append("<h1>Earthquake clusters in catalog v4 (nearest-neighbor method)</h1>")
    h.append("<p class='muted'>Cascadia catalog v4, 2010-2016, 172,340 events. Method of Zaliapin &amp; Ben-Zion "
             "(2013a, 2016). Code: <code>workflow/06_analysis/v4_clusters_zbz.py</code> (clustering) and "
             "<code>v4_clusters_zbz_report.py</code> (figures, validation, this page). Branch "
             "<code>cluster/zbz</code>, 2026-11-22.</p>")

    # ---- summary
    h.append("<h2>Summary</h2>")
    h.append(f"""<p>We linked every event to its nearest earlier neighbor in space, time and magnitude and cut the
links at a threshold set separately in each of nine regions. Above the regional completeness magnitude Mc
there are {len(a):,} events; {100 * (a.role == 'single').mean():.0f}% of them are background singles and the
rest fall in {len(fam):,} families: {(fam.type == 'mainshock-aftershock').sum()} mainshock-aftershock
families, {(fam.type == 'swarm').sum()} swarms and {(fam.type == 'multiplet').sum()} doublets or small
multiplets. Events below Mc or without a magnitude (144,785) were attached to these families afterwards when
their nearest above-Mc neighbor passed the same test; 17% of the whole catalog ends up in a family.</p>
<p>The main results:</p>
<ol>
<li>All nine regions show the two modes in the (T, R) plane. Time-shuffled catalogs lose them: the
bimodality index D (Ashman) drops from 1.6-3.4 to 0.3-1.6.</li>
<li>Background share splits by tectonic setting: about 50% at Axial, Endeavour-Explorer and the Blanco,
58-61% in the Gorda plate and at the Mendocino triple junction, 68-84% in the forearc, Puget Lowland and the
rest of the onshore. Ridge and transform families are mostly swarms (61-66% of families with 5 or more
events) with deep, chain-like trees (median mean leaf depth 2.5-3.0). At the Mendocino triple junction only
31% are swarms and the trees are shallow and bushy (median 1.33). This is the heat-flow ordering of
Zaliapin &amp; Ben-Zion (2013b, 2016).</li>
<li>All six large tectonic mainshocks we checked (2010 Mw 6.5 and 2014 Mw 6.8 Gorda, 2011 Mw 6.4 and 2014
Mw 6.5 Vancouver Island, 2012 Mw 6.1 Explorer plate, 2015 Mw 5.7 Mendocino fault zone) come out as
mainshock-aftershock families. The 2015 Axial eruption comes out as one 632-event swarm starting
2015-04-24 05:11 UTC.</li>
<li>The largest Blanco events behave like oceanic-transform events elsewhere (McGuire et al. 2005; Roland
&amp; McGuire 2009; Boettcher &amp; McGuire 2009): the 2012 Mw 5.8/6.0 pair and the 2015 Mw 5.8/5.4/5.9
triplet sit in swarms with M1-M2 near 0.1, and 4 of 21 Blanco M&ge;5 events have no linked event above
M 2.45.</li>
<li>At the Mendocino triple junction the events 1-2 units above Mc have 0.40 linked offspring on average,
against 0.85-1.8 at Axial, Endeavour, the Gorda plate and the Blanco; 76% have none. The two Mw 6.5-6.8
sequences hold 2,097 of the 3,272 clustered events above Mc. This fits the high background and low productivity found by
Hsu et al. (2026) with the same method on a different catalog.</li>
<li>The background/clustered label of an event is stable: 84-99% of events keep it when we change df (1.3,
2.0, regional estimate), use 3-D distance, raise Mc by 0.3 or use yearly Mc. The exception is the
threshold: a conservative eta0 that admits only 5% of links from shuffled catalogs raises the background
share by 1-32 points, most at Axial (51% to 82%).</li>
</ol>""")

    # ---- table
    h.append("<h2>Results by region</h2>")
    h.append(tab(summ, ["region", "name", "n_analysis", "Mc", "b", "log_eta0", "ashman_D", "ashman_D_shuffled",
                        "shuffled_linked_pct", "pct_background", "pct_MA", "pct_swarm", "pct_multiplet",
                        "n_MA", "n_MA_ge5", "n_swarm", "n_multiplet"],
                 rename={"n_analysis": "events &ge; Mc", "log_eta0": "log eta0", "ashman_D": "D",
                         "ashman_D_shuffled": "D shuffled", "shuffled_linked_pct": "% linked, shuffled",
                         "pct_background": "% background", "pct_MA": "% in MA", "pct_swarm": "% in swarms",
                         "pct_multiplet": "% in multiplets", "n_MA": "MA families", "n_MA_ge5": "MA n&ge;5",
                         "n_swarm": "swarms", "n_multiplet": "multiplets"}).replace("&amp;", "&"))
    h.append("<p class='muted'>Percentages are of events above Mc in the region. MA = mainshock-aftershock. "
             "A family belongs to the region of its largest event; swarms are all of size 5 or more by "
             "definition. D: Ashman bimodality index of the two-Gaussian fit to log eta (D&gt;2 means clean "
             "separation). '% linked, shuffled': share of events in time-shuffled catalogs that pass the real "
             "threshold, a false-link rate.</p>")
    h.append("<img src='fig_family_map.png' alt='Map of families'><p class='muted'>Families with 5 or more "
             "events above Mc at their largest event, sized by log n; grey dots are background events above "
             "Mc. Dashed boxes: regions.</p>")

    # ---- data and choices
    h.append("<h2>Data and choices</h2>")
    h.append("""<p><b>Catalog and magnitudes.</b> Catalog v4 (172,340 events) with the Route-A magnitudes
(ML calibrated to ComCat; Mw from moment tensors above 4.5). The catalog's own magnitude column is not used
because it is biased high offshore. 41,573 events (24%) have no magnitude. They are mostly small and poorly
recorded. They are left out of the clustering and attached afterwards (see below); the variant
<code>noM_floor</code> gives them M = Mc instead.</p>
<p><b>Regions.</b> The nine project boxes, most specific first (AX, MT, PU, EN, BL, GO, WA, OR; ON for the
rest). ON is mixed: it holds the onshore crust and slab, the Cascade volcanoes, northern Vancouver Island and
the Juan de Fuca Ridge segments between the Blanco and Axial and between Axial and Endeavour. Mc, b and the
threshold are regional. The parent search runs over the whole catalog, but a parent from another region is
allowed only within the distance cap below; that keeps the 2014 Vancouver Island aftershocks (69 of 97 in
the EN box) in one family.</p>
<p><b>Completeness.</b> Mc by maximum curvature + 0.2 for each region and year with at least 200 events;
the region Mc is the largest yearly value, so Mc is constant over 2010-2016. The OBS years (2011-2015)
reach lower magnitudes offshore, and NC/BK picks are almost absent south of 42&deg;N in 2010-2011 and 2016,
so a constant Mc throws away small events in good years to keep the catalog homogeneous. Sparse years
(fewer than 200 events, e.g. Blanco 2010, 2011 and 2016) do not set Mc and are less complete than the rest.
The b-value-stability Mc (Woessner &amp; Wiemer 2005 style) is equal within 0.15 or lower than ours in
every region except Axial, where it is 1.3 against 0.75; the Mc + 0.3 variant partly covers that case. Variants: Mc + 0.3, and Mc per region and year.</p>""")
    h.append(tab(comp, ["region", "n", "n_M", "Mc_all", "Mc_bstab", "Mc", "years_used", "n_above", "b", "b_err",
                        "df_est"],
                 rename={"n": "events", "n_M": "with M", "Mc_all": "Mc (all years)", "Mc_bstab": "Mc (b-stability)",
                         "Mc": "Mc used", "years_used": "years setting Mc", "n_above": "events &ge; Mc",
                         "b_err": "b error", "df_est": "D2 epicentres"}).replace("&amp;", "&"))
    h.append("""<p><b>Proximity.</b> eta_ij = t_ij r_ij^df 10^(-b m_i) (Baiesi &amp; Paczuski 2004), t in years,
r in km, with rescaled time T = t 10^(-q b m_i) and distance R = r^df 10^(-(1-q) b m_i), q = 0.5 (Zaliapin et
al. 2008). b is the Aki (1965) estimate above Mc of the parent's region. Distance is epicentral, floored at
0.5 km. We use 2-D distance because v4 depths are biased shallow (23.5% of events plot above the surface);
3-D distance is a variant. df = 1.6 in all regions, as in Zaliapin &amp; Ben-Zion (2013a, 2016). The
correlation dimension of the epicentres (3-30 km) is 0.8-1.4, lower than 1.6 because the offshore seismicity
lies on narrow strands and the catalog is itself clustered; df 1.3, 2.0 and the regional estimate are
variants.</p>
<p><b>Threshold.</b> A two-component Gaussian mixture fitted to log10 eta in each region; eta0 is where the
weighted components cross. The KDE valley gives lower (stricter) values by 0.4-1.3 log units. A third
choice, used as a variant, sets eta0 so that only 5% of links in time-shuffled catalogs pass it.</p>
<p><b>Distance cap.</b> A link that passes eta0 is kept only if r &le; max(10 km, 3 L), with L the subsurface
rupture length of the parent from Wells &amp; Coppersmith (1994), log10 L = -2.44 + 0.59 M (18 km for M 5.5,
76 km for M 6.5). Without it, a M 6.5 parent links events 300 km away within a few days. The cap removed
513 of 9,726 sub-threshold links above Mc. It also cuts real links when epicentres are poor: three Blanco
M 5.0-5.5 events within 2 h on 2010-05-07 (before the OBS deployments) are 21-38 km apart and stay
separate.</p>
<p><b>Families and attachment.</b> Kept links form trees; connected trees are families, events with no kept
link are background singles. Events below Mc or without M are then given their nearest above-Mc parent
(eta needs only the parent's magnitude) and join its family if the link passes the same threshold and cap
(roles ending in <code>_belowMc</code>). They do not change the family classification.</p>
<p><b>Classification.</b> For families with 2-4 events above Mc: a <i>multiplet</i> (doublet, triplet)
when M1-M2 &lt; 0.3, else <i>mainshock-aftershock</i>. For 5 or more events we use three swarm indicators,
from the topology analysis of Zaliapin &amp; Ben-Zion (2013b, 2016) and the burst criteria of Vidale &amp;
Shearer (2006): S1, no dominant event (M1-M2 &lt; 0.5); S2, late peak (at least 20% of the family occurs
before its largest event); S3, chain-like tree (mean leaf depth &ge; 2 and fewer than half of the family are
direct offspring of the largest event). A family is a <i>swarm</i> when at least two hold, otherwise a
<i>mainshock-aftershock</i> family. In mainshock-aftershock families the largest event is the mainshock and
the others are foreshocks or aftershocks by time; in swarms and multiplets all events take the family type
as role. Migration: Spearman rank correlation and Theil-Sen slope of distance from the first 10% of events
against time, for families of 10 or more; migrating when rho &ge; 0.5, p &lt; 0.01 and the slope is
positive.</p>""")

    # ---- T-R
    h.append("<h2>Bimodality and threshold</h2>")
    h.append("<img src='fig_TR_density.png' alt='T-R density per region'>")
    h.append("<p class='muted'>Density of rescaled time T and distance R of each event's nearest neighbor, "
             "events above Mc. Blue contours: one time-shuffled catalog (80% and 95% density levels). Red: "
             "the threshold line.</p>")
    h.append("<img src='fig_eta_hist.png' alt='log eta histograms'>")
    h.append(f"""<p>The clustered mode (small T and small R) is clear in BL, EN, GO and MT (D = 2.5-3.4) and
weaker in WA, OR and PU (D = 1.6-1.8), where the catalog is dominated by background and families are few.
Axial is the special case. Its 4,290 events above M 0.75 sit in a 5 km wide caldera and most occur in the
five months before the April 2015 eruption, when the rate rose steadily (Wilcock et al. 2016). Shuffling
times keeps that rate history and that spatial density, so 34% of shuffled Axial events still pass the
threshold, against 49% of real ones. At Axial the clustered/background split mostly separates the
eruption-driven rate change from the rest, and should be read as such.</p>""")
    h.append(tab(thr, ["region", "w1", "mu1", "sd1", "w2", "mu2", "sd2", "log_eta0", "log_eta0_kde",
                       "log_eta0_shuf5", "n_below_eta0", "n_cut_by_cap"],
                 rename={"w1": "w clustered", "mu1": "mu clustered", "sd1": "sd clustered", "w2": "w background",
                         "mu2": "mu background", "sd2": "sd background", "log_eta0": "log eta0 (mixture)",
                         "log_eta0_kde": "KDE valley", "log_eta0_shuf5": "5% shuffled",
                         "n_below_eta0": "links below eta0", "n_cut_by_cap": "cut by cap"}))

    # ---- validation
    h.append("<h2>Validation</h2><h3>Known sequences</h3>")
    v = val.copy()
    h.append(tab(v, ["label", "expected", "region", "M", "role", "type", "n", "n_all", "M2",
                     "duration90_days", "extent_km", "mean_leaf_depth", "regions"],
                 rename={"n": "n &ge; Mc", "n_all": "n all", "duration90_days": "t90 (days)",
                         "extent_km": "extent (km)", "mean_leaf_depth": "leaf depth"}).replace("&amp;", "&"))
    h.append("""<p>The six tectonic mainshocks are all mainshock-aftershock families with the mainshock first
(no foreshock above Mc except two for the 2014 Gorda event) and bushy trees (mean leaf depth 1.0-1.3), except
the 2014 Mw 6.8 Gorda sequence, whose tree is deeper (3.2) because of secondary sequences inside it. The 2010
Mw 6.5 family includes the 2010-02-04 Mw 5.9 event 44 km to the south, which we count as triggered. Both
Gorda families keep linking small events for years (the 2010 family's last event is in 2016), but 90% of
their events come within 81-104 days. The Blanco events behave differently (next sections). The Axial
eruption family starts at 05:11 UTC on 24 April 2015, has 632 events above Mc (805 in all), 90% of them
within 1.2 days, and a mean leaf depth of 24: a long chain, the shape of a migrating or steadily driven
swarm. It holds 61% of all Axial events above Mc between 23 April and 31 May 2015.</p>""")
    h.append("<img src='fig_examples.png' alt='Example sequences'>")
    h.append("<p class='muted'>Magnitude and distance from the largest event against time for the "
             "validation families and the largest swarms in GO, PU, ON and BL. Colored: events above Mc; "
             "grey: attached events below Mc (missing M plotted at -0.5).</p>")
    h.append("<h3>Shuffled catalogs</h3>")
    h.append(f"""<p>Three catalogs with event times permuted among the events of each region (locations and
magnitudes kept). The two modes disappear (D shuffled in the region table: 0.3-1.6, against 1.6-3.4 for the
real catalog), and the share of events that pass the real threshold and cap drops to 3-17% outside Axial
(34% at Axial). In BL, GO and MT the real clustered share is 2.5-5 times the shuffled one; in WA, OR, PU and
ON, 2.5-6 times, on smaller clustered shares.</p>""")
    h.append(tab(shm.reset_index().round(3), ["region", "frac_below_eta0", "frac_linked", "ashman_D", "mu1", "mu2"],
                 rename={"frac_below_eta0": "share below eta0", "frac_linked": "share linked (eta0 and cap)",
                         "ashman_D": "D", "mu1": "mu 1", "mu2": "mu 2"}, fmt=lambda x: f"{x:.3f}"))
    h.append("<h3>Comparison with the short-swarm scan (v4_swarms.py)</h3>")
    h.append(f"""<p>The earlier DBSCAN scan found 455 short clusters. Most of their events are small: only 3-36%
are above the regional Mc (3% at Mount St. Helens, 11% at Mount Hood, 19% on the Blanco). Of the members that
are above Mc, 85-100% are in nearest-neighbor families (Axial: 28%). The two methods agree where both can
see the events. The nearest-neighbor analysis above a regional Mc cannot see the microswarms at the Cascade
volcanoes; we ran a zoomed analysis for them (below).</p>""")
    h.append(tab(vcomp, None, rename={"group": "DBSCAN region / volcano", "members": "DBSCAN members",
                                      "pct_above_Mc": "% above Mc",
                                      "pct_of_those_clustered_by_ZBZ": "% of those in NN families"},
                 fmt=lambda x: f"{x:.1f}"))
    ct = pd.crosstab(vs.swarm_like, vs.main_family_type)
    h.append("<p>DBSCAN 'swarm-like' flag against the type of the nearest-neighbor family that holds most of "
             "the cluster's members ('background' when none):</p>" + ct.to_html(border=0))
    h.append("""<p>The DBSCAN flag and our type agree poorly: of the 298 DBSCAN swarm-like clusters, 69 sit
mainly in our swarms, 54 in multiplets, 40 in mainshock-aftershock families and 135 in no family. The
DBSCAN flag uses all events, including those below Mc, so its magnitude gap and largest-event position come
from a different and less complete set of events.</p>""")
    h.append("<h3>Cascade volcanoes</h3>")
    h.append(tab(vo, None, rename={"n_events_15km": "events &ge; Mc within 15 km"}).replace("&amp;", "&"))
    h.append(tab(vl, None, rename={"n_all": "events within radius", "n_above": "events &ge; local Mc",
                                   "ashman_D_shuffled": "D shuffled"}))
    h.append("""<p>With a local Mc (0.35 at St. Helens, 1.45 at Hood, 1.05 at Rainier, within 20 km) the method
finds 308 families at Mount St. Helens and 13 at Mount Hood, but the test is weak: a shuffled catalog in
such a small volume is almost as bimodal (D 1.9-2.0 against 2.2-2.3). None of the volcano families meets two
swarm indicators. They are small bursts of near-equal events (the largest at St. Helens: 181 events above
M 0.35 from 2011-01-27, M 3.9 then M 3.1). We do not claim a swarm/burst distinction at the volcanoes from
this catalog; it needs relocated, lower-Mc data (e.g. waveform-based detection, Skoumal et al. 2016).</p>""")

    # ---- sensitivity
    h.append("<h2>Sensitivity</h2>")
    h.append("<img src='fig_sensitivity.png' alt='sensitivity'>")
    piv = sens.pivot(index="region", columns="variant", values="pct_background").reindex(REG_ORDER).reset_index()
    h.append("<p>% background above Mc by variant:</p>" + tab(piv, fmt=lambda x: f"{x:.1f}"))
    piv2 = sens.pivot(index="region", columns="variant",
                      values="agree_bg_vs_clustered_with_main").reindex(REG_ORDER).reset_index()
    h.append("<p>Share of events (in both runs) with the same background/clustered label as the main run:</p>"
             + tab(piv2.drop(columns="main"), fmt=lambda x: f"{x:.3f}"))
    piv3 = sens.pivot(index="region", columns="variant", values="n_swarm").reindex(REG_ORDER).reset_index()
    h.append("<p>Number of swarms by variant:</p>" + tab(piv3, fmt=lambda x: f"{x:.0f}"))
    h.append("""<p>Changing df, using 3-D distance or yearly Mc moves the background share by a few points
(3-D distance raises it by 2-16 points, most offshore Oregon and at Axial: v4 depths scatter more than
epicentres). Raising Mc by
0.3 halves the number of families of 5 or more, as expected, but leaves the background share within 6
points. The conservative threshold (5% shuffled links) is the largest change. Under it the Blanco, Endeavour and
Gorda plate (63-67% background) stay below offshore Washington, offshore Oregon and Puget (82-85%), while
Axial moves to 82%. Giving the events without
magnitude M = Mc (<code>noM_floor</code>) adds 12,800 events at Axial, most of them in the eruption period,
and turns 300 more Axial families into swarms; this variant overstates their size and is only a bound.</p>""")

    # ---- topology / setting
    h.append("<h2>Cluster topology and tectonic setting</h2>")
    h.append("<img src='fig_topology.png' alt='topology'>")
    h.append(tab(st, None, rename={"n_fam_ge5": "families n&ge;5", "pct_swarm_of_ge5": "% swarms (n&ge;5)",
                                   "median_leaf_depth_ge5": "median leaf depth (n&ge;5)",
                                   "n_events_Mc1to2": "events Mc+1 to Mc+2",
                                   "mean_offspring_Mc1to2": "mean offspring &ge; Mc",
                                   "pct_Mc1to2_no_offspring": "% with no offspring",
                                   "n_events_ge_Mc1": "events &ge; Mc+1",
                                   "pct_ge_Mc1_single": "% of those single",
                                   "pct_MA_with_foreshocks": "% MA families with foreshocks",
                                   "pct_fam_migrating_ge10": "% migrating (n&ge;10)",
                                   "n_fam_ge10": "families n&ge;10"}).replace("&amp;", "&"))
    h.append("<p class='muted'>Offspring: events above Mc whose kept nearest-neighbor link points to the "
             "event. With a constant magnitude offset above Mc this is comparable between regions if the "
             "magnitude-frequency law holds.</p>")

    # ---- faults
    h.append("<h2>Families on the v0 fault surfaces</h2>")
    h.append("""<p>The fault link is a join on <code>data/faults/v0/fault_surfaces_v0_members.csv</code> and is
rerun with <code>v4_clusters_zbz.py --faults-only</code> when the surfaces change (the Blanco surfaces are
being revised). Each family carries the fraction of its events on each surface (<code>fault_ids</code>);
<code>faults_v0_cluster_counts.csv</code> gives, per surface, the counts of background, mainshock-aftershock,
swarm and multiplet events above Mc.</p>""")
    h.append("<img src='fig_faults.png' alt='faults'>")
    h.append(tab(pfs.head(15), ["surface_id", "n_analysis", "pct_background", "pct_swarm", "n_families", "lit_match"],
                 rename={"n_analysis": "events &ge; Mc", "lit_match": "literature match"}).replace("&amp;", "&"))
    h.append("""<p>Swarm shares are highest on the Blanco step-over and depression surfaces (BL010, the cross
fault between the East Blanco and Surveyor depressions: 70% swarm events; BL002, BL012, BL015: 32-36%), on
the northern Escanaba Trough axis (GO002, 47%) and on a recurring swarm source north of Mount McLoughlin
(ON027, 71%). The 36 Puget and Mendocino surfaces with 30 or more events are 11-94% background (median 65%) and
most have no swarm events; the 2010 and 2014 Gorda sequences dominate the clustered events on MT011 and
MT008.</p>""")

    # ---- interpretation
    h.append("<h2>Interpretation by setting</h2>")
    h.append("""<h3>Blanco transform</h3>
<p>Half of the Blanco events above M 2.45 are clustered, and two thirds of the families with 5 or more events
are swarms, short (median t90 4 hours) and compact (median extent 4 km). The large events behave the same
way. The 2012 Mw 5.8 and 6.0 events, two months apart, share one 77-event swarm with M1-M2 = 0.12, and
the 2015-06-01 sequence holds three M 5.4-5.9 events within 13 hours (M1-M2 = 0.09). Several other M 5+
events have one or no linked event above Mc. Low aftershock productivity, many foreshocks and swarm-like
sequences are typical of oceanic transforms (McGuire et al. 2005; Roland &amp; McGuire 2009; Boettcher
&amp; McGuire 2009), usually explained by aseismic slip transients loading small seismic patches. On the
Blanco, Kuna et al. (2019) proposed that episodic creep in the mantle below the seismic crust triggers
swarms, and Lange et al. (2026) found bursts with interevent times mostly under 5 minutes and repeating
earthquakes in the western Blanco and the Gorda Depression. Our highest swarm shares are on the pull-apart
and step-over surfaces of the western Blanco (BL010, BL002, BL012, BL015), where Ren et al. (2023) and Lange
et al. (2026) place the extensional basins and the intra-transform spreading of the Cascadia Depression.
That is where fluids and magma could also drive swarms (Hill 1977). Mainshock-aftershock events are
spread over the Blanco surfaces (23-31% of events on the Blanco Ridge, Surveyor and Cascadia Depression
surfaces), so the swarm/burst split is not a simple east-west contrast along the transform. A caveat: the Blanco network changes between the 2012-2013
and 2013-2015 OBS deployments and is poor in 2010, 2011 and 2016, so Mc = 2.45 hides the small-event
structure that Lange et al. (2026) resolve at lower magnitude.</p>
<h3>Axial Seamount and the Juan de Fuca Ridge</h3>
<p>Axial has the lowest background share (51%) and the most swarms per event. The monthly count of
swarm families climbs from one in November 2014 to 16 in March and 13 in April 2015, ahead of the eruption,
and the eruption itself is one chain-like swarm. This matches the inflation-driven ring-fault seismicity
and the abrupt rate drop after the dike drained the reservoir (Wilcock et al. 2016), and the build-up before
the 2005-2006 East Pacific Rise eruption (Tolstoy et al. 2006). Because shuffled catalogs pass 34% of Axial
events, we treat the Axial clustered share as an upper bound; the conservative threshold gives 18%. The
ridge segments north and south of Axial (in ON) and the Gorda Ridge (GO) show the same swarm-dominated
pattern, as expected for ridge-axis sequences, which are often swarms with weak Omori decay (Bohnenstiehl et
al. 2002; Hooft et al. 2010 for Endeavour).</p>
<h3>Endeavour-Explorer and Vancouver Island</h3>
<p>The Endeavour-Explorer box has few events above M 2.25 (512). Its clustered events are mostly
mainshock-aftershock families of plate-interior events (2012 Mw 6.1 Explorer plate, 2012-02-04 Mw 5.6) and
aftershocks of the 2014 Mw 6.5 Vancouver Island event, whose epicentre is just outside the box. The 2011
and 2014 Vancouver Island events are classic mainshock-aftershock families with all or nearly all aftershocks direct
offspring of the mainshock (leaf depth 1.0-1.05). Two migrating families in EN (2012-02-04 and the 2012-11
Explorer sequence) expand at 0.2-1.2 km/day.</p>
<h3>Gorda plate and Mendocino triple junction</h3>
<p>The Gorda plate (GO) mixes swarms on the Gorda Ridge and Escanaba Trough (65% of families with 5 or more
are swarms; the 2013-05-20 swarm on the northern Escanaba Trough migrates at about 10 km/day, a dike-like
rate) with mainshock-aftershock families in the plate interior (2013-12-01 Mw 5.5, 2015-04-24 Mw 5.5).
At the Mendocino triple junction the families look different: 69% of families with 5 or more are
mainshock-aftershock families, trees are the shallowest of all regions (median leaf depth 1.33), and small
and moderate events have few offspring (0.40 per event 1-2 units above Mc). Hsu et al. (2026), using the same
nearest-neighbor method on a northern California catalog, found high background rates and low aftershock
productivity in the triple-junction region, decreasing away from it. Our catalog agrees on both. The
clustered share at MT (42%) is driven by the 2010 and 2014 Gorda sequences, which hold 2,097 of the 3,272
clustered events above Mc.</p>
<h3>Forearc, slab and Puget Lowland</h3>
<p>Offshore Washington and Oregon and the Puget Lowland are 68-84% background. Families are few and small;
offshore Washington has no mainshock-aftershock family of 5 or more events above M 1.65. The forearc and
intraslab seismicity seen by the OBS deployments is diffuse and dominated by single events (Stone et al.
2018; Morton et al. 2023), and the nearest-neighbor analysis finds the same: the clustered mode is weak
(D = 1.6-1.8) and the clustered share is two to six times the shuffled one only because the latter is very
small. Puget swarms (12) and bursts are crustal: the Hood Canal fault zone (PU009) has 40% swarm events.
Cold, thick forearc and slab lithosphere is where Zaliapin &amp; Ben-Zion (2013b, 2016) find burst-like
families and singles; our forearc results are consistent but rest on few families.</p>
<h3>Heat flow and topology</h3>
<p>Across regions, the share of swarms among families with 5 or more events goes from 31% at the
Mendocino triple junction and 0% offshore Oregon (2 families) to 61-66% at Axial, the Blanco and the Gorda
plate, and the median leaf depth goes from 1.3-1.6 to 2.5-3.0. Young, hot oceanic lithosphere and the ridge
axes have swarm-like, deep trees; the older Gorda-North America boundary and the forearc have burst-like,
shallow trees and more singles. Zaliapin &amp; Ben-Zion (2013b) found this dependence on heat flow in
southern California and Zaliapin &amp; Ben-Zion (2016) globally; geothermal fields show the same shift with
fluid injection (Martínez-Garzón et al. 2018). Our regions differ in Mc (0.75 to 2.45) as well as in heat
flow, and topology depends on how many events a family has above Mc, so we compare medians at fixed
family-size thresholds; a cleaner test would use a common Mc of 2.45, which leaves too few events outside
the Blanco and the Gorda plate.</p>""")

    # ---- limits
    h.append("<h2>Limits</h2><ul>")
    for li in [
        "Completeness changes in space and time. A constant regional Mc (0.75-2.45) keeps the analysis homogeneous but discards most events (only 27,555 of 172,340 are above Mc). Families made only of small events, such as the volcano microswarms, are invisible above Mc.",
        "Depths are biased shallow, so the main run uses epicentral distance; this merges events at different depths (slab and crust) that share an epicentre. The 3-D variant changes 3-16% of labels.",
        "spatial_sigma is not a usable relative error, so we cannot weight links by location quality. The 0.5 km floor and the rupture-length cap are fixed choices; the cap cut true links where pre-OBS epicentres are off by 20-40 km (Blanco, 2010).",
        "41,573 events have no magnitude. They cannot be parents. They are attached as children when possible, and the noM_floor variant gives them M = Mc as a bound.",
        "Magnitudes mix calibrated ML and Mw; M1-M2 gaps of a few tenths, which drive the swarm/multiplet rules, are close to the ML uncertainty (about 0.2).",
        "The swarm criteria are thresholds on continuous properties. Families near the thresholds (M1-M2 near 0.5, 20% foreshocks) can switch type with small magnitude changes.",
        "Region boxes are coarse; ON mixes the volcanic arc, the onshore slab, Vancouver Island and two ridge sections.",
        "The 2016 gap in northern California picks and the network changes between OBS deployments make rates before, during and after 2011-2015 not comparable."]:
        h.append(f"<li>{li}</li>")
    h.append("</ul>")

    # ---- files
    h.append("<h2>Files</h2><ul>")
    for fn, d in [("events_zbz.csv.gz", "per event: event_id, region, in_analysis, M, parent_id, log_eta, log_T, log_R, nn_dist_km, nn_dt_days, linked, family_id, role, time, lat, lon, depth"),
                  ("families_zbz.csv", "per family: size, Mmax, M2, start, durations, extent, type, swarm indicators, topology, migration, fault fractions"),
                  ("regions_zbz.csv", "per-region summary"), ("thresholds_zbz.csv", "mixture fits and thresholds"),
                  ("completeness_b_df.csv, completeness_by_year.csv", "Mc, b, D2"),
                  ("sensitivity_zbz.csv", "summary for each variant"), ("shuffle_test.csv", "shuffled-catalog statistics"),
                  ("faults_v0_cluster_counts.csv", "per fault surface"),
                  ("validation_sequences.csv, validation_v4_swarms.csv, volcano_families.csv, volcano_local_zbz.csv, volcano_local_families.csv, setting_stats.csv", "validation and setting tables")]:
        h.append(f"<li><code>{fn}</code>: {d}</li>")
    h.append("</ul>")

    # ---- refs
    h.append("<h2>References</h2><p class='muted'>DOIs checked against Crossref on 2026-11-22. Entries without a "
             "DOI were not checked.</p><ul>")
    for _, txt, doi in REFS:
        link = f" <a href='https://doi.org/{doi}'>doi:{doi}</a>" if doi else " (not checked on Crossref)"
        h.append(f"<li>{html.escape(txt)}{link}</li>")
    h.append("</ul></body></html>")
    (out / "catalog_v4_clusters_report.html").write_text("\n".join(h))


if __name__ == "__main__":
    main()
