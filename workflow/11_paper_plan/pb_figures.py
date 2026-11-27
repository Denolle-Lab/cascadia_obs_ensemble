#!/usr/bin/env python3
"""Draft mock-ups for the plate-boundary paper plan (D-B107). Draft quality, Crameri colour maps (D-B104):
Vp batlow reversed, Vp/Vs roma reversed centred on 1.75, 1 sigma lajolla, segment types batlowS.

  fig_sections    fault-normal sections stacked along strike, Blanco transform and Mendocino fault zone: Vp, Vp/Vs and
                  their calibrated ensemble 1 sigma, resolved nodes only, depth-good tier A-C seismicity, 7.5 km/s level
  fig_fingerprint seismicity-style fingerprint of the segments (small multiples, shared y axis)
  fig_contrasts   depth-matched structure contrasts per segment against the ensemble 1 sigma of the contrast
  fig_segments    map of the segment rule (events coloured by segment) with the feasibility class

Writes PNGs to /wd1/mdenolle_data/paper_plan/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "08_figures"))
import pb_common as P  # noqa: E402
from scm_palettes import scm_cmap, scm_colors  # noqa: E402

FIG = P.BIG / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "font.family": "DejaVu Sans"})
def batlow_s():
    """Categorical batlowS colours (index r/g/b lines of the SCM CPT), 0-1."""
    from scm_palettes import scm_path
    out = []
    for line in scm_path("batlowS").read_text().splitlines():
        t = line.split()
        if len(t) == 2 and not t[0].startswith("#") and "/" in t[1]:
            out.append([float(v) / 255 for v in t[1].split("/")])
    return np.array(out)


_S = batlow_s()
# segment types: batlowS colours with a marker shape each (redundant coding for colour-vision deficiency)
TYPE_COL = {"ridge": _S[0], "transform": _S[4], "intraplate": _S[3], "triple junction": _S[11],
            "subduction": _S[7], "crustal": _S[8], "arc": _S[12], "other": np.array([0.6, 0.6, 0.6])}
TYPE_MK = {"ridge": "o", "transform": "s", "intraplate": "D", "triple junction": "*", "subduction": "v",
           "crustal": "^", "arc": "P", "other": "."}

CM_VP = scm_cmap("batlow", reverse=True, bad="#ffffff")
CM_R = scm_cmap("roma", reverse=True, bad="#ffffff")
CM_SD = scm_cmap("lajolla", reverse=True, bad="#ffffff")   # light = small 1 sigma


def local_xy(lon, lat, lon0, lat0):
    return ((np.asarray(lon) - lon0) * P.KM * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * P.KM)


def trace_frame(xy_trace, lon, lat):
    """Signed normal distance (km, positive to the left of the trace direction) and along-trace position of points,
    from the nearest point of a densified trace; also whether the point projects inside the trace."""
    from scipy.spatial import cKDTree
    lon0, lat0 = xy_trace[:, 0].mean(), xy_trace[:, 1].mean()
    tx, ty = local_xy(xy_trace[:, 0], xy_trace[:, 1], lon0, lat0)
    o = np.argsort(tx)                                   # order west -> east
    tx, ty = tx[o], ty[o]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))]
    dx, dy = np.gradient(tx), np.gradient(ty)
    nrm = np.hypot(dx, dy)
    dx, dy = dx / nrm, dy / nrm
    px, py = local_xy(lon, lat, lon0, lat0)
    d, i = cKDTree(np.c_[tx, ty]).query(np.c_[px, py])
    vx, vy = px - tx[i], py - ty[i]
    along = s[i] + vx * dx[i] + vy * dy[i]
    normal = -vx * dy[i] + vy * dx[i]                    # left of the eastward direction = north side
    inside = (along >= 0) & (along <= s[-1])
    return normal, along, inside, s[-1]


def section_grid(m, field, mask, normal, inside, nbins, var_sd=None):
    """Mean of `field` over masked nodes per (depth layer, normal-distance bin); node counts."""
    nz = len(m.dep)
    nb = len(nbins) - 1
    out = np.full((nz, nb), np.nan)
    cnt = np.zeros((nz, nb), int)
    j = np.digitize(normal, nbins) - 1
    for k in range(nz):
        sel = mask[k] & inside[k] & (j[k] >= 0) & (j[k] < nb) & np.isfinite(field[k])
        for b in np.unique(j[k][sel]):
            v = field[k][sel & (j[k] == b)]
            out[k, b] = np.mean(v) if var_sd is None else np.median(v)
            cnt[k, b] = len(v)
    out[cnt < 2] = np.nan
    return out, cnt


def fig_sections():
    m = P.Model()
    c = pd.read_parquet(P.BIG / "events_segments_v60.parquet")
    E = c[c.is_eq & c.abc & ~c.review_P1 & c.depth_good]
    pts, _, _ = P.pb2002()
    otf = pts["OTF"]
    traces = {"BL": otf[(otf[:, 1] > 42.9) & (otf[:, 1] < 44.7) & (otf[:, 0] < -126.5)],
              "MFZ": otf[(otf[:, 1] < 41.0) & (otf[:, 0] < -124.9)]}
    names = {"BL": "Blanco transform", "MFZ": "Mendocino fault zone"}
    nbins = np.arange(-45, 45.1, 5.0)
    zed = np.r_[m.dep[0], 0.5 * (m.dep[1:] + m.dep[:-1]), m.dep[-1]]
    LO = np.broadcast_to(m.LO, m.Vp.shape)
    LA = np.broadcast_to(m.LA, m.Vp.shape)
    rows = [("Vp", "Vp (km/s)", CM_VP, dict(vmin=5.5, vmax=8.2), m.mP),
            ("sdVp", "1$\\sigma$ Vp (km/s, x1.50)", CM_SD, dict(vmin=0, vmax=0.4), m.mP),
            ("VpVs", "Vp/Vs", CM_R, dict(norm=TwoSlopeNorm(1.75, 1.60, 1.90)), m.mPS),
            ("sdVpVs", "1$\\sigma$ Vp/Vs (raw x1.50)", CM_SD, dict(vmin=0, vmax=0.12), m.mPS)]
    fig, axs = plt.subplots(4, 2, figsize=(7.2, 8.6), sharex=True, sharey=True, constrained_layout=True)
    info = {}
    for jc, seg in enumerate(["BL", "MFZ"]):
        tr = traces[seg]
        normal, along, inside, L = trace_frame(tr, LO.ravel(), LA.ravel())
        normal, inside = normal.reshape(m.Vp.shape), inside.reshape(m.Vp.shape)
        en, ea, ei, _ = trace_frame(tr, E.lon.values, E.lat.values)
        # centre the section on the seismicity: shift by the median normal distance of the segment's events
        sel = (E.seg == seg).values & ei & (np.abs(en) <= 45)
        off = float(np.median(en[sel]))
        normal = normal - off
        en = en - off
        sel = (E.seg == seg).values & ei & (np.abs(en) <= 45)
        es, en = E[sel], en[sel]
        vp_grid, cntP = section_grid(m, m.Vp, m.mP, normal, inside, nbins)
        info[seg] = dict(length_km=L, n_events=len(es), nodes_P=int(cntP.sum()), offset_km=off)
        for ir, (key, lab, cm, kw, mask) in enumerate(rows):
            field = {"Vp": m.Vp, "sdVp": m.sdVp, "VpVs": m.VpVs, "sdVpVs": m.sdVpVs}[key]
            g, cnt = section_grid(m, field, mask, normal, inside, nbins, var_sd=key.startswith("sd") or None)
            ax = axs[ir, jc]
            pc = ax.pcolormesh(nbins, zed, g, cmap=cm, shading="flat", **kw)
            if key in ("Vp", "VpVs"):
                xc = 0.5 * (nbins[1:] + nbins[:-1])
                try:
                    cs = ax.contour(xc, m.dep, vp_grid, levels=[7.5], colors="k", linewidths=0.8, linestyles="--")
                    ax.clabel(cs, fmt="7.5", fontsize=6)
                except Exception:  # noqa: BLE001
                    pass
            ax.scatter(en, es.depth, s=0.3, c="k", alpha=0.25, lw=0, rasterized=True)
            ax.axvline(0, color="0.3", lw=0.5, ls=":")
            if jc == 1:
                fig.colorbar(pc, ax=axs[ir, :], shrink=0.85, pad=0.01, label=lab)
            if ir == 0:
                ax.set_title(f"{names[seg]} ({L:.0f} km stacked)\n{len(es):,} depth-good tier A-C events",
                             fontsize=7.5)
            ax.set_ylim(40, 0)
            ax.set_xlim(-45, 45)
            if jc == 0:
                ax.set_ylabel("depth bsl (km)")
            if ir == 3:
                ax.set_xlabel("fault-normal distance (km, + north)")
            ax.text(0.02, 0.04, {"Vp": "P mask", "sdVp": "P mask", "VpVs": "P and S mask",
                                 "sdVpVs": "P and S mask"}[key], transform=ax.transAxes, fontsize=6, color="0.2")
    fig.suptitle("DRAFT: fault-normal sections, v0.4.2, resolved nodes only (rock and (checkerboard or DWS >= 100)),\n"
                 "5-km bins (>= 2 nodes); dashed: 7.5 km/s; dots: seismicity in the segment", fontsize=7.5)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_sections_BL_MFZ.png", dpi=170)
    plt.close(fig)
    return info


def fig_fingerprint():
    s = pd.read_csv(P.OUT / "seismicity_metrics.csv")
    f = pd.read_csv(P.OUT / "structure_feasibility.csv")[["seg", "class"]]
    s = s.merge(f, on="seg", how="left")
    s = s[s.type != "other"].reset_index(drop=True)
    s = s.iloc[::-1].reset_index(drop=True)              # first segment on top
    y = np.arange(len(s))
    col = [TYPE_COL[t] for t in s.type]
    panels = [
        ("n_events", "events\n(tier A-C)", None, None, "log"),
        ("Mc", "Mc\n(maxc + 0.2)", "Mc_lo", "Mc_hi", None),
        ("b", "b-value\n(95 % CI)", "b_lo", "b_hi", None),
        ("pct_bg", "background\n(%)", "pct_bg_lo", "pct_bg_hi", None),
        ("pct_MA", "mainshock-\naftershock (%)", "pct_MA_lo", "pct_MA_hi", None),
        ("pct_swarm", "swarm\n(%)", "pct_swarm_lo", "pct_swarm_hi", None),
        ("bsf_med", "depth below\nsurface (km)", "bsf_p10", "bsf_p90", None),
        ("frac_below_z75", "share below\n7.5 km/s level", "frac_below_z75_mem_p05",
         "frac_below_z75_mem_p95", None),
        ("dip_wmed", "surface dip\n(deg)", "dip_min", "dip_max", None),
        ("burst_all", "burstiness\n(events >= Mc)", None, None, None),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(11.5, 5.2), sharey=True)
    for ax, (k, lab, lo, hi, sc) in zip(axs, panels):
        v = s[k].values.astype(float)
        for i in range(len(s)):
            if not np.isfinite(v[i]):
                continue
            hollow = (k in ("bsf_med",) and s.depth_class[i] == "fair")
            if lo and np.isfinite(s[lo][i]) and np.isfinite(s[hi][i]):
                ax.plot([s[lo][i], s[hi][i]], [y[i], y[i]], color=col[i], lw=1.6, alpha=0.7)
            ax.plot(v[i], y[i], TYPE_MK[s.type[i]], ms=5 if s.type[i] != "triple junction" else 7,
                    mfc="white" if hollow else col[i], mec=col[i], mew=1.0)
        if sc == "log":
            ax.set_xscale("log")
        ax.set_title(lab, fontsize=7)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=6.5)
        for i in range(len(s) - 1):
            if s.type[i] != s.type[i + 1]:
                ax.axhline(i + 0.5, color="0.6", lw=0.4)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([f"{a} ({c[0] if isinstance(c, str) else '-'})" for a, c in zip(s.seg, s["class"])],
                           fontsize=7)
    for t in P.TYPE_ORDER[:-1]:
        axs[0].plot([], [], TYPE_MK[t], color=TYPE_COL[t], label=t)
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=6.5, loc="lower center", ncol=7, frameon=False)
    fig.suptitle("DRAFT: seismicity-style fingerprint, catalog v6.0 (tier A-C; flagged non-earthquakes and review P1 "
                 "left out); label = segment (structure class r/m/n).\nBars: 95 % bootstrap CI (Mc, b, shares); depth "
                 "p10-p90 (hollow: depth fair, median sigma_z 3-6 km); 7.5 km/s share: ensemble members 5-95 %; dip: "
                 "min-max over surfaces.", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(FIG / "fig_fingerprint.png", dpi=170)
    plt.close(fig)


def fig_contrasts():
    ct = pd.read_csv(P.OUT / "structure_contrasts.csv")
    f = pd.read_csv(P.OUT / "structure_feasibility.csv")
    keep = f[f["class"].isin(["resolved", "marginal"])].seg.tolist()
    ct = ct[ct.seg.isin(keep) & ct.value.notna()]
    segs = [s for s in P.SEG_ORDER if s in keep]
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 4.8), sharey=True)
    mk = {"near-far": ("o", 0.0), "across": ("^", 0.22), "along": ("s", -0.22)}
    units = {"Vp": "km/s", "Vs": "km/s", "VpVs": ""}
    for ax, v in zip(axs, ["Vp", "Vs", "VpVs"]):
        d = ct[ct["var"] == v]
        for i, sgm in enumerate(segs):
            yy = len(segs) - 1 - i
            for cn, (sym, off) in mk.items():
                r = d[(d.seg == sgm) & (d.contrast == cn)]
                if r.empty:
                    continue
                r = r.iloc[0]
                strong = r.ratio >= 2
                c = TYPE_COL[P.SEGMENTS[sgm][1]]
                ax.errorbar(r.value, yy + off, xerr=2 * r.sigma_members, fmt=sym, ms=4, color=c,
                            mfc=c if strong else "white", lw=0.9, capsize=1.5)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_title(f"{v} contrast {('(' + units[v] + ')') if units[v] else ''}", fontsize=8)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axs[0].set_yticks(range(len(segs)))
    axs[0].set_yticklabels(segs[::-1])
    for cn, (sym, _) in mk.items():
        axs[2].plot([], [], sym, color="0.3", mfc="white", label=cn)
    axs[2].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("DRAFT: depth-matched contrasts (deterministic v0.4.2) with +-2 sigma of the same contrast over the "
                 "100 ensemble members (x calibration factor); filled = |contrast| >= 2 sigma.\nnear-far: nodes within "
                 "5 km of the segment's seismicity/faults minus nodes 10-20 km away; across: the two sides of the "
                 "strike line; along: first vs last third.", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_contrasts.png", dpi=170)
    plt.close(fig)


def fig_segments():
    c = pd.read_parquet(P.BIG / "events_segments_v60.parquet")
    E = c[c.is_eq & c.abc & ~c.review_P1]
    f = pd.read_csv(P.OUT / "structure_feasibility.csv").set_index("seg")
    ax_t = pd.read_csv(P.OUT / "segment_axes.csv").set_index("seg")
    m = P.Model()
    pts, sub, tj = P.pb2002()
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.contour(m.lon, m.lat, m.surf, levels=[0], colors="0.55", linewidths=0.6)
    segs = [s for s in P.SEG_ORDER if s in E.seg.unique()]
    rng = np.random.default_rng(0)
    for s_ in segs:
        g = E[E.seg == s_]
        if len(g) > 6000:
            g = g.iloc[rng.choice(len(g), 6000, replace=False)]
        ax.scatter(g.lon, g.lat, s=0.4, color=TYPE_COL[P.SEGMENTS[s_][1]], lw=0, rasterized=True)
    for t in P.TYPE_ORDER:
        ax.plot([], [], "o", color=TYPE_COL[t], label=t, ms=4)
    ax.legend(fontsize=6.5, loc="lower left", frameon=True)
    for k, sty in (("OSR", "k-"), ("OTF", "k-"), ("SUB", "k--")):
        xy = pts[k]
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=0.5, color="k")
    ax.plot(tj[0], tj[1], "k*", ms=8)
    sym = {"resolved": "#1a1a1a", "marginal": "#8c8c8c", "not resolved": "#ffffff"}
    for s_ in segs:
        if s_ not in ax_t.index:
            continue
        a = ax_t.loc[s_]
        cl = f.loc[s_, "class"]
        ax.text(a.lon0, a.lat0, s_, fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=sym[cl], ec="k", lw=0.5, alpha=0.85),
                color="white" if cl == "resolved" else "k")
    ax.set_xlim(-131.8, -120.2)
    ax.set_ylim(39.2, 51.0)
    ax.set_aspect(1 / np.cos(np.radians(45)))
    ax.set_title("DRAFT: segment rule on catalog v6.0 (tier A-C earthquakes); label fill = structure class\n"
                 "black resolved, grey marginal, white not resolved; dots = PB2002 traces, star = triple junction",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_segments_map.png", dpi=170)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    info = fig_sections()
    P.log(info)
    pd.DataFrame(info).T.to_csv(P.OUT / "sections_info.csv")
    fig_fingerprint()
    fig_contrasts()
    fig_segments()


if __name__ == "__main__":
    main()
