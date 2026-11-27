"""Fig. 6: temporal character of the boundary types (the old Fig. 6 restyled, on the catalog of the configuration).

  A  Axial Seamount before the April 2015 eruption: weekly events above the clustering Mc, and swarm families per
     month (two stacked panels on one time axis; no second y axis)
  B  West Blanco swarm of June 2015: distance along the transform strike against time, size = magnitude
  C  Gorda plate source near 41.85 N 126.1 W: map, events coloured by time, fault surfaces v1, ComCat moment tensors
  D  western Mendocino transform 2012-2016: longitude against time, ocean-bottom-station coverage shaded,
     ComCat Mw >= 5 as stars
  E  Mount St Helens swarm of August 2013 (largest swarm family within 25 km of the summit): depth against time

Event colours follow the boundary type of the segment (paper_style.TYPE_COL), except in C (time). Every panel carries
the segment's burstiness (events >= Mc, seismicity_metrics.csv: burst_all).
"""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import paper_style as S
from paper_v2 import data as D

MSH = (46.191, -122.194)


def load_clusters(cfg):
    cl = cfg.path("clusters")
    tag = cfg.get("cluster_tag", "v61")
    ez = pd.read_csv(cl / f"events_zbz_{tag}.csv.gz", usecols=["event_id", "region", "in_analysis", "M", "family_id",
                                                               "family_type", "time", "lat", "lon", "depth", "tier",
                                                               "depth_good"])
    ez["time"] = pd.to_datetime(ez.time, utc=True, format="ISO8601")
    fam = pd.read_csv(cl / f"families_zbz_{tag}.csv")
    for c in ("start", "end"):
        fam[c] = pd.to_datetime(fam[c], utc=True, format="ISO8601")
    reg = pd.read_csv(cl / "regions_zbz.csv").set_index("region")
    return ez, fam, reg


def load_mt(cfg, region):
    p = cfg.opt("focal")
    if p is None:
        return None
    mt = pd.read_csv(p)
    W, E, Sth, N = region
    mt = mt[mt.lon.between(W, E) & mt.lat.between(Sth, N)].copy()
    mt["time"] = pd.to_datetime(mt.time, utc=True, format="ISO8601")
    comp = mt[["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]].to_numpy()
    m0 = np.sqrt((comp[:, :3] ** 2).sum(1) / 2 + (comp[:, 3:] ** 2).sum(1))
    mt["Mw"] = 2 / 3 * (np.log10(m0) - 9.1)
    return mt


def burst_label(ax, metrics, seg, x=0.99, y=0.97, ha="right", va="top"):
    if seg in metrics.index and np.isfinite(metrics.loc[seg, "burst_all"]):
        S.text_halo(ax, x, y, f"{seg} burstiness {metrics.loc[seg, 'burst_all']:.2f}", transform=ax.transAxes,
                    ha=ha, va=va, color=S.INK2, zorder=30)


def title(ax, text, t, fig_h_cm, dy=0.17):
    h = ax.get_position().height * fig_h_cm
    ax.plot([0.0], [1.0 + dy / h], marker=S.TYPE_MK[t], ms=4.6 * S.TYPE_MS[S.TYPE_MK[t]], mfc=S.TYPE_COL[t],
            mec="white", mew=0.4, transform=ax.transAxes, clip_on=False, ls="")
    w = ax.get_position().width * S.COL2
    ax.text(0.2 / w, 1.0 + dy / h, text, transform=ax.transAxes, fontsize=S.FS["base"], fontweight="bold",
            ha="left", va="center")


def date_axis(ax, fmt="%b", minor=None):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def make(cfg):
    S.use()
    ez, fam, reg = load_clusters(cfg)
    metrics = pd.read_csv(cfg.path("seismicity_metrics")).set_index("seg")
    types = metrics["type"].to_dict()
    E = D.events(cfg)
    st = D.stations(cfg)
    ctx = D.context(cfg)
    xl, wl, xr_, wr = 1.2, 7.25, 10.35, 6.65
    rc = [-126.45, -125.6, 41.68, 42.05]
    hC = S.map_height(rc, wl)
    hE = 1.95
    W, H = S.COL2, round(0.55 + 3.92 + 1.55 + hC + 1.3 + hE + 0.95, 2)
    fig = S.fig_cm(W, H)
    facts = {}

    # ---------------- A: Axial Seamount (two stacked panels, shared time axis)
    t0, t1 = pd.Timestamp("2014-10-01", tz="UTC"), pd.Timestamp("2015-07-01", tz="UTC")
    a = ez[(ez.region == "AX") & ez.in_analysis & ez.time.between(t0, t1)]
    wk = a.set_index("time").resample("7D").size()
    sw = fam[(fam.region == "AX") & (fam.type == "swarm") & fam.start.between(t0, t1)]
    mo = sw.groupby(sw.start.dt.tz_localize(None).dt.to_period("M")).size()
    yA = H - 0.55
    axA1 = S.axes_cm(fig, xl, yA - 2.55, wl, 2.55)
    axA2 = S.axes_cm(fig, xl, yA - 2.55 - 0.12 - 1.25, wl, 1.25, sharex=axA1)
    col = S.TYPE_COL["ridge"]
    x = wk.index.tz_localize(None)
    axA1.fill_between(x, 0, wk.values, step="post", color=col, alpha=0.14, lw=0)
    axA1.step(x, wk.values, where="post", color=col, lw=0.9)
    mc_ax = reg.loc["AX", "Mc"] if "AX" in reg.index else np.nan
    axA1.set_ylabel(f"events per week\n(M ≥ {mc_ax:g})")
    axA1.set_ylim(0, wk.max() * 1.18)
    mx = np.array([p.to_timestamp() for p in mo.index])
    axA2.bar(mx + pd.Timedelta(days=15), mo.values, width=22, color=col, lw=0, alpha=0.85)
    axA2.set_ylabel("swarm families\nper month")
    axA2.set_ylim(0, max(4, mo.max()) * 1.25)
    erupt = pd.Timestamp("2015-04-24 08:00")
    for ax in (axA1, axA2):
        ax.axvline(erupt, color=S.ACCENT, lw=0.9, zorder=5)
        S.clean(ax, grid_axis="y")
    S.text_halo(axA1, erupt - pd.Timedelta(days=3), wk.max() * 1.10, "eruption\n24 Apr 2015", ha="right", va="top",
                color=S.ACCENT)
    axA1.tick_params(labelbottom=False)
    axA2.set_xlim(t0.tz_localize(None), t1.tz_localize(None))
    axA2.xaxis.set_major_locator(mdates.MonthLocator())
    axA2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axA2.text(0.0, -0.42, "2014", transform=axA2.transAxes, fontsize=S.FS["small"], ha="left", va="top")
    axA2.text((pd.Timestamp("2015-01-01") - t0.tz_localize(None)) / (t1 - t0), -0.42, "2015", transform=axA2.transAxes,
              fontsize=S.FS["small"], ha="left", va="top")
    title(axA1, "Axial Seamount before the 2015 eruption", types.get("AX", "ridge"), H)
    burst_label(axA1, metrics, "AX", x=0.02, ha="left")
    S.letter(axA1, "A", dx=-1.1, dy=0.08)
    facts["A"] = dict(events=int(len(a)), swarm_families=int(len(sw)), Mc=float(mc_ax), peak_week=int(wk.max()))

    # ---------------- B: West Blanco swarm, June 2015
    b = fam[(fam.region == "BL") & (fam.start.dt.strftime("%Y-%m") == "2015-06")].sort_values("n_all").iloc[-1]
    e = ez[ez.family_id == int(b.family_id)].sort_values("time")
    az = 111.0
    xk = (e.lon - e.lon.mean()) * D.KM * np.cos(np.radians(e.lat.mean()))
    yk = (e.lat - e.lat.mean()) * D.KM
    dist = (xk * np.sin(np.radians(az)) + yk * np.cos(np.radians(az))).to_numpy()
    days = ((e.time - e.time.iloc[0]).dt.total_seconds() / 86400).to_numpy()
    dmax = 12.0
    keep = days <= dmax
    axB = S.axes_cm(fig, xr_, yA - 3.92, wr, 3.92)
    o = np.argsort(e.M.fillna(0).to_numpy()[keep])
    MB = e.M.to_numpy()[keep][o]
    axB.scatter(days[keep][o], dist[keep][o], s=S.msize(MB, 0.8), c=S.TYPE_COL["transform"], edgecolors="white",
                linewidths=0.2, alpha=0.9, zorder=3)
    big = keep & (e.M.to_numpy() >= 5.0)
    axB.scatter(days[big], dist[big], s=60, marker="*", c=S.ACCENT, edgecolors="white", linewidths=0.4, zorder=5)
    for dd, yy, mm in zip(days[big], dist[big], e.M.to_numpy()[big]):
        S.text_halo(axB, dd + 0.25, yy, f"M {mm:.1f}", va="center", ha="left", zorder=6)
    axB.set_xlim(-0.3, dmax)
    axB.set_ylim(dist[keep].min() - 3, dist[keep].max() + 9)
    axB.set_xlabel(f"days since {e.time.iloc[0]:%d %b %Y %H:%M} UTC")
    axB.set_ylabel(f"distance along {az:.0f}° (km)")
    S.clean(axB, grid_axis="y")
    S.text_halo(axB, 0.99, 0.03, f"{keep.sum()} events in {dmax:g} days ({len(e)} in the family, "
                f"{b.duration_days:.0f} days)", transform=axB.transAxes, ha="right", va="bottom", color=S.INK2,
                zorder=30)
    title(axB, "West Blanco swarm, June 2015", types.get("BL", "transform"), H)
    burst_label(axB, metrics, "BL")
    S.mag_legend(axB, 0.66, 0.80, mags=(2, 3, 4, 5, 6), scale=0.8, color="#b8b8b8", title="M")
    S.letter(axB, "B", dx=-1.05, dy=0.08)
    facts["B"] = dict(family=int(b.family_id), events_family=int(len(e)), events_12d=int(keep.sum()),
                      Mmax=float(np.nanmax(e.M)), duration_days=float(b.duration_days))

    # ---------------- C: Gorda plate source (map)
    yC = yA - 3.92 - 1.55
    axC = S.axes_cm(fig, xl, yC - hC, wl, hC)
    m = S.MapAx(axC, rc, ticks=(0.2, 0.1), labels="WS", fmt="{:.1f}°")
    ctx["relief"].draw(m, px=900)
    q = E[E.lon.between(*rc[:2]) & E.lat.between(*rc[2:])].sort_values("M", na_position="first")
    tyr = q.t.dt.year + (q.t.dt.dayofyear - 1) / 365.25
    norm = Normalize(2010, 2017)
    sc_ = m.scatter(q.lon, q.lat, s=S.msize(q.M, 1.0), c=tyr, cmap=S.CM_TIME, norm=norm, edgecolors="white",
                    linewidths=0.25, zorder=5)
    # fault surfaces v1 inside the window (top edges)
    tr = pd.read_csv(cfg.path("faults") / "fault_surfaces_v1_traces.csv")
    tr = tr[tr[["top_edge_lon1", "top_edge_lon2"]].mean(1).between(*rc[:2]) &
            tr[["top_edge_lat1", "top_edge_lat2"]].mean(1).between(*rc[2:])]
    for _, r in tr.iterrows():
        S.cased(axC, [r.top_edge_lon1, r.top_edge_lon2], S.merc([r.top_edge_lat1, r.top_edge_lat2]), color=S.INK,
                lw=0.8, zorder=7)
    mt = load_mt(cfg, rc)
    nmt = 0
    if mt is not None and len(mt):
        try:
            from obspy.imaging.beachball import beach
            for _, r in mt.sort_values("Mw", ascending=False).iterrows():
                inwin = pd.Timestamp("2010-01-01", tz="UTC") <= r.time < pd.Timestamp("2017-01-01", tz="UTC")
                wdeg = 0.036
                bb = beach([r.mrr, r.mtt, r.mpp, r.mrt, r.mrp, r.mtp], xy=(r.lon, float(S.merc(r.lat))), width=wdeg,
                           linewidth=0.35, facecolor="#3a3a3a" if inwin else "#a8a8a8", edgecolor="#2a2a2a",
                           zorder=9)
                axC.add_collection(bb)
                S.text_halo(axC, r.lon, float(S.merc(r.lat)) - 0.023, f"{r.time:%Y}", ha="center", va="top", zorder=10)
                nmt += 1
        except Exception as ex:  # noqa: BLE001
            print("beachballs skipped:", ex)
    m.scalebar(rc[0] + 0.04, rc[2] + 0.025, 10)
    cax = S.axes_cm(fig, xl + wl - 2.3, yC + 0.12, 2.3, 0.1)
    cb = fig.colorbar(sc_, cax=cax, orientation="horizontal", ticks=[2010, 2012, 2014, 2016])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(length=1.8, pad=1)
    cax.xaxis.set_ticks_position("top")
    cax.text(-0.04, 0.5, "year", transform=cax.transAxes, ha="right", va="center", fontsize=S.FS["small"])
    title(axC, "Gorda plate, 41.85°N 126.1°W", types.get("GP", "intraplate"), H)
    burst_label(axC, metrics, "GP", x=0.99, y=0.03, ha="right", va="bottom")
    S.letter(axC, "C", dx=-1.1, dy=0.08)
    facts["C"] = dict(events=int(len(q)), surfaces=int(len(tr)), moment_tensors=nmt)

    # ---------------- D: western Mendocino transform, longitude against time
    t0, t1 = pd.Timestamp("2012-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC")
    lon0, lon1 = -127.6, -124.9
    md = E[E.lat.between(40.25, 40.62) & E.lon.between(lon0, lon1) & E.t.between(t0, t1)].sort_values(
        "M", na_position="first")
    axD = S.axes_cm(fig, xr_, yC - hC, wr, hC)
    days_ = pd.date_range("2012-01-01", "2017-01-01", freq="7D")
    s_ = st[st.sea]
    dkm = np.hypot((s_.lon + 126.2) * D.KM * np.cos(np.radians(40.45)), (s_.lat - 40.45) * D.KM)
    s_ = s_[dkm < 120]
    nobs = np.array([((s_.epoch_start <= t) & (s_.epoch_end >= t)).sum() for t in days_])
    on = nobs > 0
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            axD.axvspan(days_[i], days_[j], color="#dfe9f3", lw=0, zorder=0)
            i = j + 1
        else:
            i += 1
    axD.scatter(md.t.dt.tz_localize(None), md.lon, s=S.msize(md.M, 0.7), c=S.TYPE_COL["transform"],
                edgecolors="white", linewidths=0.15, alpha=0.85, zorder=3)
    mtD = load_mt(cfg, [lon0, lon1, 40.25, 40.62])
    nbig = 0
    if mtD is not None:
        bigD = mtD[mtD.time.between(t0, t1) & (mtD.Mw >= 5.0)]
        nbig = len(bigD)
        axD.scatter(bigD.time.dt.tz_localize(None), bigD.lon, s=55, marker="*", c=S.ACCENT, edgecolors="white",
                    linewidths=0.4, zorder=5)
        for _, r in bigD.iterrows():
            late = r.time > pd.Timestamp("2016-07-01", tz="UTC")
            S.text_halo(axD, r.time.tz_localize(None) + pd.Timedelta(days=-40 if late else 40), r.lon,
                        f"{r.Mw:.1f}", ha="right" if late else "left", va="center", zorder=6)
    axD.set_xlim(t0.tz_localize(None), (t1 + pd.Timedelta(days=45)).tz_localize(None))
    axD.set_ylim(lon0, lon1)
    axD.yaxis.set_major_formatter(lambda v, p: f"{abs(v):.1f}°W")
    axD.set_ylabel("longitude")
    axD.xaxis.set_major_locator(mdates.YearLocator())
    axD.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    S.clean(axD, grid_axis="y")
    leg = [Line2D([], [], ls="", marker="s", ms=6, mfc="#dfe9f3", mec="none",
                  label="ocean-bottom stations within 120 km"),
           Line2D([], [], ls="", marker="*", ms=7, mfc=S.ACCENT, mec="white", mew=0.4, label="ComCat Mw ≥ 5")]
    axD.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=S.FS["small"],
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0.1)
    title(axD, "Mendocino transform, 40.25-40.62°N", types.get("MFZ", "transform"), H, dy=0.62)
    burst_label(axD, metrics, "MFZ", y=0.03, va="bottom")
    S.letter(axD, "D", dx=-1.05, dy=0.53)
    facts["D"] = dict(events=int(len(md)), mw5=int(nbig))

    # ---------------- E, F: largest swarm family of the Mount St Helens segment (within 25 km of the summit)
    dd = D.hav(fam.lat, fam.lon, MSH[0], MSH[1])
    f5 = fam[(dd < 25) & (fam.type == "swarm")].sort_values("n_all").iloc[-1]
    e5 = ez[(ez.family_id == int(f5.family_id)) & ez.tier.isin(["A", "B", "C"])].sort_values("time")
    d5 = ((e5.time - f5.start).dt.total_seconds() / 86400).to_numpy()
    ex_, ey_ = D.local_xy(e5.lon.values, e5.lat.values, e5.lon.median(), e5.lat.median())
    w_, v_ = np.linalg.eigh(np.cov(np.c_[ex_, ey_].T))
    u = v_[:, -1] if v_[0, -1] >= 0 else -v_[:, -1]               # long axis, pointing east
    azl = float(np.degrees(np.arctan2(u[0], u[1])) % 360)
    along = ex_ * u[0] + ey_ * u[1]
    r_km = float(D.hav(e5.lat.median(), e5.lon.median(), MSH[0], MSH[1]))
    bx, by = D.local_xy(e5.lon.median(), e5.lat.median(), MSH[1], MSH[0])
    bearing = D.compass(np.degrees(np.arctan2(bx, by)))
    yE = yC - hC - 1.3
    axE = S.axes_cm(fig, xl, yE - hE, W - xl - 0.35, hE)
    o = np.argsort(e5.M.fillna(0).to_numpy())
    axE.scatter(np.maximum(d5[o], 1e-3), along[o], s=S.msize(e5.M.to_numpy()[o], 1.0),
                c=S.TYPE_COL["arc"], edgecolors="white", linewidths=0.2, alpha=0.9, zorder=3)
    axE.set_xscale("log")
    axE.set_xlim(0.01, max(400, d5.max() * 1.2))
    axE.set_ylim(np.percentile(along, 0.5) - 1.0, np.percentile(along, 99.5) + 1.5)
    axE.set_xlabel(f"days since {f5.start:%d %b %Y %H:%M} UTC (log scale)")
    axE.set_ylabel(f"along {azl:.0f}° (km)")
    axE.xaxis.set_major_formatter(lambda v, p: f"{v:g}")
    S.clean(axE, grid_axis="y")
    title(axE, f"Mount St Helens segment: swarm {r_km:.0f} km {bearing} of the summit, {f5.start:%B %Y}",
          types.get("MSH", "arc"), H)
    S.text_halo(axE, 0.99, 0.04, f"{len(e5)} tier A-C events over {f5.duration_days:.0f} days; Mmax "
                f"{np.nanmax(e5.M):.1f}; median depth {np.nanmedian(e5.depth):.1f} km", transform=axE.transAxes,
                ha="right", va="bottom", color=S.INK2, zorder=30)
    burst_label(axE, metrics, "MSH", x=0.02, ha="left")
    S.letter(axE, "E", dx=-1.1, dy=0.08)
    facts["E"] = dict(family=int(f5.family_id), events=int(len(e5)), duration_days=float(f5.duration_days),
                      distance_km=round(r_km, 1), bearing=bearing, axis_az=round(azl))

    name = "fig6_temporal"
    size = S.save(fig, cfg["out"], name)
    bl = metrics.burst_all
    cap = (f"Temporal character of four boundary types, catalog {cfg.get('label', '')} with its nearest-neighbour "
           "clustering. (A) Axial Seamount, October 2014 to June 2015: events at or above the regional clustering Mc "
           f"({facts['A']['Mc']:g}) per week and swarm families per month ({facts['A']['swarm_families']} families); "
           "red line: 24 April 2015 eruption. (B) West Blanco swarm family of June 2015: distance along 111° against "
           f"time, first 12 days ({facts['B']['events_12d']} of {facts['B']['events_family']} events); stars: M ≥ 5. "
           f"(C) Gorda plate source near 41.85°N 126.1°W: tier A-C events ({facts['C']['events']:,}) coloured by time, "
           f"fault surfaces v1 (top edges, {facts['C']['surfaces']}), ComCat moment tensors (dark: 2010-2016, grey: "
           "other years; labels: year). (D) Mendocino transform between 40.25 and 40.62°N, 2012-2016: longitude "
           f"against time ({facts['D']['events']:,} events); shading: weeks with ocean-bottom stations within 120 km; "
           f"stars: ComCat Mw ≥ 5 ({facts['D']['mw5']}). (E) Largest swarm family within 25 km of the Mount St Helens "
           f"summit, {facts['E']['distance_km']:g} km {facts['E']['bearing']} of it ({facts['E']['events']} tier A-C "
           f"events): distance along its long axis ({facts['E']['axis_az']}°) against time on a log scale. Symbol size scales "
           "with magnitude; colours by boundary type as in Fig. 1 (C: time). Burstiness of the segment (events ≥ "
           "segment Mc, whole period; 0 = Poisson, 1 = most bursty): " + ", ".join(f"{k} {bl[k]:.2f}" for k in ("AX", "BL", "GP", "MFZ", "MSH")
                                                     if k in bl.index) + ".")
    return [dict(name=name, files=[f"{name}.pdf", f"{name}.png"], size_cm=size, caption=cap,
                 before=["fig6_short_lived.png"],
                 inputs=["clusters", "cluster_tag", "events_segments", "stations", "focal", "faults",
                         "seismicity_metrics", "cache"], panels=facts)]
