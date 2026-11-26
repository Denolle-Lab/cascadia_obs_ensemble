"""Figures of one event view (from the cache of rv_event.py):
  <ID>_geom.webp   regional map, zoom map with the NonLinLoc scatter, E-W depth section, pick times vs distance
  <ID>_rec.webp    record section: vertical (left) and horizontals (right) on the v5.1 origin-time axis
  <ID>_zoom.webp   P (vertical) and S (horizontal, else vertical) windows at the nearest picked stations
Colours: reference categorical palette slots 1-3 (blue, orange, aqua; all-pairs validated) for the hypotheses, with
marker and line style as second encoding; observed picks black.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rv_common as C  # noqa: E402
import rv_event as E  # noqa: E402

COL = {"v5.1": "#2a78d6", "v4": "#eb6834", "ComCat": "#1baf7a"}
MRK = {"v5.1": "*", "v4": "s", "ComCat": "D"}
LS = {"v5.1": "-", "v4": "--", "ComCat": ":"}
MS = {"v5.1": 13, "v4": 7, "ComCat": 7}
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d4"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.labelcolor": INK, "legend.fontsize": 7.5, "legend.frameon": False,
                     "savefig.facecolor": "white"})
SAVE = dict(format="webp", pil_kwargs={"lossless": False, "quality": 88, "method": 4})


def _load(eid):
    J = json.loads((C.WORK / "events" / f"{eid}.json").read_text())
    P = pd.DataFrame(J["phases"])
    f = C.WORK / "events" / f"{eid}.scat.npy"
    scat = np.load(f) if f.exists() else None
    return J, P, scat


def _coast(ax, lon0, lon1, lat0, lat1, n=220):
    lo, la = np.meshgrid(np.linspace(lon0, lon1, n), np.linspace(lat0, lat1, n))
    try:
        h = C.surface_km(lo, la)
        if np.nanmin(h) < 0 < np.nanmax(h):
            ax.contour(lo, la, h, levels=[0.0], colors=[MUTED], linewidths=0.6, zorder=1)
    except Exception:
        pass


def _aspect(ax, lat):
    ax.set_aspect(1.0 / np.cos(np.radians(lat)))


def _sta_table(J, P):
    S = C.stations()
    rows = []
    for st in dict.fromkeys(list(P.sta) + list(J.get("unpicked", {}).keys())):
        ph = set(P[P.sta == st].phase)
        info = J.get("wf_info", {}).get(st, {})
        nodata = str(info.get("status", "")).startswith("no data") or str(info.get("status", "")).startswith("error")
        rows.append(dict(sta=st, lat=S.lat.get(st, np.nan), lon=S.lon.get(st, np.nan), z=S.z_km.get(st, np.nan),
                         P="P" in ph, S="S" in ph, picked=bool(ph), nodata=nodata,
                         shown=st in J.get("stations_shown", [])))
    return pd.DataFrame(rows)


def _stations_on_map(ax, T, legend=True):
    kinds = [("P and S picks", T.P & T.S, INK, INK), ("P pick only", T.P & ~T.S, "#8a8985", INK),
             ("S pick only", ~T.P & T.S, "white", INK), ("no pick (waveform shown)", ~T.picked, "white", "#a3a29d")]
    for lab, m, fc, ec in kinds:
        if m.any():
            ax.scatter(T.lon[m], T.lat[m], marker="^", s=34, c=fc, edgecolors=ec, linewidths=0.7, zorder=4,
                       label=lab if legend else None)
    m = T.nodata & T.shown
    if m.any():
        ax.scatter(T.lon[m], T.lat[m], marker="x", s=22, c="#b3261e", linewidths=0.9, zorder=5,
                   label="no waveform data" if legend else None)


def _hyp_points(ax, J, xkey="lon", ykey="lat", xf=None, yf=None, legend=True):
    for h in E.HYPS:
        d = J["hyp"].get(h)
        if d is None or d["lat"] is None:
            continue
        x = d[xkey] if xf is None else xf(d)
        y = d[ykey] if yf is None else yf(d)
        ax.plot(x, y, MRK[h], ms=MS[h], mfc=COL[h], mec="white", mew=0.8, zorder=6,
                label=(f"{h}" if legend else None))


def fig_geometry(eid, out):
    J, P, scat = _load(eid)
    c = C.catalog().loc[eid]
    hy = J["hyp"]
    lat0, lon0, z0 = hy["v5.1"]["lat"], hy["v5.1"]["lon"], hy["v5.1"]["depth"]
    T = _sta_table(J, P)
    fig = plt.figure(figsize=(16.5, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.25, 1.15], wspace=0.28, left=0.04, right=0.99, top=0.88,
                          bottom=0.12)
    # (a) regional map
    ax = fig.add_subplot(gs[0])
    lons = np.r_[T.lon.dropna(), [hy[h]["lon"] for h in hy if hy[h]["lon"] is not None]]
    lats = np.r_[T.lat.dropna(), [hy[h]["lat"] for h in hy if hy[h]["lat"] is not None]]
    pad_lo, pad_la = max(0.15, 0.08 * np.ptp(lons)), max(0.1, 0.08 * np.ptp(lats))
    ext = [lons.min() - pad_lo, lons.max() + pad_lo, lats.min() - pad_la, lats.max() + pad_la]
    _coast(ax, *ext)
    _stations_on_map(ax, T)
    _hyp_points(ax, J)
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); _aspect(ax, lat0)
    ax.set_title("(a) stations used and epicentres", loc="left")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2, fontsize=6.8, handletextpad=0.3, columnspacing=0.8)
    # (b) zoom map with scatter
    ax = fig.add_subplot(gs[1])
    dh = [C.hav(lat0, lon0, hy[h]["lat"], hy[h]["lon"]) for h in hy if hy[h]["lat"] is not None]
    half = float(np.clip(max([10.0] + [d + 6 for d in dh] + [2.5 * (c.h_max if np.isfinite(c.h_max) else 3)]), 10, 160))
    if scat is not None:
        dsc = C.hav(lat0, lon0, scat[:, 1], scat[:, 0])
        half = float(np.clip(max(half, np.percentile(dsc, 98) * 1.15), 10, 160))
    dlat = half / 111.2
    dlon = half / (111.2 * np.cos(np.radians(lat0)))
    ext2 = [lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat]
    _coast(ax, *ext2)
    if scat is not None:
        sc = ax.scatter(scat[:, 0], scat[:, 1], c=scat[:, 2], s=2.5, cmap="viridis_r", alpha=0.6, linewidths=0, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label("scatter depth (km)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    _stations_on_map(ax, T, legend=False)
    for _, s in T.iterrows():
        if ext2[0] < s.lon < ext2[1] and ext2[2] < s.lat < ext2[3]:
            ax.annotate(s.sta.split(".")[0], (s.lon, s.lat), xytext=(3, 3), textcoords="offset points", fontsize=6,
                        color=MUTED)
    _hyp_points(ax, J, legend=False)
    ax.set_xlim(ext2[:2]); ax.set_ylim(ext2[2:]); _aspect(ax, lat0)
    ax.set_title(f"(b) zoom ±{half:.0f} km: NonLinLoc scatter" + ("" if scat is not None else " (none saved)"), loc="left")
    ax.grid(color=GRID, lw=0.4)
    # (c) E-W depth section
    ax = fig.add_subplot(gs[2])
    kx = 111.2 * np.cos(np.radians(lat0))
    xs = np.linspace(-half, half, 300)
    if scat is not None:
        ax.scatter((scat[:, 0] - lon0) * kx, scat[:, 2], s=2.5, c="#a3a29d", alpha=0.5, linewidths=0, zorder=2,
                   label="NonLinLoc scatter")
    srf = C.surface_km(lon0 + xs / kx, np.full_like(xs, lat0))
    ax.plot(xs, -srf, color=INK, lw=1.0, label="solid surface (D-B068)", zorder=3)
    slab = C.slab2_depth(lon0 + xs / kx, np.full_like(xs, lat0))
    if np.isfinite(slab).any():
        ax.plot(xs, slab, color="#6250d6", lw=1.2, ls="-.", label="Slab2 slab top", zorder=3)
    m = T.shown & (np.abs(T.lat - lat0) * 111.2 < half)
    ax.scatter((T.lon[m] - lon0) * kx, T.z[m], marker="^", s=26, c=np.where(T.picked[m], INK, "white"),
               edgecolors=INK, linewidths=0.6, zorder=4, label=f"stations within ±{half:.0f} km N-S")
    for h in E.HYPS:
        d = hy.get(h)
        if d is None or d["lat"] is None:
            continue
        ax.plot((d["lon"] - lon0) * kx, d["depth"], MRK[h], ms=MS[h], mfc=COL[h], mec="white", mew=0.8, zorder=6,
                label=f"{h} {d['depth']:.1f} km")
    if np.isfinite(c.sig_z):
        ax.errorbar(0, z0, yerr=c.sig_z, color=COL["v5.1"], lw=1.0, capsize=3, zorder=5)
    zmax = max([z0, 5.0] + [hy[h]["depth"] for h in hy if hy[h]["depth"] is not None]
               + ([float(np.percentile(scat[:, 2], 99.5))] if scat is not None else []))
    if np.isfinite(slab).any():
        zmax = max(zmax, float(np.nanmax(slab)) * 0 + zmax)
    zmin = min(-1.0, float(np.nanmin(-srf)) - 0.5 if np.isfinite(srf).any() else -1.0)
    ax.set_ylim(zmax * 1.08 + 2, zmin)
    ax.set_xlim(-half, half)
    ax.set_xlabel(f"km east of the v5.1 epicentre (section along {lat0:.3f}°N)")
    ax.set_ylabel("depth (km below sea level)")
    ax.set_title("(c) E-W depth section" + ("" if np.isfinite(slab).any() else "  (no Slab2 here)"), loc="left")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=3, fontsize=6.6, handletextpad=0.3, columnspacing=0.8)
    # (d) pick times vs distance
    ax = fig.add_subplot(gs[3])
    ot = pd.Timestamp(hy["v5.1"]["OT"])
    P["tobs_rel"] = (pd.to_datetime(P.tobs, utc=True, format="ISO8601") - ot).dt.total_seconds()
    for ph, mk in (("P", "o"), ("S", "s")):
        x = P[P.phase == ph]
        ax.plot(x.dist, x.tobs_rel, mk, ms=5.5, mfc=INK if ph == "P" else "white", mec=INK, mew=0.8, zorder=5,
                label=f"observed {ph}")
    for h in E.HYPS:
        d = hy.get(h)
        if d is None or d["OT"] is None:
            continue
        off = (pd.Timestamp(d["OT"]) - ot).total_seconds() + (d["ot_refit_shift"] or 0.0)
        for ph in ("P", "S"):
            x = P[P.phase == ph].sort_values("dist")
            y = off + x[f"tt_{h}"] + x.tcorr
            ax.plot(x.dist, y, LS[h], color=COL[h], lw=1.3, marker=MRK[h], ms=3.5 if h != "v5.1" else 6,
                    label=f"{h} predicted" if ph == "P" else None, zorder=4)
    ax.set_xlabel("epicentral distance from the v5.1 epicentre (km)")
    ax.set_ylabel("time after the v5.1 origin time (s)")
    ax.set_title("(d) picks and predictions (OT refit per hypothesis)", loc="left")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(loc="upper left", fontsize=6.8)
    fig.suptitle(f"Event {eid}  {c.time[:22]} UTC   v5.1 {lat0:.4f}°N {abs(lon0):.4f}°W {z0:.1f} km, tier {c.tier}",
                 x=0.04, ha="left", fontsize=10, color=INK)
    fig.savefig(out, dpi=95, **SAVE)
    plt.close(fig)


def _prep(segs, band, rel_lo, rel_hi):
    """Filtered, decimated pieces [(t, y)] of one channel within the window."""
    out = []
    for rel0, sr, data in segs:
        if len(data) < 2 * sr:
            continue
        y = C.bandpass(data, sr, band)
        t = rel0 + np.arange(len(y)) / sr
        q = int(max(1, round(sr / 50.0)))
        if q > 1:
            y, t = y[::q], t[::q]
        m = (t >= rel_lo) & (t <= rel_hi)
        if m.sum() > 5:
            out.append((t[m], y[m]))
    return out


def _norm(pieces):
    a = max([np.max(np.abs(y)) for _, y in pieces] + [0.0])
    return a if a > 0 else 1.0


def _pred_times(J, P, st):
    """{(phase, hyp): time after v5.1 OT} at a station (refit OT per hypothesis)."""
    hy = J["hyp"]
    ot = pd.Timestamp(hy["v5.1"]["OT"])
    out = {}
    x = P[P.sta == st]
    for h in E.HYPS:
        d = hy.get(h)
        if d is None or d["OT"] is None:
            continue
        off = (pd.Timestamp(d["OT"]) - ot).total_seconds() + (d["ot_refit_shift"] or 0.0)
        for ph in ("P", "S"):
            r = x[x.phase == ph]
            if len(r) and np.isfinite(r[f"tt_{h}"].iloc[0]):
                out[(ph, h)] = off + r[f"tt_{h}"].iloc[0] + r.tcorr.iloc[0]
    up = J.get("unpicked", {}).get(st)
    if up:
        for k, v in up["pred"].items():
            ph, h = k.split("_", 1)
            if v is not None:
                out[(ph, h)] = v
    return out


def _obs_times(J, P, st):
    ot = pd.Timestamp(J["hyp"]["v5.1"]["OT"])
    x = P[P.sta == st]
    return {ph: (pd.Timestamp(r.tobs) - ot).total_seconds() for ph, r in
            ((r.phase, r) for _, r in x.iterrows())}


def _ticks(ax, y0, pred, obs, phases=("P", "S")):
    for (ph, h), t in pred.items():
        if ph not in phases:
            continue
        if ph == "P":
            ax.plot([t, t], [y0 + 0.12, y0 + 0.46], color=COL[h], lw=1.6, ls=LS[h], zorder=5)
        else:
            ax.plot([t, t], [y0 - 0.46, y0 - 0.12], color=COL[h], lw=1.6, ls=LS[h], zorder=5)
    for ph, t in obs.items():
        if ph in phases:
            ax.plot([t, t], [y0 - 0.48, y0 + 0.48], color=INK, lw=1.4, zorder=6)
            ax.text(t, y0 + 0.5, ph, fontsize=6.5, color=INK, ha="center", va="bottom", zorder=6)


def _station_order(J, P):
    S = C.stations()
    lat0, lon0 = J["hyp"]["v5.1"]["lat"], J["hyp"]["v5.1"]["lon"]
    d = {st: float(C.hav(lat0, lon0, S.lat[st], S.lon[st])) for st in J["stations_shown"]}
    return sorted(d, key=d.get), d


def fig_record(eid, out):
    J, P, _ = _load(eid)
    W = E.load_waveforms(eid)
    band = tuple(J["band"])
    rel_lo, rel_hi = J["window_rel"]
    order, dist = _station_order(J, P)
    n = len(order)
    h = max(3.0, 0.36 * n + 1.3)
    fig, axs = plt.subplots(1, 2, figsize=(16.5, h), sharey=True,
                            gridspec_kw=dict(left=0.13, right=0.99, top=1 - 0.75 / h, bottom=0.55 / h, wspace=0.03))
    for k, st in enumerate(order):
        y0 = n - 1 - k
        chans = W.get(st, {})
        zc = [c for c in chans if c[-1] in "Z3"]
        hc = [c for c in chans if c[-1] in "EN12"]
        pred, obs = _pred_times(J, P, st), _obs_times(J, P, st)
        picked = st in set(P.sta)
        info = J["wf_info"].get(st, {})
        lab = f"{st}  {dist[st]:.0f} km"
        ph = "".join(sorted(set(P[P.sta == st].phase))) or "no pick"
        axs[0].text(rel_lo - 0.005 * (rel_hi - rel_lo), y0, f"{lab}\n{ph}  {info.get('band', '')}", ha="right",
                    va="center", fontsize=6.6, color=INK if picked else MUTED, clip_on=False)
        for ax, cl, phs in ((axs[0], zc, ("P", "S")), (axs[1], hc, ("P", "S"))):
            if not cl:
                txt = info.get("status", "no data") if not chans else ("no horizontal channel" if ax is axs[1] else "no vertical")
                ax.text(rel_lo + 0.01 * (rel_hi - rel_lo), y0, txt[:90], fontsize=6.5, color="#b3261e" if not chans else MUTED,
                        va="center")
                _ticks(ax, y0, pred, obs, phs)
                continue
            pcs = []
            for ch in cl[:2]:
                pcs.append((ch, _prep(chans[ch], band, rel_lo, rel_hi)))
            a = _norm([p for _, pp in pcs for p in pp])
            for j, (ch, pp) in enumerate(pcs):
                for t, yy in pp:
                    ax.plot(t, y0 + 0.45 * yy / a, color=INK if j == 0 else "#8a8985", lw=0.45, zorder=3)
            _ticks(ax, y0, pred, obs, phs)
    for ax, ttl in ((axs[0], "vertical"), (axs[1], "horizontals (first: dark, second: grey)")):
        ax.set_xlim(rel_lo, rel_hi)
        ax.set_ylim(-0.8, n - 0.2)
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_xlabel("time after the v5.1 origin time (s)")
        ax.set_title(f"{ttl}; band-pass {band[0]:g}-{band[1]:g} Hz zero-phase; each row scaled to its own peak",
                     loc="left")
    hnd = [plt.Line2D([], [], color=INK, lw=1.4, label="observed pick (re-timed)")]
    hnd += [plt.Line2D([], [], color=COL[h], lw=1.6, ls=LS[h], label=f"{h} predicted (P above trace, S below)")
            for h in E.HYPS if h in J["hyp"]]
    axs[1].legend(handles=hnd, loc="lower right", bbox_to_anchor=(1.0, 1.0 + 0.25 / h), ncol=4, fontsize=7)
    fig.savefig(out, dpi=90, **SAVE)
    plt.close(fig)


def fig_raw(eid, out):
    """Vertical raw counts (demeaned, no filter), same rows and window as the record section (P1: the filtered view
    turns a step of the recording into pulses)."""
    J, P, _ = _load(eid)
    W = E.load_waveforms(eid)
    rel_lo, rel_hi = J["window_rel"]
    order, dist = _station_order(J, P)
    n = len(order)
    h = max(3.0, 0.36 * n + 1.3)
    fig, ax = plt.subplots(figsize=(12.5, h), gridspec_kw=dict(left=0.17, right=0.99, top=1 - 0.6 / h, bottom=0.55 / h))
    for k, st in enumerate(order):
        y0 = n - 1 - k
        chans = W.get(st, {})
        zc = [c for c in chans if c[-1] in "Z3"]
        pred, obs = _pred_times(J, P, st), _obs_times(J, P, st)
        picked = st in set(P.sta)
        ax.text(rel_lo - 0.005 * (rel_hi - rel_lo), y0, f"{st}  {dist[st]:.0f} km", ha="right", va="center", fontsize=6.6,
                color=INK if picked else MUTED, clip_on=False)
        if not zc:
            ax.text(rel_lo + 0.01 * (rel_hi - rel_lo), y0, J["wf_info"].get(st, {}).get("status", "no data")[:80],
                    fontsize=6.5, color="#b3261e", va="center")
            continue
        pcs = []
        for rel0, sr, data in chans[zc[0]]:
            t = rel0 + np.arange(len(data)) / sr
            m = (t >= rel_lo) & (t <= rel_hi)
            if m.sum() > 5:
                pcs.append((t[m], data[m].astype(float)))
        if not pcs:
            continue
        mu = np.median(np.concatenate([y for _, y in pcs]))
        a = max(np.max(np.abs(y - mu)) for _, y in pcs) or 1.0
        for t, y in pcs:
            ax.plot(t, y0 + 0.45 * (y - mu) / a, color=INK, lw=0.45)
        _ticks(ax, y0, pred, obs)
    ax.set_xlim(rel_lo, rel_hi)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_xlabel("time after the v5.1 origin time (s)")
    ax.set_title("vertical, raw counts minus the median, no filter, each row scaled to its own peak", loc="left")
    fig.savefig(out, dpi=90, **SAVE)
    plt.close(fig)


def fig_zoom(eid, out, nmax=10):
    J, P, _ = _load(eid)
    W = E.load_waveforms(eid)
    band = tuple(J["band"])
    order, dist = _station_order(J, P)
    sts = [s for s in order if s in set(P.sta) and W.get(s)][:nmax]
    if not sts:
        return False
    n = len(sts)
    fig, axs = plt.subplots(n, 2, figsize=(12.5, 0.95 * n + 0.9), squeeze=False,
                            gridspec_kw=dict(left=0.1, right=0.99, top=1 - 0.5 / (0.95 * n + 0.9), bottom=0.45 / (0.95 * n + 0.9),
                                             hspace=0.55, wspace=0.06))
    for k, st in enumerate(sts):
        pred, obs = _pred_times(J, P, st), _obs_times(J, P, st)
        chans = W[st]
        zc = [c for c in chans if c[-1] in "Z3"]
        hc = [c for c in chans if c[-1] in "EN12"]
        for j, ph in enumerate(("P", "S")):
            ax = axs[k, j]
            ts = [v for (p, h), v in pred.items() if p == ph] + ([obs[ph]] if ph in obs else [])
            if not ts:
                ax.axis("off")
                continue
            c0 = obs.get(ph, pred.get((ph, "v5.1"), np.median(ts)))
            lo = min(ts) - (1.5 if ph == "P" else 2.5)
            hi = max(ts) + (2.5 if ph == "P" else 3.5)
            lo, hi = min(lo, c0 - 1.5), max(hi, c0 + 2.0)
            cl = zc if (ph == "P" or not hc) else hc
            used = "Z" if cl is zc else "H"
            pcs = []
            for ch in cl[:2]:
                pcs += [(ch, p) for p in _prep(chans[ch], band, lo - 0.5, hi + 0.5)]
            if pcs:
                a = max(np.max(np.abs(yy[(t >= lo) & (t <= hi)])) if ((t >= lo) & (t <= hi)).any() else 0
                        for _, (t, yy) in pcs) or 1.0
                for i, (ch, (t, yy)) in enumerate(pcs):
                    ax.plot(t, 0.9 * yy / a, color=INK if ch == cl[0] else "#8a8985", lw=0.6)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#b3261e")
            for (p, h), v in pred.items():
                if p == ph:
                    ax.axvline(v, color=COL[h], lw=1.8, ls=LS[h])
            if ph in obs:
                ax.axvline(obs[ph], color=INK, lw=1.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([])
            ax.grid(axis="x", color=GRID, lw=0.4)
            r = P[(P.sta == st) & (P.phase == ph)]
            if len(r):
                txt = "obs-pred: " + ", ".join(f"{h} {r[f'resfit_{h}'].iloc[0]:+.2f}" for h in E.HYPS
                                               if f"resfit_{h}" in r and np.isfinite(r[f"resfit_{h}"].iloc[0]))
            else:
                txt = f"no {ph} pick; predicted ticks only"
            ax.set_title(f"{st} {dist[st]:.0f} km  {ph} on {used} ({'/'.join(c[-1] for c in cl[:2])})   {txt}",
                         loc="left", fontsize=7.2)
    for ax in axs[-1]:
        ax.set_xlabel("time after the v5.1 origin time (s)")
    fig.savefig(out, dpi=90, **SAVE)
    plt.close(fig)
    return True


def make_all(eid, outdir, force=False, raw=False):
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    figs = [("geom", fig_geometry), ("rec", fig_record), ("zoom", fig_zoom)] + ([("raw", fig_raw)] if raw else [])
    for name, fn in figs:
        f = outdir / f"{eid}_{name}.webp"
        if f.exists() and not force:
            res[name] = f.name
            continue
        ok = fn(eid, f)
        res[name] = f.name if (ok is None or ok) and f.exists() else None
    return res


if __name__ == "__main__":
    import sys
    from pathlib import Path
    od = Path(sys.argv[1])
    for e in sys.argv[2:]:
        print(e, make_all(int(e), od, force=True))


def fig_thumb(eid, out, nmax=12):
    """Compact record-section thumbnail for the viewer: vertical only, nearest stations with data, picks and predicted
    P/S ticks (v5.1 blue, v4 orange, ComCat aqua), no labels beyond the distance."""
    J, P, _ = _load(eid)
    W = E.load_waveforms(eid)
    band = tuple(J["band"])
    rel_lo, rel_hi = J["window_rel"]
    order, dist = _station_order(J, P)
    sts = [s for s in order if any(c[-1] in "Z3" for c in W.get(s, {}))][:nmax]
    n = max(len(sts), 1)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.97, bottom=0.12)
    for k, st in enumerate(sts):
        y0 = n - 1 - k
        zc = [c for c in W[st] if c[-1] in "Z3"][0]
        pcs = _prep(W[st][zc], band, rel_lo, rel_hi)
        a = _norm(pcs)
        for t, yy in pcs:
            ax.plot(t, y0 + 0.45 * yy / a, color=INK, lw=0.3)
        _ticks(ax, y0, _pred_times(J, P, st), _obs_times(J, P, st))
        ax.text(rel_lo, y0 + 0.25, f"{dist[st]:.0f}", fontsize=5, color=MUTED, ha="right")
    ax.set_xlim(rel_lo, rel_hi)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    for t in ax.texts:
        if t.get_text() in ("P", "S"):
            t.set_fontsize(4.5)
    fig.savefig(out, dpi=85, format="webp", pil_kwargs={"quality": 55, "method": 6})
    plt.close(fig)
