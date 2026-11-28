"""Fig. 4: fault-zone structure. Fault-normal sections stacked along strike for the resolved plate-boundary segments,
Vp and Vp/Vs of model v0.4.2 on resolved nodes, the ensemble 1 sigma as transparency, unresolved cells hatched grey,
the segment's depth-good seismicity, the 7.5 km/s contour, Slab2 and the contrasts that pass the controls.

Sections (configuration `sections`): a frame per panel (PB2002/D-B113 transform trace, the segment's strike line from
segment_axes.csv, or a fixed line); nodes of the model are binned by their distance normal to the frame (5 km bins,
the frame's along-strike extent of the segment's events) and by depth layer; a bin shows the mean of its resolved
nodes when it has at least two (rock and checkerboard-resolved or DWS >= 100; P for Vp, P and S for Vp/Vs, as in the
plate-boundary analysis). Its 1 sigma is the spread of the same bin mean over the 100 ensemble members (x 1.50 for
Vp, x 1.00 for Vp/Vs: D-B099, D-B111), i.e. the uncertainty of the value drawn. One horizontal scale for every panel,
vertical exaggeration 1.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import paper_style as S
from paper_v2 import data as D

VE = 1.5                         # vertical exaggeration of every section
ALPHA_MIN = 0.22                 # opacity of a bin at or above the upper sigma
SIG = {"Vp": (0.05, 0.25), "VpVs": (0.010, 0.050)}   # 1 sigma (bin mean) at full opacity / at ALPHA_MIN
MIN_NODES = 2
LEVEL = 7.5


# ------------------------------------------------------------------ frames
def _densify(xy, step_km=0.5):
    x, y = D.local_xy(xy[:, 0], xy[:, 1], -125.0, xy[:, 1].mean())
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    q = np.arange(0, s[-1] + 1e-9, step_km)
    return np.c_[np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])]


def _trace_pieces(sec, cfg, ctx):
    """Transform strands of a trace frame, west to east, each oriented west to east: the oceanic-transform sections of
    the plate-axes table (plate boundaries v2 adopted, D-B117) with at least half their points in the section's bbox;
    step-overs and connectors (sections '*_step*', the A7 join) are left out, so every point is measured from the
    nearest strand. Without a plate-axes table: the PB2002 transform polylines in the bbox."""
    W, Ea, Sb, N = sec["bbox"]
    inside = lambda xy: ((xy[:, 0] > W) & (xy[:, 0] < Ea) & (xy[:, 1] > Sb) & (xy[:, 1] < N)).mean()  # noqa: E731
    pieces = []
    f = cfg.opt("plate_axes")
    if f is not None:
        a = pd.read_csv(f)
        for name, g in a[a.kind == "OTF"].groupby("section", sort=False):
            if "_step" in str(name) or str(g.source.iloc[0]).startswith("A7"):
                continue
            xy = g[["lon", "lat"]].to_numpy(float)
            if inside(xy) > 0.5:
                pieces.append(xy)
    else:
        pieces = [L for L in ctx["pb"]["TF"] if inside(L) > 0.5]
    pieces = [p if p[0, 0] <= p[-1, 0] else p[::-1] for p in pieces]
    return sorted(pieces, key=lambda p: p[:, 0].mean())


class Frame:
    """Normal and along-strike coordinates (km) of points relative to a section frame; + normal = left of the frame
    direction, flipped when needed so that + x points into the eastern half (east on the right of every panel)."""

    def __init__(self, sec, cfg, ctx, E):
        self.sec = sec
        kind = sec.get("frame", "axis")
        self.kind = kind
        if kind == "trace":
            pieces = _trace_pieces(sec, cfg, ctx)
            self.lon0 = float(np.mean([p[:, 0].mean() for p in pieces]))
            self.lat0 = float(np.mean([p[:, 1].mean() for p in pieces]))
            TX, TY, DX, DY, SS, XY = [], [], [], [], [], []
            s0 = 0.0
            for p in pieces:                                   # strands west to east, each oriented west to east
                q = _densify(p)
                tx, ty = D.local_xy(q[:, 0], q[:, 1], self.lon0, self.lat0)
                if TX:                                         # jump across a step-over between strands
                    s0 += float(np.hypot(tx[0] - TX[-1][-1], ty[0] - TY[-1][-1]))
                ss = s0 + np.r_[0, np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))]
                dx, dy = np.gradient(tx), np.gradient(ty)
                nn = np.hypot(dx, dy)
                TX.append(tx); TY.append(ty); DX.append(dx / nn); DY.append(dy / nn); SS.append(ss); XY.append(q)
                s0 = float(ss[-1])
            tx, ty, self.dx, self.dy = np.concatenate(TX), np.concatenate(TY), np.concatenate(DX), np.concatenate(DY)
            self.s = np.concatenate(SS)
            self.tx, self.ty = tx, ty
            from scipy.spatial import cKDTree
            self.tree = cKDTree(np.c_[tx, ty])
            mid = len(tx) // 2
            self.az_plus = np.degrees(np.arctan2(-self.dy[mid], self.dx[mid])) % 360     # left normal (east, north)
            self.trace_xy = np.vstack(XY)
        else:
            if kind == "axis":
                ax = pd.read_csv(cfg.path("segment_axes")).set_index("seg").loc[sec["segs"][0]]
                self.lon0, self.lat0, ux, uy = float(ax.lon0), float(ax.lat0), float(ax.ux), float(ax.uy)
            else:
                self.lon0, self.lat0 = float(sec["lon0"]), float(sec["lat0"])
                st = np.radians(float(sec["strike"]))
                ux, uy = np.sin(st), np.cos(st)
            self.u = np.array([ux, uy])
            self.n = np.array([-uy, ux])
            self.az_plus = np.degrees(np.arctan2(self.n[0], self.n[1])) % 360
        self.flip = np.sin(np.radians(self.az_plus)) < 0
        if self.flip:
            self.az_plus = (self.az_plus + 180) % 360
        self.offset = 0.0
        # along-strike extent and centring from the segment's depth-good events
        ev = E[E.seg.isin(sec["segs"])]
        nrm, alg, ins = self.coords(ev.lon.values, ev.lat.values, raw=True)
        if kind == "trace":
            sel = ins & (np.abs(nrm) <= 45)
            self.offset = float(np.median(nrm[sel]))
        if "along" in sec:
            self.a0, self.a1 = map(float, sec["along"])
        else:
            a = alg[ins]
            self.a0, self.a1 = np.percentile(a, [2, 98]) if len(a) else (0.0, 0.0)
        self.offset += float(sec.get("centre", 0.0)) * (-1 if self.flip else 1)

    def coords(self, lon, lat, raw=False):
        px, py = D.local_xy(lon, lat, self.lon0, self.lat0)
        if self.kind == "trace":
            d, i = self.tree.query(np.c_[px, py])
            vx, vy = px - self.tx[i], py - self.ty[i]
            along = self.s[i] + vx * self.dx[i] + vy * self.dy[i]
            normal = -vx * self.dy[i] + vy * self.dx[i]
            inside = (along >= 0) & (along <= self.s[-1])
        else:
            along = px * self.u[0] + py * self.u[1]
            normal = px * self.n[0] + py * self.n[1]
            inside = np.ones(np.shape(px), bool)
        if not raw:
            normal = normal - self.offset
            inside = inside & (along >= self.a0) & (along <= self.a1)
            if self.flip:
                normal = -normal
        return normal, along, inside

    @property
    def length(self):
        return self.a1 - self.a0


# ------------------------------------------------------------------ binning
def depth_edges(dep):
    e = np.r_[dep[0] - (dep[1] - dep[0]) / 2, 0.5 * (dep[1:] + dep[:-1]), dep[-1] + (dep[-1] - dep[-2]) / 2]
    return e


def section_grids(M, members, fr, xlim, binw):
    """Per (depth layer, normal bin): mean of resolved nodes, member-spread 1 sigma, node counts, rock-only count."""
    nb_edges = np.arange(xlim[0], xlim[1] + 1e-9, binw)
    nb = len(nb_edges) - 1
    x, _, ins = fr.coords(M.LO.ravel(), M.LA.ravel())
    j = np.digitize(x, nb_edges) - 1
    j[~ins | (j < 0) | (j >= nb)] = -1
    ok_col = j >= 0
    nz = len(M.dep)
    out = {}
    for var, mask in (("Vp", M.mP), ("VpVs", M.mPS)):
        F = M.field(var).reshape(nz, -1)
        mk = mask.reshape(nz, -1)
        rock = M.rock.reshape(nz, -1)
        mem = members[var].reshape(members[var].shape[0], nz, -1)
        val = np.full((nz, nb), np.nan)
        sd = np.full((nz, nb), np.nan)
        cnt = np.zeros((nz, nb), int)
        rk = np.zeros((nz, nb), int)
        infl = float(M.infl[var])
        for k in range(nz):
            sel = ok_col & mk[k] & np.isfinite(F[k])
            jj = j[sel]
            cnt[k] = np.bincount(jj, minlength=nb)
            rk[k] = np.bincount(j[ok_col & rock[k]], minlength=nb)
            s = np.bincount(jj, weights=F[k][sel], minlength=nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                val[k] = np.where(cnt[k] >= MIN_NODES, s / cnt[k], np.nan)
            mv = mem[:, k][:, sel]                                       # members x nodes
            fin = np.isfinite(mv)
            onehot = np.zeros((len(jj), nb))
            onehot[np.arange(len(jj)), jj] = 1.0
            msum = np.nan_to_num(mv) @ onehot
            mcnt = fin.astype(float) @ onehot
            with np.errstate(invalid="ignore", divide="ignore"):
                mmean = np.where(mcnt >= MIN_NODES, msum / mcnt, np.nan)
            good = np.isfinite(mmean).sum(0) >= 50
            sd[k] = np.where(good & (cnt[k] >= MIN_NODES), np.nanstd(mmean, axis=0, ddof=1) * infl, np.nan)
        out[var] = dict(val=val, sd=sd, cnt=cnt, rock=rk)
    out["edges"] = nb_edges
    return out


def frame_grid(fr, xlim, step_deg=0.01, pad=0.25):
    """A lon/lat grid covering the frame's stacked area (for profiles of the relief and of Slab2)."""
    if fr.kind == "trace":
        W, E = fr.trace_xy[:, 0].min() - 0.7, fr.trace_xy[:, 0].max() + 0.7
        Sb, N = fr.trace_xy[:, 1].min() - 0.7, fr.trace_xy[:, 1].max() + 0.7
    else:
        xs = np.array([xlim[0], xlim[1], xlim[0], xlim[1]]) * (-1 if fr.flip else 1) + fr.offset
        al = np.array([fr.a0, fr.a0, fr.a1, fr.a1])
        px = al * fr.u[0] + xs * fr.n[0]
        py = al * fr.u[1] + xs * fr.n[1]
        lon = fr.lon0 + px / (D.KM * np.cos(np.radians(fr.lat0)))
        lat = fr.lat0 + py / D.KM
        W, E, Sb, N = lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad
    LO, LA = np.meshgrid(np.arange(W, E, step_deg), np.arange(Sb, N, step_deg))
    return LO.ravel(), LA.ravel()


def surface_profile(fr, relief, xlim, step=1.0, win=1.5):
    """Median solid-surface elevation (km, + up) against normal distance over the stacked along-strike range."""
    from scipy.interpolate import RegularGridInterpolator
    xs = np.arange(xlim[0], xlim[1] + 1e-9, step)
    LO, LA = frame_grid(fr, xlim)
    x, _, ins = fr.coords(LO, LA)
    sel = ins & (x >= xlim[0] - win) & (x <= xlim[1] + win)
    f = RegularGridInterpolator((relief.lat, relief.lon), relief.z, bounds_error=False, fill_value=np.nan)
    z = f(np.c_[LA[sel], LO[sel]]) / 1000.0
    x = x[sel]
    o = np.argsort(x)
    x, z = x[o], z[o]
    out = np.full(len(xs), np.nan)
    for i, xv in enumerate(xs):
        a, b = np.searchsorted(x, [xv - win, xv + win])
        if b - a >= 3:
            out[i] = np.nanmedian(z[a:b])
    return xs, out


def slab_profile(fr, slab, xlim, step=2.0):
    xs = np.arange(xlim[0], xlim[1] + 1e-9, step)
    LO, LA = frame_grid(fr, xlim, step_deg=0.02)
    x, _, ins = fr.coords(LO, LA)
    sel = ins & (x >= xlim[0] - step) & (x <= xlim[1] + step)
    z = slab(LO[sel], LA[sel])
    x = x[sel]
    out = np.full(len(xs), np.nan)
    for i, xv in enumerate(xs):
        m = np.abs(x - xv) <= step / 2
        if m.sum() >= 5 and np.isfinite(z[m]).mean() >= 0.5:
            out[i] = np.nanmedian(z[m])
    return xs, out


# ------------------------------------------------------------------ drawing
def blend(rgba, sd, s0, s1):
    """Opacity from 1 sigma: 1 at s0 or below, ALPHA_MIN at s1 or above; composited on white (vector-safe)."""
    a = np.clip(1 - (sd - s0) / (s1 - s0) * (1 - ALPHA_MIN), ALPHA_MIN, 1.0)
    a = np.where(np.isfinite(sd), a, ALPHA_MIN)
    rgb = rgba[..., :3] * a[..., None] + (1 - a[..., None])
    return np.dstack([rgb, np.ones(a.shape)]), a


def draw_panel(ax, G, var, ze, fr, ev, xlim, zlim, surf, slab, vp_grid, show_x, show_y, cmap, norm):
    g = G[var]
    edges = G["edges"]
    nz, nb = g["val"].shape
    # water and air above the solid surface
    xs, zs = surf
    ax.add_patch(Rectangle((xlim[0], zlim[1]), xlim[1] - xlim[0], zlim[0] - zlim[1], facecolor="white", lw=0, zorder=0))
    sea = np.where(np.isfinite(zs), np.minimum(zs, 0), np.nan)
    ax.fill_between(xs, 0, -sea, where=np.isfinite(sea) & (sea < 0), color=S.WATER, lw=0, zorder=0.5)
    # unresolved rock: hatched grey; resolved: colour faded by 1 sigma
    xc_ = 0.5 * (edges[1:] + edges[:-1])
    zsurf = -np.interp(xc_, xs[np.isfinite(zs)], zs[np.isfinite(zs)]) if np.isfinite(zs).any() else np.zeros(nb)
    zc_ = 0.5 * (ze[1:] + ze[:-1])
    unres = (g["rock"] > 0) & ~np.isfinite(g["val"]) & (zc_[:, None] > zsurf[None, :])
    for k in range(nz):
        for b in np.flatnonzero(unres[k]):
            ax.add_patch(Rectangle((edges[b], ze[k]), edges[b + 1] - edges[b], ze[k + 1] - ze[k], facecolor="#ebebeb",
                                   edgecolor="#c4c4c4", hatch="//////", lw=0, zorder=1))
    rgba = cmap(norm(np.ma.masked_invalid(g["val"])))
    img, _ = blend(rgba, g["sd"], *SIG[var])
    img[~np.isfinite(g["val"])] = [1, 1, 1, 0]
    _cells(ax, img, edges, ze)
    # surface line
    S.cased(ax, xs, -zs, color="#6b6b6b", lw=0.5, casing=0.8, zorder=2)
    # seismicity
    alpha = float(np.clip(2000.0 / max(len(ev[0]), 1), 0.07, 0.3))       # dense clouds stay see-through
    ax.scatter(ev[0], ev[1], s=0.3, c=S.INK, alpha=alpha, lw=0, zorder=3, rasterized=True)
    # 7.5 km/s contour of the Vp section (resolved bins only)
    xc = 0.5 * (edges[1:] + edges[:-1])
    zc = 0.5 * (ze[1:] + ze[:-1])
    try:
        import contourpy
        gen = contourpy.contour_generator(xc, zc, np.ma.masked_invalid(vp_grid), name="serial", line_type="Separate")
        segs = [s for s in gen.lines(LEVEL) if len(s) > 2]
        # closed loops around a single bin (x extent < 10 km) are left out: a one-bin island, not a level
        segs = [s for s in segs if not (np.allclose(s[0], s[-1]) and np.ptp(s[:, 0]) < 10.0)]
    except Exception:  # noqa: BLE001
        segs = []
    for s in segs:
        S.cased(ax, s[:, 0], s[:, 1], color=S.INK, lw=S.LW["contour"], casing=1.0, zorder=4)
    if segs:
        s = max(segs, key=lambda q: np.ptp(q[:, 0]))
        i = int(np.argmax(s[:, 0]))
        right = s[i, 0] > xlim[1] - 0.08 * (xlim[1] - xlim[0])
        S.text_halo(ax, s[i, 0] + (-0.8 if right else 0.8), s[i, 1] - (1.2 if right else 0), "7.5", fs=S.FS["small"],
                    ha="right" if right else "left", va="bottom" if right else "center", zorder=6)
    # Slab2
    sx, sz = slab
    if np.isfinite(sz).sum() >= 3:
        S.cased(ax, sx, sz, color=S.INK, lw=S.LW["slab"], ls=(0, (4, 1.6)), casing=1.0, zorder=4)
        k = np.flatnonzero(np.isfinite(sz) & (sz < zlim[0] - 4) & (sz > zlim[1] + 4))
        if len(k):
            i = k[len(k) // 2]
            S.text_halo(ax, sx[i], sz[i] - 1.6, "Slab2", fs=S.FS["small"], ha="center", va="bottom", zorder=6,
                        rotation=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(zlim[0], zlim[1])
    ax.axvline(0, color="#8c8c8c", lw=0.4, ls=(0, (1, 2)), zorder=2.5)
    S.boxed(ax)
    ax.set_xticks(np.arange(np.ceil(xlim[0] / 20) * 20, xlim[1] + 1e-9, 20))
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.tick_params(labelbottom=show_x, labelleft=show_y, length=2.0)
    return segs


def _cells(ax, img, edges, ze):
    """RGBA cells on non-uniform depth layers: one image strip per layer (transparent where nothing is shown)."""
    for k in range(img.shape[0]):
        if img[k, :, 3].max() == 0:
            continue
        ax.imshow(img[k:k + 1], extent=[edges[0], edges[-1], ze[k + 1], ze[k]], origin="upper", aspect="auto",
                  interpolation="nearest", zorder=1.5)


def chips(ax, rows, x=0.015, y=0.975):
    """Contrasts that pass the controls, one line each, bottom-left of the panel."""
    if not rows:
        return
    lines = []
    for r in rows:
        sym = "●" if r["claimed"] else "○"
        unit = "" if r["var"] == "VpVs" else " km/s"
        nm = {"Vp": "Vp", "Vs": "Vs", "VpVs": "Vp/Vs"}[r["var"]]
        seg = f"{r['seg']} " if r.get("show_seg") else ""
        lines.append(f"{sym} {seg}{nm} {r['contrast'].replace('near-far', 'near−far')} {r['D_real']:+.2f}{unit}")
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, fontsize=S.FS["small"], ha="left", va="top",
            color=S.INK, zorder=8, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc="white", ec="none", alpha=0.82))


def sigma_key(fig, x, y, w, h, var, cmap, norm, title):
    """Two-dimensional key: colour (value) across, opacity (1 sigma) down."""
    ax = S.axes_cm(fig, x, y, w, h)
    lo, hi = (S.VP_RANGE if var == "Vp" else (S.VPVS_RANGE[0], S.VPVS_RANGE[2]))
    v = np.linspace(lo, hi, 120)
    s0, s1 = SIG[var]
    sig = np.linspace(0, s1 * 1.1, 6)
    rgba = cmap(norm(v))[None, :, :].repeat(len(sig), 0)
    img, _ = blend(rgba, sig[:, None] * np.ones((1, len(v))), s0, s1)
    ax.imshow(img, extent=[lo, hi, sig[-1] + (sig[1] - sig[0]) / 2, -(sig[1] - sig[0]) / 2], aspect="auto",
              origin="upper", interpolation="nearest")
    S.boxed(ax)
    ax.set_title(title, fontsize=S.FS["small"], loc="left", pad=2)
    ax.set_xticks(np.round(np.linspace(lo, hi, 3), 2))
    ax.set_yticks([s0, s1])
    ax.set_yticklabels([f"≤ {s0:g}", f"≥ {s1:g}"])
    ax.set_ylabel("1σ", fontsize=S.FS["small"], labelpad=1)
    ax.tick_params(length=1.6, pad=1)
    return ax


# ------------------------------------------------------------------ figure
def make(cfg):
    ctx = D.context(cfg)
    E_all = D.events(cfg)
    E = E_all[E_all.depth_good]
    M = D.Model(cfg)
    M.infl = cfg.get("inflation", {"Vp": 1.5, "VpVs": 1.0})
    names, members = D.load_members(cfg)
    slab = D.slab_fn(cfg)
    ct = D.contrasts(cfg)
    null_scored = bool(ct.null_scored.iloc[0])
    secs = cfg["sections"]
    half = float(cfg.get("section_half_width_km", 45))
    zlim = (float(cfg.get("section_depth_km", [-3, 40])[1]), float(cfg.get("section_depth_km", [-3, 40])[0]))
    binw = float(cfg.get("section_bin_km", 5))
    ze = depth_edges(M.dep)

    # layout: two blocks of sections (Vp row over Vp/Vs row), one horizontal scale for all
    xr = []
    for sc in secs:
        if "xlim" in sc:
            xr.append(tuple(map(float, sc["xlim"])))
        else:
            xr.append((-half, half))
    blocks = [secs[:3], secs[3:]]
    widths_km = [sum(r[1] - r[0] for r in xr[:3]), sum(r[1] - r[0] for r in xr[3:])]
    L, R, GAP = 1.05, 0.25, 0.32
    W = S.COL2
    kmcm = (W - L - R - GAP * (max(len(b) for b in blocks) - 1)) / max(widths_km)
    ph = (zlim[0] - zlim[1]) * kmcm * VE                    # panel height (cm)
    title_h, vgap, rowgap, bottom = 0.62, 0.12, 0.95, 0.75
    H = bottom + 2 * (2 * ph + vgap + title_h) + rowgap + 0.25
    fig = S.fig_cm(W, H)
    cmaps = {"Vp": (S.CM_VP, Normalize(*S.VP_RANGE)), "VpVs": (S.CM_VPVS, S.norm_vpvs())}
    facts = {}
    letters = iter("ABCDEFGH")
    idx = 0
    for bi, blk in enumerate(blocks):
        y_top = H - 0.25 - bi * (2 * ph + vgap + title_h + rowgap)
        x = L
        for sc in blk:
            xlim = xr[idx]
            idx += 1
            fr = Frame(sc, cfg, ctx, E)
            G = section_grids(M, members, fr, xlim, float(sc.get("bin_km", binw)))
            surf = surface_profile(fr, ctx["relief"], xlim)
            sl = slab_profile(fr, slab, xlim)
            en, ea, ei = fr.coords(E.lon.values, E.lat.values)
            m = ei & E.seg.isin(sc["segs"]).values & (en >= xlim[0]) & (en <= xlim[1])
            ev = (en[m], E.depth.values[m])
            w = (xlim[1] - xlim[0]) * kmcm
            axes = []
            for r, var in enumerate(("Vp", "VpVs")):
                ax = S.axes_cm(fig, x, y_top - title_h - (r + 1) * ph - r * vgap, w, ph)
                cm, nm = cmaps[var]
                draw_panel(ax, G, var, ze, fr, ev, xlim, zlim, surf, sl, G["Vp"]["val"], show_x=(r == 1),
                           show_y=(x == L), cmap=cm, norm=nm)
                rows = []
                for sg in sc["segs"]:
                    for v in (("Vp",) if var == "Vp" else ("VpVs", "Vs")):
                        q = ct[(ct.seg == sg) & (ct["var"] == v)]
                        ok = q.status.isin(["claimed", "survives (null pending)"])
                        for _, rr in q[ok].iterrows():
                            rows.append(dict(seg=sg, var=v, contrast=rr.contrast, D_real=rr.D_real,
                                             claimed=rr.status == "claimed", show_seg=len(sc["segs"]) > 1))
                chips(ax, rows)
                if x == L:
                    ax.set_ylabel("depth (km)")
                if r == 1:
                    ax.set_xlabel(f"{sc.get('xlabel', 'distance')} (km), {D.compass(fr.az_plus + 180)} – "
                                  f"{D.compass(fr.az_plus)}", labelpad=1.5)
                lab = "Vp" if var == "Vp" else "Vp/Vs"
                S.text_halo(ax, 0.985, 0.04 if var == "Vp" and not rows else 0.04, lab, fs=S.FS["small"],
                            transform=ax.transAxes, ha="right", va="bottom", zorder=9, fontweight="bold")
                axes.append(ax)
            # compass ends and title
            for side, xa, ha in ((D.compass(fr.az_plus + 180), 0.0, "left"), (D.compass(fr.az_plus), 1.0, "right")):
                axes[0].text(xa, 1.02, side, transform=axes[0].transAxes, fontsize=S.FS["small"], ha=ha, va="bottom",
                             color=S.INK2)
            n_ev = int(m.sum())
            sub = f"{fr.length:.0f} km stacked · {n_ev:,} events"
            fig.text((x + w / 2) / W, (y_top - 0.02) / H, sc["title"], fontsize=S.FS["base"], ha="center", va="top",
                     fontweight="bold")
            fig.text((x + w / 2) / W, (y_top - 0.30) / H, sub, fontsize=S.FS["small"], ha="center", va="top",
                     color=S.INK2)
            S.letter(axes[0], next(letters), dx=-0.75 if x == L else -0.1, dy=0.42)
            facts[sc["key"]] = dict(length_km=round(fr.length, 1), events=n_ev, offset_km=round(fr.offset, 1),
                                    plus=D.compass(fr.az_plus), frame=fr.kind,
                                    vp_bins=int(np.isfinite(G["Vp"]["val"]).sum()),
                                    vpvs_bins=int(np.isfinite(G["VpVs"]["val"]).sum()),
                                    vp_sd_med=float(np.nanmedian(G["Vp"]["sd"])),
                                    vpvs_sd_med=float(np.nanmedian(G["VpVs"]["sd"])))
            x += w + GAP

    # keys in the free space right of the first block
    x0 = L + widths_km[0] * kmcm + GAP * (len(blocks[0]) - 1) + 0.95
    kw = W - x0 - 0.2
    ytop = H - 0.25 - title_h
    sigma_key(fig, x0, ytop - 1.05, kw, 0.78, "Vp", S.CM_VP, Normalize(*S.VP_RANGE), "Vp (km/s)")
    sigma_key(fig, x0, ytop - 2.5, kw, 0.78, "VpVs", S.CM_VPVS, S.norm_vpvs(), "Vp/Vs")
    lg = S.axes_cm(fig, x0 - 0.55, ytop - 2 * ph - vgap - 0.1, kw + 0.6, 2 * ph + vgap - 2.85)
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    lg.axis("off")
    hgt = lg.get_position().height * H
    step = 0.42 / hgt
    yy = 1.0 - 0.5 * step
    lg.add_patch(Rectangle((0.0, yy - 0.3 * step), 0.16, 0.6 * step, facecolor="#ebebeb", edgecolor="#b9b9b9",
                           hatch="//////", lw=0))
    lg.text(0.22, yy, "unresolved", fontsize=S.FS["small"], va="center")
    yy -= step
    lg.scatter([0.03, 0.08, 0.12], [yy, yy + 0.15 * step, yy - 0.12 * step], s=1.2, c=S.INK, alpha=0.6, lw=0)
    lg.text(0.22, yy, "events (depth-good)", fontsize=S.FS["small"], va="center")
    yy -= step
    lg.plot([0.0, 0.16], [yy, yy], color=S.INK, lw=S.LW["contour"])
    lg.text(0.22, yy, "Vp 7.5 km/s", fontsize=S.FS["small"], va="center")
    yy -= step
    lg.plot([0.0, 0.16], [yy, yy], color=S.INK, lw=S.LW["slab"], ls=(0, (4, 1.6)))
    lg.text(0.22, yy, "Slab2", fontsize=S.FS["small"], va="center")
    yy -= step
    lg.add_patch(Rectangle((0.0, yy - 0.3 * step), 0.16, 0.6 * step, facecolor=S.WATER, edgecolor="none"))
    lg.text(0.22, yy, "water", fontsize=S.FS["small"], va="center")
    yy -= 1.1 * step
    lg.text(0.0, yy, ("● contrast: 2σ, S95 and\n    null-truth N95" if null_scored else
                      "○ contrast: 2σ and S95\n    (null truth pending)"), fontsize=S.FS["small"], va="top")
    yy -= 2.3 * step
    lg.text(0.0, yy, f"vertical exaggeration {VE:g}:1\nmodel {M.version}, catalog {cfg['label']}",
            fontsize=S.FS["small"], va="top", color=S.INK2)
    size = S.save(fig, cfg["out"], "fig4_fault_zone_sections")
    n_claim = int((ct.status == "claimed").sum())
    n_surv = int(ct.survives.sum())
    cap = (f"Fault-zone structure of model {M.version} at the resolved plate-boundary segments. Each panel pair is a "
           "section normal to the segment (" + ', '.join(s['key'] for s in secs) + "), stacked along strike over the "
           "extent of the segment's events (length above each panel) and binned at "
           f"{binw:g} km in distance (" + ", ".join(f"{q['key']} {q['bin_km']:g} km" for q in secs if "bin_km" in q) +
           ": few resolved nodes per 5 km bin) and by depth layer. Top: Vp; bottom: Vp/Vs. Colour: the mean of the resolved nodes "
           f"of a bin (at least {MIN_NODES}; rock and checkerboard-resolved or DWS ≥ {cfg.get('dws_min', 100)}; P for "
           "Vp, P and S for Vp/Vs). Opacity: 1σ of the same bin mean over the "
           f"{len(names)} ensemble members (× {M.infl['Vp']:g} for Vp, × {M.infl['VpVs']:g} for Vp/Vs); full colour at "
           f"≤ {SIG['Vp'][0]:g} km/s and ≤ {SIG['VpVs'][0]:g}, faintest at ≥ {SIG['Vp'][1]:g} km/s and ≥ "
           f"{SIG['VpVs'][1]:g}. Hatched: rock bins without resolved nodes. Pale blue: water. Dots: depth-good tier A-C "
           f"events of the segment (catalog {cfg['label']}). Solid line: Vp 7.5 km/s (closed loops around a single bin left out); dashed: Slab2. Text: depth-matched "
           "contrasts of the segment that pass " +
           ("the controls (2σ of the ensemble, synthetic S95 and the null truth N95; D-B111, D-B112)" if null_scored
            else "2σ of the ensemble and the synthetic S95 (D-B111); the null-truth control (D-B112) is pending") +
           f". Frames: the transform strands of the plate boundaries v2 (D-B117; distance from the nearest strand, step-overs "
           "and connectors left out) re-centred on the seismicity (BL, MFZ), the strike line of the "
           "segment's epicentres (GP, PU, MSH), a line through the PB2002 triple junction along the trench azimuth (344.5°; "
           "triple junction and Gorda slab). "
           f"Vertical exaggeration {VE:g}. The model resolves structure on a 40 km scale, not a fault damage zone.")
    return [dict(name="fig4_fault_zone_sections", files=["fig4_fault_zone_sections.pdf", "fig4_fault_zone_sections.png"],
                 size_cm=size, caption=cap, before=[], before_plan=["fig_sections_BL_MFZ.png"], facts=facts,
                 contrasts=dict(survives_DB111=n_surv, claimed=n_claim, null_scored=null_scored),
                 inputs=["model", "ensemble", "members", "events_segments", "segment_axes", "circularity_contrasts",
                         "null_truth_contrasts", "slab2", "plate_axes"])]
