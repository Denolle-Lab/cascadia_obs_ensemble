"""Supplementary figure: perspective view of the Mendocino region with the fault surfaces v1 (PyGMT).

The relief is the GMRT tile of the Mendocino zoom (cfg gmrt_tiles; else the GMRT mosaic of the cache) drawn with
grdview in the paper's elevation colours; hypocentres and fault surfaces are drawn on a flat overlay in page
coordinates (GMT's parallel projection, as in v4_figures.perspective), so events are discs facing the viewer and
everything is ordered back to front. Surfaces are coloured by dip with the colour map of Fig. 3; events are grey
(darker: fit members of a surface). Size from paper_style.msize.
"""
from __future__ import annotations

import glob
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import paper_style as S
from paper_v2 import data as D
from paper_v2.f3_faults import CM_DIP, DIP_NORM

REGION = [-125.6, -123.2, 39.6, 41.4]
VIEW = dict(az=210.0, el=26.0, zmax=40.0, exag=1.5, width=12.0)


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c[:3])


def relief_grid(cfg, region, max_nodes=520):
    import xarray as xr
    W, E, Sth, N = region
    best = None
    for f in glob.glob(str(Path(cfg["gmrt_tiles"]) / "gmrt_*.nc")):
        w, e, s, n = map(float, Path(f).stem.split("_")[1:5])
        if w <= W + 0.02 and e >= E - 0.02 and s <= Sth + 0.02 and n >= N - 0.02 and \
                (best is None or (e - w) * (n - s) < best[0]):
            best = ((e - w) * (n - s), f)
    if best:
        ds = xr.open_dataset(best[1])
        z = ds[list(ds.data_vars)[0]].sel(lon=slice(W, E))
        z = z.sel(lat=slice(Sth, N)) if z.lat[0] < z.lat[-1] else z.sel(lat=slice(N, Sth)).isel(lat=slice(None, None, -1))
        src = Path(best[1]).name
    else:
        R = S.Relief(cfg["cache"])
        i = (R.lat >= Sth) & (R.lat <= N)
        j = (R.lon >= W) & (R.lon <= E)
        z = xr.DataArray(R.z[np.ix_(i, j)], coords={"lat": R.lat[i], "lon": R.lon[j]}, dims=("lat", "lon"))
        src = "GMRT mosaic (cache)"
    sx = max(1, int(np.ceil(z.sizes["lon"] / max_nodes)))
    sy = max(1, int(np.ceil(z.sizes["lat"] / max_nodes)))
    if sx > 1 or sy > 1:
        z = z.coarsen(lon=sx, lat=sy, boundary="trim").mean()
    return z.astype("float32").rename("z"), src


def relief_cpt(path):
    """The paper elevation colours (paper_style._SEA, _LAND) as a GMT CPT in km."""
    rows = []
    for stops in (S._SEA, S._LAND):
        for (z0, c0), (z1, c1) in zip(stops[:-1], stops[1:]):
            f = lambda c: "/".join(str(int(round(255 * v))) for v in c)  # noqa: E731
            rows.append(f"{z0 / 1000:g} {f(c0)} {z1 / 1000:g} {f(c1)}")
    f = lambda c: "/".join(str(int(round(255 * v))) for v in c)  # noqa: E731
    Path(path).write_text("\n".join(rows) + f"\nB {f(S._SEA[0][1])}\nF {f(S._LAND[-1][1])}\nN white\n")
    return str(path)


def dip_cpt(path, n=12):
    v = np.linspace(DIP_NORM.vmin, DIP_NORM.vmax, n + 1)
    rows = []
    for a, b in zip(v[:-1], v[1:]):
        ca, cb = (CM_DIP(DIP_NORM(x)) for x in (a, b))
        rows.append(f"{a:g} {'/'.join(str(int(round(255 * t))) for t in ca[:3])} {b:g} "
                    f"{'/'.join(str(int(round(255 * t))) for t in cb[:3])}")
    Path(path).write_text("\n".join(rows) + "\n")
    return str(path)


def hillshade(z, amp=0.22):
    import pygmt
    g = pygmt.grdgradient(z, azimuth=315)
    v = g.values
    med, sd = np.nanmedian(v), np.nanstd(v)
    g.values = (amp * np.tanh((v - med) / (2 * sd if sd > 0 else 1))).astype("float32")
    return g


def view_projector(r3, width, zsize, az, el):
    """lon, lat, z (km, up) -> page x, y (cm, from the lower-left of the projected box) and distance from the viewer
    (GMT's parallel view, as in v4_figures.view_projector)."""
    W, E, Sth, N, z0, z1 = r3
    m = lambda la: np.log(np.tan(np.pi / 4 + np.radians(np.asarray(la, float)) / 2))  # noqa: E731
    sa, ca, se, ce = np.sin(np.radians(az)), np.cos(np.radians(az)), np.sin(np.radians(el)), np.cos(np.radians(el))

    def raw(lon, lat, z):
        x = width * (np.asarray(lon, float) - W) / (E - W)
        y = width * (m(lat) - m(Sth)) / np.radians(E - W)
        zc = zsize * (np.asarray(z, float) - z0) / (z1 - z0)
        return -x * ca + y * sa, -(x * sa + y * ca) * se + zc * ce, -(x * sa + y * ca) * ce - zc * se
    cx, cy, _ = raw(*np.array([[x, y, z] for x in (W, E) for y in (Sth, N) for z in (z0, z1)]).T)

    def page(lon, lat, z):
        px, py, far = raw(lon, lat, z)
        return px - cx.min(), py - cy.min(), far
    return page, float(cx.max() - cx.min()), float(cy.max() - cy.min())


def make(cfg):
    import pygmt
    S.use()
    tmp = Path(tempfile.mkdtemp())
    W, E, Sth, N = REGION
    az, el, zmax, exag, width = (VIEW[k] for k in ("az", "el", "zmax", "exag", "width"))
    lat0 = (Sth + N) / 2
    km_w = (E - W) * D.KM * np.cos(np.radians(lat0))
    z, src = relief_grid(cfg, REGION)
    z = z / 1000.0
    ztop = max(0.5, float(z.max()))
    zsize = (zmax + ztop) * exag * width / km_w
    r3 = [W, E, Sth, N, -zmax, ztop]
    fig = pygmt.Figure()
    S.gmt_config()
    shf = tmp / "shade.nc"
    hillshade(z * 1000).to_netcdf(shf)
    fig.grdview(z, region=r3, projection=f"M{width}c", zsize=f"{zsize:.3f}c", perspective=[az, el], surftype="i",
                cmap=relief_cpt(tmp / "relief.cpt"), shading=str(shf), transparency=30,
                frame=["xa1f0.5", "ya1f0.5", "za10f5+lelevation (km)", "wSEnZ+b"])
    page, pw, ph = view_projector(r3, width, zsize, az, el)
    # fault surfaces v1 of the region, events
    fdir = cfg.path("faults")
    summ = pd.read_csv(fdir / "fault_surfaces_v1_summary.csv").set_index("surface_id")
    rect = {f["properties"]["surface_id"]: np.asarray(f["geometry"]["coordinates"][0], float)
            for f in json.loads((fdir / "fault_surfaces_v1_rectangles.geojson").read_text())["features"]}
    inf = {k: v for k, v in rect.items() if W <= v[:4, 0].mean() <= E and Sth <= v[:4, 1].mean() <= N and
           "deep population" not in str(summ.loc[k, "flags"])}
    mem = set(pd.read_csv(fdir / "fault_surfaces_v1_members.csv").event_id.astype(np.int64))
    ev = D.events(cfg)
    e = ev[ev.lon.between(W, E) & ev.lat.between(Sth, N) & ev.depth.between(-ztop, zmax)].copy()
    e["px"], e["py"], e["far"] = page(e.lon.values, e.lat.values, -e.depth.values)
    e["d"] = np.minimum(np.sqrt(S.msize(e.M, 0.85)) / 72 * 2.54, 0.40)          # cm
    e["on"] = e.ID.astype(np.int64).isin(mem)
    ov = dict(region=[0, 40, 0, 40], projection="x1c", no_clip=True)
    items = []
    for sid, v in inf.items():
        c4 = v[:4]
        cx, cy, cf = page(c4[:, 0], c4[:, 1], c4[:, 2] / 1000.0)
        top = np.argsort(-c4[:, 2])[:2]
        col = _hex(CM_DIP(DIP_NORM(summ.loc[sid, "dip"])))
        items.append((float(cf.mean()), "f", (cx, cy, cx[top], cy[top], col)))
    e = e.sort_values("far", ascending=False)
    for g in np.array_split(e, 16):
        if len(g):
            items.append((float(g.far.min()), "e", g))
    for _, kind, obj in sorted(items, key=lambda t: -t[0]):
        if kind == "f":
            cx, cy, tx, ty, c_ = obj
            fig.plot(x=np.r_[cx, cx[:1]], y=np.r_[cy, cy[:1]], fill=c_ + "@78", pen=f"0.35p,{c_}@20", **ov)
            fig.plot(x=tx, y=ty, pen=f"1.0p,{c_}", **ov)
            continue
        g = obj
        for on, h in g.groupby("on"):
            fig.plot(x=h.px, y=h.py, size=h.d, style="c", fill="#4a4a4a" if on else "#a8a8a8",
                     pen="0.1p,white", transparency=15 if on else 55, **ov)
    # labels: vertical exaggeration, dip colour bar, magnitude key
    bx, by, _ = page(np.array([E]), np.array([Sth]), np.array([-zmax]))
    fig.text(x=bx[0], y=by[0] - 0.35, text=f"vertical exaggeration {exag:g}:1", font=f"{S.FS['small']:g}p,Helvetica,{S.INK2}",
             justify="TR", **ov)
    # dip colour bar drawn on the overlay (GMT scales colorbar fonts on short bars)
    cbx, cby, cbw, cbh = 0.3, ph - 0.62, 3.2, 0.13
    nseg = 60
    for k in range(nseg):
        dv = DIP_NORM.vmin + (k + 0.5) / nseg * (DIP_NORM.vmax - DIP_NORM.vmin)
        x0k = cbx + k * cbw / nseg
        fig.plot(x=[x0k, x0k + cbw / nseg + 0.005, x0k + cbw / nseg + 0.005, x0k],
                 y=[cby, cby, cby + cbh, cby + cbh], fill=_hex(CM_DIP(DIP_NORM(dv))), close=True, **ov)
    fig.plot(x=[cbx, cbx + cbw, cbx + cbw, cbx, cbx], y=[cby, cby, cby + cbh, cby + cbh, cby],
             pen=f"0.4p,{S.AXIS}", **ov)
    for dv in (30, 45, 60, 75, 90):
        xx = cbx + (dv - DIP_NORM.vmin) / (DIP_NORM.vmax - DIP_NORM.vmin) * cbw
        fig.plot(x=[xx, xx], y=[cby, cby - 0.06], pen=f"0.4p,{S.AXIS}", **ov)
        fig.text(x=xx, y=cby - 0.1, text=f"{dv}", justify="TC", font=f"{S.FS['small']:g}p,Helvetica,{S.INK}", **ov)
    fig.text(x=cbx + cbw / 2, y=cby + cbh + 0.08, text="surface dip (@.)", justify="BC",
             font=f"{S.FS['small']:g}p,Helvetica,{S.INK}", **ov)
    mk = [2, 3, 4, 5]
    x0 = 4.6
    for i, mm in enumerate(mk):
        d = float(np.sqrt(S.msize(mm, 0.85)) / 72 * 2.54)
        fig.plot(x=[x0 + i * 0.45], y=[ph - 0.55], size=[d], style="c", fill="#a8a8a8", pen="0.1p,white", **ov)
        fig.text(x=x0 + i * 0.45, y=ph - 0.85, text=f"{mm}", font=f"{S.FS['small']:g}p,Helvetica,{S.INK}", **ov)
    fig.text(x=x0 - 0.3, y=ph - 0.55, text="M", font=f"{S.FS['small']:g}p,Helvetica,{S.INK}", justify="RM", **ov)
    fig.plot(x=[x0 + 2.2], y=[ph - 0.48], style="c0.1c", fill="#4a4a4a", **ov)
    fig.text(x=x0 + 2.35, y=ph - 0.48, text="fit member of a surface", justify="LM",
             font=f"{S.FS['small']:g}p,Helvetica,{S.INK}", **ov)
    fig.plot(x=[x0 + 2.2], y=[ph - 0.78], style="c0.1c", fill="#a8a8a8", **ov)
    fig.text(x=x0 + 2.35, y=ph - 0.78, text="other tier A-C event", justify="LM",
             font=f"{S.FS['small']:g}p,Helvetica,{S.INK}", **ov)
    name = "figS_perspective_mendocino"
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=S.DPI)
    from PIL import Image
    with Image.open(out / f"{name}.png") as im:
        size = (round(im.size[0] / S.DPI * 2.54, 2), round(im.size[1] / S.DPI * 2.54, 2))
    cap = (f"Perspective view of the Mendocino region ({-W:g}° to {-E:g}°W, {Sth:g}° to {N:g}°N, to {zmax:g} km depth) from "
           f"azimuth {az:g}°, elevation {el:g}°, vertical exaggeration {exag:g}:1. Relief: GMRT ({src}). Fault "
           f"surfaces v1 refit on catalog {cfg.get('label', '')} ({len(inf)} surfaces with their centre in the box): "
           "rectangle and top edge, coloured by dip. Tier A-C events of the catalog "
           f"({len(e):,}; dark grey: fit members of a surface); symbol size scales with magnitude.")
    return [dict(name=name, files=[f"{name}.pdf", f"{name}.png"], size_cm=size, caption=cap,
                 before=["perspective_mendocino_fault.png"], inputs=["faults", "events_segments", "gmrt_tiles", "cache"],
                 panels=dict(surfaces=len(inf), events=int(len(e))))]
