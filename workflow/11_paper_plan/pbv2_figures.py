#!/usr/bin/env python3
"""Plate boundaries v2 (D-B116): figures and the comparison with other boundary models.

  BIG/figures/map_proposal.png        PB2002 vs the proposal over the shaded D-B068 surface (GMRT where available),
                                      v6.1 S0 seismicity, deformation zones, corners and triple junction (PyGMT)
  BIG/figures/zoom_<name>.png         section zooms (+ 68 % band, bathymetric picks, MAPRIDGES, Blanco strands)
  BIG/figures/models.png              proposal vs PB2002, NNR-MORVEL56, Hasterok et al. 2022, MAPRIDGES v2602
  BIG/figures/profiles.png            per-section along-strike view: picks, bootstrap band, segments, references
  BIG/figures/bias.png                offsets from PB2002 and the bias terms per section
  OUT/model_comparison.csv            median offset of each model from the proposal, per section
Colours: Okabe-Ito (colourblind-safe, D-B104), each layer also has its own line style.
Usage: python -I pbv2_figures.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pbv2_common as C  # noqa: E402
from pbv2_common import SECTIONS, log  # noqa: E402

FIG = C.BIG / "figures"
VER, BLU, GRN, PUR, ORA, SKY, YEL = "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"
ZOOMS = {
    "mendocino": ("Mendocino transform and triple junction", (-127.9, -124.25, 40.12, 40.95)),
    "gorda_south": ("Southern Gorda Ridge and the GR-MFZ corner", (-128.0, -126.8, 40.3, 41.95)),
    "gorda_north": ("Northern Gorda Ridge and the GR-Blanco corner", (-127.7, -126.2, 41.55, 43.45)),
    "blanco": ("Blanco transform: three segments, two step-overs", (-130.6, -126.4, 42.85, 44.7)),
    "jdf": ("Juan de Fuca Ridge: Cleft, Axial, Endeavour", (-130.75, -128.55, 44.35, 48.6)),
    "explorer": ("Explorer region (not resolved)", (-131.3, -126.8, 48.3, 50.9)),
}


def load():
    gj = json.loads((C.BIG / "proposed_boundaries_v1.geojson").read_text())
    with open(C.BIG / "work" / "fit.pkl", "rb") as fh:
        F = pickle.load(fh)
    cat = C.load_v61()
    return gj, F, cat


def topo_da(region, pad=0.05):
    lon, lat, z = C.topo_grid()
    a, b, c, d = region
    i = (lon >= a - pad) & (lon <= b + pad)
    j = (lat >= c - pad) & (lat <= d + pad)
    return xr.DataArray(z[np.ix_(j, i)], coords=dict(lat=lat[j], lon=lon[i]), dims=("lat", "lon"))


def bands(F, adopted=None):
    """68 % across-strike band polygons (lon/lat) for each proposed (or adopted) segment."""
    out = []
    for code, r in F["res"].items():
        sec = SECTIONS[code]
        hw_all = (r["band"][3] - r["band"][1]) / 2
        if adopted is not None and code == "JRN":
            for g in adopted["jrn"]:
                s = np.linspace(g["s0"], g["s1"], 40)
                hw = C.line_se(g["cov"], g["sm"], s)
                u = g["a"] + g["b"] * (s - g["sm"])
                lo1, la1 = sec.lonlat(s, u - hw)
                lo2, la2 = sec.lonlat(s[::-1], (u + hw)[::-1])
                out.append(np.c_[np.r_[lo1, lo2], np.r_[la1, la2]])
            continue
        for k, ln in enumerate(r["lines"]):
            s = np.linspace(ln["s0"], ln["s1"], 40)
            g = r["geo"][k]
            if ln["source"] in ("both", "bathymetry") and g.get("bathy") is not None:
                hw = C.line_se(g["bathy"]["cov"], g["bathy"]["sm"], s)
            elif "cov" in g:
                hw = C.line_se(g["cov"], g["sm"], s)
            else:
                hw = np.interp(s, r["sgrid"], hw_all)
            u = ln["a"] + ln["b"] * (s - ln["sm"])
            lo1, la1 = sec.lonlat(s, u - hw)
            lo2, la2 = sec.lonlat(s[::-1], (u + hw)[::-1])
            out.append(np.c_[np.r_[lo1, lo2], np.r_[la1, la2]])
    return out


def draw_map(fig, region, proj, gj, F, cat, pts, sub, detail=False, frame=None, title=None, adopted=None):
    import pygmt
    da = topo_da(region)
    pygmt.makecpt(cmap="gray", series=[-12000, 2500])
    fig.basemap(region=region, projection=proj, frame=frame or [("WSne+t" + title) if title else "WSne", "xaf", "yaf"])
    fig.grdimage(grid=da, cmap=True, shading="+a315+ne0.45", region=region, projection=proj)
    fig.coast(shorelines="0.25p,gray25", region=region, projection=proj)
    for z in [f for f in gj["features"] if f["properties"]["type"] == "deformation zone"]:
        xy = np.asarray(z["geometry"]["coordinates"][0])
        fig.plot(x=xy[:, 0], y=xy[:, 1], fill=PUR + "@65", pen="0.6p," + PUR, close=True)
    e = cat[cat.S0]
    a, b, c, d = region
    m = (e.lon > a) & (e.lon < b) & (e.lat > c) & (e.lat < d)
    if detail:
        e2 = cat[cat.S2 & ~cat.S0]
        m2 = (e2.lon > a) & (e2.lon < b) & (e2.lat > c) & (e2.lat < d)
        fig.plot(x=e2.lon[m2], y=e2.lat[m2], style="c0.025c", fill=SKY + "@40")
    fig.plot(x=e.lon[m], y=e.lat[m], style="c0.035c" if not detail else "c0.045c", fill="black")
    if detail:
        for poly in bands(F, adopted):
            fig.plot(x=poly[:, 0], y=poly[:, 1], fill=VER + "@55", close=True)
        for f in C.mapridges():
            fig.plot(x=f["xy"][:, 0], y=f["xy"][:, 1], pen="1.1p," + ORA + ",4_2:0p")
        for code, r in F["res"].items():
            sec = SECTIONS[code]
            dB = r["bathy_dec"]
            if len(dB):
                lo, la = sec.lonlat(dB[:, 0], dB[:, 1])
                fig.plot(x=lo, y=la, style="s0.14c", pen="0.7p," + GRN)
    for k, cl in (("OSR", "OSR"), ("OTF", "OTF")):
        for piece in pieces(pts[k]):
            fig.plot(x=piece[:, 0], y=piece[:, 1], pen="1.3p," + BLU + ",6_3:0p")
    fig.plot(x=sub[:, 0], y=sub[:, 1], pen="1.0p,gray15,2_2:0p")
    for f in gj["features"]:
        p, g = f["properties"], f["geometry"]
        if g["type"] != "LineString":
            continue
        xy = np.asarray(g["coordinates"])
        t = p["type"]
        if "strand" in t:
            if detail:
                fig.plot(x=xy[:, 0], y=xy[:, 1], pen="0.6p," + SKY)
        elif "alternative" in t:
            fig.plot(x=xy[:, 0], y=xy[:, 1], pen=("1.6p," if detail else "1.0p,") + GRN + ",2_2:0p")
        elif t == "step-over":
            fig.plot(x=xy[:, 0], y=xy[:, 1], pen="2.2p," + VER + ",3_2:0p")
        elif t == "bend":
            fig.plot(x=xy[:, 0], y=xy[:, 1], pen="2.6p," + VER)
        elif p.get("source") == "PB2002":
            pass                                        # north of 48.4 N the proposal keeps PB2002 (drawn above)
        else:
            fig.plot(x=xy[:, 0], y=xy[:, 1], pen="4.2p,white@20")
            fig.plot(x=xy[:, 0], y=xy[:, 1], pen="2.6p," + VER)
    for f in gj["features"]:
        p, g = f["properties"], f["geometry"]
        if g["type"] != "Point" or "not adopted" in p["type"]:
            continue
        lo, la = g["coordinates"]
        if p.get("uncertainty_km"):
            fig.plot(data=np.array([[lo, la, p.get("uncertainty_az_deg") or 0, 2 * p["uncertainty_km"],
                                     2 * (p.get("uncertainty_minor_km") or p["uncertainty_km"])]]),
                     style="E", pen="1p," + PUR)
        fig.plot(x=[lo], y=[la], style="a0.42c" if p["type"] == "triple junction" else "d0.3c", fill=PUR,
                 pen="0.6p,black")


def pieces(xy, gap=3.0):
    d = C.P.hav(xy[:-1, 1], xy[:-1, 0], xy[1:, 1], xy[1:, 0])
    cut = np.r_[0, np.flatnonzero(d > gap) + 1, len(xy)]
    return [xy[a:b] for a, b in zip(cut[:-1], cut[1:]) if b - a >= 2]


def legend(fig, pos="JBL+jBL+o0.2c", detail=False):
    import io
    items = [("-", f"2.6p,{VER}", "proposal v2 (straight segments)"),
             ("-", f"2.2p,{VER},3_2:0p", "step-over"),
             ("-", f"1.3p,{BLU},6_3:0p", "PB2002 (Bird 2003)"),
             ("-", f"1.6p,{GRN},2_2:0p", "alternative trace (flagged, not used)"),
             ("-", "1.0p,gray15,2_2:0p", "trench (PB2002)")]
    lines = ["N 1"]
    for sym, pen, lab in items:
        lines.append(f"S 0.35c - 0.6c - {pen} 0.8c {lab}")
    if detail:
        lines.append(f"S 0.35c - 0.6c - 1.1p,{ORA},4_2:0p 0.8c MAPRIDGES v2602")
        lines.append(f"S 0.35c s 0.14c - 0.7p,{GRN} 0.8c bathymetric picks (10 km)")
        lines.append(f"S 0.35c s 0.3c {VER}@55 - 0.8c 68 % band")
        lines.append(f"S 0.35c c 0.1c {SKY} - 0.8c tier C (S2 only)")
    lines.append("S 0.35c c 0.1c gray10 - 0.8c tier A-B earthquakes (S0)")
    lines.append(f"S 0.35c s 0.3c {PUR}@65 0.6p,{PUR} 0.8c deformation zone (80 % density)")
    lines.append(f"S 0.35c d 0.25c {PUR} 0.6p,black 0.8c corner (+68 % ellipse)")
    lines.append(f"S 0.35c a 0.35c {PUR} 0.6p,black 0.8c triple junction")
    spec = Path(FIG / "legend.txt")
    spec.write_text("\n".join(lines) + "\n")
    fig.legend(spec=str(spec), position=pos, box="+gwhite@15+p0.5p,gray40")


def main_map(gj, F, cat, pts, sub, fname="map_proposal.png", tag="D-B117 adopted v2"):
    import pygmt
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="8p", FONT_LABEL="9p", FONT_TITLE="11p", MAP_FRAME_TYPE="plain")
    region = (-131.8, -123.6, 39.9, 50.7)
    draw_map(fig, region, "M13c", gj, F, cat, pts, sub, frame=["WSne", "xa2f1", "ya2f1"])
    fig.text(x=-129.6, y=49.75, text="Explorer, Sovanco, Nootka: not resolved (PB2002 kept)", font="7p,Helvetica-Oblique,black",
             fill="white@30")
    legend(fig, pos="JBL+jBL+o0.15c")
    fig.text(position="TR", text=f"v6.1 S0, {tag}", font="8p,Helvetica-Bold,black", offset="-0.2c/-0.2c",
             fill="white@20")
    fig.savefig(FIG / fname, dpi=220)


def zooms(gj, F, cat, pts, sub, only=None, suffix="", adopted=None):
    import pygmt
    for nm, (title, region) in ZOOMS.items():
        if only and nm not in only:
            continue
        if suffix:
            title = title.split(":")[0] + " (D-B117 adopted)"
        fig = pygmt.Figure()
        pygmt.config(FONT_ANNOT_PRIMARY="8p", FONT_TITLE="10p", MAP_FRAME_TYPE="plain")
        a, b, c, d = region
        w = 15.0 if (b - a) * np.cos(np.radians((c + d) / 2)) >= (d - c) * 0.8 else 10.0
        draw_map(fig, region, f"M{w}c", gj, F, cat, pts, sub, detail=True, title=title, adopted=adopted)
        annotate(fig, gj, region)
        fig.basemap(map_scale=f"jBR+w50k+o0.3c/0.3c+f", region=region, projection=f"M{w}c")
        fig.savefig(FIG / f"zoom_{nm}{suffix}.png", dpi=200)
        log(f"zoom {nm}{suffix}")
    if suffix:
        return
    # legend alone for the other zooms
    fig = pygmt.Figure()
    fig.basemap(region=[0, 1, 0, 1], projection="X9c/6.5c", frame="+n")
    legend(fig, pos="jTL+jTL+o0.1c", detail=True)
    fig.savefig(FIG / "legend_zoom.png", dpi=200)


def annotate(fig, gj, region):
    a, b, c, d = region
    for f in gj["features"]:
        p, g = f["properties"], f["geometry"]
        if g["type"] == "LineString" and p["type"] == "step-over" and p.get("origin") != "A4 Axial node":
            xy = np.asarray(g["coordinates"])
            mx, my = xy.mean(0)
            if a < mx < b and c < my < d:
                o = p.get("offset_fit_km", p["offset_km"])
                txt = f"step {abs(o):.1f} km"
                if p.get("off_p16") is not None:
                    lo_, hi_ = sorted([abs(p["off_p16"]), abs(p["off_p84"])])
                    txt += f" (68 %: {lo_:.1f}-{hi_:.1f})"
                fig.text(x=mx, y=my, text=txt, font="7p,Helvetica-Bold," + VER, justify="LM", offset="0.15c/0c",
                         fill="white@30")
        if g["type"] == "Point" and "not adopted" not in p["type"]:
            lo, la = g["coordinates"]
            if a < lo < b and c < la < d:
                u = p.get("uncertainty_km")
                right = lo > (a + b) / 2
                fig.text(x=lo, y=la, text=p["name"].split(" (")[0] + (f" (+-{u:.1f} km)" if u else ""),
                         font="7p,Helvetica,black", justify="RB" if right else "LB",
                         offset="-0.25c/0.2c" if right else "0.25c/0.2c", fill="white@30")


# ------------------------------------------------------------------ other models
def model_lines():
    M = {"PB2002 (Bird 2003)": [], "NNR-MORVEL56 (Argus et al. 2011)": [], "Hasterok et al. 2022": [],
         "MAPRIDGES v2602 (2026)": []}
    pts, _, _ = C.pb2002_points()
    for k in ("OSR", "OTF"):
        M["PB2002 (Bird 2003)"] += pieces(pts[k])
    jf = C.morvel56_jf()
    M["NNR-MORVEL56 (Argus et al. 2011)"] += pieces(C.densify(jf, 1.0), gap=5.0)
    for f in C.hasterok():
        if f["type"] in ("spreading center", "dextral transform", "sinistral transform", "extension zone") and \
                np.all(f["xy"][:, 0] < -124.2):
            M["Hasterok et al. 2022"].append(C.densify(f["xy"], 1.0))
    for f in C.mapridges():
        M["MAPRIDGES v2602 (2026)"].append(C.densify(f["xy"], 1.0))
    return M


def model_comparison(F, M):
    rows = []
    for code, sec in SECTIONS.items():
        r = F["res"][code]
        L = r["lines"]
        for name, lines in list(M.items()):
            xy = np.vstack(lines)
            s, u = sec.su(xy[:, 0], xy[:, 1])
            lu = np.full(len(s), np.nan)
            for ln in L:
                m = (s >= ln["s0"]) & (s <= ln["s1"])
                lu[m] = ln["a"] + ln["b"] * (s[m] - ln["sm"])
            d = u - lu
            ok = np.isfinite(d) & (np.abs(d) <= 40)
            # one value per 5 km bin: the model point nearest the proposal
            vals = []
            for e0 in np.arange(0, sec.length, C.BIN):
                mm = ok & (s >= e0) & (s < e0 + C.BIN)
                if mm.any():
                    vals.append(d[mm][np.argmin(np.abs(d[mm]))])
            vals = np.array(vals)
            rows.append(dict(section=code, model=name, n_bins=len(vals),
                             median_offset_km=float(np.median(vals)) if len(vals) else np.nan,
                             median_abs_offset_km=float(np.median(np.abs(vals))) if len(vals) else np.nan,
                             p90_abs_offset_km=float(np.percentile(np.abs(vals), 90)) if len(vals) else np.nan,
                             sign="model minus proposal, + = " + C.SIGN[sec.kind]))
    t = pd.DataFrame(rows)
    t.to_csv(C.OUT / "model_comparison.csv", index=False, float_format="%.2f")
    return t


def models_fig(gj, F, cat, pts, sub, M):
    import pygmt
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="7p", FONT_TITLE="9p", MAP_FRAME_TYPE="plain")
    region = (-131.2, -124.0, 40.0, 48.8)
    e = cat[cat.S0 & (cat.lon < -124.0)]
    with fig.subplot(nrows=1, ncols=4, subsize=("6.2c", "9.6c"), margins=["0.15c", "0.1c"], frame=["WSne", "xa2f1", "ya2f1"]):
        for i, (name, lines) in enumerate(M.items()):
            with fig.set_panel(panel=i):
                fig.basemap(region=region, projection="M?", frame=[("WSne" if i == 0 else "wSne") + "+t" + name, "xa2f1", "ya2f1"])
                fig.coast(land="gray85", water="white", shorelines="0.2p,gray40")
                fig.plot(x=e.lon, y=e.lat, style="c0.02c", fill="gray55")
                for xy in lines:
                    for p in pieces(xy, gap=3.0):
                        fig.plot(x=p[:, 0], y=p[:, 1], pen="1.0p," + BLU + ",4_2:0p")
                for f in gj["features"]:
                    p, g = f["properties"], f["geometry"]
                    if g["type"] == "LineString" and p["type"] in ("transform", "spreading ridge", "step-over") \
                            and p.get("source") != "PB2002":
                        xy = np.asarray(g["coordinates"])
                        fig.plot(x=xy[:, 0], y=xy[:, 1], pen="1.4p," + VER)
    fig.savefig(FIG / "models.png", dpi=200)


# ------------------------------------------------------------------ profiles and bias charts (matplotlib)
def profiles(F, cat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mr = C.mapridges()
    fig, axs = plt.subplots(5, 1, figsize=(11, 15.5))
    for ax, (code, sec) in zip(axs, SECTIONS.items()):
        r = F["res"][code]
        ax.scatter(r["s"], r["u"], s=1.2, c="0.55", lw=0, rasterized=True, label="tier A-B (S0)")
        sc = sec.centres
        ax.plot(sc, r["pk2"], "x", color=SKY, ms=4, mew=1, label="tier A-C pick (S2)")
        ok = r["keep"]
        ax.errorbar(sc[ok], r["pk"][ok], yerr=r["sig_boot"][ok], fmt="o", color="k", ms=3, lw=0.8,
                    label="S0 pick (68 % bootstrap)")
        ax.plot(r["ps"], r["pu"], ".", color=BLU, ms=1.5, label="PB2002")
        for i, f in enumerate(mr):
            xy = C.densify(f["xy"], 1.0)
            s, u = sec.su(xy[:, 0], xy[:, 1])
            m = (s > -5) & (s < sec.length + 5) & (u > sec.uwin[0] - 5) & (u < sec.uwin[1] + 5)
            if m.sum() > 2:
                ax.plot(s[m], u[m], "--", color=ORA, lw=1, label="MAPRIDGES" if i == 0 else None)
        dB = r["bathy_dec"]
        if len(dB):
            ax.plot(dB[:, 0], dB[:, 1], "s", mfc="none", mec=GRN, ms=5, label="bathymetric pick (10 km)")
        hw = (r["band"][3] - r["band"][1]) / 2
        for k, ln in enumerate(r["lines"]):
            s = np.linspace(ln["s0"], ln["s1"], 50)
            u = ln["a"] + ln["b"] * (s - ln["sm"])
            g = r["geo"][k]
            if ln["source"] in ("both", "bathymetry") and g.get("bathy") is not None:
                h = C.line_se(g["bathy"]["cov"], g["bathy"]["sm"], s)
            elif "cov" in g:
                h = C.line_se(g["cov"], g["sm"], s)
            else:
                h = np.interp(s, r["sgrid"], hw)
            ax.fill_between(s, u - h, u + h, color=VER, alpha=0.3, lw=0)
            ax.plot(s, u, "-", color=VER, lw=2, label="proposal" if k == 0 else None)
            if g.get("bathy") is not None and ln["source"] == "seismicity":
                ub = g["bathy"]["a"] + g["bathy"]["b"] * (s - g["bathy"]["sm"])
                ax.plot(s, ub, ":", color=GRN, lw=1.8, label="bathymetric line (disagrees)" if k == 0 else None)
        if code == "JRN":
            for g in r["bathy_fit"]["geo"]:
                s = np.linspace(g["s0"], g["s1"], 30)
                ax.plot(s, C.line_u(g, s), ":", color=GRN, lw=1.8)
        ax.set_xlim(-3, sec.length + 3)
        ax.set_ylim(*sec.uwin)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        ax.set_ylabel(f"u (km, + {C.SIGN[sec.kind]})")
        tag = {"seismicity": "seismicity axis", "bathymetry": "bathymetry only (seismicity too sparse)"}.get(
            r["mode"], r["mode"])
        ax.set_title(f"{sec.name} ({code}): {tag}; drawn K = {r['K']} breaks, BIC K = {r['K_bic']}", fontsize=9,
                     loc="left")
        ax.grid(lw=0.3, color="0.9")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(fontsize=7, ncol=4, loc="upper left", frameon=False)
    axs[-1].set_xlabel("s, distance along the section frame (km)")
    fig.tight_layout()
    fig.savefig(FIG / "profiles.png", dpi=160)
    plt.close(fig)


def bias_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    off = pd.read_csv(C.OUT / "offsets.csv")
    t = pd.read_csv(C.OUT / "bias_tests.csv")
    v = pd.read_csv(C.OUT / "verdict.csv")
    secs = ["MFZ", "GR", "BL"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=False)
    for ax, code in zip(axs, secs):
        o = off[(off.section == code) & (off.span == "section")].iloc[0]
        tt = t[t.section == code]
        items = [("v6.1 seismicity axis - PB2002", o.O_pb2002, o.O_pb2002_lo, o.O_pb2002_hi, "k", "o"),
                 ("bathymetric axis - PB2002 (= O - O_bathy)", o.O_pb2002 - o.O_bathy, np.nan, np.nan, GRN, "s")]
        for r in tt[tt.test == "independent catalog"].itertuples():
            if np.isfinite(getattr(r, "own_O_pb2002", np.nan)) and r.own_n_bins >= 3:
                items.append((f"{r.label}: own axis - PB2002 ({int(r.own_n_bins)} bins)", r.own_O_pb2002, np.nan, np.nan,
                              BLU, "D"))
        for r in tt[tt.test == "version"].itertuples():
            items.append((f"{r.label} (same events) - PB2002", r.O_pb2002, np.nan, np.nan, "0.4", "^"))
        base = o.O_pb2002
        for r in tt[tt.test.isin(["geometry", "CI year", "selection"])].itertuples():
            if np.isfinite(r.D_km) and r.n_bins >= 3:
                items.append((f"{r.label} ({int(r.n_bins)} bins)", base + r.D_km, np.nan, np.nan, ORA, "v"))
        for r in tt[tt.test == "independent catalog"].itertuples():
            if np.isfinite(getattr(r, "du_median_km", np.nan)):
                items.append((f"shift vs {r.label} (n={int(r.n_pairs)})", base + r.du_median_km, base + r.du_lo,
                              base + r.du_hi, PUR, "P"))
        y = np.arange(len(items))[::-1]
        for yy, (lab, x, lo, hi, col, mk) in zip(y, items):
            if np.isfinite(lo):
                ax.plot([lo, hi], [yy, yy], color=col, lw=1.5)
            ax.plot([x], [yy], mk, color=col, ms=6)
        ax.set_yticks(y)
        ax.set_yticklabels([i[0] for i in items], fontsize=7)
        vv = v[v.section == code].iloc[0]
        if np.isfinite(vv.B):
            ax.axvspan(base - vv.B, base + vv.B, color="0.9", zorder=0)
        ax.axvline(0, color=BLU, lw=1, ls="--")
        ax.axvline(base, color="k", lw=0.6)
        ax.set_xlabel(f"km across strike (+ {C.SIGN[SECTIONS[code].kind]}); 0 = PB2002 (dashed)", fontsize=8)
        ax.set_title(f"{code}: O = {base:+.1f} km, B = {vv.B:.1f} km\n{vv.verdict}", fontsize=8.5, loc="center")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", lw=0.3, color="0.9")
    fig.suptitle("Offset of the seismicity axis from PB2002 and the location-bias terms (grey band: +-B around O; "
                 "shifts and subsets plotted as O + shift)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "bias.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    gj, F, cat = load()
    pts, sub, _ = C.pb2002_points()
    M = model_lines()
    log(model_comparison(F, M).to_string())
    profiles(F, cat)
    bias_fig()
    main_map(gj, F, cat, pts, sub, fname="map_proposal_D-B116.png", tag="D-B116 proposal")
    zooms(gj, F, cat, pts, sub)
    ad = C.BIG / "adopted_boundaries_v2.geojson"
    if ad.exists():                                   # D-B117: the adopted trace is the solid line of the main map
        gja = json.loads(ad.read_text())
        with open(C.BIG / "work" / "adopted.pkl", "rb") as fh:
            A = pickle.load(fh)
        main_map(gja, F, cat, pts, sub, fname="map_proposal.png", tag="D-B117 adopted v2")
        zooms(gja, F, cat, pts, sub, only=("jdf", "mendocino"), suffix="_adopted", adopted=A)
    models_fig(gj, F, cat, pts, sub, M)


if __name__ == "__main__":
    main()
