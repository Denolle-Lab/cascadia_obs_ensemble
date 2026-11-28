#!/usr/bin/env python3
"""One style for every figure of the Science Advances paper (figures v2, D-B114).

Everything a figure shares lives here, so a change of look is one edit:

  sizes        COL1 = 8.6 cm, COL2 = 17.8 cm (Science Advances column widths), MAX_H = 22 cm; fig_cm(w, h)
  type         one sans-serif family (Nimbus Sans = Helvetica metrics in matplotlib, Helvetica in GMT), 6-8 pt:
               FS["small"] 6 pt (ticks, notes), FS["base"] 7 pt (labels), FS["letter"] 8 pt bold (panel letters)
  lines        LW (pt): axes and ticks 0.5, grid 0.35, data 0.9, plate boundaries 0.8 (ridges 1.1), coast 0.4
  ink          INK (text and frames), INK2 (secondary text), GRID, UNRES (unresolved model cells), WATER
  categories   TYPE_COL / TYPE_MK / TYPE_ORDER: one colour and one marker per plate-boundary type (ridge, transform,
               intraplate, triple junction, subduction, crustal, arc, other); colour-blind checks in check_palette()
  colour maps  Crameri SCM (D-B104, scm_palettes.py): Vp batlow reversed, Vp/Vs roma reversed centred on 1.75,
               dV vik reversed; depth lajolla, time lipari, Mc batlow
  events       msize(M): marker area (pt^2) from magnitude, one law for every map and section; mag_legend()
  relief       Relief: GMRT mosaic (GridServer 'med' tiles, cached) as a light hillshade, land warm grey, sea grey-blue
  maps         MapAx: Mercator axes in matplotlib (x = lon, y = Mercator latitude in degrees), ticks in degrees,
               coast (GMT high-resolution, cached), PB2002 boundaries with trench teeth, scale bar, north arrow
  output       save(fig, out, name): vector PDF (dense layers rasterised at 300 dpi) + 300 dpi PNG; size check
  GMT          gmt_config(): the same fonts and pens for the PyGMT perspective view

The categorical palette was chosen with check_palette(): the dataviz checks (OKLCH lightness band and chroma floor,
OKLab Delta E x 100 under protanopia and deuteranopia simulated with Machado, Oliveira and Fernandes 2009 at severity 1,
the normal-vision floor, contrast against white) on adjacent legend pairs and on the boundary types that touch on the
map. All seven types cannot be pairwise distinct under dichromacy (no seven hues can), so every use of the type colour
also carries the type marker or the segment code (secondary encoding).
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import scm_palettes as scm  # noqa: E402

# ------------------------------------------------------------------ sizes, type, lines, ink
CM = 1 / 2.54
COL1, COL2, MAX_H = 8.6, 17.8, 22.0
FONT = ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"]
FS = {"small": 6.0, "base": 7.0, "title": 7.0, "letter": 8.0}
LW = {"axis": 0.5, "tick": 0.5, "grid": 0.35, "data": 0.9, "thin": 0.5, "coast": 0.4, "border": 0.35,
      "plate": 0.8, "ridge": 1.1, "contour": 0.75, "slab": 0.9, "casing": 1.3}
INK, INK2, GRID, AXIS = "#1b1b1b", "#5c5c5c", "#e4e4e4", "#3a3a3a"
UNRES = "#d9d9d9"          # rock cells of the model that fail the resolution mask
WATER = "#eef3f6"          # water column in sections
DPI = 300

# ------------------------------------------------------------------ plate-boundary types (D-B107)
TYPE_ORDER = ["ridge", "transform", "intraplate", "triple junction", "subduction", "crustal", "arc", "other"]
TYPE_COL = {"ridge": "#2B6CB0", "transform": "#E06A2C", "intraplate": "#16A4AE", "triple junction": "#6E4BA3",
            "subduction": "#557A1F", "crustal": "#D9A21B", "arc": "#D35D8E", "other": "#9B9B9B"}
TYPE_MK = {"ridge": "o", "transform": "s", "intraplate": "D", "triple junction": "h", "subduction": "v",
           "crustal": "^", "arc": "P", "other": "."}
TYPE_MS = {"o": 1.0, "s": 0.9, "D": 0.8, "h": 1.05, "v": 1.0, "^": 1.0, "P": 1.05, ".": 1.0}   # equal visual weight
# boundary types that touch on the map: these pairs must stay apart under dichromacy (checked in check_palette)
TYPE_NEIGHBOURS = [("ridge", "transform"), ("transform", "intraplate"), ("transform", "triple junction"),
                   ("triple junction", "subduction"), ("intraplate", "triple junction"), ("intraplate", "subduction"),
                   ("crustal", "arc"), ("crustal", "subduction"), ("arc", "subduction"), ("ridge", "intraplate")]
STATION_COL = {"sea": "#2B6CB0", "land": "#ffffff"}
ACCENT = "#C0392B"         # one accent for marked events (eruption line, M >= 5 stars), never a series colour

# ------------------------------------------------------------------ colour maps (D-B104)
CM_VP = scm.scm_cmap("batlow", reverse=True)
CM_VPVS = scm.scm_cmap("roma", reverse=True)
CM_DV = scm.scm_cmap("vik", reverse=True)
CM_DEPTH = scm.scm_cmap("lajolla", reverse=False, lo=0.06, hi=0.94)
CM_TIME = scm.scm_cmap("lipari", reverse=False, lo=0.05, hi=0.92)
CM_MC = scm.scm_cmap("batlow")
VP_RANGE = (5.6, 8.2)               # km/s, shared by every Vp panel
VPVS_RANGE = (1.60, 1.75, 1.90)     # low, centre, high: shared by every Vp/Vs panel


def norm_vpvs():
    return TwoSlopeNorm(VPVS_RANGE[1], VPVS_RANGE[0], VPVS_RANGE[2])


# ------------------------------------------------------------------ matplotlib style
def use():
    """Apply the paper rcParams (call once per process, before any figure)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": FS["base"],
        "axes.titlesize": FS["title"], "axes.titleweight": "normal", "axes.labelsize": FS["base"],
        "xtick.labelsize": FS["small"], "ytick.labelsize": FS["small"], "legend.fontsize": FS["small"],
        "legend.title_fontsize": FS["small"], "figure.titlesize": FS["base"],
        "axes.linewidth": LW["axis"], "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": LW["tick"], "ytick.major.width": LW["tick"], "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4, "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5, "axes.labelpad": 2.0, "axes.titlepad": 3.0, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": LW["grid"],
        "lines.linewidth": LW["data"], "lines.solid_capstyle": "round", "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 1.0, "legend.labelspacing": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white", "image.interpolation": "nearest",
        "mathtext.default": "regular", "axes.unicode_minus": True, "hatch.linewidth": 0.4,
        "hatch.color": "#8a8a8a"})


def fig_cm(w, h):
    """A figure of w x h cm (w one of COL1, COL2; h <= MAX_H)."""
    assert h <= MAX_H + 1e-6, f"height {h} cm > {MAX_H} cm"
    return plt.figure(figsize=(w * CM, h * CM))


def axes_cm(fig, x, y, w, h, **kw):
    """Axes at (x, y) cm from the figure's bottom-left corner, w x h cm."""
    W, H = fig.get_size_inches() / CM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def letter(ax, s, dx=-0.55, dy=0.12, fig=None):
    """Bold panel letter, its baseline dy cm above the axes top and its left edge dx cm from the axes left edge."""
    fig = fig or ax.figure
    W, H = fig.get_size_inches() / CM
    b = ax.get_position()
    fig.text(b.x0 + dx / W, b.y1 + dy / H, s, fontsize=FS["letter"], fontweight="bold", ha="left", va="bottom",
             color=INK)


def halo(lw=1.6, fg="white"):
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def text_halo(ax, x, y, s, fs=None, color=INK, lw=1.6, **kw):
    return ax.text(x, y, s, fontsize=fs or FS["small"], color=color, path_effects=halo(lw), **kw)


def cased(ax, x, y, color=INK, lw=0.8, ls="-", casing=None, zorder=6.0, alpha=1.0, **kw):
    """A line over a thin white casing, so it reads over any colour."""
    c = LW["casing"] if casing is None else casing
    ax.plot(x, y, color="white", lw=lw + c, solid_capstyle="round", alpha=0.9 * alpha, zorder=zorder, **kw)
    return ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=zorder + 0.01, alpha=alpha, **kw)


def clean(ax, grid_axis=None, left=True, bottom=True):
    """Recessive frame: left and bottom spines only, optional hairline grid along one axis."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=LW["grid"], zorder=0)
        ax.set_axisbelow(True)


def boxed(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(LW["axis"])
        s.set_color(AXIS)


# ------------------------------------------------------------------ events
M_SIZE_REF = 0.8                     # marker diameter (pt) at M 0


def msize(M, scale=1.0, mmin=-0.5, mmax=7.0):
    """Marker area (pt^2) for matplotlib scatter: diameter = 0.8 pt x 10^(0.2 M) x scale (M 2: 2 pt, M 4: 5 pt,
    M 6: 12.7 pt). Missing magnitudes draw as M 0."""
    M = np.clip(np.nan_to_num(np.asarray(M, float), nan=0.0), mmin, mmax)
    d = M_SIZE_REF * 10 ** (0.2 * M) * scale
    return d ** 2


def mag_legend(ax, x, y, mags=(2, 3, 4, 5, 6), scale=1.0, color="#9a9a9a", ec="white", dx=None, transform=None,
               title="M", fs=None, horizontal=True):
    """Size legend for msize(): circles with their magnitude below (axes fraction coordinates by default)."""
    tr = transform or ax.transAxes
    fs = fs or FS["small"]
    dmax = np.sqrt(msize(max(mags), scale)) / 72 * 2.54          # cm
    W = (ax.get_position().width * ax.figure.get_size_inches()[0]) * 2.54
    step = dx if dx is not None else max(dmax * 1.05, 0.38) / W
    if title:
        ax.text(x - 0.6 * step, y, title, transform=tr, fontsize=fs, ha="right", va="center", color=INK,
                path_effects=halo(1.4))
    for i, m in enumerate(mags):
        xx = x + i * step if horizontal else x
        yy = y if horizontal else y - i * step
        ax.scatter([xx], [yy], s=msize(m, scale), c=color, edgecolors=ec, linewidths=0.3, transform=tr,
                   zorder=30, clip_on=False)
        if horizontal:
            ax.text(xx, y - 0.5 * dmax / (ax.get_position().height * ax.figure.get_size_inches()[1] * 2.54) - 0.012,
                    f"{m:g}", transform=tr, fontsize=fs, ha="center", va="top", color=INK, path_effects=halo(1.4),
                    zorder=31)
        else:
            ax.text(xx + 0.5 * step, yy, f"{m:g}", transform=tr, fontsize=fs, ha="left", va="center", color=INK)


def type_handles(types=None, ms=4.5, with_other=False):
    from matplotlib.lines import Line2D
    types = types or [t for t in TYPE_ORDER if with_other or t != "other"]
    return [Line2D([], [], ls="", marker=TYPE_MK[t], ms=ms * TYPE_MS[TYPE_MK[t]], mfc=TYPE_COL[t], mec="white",
                   mew=0.4, label=t) for t in types]


# ------------------------------------------------------------------ colour checks (dataviz method, Machado 2009)
_MACH = scm.MACHADO


def _lin(h):
    h = h.lstrip("#")
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(l):
    r, g, b = l
    L_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * L_ + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * L_ - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * L_ + 0.7827717662 * m - 0.8086757660 * s])


def delta_e(a, b, view=None):
    """OKLab Delta E x 100 between two hex colours, optionally both seen with a dichromacy (protan/deutan/tritan)."""
    la, lb = _lin(a), _lin(b)
    if view:
        la, lb = np.clip(_MACH[view] @ la, 0, 1), np.clip(_MACH[view] @ lb, 0, 1)
    return float(100 * np.linalg.norm(_oklab(la) - _oklab(lb)))


def oklch(h):
    L, a, b = _oklab(_lin(h))
    return float(L), float(np.hypot(a, b))


def contrast(a, b="#ffffff"):
    y = lambda h: float(_lin(h) @ np.array([0.2126, 0.7152, 0.0722]))  # noqa: E731
    hi, lo = sorted([y(a), y(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def check_palette(types=None):
    """The dataviz checks on the type palette: per colour OKLCH L (band 0.43-0.78) and C (floor 0.10) and contrast on
    white; worst CVD (min of protan, deutan) and normal-vision Delta E over the adjacent legend pairs, over all pairs
    and over the map neighbours. Returns a dict (written to the review page)."""
    types = types or [t for t in TYPE_ORDER if t != "other"]
    hexes = [TYPE_COL[t] for t in types]
    out = {"colours": [], "pairs": {}}
    for t, h in zip(types, hexes):
        L, C = oklch(h)
        out["colours"].append(dict(type=t, hex=h, L=round(L, 3), C=round(C, 3), contrast=round(contrast(h), 2),
                                   band=bool(0.43 <= L <= 0.78), chroma=bool(C >= 0.10)))
    n = len(hexes)
    lists = {"adjacent (legend order)": [(i, i + 1) for i in range(n - 1)],
             "all pairs": list(itertools.combinations(range(n), 2)),
             "map neighbours": [(types.index(a), types.index(b)) for a, b in TYPE_NEIGHBOURS]}
    for k, pl in lists.items():
        cvd = min((min(delta_e(hexes[i], hexes[j], v) for v in ("protan", "deutan")), types[i], types[j])
                  for i, j in pl)
        nv = min((delta_e(hexes[i], hexes[j]), types[i], types[j]) for i, j in pl)
        tri = min((delta_e(hexes[i], hexes[j], "tritan"), types[i], types[j]) for i, j in pl)
        out["pairs"][k] = dict(cvd=round(cvd[0], 1), cvd_pair=f"{cvd[1]} / {cvd[2]}", normal=round(nv[0], 1),
                               normal_pair=f"{nv[1]} / {nv[2]}", tritan=round(tri[0], 1),
                               tritan_pair=f"{tri[1]} / {tri[2]}",
                               verdict=("pass" if cvd[0] >= 8 and nv[0] >= 15 else
                                        "floor (secondary encoding)" if cvd[0] >= 6 and nv[0] >= 15 else
                                        "below floor: marker and label carry identity"))
    return out


def palette_swatch_png(path, types=None):
    """Swatches of the type palette under normal vision, deuteranopia, protanopia, tritanopia and greyscale."""
    types = types or [t for t in TYPE_ORDER if t != "other"]
    rgb = np.array([[int(TYPE_COL[t].lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)] for t in types])
    use()
    fig = fig_cm(COL2, 4.2)
    views = scm.VIEWS
    for r, v in enumerate(views):
        ax = axes_cm(fig, 2.3, 3.4 - r * 0.72, 15.0, 0.6)
        sim = scm.simulate(rgb, v)
        for i, (t, c) in enumerate(zip(types, sim)):
            ax.add_patch(plt.Rectangle((i + 0.04, 0.05), 0.92, 0.9, color=c))
            ax.plot(i + 0.5, 0.5, TYPE_MK[t], ms=4.5 * TYPE_MS[TYPE_MK[t]], mfc="white", mec="none")
            if r == 0:
                ax.text(i + 0.5, 1.12, t, ha="center", va="bottom", fontsize=FS["small"])
        ax.set_xlim(0, len(types)); ax.set_ylim(0, 1); ax.axis("off")
        fig.text(2.2 / COL2, (3.4 - r * 0.72 + 0.3) / 4.2, scm.VIEW_LABEL[v], ha="right", va="center",
                 fontsize=FS["small"])
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ output
def save(fig, out, name, png=True, pdf=True):
    """name.pdf (vector; rasterised layers at 300 dpi) and name.png (300 dpi). Returns the size in cm."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches() / CM
    assert h <= MAX_H + 0.05, f"{name}: {h:.1f} cm tall"
    if pdf:
        fig.savefig(out / f"{name}.pdf", dpi=DPI, metadata={"Creator": "paper_figures_v2.py"})
    if png:
        fig.savefig(out / f"{name}.png", dpi=DPI)
    plt.close(fig)
    return round(float(w), 2), round(float(h), 2)


# ------------------------------------------------------------------ maps
def merc(lat):
    """Mercator y (degrees) of latitude (degrees)."""
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, float)) / 2)))


def imerc(y):
    return np.degrees(2 * np.arctan(np.exp(np.radians(np.asarray(y, float)))) - np.pi / 2)


def map_height(region, width):
    """Height (cm) of a Mercator map of `region` (W, E, S, N) drawn `width` cm wide."""
    W, E, S, N = region
    return width * (merc(N) - merc(S)) / (E - W)


class MapAx:
    """Mercator map in plain matplotlib axes: x = longitude, y = merc(latitude) (aspect 1)."""

    def __init__(self, ax, region, ticks=(2.0, 2.0), labels="WS", minor=None, fmt="{:g}°"):
        self.ax, self.region = ax, list(region)
        W, E, S, N = region
        ax.set_xlim(W, E)
        ax.set_ylim(merc(S), merc(N))
        ax.set_aspect(1.0)
        boxed(ax)
        dx, dy = ticks
        xt = np.arange(np.ceil(W / dx - 1e-9) * dx, E + 1e-9, dx)
        yt = np.arange(np.ceil(S / dy - 1e-9) * dy, N + 1e-9, dy)
        ax.set_xticks(xt)
        ax.set_yticks(merc(yt))
        lon_lab = [(fmt.format(abs(x)) + ("W" if x < 0 else "E")) for x in xt]
        lat_lab = [(fmt.format(abs(y)) + ("N" if y >= 0 else "S")) for y in yt]
        ax.set_xticklabels(lon_lab if ("S" in labels or "N" in labels) else [])
        ax.set_yticklabels(lat_lab if ("W" in labels or "E" in labels) else [])
        ax.tick_params(top="N" in labels, labeltop="N" in labels, bottom=True, labelbottom="S" in labels,
                       right="E" in labels, labelright="E" in labels, left=True, labelleft="W" in labels,
                       length=2.0, pad=1.2)
        if "N" in labels and "S" not in labels:
            ax.tick_params(bottom=True, labelbottom=False)

    def xy(self, lon, lat):
        return np.asarray(lon, float), merc(lat)

    def plot(self, lon, lat, *a, **kw):
        return self.ax.plot(np.asarray(lon, float), merc(lat), *a, **kw)

    def scatter(self, lon, lat, **kw):
        return self.ax.scatter(np.asarray(lon, float), merc(lat), **kw)

    def text(self, lon, lat, s, **kw):
        return self.ax.text(lon, merc(lat), s, **kw)

    def lines(self, segs, **kw):
        segs = [np.c_[s[:, 0], merc(s[:, 1])] for s in segs if len(s) > 1]
        lc = LineCollection(segs, **kw)
        self.ax.add_collection(lc)
        return lc

    def scalebar(self, lon, lat, km, label=None, h_cm=0.07, fs=None):
        """Black-and-white scale bar of `km` starting at (lon, lat), label above."""
        dlon = km / (111.195 * np.cos(np.radians(lat)))
        y = merc(lat)
        hh = h_cm / (self.ax.get_position().height * self.ax.figure.get_size_inches()[1] * 2.54) * \
            (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
        for i in range(2):
            self.ax.add_patch(plt.Rectangle((lon + i * dlon / 2, y), dlon / 2, hh, facecolor=INK if i == 0 else "white",
                                            edgecolor=INK, lw=0.4, zorder=25))
        for xx, t in ((lon, "0"), (lon + dlon, f"{km:g} km" if label is None else label)):
            self.ax.text(xx, y + hh * 1.6, t, ha="center", va="bottom", fontsize=fs or FS["small"], zorder=25,
                         path_effects=halo(1.4))

    def north(self, lon, lat, size_cm=0.45, fs=None):
        """North arrow: a slim filled arrow with N above, its foot at (lon, lat)."""
        ax = self.ax
        H = ax.get_position().height * ax.figure.get_size_inches()[1] * 2.54
        dy = size_cm / H * (ax.get_ylim()[1] - ax.get_ylim()[0])
        y = merc(lat)
        ax.annotate("", xy=(lon, y + dy), xytext=(lon, y), zorder=25,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", lw=0.7, color=INK,
                                    shrinkA=0, shrinkB=0))
        ax.text(lon, y + dy * 1.08, "N", ha="center", va="bottom", fontsize=fs or FS["small"], fontweight="bold",
                zorder=25, path_effects=halo(1.4))


# ------------------------------------------------------------------ relief, coast, plate boundaries
class Relief:
    """GMRT relief for every map: a mosaic of GridServer 'med' tiles (cache/tiles, fetched once), resampled to a
    regular grid and stored as cache/gmrt_mosaic.npz. Missing tiles fall back to GMT earth_relief 30s."""

    def __init__(self, cache, region=(-132.5, -119.0, 38.5, 51.5), step=0.004):
        self.cache = Path(cache)
        f = self.cache / "gmrt_mosaic.npz"
        if f.exists():
            d = np.load(f)
            self.lon, self.lat, self.z = d["lon"], d["lat"], d["z"]
            return
        import xarray as xr
        W, E, S, N = region
        lon = np.arange(W, E + 1e-9, step)
        lat = np.arange(S, N + 1e-9, step)
        Z = np.full((len(lat), len(lon)), np.nan, "float32")
        tiles = sorted((self.cache / "tiles").glob("gmrt_*.nc"))
        from scipy.interpolate import RegularGridInterpolator
        for t in tiles:
            ds = xr.open_dataset(t)
            v = ds[list(ds.data_vars)[0]]
            tl, ta = v.lon.values, v.lat.values
            zz = v.values.astype("float32")
            if ta[0] > ta[-1]:
                ta, zz = ta[::-1], zz[::-1]
            i = (lat >= ta[0]) & (lat <= ta[-1])
            j = (lon >= tl[0]) & (lon <= tl[-1])
            if not i.any() or not j.any():
                continue
            fi = RegularGridInterpolator((ta, tl), zz, bounds_error=False, fill_value=np.nan)
            LA, LO = np.meshgrid(lat[i], lon[j], indexing="ij")
            sub = fi(np.c_[LA.ravel(), LO.ravel()]).reshape(LA.shape).astype("float32")
            blk = Z[np.ix_(i, j)]
            Z[np.ix_(i, j)] = np.where(np.isfinite(blk), blk, sub)
        if np.isnan(Z).any():
            import pygmt
            r = pygmt.datasets.load_earth_relief(resolution="30s", region=[W, E, S, N], registration="gridline")
            fi = RegularGridInterpolator((r.lat.values, r.lon.values), r.values.astype("float32"), bounds_error=False,
                                         fill_value=np.nan)
            m = np.isnan(Z)
            LA, LO = np.meshgrid(lat, lon, indexing="ij")
            Z[m] = fi(np.c_[LA[m], LO[m]])
        self.lon, self.lat, self.z = lon, lat, Z
        np.savez_compressed(f, lon=lon, lat=lat, z=Z)

    def rgb(self, region, px=1600, az=315.0, alt=42.0, ve=8.0, strength=0.42):
        """Shaded relief image for `region` on a Mercator grid: (rgb array, extent in MapAx coordinates)."""
        from matplotlib.colors import LightSource
        from scipy.interpolate import RegularGridInterpolator
        W, E, S, N = region
        nx = int(px)
        ny = int(px * (merc(N) - merc(S)) / (E - W))
        x = np.linspace(W, E, nx)
        y = np.linspace(merc(S), merc(N), ny)
        lat = imerc(y)
        f = RegularGridInterpolator((self.lat, self.lon), self.z, bounds_error=False, fill_value=np.nan)
        LA, LO = np.meshgrid(lat, x, indexing="ij")
        z = f(np.c_[LA.ravel(), LO.ravel()]).reshape(LA.shape)
        z = np.where(np.isfinite(z), z, 0.0)
        dx = (x[1] - x[0]) * 111195 * np.cos(np.radians(0.5 * (S + N)))
        dy = dx                                             # Mercator: square pixels on the map
        ls = LightSource(azdeg=az, altdeg=alt)
        hs = ls.hillshade(z, vert_exag=ve, dx=dx, dy=dy, fraction=1.0)
        base = elevation_colors(z)
        shade = 1.0 - strength * (1.0 - hs)
        shade = shade / np.percentile(shade, 65)
        img = np.clip(base * shade[..., None], 0, 1)
        return img, [W, E, merc(S), merc(N)]

    def draw(self, m: MapAx, px=1600, zorder=0.2, **kw):
        img, ext = self.rgb(m.region, px=px, **kw)
        m.ax.imshow(img, extent=ext, origin="lower", interpolation="bilinear", zorder=zorder, aspect="auto",
                    rasterized=True)
        m.ax.set_xlim(ext[0], ext[1])
        m.ax.set_ylim(ext[2], ext[3])


# elevation colour stops (m, sRGB 0-1): pale grey-blue sea (deeper = a little darker), warm light grey land
_SEA = [(-6000, (0.690, 0.749, 0.788)), (-3500, (0.737, 0.788, 0.824)), (-2000, (0.800, 0.839, 0.867)),
        (-200, (0.871, 0.898, 0.914)), (0, (0.910, 0.929, 0.937))]
_LAND = [(0, (0.925, 0.918, 0.898)), (600, (0.890, 0.878, 0.851)), (1800, (0.840, 0.824, 0.792)),
         (4500, (0.780, 0.761, 0.725))]


def elevation_colors(z):
    z = np.asarray(z, float)
    out = np.zeros(z.shape + (3,))
    for stops, sel in ((_SEA, z < 0), (_LAND, z >= 0)):
        zs = np.array([s[0] for s in stops], float)
        cs = np.array([s[1] for s in stops], float)
        for k in range(3):
            out[..., k] = np.where(sel, np.interp(z, zs, cs[:, k]), out[..., k])
    return out


def gmt_lines(cache, name, args, region=(-134, -118, 37.5, 52.5)):
    """GMT coast/border lines (high resolution) as a list of (N, 2) lon/lat arrays, cached as text."""
    f = Path(cache) / f"{name}.txt"
    if not f.exists():
        import pygmt
        W, E, S, N = region
        f.parent.mkdir(parents=True, exist_ok=True)
        with pygmt.clib.Session() as lib:
            lib.call_module("coast", f"-R{W}/{E}/{S}/{N} {args} -M ->{f}")
    segs, cur = [], []
    for line in f.read_text().splitlines():
        if line.startswith(">"):
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = []
        elif line.strip():
            cur.append(list(map(float, line.split()[:2])))
    if len(cur) > 1:
        segs.append(np.array(cur))
    return segs


def pb2002_lines(path, region=(-134, -118, 37.5, 52.5), axes=None, oceanic_only=False):
    """PB2002 steps joined into polylines: {'OSR': [...], 'TF': [...], 'SUB': [...]} (lon, lat arrays).
    axes: the plate-axes table of the segment rule (pb_common.AXES_FILES; the paper uses plate boundaries v2 adopted,
    plate_axes_v2_adopted.csv, D-B117): its spreading (OSR) and oceanic transform (OTF) sections, step-overs and
    connectors replace the PB2002 ones, so the maps draw the traces the segments were made with. The subduction trace
    and the continental boundaries stay PB2002. oceanic_only: PB2002 ridges and oceanic transforms only (reference)."""
    W, E, S, N = region
    groups = {"SUB": [], "OSR": [], "TF": []}
    ax_tab = None
    if axes is not None and Path(axes).exists():
        import pandas as pd
        ax_tab = pd.read_csv(axes)
    for ft in json.loads(Path(path).read_text())["features"]:
        c = np.asarray(ft["geometry"]["coordinates"], float)
        if not ((c[:, 0] > W).all() and (c[:, 0] < E).all() and (c[:, 1] > S).all() and (c[:, 1] < N).all()):
            continue
        cls = ft["properties"]["STEPCLASS"]
        if ax_tab is not None and cls in ("OSR", "OTF"):
            continue
        if oceanic_only and cls not in ("OSR", "OTF"):
            continue
        groups["SUB" if cls == "SUB" else "OSR" if cls == "OSR" else "TF"].append(c)
    if ax_tab is not None:
        extra = {"OSR": [], "TF": []}
        for (kind, _), g in ax_tab.groupby(["kind", "section"], sort=False):
            extra["OSR" if kind == "OSR" else "TF"].append(g[["lon", "lat"]].to_numpy(float))
    out = {}
    for k, steps in groups.items():
        lines = []
        for s in steps:
            for L in lines:
                if np.allclose(L[-1], s[0], atol=1e-3):
                    L.extend(s[1:].tolist())
                    break
                if np.allclose(L[0], s[-1], atol=1e-3):
                    L[:0] = s[:-1].tolist()
                    break
            else:
                lines.append(s.tolist())
        out[k] = [np.asarray(L) for L in lines]
    if ax_tab is not None:
        for k, v in extra.items():
            out[k] = out[k] + v
    return out


def draw_plates(m: MapAx, pb, scale=1.0, color=INK, teeth_km=38.0, zorder=7):
    """Ridges thicker than transforms; the trench with filled teeth on the upper-plate (east) side."""
    ax = m.ax
    for key, lw in (("OSR", LW["ridge"]), ("TF", LW["plate"]), ("SUB", LW["plate"])):
        for L in pb[key]:
            ax.plot(L[:, 0], merc(L[:, 1]), color=color, lw=lw * scale, zorder=zorder, solid_capstyle="butt")
    for L in pb["SUB"]:
        L = L[np.argsort(-L[:, 1])]                                   # north to south
        x, y = L[:, 0], merc(L[:, 1])
        kx = 111.195 * np.cos(np.radians(L[:, 1]))
        s = np.r_[0, np.cumsum(np.hypot(np.diff(x) * kx[1:], np.diff(L[:, 1]) * 111.195))]
        H = ax.get_position().height * ax.figure.get_size_inches()[1] * 2.54
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        size = 0.085 * scale / H * yr                                 # tooth height in y units (0.085 cm)
        for d in np.arange(teeth_km / 2, s[-1], teeth_km):
            j = min(np.searchsorted(s, d), len(s) - 1)
            j0 = max(j - 1, 0)
            tx, ty = x[j] - x[j0], y[j] - y[j0]
            nrm = np.hypot(tx, ty) or 1.0
            tx, ty = tx / nrm, ty / nrm
            nx, ny = -ty, tx                                          # left of southward travel = east
            if nx < 0:
                nx, ny = -nx, -ny
            px, py = x[j], y[j]
            tri = np.array([[px - tx * size * 0.75, py - ty * size * 0.75], [px + tx * size * 0.75, py + ty * size * 0.75],
                            [px + nx * size * 1.25, py + ny * size * 1.25]])
            ax.add_patch(plt.Polygon(tri, closed=True, facecolor=color, edgecolor="none", zorder=zorder))


def draw_plates_reference(m: MapAx, pb_ref, color="#4d4d4d", lw=0.5, zorder=5.5):
    """PB2002 (Bird 2003) ridges and oceanic transforms as a faint dashed reference under the adopted traces."""
    for key in ("OSR", "TF"):
        for L in pb_ref.get(key, []):
            m.ax.plot(L[:, 0], merc(L[:, 1]), color=color, lw=lw, ls=(0, (2.2, 1.8)), zorder=zorder,
                      solid_capstyle="butt", dash_capstyle="butt")


def draw_coast(m: MapAx, coast, borders=None, zorder=3):
    m.lines(coast, colors="#4a4a4a", linewidths=LW["coast"], zorder=zorder)
    if borders:
        m.lines(borders, colors="#8a8a8a", linewidths=LW["border"], linestyles=(0, (3, 1.5)), zorder=zorder)


# ------------------------------------------------------------------ plate motions (NNR-MORVEL56, Argus et al. 2011)
MORVEL = {"PA": (-63.58, 114.70, 0.651), "NA": (-4.85, -80.64, 0.209), "JF": (-38.31, 60.04, 0.951)}


def plate_velocity(moving, fixed, lat, lon):
    """Velocity of `moving` relative to `fixed` at (lat, lon): (mm/yr, azimuth in degrees)."""
    def om(p):
        la, lo, r = np.radians(p[0]), np.radians(p[1]), np.radians(p[2]) / 1e6
        return r * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    w = om(MORVEL[moving]) - om(MORVEL[fixed])
    la, lo = np.radians(lat), np.radians(lon)
    r = 6371e3 * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    v = np.cross(w, r) * 1e3
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    return float(np.hypot(v @ e, v @ n)), float(np.degrees(np.arctan2(v @ e, v @ n)) % 360)


def draw_plate_motion(m: MapAx, moving, fixed, lat, lon, km_per_mm=3.0, label=True, fs=None):
    rate, az = plate_velocity(moving, fixed, lat, lon)
    L = rate * km_per_mm
    a = np.radians(az)
    lat2 = lat + L * np.cos(a) / 111.195
    lon2 = lon + L * np.sin(a) / (111.195 * np.cos(np.radians(lat)))
    m.ax.annotate("", xy=(lon2, merc(lat2)), xytext=(lon, merc(lat)), zorder=24,
                  arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", lw=1.1, color=INK, shrinkA=0,
                                  shrinkB=0, path_effects=[pe.withStroke(linewidth=2.4, foreground="white")]))
    if label:
        m.text(0.5 * (lon + lon2), 0.5 * (lat + lat2), f"{rate:.0f}", fontsize=fs or FS["small"], ha="left",
               va="top" if az < 180 else "bottom", color=INK, path_effects=halo(1.5), zorder=26)
    return rate, az


# ------------------------------------------------------------------ GMT (perspective view)
def gmt_config():
    import pygmt
    pygmt.config(FONT_ANNOT_PRIMARY=f"{FS['small']:g}p,Helvetica,{INK}", FONT_ANNOT_SECONDARY=f"{FS['small']:g}p,Helvetica,{INK}",
                 FONT_LABEL=f"{FS['base']:g}p,Helvetica,{INK}", FONT_TAG=f"{FS['letter']:g}p,Helvetica-Bold,{INK}",
                 FONT_TITLE=f"{FS['base']:g}p,Helvetica,{INK}", MAP_FRAME_TYPE="plain", MAP_FRAME_PEN=f"{LW['axis']}p,{AXIS}",
                 MAP_TICK_LENGTH_PRIMARY="2.5p", MAP_TICK_PEN_PRIMARY=f"{LW['tick']}p,{AXIS}", MAP_ANNOT_OFFSET_PRIMARY="1.5p",
                 MAP_LABEL_OFFSET="3p", FORMAT_GEO_MAP="dddF", PS_CHAR_ENCODING="ISOLatin1+", COLOR_NAN="white")


def git_head(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    print(json.dumps(check_palette(), indent=1))
