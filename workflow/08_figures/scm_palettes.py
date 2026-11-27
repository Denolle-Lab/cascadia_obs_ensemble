#!/usr/bin/env python3
"""Crameri Scientific Colour Maps (GMT's SCM CPTs) for matplotlib, colour-vision-deficiency simulation and
perceptual metrics: the tools behind the velocity-model palette choice (model_figures.py palette / ramps).

    read_scm(name)           256 x 3 sRGB (0-1) from <gmt share>/cpt/SCM/<name>.cpt (no cmcrameri needed)
    scm_cmap(name, reverse)  matplotlib ListedColormap
    simulate(rgb, view)      view: normal | deutan | protan | tritan | grey; any (..., 3) sRGB array (0-1)
    rgb_to_lab(rgb), de2000(lab1, lab2)
    palette_metrics(colors, diverging, dot)   numbers for the palette sheet
    js_ramp(colors, n)       [[t, r, g, b], ...], t in [-1, 1], for the 3-D atlas (rampCol in atlas_template.html)

CVD: Machado, Oliveira & Fernandes (2009, IEEE TVCG 15(6), 1291-1298) matrices at severity 1.0 (dichromacy),
applied to linear RGB (IEC 61966-2-1 sRGB transfer function), clipped to [0, 1]. Greyscale: the CIE luminance Y of
linear sRGB (Rec. 709 weights) shown as the neutral grey of the same Y. CIELAB with the D65 white; colour
differences as CIEDE2000 (Sharma, Wu & Dalal 2005).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

GMT_SHARE = Path(sys.executable).resolve().parents[1] / "share" / "gmt"
SCM_DIRS = [GMT_SHARE / "cpt" / "SCM", Path("/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/share/gmt/cpt/SCM")]

# Machado et al. (2009), severity 1.0, rows act on linear (R, G, B)
MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}
VIEWS = ("normal", "deutan", "protan", "tritan", "grey")
VIEW_LABEL = {"normal": "normal vision", "deutan": "deuteranopia", "protan": "protanopia", "tritan": "tritanopia",
              "grey": "greyscale"}


# ------------------------------------------------------------------ colour maps
def scm_path(name):
    for d in SCM_DIRS:
        f = d / f"{name}.cpt"
        if f.exists():
            return f
    raise FileNotFoundError(f"{name}.cpt not in {[str(d) for d in SCM_DIRS]}")


def read_scm(name):
    """The colours of an SCM CPT, low to high, sRGB 0-1: the start colour of every slice and the end of the last."""
    rows = []
    for line in scm_path(name).read_text().splitlines():
        t = line.split()
        if not t or t[0][0] in "#BFN":
            continue
        if "/" in t[1]:                                   # z0 r/g/b z1 r/g/b
            c0, c1 = (list(map(float, x.split("/"))) for x in (t[1], t[3]))
        else:                                             # z0 r g b z1 r g b
            c0, c1 = list(map(float, t[1:4])), list(map(float, t[5:8]))
        rows.append((c0, c1))
    c = np.array([r[0] for r in rows] + [rows[-1][1]]) / 255.0
    return c


def scm_colors(name, reverse=False, lo=0.0, hi=1.0, n=256):
    """Colours of `name` (optionally reversed and cut to the fraction [lo, hi] of the map), resampled to n."""
    c = read_scm(name)
    if reverse:
        c = c[::-1]
    x = np.linspace(0, 1, len(c))
    q = np.linspace(lo, hi, n)
    return np.stack([np.interp(q, x, c[:, j]) for j in range(3)], axis=1)


def scm_cmap(name, reverse=False, lo=0.0, hi=1.0, bad=None):
    cm = ListedColormap(scm_colors(name, reverse, lo, hi), name=f"{name}{'_r' if reverse else ''}")
    if bad is not None:
        cm.set_bad(bad)
    return cm


# ------------------------------------------------------------------ colour science
def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def luminance(rgb):
    return srgb_to_linear(rgb) @ np.array([0.2126, 0.7152, 0.0722])


def simulate(rgb, view):
    """sRGB (..., 3) in 0-1 as seen with a dichromacy (Machado 2009, severity 1) or in greyscale."""
    rgb = np.asarray(rgb, float)
    if view == "normal":
        return rgb
    if view == "grey":
        y = linear_to_srgb(luminance(rgb))
        return np.repeat(y[..., None], 3, axis=-1)
    lin = srgb_to_linear(rgb) @ MACHADO[view].T
    return linear_to_srgb(lin)


_M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    xyz = srgb_to_linear(rgb) @ _M_XYZ.T / _WHITE
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def de2000(lab1, lab2):
    """CIEDE2000 colour difference (kL = kC = kH = 1), broadcasting over leading axes."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, float), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, float), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hs = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2)
    hbp = np.where(C1p * C2p == 0, hs, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# ------------------------------------------------------------------ metrics
def palette_metrics(colors, diverging=False, dot=(0.10, 0.10, 0.10), halo=(1.0, 1.0, 1.0)):
    """Numbers for one colour map (256 x 3 sRGB, low to high) under every view.

    lightness: L* at the low end, the centre and the high end; monotonic (sequential) or the largest asymmetry of the
      two arms |L*(t) - L*(1 - t)| (diverging), normal vision.
    uniformity: coefficient of variation of the CIEDE2000 step between neighbouring colours (0 = perfectly even).
    per view: length = summed CIEDE2000 along the map (how many just-noticeable steps it holds), as a fraction of
      normal vision; weakest = smallest CIEDE2000 between colours 1/8 of the range apart (9-class legibility);
      arms (diverging) = smallest CIEDE2000 between colours at the same distance either side of the centre, for
      distances 0.25-0.5 of the range (can a slow and a fast anomaly of equal size be told apart).
    dots: smallest CIEDE2000 between the event dot core and any colour of the map, and the fraction of the map
      closer than 15 to the core (where the white halo has to carry the dot).
    """
    colors = np.asarray(colors, float)
    out = {}
    lab = rgb_to_lab(colors)
    L = lab[:, 0]
    n = len(colors)
    step = de2000(lab[:-1], lab[1:])
    out["L_lo"], out["L_mid"], out["L_hi"] = float(L[0]), float(L[n // 2]), float(L[-1])
    if diverging:
        out["L_asym"] = float(np.max(np.abs(L - L[::-1])))
    else:
        dL = np.diff(L)
        out["L_monotonic"] = bool(np.all(dL > 0) or np.all(dL < 0))
    out["uniformity_cv"] = float(np.std(step) / np.mean(step))
    k8 = (n - 1) // 8
    base_len = None
    for v in VIEWS:
        labv = rgb_to_lab(simulate(colors, v))
        ln = float(de2000(labv[:-1], labv[1:]).sum())
        base_len = ln if v == "normal" else base_len
        out[f"len_{v}"] = ln
        out[f"lenfrac_{v}"] = ln / base_len
        out[f"weakest_{v}"] = float(de2000(labv[:-k8], labv[k8:]).min())
        if diverging:
            c = n // 2
            ds = np.arange(int(0.25 * n), int(0.5 * n))
            out[f"arms_{v}"] = float(de2000(labv[np.clip(c - ds, 0, n - 1)], labv[np.clip(c + ds, 0, n - 1)]).min())
    dd = de2000(rgb_to_lab(np.array(dot))[None], lab)
    out["dot_min"] = float(dd.min())
    out["dot_close_frac"] = float((dd < 15).mean())
    out["halo_min"] = float(de2000(rgb_to_lab(np.array(halo))[None], lab).min())
    return out


# ------------------------------------------------------------------ atlas
def js_ramp(colors, n=17):
    """n stops [[t, r, g, b], ...] with t from -1 (low end) to 1 (high end), RGB 0-255 integers."""
    colors = np.asarray(colors, float)
    x = np.linspace(0, 1, len(colors))
    q = np.linspace(0, 1, n)
    rgb = np.stack([np.interp(q, x, colors[:, j]) for j in range(3)], axis=1)
    return [[round(float(2 * t - 1), 4)] + [int(round(255 * v)) for v in c] for t, c in zip(q, rgb)]


def js_ramp_error(colors, ramp):
    """Largest CIEDE2000 between the full map and the linear-in-sRGB interpolation of the ramp stops."""
    colors = np.asarray(colors, float)
    t = np.array([r[0] for r in ramp])
    c = np.array([r[1:] for r in ramp], float) / 255
    q = np.linspace(-1, 1, len(colors))
    rec = np.stack([np.interp(q, t, c[:, j]) for j in range(3)], axis=1)
    return float(de2000(rgb_to_lab(colors), rgb_to_lab(rec)).max())
