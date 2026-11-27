"""Quick-look maps and the first-look scatter plots of the external data (D-B114); for checking, not for the paper.

Writes derived/figures/quicklook_lattice.png and derived/figures/first_look_<tag>.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402


def main():
    fig_dir = C.DER / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    L = xr.open_dataset(C.DER / "lattice_external_geo_0p1.nc")
    sites = pd.read_csv(C.DER / "ghfdb_sites_area.csv")
    panels = [("age_myr", "plate age (Myr), Seton+2020", dict(cmap="viridis", vmin=0, vmax=12)),
              ("iso600_hsc_bsf", "600 C isotherm, HSC (km bsf)", dict(cmap="magma_r", vmin=0, vmax=20)),
              ("sed_thk_m", "sediment (m), GlobSed v3", dict(cmap="cividis", vmin=0, vmax=2000)),
              ("hf_smooth", "heat flow, smoothed (mW/m2)", dict(cmap="inferno", vmin=0, vmax=400)),
              ("rough10", "roughness 10 km (m)", dict(cmap="viridis", vmin=0, vmax=300)),
              ("fabric_az20", "fabric azimuth 20 km (deg)", dict(cmap="twilight", vmin=0, vmax=180))]
    fig, ax = plt.subplots(2, 3, figsize=(14, 11))
    for a, (k, t, kw) in zip(ax.ravel(), panels):
        v = L[k].values.astype(float)
        if k in ("rough10", "fabric_az20"):
            b = xr.open_dataset(C.DER / "bathy_roughness_lattice_0p1.nc")
            v = np.where(b.landfrac20.values > 0.05, np.nan, b[k].values)
        m = a.pcolormesh(L.lon, L.lat, v, shading="auto", **kw)
        if k == "hf_smooth":
            s = sites[sites.in_area]
            a.scatter(s.lon, s.lat, s=1, c="c", lw=0)
        a.set_title(t, fontsize=9)
        a.set_aspect(1 / np.cos(np.radians(45.5)))
        plt.colorbar(m, ax=a, shrink=0.8)
    plt.tight_layout()
    plt.savefig(fig_dir / "quicklook_lattice.png", dpi=80)

    seg = pd.read_csv(C.DER / f"segments_external_geo_{C.CATTAG}.csv")
    off = seg[seg.seg_type.isin(["ridge", "transform", "intraplate", "triple junction"])]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for a, x in zip(ax[:3], ("hf_med20", "rough10_med", "age_myr_med")):
        a.scatter(off[x], off.pct_swarm)
        for r in off.itertuples():
            a.annotate(r.seg, (getattr(r, x), r.pct_swarm), fontsize=8)
        a.set_xlabel(x)
        a.set_ylabel("pct_swarm")
    a = ax[3]
    for s in ("GP", "BL", "MFZ", "GR", "AX", "MTJ"):
        r = seg[seg.seg == s]
        if r.empty or not np.isfinite(r.bsf_p90.iloc[0]):
            continue
        r = r.iloc[0]
        a.plot([r.iso600_hsc_deep10_med, r.iso600_hsc_deep10_med], [0, 0], alpha=0)
        a.scatter(r.iso600_hsc_deep10_med, r.bsf_p90, c="C0")
        a.scatter(r.iso600_plate_deep10_med, r.bsf_p90, c="C1", marker="s")
        a.annotate(s, (r.iso600_hsc_deep10_med, r.bsf_p90), fontsize=8)
    lim = [0, 40]
    a.plot(lim, lim, "k:")
    a.set_xlabel("600 C isotherm at the deepest 10 % (km bsf; o HSC, s GDH1)")
    a.set_ylabel("bsf p90 (km)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"first_look_{C.CATTAG}.png", dpi=80)
    C.log(f"wrote figures in {fig_dir}")


if __name__ == "__main__":
    main()
