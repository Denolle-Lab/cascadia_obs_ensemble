"""Track B, M0 (D-B044, D-B046): the whole-margin model itself (absolute Vp, Vs, Vp/Vs), not its change from the start.

Run margin_m0_v2/<tag> (default d200_s480). Maps at 3, 9, 15, 25 and 40 km on the interior nodes, with the
coastline, the five tile boxes, and the nodes that the final rays do not sample (P DWS < 1 in the last DWS
block in tomoDD.vel) greyed out: their values are the start model, not data. East-west sections at 41.0, 44.5 and
47.5 N (Vp and Vp/Vs, 0-60 km) with the relocated M0 events within 15 km of the section.
Outputs: data/tomo_v1/b2_merge/m0/model_maps_<tag>.png, model_sections_<tag>.png.

  python b2_m0_model_figs.py [tag]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b2_tile_overlap import read_reloc  # noqa: E402

T = load_params()["tomodd"]
REG = "margin_m0_v2"
RUNS = Path(T["runs"])
OUT = REPO / "data" / "tomo_v1" / "b2_merge" / "m0"
COAST = REPO / "data" / "tomo_v1" / "b2_res_tests" / "washington_v2" / "coast_i.txt"
TILES = ["mendocino_gorda_c_v2", "blanco_v2", "oregon_v2", "washington_v2", "ridges_north_v2"]
DEPTHS = [3, 9, 15, 25, 40]
SECTIONS = [41.0, 44.5, 47.5]
DWS_MIN = 1      # no ray at the node: value = start model. (tile threshold 100 does not transfer: coarse grid, dataset weight scale)


def coast_segments():
    segs, cur = [], []
    for line in COAST.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                segs.append(np.array(cur))
            cur = []
        elif line.strip():
            cur.append([float(v) for v in line.split()[:2]])
    if cur:
        segs.append(np.array(cur))
    return segs


def dws(run, shape):
    lines = (run / "Output_Files" / "tomoDD.vel").read_text().split("\n")
    starts = [i for i, l in enumerate(lines) if "DWS for P-wave" in l]
    if not starts:
        return None
    nx, ny, nz = shape
    i0 = starts[-1]
    a = np.array([lines[i0 + 1 + r].split() for r in range(ny * nz)], float).reshape(nz, ny, nx)
    return a.transpose(2, 1, 0)


def deco(ax, segs, lon_lim, lat_lim):
    for s in segs:
        ax.plot(s[:, 0], s[:, 1], color="k", lw=0.4)
    for t in TILES:
        r = T["regions"][t]
        (a, b), (c, d) = r["events_lat"], r["events_lon"]
        ax.plot([c, d, d, c, c], [a, a, b, b, a], color="0.35", lw=0.5, ls="--")
    ax.set_xlim(*lon_lim)
    ax.set_ylim(*lat_lim)
    ax.set_aspect(1 / np.cos(np.radians(45)))
    ax.tick_params(labelsize=6)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = sys.argv[1] if len(sys.argv) > 1 else "d200_s480"
    run = RUNS / REG / tag
    g = np.load(run / "Output_Files" / "model_geo.npz")
    lon, lat, z = g["lon"][1:-1], g["lat"][1:-1], g["z"][1:-1]
    vp, vs = g["vp"][1:-1, 1:-1, 1:-1], g["vs"][1:-1, 1:-1, 1:-1]
    D = dws(run, g["vp"].shape)
    samp = D[1:-1, 1:-1, 1:-1] >= DWS_MIN if D is not None else np.ones_like(vp, bool)
    segs = coast_segments()
    lon_lim, lat_lim = (lon[0] - 0.2, lon[-1] + 0.2), (lat[0] - 0.15, lat[-1] + 0.15)
    rows = [("Vp (km/s)", vp, "RdBu"), ("Vs (km/s)", vs, "RdBu"), ("Vp/Vs", vp / vs, "RdBu_r")]
    fig, ax = plt.subplots(3, len(DEPTHS), figsize=(2.9 * len(DEPTHS), 12.2), squeeze=False)
    for j, zd in enumerate(DEPTHS):
        k = int(np.abs(z - zd).argmin())
        for i, (name, arr, cmap) in enumerate(rows):
            a = ax[i, j]
            v = arr[:, :, k]
            ok = samp[:, :, k]
            if name == "Vp/Vs":
                lo, hi = 1.60, 2.00
            else:
                ref = v[ok] if ok.sum() > 10 else v.ravel()
                lo, hi = np.percentile(ref, [2, 98])
            pc = a.pcolormesh(lon, lat, v.T, cmap=cmap, vmin=lo, vmax=hi, shading="nearest")
            grey = np.where(ok, np.nan, 1.0)
            a.pcolormesh(lon, lat, grey.T, cmap="Greys", vmin=0, vmax=1.6, alpha=0.75, shading="nearest")
            deco(a, segs, lon_lim, lat_lim)
            a.set_title(f"M0 {name.split()[0]}, {z[k]:g} km", fontsize=8)
            fig.colorbar(pc, ax=a, orientation="horizontal", fraction=0.045, pad=0.07).ax.tick_params(labelsize=6)
    fig.suptitle(f"M0 = whole-margin tomoDD model ({REG} {tag}); grey = P DWS < {DWS_MIN} (start model, "
                 "not data); dashed = tile boxes. Colour limits per panel.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    f1 = OUT / f"model_maps_{tag}.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    ev = read_reloc(run / "Output_Files" / "tomoDD.reloc_geo")
    zz = z[(z >= 0) & (z <= 60)]
    kz = np.isin(z, zz)
    fig, ax = plt.subplots(len(SECTIONS), 2, figsize=(13, 3.0 * len(SECTIONS)), squeeze=False)
    for i, la in enumerate(SECTIONS):
        jy = int(np.abs(lat - la).argmin())
        near = ev[(np.abs(ev.lat - lat[jy]) * 111.2 <= 15)]
        for c, (name, arr, cmap, lim) in enumerate((("Vp (km/s)", vp, "RdBu", (4.5, 8.2)),
                                                     ("Vp/Vs", vp / vs, "RdBu_r", (1.6, 2.0)))):
            a = ax[i, c]
            sec = arr[:, jy, :][:, kz]
            pc = a.pcolormesh(lon, zz, sec.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="nearest")
            grey = np.where(samp[:, jy, :][:, kz], np.nan, 1.0)
            a.pcolormesh(lon, zz, grey.T, cmap="Greys", vmin=0, vmax=1.6, alpha=0.75, shading="nearest")
            if name.startswith("Vp ("):
                cs = a.contour(lon, zz, sec.T, levels=[6.0, 7.0, 7.8], colors="k", linewidths=0.5)
                a.clabel(cs, fontsize=6, fmt="%.1f")
            a.scatter(near.lon, near.depth, s=1.5, c="k", alpha=0.5, linewidths=0)
            a.invert_yaxis()
            a.set(xlim=lon_lim, ylabel="depth (km bsl)")
            a.set_title(f"M0 {name.split()[0]}, E-W at {lat[jy]:.1f} N ({len(near)} M0 events within 15 km)", fontsize=8)
            a.tick_params(labelsize=7)
            fig.colorbar(pc, ax=a, fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)
    fig.tight_layout()
    f2 = OUT / f"model_sections_{tag}.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    write_manifest(f"b2_m0_model_figs_{tag}", OUT, inputs=[run / "Output_Files" / "model_geo.npz",
                                                           run / "Output_Files" / "tomoDD.vel"],
                   outputs=[f1, f2], params={"tag": tag, "dws_min": DWS_MIN, "depths": DEPTHS, "sections": SECTIONS},
                   counts={"sampled_nodes_frac": float(samp.mean())})
    print(f1, f2, f"sampled fraction {samp.mean():.2f}")


if __name__ == "__main__":
    main()
