"""Track B: maps of every run d*_s* of one region (default M0 = margin_m0_v2, D-B044; also the tiles restarted
from M0, D-B046): dlnVp and dlnVs (change from that run's start, %) at 7, 15, 25 and 40 km on the interior nodes,
the five tile boxes, how many nodes moved; plus the start model (Vp) at the same depths.
Outputs: <outdir> (default data/tomo_v1/b2_merge/m0/) maps_<tag>.png, start_vp.png, summary.md.

  python b2_m0_report.py [REGION [OUTDIR]]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b2_fwd_common import read_mod  # noqa: E402

T = load_params()["tomodd"]
REG = sys.argv[1] if len(sys.argv) > 1 else "margin_m0_v2"
RUNS = Path(T["runs"])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "data" / "tomo_v1" / "b2_merge" / "m0"
TILES = ["mendocino_gorda_c_v2", "blanco_v2", "oregon_v2", "washington_v2", "ridges_north_v2"]
DEPTHS = [7, 15, 25, 40]


def tile_boxes(ax):
    for t in TILES:
        r = T["regions"][t]
        (a, b), (c, d) = r["events_lat"], r["events_lon"]
        ax.plot([c, d, d, c, c], [a, a, b, b, a], color="0.3", lw=0.6)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    tags = sorted(p.name for p in (RUNS / REG).glob("d*_s*") if "partial" not in p.name and (p / "Output_Files" / "model_geo.npz").exists())
    lines = [f"# Runs of {REG}", "",
             "dlnV = change from the run's start (M0: D-B005 model averaged over 31 km boxes; tiles *_m0: M0), "
             "interior nodes; % of nodes "
             "changed by more than 0.5 %; rms over nodes changed.", "",
             "| tag | depth km | Vp nodes changed % | rms dlnVp changed % | Vs nodes changed % | rms dlnVs changed % |",
             "|---|---|---|---|---|---|"]
    start = None
    for tag in tags:
        run = RUNS / REG / tag
        g = np.load(run / "Output_Files" / "model_geo.npz")
        m0 = read_mod(run / "MOD")
        lon, lat, z = g["lon"][1:-1], g["lat"][1:-1], g["z"][1:-1]
        dp = g["vp"][1:-1, 1:-1, 1:-1] / m0["vp"][1:-1, 1:-1, 1:-1] - 1
        ds = g["vs"][1:-1, 1:-1, 1:-1] / m0["vs"][1:-1, 1:-1, 1:-1] - 1
        start = (lon, lat, z, m0["vp"][1:-1, 1:-1, 1:-1])
        fig, ax = plt.subplots(2, len(DEPTHS), figsize=(3.2 * len(DEPTHS), 7.4), squeeze=False)
        for j, zd in enumerate(DEPTHS):
            k = int(np.abs(z - zd).argmin())
            for i, (d, nm) in enumerate(((dp, "Vp"), (ds, "Vs"))):
                a = ax[i, j]
                pc = a.pcolormesh(lon, lat, 100 * d[:, :, k].T, cmap="RdBu", vmin=-6, vmax=6, shading="nearest")
                tile_boxes(a)
                a.set_xlim(lon[0], lon[-1]); a.set_ylim(lat[0], lat[-1])
                a.set_aspect(1 / np.cos(np.radians(45)))
                a.set_title(f"{tag} dln{nm} {z[k]:g} km", fontsize=8)
                a.tick_params(labelsize=7)
                ch = np.abs(d[:, :, k]) > 0.005
                lines_row = (tag, z[k])
                if nm == "Vp":
                    vp_ch, vp_rms = 100 * ch.mean(), 100 * np.sqrt(np.mean(d[:, :, k][ch] ** 2)) if ch.any() else 0
                else:
                    lines.append(f"| {lines_row[0]} | {lines_row[1]:g} | {vp_ch:.0f} | {vp_rms:.2f} | {100 * ch.mean():.0f} | "
                                 f"{100 * np.sqrt(np.mean(d[:, :, k][ch] ** 2)) if ch.any() else 0:.2f} |")
        fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.03, pad=0.06, label="change from start (%)")
        fig.suptitle(f"{REG} {tag}: change from start; tile boxes in grey", fontsize=9)
        fig.savefig(OUT / f"maps_{tag}.png", dpi=130)
        plt.close(fig)
    if start is not None:
        lon, lat, z, vp = start
        fig, ax = plt.subplots(1, len(DEPTHS), figsize=(3.2 * len(DEPTHS), 3.9), squeeze=False)
        for j, zd in enumerate(DEPTHS):
            k = int(np.abs(z - zd).argmin())
            pc = ax[0, j].pcolormesh(lon, lat, vp[:, :, k].T, cmap="viridis", shading="nearest")
            tile_boxes(ax[0, j])
            ax[0, j].set_aspect(1 / np.cos(np.radians(45)))
            ax[0, j].set_title(f"start Vp {z[k]:g} km", fontsize=8)
            ax[0, j].tick_params(labelsize=7)
            fig.colorbar(pc, ax=ax[0, j], orientation="horizontal", fraction=0.05, pad=0.12, label="km/s")
        fig.savefig(OUT / "start_vp.png", dpi=130)
        plt.close(fig)
    scan = RUNS / REG / "summary_runs.csv"
    if scan.exists():
        s = pd.read_csv(scan)
        cols = ["tag", "damp", "rms_wt_final", "cnd_first", "cnd_last", "aq_repeats", "events", "rms_dlnvp_pct",
                "rms_dlnvs_pct", "shift_h_med_km", "shift_z_abs_med_km"]
        s = s[[c for c in cols if c in s.columns]]
        lines += ["", "## Runs", "", "| " + " | ".join(s.columns) + " |", "|" + "---|" * len(s.columns)]
        lines += ["| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |"
                  for r in s.itertuples(index=False)]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    write_manifest("b2_m0_report", OUT, inputs=[RUNS / REG / t / "Output_Files" / "model_geo.npz" for t in tags],
                   outputs=sorted(OUT.glob("*.png")) + [OUT / "summary.md"], params={"region": T["regions"][REG]})
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
