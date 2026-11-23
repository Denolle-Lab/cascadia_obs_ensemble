"""Track B, step B2: summarise tomoDD runs of one region (L-curve and run diagnostics).

For each run directory <runs>/<region>/<tag> with Output_Files/Vp_model.dat:
  final absolute and weighted catalog RMS (last "Residual summary" of tomoDD-SE.log), first-iteration RMS,
  last LSQR condition number, number of passes and air-quake repeats, events relocated,
  model change against the starting MOD on interior nodes: RMS and max of dlnVp, dlnVs, d(Vp/Vs),
  fraction of interior nodes changed by > 0.5%, mean hypocentre shift from the v4-NLL3D start.
Writes <runs>/<region>/summary_runs.csv and data/tomo_v1/b2_tomodd/<region>_lcurve.png.

  python b2_tomodd_summary.py REGION [TAG_GLOB]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402

T = load_params()["tomodd"]


def read_mod(path):
    lines = Path(path).read_text().split("\n")
    _, nx, ny, nz = lines[0].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    vals = np.array(" ".join(lines[4:]).split(), float)
    vp = vals[: nx * ny * nz].reshape(nz, ny, nx)
    r = vals[nx * ny * nz: 2 * nx * ny * nz].reshape(nz, ny, nx)
    return vp, vp / r


def read_out(path, shape):
    v = np.array(Path(path).read_text().split(), float)
    return v.reshape(shape)


def parse_log(log):
    txt = Path(log).read_text(errors="replace")
    abs_rms = [float(x) for x in re.findall(r"absolute ct rms \[s\] =\s+([-\d.]+)", txt)]
    wt_rms = [float(x) for x in re.findall(r"weighted ct rms \[s\][^=]*=\s+([-\d.]+)", txt)]
    cnd = [float(x) for x in re.findall(r"acond \(CND\)=\s*([-\d.]+)", txt)]
    aq = [int(x) for x in re.findall(r"Number of air quakes \(AQ\) =\s+(\d+)", txt)]
    its = re.findall(r"===ITERATION\s+(\d+)\s+\(\s*(\d+)\)", txt)
    return dict(rms_abs_start=abs_rms[0] if abs_rms else np.nan, rms_abs_final=abs_rms[-1] if abs_rms else np.nan,
                rms_wt_start=wt_rms[0] if wt_rms else np.nan, rms_wt_final=wt_rms[-1] if wt_rms else np.nan,
                cnd_last=cnd[-1] if cnd else np.nan, cnd_first=cnd[0] if cnd else np.nan,
                passes=len(its), iterations=int(its[-1][0]) if its else 0, aq_repeats=sum(1 for a in aq if a > 0))


def main():
    region = sys.argv[1]
    pat = sys.argv[2] if len(sys.argv) > 2 else "*"
    base = Path(T["runs"]) / region
    vp0, vs0 = read_mod(base / "inputs" / "MOD")
    inner = (slice(1, -1), slice(1, -1), slice(1, -1))
    start = pd.read_csv(base / "inputs" / "events.csv").set_index("ID")
    rows = []
    for run in sorted(base.glob(pat)):
        f = run / "Output_Files" / "Vp_model.dat"
        if not f.exists() or run.name == "inputs":
            continue
        inp = (run / "tomoDD-SE.inp").read_text().split("\n")
        i = [k for k, l in enumerate(inp) if l.startswith("* weight1")][0]
        smooth = inp[i + 1].split()[:3]
        sets = [l.split() for l in inp if re.match(r"^\s+\d+\s+-9", l)]
        vp = read_out(f, vp0.shape)
        vs = read_out(run / "Output_Files" / "Vs_model.dat", vp0.shape)
        dlp = (vp / vp0 - 1)[inner]
        dls = (vs / vs0 - 1)[inner]
        drat = (vp / vs - vp0 / vs0)[inner]
        rel = pd.read_csv(run / "Output_Files" / "tomoDD.reloc", sep=r"\s+", header=None)
        rel = rel.set_index(0)
        s = start.reindex(rel.index)
        dh = np.hypot((rel[1] - s.lat) * 111.19, (rel[2] - s.lon) * 111.19 * np.cos(np.radians(s.lat)))
        dz = rel[3] - s.depth_bsl_km
        rows.append(dict(tag=run.name, damp=float(sets[0][10]) if sets else np.nan, smooth_x=float(smooth[0]),
                         smooth_z=float(smooth[2]), **parse_log(run / "tomoDD-SE.log"),
                         events=len(rel), rms_dlnvp_pct=100 * np.sqrt(np.mean(dlp ** 2)),
                         rms_dlnvs_pct=100 * np.sqrt(np.mean(dls ** 2)), max_abs_dlnvp_pct=100 * np.abs(dlp).max(),
                         max_abs_dlnvs_pct=100 * np.abs(dls).max(), rms_dvpvs=np.sqrt(np.mean(drat ** 2)),
                         frac_nodes_changed=float(np.mean((np.abs(dlp) > 0.005) | (np.abs(dls) > 0.005))),
                         shift_h_med_km=float(dh.median()), shift_z_med_km=float(dz.median()),
                         shift_z_abs_med_km=float(dz.abs().median())))
    df = pd.DataFrame(rows).round(4)
    df.to_csv(base / "summary_runs.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, lab in ((axs[0], "rms_dlnvp_pct", "RMS dlnVp (%)"), (axs[1], "rms_dlnvs_pct", "RMS dlnVs (%)")):
        for sx, g in df.groupby("smooth_x"):
            g = g.sort_values("damp")
            ax.plot(g[col], g.rms_abs_final, "o-", label=f"smooth {sx:g}")
            for r in g.itertuples():
                ax.annotate(f"{r.damp:g}", (getattr(r, col), r.rms_abs_final), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_xlabel(lab)
        ax.set_ylabel("final absolute RMS (s)")
        ax.legend(fontsize=7)
        ax.set_title(f"{region}: L-curve (labels = damping)", fontsize=9)
    fig.tight_layout()
    out = REPO / "data/tomo_v1/b2_tomodd"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{region}_lcurve.png", dpi=150)
    df.to_csv(out / f"{region}_summary_runs.csv", index=False)


if __name__ == "__main__":
    main()
