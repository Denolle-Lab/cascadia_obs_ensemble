"""QC figure for the A1 model: Vs and Vp depth slices, two E-W cross-sections, Gen0 weight."""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
QC = Path(__file__).resolve().parents[2] / "data/reloc_v5/a1_model"
d = np.load(QC / "qc_slices.npz")
deps = (2, 5, 10, 20, 30, 45)
fig, ax = plt.subplots(3, 6, figsize=(18, 13), constrained_layout=True)
for j, z in enumerate(deps):
    for i, (k, lo, hi) in enumerate((("vs", 1.5, 4.7), ("vp", 3, 8.2))):
        a = ax[i, j]
        im = a.pcolormesh(d["LON"], d["LAT"], d[k][:, :, j], vmin=lo, vmax=hi, cmap="turbo_r", shading="auto", rasterized=True)
        a.contour(d["LON"], d["LAT"], d["elev"], [0], colors="k", linewidths=0.5)
        a.set_title(f"{k[1].upper()} at {z} km bsl", fontsize=9); a.set_aspect(1.4)
    if j == 5:
        fig.colorbar(im, ax=ax[1, :], shrink=0.6, label="Vp km/s")
a = ax[2, 0]; a.pcolormesh(d["LON"], d["LAT"], d["w_c"], cmap="gray", shading="auto", rasterized=True)
a.contour(d["LON"], d["LAT"], d["elev"], [0], colors="r", linewidths=0.5); a.set_title("Gen0 weight (white=Gen0)", fontsize=9); a.set_aspect(1.4)
for n, (key, lab) in enumerate((("xs", "y=0 (45.0N, LAMBERT row)"), ("xs2", "41.3N (Gorda)"))):
    for m, ph in enumerate(("vs", "vp")):
        a = ax[2, 1 + 2 * n + m] if 1 + 2 * n + m < 6 else None
        if a is None: continue
        lo, hi = (1.5, 4.7) if ph == "vs" else (3, 8.2)
        a.pcolormesh(d["xs"], d["zs"], d[f"{ph}_{key}"].T, vmin=lo, vmax=hi, cmap="turbo_r", shading="auto", rasterized=True)
        a.set_ylim(60, -5); a.set_title(f"{ph[1].upper()} section {lab}", fontsize=9); a.set_xlabel("x km"); a.set_ylabel("km bsl")
ax[2, 5].axis("off") if False else None
fig.savefig(QC / "a1_model_qc.png", dpi=110)
print(QC / "a1_model_qc.png")
