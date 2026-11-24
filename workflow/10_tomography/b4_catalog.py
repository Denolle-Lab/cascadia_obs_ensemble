"""Track B, step B4: catalog from the NLLoc relocation in a registered model, depths back from the Earth-flattened
frame, and a comparison with v4-NLL3D (Track A iter1).

Input: Track A's catalog of the run (a2_build_catalog.py RUN -> /wd1/mdenolle_data/reloc_v5/catalog/
catalog_v5_trackA_<RUN>.csv: every v4 event, NLLoc location or v4 kept (tier U), tiers A-D/U). The time grids of
b4_time_grids.py use flattened depth z'; here depth, exp_depth and shift_z_km of the located events (loc_source
nll3d) are mapped back with z = Re (1 - exp(-z' / Re)) (0.07 km at 30 km, 0.2 km at 50 km). Tiers are kept (their
depth conditions move by less than the change).
Outputs: /wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_<RUN>.csv, and data/tomo_v1/b4/<RUN>/ (compare.md,
compare_by_region.csv, depth_hist.png).

  python b4_catalog.py RUN [--ref iter1_vpvs176]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, write_manifest  # noqa: E402

RE = 6371.0
CAT = Path("/wd1/mdenolle_data/reloc_v5/catalog")


def unflatten(z):
    return RE * (1 - np.exp(-np.asarray(z, float) / RE))


def km_h(a, b):
    y = (a.lat.values - b.lat.values) * 111.195
    x = (a.lon.values - b.lon.values) * 111.195 * np.cos(np.radians(a.lat.values))
    return np.hypot(x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default="iter1_vpvs176")
    a = ap.parse_args()
    c = pd.read_csv(CAT / f"catalog_v5_trackA_{a.run}.csv")
    nll = c.loc_source == "nll3d"
    for col in ("depth", "exp_depth"):
        c.loc[nll, col] = unflatten(c.loc[nll, col])
    c.loc[nll, "shift_z_km"] = c.loc[nll, "depth"] - c.loc[nll, "v4_depth"]
    c["depth_frame"] = np.where(nll, "true depth (EFT undone, b4_catalog.py)", "v4")
    out = CAT / f"catalog_b4_{a.run}.csv"
    c.to_csv(out, index=False, float_format="%.5g")
    r = pd.read_csv(CAT / f"catalog_v5_trackA_{a.ref}.csv").set_index("ID")
    c = c.set_index("ID")
    both = c.index[(c.loc_source == "nll3d")].intersection(r.index[r.loc_source == "nll3d"])
    cb, rb = c.loc[both], r.loc[both]
    d = pd.DataFrame({"region": cb.region, "tier_b4": cb.tier, "tier_ref": rb.tier,
                      "dz": cb.depth - rb.depth, "dh": km_h(cb, rb),
                      "dt": (pd.to_datetime(cb.time) - pd.to_datetime(rb.time)).dt.total_seconds(),
                      "rms_b4": cb.rms, "rms_ref": rb.rms, "above_b4": cb.above_surface, "above_ref": rb.above_surface})
    od = REPO / "data" / "tomo_v1" / "b4" / a.run
    od.mkdir(parents=True, exist_ok=True)
    g = d.groupby("region").agg(n=("dz", "size"), dz_med=("dz", "median"), absdz_med=("dz", lambda x: x.abs().median()),
                                absdz_p90=("dz", lambda x: x.abs().quantile(0.9)), dh_med=("dh", "median"),
                                dh_p90=("dh", lambda x: x.quantile(0.9)), dt_med=("dt", "median"),
                                rms_b4=("rms_b4", "median"), rms_ref=("rms_ref", "median"))
    g.loc["all"] = [len(d), d.dz.median(), d.dz.abs().median(), d.dz.abs().quantile(0.9), d.dh.median(),
                    d.dh.quantile(0.9), d.dt.median(), d.rms_b4.median(), d.rms_ref.median()]
    g.round(3).to_csv(od / "compare_by_region.csv")
    tiers = pd.DataFrame({"b4": c.tier.value_counts(), "ref": r.tier.value_counts()}).fillna(0).astype(int)
    L = [f"# B4 relocation {a.run} against {a.ref}", "",
         f"Catalog: {out} ({len(c)} events, {int(nll.sum())} located in 3D). Depths in km below sea level, "
         "EFT undone. Differences B4 minus reference, events located in both.", "",
         "## Tiers", "", "| tier | B4 | reference |", "|---|---|---|"]
    L += [f"| {t} | {row.b4} | {row.ref} |" for t, row in tiers.sort_index().iterrows()]
    L += ["", "## By region", "", "| region | n | dz med | abs dz med / p90 | dh med / p90 | dt med s | rms B4 / ref |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {int(v.n)} | {v.dz_med:+.2f} | {v.absdz_med:.2f} / {v.absdz_p90:.2f} | {v.dh_med:.2f} / "
          f"{v.dh_p90:.2f} | {v.dt_med:+.3f} | {v.rms_b4:.3f} / {v.rms_ref:.3f} |" for k, v in g.iterrows()]
    L += ["", f"Above the surface: B4 {100 * d.above_b4.mean():.1f} %, reference {100 * d.above_ref.mean():.1f} %.",
          "The ANSS check per tier is in the tier summary of a2_build_catalog.py (copied next to this file)."]
    (od / "compare.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regs = list(g.index[:-1])
    fig, ax = plt.subplots(1, len(regs), figsize=(2.6 * len(regs), 3.4), sharey=True, squeeze=False)
    for i, rg in enumerate(regs):
        s = d.region == rg
        bins = np.arange(-5, 61, 2)
        ax[0, i].hist(rb.depth[s.values], bins=bins, orientation="horizontal", histtype="step", color="0.5", label="ref")
        ax[0, i].hist(cb.depth[s.values], bins=bins, orientation="horizontal", histtype="step", color="C3", label="B4")
        ax[0, i].set_title(f"{rg} ({int(s.sum())})", fontsize=8)
        ax[0, i].tick_params(labelsize=7)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel("depth (km bsl)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(od / "depth_hist.png", dpi=130)
    write_manifest(f"b4_catalog_{a.run}", od, inputs=[CAT / f"catalog_v5_trackA_{a.run}.csv",
                                                      CAT / f"catalog_v5_trackA_{a.ref}.csv"],
                   outputs=[out, od / "compare.md", od / "compare_by_region.csv", od / "depth_hist.png"],
                   counts={"events": len(c), "nll3d": int(nll.sum()), "both": len(both)})
    print((od / "compare.md").read_text())


if __name__ == "__main__":
    main()
