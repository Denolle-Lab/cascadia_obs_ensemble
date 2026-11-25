"""Track B, B4: acceptance checks of a B4 catalog (handoff 2026-11-24 07:55, step 2) -> data/tomo_v1/b4/<RUN>/acceptance.md.

Axial: share of tier A-B events within 1.6 and 2.0 km below the seafloor (depth + surface_km); Waldhauser et al.
(2020): 85 % above 1.6 km, 90 % above 2.0 km. Other regions: median depth (km bsl) of the events located in this run,
B4 v0.3.0 and v4-NLL3D (iter1_vpvs176), and the median change from v0.3.0 (Puget and onshore should not move; Blanco
against Lange et al. 2026 is in data/tomo_v1/depth_literature/depth_literature.html: v0.3.0 was +2.2 km).

  python b4_acceptance.py RUN [--old b4_iter1_v0.3.0] [--ref iter1_vpvs176]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, write_manifest  # noqa: E402
from b4_catalog import CAT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--old", default="b4_iter1_v0.3.0")
ap.add_argument("--ref", default="iter1_vpvs176")
a = ap.parse_args()
f_new, f_old, f_ref = (CAT / f"catalog_b4_{a.run}.csv", CAT / f"catalog_b4_{a.old}.csv",
                       CAT / f"catalog_v5_trackA_{a.ref}.csv")
new, old, ref = (pd.read_csv(f, dtype={"tier": str}).set_index("ID") for f in (f_new, f_old, f_ref))
loc = new.loc_source == "nll3d"
ab = new.above_surface.astype("boolean").fillna(False)   # empty (no hypocentre, v5.1 tier U) is not above
L = [f"# Acceptance checks of {a.run}", "", f"Catalog {f_new}; compared with {a.old} and v4-NLL3D {a.ref}.", "",
     f"Located {int(loc.sum()):,}; above the surface: located {int((ab & loc).sum()):,}, "
     f"tier U at the v4 position {int((ab & (new.loc_source == 'v4')).sum()):,}; "
     f"tier U without a hypocentre {int((new.loc_source == 'none').sum()):,}.", "",
     "## Axial (Waldhauser et al. 2020: 85 % within 1.6 km, 90 % within 2.0 km below the seafloor)", "",
     "| set | n | median bsf | p10-p90 | <= 1.6 km | <= 2.0 km | pass (>= 85 % within 2.0 km) |", "|---|---|---|---|---|---|---|"]
ax = new[loc & (new.region == "Axial")]
for t, s in (("A-B", ax[ax.tier.isin(["A", "B"])]), ("A-D", ax)):
    z = s.depth + s.surface_km
    L.append(f"| {t} | {len(s):,} | {z.median():.2f} | {z.quantile(.1):.2f} to {z.quantile(.9):.2f} | "
             f"{100 * (z <= 1.6).mean():.0f} % | {100 * (z <= 2.0).mean():.0f} % | "
             f"{'yes' if (z <= 2.0).mean() >= 0.85 else 'no'} |")
both = new.index[loc].intersection(old.index[old.loc_source == "nll3d"]).intersection(ref.index[ref.loc_source == "nll3d"])
d = pd.DataFrame({"region": new.loc[both, "region"], "new": new.loc[both, "depth"], "old": old.loc[both, "depth"],
                  "ref": ref.loc[both, "depth"]})
d["dz"] = d.new - d.old
g = d.groupby("region").agg(n=("new", "size"), new=("new", "median"), old=("old", "median"), ref=("ref", "median"),
                            dz=("dz", "median"))
L += ["", "## Median depth by region (km bsl, events located in all three)", "",
      f"| region | n | {a.run} | {a.old} | v4-NLL3D | median change from {a.old} |", "|---|---|---|---|---|---|"]
L += [f"| {k} | {int(v.n):,} | {v.new:.2f} | {v.old:.2f} | {v.ref:.2f} | {v.dz:+.2f} |" for k, v in g.iterrows()]
od = REPO / "data" / "tomo_v1" / "b4" / a.run
od.mkdir(parents=True, exist_ok=True)
(od / "acceptance.md").write_text("\n".join(L) + "\n")
write_manifest(f"b4_acceptance_{a.run}", od, inputs=[f_new, f_old, f_ref], outputs=[od / "acceptance.md"],
               counts={"axial_AB": int(ax.tier.isin(["A", "B"]).sum()), "events_compared": len(both)})
print("\n".join(L))
