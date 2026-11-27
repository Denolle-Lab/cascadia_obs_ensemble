"""Paper numbers of two products folders, before and after (the definitions of mag_converged_paper.numbers).

Each folder holds data/catalog_v4/clusters_zbz, data/catalog_v4/mc_maps and paper_sciadv/figures laid out as in the
repository. Writes one row per paper number (where it is, what the text says now, before, after, changed, how). Used
for D-B105 (Mc maps without the events flagged as not earthquakes; before = main 08d9dbe4) and D-B106 (clustering
without them; before = 8452c6e5, the D-B105 commit). Fig. 5 rows: the numbers printed on the figure.

    python paper_numbers_compare.py --before <folder> --after <folder> --out <csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import mag_converged_paper as mcp
import mag_paths


SETTINGS = [("ridges and transforms", ["AX", "EN", "BL", "GO"]), ("Mendocino triple junction", ["MT"]),
            ("forearc and onshore", ["WA", "OR", "PU", "ON"])]   # Fig. 5 A-C (workflow/08_figures/v4_figures.py)


def fig5_rows(p) -> dict:
    """The numbers printed on Fig. 5 (same formulas as v4_figures.fig_clusters)."""
    e = p.ev[p.ev.in_analysis & p.ev.log_T.notna() & p.ev.log_R.notna()]
    eta0 = p.reg.log_eta0
    abc = "; ".join(f"{name}: {len(x):,} events, {(x.log_eta < x.region.map(eta0)).mean() * 100:.0f}% clustered"
                    for name, regs in SETTINGS for x in [e[e.region.isin(regs)]])
    d = ", ".join(f"{r} {int(p.reg.loc[r, 'n_analysis']):,}" for _, regs in SETTINGS for r in regs)
    ee = ", ".join(f"{r} ({int(p.st.loc[r, 'n_fam_ge5'])})" for _, regs in SETTINGS for r in regs)
    return {"Fig. 5A-C labels": abc, "Fig. 5D event counts (right of the bars)": d,
            "Fig. 5E labels (families >= 5)": ee}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, default=mcp.REPO)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mag = mag_paths.v41("converged")
    pa = mcp.Products(a.after, mag)
    t = mcp.numbers(mcp.Products(a.before, mag), pa)
    t = t.rename(columns={"stored": "before", "converged": "after"})
    # rows numbers() fills from fixed text rather than from the products (e.g. the ComCat anchors) differ even when
    # a folder is compared with itself: they do not depend on the products, so before = after there
    self_ = mcp.numbers(pa, pa)
    fixed = self_.stored.astype(str) != self_.converged.astype(str)
    t.loc[fixed, "before"] = t.loc[fixed, "after"]
    pb = mcp.Products(a.before, mag)
    fb, fa = fig5_rows(pb), fig5_rows(pa)
    t = pd.concat([t, pd.DataFrame([dict(section="Fig. 5", item=k, text="(figure)", before=fb[k], after=fa[k],
                                         how="events_zbz.csv.gz, regions_zbz.csv, setting_stats.csv (v4_figures.py)")
                                    for k in fb])], ignore_index=True)
    t["changed"] = t.before.astype(str) != t.after.astype(str)
    t[["section", "item", "text", "before", "after", "changed", "how"]].to_csv(a.out, index=False)
    for r in t[t.changed].itertuples():
        print(f"- {r.section} | {r.item}\n    before: {r.before}\n    after:  {r.after}")
    print(f"{int(t.changed.sum())} of {len(t)} numbers change; wrote {a.out}")


if __name__ == "__main__":
    main()
