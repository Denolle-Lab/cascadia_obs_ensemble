"""Paper numbers of two products folders, before and after (the definitions of mag_converged_paper.numbers).

Each folder holds data/catalog_v4/clusters_zbz, data/catalog_v4/mc_maps and paper_sciadv/figures laid out as in the
repository. Writes one row per paper number (where it is, what the text says now, before, after, changed, how). Used
for D-B105 (Mc maps without the events flagged as not earthquakes): before = main, after = this branch.

    python paper_numbers_compare.py --before <folder> --after <folder> --out <csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mag_converged_paper as mcp
import mag_paths


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
    t["changed"] = t.before.astype(str) != t.after.astype(str)
    t[["section", "item", "text", "before", "after", "changed", "how"]].to_csv(a.out, index=False)
    for r in t[t.changed].itertuples():
        print(f"- {r.section} | {r.item}\n    before: {r.before}\n    after:  {r.after}")
    print(f"{int(t.changed.sum())} of {len(t)} numbers change; wrote {a.out}")


if __name__ == "__main__":
    main()
