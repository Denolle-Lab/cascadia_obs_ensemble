"""Track B, B4 acceptance: event-paired depth comparison with two published local catalogs.

Pairs (v4 event ID <-> published event) were built once in the literature comparison of 2026-11-24
(data/tomo_v1/depth_literature/depth_literature.html): one-to-one, greedy by |dt| then distance.
* Mount St Helens, Hirao et al. (2025): |dt| <= 2 s, <= 5 km; 2,369 pairs; both catalogs in km below sea level.
* Blanco, Lange et al. (2026) OBS catalog, absolute and relative (double-difference) locations: |dt| <= 3 s, <= 20 km;
  11,298 / 10,706 pairs. Lange's depth datum is not stated; read here as km below sea level, as in the report.
Pair tables: /wd1/mdenolle_data/tomo_v1/depth_literature/pairs/*.csv (ID, ref_depth, ...). A pair stays valid for
any relocation of the same v4 events, so each catalog is compared with the same published events.

For every catalog given (B4 relocation catalogs or v4-NLL3D): median and 10/90th percentiles of (catalog - published)
depth, by location tier (A-D, A-C, A-B) and, for Blanco, by segment (longitude of the published epicentre, the
segments of the report). Only events located in 3D (tier A-D) are used.

  python b4_lit_paired.py [--out DIR] LABEL=CATALOG.csv [LABEL=CATALOG.csv ...]
      -> data/tomo_v1/depth_literature/paired_<first label>.md (and .csv); --out DIR writes them to DIR instead
         (pipeline runs into a scratch root, D-B110)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, write_manifest  # noqa: E402

PAIRS = Path("/wd1/mdenolle_data/tomo_v1/depth_literature/pairs")
OUT = REPO / "data" / "tomo_v1" / "depth_literature"
TIERS = {"A-D": list("ABCD"), "A-C": list("ABC"), "A-B": list("AB")}
SEGS = [(-131.0, -129.3, "West (WBD/EBD)"), (-129.3, -128.75, "Surveyor"), (-128.75, -128.45, "Cascadia Depression"),
        (-128.45, -127.1, "Blanco Ridge"), (-127.1, -126.0, "Gorda Depression")]


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, med=np.nan, p10=np.nan, p90=np.nan)
    p = np.percentile(x, [10, 50, 90])
    return dict(n=len(x), med=p[1], p10=p[0], p90=p[2])


def compare(cats, pairs, ref, segs=None):
    rows = []
    for lab, c in cats.items():
        m = pairs.merge(c[["ID", "depth", "tier"]], on="ID", how="inner")
        for tl, tt in TIERS.items():
            s = m[m.tier.isin(tt)]
            groups = [("all", s)]
            if segs is not None:
                groups += [(name, s[(s.ref_lon >= a) & (s.ref_lon < b)]) for a, b, name in segs]
            for g, d in groups:
                r = q(d.depth - d.ref_depth)
                rows.append(dict(reference=ref, catalog=lab, tiers=tl, group=g, **r,
                                 cat_med=float(np.median(d.depth)) if len(d) else np.nan,
                                 ref_med=float(np.median(d.ref_depth)) if len(d) else np.nan))
    return pd.DataFrame(rows)


def main():
    global OUT
    if len(sys.argv) > 2 and sys.argv[1] == "--out":
        OUT = Path(sys.argv[2])
        del sys.argv[1:3]
    cats = {}
    for a in sys.argv[1:]:
        lab, path = a.split("=", 1)
        cats[lab] = pd.read_csv(path, usecols=["ID", "depth", "tier"])
    if not cats:
        raise SystemExit(__doc__)
    hir = pd.read_csv(PAIRS / "msh_hirao2025_pairs.csv")
    tabs = [compare(cats, hir, "Mount St Helens, Hirao et al. 2025")]
    for k in ("abs", "rel"):
        lg = pd.read_csv(PAIRS / f"blanco_lange2026_{k}_pairs.csv")
        tabs.append(compare(cats, lg, f"Blanco, Lange et al. 2026 ({'absolute' if k == 'abs' else 'relative'})", SEGS))
    t = pd.concat(tabs, ignore_index=True)
    first = next(iter(cats))
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"paired_{first}.csv", index=False, float_format="%.3f")
    L = [f"# Event-paired depths against published local catalogs ({', '.join(cats)})", "",
         "Catalog minus published depth, km (positive = our event deeper); median [p10, p90] and n. Events located in 3D "
         "only. Mount St Helens: both in km bsl. Blanco: Lange datum read as km bsl. Script: "
         "workflow/10_tomography/b4_lit_paired.py.", ""]
    for ref, g in t.groupby("reference", sort=False):
        L += [f"## {ref}", "", "| tiers | group | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
        for (tl, grp), h in g.groupby(["tiers", "group"], sort=False):
            cells = []
            for lab in cats:
                r = h[h.catalog == lab].iloc[0]
                cells.append("n=0" if r.n == 0 else f"{r.med:+.2f} [{r.p10:+.1f}, {r.p90:+.1f}] n={int(r.n)}")
            L.append(f"| {tl} | {grp} | " + " | ".join(cells) + " |")
        L.append("")
    (OUT / f"paired_{first}.md").write_text("\n".join(L) + "\n")
    write_manifest(f"b4_lit_paired_{first}", OUT, inputs=[Path(a.split("=", 1)[1]) for a in sys.argv[1:]]
                   + sorted(PAIRS.glob("*.csv")), outputs=[OUT / f"paired_{first}.md", OUT / f"paired_{first}.csv"],
                   params={"catalogs": {k: a.split("=", 1)[1] for k, a in zip(cats, sys.argv[1:])}},
                   counts={"rows": len(t)})
    print("\n".join(L))


if __name__ == "__main__":
    main()
