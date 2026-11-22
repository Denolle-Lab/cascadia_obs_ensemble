"""Merge updated literature matches and verified DOIs into the fault-surface product.

    python workflow/07_faults/merge_literature.py --matches blanco_strands_matches.csv --dois dois_nonblanco.csv

--matches: rows that replace lit_match, lit_confidence, lit_class, lit_new_candidate, lit_evidence, lit_refs and
           lit_dois for their surface_id (e.g. the re-matched Blanco strands).
--dois:    rows that add lit_dois, lit_urls, lit_refs_full (and doi_note) for their surface_id.
Updates fault_surfaces_v0_literature.csv, the summary and the GeoJSON properties. Every surface must end up
with at least one DOI; the script lists those without.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

V0 = Path(__file__).resolve().parents[2] / "data" / "faults" / "v0"
STEM = "fault_surfaces_v0"
LIT = ["lit_match", "lit_confidence", "lit_class", "lit_new_candidate", "lit_evidence", "lit_refs", "lit_dois", "lit_urls",
       "lit_refs_full", "doi_note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", default=[])
    ap.add_argument("--dois", nargs="*", default=[])
    a = ap.parse_args()
    lit = pd.read_csv(V0 / f"{STEM}_literature.csv").set_index("surface_id")
    for c in LIT:
        if c not in lit:
            lit[c] = ""
    lit = lit.astype({c: object for c in LIT})
    for f in a.matches:
        m = pd.read_csv(f).set_index("surface_id")
        for sid, r in m.iterrows():
            if sid not in lit.index:
                print("matches: not in product, skipped:", sid); continue
            for c in m.columns:
                if c in LIT and pd.notna(r[c]):
                    lit.at[sid, c] = r[c]
    for f in a.dois:
        d = pd.read_csv(f).set_index("surface_id")
        for sid, r in d.iterrows():
            if sid not in lit.index:
                print("dois: not in product, skipped:", sid); continue
            for c in ("lit_dois", "lit_urls", "lit_refs_full", "doi_note"):
                if c in d.columns and pd.notna(r[c]) and str(r[c]).strip():
                    lit.at[sid, c] = r[c]
    lit["lit_new_candidate"] = lit.lit_new_candidate.map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
    lit = lit.fillna("")
    lit.reset_index().to_csv(V0 / f"{STEM}_literature.csv", index=False)
    no = [s for s, r in lit.iterrows() if not str(r.lit_dois).strip()]
    print(f"{len(lit)} surfaces; {len(lit) - len(no)} with >= 1 DOI; without: {no}")
    keep = ["lit_match", "lit_confidence", "lit_class", "lit_new_candidate", "lit_refs", "lit_dois", "lit_urls"]
    s = pd.read_csv(V0 / f"{STEM}_summary.csv")
    s = s.drop(columns=[c for c in keep if c in s]).merge(lit[keep].reset_index(), on="surface_id", how="left")
    s.to_csv(V0 / f"{STEM}_summary.csv", index=False)
    g = json.load(open(V0 / f"{STEM}.geojson"))
    for f in g["features"]:
        r = lit.loc[f["properties"]["surface_id"]]
        f["properties"].update({c: (bool(r[c]) if c == "lit_new_candidate" else str(r[c])) for c in keep})
    (V0 / f"{STEM}.geojson").write_text(json.dumps(g))


if __name__ == "__main__":
    main()
