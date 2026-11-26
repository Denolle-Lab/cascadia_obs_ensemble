"""Track B, catalog v6.0: acceptance report of the v6 relocation (HTML and Markdown).

Sections: located counts and why the others are not located; events above the solid surface (hard requirement: 0, D-B055;
any event above is a computation error to fix at the root, never clamped or moved: the script exits 1 and the chain stops);
floor-limited events by region; tiers (v6.0 against v5.1); shifts from v5.1 by region; event-paired depths against the
published local catalogs (b4_lit_paired.py: Hirao et al. 2025 at Mount St Helens, Lange et al. 2026 at Blanco; v5.1 in
the same table); the Axial share of tier A-B events within 2 km of the seafloor (v5.1 81 %, target 85 %, Waldhauser et al.
2020: 90 %); unweighted residual rms by region; the held-out effect of the coherence stage by region (b4_v6_heldout.py,
the termtest protocol on nine clusters); pass statistics (NLLoc passes, Loc2ssst, coherence, deep and edge passes).

  python b4_v6_acceptance.py CATALOG.csv [--paired data/tomo_v1/depth_literature/paired_v6.0.md] [--heldout DIR]
      [--outdir data/tomo_v1/b4/v6.0] [--smoke]
"""
import argparse
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO  # noqa: E402

V51 = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
V51_PAIRED = Path("/wd1/mdenolle_data/catalog_v5/v5.1/paired_v5.1.md")
REGION_ORDER = ["Axial", "Endeavour", "Explorer", "JdF ridge/plate", "Blanco", "Gorda", "Mendocino", "Offshore forearc",
                "Puget/Georgia", "Cascades", "Other onshore"]


def table(df, fmt=None, index=True):
    """DataFrame -> (html, markdown)."""
    d = df.reset_index() if index else df.copy()
    fmt = fmt or {}
    def f(c, v):
        if pd.isna(v):
            return ""
        if c in fmt:
            return fmt[c].format(v)
        if isinstance(v, (float, np.floating)):
            return f"{v:,.2f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    rows = [[f(c, v) for c, v in zip(d.columns, r)] for r in d.itertuples(index=False)]
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns) + "</tr></thead><tbody>"
    h += "".join("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in r) + "</tr>" for r in rows) + "</tbody></table>"
    m = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    m += "".join("| " + " | ".join(r) + " |\n" for r in rows)
    return h, m


def md_tables_to_html(md):
    """the pipe tables and headings of a Markdown file -> HTML (b4_lit_paired.py output)."""
    out, rows = [], []
    def flush():
        if rows:
            hdr, body = rows[0], [r for r in rows[2:]]
            out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in hdr) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table>")
            rows.clear()
    for line in md.splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        flush()
        if line.startswith("## "):
            out.append(f"<h3>{html.escape(line[3:])}</h3>")
        elif line.startswith("# "):
            continue
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    flush()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("catalog")
    ap.add_argument("--paired")
    ap.add_argument("--heldout")
    ap.add_argument("--outdir", default=str(REPO / "data/tomo_v1/b4/v6.0"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cat = Path(a.catalog)
    man = json.loads(cat.with_suffix(".manifest.json").read_text())
    c = pd.read_csv(cat, dtype={"tier": str}, low_memory=False)
    v = pd.read_csv(V51, dtype={"tier": str}, low_memory=False)
    if a.smoke:                    # only the events of the smoke tiles
        c = c[c.loc_source.eq("nll3d") | ~c.u_reason.fillna("").eq("not in the run's observation files")]
        v = v[v.ID.isin(c.ID)]
    nll = c.loc_source.eq("nll3d")
    vn = v.loc_source.eq("nll3d")
    B = lambda s: s.astype(str).str.lower().eq("true")
    H, M = [], []

    def sec(title, intro=""):
        H.append(f"<h2>{html.escape(title)}</h2>" + (f"<p>{html.escape(intro)}</p>" if intro else ""))
        M.append(f"\n## {title}\n\n" + (f"{intro}\n\n" if intro else ""))

    def add(df, **k):
        h, m = table(df, **k)
        H.append(h)
        M.append(m + "\n")

    def para(t):
        H.append(f"<p>{html.escape(t)}</p>")
        M.append(t + "\n\n")

    # 1. located
    above = int(B(c.above_surface).sum())
    sec("Located events")
    lp = c.loc_pass[nll].value_counts()
    t = pd.DataFrame({"events": [len(c), len(v)], "located": [int(nll.sum()), int(vn.sum())],
                      "not located": [int((~nll).sum()), int((~vn).sum())]}, index=pd.Index(["v6.0", "v5.1"], name="catalog"))
    add(t)
    add(pd.DataFrame({"events": lp}).rename_axis("loc_pass"))
    ur = c.u_reason[~nll].fillna("").str.replace(r"rejected: .*boundary (\d+).*", r"rejected: max likelihood on box face \1",
                                                  regex=True).value_counts().rename("events").to_frame().rename_axis("u_reason")
    if len(ur):
        add(ur)
    # 2. surface
    sec("Events above the solid surface (D-B055)",
        "Hard requirement: 0. An event above the ground or seafloor is a computation error to fix at its root; no depth is "
        "clamped or moved.")
    para(f"Above the surface: {above} of {int(nll.sum()):,} located events "
         f"({'PASS' if above == 0 else 'FAIL: computation error, chain stopped'}). Located events above the NLLoc floor "
         f"file of their run: {man['counts'].get('above_floor_file')} (expected 0). Surface: {man.get('surface_source')}.")
    # 3. floor-limited by region
    sec("Floor-limited events by region", "floor_limited: maximum likelihood within 1.5 final oct-tree cells of the NLLoc "
        "floor (depth set by the floor, not the data).")
    g6 = c[nll].groupby("region").agg(located=("ID", "size"), floor_limited=("floor_limited", lambda s: int(B(s).sum())))
    g5 = v[vn].groupby("region").agg(located_v51=("ID", "size"), floor_limited_v51=("floor_limited", lambda s: int(B(s).sum())))
    g = g6.join(g5, how="outer").reindex([r for r in REGION_ORDER if r in set(g6.index) | set(g5.index)])
    g.loc["all"] = g.sum()
    g["share_v6_%"] = 100 * g.floor_limited / g.located
    g["share_v51_%"] = 100 * g.floor_limited_v51 / g.located_v51
    add(g.fillna(0), fmt={"share_v6_%": "{:.1f}", "share_v51_%": "{:.1f}", "located": "{:,.0f}", "floor_limited": "{:,.0f}",
                         "located_v51": "{:,.0f}", "floor_limited_v51": "{:,.0f}"})
    # 4. tiers
    sec("Location-quality tiers", "Tier rules of a2_build_catalog.py (D tier also for every 5-phase event, D-B066).")
    tt = pd.DataFrame({"v6.0": c.tier.value_counts(), "v5.1": v.tier.value_counts()}).fillna(0).astype(int).sort_index()
    add(tt.rename_axis("tier"))
    ct = pd.crosstab(c.region, c.tier).reindex([r for r in REGION_ORDER if r in set(c.region)])
    add(ct.rename_axis("region (v6.0)"))
    # 5. shifts vs v5.1
    sec("Change from catalog v5.1 by region", "Events located in both; dz positive = v6.0 deeper.")
    b = c[c.shift_h_v51_km.notna()]
    sh = b.groupby("region").agg(n=("ID", "size"), dh_med=("shift_h_v51_km", "median"),
                                 dh_p90=("shift_h_v51_km", lambda s: s.quantile(.9)),
                                 dz_med=("shift_z_v51_km", "median"),
                                 absdz_med=("shift_z_v51_km", lambda s: s.abs().median()),
                                 absdz_p90=("shift_z_v51_km", lambda s: s.abs().quantile(.9)))
    sh = sh.reindex([r for r in REGION_ORDER if r in sh.index])
    sh.loc["all"] = [len(b), b.shift_h_v51_km.median(), b.shift_h_v51_km.quantile(.9), b.shift_z_v51_km.median(),
                     b.shift_z_v51_km.abs().median(), b.shift_z_v51_km.abs().quantile(.9)]
    add(sh, fmt={"n": "{:,.0f}"})
    sh.to_csv(Path(a.outdir) / "shift_v51_by_region.csv") if Path(a.outdir).exists() else None
    # 6. literature
    sec("Depths against published local catalogs (event-paired)",
        "b4_lit_paired.py: catalog minus published depth (km, positive = ours deeper), median [p10, p90] and n; "
        f"v5.1 in the same table (v5.1 report: {V51_PAIRED}).")
    if a.paired and Path(a.paired).exists():
        txt = Path(a.paired).read_text()
        H.append(md_tables_to_html(txt))
        M.append(txt.split("\n", 1)[1] if txt.startswith("#") else txt)
    else:
        para("paired comparison not available (smoke test, or b4_lit_paired.py not run)")
    # 7. Axial
    sec("Axial: share within 2 km of the seafloor",
        "Tier A-B Axial events, depth below the seafloor = depth + surface_km. v5.1: 81 %; target 85 %; Waldhauser et al. "
        "(2020): 85 % within 1.6 km, 90 % within 2.0 km.")
    rows = []
    for lab, cc, nn in (("v6.0", c, nll), ("v5.1", v, vn)):
        for tl, tiers in (("A-B", ["A", "B"]), ("A-D", list("ABCD"))):
            s = cc[nn & cc.region.eq("Axial") & cc.tier.isin(tiers)]
            z = s.depth + s.surface_km
            rows.append(dict(catalog=lab, tiers=tl, n=len(s), median_bsf_km=z.median(),
                             within_1_6_pct=100 * (z <= 1.6).mean() if len(s) else np.nan,
                             within_2_0_pct=100 * (z <= 2.0).mean() if len(s) else np.nan))
    ax = pd.DataFrame(rows)
    add(ax, index=False, fmt={"n": "{:,.0f}", "within_1_6_pct": "{:.1f}", "within_2_0_pct": "{:.1f}"})
    ab = ax[(ax.catalog == "v6.0") & (ax.tiers == "A-B")].within_2_0_pct.iloc[0]
    para(f"v6.0 tier A-B within 2 km: {ab:.1f} % (target 85 %: {'met' if ab >= 85 else 'not met'}).")
    # 8. rms
    sec("Residual rms by region", "rms_unweighted: unweighted rms of the residuals of the used arrivals (s); rms: NLLoc's "
        "EDT-weighted RMS (D-B069), also for v5.1.")
    r6 = c[nll].groupby("region").agg(n=("ID", "size"), rms_unweighted_med=("rms_unweighted", "median"),
                                      rms_unweighted_p90=("rms_unweighted", lambda s: s.quantile(.9)),
                                      rms_med=("rms", "median"))
    r5 = v[vn].groupby("region").agg(rms_v51_med=("rms", "median"))
    rr = r6.join(r5).reindex([r for r in REGION_ORDER if r in r6.index])
    rr.loc["all"] = [int(nll.sum()), c.rms_unweighted[nll].median(), c.rms_unweighted[nll].quantile(.9),
                     c.rms[nll].median(), v.rms[vn].median()]
    add(rr, fmt={"n": "{:,.0f}", "rms_unweighted_med": "{:.3f}", "rms_unweighted_p90": "{:.3f}", "rms_med": "{:.3f}",
                 "rms_v51_med": "{:.3f}"})
    # 9. coherence held out
    sec("Held-out effect of the coherence stage by region",
        "b4_v6_heldout.py: the v6.0 recipe (v0.4.1 grids, SSST to 4 km, then coherence) on nine clusters with waveform CC "
        "(termtest protocol: 3 station folds, held-out CC differential times, pair-demeaned); MAD per phase pooled over the "
        "folds on the rows located in both arms; coherence - ssst with a 95 % paired bootstrap interval over event pairs.")
    cr = c[nll].groupby("region").agg(located=("ID", "size"),
                                      coherence_relocated=("coherence_relocated", lambda s: int(B(s).sum())),
                                      with_partner=("coherence_partners", lambda s: int((s.fillna(0) > 0).sum())),
                                      **({"skipped_region": ("coherence_skipped_region", lambda s: int(B(s).sum()))}
                                         if "coherence_skipped_region" in c else {}))
    add(cr.reindex([r for r in REGION_ORDER if r in cr.index]))
    para("D-B098 (PI 2026-11-25 20:35): no coherence relocation in the Mendocino region (v5.1 region label), where the "
         "held-out test on the 2015 sequence found the stage worse (P +8 ms, S +20 ms); those compact events keep their SSST "
         "4 km location (skipped_region) and can still be stacking partners elsewhere. Provisional: one Mendocino test cluster.")
    rc = Path(a.heldout) / "coh_rule_check.json" if a.heldout else None
    para("Partner rule of v6.0 (R3, coordinator 2026-11-25 20:00, notes/v6_plan.md): separation <= 3 km, >= 3 accepted "
         "CC rows, median cc_max >= 0.5 over the measured rows, rows straddling a confirmed clock epoch left out; the "
         "test sets' rule (median cc_max >= 0.5 over >= 3 measured rows) passes 55 % of all-family pairs 10-20 km apart.")
    if rc and rc.exists():
        ck = json.loads(rc.read_text())
        para(f"Rule check before the coherence stage (puget, mtj; decision rule: {ck.get('decision_rule')}): "
             f"use {ck.get('use')}.")
        add(pd.DataFrame(ck["rows"])[["set", "phase", "pairs", "mad_ssst_ms", "mad_test_ms", "mad_r3_ms",
                                      "r3_minus_test_ms", "lo_ms", "hi_ms", "r3_worse"]], index=False,
            fmt={"mad_ssst_ms": "{:.1f}", "mad_test_ms": "{:.1f}", "mad_r3_ms": "{:.1f}", "r3_minus_test_ms": "{:+.1f}",
                 "lo_ms": "{:+.1f}", "hi_ms": "{:+.1f}", "pairs": "{:,.0f}"})
    hf = Path(a.heldout) / "heldout.csv" if a.heldout else None
    if hf and hf.exists():
        h = pd.read_csv(hf)
        cols = ["set", "region", "phase", "pairs", "mad_ssst_ms", "mad_coh_ms", "diff_ms", "effect"]
        if "mad_coh_r3_ms" in h:
            cols += ["mad_coh_r3_ms", "diff_r3_ms", "lo_r3_ms", "hi_r3_ms", "effect_r3"]
        para("Held-out MAD (ms): ssst = SSST to 4 km alone; coh = coherence with the test rule; coh_r3 = coherence with "
             "R3 (the v6.0 rule); diff = coherence - ssst with its 95 % interval (effect: helps / hurts / no clear effect).")
        add(h[cols], index=False, fmt={k: ("{:+.1f}" if k.startswith(("diff", "lo", "hi")) else "{:.1f}")
                                       for k in cols if k.endswith("_ms")} | {"pairs": "{:,.0f}"})
        js = json.loads((Path(a.heldout) / "scores.json").read_text())
        if js.get("missing"):
            para(f"Clusters not scored (relocation not finished or failed): {', '.join(js['missing'])}.")
    else:
        para("held-out scores not available" + (" (smoke test)" if a.smoke else ""))
    # 10. passes
    sec("Passes")
    runs = man.get("runs", {})
    pr = pd.DataFrame([{"run": k, **{kk: (json.dumps(vv) if isinstance(vv, dict) else vv) for kk, vv in r.items()
                                       if kk in ("events", "status", "rejected_lateral", "rejected_bottom", "located_deep",
                                                 "wall_min", "compact", "compact_with_compact_partner", "pairs_used",
                                                 "selected", "straddling_rows_dropped", "n_failed")}}
                       for k, r in runs.items()])
    if len(pr):
        add(pr, index=False)
    ls = pd.DataFrame([{"pass": k, **{kk: r.get(kk) for kk in ("char_dist_km", "tiles", "instances", "events_accepted_sum",
                                                                 "corrected_grids", "kept_previous", "negative_nodes",
                                                                 "tiles_with_negative", "t_min", "tiles_partial_input",
                                                                 "events_dropped_arrival_limit", "wall_min")}}
                       for k, r in man.get("loc2ssst", {}).items()])
    if len(ls):
        para("Loc2ssst: corrected grids per pass; kept_previous = station-phases without accepted arrivals (previous grid "
             "kept); negative_nodes = corrected time < 0 near a station (NLLoc does not use a negative predicted time); "
             "events_dropped_arrival_limit = events left out of a tile's input by the 20,000 arrivals per station-phase "
             "limit (nearest kept).")
        add(ls, index=False)
    k = man["counts"]
    para(f"Flags: {json.dumps(k.get('flags'))}. Review indicators carried on {k.get('review_rows'):,} events "
         f"(reviewer columns empty, D-B097). Magnitudes: {json.dumps(k.get('magnitudes', {}).get('v51'))} from the v5.1 "
         f"converged table, {json.dumps(k.get('magnitudes', {}).get('v41'))} from the v4.1 converged table (D-B091).")
    # write
    head = (f"Catalog {cat} (md5 {man.get('catalog_md5')}), model {man.get('model')}, grid manifest md5 "
            f"{man.get('grid_manifest_md5')}, floor {man.get('floor')} (md5 {man.get('floor_md5')}), picks {man.get('picks')}. "
            f"Built {time.strftime('%F %T')}.")
    title = "Catalog v6.0 acceptance" + (" (smoke test)" if a.smoke else "")
    css = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1d1d1f;"
           "background:#fff}table{border-collapse:collapse;margin:8px 0 18px;font-size:13px}th,td{border:1px solid #ccc;"
           "padding:3px 7px;text-align:right}th{background:#f2f2f2}td:first-child,th:first-child{text-align:left}"
           "h2{margin-top:30px;border-bottom:1px solid #ddd}p{line-height:1.45}"
           "@media (prefers-color-scheme: dark){body{background:#151515;color:#e8e8e8}th{background:#2a2a2a}"
           "th,td{border-color:#444}}")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
           f"<title>{title}</title><style>{css}</style></head><body><h1>{title}</h1><p>{html.escape(head)}</p>"
           + "\n".join(H) + "</body></html>")
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "acceptance.html").write_text(doc)
    (od / "acceptance.md").write_text(f"# {title}\n\n{head}\n" + "".join(M))
    print(f"acceptance -> {od}/acceptance.html; above surface {above}; Axial A-B within 2 km {ab:.1f} %")
    sys.exit(1 if above else 0)


if __name__ == "__main__":
    main()
