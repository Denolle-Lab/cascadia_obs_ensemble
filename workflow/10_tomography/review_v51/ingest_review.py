"""Ingest returned review CSVs of the v5.1 location review (review_v51_events.csv filled in the viewer, or the CSV of
a report page).

  python ingest_review.py returned1.csv [returned2.csv ...] [--events review_v51_events.csv] [--out DIR]
  python ingest_review.py --selftest            # synthetic filled CSVs, checks the outputs

Validation
  * every row must name a known (ID, problem) of review_v51_events.csv (report-page CSVs: event_id, problem);
  * allowed values: is_earthquake yes / no / unsure; depth_verdict plausible / implausible / cannot_tell; action keep /
    flag_not_earthquake / flag_depth_unreliable / relocate / other; corrected_depth_km a number in -5..700 or empty;
  * a reviewed row (any answer filled) without a reviewer name is kept and named "unnamed:<file>";
  * conflict: the same ID answered differently (is_earthquake, depth_verdict or action) by two reviewers, in any problem.
Outputs (DIR, default /wd1/mdenolle_data/catalog_v5/v5.1_review/returned/)
  review_answers_merged.csv     every valid reviewed row (one per reviewer, ID, problem; latest reviewed_utc kept)
  review_flag_not_earthquake.csv  rule: any reviewer's action = flag_not_earthquake (conflicts column says if others disagree)
  review_depth_unreliable.csv   rule: any reviewer's action = flag_depth_unreliable, or depth_verdict = implausible
  review_relocate.csv           rule: any reviewer's action = relocate (with the corrected depths given)
  review_conflicts.csv, review_invalid_rows.csv, review_summary.md
Nothing in the catalog is changed: these lists are inputs for the next catalog version.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = Path("/wd1/mdenolle_data/catalog_v5/v5.1_review/review_v51_events.csv")
OUTDIR = Path("/wd1/mdenolle_data/catalog_v5/v5.1_review/returned")
ALLOWED = {"is_earthquake": {"yes", "no", "unsure"}, "depth_verdict": {"plausible", "implausible", "cannot_tell"},
           "action": {"keep", "flag_not_earthquake", "flag_depth_unreliable", "relocate", "other"}}
ANSWER = ["is_earthquake", "depth_verdict", "corrected_depth_km", "action", "comment"]
REVIEW = ["reviewer"] + ANSWER + ["reviewed_utc"]


def read_returned(f):
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    if "is_earthquake" not in d.columns and {"event_id", "earthquake"} <= set(d.columns):   # report-page CSV
        eq = {"earthquake": "yes", "not_earthquake": "no", "unsure": "unsure"}
        d = pd.DataFrame({"ID": d.event_id, "problem": d.problem, "reviewer": d.get("reviewer", ""),
                          "is_earthquake": d.earthquake.map(lambda v: eq.get(v, v)), "depth_verdict": d.v51_depth,
                          "corrected_depth_km": "", "action": "", "comment": d.get("comment", ""),
                          "reviewed_utc": d.get("answered_at_utc", "")})
    missing = [c for c in ["ID", "problem"] + REVIEW if c not in d.columns]
    if missing:
        raise SystemExit(f"{f}: missing columns {missing}")
    d = d[["ID", "problem"] + REVIEW].copy()
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()
    d["file"] = Path(f).name
    return d


def validate(d, known):
    bad = []
    rev = d[(d[ANSWER] != "").any(axis=1)].copy()
    for i, r in rev.iterrows():
        why = []
        if (r.ID, r.problem) not in known:
            why.append("unknown ID/problem")
        for k, allowed in ALLOWED.items():
            if r[k] and r[k] not in allowed:
                why.append(f"{k}={r[k]!r} not allowed")
        if r.corrected_depth_km:
            try:
                z = float(r.corrected_depth_km)
                if not -5 <= z <= 700:
                    why.append(f"corrected_depth_km {z} outside -5..700")
            except ValueError:
                why.append(f"corrected_depth_km={r.corrected_depth_km!r} not a number")
        if why:
            bad.append(dict(file=r.file, ID=r.ID, problem=r.problem, reviewer=r.reviewer, problems="; ".join(why)))
    badkeys = {(b["file"], b["ID"], b["problem"], b["reviewer"]) for b in bad}
    rev = rev[[(r.file, r.ID, r.problem, r.reviewer) not in badkeys for r in rev.itertuples()]]
    rev.loc[rev.reviewer == "", "reviewer"] = "unnamed:" + rev.loc[rev.reviewer == "", "file"]
    rev = rev.sort_values("reviewed_utc").drop_duplicates(["reviewer", "ID", "problem"], keep="last")
    return rev, pd.DataFrame(bad, columns=["file", "ID", "problem", "reviewer", "problems"])


def conflicts(rev):
    out = []
    for i, g in rev.groupby("ID"):
        if g.reviewer.nunique() < 2:
            continue
        for k in ("is_earthquake", "depth_verdict", "action"):
            vals = {v for v in g[k] if v}
            if len(vals) > 1:
                out.append(dict(ID=i, field=k, answers="; ".join(f"{r.reviewer} ({r.problem}): {r[k] or '-'}" for _, r in g.iterrows())))
    return pd.DataFrame(out, columns=["ID", "field", "answers"])


def lists(rev, ev, conf):
    cf = set(conf.ID)
    info = ev.drop_duplicates("ID").set_index("ID")[["time_utc", "lat", "lon", "depth_v51", "depth_v4", "tier", "region",
                                                     "magnitude", "auto_indicator_rule_not_verdict"]]

    def agg(sel, rule):
        x = rev[sel]
        if not len(x):
            return pd.DataFrame(columns=["ID", "rule", "reviewers", "problems", "corrected_depth_km", "comments",
                                         "conflict"] + list(info.columns))
        g = x.groupby("ID").agg(reviewers=("reviewer", lambda s: ";".join(sorted(set(s)))),
                                problems=("problem", lambda s: ";".join(sorted(set(s)))),
                                corrected_depth_km=("corrected_depth_km", lambda s: ";".join(v for v in s if v)),
                                comments=("comment", lambda s: " | ".join(v for v in s if v))).reset_index()
        g.insert(1, "rule", rule)
        g["conflict"] = np.where(g.ID.isin(cf), "yes: see review_conflicts.csv", "no")
        return g.merge(info, left_on="ID", right_index=True, how="left")
    flag = agg(rev.action.eq("flag_not_earthquake"), "reviewer action = flag_not_earthquake")
    depth = agg(rev.action.eq("flag_depth_unreliable") | rev.depth_verdict.eq("implausible"),
                "reviewer action = flag_depth_unreliable, or depth_verdict = implausible")
    reloc = agg(rev.action.eq("relocate"), "reviewer action = relocate")
    return flag, depth, reloc


def run(files, events=EVENTS, out=OUTDIR):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(events, dtype=str, keep_default_na=False)
    known = set(zip(ev.ID, ev.problem))
    d = pd.concat([read_returned(f) for f in files], ignore_index=True)
    rev, bad = validate(d, known)
    conf = conflicts(rev)
    flag, depth, reloc = lists(rev, ev, conf)
    rev.to_csv(out / "review_answers_merged.csv", index=False)
    flag.to_csv(out / "review_flag_not_earthquake.csv", index=False)
    depth.to_csv(out / "review_depth_unreliable.csv", index=False)
    reloc.to_csv(out / "review_relocate.csv", index=False)
    conf.to_csv(out / "review_conflicts.csv", index=False)
    bad.to_csv(out / "review_invalid_rows.csv", index=False)
    s = [f"# v5.1 location review: returned answers\n", f"Files: {', '.join(Path(f).name for f in files)}\n",
         f"- rows read: {len(d)}; reviewed rows (any answer): {int((d[ANSWER] != '').any(axis=1).sum())}; valid: {len(rev)}; "
         f"invalid: {len(bad)}",
         f"- reviewers: {', '.join(f'{k} {v}' for k, v in rev.reviewer.value_counts().items())}",
         f"- events reviewed: {rev.ID.nunique()} of {ev.ID.nunique()}; by problem: "
         + ", ".join(f"{k} {v}" for k, v in rev.groupby('problem').ID.nunique().items()),
         f"- conflicts (same ID answered differently by two reviewers): {conf.ID.nunique()} events, {len(conf)} fields",
         f"- flag as not an earthquake (rule: reviewer action = flag_not_earthquake): {len(flag)} events",
         f"- depth unreliable (rule: action = flag_depth_unreliable or depth_verdict = implausible): {len(depth)} events",
         f"- relocate (rule: action = relocate): {len(reloc)} events",
         "", "| problem | is_earthquake yes / no / unsure | depth plausible / implausible / cannot_tell | actions |", "|---|---|---|---|"]
    for p, g in rev.groupby("problem"):
        e = g.is_earthquake.value_counts()
        z = g.depth_verdict.value_counts()
        a = g.action.value_counts()
        s.append(f"| {p} | {e.get('yes', 0)} / {e.get('no', 0)} / {e.get('unsure', 0)} | {z.get('plausible', 0)} / "
                 f"{z.get('implausible', 0)} / {z.get('cannot_tell', 0)} | " + ", ".join(f"{k} {v}" for k, v in a.items() if k) + " |")
    (out / "review_summary.md").write_text("\n".join(s) + "\n")
    print("\n".join(s))
    return dict(rev=rev, bad=bad, conf=conf, flag=flag, depth=depth, reloc=reloc)


def selftest():
    """Two synthetic reviewers + one report-page CSV, with known conflicts and invalid rows; checks the outputs."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="ingest_selftest_", dir=str(OUTDIR.parent / "work")))
    ev = pd.read_csv(EVENTS, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(0)
    a = ev.copy()
    pick = a.index[a.problem.eq("P1")][:40]
    a.loc[pick, "reviewer"] = "Alice"
    a.loc[pick, "is_earthquake"] = "no"
    a.loc[pick, "depth_verdict"] = "implausible"
    a.loc[pick, "action"] = "flag_not_earthquake"
    a.loc[pick, "reviewed_utc"] = "2026-11-26T10:00:00Z"
    p2 = a.index[a.problem.eq("P2")][:10]
    a.loc[p2, ["reviewer", "is_earthquake", "depth_verdict", "action", "corrected_depth_km", "reviewed_utc"]] = \
        ["Alice", "yes", "implausible", "relocate", "15", "2026-11-26T10:05:00Z"]
    a.loc[a.index[a.problem.eq("P4")][0], ["reviewer", "is_earthquake", "action"]] = ["Alice", "maybe", "keep"]       # invalid
    a.loc[a.index[a.problem.eq("P4")][1], ["reviewer", "is_earthquake", "corrected_depth_km"]] = ["Alice", "yes", "deep"]  # invalid
    b = ev.copy()
    b.loc[pick[:20], ["reviewer", "is_earthquake", "depth_verdict", "action", "reviewed_utc"]] = \
        ["Bob", "no", "implausible", "flag_not_earthquake", "2026-11-26T11:00:00Z"]                   # agree with Alice
    b.loc[pick[20:25], ["reviewer", "is_earthquake", "depth_verdict", "action", "reviewed_utc"]] = \
        ["Bob", "yes", "plausible", "keep", "2026-11-26T11:00:00Z"]                                   # 5 conflicts
    b.loc[b.index[b.problem.eq("P3")][:6], ["reviewer", "is_earthquake", "depth_verdict", "action", "reviewed_utc"]] = \
        ["Bob", "yes", "implausible", "flag_depth_unreliable", "2026-11-26T11:00:00Z"]
    page = pd.DataFrame({"report": "P1_msh_deep", "problem": "P1", "event_id": ev.ID[pick[25:28]].values, "view_group": "x",
                         "origin_time_utc": "", "v51_depth_km": "", "reviewer": "Carol", "earthquake": "not_earthquake",
                         "v51_depth": "implausible", "comment": "step on analog channels", "answered_at_utc": "2026-11-26T12:00:00Z",
                         "exported_at_utc": "2026-11-26T12:01:00Z"})
    page.loc[len(page)] = ["P1_msh_deep", "P1", "99999999", "x", "", "", "Carol", "earthquake", "plausible", "", "", ""]  # unknown ID
    fa, fb, fc = tmp / "alice.csv", tmp / "bob.csv", tmp / "carol_page.csv"
    a.to_csv(fa, index=False)
    b.to_csv(fb, index=False)
    page.to_csv(fc, index=False)
    r = run([fa, fb, fc], out=tmp / "out")
    checks = [("valid reviewed rows = 40 + 10 + 20 + 5 + 6 + 3", len(r["rev"]) == 84),
              ("invalid rows = 3 (two Alice P4, one unknown ID)", len(r["bad"]) == 3),
              ("conflicting events = 5", r["conf"].ID.nunique() == 5),
              ("flag_not_earthquake events = 40", len(r["flag"]) == 40),
              ("flag list marks the 5 conflicts", int((r["flag"].conflict != "no").sum()) == 5),
              ("depth unreliable = 40 (implausible) + 10 (P2 implausible) + 6 (P3) = 56", len(r["depth"]) == 56),
              ("relocate = 10 with corrected depth 15", len(r["reloc"]) == 10 and set(r["reloc"].corrected_depth_km) == {"15"}),
              ("Carol's report-page rows mapped (3 valid)", int((r["rev"].reviewer == "Carol").sum()) == 3)]
    ok = True
    for what, c in checks:
        print(("PASS " if c else "FAIL ") + what)
        ok &= bool(c)
    print("selftest outputs:", tmp / "out")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--events", default=str(EVENTS))
    ap.add_argument("--out", default=str(OUTDIR))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.files:
        ap.error("give one or more returned CSV files (or --selftest)")
    run(a.files, Path(a.events), Path(a.out))
