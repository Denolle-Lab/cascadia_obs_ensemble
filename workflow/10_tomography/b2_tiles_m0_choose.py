"""Track B, D-B046: smoothing of each tile restarted from M0, by the rule of P-M11.

For each tile <t>_m0, the damping-200 smoothing scan (params tomodd.merge.tile_smooth_scan) is read from
<runs>/<t>_m0/summary_runs.csv (b2_tomodd_summary.py). Chosen smoothing = the largest whose weighted RMS is within
merge.smooth_rule_tol of the scan's best (the M0 rule). Writes data/tomo_v1/b2_merge/tiles_m0/choice.csv and
summary.md (whole scan per tile, chosen row marked), and compares with the tile's earlier damping-200 smoothing-20
run on the D-B005 start (<t>_v2 d200_s20 or d1000_s20 when no d200 run exists).

  python b2_tiles_m0_choose.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402

T = load_params()["tomodd"]
M = T["merge"]
RUNS = Path(T["runs"])
OUT = REPO / "data" / "tomo_v1" / "b2_merge" / "tiles_m0"
TILES = ["mendocino_gorda_c", "blanco", "oregon", "washington", "ridges_north"]
COLS = ["tag", "smooth_x", "rms_abs_final", "rms_wt_final", "cnd_first", "aq_repeats", "events", "rms_dlnvp_pct",
        "rms_dlnvs_pct", "max_abs_dlnvp_pct", "shift_h_med_km", "shift_z_abs_med_km"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tol = float(M["smooth_rule_tol"])
    want = {f"d200_s{w[0]}" for w in M["tile_smooth_scan"]}
    rows, L = [], ["# Tiles restarted from M0: smoothing scan at damping 200 (D-B046)", "",
                   f"Rule: chosen smoothing = the largest whose weighted RMS <= (1 + {tol}) x the best of the scan. "
                   "Previous run = the tile on the D-B005 start (v2, per-tile weight scale).", ""]
    for t in TILES:
        f = RUNS / f"{t}_m0" / "summary_runs.csv"
        if not f.exists():
            L += [f"## {t}: no scan summary", ""]
            continue
        s = pd.read_csv(f)
        s = s[s.tag.isin(want)].sort_values("smooth_x")
        if s.empty:
            continue
        best = s.rms_wt_final.min()
        ok = s[s.rms_wt_final <= (1 + tol) * best]
        pick = ok.sort_values("smooth_x").iloc[-1]
        rows.append(dict(tile=t, region=f"{t}_m0", tag=pick.tag, smooth=[int(pick.smooth_x), int(pick.smooth_x),
                                                                          int(pick.smooth_z)],
                         rms_wt=pick.rms_wt_final, rms_wt_best=best, n_scan=len(s)))
        prev = None
        for g in ("d200_s20", "d1000_s20"):
            pf = RUNS / f"{t}_v2" / "summary_runs.csv"
            if pf.exists():
                p = pd.read_csv(pf)
                if (p.tag == g).any():
                    prev = p[p.tag == g].iloc[0]
                    break
        L += [f"## {t}_m0 (chosen {pick.tag})", "", "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
        for r in s.itertuples(index=False):
            d = r._asdict()
            mark = " **chosen**" if d["tag"] == pick.tag else ""
            L.append("| " + " | ".join((f"{d[c]:.4g}" if isinstance(d[c], float) else str(d[c])) + (mark if c == "tag" else "")
                                       for c in COLS) + " |")
        if prev is not None:
            L.append("| " + " | ".join(f"{prev[c]:.4g}" if isinstance(prev[c], float) else f"v2 {prev[c]}" for c in COLS) + " |")
        L.append("")
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "choice.csv", index=False)
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    write_manifest("b2_tiles_m0_choose", OUT, inputs=[RUNS / f"{t}_m0" / "summary_runs.csv" for t in TILES],
                   outputs=[OUT / "choice.csv", OUT / "summary.md"], params={"merge": M})
    print(c.to_string(index=False))


if __name__ == "__main__":
    main()
