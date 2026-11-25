"""D-B078 comparison report: every leg and variant scored by b5_cmp_common.score_set on every set, in one pass.

Main score = held-out waveform-CC differential times (3 station folds) in the v5.1 NonLinLoc grids, each pair's median
training residual removed; the leg's `_filled` score (events a code leaves unrelocated kept at their v5.1 position, as
production would) is the operational one, `v5.1` the reference, `v5.1_same` the reference on the pairs the leg
relocated. Full runs: events above the solid surface (D-B055), cluster thickness, median shifts.

  python b5_cmp_report.py   ->  <cmp>/report.json, data/tomo_v1/b5/cmp/report.md (repository)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cmp_common as C  # noqa: E402
from manifest import REPO, write_manifest  # noqa: E402

LEGS = ["tomodd", "tomodd_d300_o0", "tomodd_d100_o4", "tomodd_d300_o4", "tomodd_snll", "tomodd_d300_o0_snll", "growclust", "nllbase", "nllssst", "nllcoh"]


def main():
    R = {}
    for leg in LEGS:
        if not (C.CMP / leg).exists():
            continue
        for s in C.SETS:
            if (C.CMP / leg / s / "full" / "reloc.csv").exists():
                R.setdefault(leg, {})[s] = C.score_set(leg, s)
                if s == "axial":     # sensitivity: rows straddling unconfirmed candidate clock epochs (Axial array)
                    R[leg]["axial_no_candidates"] = C.score_set(leg, s, drop_candidates=True)
                print(leg, s, flush=True)
    (C.CMP / "report.json").write_text(json.dumps(R, indent=1))
    L = ["# D-B078: B5 code comparison on waveform cross-correlation data", "",
         "Held-out CC dt in the v5.1 NonLinLoc grids (3 station folds, each pair's training median removed): MAD / rms "
         "(ms), P and S. `filled` = the code's events, with events it left unrelocated kept at v5.1 (the operational "
         "score). Full runs: relocated / events, above the solid surface, cluster thickness (smallest principal axis, km; "
         "v5.1 in brackets).", "",
         "**Reading the v5.1 row:** v5.1 was located with every station, the held-out ones included, so its held-out "
         "score is in-sample and flattered. The fair NonLinLoc baseline is `nllbase` (NonLinLoc with the v5.1 settings, "
         "relocated without the held-out stations, like every other code). `axial_no_candidates`: axial scored without "
         "the CC rows that straddle unconfirmed candidate clock epochs of the Axial array (D-B090 screen).", ""]
    for s in list(C.SETS) + ["axial_no_candidates"]:
        L += [f"## {s}", "", "| code | P MAD | P rms | S MAD | S rms | relocated | above surface | thickness km |",
              "|---|---|---|---|---|---|---|---|"]
        ref = next((R[l][s]["v5.1"] for l in R if s in R[l]), None)
        if ref:
            L.append(f"| v5.1 (NLLoc) | {ref['P']['mad_ms']} | {ref['P']['rms_ms']} | {ref['S']['mad_ms']} | "
                     f"{ref['S']['rms_ms']} | - | 0 | - |")
        for leg in R:
            if s not in R[leg]:
                continue
            x = R[leg][s]
            f = x.get(f"{leg}_filled", {})
            full = x.get("full", {})
            if not f:
                continue
            L.append(f"| {leg} | {f['P']['mad_ms']} | {f['P']['rms_ms']} | {f['S']['mad_ms']} | {f['S']['rms_ms']} | "
                     f"{full.get('relocated')}/{full.get('of')} | {full.get('above_surface')} | "
                     f"{full.get('thickness_km')} ({full.get('thickness_v51_km')}) |")
        L.append("")
    od = REPO / "data" / "tomo_v1" / "b5" / "cmp"
    od.mkdir(parents=True, exist_ok=True)
    (od / "report.md").write_text("\n".join(L) + "\n")
    (od / "report.json").write_text(json.dumps(R, indent=1))
    write_manifest("b5_cmp_report", od, outputs=[od / "report.md", od / "report.json"], params={"legs": LEGS},
                   counts={leg: list(R[leg]) for leg in R})
    print("\n".join(L))


if __name__ == "__main__":
    main()
