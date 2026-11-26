"""v0.4.0: would the B1 pick-QC rules (D-B009: 4-MAD residual cut, 5-MAD S-P cut; D-B003/D-B023 post-QC event rule)
choose differently with the re-timed picks? v0.4.0 keeps v0.3.1's choice (b1t_retime_dataset.py); this only reports.

1. Rebuild the B1 candidate picks exactly as b1_select_events.py does (select_events, load_stations, load_picks: the
   v4.0 picks matched to the iter0/iter1 NLLoc phases) and run its pick_qc + event_check: the kept set must equal
   b1_dataset_c/picks.csv (reproduction check; stop otherwise).
2. Shift every candidate pick by its D-B075 correction (pick time, travel time and the NLLoc residuals res_i1, res_i0,
   res_noterm_i1: at a fixed hypocentre a pick time shift moves its residual by the same amount; the NLLoc matching
   is done before the shift, so no pick is lost to it) and run the same QC again. The medians and MAD of the
   thresholds are recomputed from the shifted residuals, as the rule would.
3. Write <out>/qc_check/: qc_before_after.csv (one row per candidate pick with both reasons), qc_summary.json,
   qc_counts.csv (picks removed per filter and phase, before / after).

  python b1t_qc_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b1_select_events as B1  # noqa: E402
from b1t_retime_dataset import catalog_rows, KEY  # noqa: E402
from manifest import load_params, write_manifest  # noqa: E402

P = load_params()
RT = P["retime"]
BASE = Path(P["paths"]["base"])
OUT = Path(RT["out"]) / "qc_check"


def run_qc(pk, k):
    pk, tab, thr, _ = B1.pick_qc(pk.copy())
    kept = pk[pk.qc == ""].copy()
    ev, below = B1.event_check(k, kept)
    drop = set(below.loc[below.decision == "drop", "ID"])
    pk.loc[pk.ID.isin(drop) & (pk.qc == ""), "qc"] = "event_dropped"
    return pk, tab, thr, drop


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat, cand, k, funnel = B1.select_events()
    st = B1.load_stations()
    pk = B1.load_picks(k, st)
    # 1. reproduction of v0.3.1
    q0, tab0, thr0, drop0 = run_qc(pk, k)
    ref = pd.read_csv(BASE / RT["datasets"]["b1"][0] / "picks.csv", usecols=["event_ID", "station", "phase", "pick_time"])
    kept0 = q0[q0.qc == ""]
    a = set(zip(kept0.ID, kept0.station, kept0.phase, kept0.pick_time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")))
    b = set(zip(ref.event_ID, ref.station, ref.phase, ref.pick_time))
    if a != b:
        raise SystemExit(f"B1 QC not reproduced: {len(a - b)} picks kept here only, {len(b - a)} in b1_dataset_c only")
    print(f"reproduced b1_dataset_c: {len(a)} picks", flush=True)
    # 2. shift by the corrections (join on the pick ID: event, station, phase, v4.0 time)
    rows, _ = catalog_rows()
    pk["t_old"] = pk.pick_time.astype("int64")
    m = pk.reset_index().merge(rows[KEY + ["t_old", "corr_ns", "epoch_class"]].rename(columns={"event_ID": "ID"}),
                               on=["ID", "station", "phase", "t_old"], how="left")
    if len(m) != len(pk) or m.corr_ns.isna().any():
        raise SystemExit(f"candidate picks: join not one-to-one ({len(m)} rows for {len(pk)}, "
                         f"{int(m.corr_ns.isna().sum())} unmatched)")
    m = m.set_index("index").loc[pk.index]
    c = m.corr_ns.values.astype("int64")
    cs = c / 1e9
    pk1 = pk.copy()
    pk1["pick_time"] = pd.to_datetime(pk.t_old.values + c)
    for col in ("tt", "res_i1", "res_i0", "res_noterm_i1"):
        pk1[col] = pk[col] + cs
    q1, tab1, thr1, drop1 = run_qc(pk1, k)
    # 3. compare
    d = pd.DataFrame({"event_ID": pk.ID.values, "station": pk.station.values, "phase": pk.phase.values,
                      "is_obs": pk.is_obs.values, "rule": pk.rule.values, "correction_s": np.round(cs, 6),
                      "qc_v031": q0.qc.values, "qc_retimed": q1.qc.values})
    d.to_csv(OUT / "qc_before_after.csv.gz", index=False, compression="gzip")
    cnt = []
    for name, q in (("v0.3.1 picks", d.qc_v031), ("re-timed picks", d.qc_retimed)):
        for f in ["no_nll", "w0", "duplicate", "res", "sp", "event_dropped", ""]:
            for ph in ("P", "S"):
                cnt.append(dict(picks=name, filter=f or "kept", phase=ph, n=int(((q == f) & (d.phase == ph)).sum())))
    cnt = pd.DataFrame(cnt).pivot_table(index=["filter", "phase"], columns="picks", values="n").reset_index()
    cnt.to_csv(OUT / "qc_counts.csv", index=False)
    ch = d[d.qc_v031 != d.qc_retimed]
    trans = ch.groupby(["qc_v031", "qc_retimed"]).size().rename("picks").reset_index()
    summ = dict(candidates=len(d), corrected=int((c != 0).sum()),
                kept_v031=int((d.qc_v031 == "").sum()), kept_retimed=int((d.qc_retimed == "").sum()),
                changed=len(ch), changed_corrected=int((ch.correction_s != 0).sum()),
                transitions=trans.to_dict("records"),
                events_dropped_v031=len(drop0), events_dropped_retimed=len(drop1),
                events_dropped_only_v031=sorted(map(int, drop0 - drop1)),
                events_dropped_only_retimed=sorted(map(int, drop1 - drop0)),
                thresholds_v031={k_: {kk: float(v) for kk, v in t.items()} for k_, t in thr0.items()},
                thresholds_retimed={k_: {kk: float(v) for kk, v in t.items()} for k_, t in thr1.items()})
    (OUT / "qc_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    print(cnt.to_string())
    write_manifest("b1t_qc_check", OUT, inputs=[RT["picks_v40"], RT["corrections"]],
                   outputs=[OUT / "qc_before_after.csv.gz", OUT / "qc_counts.csv", OUT / "qc_summary.json"],
                   params={"pick_qc": P["pick_qc"], "selection": P["selection"]}, counts=summ)


if __name__ == "__main__":
    main()
