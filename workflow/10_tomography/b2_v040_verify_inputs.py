"""v0.4.0: check that a re-timed region's tomoDD inputs differ from its v0.3.1 counterpart only in the pick times.

  python b2_v040_verify_inputs.py NEW OLD      e.g. margin_m0_v3 margin_m0_v2, blanco_m0t blanco_m0

Checks (exit 1 on a structural difference):
  event.dat, station.dat, station_codes.csv, events.csv, forward.json: byte-identical (same events, start hypocentres,
      stations, codes, coordinate mapping);
  picks_used.csv: same rows (event, station, code, phase) in the same order, same weights (w, w_raw, w_bal, sigma),
      same OBS statics; travel_time_s - v0.3.1 = the pick's timing correction (to the 1e-4 s of the 4-decimal times);
  absolute.dat, dt.ct: same lines apart from the times (event headers, pairs, codes, phases, weights); time changes;
  station_corr.csv (P-4): equal, or the largest difference reported (M0 recomputes it from a forward run whose
      predicted times are observed - residual, so a re-timed pick can move it by rounding);
  MOD: identical for M0 (same start model); for tiles the difference is the new M0's change (reported).
Writes <runs>/NEW/inputs/verify_vs_OLD.json.
"""
import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params  # noqa: E402
from b2_fwd_common import read_mod  # noqa: E402

P = load_params()
T = P["tomodd"]


def split_lines(path):
    hdr, rows = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        t = line.split()
        if line.startswith("#"):
            hdr.append(line)
            rows.append(("#", tuple(t)))
        else:
            rows.append(("d", tuple(t)))
    return rows


def compare_times(new, old, ntime):
    """Lines of absolute.dat (ntime 1) or dt.ct (ntime 2): structure equal, time differences returned."""
    a, b = split_lines(new), split_lines(old)
    if len(a) != len(b):
        return None, f"{Path(new).name}: {len(a)} lines, v0.3.1 {len(b)}"
    d = []
    for (ka, ta), (kb, tb) in zip(a, b):
        if ka != kb:
            return None, f"{Path(new).name}: line kinds differ"
        if ka == "#":
            if ta != tb:
                return None, f"{Path(new).name}: header {ta} != {tb}"
            continue
        # absolute: code tt w ph ; dt.ct: code t1 t2 w ph
        if ta[0] != tb[0] or ta[-1] != tb[-1] or float(ta[-2]) != float(tb[-2]):
            return None, f"{Path(new).name}: line {ta} vs {tb}"
        d.append([float(x) - float(y) for x, y in zip(ta[1:1 + ntime], tb[1:1 + ntime])])
    return np.array(d), None


def main():
    new, old = sys.argv[1], sys.argv[2]
    R = Path(T["runs"])
    A, B = R / new / "inputs", R / old / "inputs"
    out, bad = {"new": new, "old": old}, []
    for f in ("event.dat", "station.dat", "station_codes.csv", "events.csv", "forward.json"):
        same = filecmp.cmp(A / f, B / f, shallow=False)
        out[f] = "identical" if same else "DIFFERENT"
        if not same:
            bad.append(f)
    pa, pb = pd.read_csv(A / "picks_used.csv"), pd.read_csv(B / "picks_used.csv")
    keys = ["event_ID", "station", "code", "phase", "static_s", "sigma_s", "w_bal", "w_raw", "w", "is_obs"]
    if len(pa) != len(pb) or not all(pa[k].equals(pb[k]) for k in keys):
        bad.append("picks_used.csv rows/weights/statics")
    else:
        ds = R.parent / T["regions"][new]["b1b"] / "picks.csv"
        c = pd.read_csv(ds, usecols=["event_ID", "station", "phase", "timing_correction_s"])
        m = pa[["event_ID", "station", "phase"]].merge(c, on=["event_ID", "station", "phase"], how="left")
        dtt = pa.travel_time_s - pb.travel_time_s
        err = (dtt - m.timing_correction_s.values).abs()
        out["picks_used"] = dict(rows=len(pa), retimed=int((m.timing_correction_s != 0).sum()),
                                 tt_changed=int((dtt != 0).sum()), tt_minus_corr_absmax_s=float(err.max()),
                                 corr_absmax_s=float(m.timing_correction_s.abs().max()),
                                 station_corr_absdiff_max_s=float((pa.station_corr_s - pb.station_corr_s).abs().max()))
        if err.max() > 1.0001e-4:
            bad.append("picks_used travel time change != timing correction")
    for f, n in (("absolute.dat", 1), ("dt.ct", 2)):
        fa = A / f if not (A / "absolute_unperturbed.dat").exists() else \
            A / {"absolute.dat": "absolute_unperturbed.dat", "dt.ct": "dt_unperturbed.ct"}[f]
        d, err = compare_times(fa, B / f, n)
        if err:
            bad.append(err)
            continue
        out[f] = dict(lines=len(d), changed=int((d != 0).any(axis=1).sum()), absmax_s=float(np.abs(d).max()),
                      median_abs_changed_s=float(np.median(np.abs(d[(d != 0)]))) if (d != 0).any() else 0.0)
    ca, cb = pd.read_csv(A / "station_corr.csv"), pd.read_csv(B / "station_corr.csv")
    mm = ca.merge(cb, on=["station", "phase"], suffixes=("", "_old"), how="outer", indicator=True)
    out["station_corr"] = dict(rows=len(ca), rows_old=len(cb), unmatched=int((mm._merge != "both").sum()),
                               absdiff_max_s=float((mm.corr_s - mm.corr_s_old).abs().max()),
                               n_diff=int(((mm.corr_s - mm.corr_s_old).abs() > 0).sum()),
                               identical_file=filecmp.cmp(A / "station_corr.csv", B / "station_corr.csv", shallow=False))
    if out["station_corr"]["unmatched"]:
        bad.append("station_corr station-phase sets differ")
    ma, mb = read_mod(A / "MOD"), read_mod(B / "MOD")
    out["MOD"] = dict(identical=filecmp.cmp(A / "MOD", B / "MOD", shallow=False),
                      dlnvp_absmax_pct=float(100 * np.abs(ma["vp"] / mb["vp"] - 1).max()),
                      dlnvs_absmax_pct=float(100 * np.abs(ma["vs"] / mb["vs"] - 1).max()))
    out["problems"] = bad
    (A / f"verify_vs_{old}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
