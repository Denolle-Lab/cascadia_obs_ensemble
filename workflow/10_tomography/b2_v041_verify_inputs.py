"""v0.4.1: check that a floored stage's tomoDD inputs are the v0.4.0 inputs (D-B083: "the same inputs").

  python b2_v041_verify_inputs.py NEW OLD      e.g. margin_m0_v4 margin_m0_v3, blanco_m0f blanco_m0t,
                                                    margin_m0_v4nf margin_m0_v3nf (noise floor: unperturbed files)

Byte-identical required (exit 1 otherwise): event.dat, station.dat, station_codes.csv, events.csv, forward.json,
picks_used.csv, station_corr.csv (P-4: M0 recomputes it with the 2026-11-25 binary, tiles reuse <tile>_v2's), absolute.dat
and dt.ct (noise-floor regions: absolute_unperturbed.dat and dt_unperturbed.ct; the perturbed files differ by the
region-name seed of b2_v040_noise.py), summary.json apart from the region name.
MOD: byte-identical for M0 (same start); for tiles it is the start from the new M0 (difference reported).
Writes <runs>/NEW/inputs/verify_vs_OLD.json.
"""
import filecmp
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params  # noqa: E402
from b2_fwd_common import read_mod  # noqa: E402

T = load_params()["tomodd"]
RUNS = Path(T["runs"])


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    new, old = sys.argv[1], sys.argv[2]
    a, b = RUNS / new / "inputs", RUNS / old / "inputs"
    nf = (a / "absolute_unperturbed.dat").exists()
    same = ["event.dat", "station.dat", "station_codes.csv", "events.csv", "forward.json", "picks_used.csv",
            "station_corr.csv"]
    same += ["absolute_unperturbed.dat", "dt_unperturbed.ct"] if nf else ["absolute.dat", "dt.ct"]
    tile = T["regions"][new].get("start_model") is not None
    if not tile:
        same.append("MOD")
    problems, res = [], {}
    for f in same:
        if not (a / f).exists() or not (b / f).exists():
            problems.append(f"{f}: missing ({(a / f).exists()}, {(b / f).exists()})")
            continue
        ok = filecmp.cmp(a / f, b / f, shallow=False)
        res[f] = dict(identical=ok, md5=md5(a / f))
        if not ok:
            problems.append(f"{f} differs")
    sa, sb = json.loads((a / "summary.json").read_text()), json.loads((b / "summary.json").read_text())
    sa.pop("region"), sb.pop("region")
    res["summary.json"] = dict(identical_except_region=sa == sb)
    if sa != sb:
        problems.append(f"summary.json differs: {[k for k in sa if sa.get(k) != sb.get(k)]}")
    if nf:
        res["perturbed_files_identical"] = {f: filecmp.cmp(a / f, b / f, shallow=False) for f in ("absolute.dat", "dt.ct")}
        res["noise"] = json.loads((a / "noise.json").read_text())
    if tile:
        ma, mb = read_mod(a / "MOD"), read_mod(b / "MOD")
        geo = all(np.array_equal(ma[k], mb[k]) for k in ("lon", "lat", "z"))
        if not geo:
            problems.append("MOD node coordinates differ")
        dvp = 100 * (ma["vp"] / mb["vp"] - 1)
        dvs = 100 * (ma["vs"] / mb["vs"] - 1)
        res["MOD"] = dict(identical=filecmp.cmp(a / "MOD", b / "MOD", shallow=False), same_nodes=geo,
                          start_from=T["regions"][new]["start_model"],
                          dvp_pct_rms_max=[round(float(np.sqrt((dvp ** 2).mean())), 4), round(float(np.abs(dvp).max()), 3)],
                          dvs_pct_rms_max=[round(float(np.sqrt((dvs ** 2).mean())), 4), round(float(np.abs(dvs).max()), 3)])
    out = dict(new=new, old=old, noise_floor_region=nf, files=res, problems=problems)
    (a / f"verify_vs_{old}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
