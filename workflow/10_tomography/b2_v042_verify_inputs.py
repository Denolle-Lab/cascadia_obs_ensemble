"""v0.4.2 (D-B099): check that a rebuilt tile's tomoDD inputs are v0.4.1's minus exactly the self-paired differential times.

  python b2_v042_verify_inputs.py NEW OLD      e.g. washington_m0g washington_m0f,
                                                    washington_m0gnf washington_m0fnf (noise floor: unperturbed files)

The fixed b2_tomodd_inputs.diff_times drops the event itself by index instead of dropping column 0 of the KD-tree query;
with co-located events the old code had paired an event with itself ("#  ID  ID"), which tomoDD reads as an absolute
time of 0 s. Required (exit 1 otherwise):
* byte-identical: event.dat, station.dat, station_codes.csv, events.csv, forward.json, picks_used.csv, station_corr.csv,
  absolute.dat (noise floor: absolute_unperturbed.dat), MOD (same start model), and surface.dat when both stages have
  it (same floor); floor.json equal apart from its region and inputs keys;
* dt.ct (noise floor: dt_unperturbed.ct) equal, line for line, to OLD's with the self-pair blocks (header + data lines)
  removed, and no self-pair left in NEW;
* summary.json equal apart from region, dt_pairs and dt_data, which drop by the number of removed blocks and lines.
Noise-floor regions: the perturbed absolute.dat / dt.ct differ by the region-name seed of b2_v040_noise.py (reported).
Writes <runs>/NEW/inputs/verify_vs_OLD.json.
"""
import filecmp
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params  # noqa: E402

T = load_params()["tomodd"]
RUNS = Path(T["runs"])


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def blocks(path):
    """dt.ct as a list of (header, [data lines])."""
    out = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#"):
            out.append((ln, []))
        elif ln.strip():
            out[-1][1].append(ln)
    return out


def is_self(header):
    t = header.split()
    return t[1] == t[2]


def main():
    new, old = sys.argv[1], sys.argv[2]
    a, b = RUNS / new / "inputs", RUNS / old / "inputs"
    nf = (a / "absolute_unperturbed.dat").exists()
    absf, dtf = ("absolute_unperturbed.dat", "dt_unperturbed.ct") if nf else ("absolute.dat", "dt.ct")
    same = ["event.dat", "station.dat", "station_codes.csv", "events.csv", "forward.json", "picks_used.csv",
            "station_corr.csv", absf, "MOD"]
    same += [f for f in ("surface.dat",) if (a / f).exists() and (b / f).exists()]
    problems, res = [], {}
    for f in same:
        if not (a / f).exists() or not (b / f).exists():
            problems.append(f"{f}: missing ({(a / f).exists()}, {(b / f).exists()})")
            continue
        ok = filecmp.cmp(a / f, b / f, shallow=False)
        res[f] = dict(identical=ok, md5=md5(a / f))
        if not ok:
            problems.append(f"{f} differs")
    if (a / "floor.json").exists() and (b / "floor.json").exists():
        fa, fb = (json.loads((x / "floor.json").read_text()) for x in (a, b))
        dk = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
        res["floor.json"] = dict(differing_keys=dk)
        if set(dk) - {"region", "inputs"}:
            problems.append(f"floor.json differs in {dk}")
    bn, bo = blocks(a / dtf), blocks(b / dtf)
    removed = [h for h, _ in bo if is_self(h)]
    expect = [(h, d) for h, d in bo if not is_self(h)]
    n_lines = sum(len(d) for h, d in bo if is_self(h))
    left = [h for h, _ in bn if is_self(h)]
    ok_dt = bn == expect
    res[dtf] = dict(equals_old_minus_self_pairs=ok_dt, self_pairs_removed=[int(h.split()[1]) for h in removed],
                    self_pair_data_lines_removed=n_lines, pairs_old=len(bo), pairs_new=len(bn),
                    data_old=sum(len(d) for _, d in bo), data_new=sum(len(d) for _, d in bn),
                    self_pairs_left=[int(h.split()[1]) for h in left], md5=md5(a / dtf))
    if not ok_dt:
        problems.append(f"{dtf} is not OLD minus the self-pair blocks")
    if left:
        problems.append(f"{dtf} still has self-pairs {left}")
    sa, sb = json.loads((a / "summary.json").read_text()), json.loads((b / "summary.json").read_text())
    diff = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    want = dict(dt_pairs=sb["dt_pairs"] - len(removed), dt_data=sb["dt_data"] - n_lines)
    ok_s = set(diff) <= {"region", "dt_pairs", "dt_data"} and all(sa[k] == v for k, v in want.items())
    res["summary.json"] = dict(differing_keys=diff, dt_pairs=[sb["dt_pairs"], sa["dt_pairs"]],
                               dt_data=[sb["dt_data"], sa["dt_data"]], as_expected=ok_s)
    if not ok_s:
        problems.append(f"summary.json: differing keys {diff}, expected dt_pairs/dt_data {want}")
    if nf:
        res["perturbed_files_identical"] = {f: filecmp.cmp(a / f, b / f, shallow=False) for f in ("absolute.dat", "dt.ct")}
        res["noise"] = json.loads((a / "noise.json").read_text())
    out = dict(new=new, old=old, noise_floor_region=nf, files=res, problems=problems)
    (a / f"verify_vs_{old}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
