"""v0.4.0 numerical noise floor (notes/retime_v040.md; design as notes/bayes_design.md section 3: identical inputs
except a 1e-6 s data perturbation).

For a region whose inputs were built by b2_tomodd_inputs.py (params tomodd.regions.<region>, e.g. margin_m0_v3nf =
margin_m0_v3), add e ~ N(0, retime.noise_sigma_s) to every observed time, one draw per pick (event, station code,
phase; seed retime.noise_seed + a hash of the region name), the same draw in absolute.dat and in both dt.ct columns
that use the pick. Times are written with 6 decimals (b2_tomodd_inputs.py writes 4, which would round the draw away);
tomoDD reads them free-format into REAL*4, so the perturbation is of the order of one float32 ulp of a travel time:
the numerical noise floor of the chain. The unperturbed files are kept as absolute_unperturbed.dat, dt_unperturbed.ct;
rerunning regenerates the perturbed files from them (idempotent).

  python b2_v040_noise.py REGION
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import load_params, write_manifest  # noqa: E402

P = load_params()
T = P["tomodd"]
RT = P["retime"]


def main():
    region = sys.argv[1]
    inp = Path(T["runs"]) / region / "inputs"
    a0, d0 = inp / "absolute_unperturbed.dat", inp / "dt_unperturbed.ct"
    if not a0.exists():
        (inp / "absolute.dat").rename(a0)
        (inp / "dt.ct").rename(d0)
    sig = float(RT["noise_sigma_s"])
    rng = np.random.default_rng(int(RT["noise_seed"]) + zlib.crc32(region.encode()))
    eps = {}

    def e(ev, code, ph):
        k = (ev, code, ph)
        if k not in eps:
            eps[k] = rng.normal(0.0, sig)
        return eps[k]

    out, ev, n = [], None, 0
    for line in a0.read_text().splitlines():
        if line.startswith("#"):
            ev = int(line.split()[1])
            out.append(line)
            continue
        code, tt, w, ph = line.split()
        out.append(f"{code:<7s} {float(tt) + e(ev, code, ph):11.6f} {float(w):6.3f} {ph}")
        n += 1
    (inp / "absolute.dat").write_text("\n".join(out) + "\n")
    out, nd = [], 0
    for line in d0.read_text().splitlines():
        if line.startswith("#"):
            _, e1, e2 = line.split()
            e1, e2 = int(e1), int(e2)
            out.append(line)
            continue
        if not line.strip():
            continue
        code, t1, t2, w, ph = line.split()
        out.append(f"{code:<7s} {float(t1) + e(e1, code, ph):11.6f} {float(t2) + e(e2, code, ph):11.6f} "
                   f"{float(w):6.3f} {ph}")
        nd += 1
    (inp / "dt.ct").write_text("\n".join(out) + "\n")
    v = np.array(list(eps.values()))
    counts = dict(region=region, absolute=n, dt=nd, picks_drawn=len(v), sigma_s=sig,
                  draw_std_s=float(v.std()), draw_absmax_s=float(np.abs(v).max()))
    (inp / "noise.json").write_text(json.dumps(counts, indent=1))
    print(counts)
    write_manifest(f"b2_v040_noise_{region}", inp, inputs=[a0, d0], outputs=[inp / "absolute.dat", inp / "dt.ct"],
                   params={"retime": RT}, counts=counts)


if __name__ == "__main__":
    main()
