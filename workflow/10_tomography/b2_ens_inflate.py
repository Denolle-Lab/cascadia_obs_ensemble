"""Track B Bayesian ensemble: D-B094 inflation factors of the Vp and Vs posterior spreads from a synthetic calibration.

  B2_ENS_CONFIG=<c> python b2_ens_inflate.py [--out DIR]        (default env)
      -> DIR/inflation.json, DIR/inflation.md (default ROOT/calibration/ of configuration <c>)

Source: configs.<c>.calibration_root (or ROOT when it has none): the synthetic members of its synth/coverage.json (the
same members as its calibrate verdict), its truth (synth/truth/truth_model.npz) and the masks of b2_ens_coverage.py,
recomputed here and checked against coverage.json (factor 1 must reproduce its coverages):
* resolved: checkerboard resolution > 0 in v<truth_model> (D-B018) and the tiles' share R > 0 in every member (gate mask);
* sampled: DWS >= 100 (P for Vp, S for Vs) in >= member_frac of the members and R > 0 in every member.
Quantities: the merged member models before the D-B073 node rule (merged.npz vp, vs) against the truth.
Rule (D-B094, PI 2026-11-25 15:37): inflated members x' = median + f (x - median), node by node, so every empirical
percentile interval widens by f about the ensemble median (and the std by f). Per node, the smallest f whose interval
holds the truth is (truth - median) / (q_hi - median) above the median, (median - truth) / (median - q_lo) below it;
f68 = the smallest f with 16-84 % coverage >= 0.68 on the resolved nodes, f95 = the smallest f with 2.5-97.5 %
coverage >= 0.95; the factor is max(f68, f95) rounded UP to the next 0.05 (PI: "round up because that's still a bit
too narrow"). Hypocentres and Vp/Vs are not inflated. Reported as a check, not used: the factors on the sampled nodes,
per tile (a node belongs to the tile with the largest merge weight of its phase, prior/merge_geometry.pkl), and the
Gaussian version (mean +- f std, 1 and 1.96 std).
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
from b2_ens_common import CFG, E, ROOT, T  # noqa: E402

LEVELS = {"68": (16.0, 84.0, 0.68), "95": (2.5, 97.5, 0.95)}
STEP = 0.05


def round_up(f, step=STEP):
    """Up to the next multiple of step (a value on a multiple stays; 1e-9 guards float noise)."""
    return round(math.ceil(f / step - 1e-9) * step, 10)


def need_factor(X, truth, lo, hi):
    """Per node: the smallest f with truth inside median + f (q_lo - median) .. median + f (q_hi - median)."""
    q = np.nanpercentile(X, [lo, 50.0, hi], axis=0)
    ql, med, qh = q
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(qh > med, (truth - med) / (qh - med), np.inf)
        dn = np.where(med > ql, (med - truth) / (med - ql), np.inf)
    return np.where(truth > med, up, np.where(truth < med, dn, 0.0)), ql, med, qh


def covered(truth, ql, med, qh, f):
    return (truth >= med + f * (ql - med)) & (truth <= med + f * (qh - med))


def smallest(need, p):
    """Smallest f with share(need <= f) >= p."""
    s = np.sort(need)
    k = int(math.ceil(p * len(s) - 1e-9))
    return float(s[max(k, 1) - 1])


def analyse(X, truth, m):
    """Factors and coverages on mask m (quantile and Gaussian versions)."""
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    Xm, tm = X[:, m], truth[m]
    out = dict(n=n)
    nf = {}
    for lev, (lo, hi, p) in LEVELS.items():
        need, ql, med, qh = need_factor(Xm, tm, lo, hi)
        nf[lev] = (need, ql, med, qh)
        out[f"cov{lev}_raw"] = round(float(covered(tm, ql, med, qh, 1.0).mean()), 4)
        out[f"f{lev}"] = round(smallest(need, p), 4)
    out["f_max"] = max(out["f68"], out["f95"])
    out["factor"] = round_up(out["f_max"])
    mu, sd = np.nanmean(Xm, axis=0), np.nanstd(Xm, axis=0, ddof=1)
    zg = np.abs(tm - mu) / np.where(sd > 0, sd, np.nan)
    zg = np.where(np.isfinite(zg), zg, np.inf)
    out["gauss_cov68_raw"] = round(float((zg <= 1.0).mean()), 4)
    out["gauss_cov95_raw"] = round(float((zg <= 1.96).mean()), 4)
    out["gauss_f68"] = round(smallest(zg, 0.68) / 1.0, 4)
    out["gauss_f95"] = round(smallest(zg, 0.95) / 1.96, 4)
    out["rms_z_raw"] = round(float(np.sqrt(np.nanmean(((mu - tm) / np.where(sd > 0, sd, np.nan)) ** 2))), 3)
    out["nonfinite_nodes"] = int((~np.isfinite(Xm).all(0) | ~np.isfinite(tm)).sum())
    out["_nf"] = nf
    out["_g"] = (zg, mu, sd)
    return out


def after(r, f):
    """Coverages of r's nodes with spreads x f (quantile and Gaussian)."""
    if not r.get("n"):
        return {}
    tm_cov = {}
    for lev in LEVELS:
        need = r["_nf"][lev][0]
        tm_cov[f"cov{lev}"] = round(float((need <= f + 1e-12).mean()), 4)
    zg = r["_g"][0]
    tm_cov["gauss_cov68"] = round(float((zg <= f).mean()), 4)
    tm_cov["gauss_cov95"] = round(float((zg <= 1.96 * f).mean()), 4)
    tm_cov["rms_z"] = round(r["rms_z_raw"] / f, 3)
    return tm_cov


def home_tile(geo, phase, shape):
    """Index of the tile with the largest merge weight per node (-1: no tile)."""
    W = []
    for g in geo["geo"]:
        w = np.zeros(int(np.prod(shape)))
        w[g["sel"]] = g[f"w_{phase}"]
        W.append(w)
    W = np.array(W)
    h = np.where(W.max(0) > 0, W.argmax(0), -1)
    return h.reshape(shape), [g["tile"]["stage"] for g in geo["geo"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    cr = Path(CFG.get("calibration_root") or ROOT)
    sd = cr / "synth"
    cj = sd / "coverage.json"
    cov = json.loads(cj.read_text())
    pri = json.loads((cr / "prior" / "prior.json").read_text())
    ccfg = E["configs"][pri["config"]]                       # the configuration the calibration root was run with
    names = cov["names"]
    miss = [n for n in names if not (sd / n / "MERGED").exists()]
    if miss:
        raise SystemExit(f"members of {cj} not merged: {miss}")
    keys = ("vp", "vs", "r_p", "r_s", "dws_p", "dws_s")
    M = {k: [] for k in keys}
    for n in names:
        z = np.load(sd / n / "merged.npz")
        for k in keys:
            M[k].append(z[k])
    M = {k: np.array(v, "float64") for k, v in M.items()}
    tm = np.load(sd / "truth" / "truth_model.npz")
    nd = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{ccfg['truth_model']}.nc")
    resp, ress = (nd[v].values.transpose(2, 1, 0) > 0 for v in ("resolution_p", "resolution_s"))
    rp_all, rs_all = (M["r_p"] > 0).all(0), (M["r_s"] > 0).all(0)
    fr = E["calibration"]["member_frac"]
    masks = {"resolved": {"Vp": resp & rp_all, "Vs": ress & rs_all},
             "sampled": {"Vp": ((M["dws_p"] >= 100).mean(0) >= fr) & rp_all,
                         "Vs": ((M["dws_s"] >= 100).mean(0) >= fr) & rs_all}}
    X = {"Vp": (M["vp"], tm["vp"].astype("float64"), "P"), "Vs": (M["vs"], tm["vs"].astype("float64"), "S")}
    with open(cr / "prior" / "merge_geometry.pkl", "rb") as f:
        GEO = pickle.load(f)
    shape = tuple(GEO["shape"])
    res = {}
    for mk, mm in masks.items():
        for q, (Xq, tq, ph) in X.items():
            res[f"{mk}_{q}"] = analyse(Xq, tq, mm[q])
            ht, tiles = home_tile(GEO, ph, shape)
            for i, t in enumerate(tiles):
                res[f"{mk}_{q}_{t}"] = analyse(Xq, tq, mm[q] & (ht == i))
    # check against the calibrate verdict (factor 1 = coverage.json)
    chk = {}
    for k in ("resolved_Vp", "resolved_Vs", "sampled_Vp", "sampled_Vs"):
        c = cov["nodes"][k]
        chk[k] = dict(n=(res[k]["n"], c["n"]), cov68=(res[k]["cov68_raw"], c["e68"]), cov95=(res[k]["cov95_raw"], c["e95"]))
        if res[k]["n"] != c["n"] or abs(res[k]["cov68_raw"] - c["e68"]) > 1e-3 or abs(res[k]["cov95_raw"] - c["e95"]) > 1e-3:
            raise SystemExit(f"{k}: recomputed coverage {chk[k]} differs from {cj}")
    factors = {q: res[f"resolved_{q}"]["factor"] for q in ("Vp", "Vs")}
    for k, r in res.items():
        q = "Vp" if "_Vp" in k else "Vs"
        r["after_used_factor"] = after(r, factors[q])
        r["after_own_factor"] = after(r, r["factor"]) if r.get("n") else {}
    clean = {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in res.items()}
    out = dict(decision="D-B094", created=time.strftime("%F %T"), configuration=C.CONFIG,
               rule=("x' = median + f (x - median) per node (percentile intervals about the median and std x f); "
                     "f = max(f68, f95) on the resolved nodes, rounded up to the next 0.05; f68 / f95 = smallest f with "
                     "16-84 % / 2.5-97.5 % coverage of the truth >= 0.68 / 0.95; hypocentres and Vp/Vs not inflated"),
               source=dict(calibration_root=str(cr), coverage_json=str(cj), coverage_json_md5=C.md5(cj),
                           calibration_config=pri["config"], truth_model=ccfg["truth_model"], members=len(names),
                           result=cov["result"]),
               factors=factors, check_against_coverage_json=chk, items=clean)
    od = Path(a.out) if a.out else ROOT / "calibration"
    od.mkdir(parents=True, exist_ok=True)
    C.jdump(od / "inflation.json", out)
    md = [f"# D-B094 inflation factors of the Vp and Vs spreads (configuration {C.CONFIG})", "",
          f"Source: {cj} (md5 {out['source']['coverage_json_md5']}; configuration {pri['config']}, truth "
          f"v{ccfg['truth_model']}, {len(names)} synthetic members, result {cov['result']}). Created {out['created']}.", "",
          f"Rule: {out['rule']}.", "",
          f"**Factors used: Vp {factors['Vp']:.2f}, Vs {factors['Vs']:.2f}** (resolved nodes).", "",
          "| item | nodes | 68 % raw | 95 % raw | f68 | f95 | max | rounded up | 68 % after (used f) | 95 % after (used f) "
          "| rms z raw / after | Gauss 68 / 95 raw | Gauss 68 / 95 after |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in clean.items():
        if not r.get("n"):
            md.append(f"| {k} | 0 | | | | | | | | | | | |")
            continue
        a_ = r["after_used_factor"]
        md.append(f"| {k} | {r['n']} | {r['cov68_raw']} | {r['cov95_raw']} | {r['f68']:.3f} | {r['f95']:.3f} | "
                  f"{r['f_max']:.3f} | {r['factor']:.2f} | {a_['cov68']} | {a_['cov95']} | {r['rms_z_raw']} / {a_['rms_z']} | "
                  f"{r['gauss_cov68_raw']} / {r['gauss_cov95_raw']} | {a_['gauss_cov68']} / {a_['gauss_cov95']} |")
    (od / "inflation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
