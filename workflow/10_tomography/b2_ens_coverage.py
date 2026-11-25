"""Track B Bayesian ensemble: the D-B030 calibration gate on the synthetic ensemble (bayes_design section 5).

  python b2_ens_coverage.py [--members s000-s099] [--no-exit]      -> ROOT/synth/coverage.{json,md}; exit 0 PASS, 2 FAIL

Nodes (merge grid, the merged member models before the D-B073 node rule, against the synthetic truth model, which has
no rule either): for Vp, Vs and Vp/Vs, the ensemble 68 % and 95 % intervals (empirical quantiles 16-84 and 2.5-97.5;
Gaussian mean +- 1 and 1.96 std reported as well) and the share of nodes whose truth lies inside, on
* resolved: nodes resolved by the deterministic checkerboard tests (v<truth_model> resolution_p / resolution_s > 0,
  D-B018) with the tiles' share R > 0 in every member (the gate mask);
* sampled: nodes with DWS >= 100 (P for Vp, S for Vs) in >= member_frac of the members and R > 0 (reported).
Hypocentres: per tile, the members' tomoDD relocations (east, north, depth, km) against the truth hypocentres, for
events relocated in >= member_frac of the members; empirical intervals as above.
PASS when, on the gate mask with >= min_nodes nodes, Vp and Vs and each hypocentre coordinate have 68 % coverage in
calibration.cov68 and 95 % coverage in calibration.cov95. Vp/Vs is reported, not gated.
Noise floor: the clones nf* of s000 (identical draws, 1e-6 s data perturbation): their node and hypocentre spread,
reported against the ensemble spread.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
from b2_ens_common import CFG, E, ROOT, T  # noqa: E402

SD = ROOT / "synth"
CAL = E["calibration"]


def members(kind, names=None):
    d = ROOT / kind
    names = names or sorted(p.name for p in d.iterdir() if p.is_dir() and p.name[0] in "sr")
    return [n for n in names if (d / n / "MERGED").exists()]


def load_merged(kind, names):
    keys = ("vp", "vs", "r_p", "r_s", "dws_p", "dws_s")
    out = {k: [] for k in keys}
    for n in names:
        z = np.load(ROOT / kind / n / "merged.npz")
        for k in keys:
            out[k].append(z[k])
    return {k: np.array(v) for k, v in out.items()}


def intervals(X, truth):
    q = np.nanpercentile(X, [2.5, 16, 84, 97.5], axis=0)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1)
    return dict(e68=(truth >= q[1]) & (truth <= q[2]), e95=(truth >= q[0]) & (truth <= q[3]),
                g68=np.abs(truth - mu) <= sd, g95=np.abs(truth - mu) <= 1.96 * sd, sd=sd,
                z=(mu - truth) / np.where(sd > 0, sd, np.nan))


def row(I, m):
    n = int(m.sum())
    if not n:
        return dict(n=0)
    return dict(n=n, e68=round(float(I["e68"][m].mean()), 4), e95=round(float(I["e95"][m].mean()), 4),
                g68=round(float(I["g68"][m].mean()), 4), g95=round(float(I["g95"][m].mean()), 4),
                rms_z=round(float(np.sqrt(np.nanmean(I["z"][m] ** 2))), 3), sd_median=float(np.nanmedian(I["sd"][m])))


def ok(r):
    return r.get("n", 0) >= CAL["min_nodes"] and CAL["cov68"][0] <= r["e68"] <= CAL["cov68"][1] and \
        CAL["cov95"][0] <= r["e95"] <= CAL["cov95"][1]


def hypo_cov(names):
    tr = pd.read_parquet(SD / "truth" / "truth_hypo.parquet").set_index("ID")
    out = {}
    for tile in CFG["tiles"]:
        rel = []
        for n in names:
            f = SD / n / tile / "run" / "Output_Files" / "tomoDD.reloc_geo"
            rel.append(C.read_reloc(f)[["lat", "lon", "depth"]].add_prefix(f"{n}_"))
        R = pd.concat(rel, axis=1)
        frac = R.filter(like="_lat").notna().mean(axis=1)
        R = R[frac >= CAL["member_frac"]]
        R = R[R.index.isin(tr.index)]
        t = tr.loc[R.index]
        lat = R.filter(like="_lat").values
        lon = R.filter(like="_lon").values
        dep = R.filter(like="_depth").values
        km_n = (lat - t.lat.values[:, None]) * 111.19
        km_e = (lon - t.lon.values[:, None]) * 111.19 * np.cos(np.radians(t.lat.values[:, None]))
        km_z = dep - t.depth.values[:, None]
        res = {}
        for nm, X in (("east", km_e), ("north", km_n), ("depth", km_z)):
            I = intervals(X.T, np.zeros(X.shape[0]))
            res[nm] = row(I, np.ones(X.shape[0], bool))
            res[nm]["median_abs_err_km"] = round(float(np.nanmedian(np.abs(np.nanmean(X, axis=1)))), 3)
        out[tile] = dict(events=int(len(R)), **res)
    return out


def noise_floor():
    nf = sorted(p.name for p in SD.iterdir() if p.is_dir() and p.name.startswith("nf") and (p / "MERGED").exists())
    if not nf or not (SD / "s000" / "MERGED").exists():
        return dict(n=0)
    M = load_merged("synth", ["s000"] + nf)
    sdp, sds = np.nanstd(M["vp"], axis=0, ddof=1), np.nanstd(M["vs"], axis=0, ddof=1)
    cov = (M["r_p"] > 0).all(0)
    out = dict(n=len(nf) + 1, vp_std_median=float(np.median(sdp[cov])), vp_std_p95=float(np.percentile(sdp[cov], 95)),
               vs_std_median=float(np.median(sds[cov])), vs_std_p95=float(np.percentile(sds[cov], 95)))
    hz = []
    for tile in CFG["tiles"]:
        R = pd.concat([C.read_reloc(SD / n / tile / "run" / "Output_Files" / "tomoDD.reloc_geo").depth.rename(n)
                       for n in ["s000"] + nf], axis=1).dropna()
        hz.append(R.std(axis=1, ddof=1).values)
    hz = np.concatenate(hz)
    out.update(depth_std_median_km=float(np.median(hz)), depth_std_p95_km=float(np.percentile(hz, 95)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", help="range like s000-s099 (default: every merged synth member s*)")
    ap.add_argument("--no-exit", action="store_true")
    a = ap.parse_args()
    if a.members:
        lo, hi = (int(v[1:]) for v in a.members.split("-"))
        names = members("synth", [f"s{i:03d}" for i in range(lo, hi + 1)])
    else:
        names = members("synth")
    N = len(names)
    tm = np.load(SD / "truth" / "truth_model.npz")
    M = load_merged("synth", names)
    nd = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{CFG['truth_model']}.nc")
    resp, ress = (nd[v].values.transpose(2, 1, 0) > 0 for v in ("resolution_p", "resolution_s"))
    rp_all, rs_all = (M["r_p"] > 0).all(0), (M["r_s"] > 0).all(0)
    f = CAL["member_frac"]
    masks = {"resolved": (resp & rp_all, ress & rs_all),
             "sampled": ((M["dws_p"] >= 100).mean(0) >= f) & rp_all, }
    masks["sampled"] = (masks["sampled"], ((M["dws_s"] >= 100).mean(0) >= f) & rs_all)
    quant = {"Vp": (M["vp"], tm["vp"], 0), "Vs": (M["vs"], tm["vs"], 1), "VpVs": (M["vp"] / M["vs"], tm["vp"] / tm["vs"], 2)}
    nodes = {}
    for mk, (mp, ms) in masks.items():
        for q, (X, tr, kind) in quant.items():
            m = mp if kind == 0 else ms if kind == 1 else (mp & ms)
            nodes[f"{mk}_{q}"] = row(intervals(X, tr), m)
    hyp = hypo_cov(names)
    gate_nodes = {q: nodes[f"resolved_{q}"] for q in ("Vp", "Vs")}
    gate_hypo = {f"{t}_{c}": h[c] for t, h in hyp.items() for c in ("east", "north", "depth") if h["events"] > 0}
    verdict = {k: ok(r) for k, r in gate_nodes.items()}
    verdict.update({k: (CAL["cov68"][0] <= r["e68"] <= CAL["cov68"][1] and CAL["cov95"][0] <= r["e95"] <= CAL["cov95"][1])
                    for k, r in gate_hypo.items()})
    passed = bool(N >= 2 and all(verdict.values()))
    out = dict(members=N, names=names, criteria=CAL, nodes=nodes, hypocentres=hyp, verdict=verdict,
               result="PASS" if passed else "FAIL", noise_floor=noise_floor())
    C.jdump(SD / "coverage.json", out)
    md = [f"# Calibration of the {C.CONFIG} ensemble on synthetic data ({N} members)", "",
          f"Result: **{out['result']}** (68 % coverage in {CAL['cov68']}, 95 % in {CAL['cov95']}, gate mask 'resolved', "
          f">= {CAL['min_nodes']} nodes; hypocentres per tile and coordinate)", "",
          "| mask / quantity | nodes | 68 % (quantile) | 95 % (quantile) | 68 % (Gauss) | 95 % (Gauss) | rms z | median sd |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in nodes.items():
        md.append(f"| {k} | {r.get('n', 0)} | {r.get('e68', '')} | {r.get('e95', '')} | {r.get('g68', '')} | "
                  f"{r.get('g95', '')} | {r.get('rms_z', '')} | {r.get('sd_median', '')} |")
    md += ["", "| tile | coordinate | events | 68 % | 95 % | median abs error of the mean (km) |", "|---|---|---|---|---|---|"]
    for t, h in hyp.items():
        for c in ("east", "north", "depth"):
            r = h[c]
            md.append(f"| {t} | {c} | {h['events']} | {r.get('e68', '')} | {r.get('e95', '')} | {r.get('median_abs_err_km', '')} |")
    md += ["", f"Noise floor: {json.dumps(out['noise_floor'])}", "", f"Verdict per item: {json.dumps(verdict)}"]
    (SD / "coverage.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if not a.no_exit:
        sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
