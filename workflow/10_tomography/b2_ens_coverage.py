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
N_eff-aware consistency (review 2026-11-25), reported next to the D-B030 verdict without changing it: per item the
observed coverage, N_eff (n^2 / sum rho, and / sum rho^2, rho the prior's 30 km / 8 km correlation; events binned
5 x 5 x 2 km) and the Clopper-Pearson 95 % interval with n = N_eff, and whether 0.68 / 0.95 lies inside.
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
from b2_ens_common import CFG, E, PR, ROOT, T  # noqa: E402

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
        res = {"neff": neff_events(t.lat.values, t.lon.values, t.depth.values) if len(t) else (0, 0.0, 0.0)}
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


# ---------------------------------------------------------------- N_eff-aware consistency (review 2026-11-25; reported,
# not part of the D-B030 pass rule, which is the PI's): the truth is ONE random field with 30 km / 8 km correlation, so
# the gate's nodes (and events) are not independent trials. N_eff = n^2 / sum_ij rho_ij^k with rho the prior's
# correlation exp(-(dh/Lh)^2) exp(-(dz/Lz)^2): k = 1 for the field itself (conservative), k = 2 for the coverage
# indicators (leading term). Clopper-Pearson 95 % interval of the coverage with n = N_eff; consistent = the nominal
# value (0.68 or 0.95) lies inside.
def _rho_sums(x, y, z, w=None, chunk=2000):
    Lh, Lz = float(PR["rf"]["len_lat_km"]), float(PR["rf"]["len_z_km"])
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    s1 = s2 = 0.0
    for i in range(0, len(x), chunk):
        sl = slice(i, i + chunk)
        r = (np.exp(-((x[sl, None] - x[None]) ** 2 + (y[sl, None] - y[None]) ** 2) / Lh ** 2)
             * np.exp(-((z[sl, None] - z[None]) / Lz) ** 2))
        ww = w[sl, None] * w[None]
        s1 += float((ww * r).sum())
        s2 += float((ww * r ** 2).sum())
    n = float(w.sum())
    return n, n ** 2 / s1, n ** 2 / s2


def neff_nodes(mask, lon, lat, z):
    LO, LA, Z = np.meshgrid(lon, lat, z, indexing="ij")
    X, Y = C.proj()(LO[mask], LA[mask])
    return _rho_sums(X / 1e3, Y / 1e3, Z[mask])


def neff_events(lat, lon, depth, cell_km=5.0, cell_z_km=2.0):
    """Events binned into cell_km x cell_km x cell_z_km cells (counts as weights; within-cell rho taken as 1)."""
    X, Y = C.proj()(np.asarray(lon), np.asarray(lat))
    k = pd.DataFrame({"i": np.floor(X / 1e3 / cell_km), "j": np.floor(Y / 1e3 / cell_km),
                      "k": np.floor(np.asarray(depth) / cell_z_km)}).value_counts().reset_index(name="n")
    return _rho_sums((k.i.values + 0.5) * cell_km, (k.j.values + 0.5) * cell_km, (k.k.values + 0.5) * cell_z_km, k.n.values)


def cp_interval(p, n):
    from scipy.stats import beta
    x = p * n
    lo = float(beta.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0
    hi = float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0
    return [round(lo, 3), round(hi, 3)]


def consistency(r, n_field, n_ind):
    out = dict(n=r.get("n", 0), neff_field=round(n_field, 1), neff_indicator=round(n_ind, 1))
    for lev, key in ((0.68, "e68"), (0.95, "e95")):
        if key not in r:
            continue
        obs = r[key]
        for lab, ne in (("field", n_field), ("indicator", n_ind)):
            iv = cp_interval(obs, ne)
            out[f"{key}_{lab}_ci95"] = iv
            out[f"{key}_{lab}_nominal_inside"] = bool(iv[0] <= lev <= iv[1])
        out[key] = obs
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
    cons = {}
    for mk, (mp, ms) in masks.items():
        for q, kind in (("Vp", 0), ("Vs", 1), ("VpVs", 2)):
            m = mp if kind == 0 else ms if kind == 1 else (mp & ms)
            if m.sum():
                _, nf, ni = neff_nodes(m, tm["lon"], tm["lat"], tm["z"])
                cons[f"{mk}_{q}"] = consistency(nodes[f"{mk}_{q}"], nf, ni)
    for t, h in hyp.items():
        if h["events"]:
            _, nf, ni = h["neff"]
            for c in ("east", "north", "depth"):
                cons[f"{t}_{c}"] = consistency(h[c], nf, ni)
    out = dict(members=N, names=names, criteria=CAL, nodes=nodes, hypocentres=hyp, verdict=verdict,
               result="PASS" if passed else "FAIL", neff_consistency=cons, noise_floor=noise_floor())
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
    md += ["", f"Noise floor: {json.dumps(out['noise_floor'])}", "", f"Verdict per item (D-B030 rule): {json.dumps(verdict)}",
           "", "## N_eff-aware consistency (reported next to the D-B030 verdict; does not change it)", "",
           "One truth field with 30 km / 8 km correlation: nodes and events are not independent. N_eff = n^2 / sum rho^k "
           "(k = 1 field, k = 2 coverage indicators). Clopper-Pearson 95 % interval of the observed coverage with n = "
           "N_eff; 'in' = the nominal value lies inside (consistent with calibration). Gate items (D-B030) marked *.", "",
           "| item | n | N_eff field / indicator | 68 % obs | 95 % CI (field) | 95 % CI (indicator) | 0.68 in (f / i) | "
           "95 % obs | 95 % CI (field) | 95 % CI (indicator) | 0.95 in (f / i) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    gate_items = {"resolved_Vp", "resolved_Vs"} | {f"{t}_{c}" for t in hyp for c in ("east", "north", "depth")}
    for k, r in cons.items():
        if "e68" not in r:
            continue
        md.append(f"| {k}{' *' if k in gate_items else ''} | {r['n']} | {r['neff_field']} / {r['neff_indicator']} | "
                  f"{r['e68']} | {r['e68_field_ci95']} | {r['e68_indicator_ci95']} | "
                  f"{'yes' if r['e68_field_nominal_inside'] else 'no'} / {'yes' if r['e68_indicator_nominal_inside'] else 'no'} | "
                  f"{r['e95']} | {r['e95_field_ci95']} | {r['e95_indicator_ci95']} | "
                  f"{'yes' if r['e95_field_nominal_inside'] else 'no'} / {'yes' if r['e95_indicator_nominal_inside'] else 'no'} |")
    (SD / "coverage.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if not a.no_exit:
        sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
