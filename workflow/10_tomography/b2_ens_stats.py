"""Track B Bayesian ensemble: per-node statistics, the versioned ensemble product (D-B047) and the HTML report.

  python b2_ens_stats.py stats KIND                       (default env)  -> ROOT/stats/KIND/stats.npz, stats.json
  python b2_ens_stats.py products KIND [--dry-run DIR]    (models env)   -> dry-run netCDF + report HTML (default);
        B2_ENS_REGISTER=1 registers the immutable version <model version>+ens.<members> (PI's go needed)
Statistics over the merged members (merge grid of b2_merge.py: 0.125 x 0.1 deg, tomodd.node_z):
* Vp, Vs, VpVs as v0.3.2 defines them (medium from the node model: water Vp 1.5 and NaN Vs, air NaN; Vs with the D-B073
  node rule at the member's drawn Vp/Vs and thickness), and Vp_tomography, Vs_tomography (the inversion's values, no
  surface, no rule): mean, median, std, 5 and 95 % quantiles, number of members;
* prob_increase_p / _s: share of members whose inverted velocity exceeds that member's own starting model (dlnV > 0;
  0 where no tile resolves the node, R = 0);
* tile_disagreement_p / _s (P-M5, merge_design M4): sqrt(sum_k w_k (mu_k - mu)^2) of the tiles' own ensemble means
  mu_k (each tile's member models at the grid node) in tile overlaps, w_k the normalised merge weights; std_total =
  sqrt(std^2 + disagreement^2) for the tomography velocities;
* resolved_p / _s: deterministic checkerboard resolution > 0 (D-B018, v<node_model> resolution_*) and R > 0 in every
  member; sampled_p / _s: DWS >= 100 in >= member_frac of the members.
stats.npz / stats.json hold the raw member statistics. Calibration (D-B094), applied by `products` when
configs.<c>.calibration.inflation is set (factor f per Vp / Vs, from b2_ens_inflate.py on the synthetic members of the
calibration root): Vp_std, Vs_std, Vp_tomography_std, Vs_tomography_std -> f x std; their p05 / p95 -> median + f (p -
median); the tomography std_total from the inflated std and the (not inflated) tile disagreement. The raw values stay
in the product as <name>_raw; mean, median, n, VpVs and hypocentres are not changed. The factors, the rule, the
decision, the source calibration (coverage.json md5) and the derivation (ROOT/calibration/inflation.json) are in the
product attributes (provenance_uncertainty_calibration, the variables' calibration attributes) and the HTML report.
"""
import argparse
import base64
import io
import json
import pickle
import sys
import time
from html import escape as html_escape
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_ens_common as C  # noqa: E402
from b2_ens_common import CFG, E, ROOT, T  # noqa: E402

KEYS = ("vp", "vs", "vp_start", "vs_start", "vp_medium", "vs_rule", "r_p", "r_s", "dws_p", "dws_s")


def merged_names(kind):
    d = ROOT / kind
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name[0] in "sr" and (p / "MERGED").exists())


def stats(kind):
    from b2_fwd_common import mod_velocity
    names = merged_names(kind)
    N = len(names)
    out = ROOT / "stats" / kind
    out.mkdir(parents=True, exist_ok=True)
    M = {k: [] for k in KEYS}
    for n in names:
        z = np.load(ROOT / kind / n / "merged.npz")
        for k in KEYS:
            M[k].append(z[k])
    M = {k: np.array(v, "float64") for k, v in M.items()}
    S = {}
    Q = {"Vp": M["vp_medium"], "Vs": M["vs_rule"], "VpVs": M["vp_medium"] / M["vs_rule"], "Vp_tomography": M["vp"],
         "Vs_tomography": M["vs"]}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for q, X in Q.items():
            S[f"{q}_mean"] = np.nanmean(X, 0)
            S[f"{q}_median"] = np.nanmedian(X, 0)
            S[f"{q}_std"] = np.nanstd(X, 0, ddof=1) if N > 1 else np.full(X.shape[1:], np.nan)
            S[f"{q}_p05"], S[f"{q}_p95"] = np.nanpercentile(X, [5, 95], axis=0)
            S[f"{q}_n"] = np.isfinite(X).sum(0).astype("int16")
    S["prob_increase_p"] = ((M["vp"] / M["vp_start"] - 1) > 0).mean(0)
    S["prob_increase_s"] = ((M["vs"] / M["vs_start"] - 1) > 0).mean(0)
    # P-M5 tile disagreement
    with open(C.PRIOR / "merge_geometry.pkl", "rb") as f:
        GEO = pickle.load(f)
    shp = GEO["shape"]
    flo, fla, fz = GEO["flo"], GEO["fla"], GEO["fz"]
    for ph, key in (("P", "vp"), ("S", "vs")):
        mus, ws = [], []
        for g in GEO["geo"]:
            sel, st = g["sel"], g["tile"]["stage"]
            acc = np.zeros(int(sel.sum()))
            for n in names:
                gg = np.load(ROOT / kind / n / st / "run" / "Output_Files" / "model_geo.npz")
                m = dict(lon=gg["lon"], lat=gg["lat"], z=gg["z"], vp=gg["vp"], vs=gg["vs"])
                acc += mod_velocity(m, ph, flo[sel], fla[sel], fz[sel], clamp_surface=False)
            mu = np.full(flo.size, np.nan)
            mu[sel] = acc / max(N, 1)
            w = np.zeros(flo.size)
            w[sel] = g[f"w_{ph}"]
            mus.append(mu)
            ws.append(w)
        W = np.array(ws)
        MU = np.nan_to_num(np.array(mus))
        den = W.sum(0)
        mbar = np.where(den > 0, (W * MU).sum(0) / np.where(den > 0, den, 1), np.nan)
        dis = np.where(den > 0, np.sqrt(np.maximum((W * (MU - mbar) ** 2).sum(0) / np.where(den > 0, den, 1), 0)), 0.0)
        dis = np.where((W > 0).sum(0) >= 2, dis, 0.0)
        S[f"tile_disagreement_{ph.lower()}"] = dis.reshape(shp)
        S[f"{'Vp' if ph == 'P' else 'Vs'}_tomography_std_total"] = np.sqrt(
            S[f"{'Vp' if ph == 'P' else 'Vs'}_tomography_std"] ** 2 + S[f"tile_disagreement_{ph.lower()}"] ** 2)
    import xarray as xr
    nd = xr.open_dataset(Path(T["models"]["store"]) / T["models"]["name"] / f"v{CFG['node_model']}.nc")
    f = E["calibration"]["member_frac"]
    S["resolved_p"] = ((nd.resolution_p.values.transpose(2, 1, 0) > 0) & (M["r_p"] > 0).all(0)).astype("int8")
    S["resolved_s"] = ((nd.resolution_s.values.transpose(2, 1, 0) > 0) & (M["r_s"] > 0).all(0)).astype("int8")
    S["sampled_p"] = (((M["dws_p"] >= 100).mean(0) >= f) & (M["r_p"] > 0).all(0)).astype("int8")
    S["sampled_s"] = (((M["dws_s"] >= 100).mean(0) >= f) & (M["r_s"] > 0).all(0)).astype("int8")
    S["medium"] = nd.medium.values.transpose(2, 1, 0).astype("int8")
    np.savez_compressed(out / "stats.npz", **{k: np.asarray(v, "float32") if v.dtype.kind == "f" else v
                                              for k, v in S.items()})
    runs = []
    for n in names:
        rec = dict(member=n, draws=json.loads((ROOT / kind / n / "draws.json").read_text()))
        for st in C.STAGES:
            rec[st] = json.loads((ROOT / kind / n / st / "DONE").read_text())
        rec["merged"] = json.loads((ROOT / kind / n / "MERGED").read_text())
        runs.append(rec)
    rock = S["medium"] == 0
    summ = dict(kind=kind, members=N, names=names, config=C.CONFIG, created=time.strftime("%F %T"),
                vp_std_rock_median=float(np.nanmedian(S["Vp_std"][rock])),
                vs_std_rock_median=float(np.nanmedian(S["Vs_std"][rock])),
                vp_std_resolved_median=float(np.nanmedian(S["Vp_std"][S["resolved_p"] > 0])) if S["resolved_p"].any() else None,
                vs_std_resolved_median=float(np.nanmedian(S["Vs_std"][S["resolved_s"] > 0])) if S["resolved_s"].any() else None,
                resolved_nodes=[int(S["resolved_p"].sum()), int(S["resolved_s"].sum())],
                sampled_nodes=[int(S["sampled_p"].sum()), int(S["sampled_s"].sum())],
                tile_disagreement_p_median_overlap=float(np.median(S["tile_disagreement_p"][S["tile_disagreement_p"] > 0]))
                if (S["tile_disagreement_p"] > 0).any() else None)
    if C.FLOOR:                              # D-B083 floor per stage over the members (DONE records)
        fl = {}
        for st in C.STAGES:
            recs = [r[st].get("floor") or {} for r in runs]
            fl[st] = {k: [int(np.min([x.get(k, 0) for x in recs])), float(np.median([x.get(k, 0) for x in recs])),
                          int(np.max([x.get(k, 0) for x in recs]))]
                      for k in ("lines", "events_hit", "end_on_floor", "start_above_floor", "above_floor_by_more_than_2m",
                                "not_met", "air_quake_lines")}
        summ["floor_min_median_max"] = fl
    C.jdump(out / "stats.json", summ)
    C.jdump(out / "runs.json", runs)
    print(json.dumps(summ, indent=1, default=str)[:2000])


# ---------------------------------------------------------------- product + report
VAR_DOC = {
    "mean": "ensemble mean", "median": "ensemble median", "std": "ensemble standard deviation",
    "p05": "5 % quantile of the ensemble", "p95": "95 % quantile of the ensemble", "n": "number of members with a value"}
QDOC = {"Vp": ("P-wave velocity (v0.3.2 convention: water 1.5, air NaN)", "km/s"),
        "Vs": ("S-wave velocity with the D-B073 node rule (drawn Vp/Vs and thickness); NaN in water and air", "km/s"),
        "VpVs": ("Vp/Vs ratio (NaN above the solid surface)", "1"),
        "Vp_tomography": ("P-wave velocity of the inversions (no surface, no rule)", "km/s"),
        "Vs_tomography": ("S-wave velocity of the inversions (no surface, no rule)", "km/s")}


INFLATED = {"Vp": ("Vp", "Vp_tomography"), "Vs": ("Vs", "Vs_tomography")}   # D-B094 factor -> quantities it scales
SPREAD = ("std", "p05", "p95")
RULE = ("D-B094: spreads multiplied by f about the ensemble median (std -> f std; p05 / p95 -> median + f (p - median)); "
        "f = max(f68, f95) on the resolved nodes of the synthetic calibration, rounded up to the next 0.05, f68 / f95 = "
        "the smallest factor bringing the 16-84 % / 2.5-97.5 % coverage of the truth to 0.68 / 0.95 "
        "(b2_ens_inflate.py); hypocentres and Vp/Vs not inflated; raw member statistics kept as <name>_raw")


def inflate(S):
    """D-B094 calibration of the spreads in place (configs.<c>.calibration.inflation, factor f per Vp / Vs): for Vp, Vs
    (node rule) and Vp_tomography, Vs_tomography, std -> f std and p05 / p95 -> median + f (p - median); the tomography
    std_total recomputed from the inflated std and the (not inflated) tile disagreement. The raw values are kept as
    <name>_raw. Mean, median, n and VpVs are not changed. Returns {quantity: factor} inflated ({} = nothing)."""
    applied = {}
    for base, f in C.inflation().items():
        for q in INFLATED[base]:
            med = S[f"{q}_median"]
            for st in SPREAD:
                S[f"{q}_{st}_raw"] = S[f"{q}_{st}"]
            S[f"{q}_std"] = f * S[f"{q}_std_raw"]
            for st in ("p05", "p95"):
                S[f"{q}_{st}"] = med + f * (S[f"{q}_{st}_raw"] - med)
            applied[q] = f
        k = f"{base}_tomography_std_total"
        S[f"{k}_raw"] = S[k]
        S[k] = np.sqrt(S[f"{base}_tomography_std"] ** 2 + S[f"tile_disagreement_{base[-1].lower()}"] ** 2)
    return applied


def calibration_info(applied):
    """What the product's spreads are calibrated with: source calibration (root, coverage.json md5 and result), the
    accepted decision and factors (configs.<c>.calibration), the derivation file ROOT/calibration/inflation.json
    (md5, its factors, whether they equal the configuration's, the resolved-node coverages before / after)."""
    cr = C.calibration_root()
    cj = cr / "synth" / "coverage.json"
    acc = C.calibration_entry()
    info = dict(source_root=str(cr), coverage_json=str(cj), coverage_json_md5=C.md5(cj) if cj.exists() else None,
                result=json.loads(cj.read_text()).get("result") if cj.exists() else None)
    if acc:
        info.update(decision=acc.get("accepted"), inflation=C.inflation(), applied_to=applied, rule=RULE,
                    coverage_md5_accepted=acc.get("coverage_md5"),
                    not_inflated="mean, median, n, VpVs, hypocentres (not in this product)")
    der = ROOT / "calibration" / "inflation.json"
    if der.exists():
        d = json.loads(der.read_text())
        info["derivation"] = dict(
            file=str(der), md5=C.md5(der), factors=d["factors"],
            matches_configuration=all(abs(float(d["factors"].get(q, np.nan)) - f) < 1e-9 for q, f in C.inflation().items()),
            resolved={q: {k: d["items"][f"resolved_{q}"].get(k) for k in ("n", "cov68_raw", "cov95_raw", "f68", "f95",
                                                                           "factor", "after_used_factor")}
                      for q in ("Vp", "Vs")})
    return info


def dataset(kind):
    import xarray as xr
    import b2_model_export as EX
    import b2_merge as BM
    S = dict(np.load(ROOT / "stats" / kind / "stats.npz"))
    summ = json.loads((ROOT / "stats" / kind / "stats.json").read_text())
    applied = inflate(S)
    summ["calibration"] = calibration_info(applied)
    lon, lat, z = BM.grid_axes()
    tr = lambda a: np.ascontiguousarray(np.asarray(a).transpose(2, 1, 0))
    src = f"b2_ens_stats.py, {summ['members']} members"
    V = {}
    for k, a in S.items():
        q, _, stat = k.rpartition("_")
        if q in QDOC and stat in VAR_DOC:
            ln, un = QDOC[q]
            at = dict(long_name=f"{VAR_DOC[stat]} of {ln}", display_name=f"{q} {stat}",
                      units=("1" if stat == "n" else un), source=src)
            if q in applied and stat in SPREAD:
                f = applied[q]
                at.update(long_name=f"{at['long_name']}, calibrated: x {f:.2f} about the ensemble median (D-B094)",
                          calibration=f"D-B094 inflation factor {f:.2f} of the member spread (raw: {q}_{stat}_raw)",
                          inflation_factor=f)
            V[f"{q}_{stat}"] = (tr(a), at)
    for q, f in applied.items():                 # raw (uninflated) member statistics, kept beside the calibrated ones
        ln, un = QDOC[q]
        for st in SPREAD:
            V[f"{q}_{st}_raw"] = (tr(S[f"{q}_{st}_raw"]), dict(
                long_name=f"{VAR_DOC[st]} of {ln}, raw member spread (not calibrated)", display_name=f"{q} {st} raw",
                units=un, source=src, calibration=f"none (the calibrated value, x {f:.2f}, is {q}_{st})"))
    extra = {"prob_increase_p": "share of members with Vp above their own starting model",
             "prob_increase_s": "share of members with Vs above their own starting model",
             "tile_disagreement_p": "spread of the tiles' ensemble means in overlaps, Vp (P-M5)",
             "tile_disagreement_s": "spread of the tiles' ensemble means in overlaps, Vs (P-M5)",
             "Vp_tomography_std_total": "sqrt(std^2 + tile disagreement^2), Vp_tomography",
             "Vs_tomography_std_total": "sqrt(std^2 + tile disagreement^2), Vs_tomography",
             "resolved_p": "1 = resolved by the checkerboard tests (D-B018) and covered by a tile in every member, P",
             "resolved_s": "1 = resolved by the checkerboard tests (D-B018) and covered by a tile in every member, S",
             "sampled_p": "1 = DWS P >= 100 in >= 90 % of the members", "sampled_s": "1 = DWS S >= 100 in >= 90 % of the members",
             "medium": "0 rock, 1 water, 2 air (node model)"}
    for b in ("Vp", "Vs"):
        k = f"{b}_tomography_std_total"
        if f"{k}_raw" in S:
            f = applied[f"{b}_tomography"]
            extra[k] = f"sqrt(std^2 + tile disagreement^2), {b}_tomography, std calibrated x {f:.2f} (D-B094)"
            extra[f"{k}_raw"] = f"sqrt(raw std^2 + tile disagreement^2), {b}_tomography (not calibrated)"
    for k, ln in extra.items():
        un = "km/s" if ("disagreement" in k or "std_total" in k) else "1"
        V[k] = (tr(S[k]), dict(long_name=ln, display_name=k, units=un, source="b2_ens_stats.py"))
    ds = xr.Dataset({k: (("depth", "latitude", "longitude"), a, at) for k, (a, at) in V.items()},
                    coords=EX.coords(z, lat, lon))
    return ds, summ


def fig_png(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def cal_html(cal):
    """Report section of the uncertainty calibration (D-B094): decision, factors, rule, source, derivation table."""
    if not cal.get("applied_to"):
        return ["<h2>Uncertainty calibration</h2><p>None: the spreads are the raw member spreads (source calibration "
                f"{html_escape(str(cal.get('coverage_json')))}, result {cal.get('result')}).</p>"]
    f = cal["inflation"]
    out = ["<h2>Uncertainty calibration (" + html_escape(str(cal["decision"])) + ")</h2>",
           f"<p><b>Vp and Vs spreads multiplied by {f.get('Vp', 1):.2f} (Vp) and {f.get('Vs', 1):.2f} (Vs)</b>, applied to "
           f"{', '.join(f'{q} (x {v:.2f})' for q, v in cal['applied_to'].items())}; the raw member statistics are in "
           f"the variables *_raw. Not inflated: {html_escape(cal['not_inflated'])}.</p>",
           f"<p>Rule: {html_escape(cal['rule'])}.</p>",
           f"<p>Source: synthetic calibration {html_escape(cal['coverage_json'])} (md5 {cal['coverage_json_md5']}, "
           f"result {cal['result']}; accepted md5 {cal['coverage_md5_accepted']}).</p>"]
    d = cal.get("derivation")
    if not d:
        return out + ["<p>No derivation file (ROOT/calibration/inflation.json, b2_ens_inflate.py).</p>"]
    out.append(f"<p>Derivation: {html_escape(d['file'])} (md5 {d['md5']}), factors {json.dumps(d['factors'])}"
               + ("" if d["matches_configuration"] else
                  " <b style='color:#b00'>differ from the configuration's factors used here</b>") + ".</p>")
    out.append("<table><tr><th>resolved nodes</th><th>nodes</th><th>68 % raw</th><th>95 % raw</th><th>f68</th>"
               "<th>f95</th><th>factor (rounded up)</th><th>68 % after</th><th>95 % after</th></tr>")
    for q, r in d["resolved"].items():
        a = r.get("after_used_factor") or {}
        out.append(f"<tr><td>{q}</td><td>{r['n']}</td><td>{r['cov68_raw']}</td><td>{r['cov95_raw']}</td>"
                   f"<td>{r['f68']:.3f}</td><td>{r['f95']:.3f}</td><td>{r['factor']:.2f}</td><td>{a.get('cov68')}</td>"
                   f"<td>{a.get('cov95')}</td></tr>")
    md = Path(d["file"]).with_suffix(".md")
    out += ["</table>"] + ([f"<details><summary>All items (sampled nodes, per tile, Gaussian)</summary><pre>"
                            f"{html_escape(md.read_text())}</pre></details>"] if md.exists() else [])
    return out


def report(kind, ds, summ, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imgs = []
    cal = summ.get("calibration") or {}
    fac = cal.get("applied_to") or {}
    lab = lambda q: f"{q} std, calibrated x {fac[q]:.2f}" if q in fac else f"{q} std"
    depths = [3.0, 7.0, 15.0, 25.0]
    for q in ("Vp", "Vs", "VpVs"):
        fig, axs = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 4.6))
        for ax, zz in zip(axs, depths):
            sl = ds[f"{q}_std"].sel(depth=zz, method="nearest")
            sl.where(ds.medium.sel(depth=zz, method="nearest") == 0).plot(ax=ax, cmap="viridis", robust=True,
                                                                           cbar_kwargs=dict(shrink=0.7, label=lab(q)))
            ax.set_title(f"{lab(q)}, {float(sl.depth):g} km")
            ax.set_aspect(1 / np.cos(np.radians(45)))
            ax.set_facecolor("0.85")
        imgs.append((f"{q}: ensemble standard deviation at depth{' (D-B094 calibrated: raw member std x ' + format(fac[q], '.2f') + ')' if q in fac else ' (not inflated)'}"
                     " (grey: water, air or no value)", fig_png(fig)))
    secs = {"E-W at 44.5 N": dict(latitude=44.5), "E-W at 47.5 N": dict(latitude=47.5), "N-S at 125.0 W": dict(longitude=-125.0)}
    for nm, sel in secs.items():
        fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
        for ax, q in zip(axs, ("Vp", "Vs")):
            s = ds[f"{q}_std"].sel(**sel, method="nearest").sel(depth=slice(-3, 60))
            s.plot(ax=ax, cmap="viridis", robust=True, yincrease=False, cbar_kwargs=dict(label=f"{lab(q)} (km/s)"))
            ax.set_title(f"{lab(q)}, {nm}")
        imgs.append((f"Sections of the standard deviation, {nm}{' (calibrated, D-B094)' if fac else ''}", fig_png(fig)))
    runs = json.loads((ROOT / "stats" / kind / "runs.json").read_text())
    rows = []
    for st in C.STAGES:
        w = [r[st].get("wall_min") or np.nan for r in runs]
        m = [r[st].get("max_rss_gb") or np.nan for r in runs]
        k = [r[st].get("kept_mb") or np.nan for r in runs]
        rows.append(f"<tr><td>{st}</td><td>{np.nanmedian(w):.1f}</td><td>{np.nanmax(w):.1f}</td><td>{np.nanmax(m):.2f}</td>"
                    f"<td>{np.nanmean(k):.0f}</td></tr>")
    cov = C.calibration_root() / "synth" / "coverage.md"
    covtxt = cov.read_text() if cov.exists() else "(no calibration run)"
    frows = []
    for st, v in (summ.get("floor_min_median_max") or {}).items():
        frows.append(f"<tr><td>{st}</td>" + "".join(f"<td>{v[k][0]} / {v[k][1]:g} / {v[k][2]}</td>" for k in
                                                     ("lines", "events_hit", "end_on_floor", "start_above_floor",
                                                      "above_floor_by_more_than_2m")) + "</tr>")
    ftab = (["<h2>Floor (D-B083) per stage: min / median / max over the members</h2><table><tr><th>stage</th>"
             "<th>floor stops</th><th>events hit</th><th>end on the floor</th><th>start above the floor</th>"
             "<th>above the floor by &gt; 2 m at the end (errors)</th></tr>", *frows, "</table>"] if frows else [])
    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>Ensemble {product_label(summ['members'])}</title>",
            "<style>body{font-family:sans-serif;max-width:1300px;margin:auto;padding:16px;background:#fff;color:#111}"
            "img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 8px}"
            "pre{white-space:pre-wrap;background:#f4f4f4;padding:8px}</style></head><body>",
            f"<h1>Bayesian ensemble of model v{C.product_version()} (configuration {C.CONFIG}; {kind}, "
            f"{summ['members']} members)</h1>",
            f"<p>Created {summ['created']}. Product {product_label(summ['members'])}. Median std on rock nodes: Vp "
            f"{summ['vp_std_rock_median']:.3f} km/s, Vs {summ['vs_std_rock_median']:.3f} km/s; resolved nodes P/S "
            f"{summ['resolved_nodes']}; sampled nodes P/S {summ['sampled_nodes']} (raw member spreads; calibrated: x the "
            f"factors below).</p>", *cal_html(cal),
            "<h2>Run cost per member</h2><table><tr><th>stage</th><th>median wall (min)</th><th>max wall (min)</th>"
            "<th>max RSS (GB)</th><th>kept (MB, mean)</th></tr>", *rows, "</table>", *ftab,
            f"<h2>Synthetic calibration ({cov})</h2><pre>", html_escape(covtxt), "</pre>"]
    for cap, b in imgs:
        html += [f"<h3>{cap}</h3>", f"<img src='data:image/png;base64,{b}'>"]
    html.append("</body></html>")
    f = outdir / f"report_{kind}.html"
    f.write_text("\n".join(html))
    return f


def product_label(n_members):
    """Version label of the ensemble product: the configuration's model version (product_version before any '+')
    with '+ens.<number of merged members>' (D-B047 build metadata), e.g. 0.3.2+ens.97."""
    return f"{C.product_version()}+ens.{int(n_members)}"


def products(kind, dry):
    """Default: dry run (netCDF in ROOT/products or --dry-run DIR) + the HTML report. Registration of the immutable
    version only with B2_ENS_REGISTER=1 in the environment (needs the PI's go), through b2_model_export.export with the
    read-back on Vp_mean before the registry is touched."""
    import os
    import b2_model_export as EX
    ds, summ = dataset(kind)
    version = product_label(summ["members"])
    register = os.environ.get("B2_ENS_REGISTER") == "1" and not dry
    desc = (f"Bayesian (RTO) ensemble statistics of model v{C.product_version()} (configuration {C.CONFIG}): "
            f"{summ['members']} members of M0 + {', '.join(CFG['tiles'])} merged with the D-B049..D-B051 rules "
            f"(tomodd.merge.{CFG.get('merge_product', 'product')}), D-B073 node rule with drawn parameters"
            + ("; tomoDD's D-B081 floor in every inversion (D-B083)" if C.FLOOR else ""))
    cal = summ["calibration"]
    if cal.get("applied_to"):
        f = cal["inflation"]
        desc += (f"; Vp and Vs spreads (std, 5-95 % about the median) calibrated by the synthetic ensemble: x "
                 f"{f.get('Vp', 1):.2f} (Vp), x {f.get('Vs', 1):.2f} (Vs) ({cal['decision']}; raw spreads in *_raw)")
    cj = Path(cal["coverage_json"])
    prov = dict(run=f"ensemble {C.CONFIG} ({ROOT})", members=summ["members"], config=json.dumps(CFG),
                prior=json.dumps(C.PR, default=str), calibration=cj.read_text()[:2000] if cj.exists() else "none",
                uncertainty_calibration=json.dumps(cal, default=str))
    EX.add_attrs(ds, version if register else "0.0.0-dryrun", desc, prov, "b2_ens_stats.py")
    out = Path(dry) if dry else ROOT / "products"
    out.mkdir(parents=True, exist_ok=True)
    if register:
        EX.export(ds, version, desc, nll=False, status="intermediate", readback_var="Vp_mean")
    else:
        f = out / f"ensemble_{kind}_{version}_dryrun.nc"
        ds.to_netcdf(f, format="NETCDF4_CLASSIC", encoding={k: dict(zlib=True, complevel=4) for k in ds.data_vars})
        print(f"dry run (not registered; B2_ENS_REGISTER=1 registers v{version}):", f, f.stat().st_size)
    print("report:", report(kind, ds, summ, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "products"])
    ap.add_argument("kind")
    ap.add_argument("--dry-run")
    a = ap.parse_args()
    stats(a.kind) if a.cmd == "stats" else products(a.kind, a.dry_run)


if __name__ == "__main__":
    main()
