"""Forward-accuracy test, step 4: merge all forward times of the subset, build the references and split the bias.

Inputs: OUT/subset/picks.csv (t_nll_prod = production NLL grids), OUT/nll/*.csv (t_<variant>, incl. the TauP
1D reference), OUT/tomodd_runs/*/pred.csv (t_tdd_<run>).

References ("exact" times):
* 1D: TauP (b2_fwd_taup.py).
* 3D: Richardson extrapolation of the cell-centred, Earth-flattened Grid2Time times at 0.5 and 0.25 km,
  T0 = T025 - k (T05 - T025), with k calibrated on the 1D model against TauP (median), checked there.
  ref_orig = v4-NLL3D model as a continuous field (trilinear between the NLL nodes, 'origlin');
  ref_mod  = the tomoDD MOD of check01 (inputs/MOD, tomoDD's own trilinear lookup);
  ref_<v>  = other MODs: T05(<v>) - [T05(mod) - ref_mod] (FD error transferred from the MOD).

Split of the finding (all differences a - b in s, per pick, then medians by phase x station type x distance):
  finding = tomoDD(MOD) - NLL production
          = [tomoDD(MOD) - ref_mod]          (b) tomoDD: sphere coordinates + ray tracer
          + [ref_mod - ref_orig]             (a) MOD sampling of the NLL model
          - [NLL production - ref_orig]      (c) NLL: half-cell shift + flat Earth + Podvin-Lecomte FD error

  python b2_fwd_compare.py   -> OUT/compare/{decomposition,options,station_corr}.md/.csv, all_times.csv
"""
import numpy as np
import pandas as pd

from b2_fwd_common import OUT, bias_table, md_table, summary_row

KEY = ["event_ID", "code", "phase"]


def load():
    p = pd.read_csv(OUT / "subset/picks.csv")
    for f in sorted((OUT / "nll").glob("*.csv")):
        p = p.merge(pd.read_csv(f), on=KEY, how="left")
    for d in sorted((OUT / "tomodd_runs").glob("*/pred.csv")):
        t = pd.read_csv(d)[KEY + ["t_tomodd"]].rename(columns={"t_tomodd": f"t_tdd_{d.parent.name}"})
        p = p.merge(t, on=KEY, how="left")
    return p


def references(p, log):
    # k from the 1D model: TauP = T025 - k (T05 - T025)
    k = {}
    for ph in ("P", "S"):
        m = (p.phase == ph) & np.isfinite(p.t_oned_c025_eft) & np.isfinite(p.t_taup_oned_ell)
        num = (p.t_oned_c025_eft - p.t_taup_oned_ell)[m]
        den = (p.t_oned_c05_eft - p.t_oned_c025_eft)[m]
        ok = den.abs() > 0.005
        k[ph] = float(np.median(num[ok] / den[ok]))
    kk = p.phase.map(k)
    p["ref_oned"] = p.t_oned_c025_eft - kk * (p.t_oned_c05_eft - p.t_oned_c025_eft)
    p["ref_orig"] = p.t_origlin_c025_eft - kk * (p.t_origlin_c05_eft - p.t_origlin_c025_eft)
    p["ref_mod"] = p.t_mod_c025_eft - kk * (p.t_mod_c05_eft - p.t_mod_c025_eft)
    fd_mod = p.t_mod_c05_eft - p.ref_mod
    for v in ("slow5", "slow11", "point", "pointz2", "slow5z2", "pointz1", "slow5z"):
        if f"t_{v}_c05_eft" in p:
            p[f"ref_{v}"] = p[f"t_{v}_c05_eft"] - fd_mod
    e = p.ref_oned - p.t_taup_oned_ell
    log.append(f"Richardson factor k: P {k['P']:.3f}, S {k['S']:.3f} (1/(2^p - 1); p = {np.log2(1 + 1 / k['P']):.2f} / "
               f"{np.log2(1 + 1 / k['S']):.2f})")
    for ph in ("P", "S"):
        x = e[(p.phase == ph)].dropna()
        log.append(f"1D check, extrapolated NLL - TauP, {ph}: median {x.median():+.4f} s, p95 |d| "
                   f"{np.percentile(x.abs(), 95):.4f} s (n {len(x)})")
    return p


DECOMP = [
    ("finding: tomoDD(MOD) - NLL production", "t_tdd_mod_check01", "t_nll_prod"),
    ("(b) tomoDD(MOD) - ref_mod", "t_tdd_mod_check01", "ref_mod"),
    ("(b1) sphere coordinates: tomoDD(MOD) - tomoDD(MOD, WGS-84-scaled coords)", "t_tdd_mod_check01", "t_tdd_mod_ellcoord"),
    ("(b2) ray tracer: tomoDD(MOD, scaled coords) - ref_mod", "t_tdd_mod_ellcoord", "ref_mod"),
    ("(a) MOD sampling: ref_mod - ref_orig", "ref_mod", "ref_orig"),
    ("(c) NLL production - ref_orig", "t_nll_prod", "ref_orig"),
    ("(c1) half-cell shift: production - cell-centred (1 km, flat)", "t_nll_prod", "t_orig_c1"),
    ("(c2) flat Earth: cell-centred 1 km flat - EFT", "t_orig_c1", "t_orig_c1_eft"),
    ("(c3) FD error (1 km, blocky): cell-centred 1 km EFT - ref_orig", "t_orig_c1_eft", "ref_orig"),
    ("1D: tomoDD - TauP (sphere distances): ray tracer", "t_tdd_oned", "t_taup_oned_sph"),
    ("1D: tomoDD - TauP (WGS-84 distances)", "t_tdd_oned", "t_taup_oned_ell"),
    ("1D: tomoDD(scaled coords) - TauP (WGS-84)", "t_tdd_oned_ellcoord", "t_taup_oned_ell"),
    ("1D: NLL 0.5 km EFT - TauP", "t_oned_c05_eft", "t_taup_oned_ell"),
    ("1D: NLL 0.25 km EFT - TauP", "t_oned_c025_eft", "t_taup_oned_ell"),
    ("1D: NLL 0.5 km flat - TauP", "t_oned_c05", "t_taup_oned_ell"),
    ("1D: NLL extrapolated - TauP", "ref_oned", "t_taup_oned_ell"),
]

OPTIONS = [
    # MOD sampling variants (against the continuous NLL model)
    ("MOD as built (velocity mean, 11 km box, inputs/MOD)", "ref_mod", "ref_orig"),
    ("MOD slowness mean 11 km", "ref_slow11", "ref_orig"),
    ("MOD slowness mean 5 km", "ref_slow5", "ref_orig"),
    ("MOD nearest column (no box)", "ref_point", "ref_orig"),
    ("MOD slowness 5 km + 1 km nodes to 9 km", "ref_slow5z", "ref_orig"),
    ("MOD slowness 5 km, 1 km nodes to 9 km, 2 km to 49 km", "ref_slow5z2", "ref_orig"),
    ("MOD nearest column, 1 km nodes to 9 km, 2 km to 49 km", "ref_pointz2", "ref_orig"),
    ("MOD nearest column, 1 km nodes everywhere (NLL only)", "ref_pointz1", "ref_orig"),
    # tomoDD in candidate set-ups (tracer + coordinates + sampling) against the continuous NLL model
    ("tomoDD as check01", "t_tdd_mod_check01", "ref_orig"),
    ("tomoDD, MOD as built, scaled coords", "t_tdd_mod_ellcoord", "ref_orig"),
    ("tomoDD, nearest column, scaled coords", "t_tdd_point_ellcoord", "ref_orig"),
    ("tomoDD, slowness 5 km + 1/2 km nodes, scaled coords", "t_tdd_slow5z2_ellcoord", "ref_orig"),
    ("tomoDD, nearest column + 1/2 km nodes, scaled coords (proposed)", "t_tdd_pointz2_ellcoord", "ref_orig"),
    ("tomoDD ray tracer only, proposed MOD: tomoDD - ref_pointz2", "t_tdd_pointz2_ellcoord", "ref_pointz2"),
    # NLL set-ups against the continuous NLL model
    ("NLL production (node value on cell [k,k+1], 1 km, flat)", "t_nll_prod", "ref_orig"),
    ("NLL cell-centred 1 km, flat", "t_origlin_c1", "ref_orig"),
    ("NLL cell-centred 1 km, EFT", "t_origlin_c1_eft", "ref_orig"),
    ("NLL cell-centred 0.5 km, flat", "t_origlin_c05", "ref_orig"),
    ("NLL cell-centred 0.5 km, EFT", "t_origlin_c05_eft", "ref_orig"),
    ("NLL cell-centred 0.25 km, EFT", "t_origlin_c025_eft", "ref_orig"),
    # end to end: proposed tomoDD against NLL set-ups
    ("tomoDD proposed - NLL production", "t_tdd_pointz2_ellcoord", "t_nll_prod"),
    ("tomoDD proposed - NLL cell-centred 1 km EFT", "t_tdd_pointz2_ellcoord", "t_origlin_c1_eft"),
    ("tomoDD proposed - NLL cell-centred 0.5 km EFT", "t_tdd_pointz2_ellcoord", "t_origlin_c05_eft"),
]


def run_set(p, items, tag, out):
    rows, md = [], []
    for name, a, b in items:
        if a not in p or b not in p:
            continue
        p["d"] = p[a] - p[b]
        if p.d.notna().sum() == 0:
            continue
        rows.append({"comparison": name, "n": int(p.d.notna().sum()), **summary_row(p, "d")})
        md += [f"#### {name}", "", f"`{a}` - `{b}`", "", md_table(bias_table(p, "d").round(3).reset_index()), ""]
    s = pd.DataFrame(rows)
    s.to_csv(out / f"{tag}.csv", index=False)
    (out / f"{tag}_tables.md").write_text("\n".join(md))
    return s


def station_correction(p, a, b, out):
    """Per station-phase median of (a - b) as a correction; residual after it, by phase x type x bin."""
    p["d"] = p[a] - p[b]
    med = p.groupby(["station", "phase"]).d.transform("median")
    p["d_corr"] = p.d - med
    tab = p.groupby(["station", "phase"]).d.median().unstack().round(3)
    tab.to_csv(out / "station_corr.csv")
    return summary_row(p, "d_corr"), tab


def wide(s):
    t = s.set_index("comparison").drop(columns="n")
    return t


def main():
    p = load()
    out = OUT / "compare"
    out.mkdir(exist_ok=True)
    log = []
    p = references(p, log)
    s1 = run_set(p, DECOMP, "decomposition", out)
    s2 = run_set(p, OPTIONS, "options", out)
    sc, tab = station_correction(p, "t_tdd_pointz2_ellcoord", "ref_pointz2", out)
    p.drop(columns=["d", "d_corr"], errors="ignore").to_csv(out / "all_times.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    print("\n".join(log))
    for s in (s1, s2):
        w = wide(s)
        print(w[[c for c in w.columns if c.startswith("P")]].to_string())
        print(w[[c for c in w.columns if c.startswith("S")]].to_string())
    print("tomoDD tracer error per station (proposed MOD), median s:\n", tab.to_string())
    print("after a per station-phase correction:", sc)
    (out / "log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
