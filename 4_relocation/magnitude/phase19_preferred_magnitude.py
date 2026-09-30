"""Phase 19: preferred magnitude -- Hutton-Boore ML (with station terms) below M 4.5, Mw above.

ML (every event with >= 3 amplitudes):
    station ML = log10 A_WA(mm) + 1.110 log10(r/100) + 0.00189 (r - 100) + 3.0
    (Hutton & Boore 1987; r hypocentral km). The picks within RMAX km (P and S) are
    inverted jointly for an event term M_i and a station-phase term C_jp,
        station ML_ijp = M_i + C_jp,  sum_j C_jp = 0 per phase,
    by damped LSQR (stations with >= MIN_STA_OBS picks; one refit after rejecting
    residuals > REJECT_MAD scaled MADs). The distance correction is FIXED: the
    decay fitted by phase3 is too shallow beyond ~200 km and pulls large, distantly
    recorded events low (phase18). The station terms carry the site amplification
    (large at sediment-covered OBS); without them the event ML scatters by ~0.6.
    Events with >= 3 retained picks within RMAX km get ML_method "near"; one offset
    ties M_i to the ComCat ML anchors (slope fixed at 1). The others (recorded only
    far away, mostly far offshore) get the phase3 inversion ML mapped onto this scale
    by a Theil-Sen fit over the "near" events: ML_method "mapped". (Hutton-Boore
    magnitudes from their nearest picks were tried instead and rejected: they are on
    a different scale for these events and flatten the regional b-values, e.g.
    Blanco b 0.6.) Both routes are too low for large, far-recorded events; those
    with a ComCat tensor get its Mw below.
Preferred magnitude M (M_type):
    "Mw"      the matched ComCat moment tensor (phase18) has Mw >= M_SWITCH;
    "Mw_cal"  no such tensor, ML_method "near", and Mw_cal = ML + offset >= M_SWITCH.
              The offset is the median (tensor Mw - ML) of the "near" pairs with
              ML >= FIT_MIN (~0). A constant, not a fitted slope: the tensor sample
              stops at Mw ~4, and with ~0.45 scatter a fitted slope is diluted (~0.65);
    "ML"      otherwise (a tensor Mw < M_SWITCH is kept in Mw_mt, not used as M).
    "mapped" events are NOT calibrated: against tensors their ML is ~1.6 low, while
    their small events agree with ComCat (a small event is picked far away only when
    it is loud), so no single offset holds. Their ML above ~4 is a lower bound.
Uncertainty: "near" ML_unc = sqrt(SE^2 + s_anchor^2) with SE = 1.4826 MAD / sqrt(n) of
the event's station residuals and s_anchor the scatter about the anchor offset after
removing the mean SE^2 of the anchors; "mapped" sqrt((slope ML_inv_unc)^2 + s_map^2);
Mw: MW_MT_UNC; Mw_cal: std of (Mw - ML - offset) of its pairs.

Writes ../../data/magnitude/cascadia_catalog_M_routeA.csv (same events as
cascadia_catalog_ML_routeA.csv; ML_inv keeps the inversion ML).

Usage (default env, run from 4_relocation/magnitude, after phase18):
    python phase19_preferred_magnitude.py
"""
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import theilslopes
from phase18_moment_tensor_match import hutton_boore, RMAX

D = "../../data/magnitude"
M_SWITCH = 4.5
FIT_MIN = 3.5
MW_MT_UNC = 0.1
MIN_STA_OBS = 8
REJECT_MAD = 4.0


def station_ml():
    ds = pd.read_csv(f"{D}/amp_distance_dataset_routeA.csv",
                     usecols=["event_id", "station", "phase", "dist_hypo_km", "log10A"])
    ds["m"] = hutton_boore(ds.log10A, ds.dist_hypo_km)
    ds["sp"] = ds.station + "|" + ds.phase
    return ds


def invert(d):
    """station ML = M_i + C_jp (sum C = 0 per phase). Returns M (by event), C (by sp)
    and the per-pick residuals."""
    e, ev_ids = pd.factorize(d.event_id)
    j, sp_ids = pd.factorize(d.sp)
    n, ne, ns = len(d), len(ev_ids), len(sp_ids)
    ph = np.array([k.split("|")[1] for k in sp_ids])
    rows, cols, vals, rhs = [np.arange(n)] * 2, [e, ne + j], [np.ones(n)] * 2, list(d.m)
    for i, p in enumerate(("P", "S")):              # gauge rows
        sel = np.flatnonzero(ph == p)
        rows.append(np.full(len(sel), n + i)); cols.append(ne + sel)
        vals.append(np.full(len(sel), 1000.0)); rhs.append(0.0)
    A = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n + 2, ne + ns)).tocsr()
    x = lsqr(A, np.asarray(rhs), damp=1e-3, atol=1e-8, btol=1e-8, iter_lim=3000)[0]
    M, C = pd.Series(x[:ne], index=ev_ids), pd.Series(x[ne:], index=sp_ids)
    return M, C, d.m.to_numpy() - x[e] - x[ne + j]


def event_ml(ds):
    near = ds[ds.dist_hypo_km <= RMAX]
    cnt = near.sp.value_counts()
    near = near[near.sp.isin(cnt[cnt >= MIN_STA_OBS].index)]
    M, C, res = invert(near)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    near = near[np.abs(res) < REJECT_MAD * mad]
    M, C, res = invert(near)
    near = near.assign(res=res)
    g = near.groupby("event_id").res
    ev = pd.DataFrame({"m": M, "n": g.size(),
                       "se": 1.4826 * g.apply(lambda v: np.median(np.abs(v - v.median()))) / np.sqrt(g.size())})
    ev = ev[ev.n >= 3].assign(ML_method="near")
    print(f"inversion within {RMAX:.0f} km: {len(near):,} picks, {len(C):,} station-phase terms, "
          f"residual robust std {1.4826 * np.median(np.abs(res - np.median(res))):.3f}")
    return ev


def main():
    cat = pd.read_csv(f"{D}/cascadia_catalog_ML_routeA.csv")
    ev = event_ml(station_ml())
    an = pd.read_csv(f"{D}/route_b_ml_anchors_routeA.csv")[["event_id", "ml"]].join(ev, on="event_id", how="inner")
    off = np.median(an.ml - an.m)
    s_anchor = np.sqrt(max(np.var(an.ml - an.m - off) - np.mean(an.se ** 2), 0))
    ev["ML"] = ev.m + off
    ev["ML_unc"] = np.sqrt(ev.se ** 2 + s_anchor ** 2)
    print(f"ML offset to {len(an):,} ComCat anchors {off:+.3f}; anchor residual std "
          f"{np.std(an.ml - an.m - off):.3f} (s_anchor {s_anchor:.3f})")

    out = cat.rename(columns={"ML": "ML_inv", "ML_unc": "ML_inv_unc"})[
        ["event_id", "otime", "evla", "evlo", "evdp", "ML_inv", "ML_inv_unc"]]
    out = out.join(ev[["ML", "ML_unc", "n", "ML_method"]].rename(columns={"n": "n_ML"}), on="event_id")
    nr = out[out.ML_method == "near"]
    sl, ic, _, _ = theilslopes(nr.ML, nr.ML_inv)
    s_map = np.std(nr.ML - ic - sl * nr.ML_inv)
    miss = out.ML.isna()
    out.loc[miss, "ML"] = ic + sl * out.ML_inv[miss]
    out.loc[miss, "ML_unc"] = np.sqrt((sl * out.ML_inv_unc[miss]) ** 2 + s_map ** 2)
    out.loc[miss, "ML_method"] = "mapped"
    print(f"mapped (no 3 picks within {RMAX:.0f} km): ML = {ic:.3f} + {sl:.3f} ML_inv "
          f"(fit over near events, residual std {s_map:.2f})")
    print("ML_method:", out.ML_method.value_counts().to_dict())

    mt = pd.read_csv("../../data/focal/comcat_mt_matched.csv")[["event_id", "comcat_id", "Mw_mt"]]
    mt = mt.sort_values("Mw_mt", ascending=False).drop_duplicates("event_id")
    out = out.merge(mt, on="event_id", how="left")
    pairs = out[out.Mw_mt.notna() & (out.ML >= FIT_MIN)]
    off_m, std_m = {}, {}
    for meth, g in pairs.groupby("ML_method"):
        d = g.Mw_mt - g.ML
        off_m[meth], std_m[meth] = d.median(), np.std(d - d.median())
        bb, aa, _, _ = theilslopes(g.Mw_mt, g.ML)
        print(f"Mw_cal ({meth}) = ML {off_m[meth]:+.2f}  ({len(g)} tensor pairs with ML >= {FIT_MIN}; "
              f"std {std_m[meth]:.2f}; a fitted Theil-Sen slope would be {bb:.2f})")
    out["M"], out["M_unc"], out["M_type"] = out.ML, out.ML_unc, "ML"
    cal = out.ML + out.ML_method.map(off_m)
    use_mt = out.Mw_mt >= M_SWITCH
    use_cal = ~use_mt & (out.ML_method == "near") & (cal >= M_SWITCH)
    out.loc[use_mt, "M"] = out.Mw_mt[use_mt]
    out.loc[use_mt, "M_unc"] = MW_MT_UNC
    out.loc[use_mt, "M_type"] = "Mw"
    out.loc[use_cal, "M"] = cal[use_cal]
    out.loc[use_cal, "M_unc"] = out.ML_method[use_cal].map(std_m)
    out.loc[use_cal, "M_type"] = "Mw_cal"
    print("M_type:", out.M_type.value_counts().to_dict(), "; uncalibrated 'mapped' ML >= 4:",
          int(((out.M_type == "ML") & (out.ML_method == "mapped") & (out.ML >= 4)).sum()))

    t = out[out.Mw_mt.notna()]
    for lo, hi in [(3, 4.5), (4.5, 5.5), (5.5, 7.5)]:
        s = t[t.Mw_mt.between(lo, hi, inclusive="left")]
        print(f"  tensor Mw {lo}-{hi}: n={len(s):3d}  ML - Mw {np.median(s.ML - s.Mw_mt):+.2f}  "
              f"(inversion ML {np.median(s.ML_inv - s.Mw_mt):+.2f})")
    d = out.ML - out.ML_inv
    print(f"ML - ML_inv: median {d.median():+.2f}, std {d.std():.2f}; ML median {out.ML.median():.2f}")
    cols = ["event_id", "otime", "evla", "evlo", "evdp", "M", "M_unc", "M_type", "ML", "ML_unc",
            "n_ML", "ML_method", "Mw_mt", "comcat_id", "ML_inv", "ML_inv_unc"]
    out[cols].round({"M": 3, "M_unc": 3, "ML": 3, "ML_unc": 3, "ML_inv": 3, "ML_inv_unc": 3}).to_csv(
        f"{D}/cascadia_catalog_M_routeA.csv", index=False)
    print(f"wrote {D}/cascadia_catalog_M_routeA.csv ({len(out):,} events)")


if __name__ == "__main__":
    main()
