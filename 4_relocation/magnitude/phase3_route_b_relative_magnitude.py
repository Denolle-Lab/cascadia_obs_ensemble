#!/usr/bin/env python3
"""
Route B (Phase 3): relative magnitudes from counts via a joint linear inversion.

Solves, jointly over all events / stations / phases:

    log10(A_ijp) = M_i + C_{j,p} - [ n_p * log10(r_ij / r_ref) + k_p * r_ij ] + eps

where
    M_i        per-event RELATIVE magnitude (unknown; NOT yet calibrated to ML)
    C_{j,p}    per-(station, phase) term -- absorbs the (in-band) instrument
               response gain + site amplification. This is the "relative station
               magnitude" correction; because amplitudes are counts, this term is
               what makes stations comparable (per the review on issue #10). Terms
               are created ONLY for (station, phase) pairs that are actually
               observed (no phantom, data-less columns).
    n_p, k_p   phase-specific geometric + anelastic attenuation
    r          hypocentral distance (km)

Gauge: sum_j C_{j,p} = 0 within each phase => M_i is on a definite RELATIVE scale.
Absolute ML calibration (ComCat/Morton anchor) is a SEPARATE downstream step.

Amplitudes are raw counts (instrument response NOT removed -- confirmed in
calculate_amplitudes.py); this is the response-free relative route and does NOT
by itself yield absolute ML.

Usage:
    python phase3_route_b_relative_magnitude.py \
        --dataset ../../data/magnitude/amp_distance_dataset.csv \
        --outdir  ../../data/magnitude
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

R_REF_KM = 100.0


def solve(df, damp, gauge_w, fix_n=None):
    """Assemble the sparse design matrix (station-phase terms only for observed
    pairs) and solve with LSQR. Returns a result dict aligned to df's row order.

    fix_n : if None, solve the geometric spreading exponent n_p together with the
        anelastic k_p (may yield unphysical k<0 due to n-k collinearity + the
        magnitude-distance selection bias). If a float, FIX n_p = fix_n and solve
        only k_p -- pinning the (shallower) geometric term forces the real distance
        decay into the anelastic term, so k comes out positive and physical."""
    ev_ids = np.sort(df["event_id"].unique())
    ev_idx = pd.factorize(pd.Categorical(df["event_id"], categories=ev_ids))[0]
    n_ev = len(ev_ids)

    sp_key = df["station"].astype(str) + "|" + df["phase"].astype(str)
    sp_ids = np.sort(sp_key.unique())
    sp_idx = pd.factorize(pd.Categorical(sp_key, categories=sp_ids))[0]
    n_sp = len(sp_ids)
    sp_station = np.array([k.split("|")[0] for k in sp_ids])
    sp_phase = np.array([k.split("|")[1] for k in sp_ids])

    is_s = (df["phase"].to_numpy() == "S")
    r = df["dist_hypo_km"].to_numpy()
    L = np.log10(r / R_REF_KM)
    b = df["log10A"].to_numpy()
    n_obs = len(df)

    base_c = n_ev
    base_d = n_ev + n_sp
    ridx = np.arange(n_obs)
    if fix_n is None:                     # solve geometric n_p AND anelastic k_p
        n_cols = base_d + 4               # n_P, k_P, n_S, k_S
        rows = [ridx, ridx, ridx, ridx]
        cols = [ev_idx, base_c + sp_idx,
                np.where(is_s, base_d + 2, base_d + 0),      # n_p
                np.where(is_s, base_d + 3, base_d + 1)]      # k_p
        vals = [np.ones(n_obs), np.ones(n_obs), -L, -r]
        rhs = list(b)
    else:                                 # fix geometric spreading; solve k_p only
        n_cols = base_d + 2               # k_P, k_S
        rows = [ridx, ridx, ridx]
        cols = [ev_idx, base_c + sp_idx, np.where(is_s, base_d + 1, base_d + 0)]  # k_p
        vals = [np.ones(n_obs), np.ones(n_obs), -r]
        rhs = list(b + fix_n * L)         # move the known geometric term to the LHS

    # per-phase gauge rows: sum of C over that phase's station-phase columns = 0
    grow = n_obs
    for ph in ("P", "S"):
        sel = np.where(sp_phase == ph)[0]
        if len(sel):
            rows.append(np.full(len(sel), grow))
            cols.append(base_c + sel)
            vals.append(np.full(len(sel), gauge_w))
            rhs.append(0.0)
            grow += 1

    A = sparse.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(grow, n_cols)).tocsr()
    x = lsqr(A, np.asarray(rhs), damp=damp, atol=1e-8, btol=1e-8, iter_lim=2000)[0]

    M = x[:n_ev]
    C_sp = x[base_c:base_c + n_sp]
    if fix_n is None:
        nP, kP, nS, kS = x[base_d:base_d + 4]
    else:
        nP = nS = fix_n
        kP, kS = x[base_d:base_d + 2]
    D = np.where(is_s, nS * L + kS * r, nP * L + kP * r)
    Cobs = C_sp[sp_idx]
    resid = b - (M[ev_idx] + Cobs - D)
    sp_tbl = pd.DataFrame({"station": sp_station, "phase": sp_phase, "C": C_sp})
    return dict(M=M, ev_ids=ev_ids, sp_tbl=sp_tbl, nP=nP, kP=kP, nS=nS, kS=kS,
                resid=resid, D=D, Cobs=Cobs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="../../data/magnitude/amp_distance_dataset.csv")
    p.add_argument("--outdir", default="../../data/magnitude")
    p.add_argument("--min-log10a", type=float, default=0.0, help="drop amplitudes below (noise floor)")
    p.add_argument("--min-sta-obs", type=int, default=8)
    p.add_argument("--min-ev-obs", type=int, default=3)
    p.add_argument("--damp", type=float, default=1e-3)
    p.add_argument("--gauge-w", type=float, default=1000.0)
    p.add_argument("--reject-mad", type=float, default=4.0, help="robust: drop |resid|>k*MAD, refit")
    p.add_argument("--fix-n", type=float, default=None,
                   help="fix geometric spreading exponent n and solve only k (yields k>0); e.g. 1.0")
    p.add_argument("--suffix", default="", help="suffix for output filenames, e.g. _kpos")
    args = p.parse_args(argv)

    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(os.path.expanduser(args.dataset))
    n_total_ev = df["event_id"].nunique()
    n0 = len(df)

    # ---------------- QC ----------------
    drop = {}
    m = df["log10A"] >= args.min_log10a
    drop["amp<floor"] = int((~m).sum()); df = df[m]
    while True:                                      # min-obs per station and per event
        sc = df["station"].value_counts()
        keep_sta = sc[sc >= args.min_sta_obs].index
        m = df["station"].isin(keep_sta)
        if not m.all():
            df = df[m]; continue
        ec = df["event_id"].value_counts()
        keep_ev = ec[ec >= args.min_ev_obs].index
        m2 = df["event_id"].isin(keep_ev)
        if m2.all():
            break
        df = df[m2]
    drop["sta/ev<min-obs"] = n0 - drop["amp<floor"] - len(df)
    df = df.reset_index(drop=True)

    # ---------------- solve (+1 robust pass) ----------------
    out = solve(df, args.damp, args.gauge_w, args.fix_n)
    mad = 1.4826 * np.median(np.abs(out["resid"] - np.median(out["resid"])))
    keep = np.abs(out["resid"]) <= args.reject_mad * mad
    n_rej = int((~keep).sum())
    if n_rej:
        df = df[keep].reset_index(drop=True)
        out = solve(df, args.damp, args.gauge_w, args.fix_n)

    # ---------------- per-event magnitudes + station-magnitude scatter ----------------
    df["resid"] = out["resid"]
    df["m_sta"] = df["log10A"].to_numpy() + out["D"] - out["Cobs"]     # single-station magnitude
    ev_ids = out["ev_ids"]
    g = df.groupby("event_id")
    isP = df["phase"].eq("P")
    ev = pd.DataFrame({"event_id": ev_ids, "M_rel": out["M"]})
    ev["n_obs"] = g.size().reindex(ev_ids).to_numpy()
    ev["n_P"] = isP.groupby(df["event_id"]).sum().reindex(ev_ids).to_numpy()
    ev["n_S"] = (~isP).groupby(df["event_id"]).sum().reindex(ev_ids).to_numpy()
    ev["M_sta_std"] = g["m_sta"].std().reindex(ev_ids).to_numpy()
    for c in ("evla", "evlo", "evdp"):
        ev[c] = g[c].first().reindex(ev_ids).to_numpy()
    ev["M_rel_sem"] = ev["M_sta_std"] / np.sqrt(ev["n_obs"])

    sta = out["sp_tbl"].merge(
        df.groupby(["station", "phase"]).size().rename("n_obs").reset_index(),
        on=["station", "phase"], how="left")

    ev_out = os.path.join(outdir, f"route_b_event_relative_mag{args.suffix}.csv")
    st_out = os.path.join(outdir, f"route_b_station_terms{args.suffix}.csv")
    ev.to_csv(ev_out, index=False); sta.to_csv(st_out, index=False)

    # ---------------- report ----------------
    rmad = 1.4826 * np.median(np.abs(out["resid"] - np.median(out["resid"])))
    print("=== Route B relative-magnitude inversion ===")
    print(f"input observations     : {n0:,}   events total: {n_total_ev:,}")
    print(f"dropped amp<floor      : {drop['amp<floor']:,}")
    print(f"dropped sta/ev<min-obs : {drop['sta/ev<min-obs']:,}")
    print(f"robust-rejected obs    : {n_rej:,}")
    print(f"final observations     : {len(df):,}")
    print(f"events WITH magnitude  : {len(ev):,}  ({100*len(ev)/n_total_ev:.1f}% of all events)")
    print(f"events MISSED          : {n_total_ev - len(ev):,}  (too few well-recorded stations)")
    print(f"station-phase terms    : {len(sta):,}  (|C|>10 pathological: {(sta.C.abs()>10).sum()})")
    print(f"attenuation  P: n={out['nP']:.3f} k={out['kP']:.5f}   S: n={out['nS']:.3f} k={out['kS']:.5f}")
    print(f"residual RMS={np.std(out['resid']):.3f}  robustMAD={rmad:.3f}  (log10 units)")
    print(f"median single-station-magnitude scatter per event = {ev['M_sta_std'].median():.3f}")
    print(f"M_rel range p1..p99    : {ev.M_rel.quantile(.01):.2f} .. {ev.M_rel.quantile(.99):.2f}")
    print(f"wrote {ev_out}\nwrote {st_out}")
    print("NOTE: M_rel is RELATIVE (uncalibrated). Absolute ML needs a ComCat/Morton anchor (next step).")

    # ---------------- diagnostics ----------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        ax[0, 0].hist(ev.M_rel, bins=80); ax[0, 0].set_title("relative magnitude M_rel"); ax[0, 0].set_xlabel("M_rel")
        for ph in ("P", "S"):
            ax[0, 1].hist(sta[sta.phase == ph]["C"].dropna(), bins=40, alpha=0.5, label=ph)
        ax[0, 1].set_title("station terms C (response+site, log10)"); ax[0, 1].legend(); ax[0, 1].set_xlabel("C")
        s = df.sample(min(120000, len(df)), random_state=0)
        ax[1, 0].hexbin(s.dist_hypo_km, s["resid"], gridsize=60, bins="log", mincnt=1)
        ax[1, 0].axhline(0, color="w", lw=.5); ax[1, 0].set_title("residual vs distance (should be flat)")
        ax[1, 0].set_xlabel("hypocentral distance (km)"); ax[1, 0].set_ylabel("residual (log10)")
        ax[1, 1].hist(ev.M_sta_std.dropna(), bins=60)
        ax[1, 1].set_title("per-event single-station-magnitude scatter"); ax[1, 1].set_xlabel("std (log10 units)")
        fig.suptitle("Route B relative-magnitude inversion — diagnostics"); fig.tight_layout()
        png = os.path.join(outdir, f"route_b_diagnostics{args.suffix}.png"); fig.savefig(png, dpi=130)
        print(f"wrote {png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
