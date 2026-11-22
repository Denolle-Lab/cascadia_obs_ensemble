"""Waveform check of OBS S picks with large Track A S station terms (D-B010 follow-up).

Question: at deep-water OBS whose S term is strongly negative (S observed 0.5-2.4 s earlier than the
3D model predicts), is the deep-learning S pick the S wave, or a P-to-S conversion at the sediment-basement
contact (Ps, a near-constant delay after P)? At shelf OBS with large positive terms, is the S delay of the
size expected from slow sediments?

Inputs: Track A iter1 NLLoc phases (cache of b1_hyp_phases.py: observed pick, TTpred, residual, Tcorr,
epicentral distance), Track A iter1 locations, b1 station table and station-term analysis.
Waveforms: UW pnwstore archive read directly through its sqlite index (no pnwstore package needed),
EarthScope FDSN as fallback. Raw windows saved under <base>/obs_check/raw (not in git).

Per station:
  * pick statistics on all weighted iter1 picks: S residual without term vs distance, observed S minus
    predicted P (tS - tP_pred) vs predicted S-P. Ps picked as S would give tS - tP_pred ~ constant
    (0.5-2 s) with zero slope on predicted S-P; a true S gives slope ~1 and intercept ~ (S term - P term).
  * record section of up to N_EV events (Z, H1, H2; 2-12 Hz), marks: P pick, S pick, predicted P and S
    (3D model, no term), predicted S + Track A S term.

Outputs: data/tomo_v1/obs_check/*.png, obs_check_station_stats.csv, obs_check_events.csv, manifest.
Usage: python b1_obs_waveform_check.py [--no-download]
"""
import argparse
import io
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from manifest import REPO, load_params, write_manifest  # noqa: E402

VPVS_MODEL = 1.76          # Track A run iter1_vpvs176: Vs = Vp / 1.76 everywhere -> tP_pred = tS_pred / 1.76
N_EV = 8                   # events plotted per station
N_MEAS = 25                # events measured per station (onset test)
BAND = (2.0, 12.0)
INDEX = "/wd1/PNWstore_sqlite/{year}.sqlite"

GROUPS = {
    "neg": ["W01.7A", "J35A.7D", "M03C.7D", "W02.7A", "FN13C.7D", "J35C.7D", "J58C.7D", "J67C.7D",
            "J34A.7D", "W03.7A", "J11B.7D", "GB220.Z5", "G20D.7D", "GB281.Z5", "J20D.7D", "J28D.7D"],
    "shelf": ["FC03D.7D", "FS15D.7D", "J33A.7D", "FS18B.7D", "FS14D.7D", "FS42D.7D", "FS08D.7D", "M01C.7D"],
    "ctrl_obs": ["BB320.X9", "J06B.7D", "BB300.X9"],
    "ctrl_land": [],         # filled from the term table: land, many picks, |S term| < 0.1, near the coast
}


def pnw_read(net, sta, t0, t1):
    """Read a window from the pnwstore day files (path /data/wdNN -> /auto/pnwstore1-wdNN)."""
    from obspy import Stream, read
    out = Stream()
    for t in sorted({(t0.year, t0.julday), (t1.year, t1.julday)}):
        try:
            con = sqlite3.connect(INDEX.format(year=t[0]))
        except Exception:
            continue
        rows = con.execute(
            "SELECT channel, filename, byteoffset, bytes, starttime FROM tsindex WHERE network=? AND "
            "station=? AND substr(starttime,1,4)=? ", (net, sta, str(t[0]))).fetchall()
        con.close()
        jd = f"{t[0]}.{t[1]:03d}"
        rows = [r for r in rows if r[1].endswith(jd)]
        chans = {r[0] for r in rows}
        band = next((b for b in ("HH", "BH", "EH", "HN") if {b + "Z"} <= chans), None)
        if band is None:
            continue
        for ch, fn, off, nb, _ in rows:
            if ch[:2] != band or ch[2] not in "Z12NE":
                continue
            fn = fn.replace("/data/wd", "/auto/pnwstore1-wd")
            try:
                with open(fn, "rb") as f:
                    f.seek(off)
                    buf = io.BytesIO(f.read(nb))
                out += read(buf, starttime=t0, endtime=t1)
            except Exception:
                pass
    out.merge(fill_value=0)
    return out.trim(t0, t1)


def fdsn_read(net, sta, t0, t1):
    from obspy.clients.fdsn import Client
    for c in ("EARTHSCOPE", "IRIS"):
        try:
            return Client(c, timeout=60).get_waveforms(net, sta, "*", "HH?,BH?,EH?", t0, t1)
        except Exception:
            continue
    from obspy import Stream
    return Stream()


def get_window(net, sta, t0, t1, rawdir, eid, download=True):
    from obspy import read
    f = rawdir / f"{sta}.{net}.{eid}.mseed"
    if f.exists():
        return read(str(f)), "cache"
    if not download:
        from obspy import Stream
        return Stream(), "none"
    st, src = pnw_read(net, sta, t0, t1), "pnwstore"
    if len(st.select(channel="??Z")) == 0:
        st, src = fdsn_read(net, sta, t0, t1), "fdsn"
        if len(st):
            bands = {tr.stats.channel[:2] for tr in st}
            b = next((x for x in ("HH", "BH", "EH") if x in bands), None)
            st = st.select(channel=f"{b}?") if b else st
            st.merge(fill_value=0)
    if len(st):
        for tr in st:
            tr.data = tr.data.astype(np.float32)
        st.write(str(f), format="MSEED")
    return st, src


def pick_table(ph, loc):
    """All weighted iter1 picks with origin time, predicted times and the no-term residual."""
    ph = ph[ph.w > 0].copy()
    ph["sta"] = ph.sta.astype(str)
    ph["phase"] = ph.phase.astype(str)
    ph["pick_time"] = pd.to_datetime(ph.pick_time)
    ph["ot"] = ph.pick_time - pd.to_timedelta(ph.ttpred + ph.tcorr + ph.res, unit="s")
    ph["tt"] = (ph.pick_time - ph.ot).dt.total_seconds()
    ph["res0"] = ph.res + ph.tcorr                  # residual against the 3D model without station term
    ph = ph.merge(loc[["ID", "depth"]], on="ID", how="left")
    ph["hyp"] = np.hypot(ph.dist, ph.depth)
    return ph


def station_stats(ph, sta, terms):
    s = ph[(ph.sta == sta) & (ph.phase == "S")].copy()
    p = ph[(ph.sta == sta) & (ph.phase == "P")][["ID", "tt", "res0"]].rename(columns={"tt": "ttP", "res0": "res0P"})
    s = s.merge(p, on="ID", how="left")
    s["tP_pred"] = s.ttpred / VPVS_MODEL
    s["sp_pred"] = s.ttpred - s.tP_pred
    s["s_minus_ppred"] = s.tt - s.tP_pred          # Ps hypothesis: ~constant; S: ~ sp_pred + term
    s["sp_obs"] = s.tt - s.ttP
    r = {"station": sta, "n_S": len(s), "n_SP": int(s.ttP.notna().sum()),
         "dist_med": s.dist.median(), "dist_min": s.dist.min(),
         "term_S": terms.get(sta, (np.nan, np.nan))[1], "term_P": terms.get(sta, (np.nan, np.nan))[0],
         "res0S_med": s.res0.median()}
    if len(s) >= 3 and s.sp_pred.std() > 0.3:
        a, b = np.polyfit(s.sp_pred, s.s_minus_ppred, 1)
        r.update(slope_SmPpred=a, icpt_SmPpred=b)
        a2, b2 = np.polyfit(s.dist, s.res0, 1)
        r.update(slope_res0S_per_100km=100 * a2)
    sp = s.dropna(subset=["ttP"])
    if len(sp) >= 3 and sp.sp_pred.std() > 0.3:
        a, b = np.polyfit(sp.sp_pred, sp.sp_obs, 1)
        r.update(slope_SPobs=a, icpt_SPobs=b, res0P_med=sp.res0P.median())
    r["SmPpred_min"] = s.s_minus_ppred.min()
    r["SmPpred_med"] = s.s_minus_ppred.median()
    return r, s


def choose_events(s, n=N_EV):
    s = s.copy()
    s["hasP"] = s.ttP.notna()
    s = s.sort_values(["hasP", "dist"], ascending=[False, True])
    near = s[s.dist <= 80]
    sel = near.head(n)
    if len(sel) < n:
        sel = pd.concat([sel, s[~s.ID.isin(sel.ID)].sort_values("dist").head(n - len(sel))])
    return sel.sort_values("dist")


def aic_onset(x):
    """Index of the AIC minimum (Maeda 1985) of a trace segment."""
    n = len(x)
    k = np.arange(5, n - 5)
    v1 = np.array([np.var(x[:i]) for i in k]) + 1e-30
    v2 = np.array([np.var(x[i:]) for i in k]) + 1e-30
    return k[np.argmin(k * np.log(v1) + (n - k - 1) * np.log(v2))]


def measure(st, ot, tpick):
    """Onset of S energy on Z and on the horizontals near the S pick (AIC in pick -2.5 / +3 s),
    and the horizontal / vertical RMS ratio in the 1 s after the pick and 1-2 s after it.
    Sp picked as S: Z onset at the pick, horizontal onset later (tH - tZ ~ 0.5-3 s), H/Z < 1 at the pick."""
    z = st.select(channel="??Z")
    h = [tr for tr in st if tr.stats.channel[-1] in "12NE"]
    if not z or len(h) < 2:
        return {}
    z = z[0]
    out = {}
    w0, w1 = ot + tpick - 2.5, ot + tpick + 3.0
    zz = z.slice(w0, w1).data.astype(float)
    hh = [tr.slice(w0, w1).data.astype(float) for tr in h[:2]]
    n = min(len(zz), *(len(x) for x in hh))
    if n < 50:
        return {}
    dt = z.stats.delta
    hs = np.hypot(hh[0][:n], hh[1][:n])
    out["tZ_minus_pick"] = aic_onset(zz[:n]) * dt - 2.5
    out["tH_minus_pick"] = aic_onset(hs) * dt - 2.5   # AIC on the horizontal modulus
    i0 = int(2.5 / dt)
    i1, i2 = i0 + int(1 / dt), i0 + int(2 / dt)
    rz = lambda a, b: np.sqrt(np.mean(zz[a:b] ** 2)) + 1e-20  # noqa: E731
    rh = lambda a, b: np.sqrt(np.mean(hs[a:b] ** 2) / 2)  # noqa: E731
    out["HZ_0_1s"] = rh(i0, i1) / rz(i0, i1)
    out["HZ_1_2s"] = rh(i1, i2) / rz(i1, i2)
    out["HZ_pre"] = rh(0, i0) / rz(0, i0)
    return out


def measure_station(sta, grp, sel, rawdir, terms, download):
    from obspy import UTCDateTime
    net, code = sta.split(".")[1], sta.split(".")[0]
    rows = []
    for _, e in sel.iterrows():
        ot = UTCDateTime(e.ot.to_pydatetime())
        marks = [e.tP_pred, e.ttpred, e.tt] + ([e.ttP] if pd.notna(e.ttP) else [])
        st, src = get_window(net, code, ot + min(marks) - 6, ot + max(marks) + 10, rawdir, int(e.ID), download)
        r = dict(station=sta, group=grp, ID=int(e.ID), dist=e.dist, depth=e.depth, ttP=e.ttP, ttS=e.tt,
                 tP_pred=e.tP_pred, tS_pred=e.ttpred, res0S=e.res0, src=src, ntr=len(st))
        if len(st) >= 3:
            st.detrend("demean")
            st.taper(0.05)
            st.filter("bandpass", freqmin=BAND[0], freqmax=min(BAND[1], 0.45 * st[0].stats.sampling_rate),
                      corners=4, zerophase=True)
            r.update(measure(st, ot, e.tt))
        rows.append(r)
    return rows


def plot_station(sta, grp, sel, rawdir, outdir, terms, download, log):
    from obspy import UTCDateTime
    net, code = sta.split(".")[1], sta.split(".")[0]
    tP_, tS_ = terms.get(sta, (np.nan, np.nan))
    nev = len(sel)
    fig, axes = plt.subplots(nev, 1, figsize=(7.0, 1.05 * nev + 0.9), sharex=False, squeeze=False)
    axes = axes[:, 0]
    rows = []
    for ax, (_, e) in zip(axes, sel.iterrows()):
        ot = UTCDateTime(e.ot.to_pydatetime())
        tPp, tSp = e.tP_pred, e.ttpred
        marks = [tPp, tSp, e.tt] + ([e.ttP] if pd.notna(e.ttP) else [])
        t0, t1 = ot + min(marks) - 6, ot + max(marks) + 10
        st, src = get_window(net, code, t0, t1, rawdir, int(e.ID), download)
        row = dict(station=sta, group=grp, ID=int(e.ID), dist=e.dist, depth=e.depth, ttP=e.ttP, ttS=e.tt,
                   tP_pred=tPp, tS_pred=tSp, tS_pred_term=tSp + (tS_ if pd.notna(tS_) else 0),
                   src=src, ntr=len(st))
        rows.append(row)
        if len(st) == 0:
            ax.text(0.5, 0.5, f"ev {int(e.ID)}: no data", transform=ax.transAxes, ha="center", fontsize=7)
            ax.set_yticks([])
            continue
        st.detrend("demean")
        st.taper(0.05)
        st.filter("bandpass", freqmin=BAND[0], freqmax=min(BAND[1], 0.45 * st[0].stats.sampling_rate),
                  corners=4, zerophase=True)
        order = sorted(st, key=lambda tr: {"Z": 0, "1": 1, "N": 1, "2": 2, "E": 2}.get(tr.stats.channel[-1], 3))
        cols = {"Z": "k", "1": "tab:blue", "N": "tab:blue", "2": "tab:green", "E": "tab:green"}
        for k, tr in enumerate(order[:3]):
            t = tr.times() + (tr.stats.starttime - ot)
            d = tr.data / (np.abs(tr.data).max() + 1e-20)
            ax.plot(t, d * 0.45 - k, lw=0.4, color=cols.get(tr.stats.channel[-1], "0.4"))
        ax.axvline(tPp, color="tab:orange", ls=":", lw=1)
        if pd.notna(e.ttP):
            ax.axvline(e.ttP, color="tab:orange", lw=1.2)
        ax.axvline(tSp, color="tab:red", ls=":", lw=1)
        if pd.notna(tS_):
            ax.axvline(tSp + tS_, color="tab:purple", ls="--", lw=1)
        ax.axvline(e.tt, color="tab:red", lw=1.2)
        ax.set_xlim(t0 - ot, t1 - ot)
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels([tr.stats.channel for tr in order[:3]], fontsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.text(1.0, 1.0, f"ev {int(e.ID)}  {e.dist:.0f} km, z {e.depth:.0f} km  S-Ppred {e.tt - tPp:.1f} s",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    axes[-1].set_xlabel("time after origin (s)", fontsize=7)
    fig.suptitle(f"{sta} ({grp}): S term {tS_:+.2f} s, P term {tP_:+.2f} s; {BAND[0]:g}-{BAND[1]:g} Hz\n"
                 "solid orange P pick, dotted orange P pred; solid red S pick, dotted red S pred (3D), "
                 "dashed purple S pred + term", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = outdir / f"wf_{grp}_{sta}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log.append(out)
    return rows


def plot_summary(allS, stats, outdir):
    grps = [("neg", "deep-water OBS, S term < -0.8 s"), ("shelf", "shelf OBS, S term > +0.5 s"),
            ("ctrl_obs", "control OBS, |S term| < 0.15 s"), ("ctrl_land", "control land")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True)
    for ax, (g, title) in zip(axes, grps):
        d = allS[allS.group == g]
        for sta, ds in d.groupby("sta"):
            ax.plot(ds.sp_pred, ds.s_minus_ppred, ".", ms=2, alpha=0.5, label=f"{sta.split('.')[0]} ({len(ds)})")
        x = np.linspace(0, 25, 10)
        ax.plot(x, x, "k-", lw=0.8)
        ax.axhspan(0.3, 2.5, color="0.85", zorder=0)
        ax.set_xlim(0, 25)
        ax.set_ylim(-1, 27)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("predicted S - P (s), 3D model", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=4.5, markerscale=3, ncol=2, loc="upper left")
    axes[0].set_ylabel("picked S - predicted P (s)", fontsize=7)
    fig.suptitle("Black line: true S with zero term. Grey band: Ps / Sp sediment conversion window "
                 "(0.3-2.5 s after P, independent of distance)", fontsize=8)
    fig.tight_layout()
    out = outdir / "summary_S_minus_Ppred.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    # residual vs predicted S-P
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, (g, title) in zip(axes, grps):
        d = allS[allS.group == g]
        for sta, ds in d.groupby("sta"):
            ax.plot(ds.sp_pred, ds.res0, ".", ms=2, alpha=0.5)
        x = np.linspace(0, 25, 10)
        ax.plot(x, 1.0 - x, "k:", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(0, 25)
        ax.set_ylim(-6, 4)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("predicted S - P (s), 3D model", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("S residual, no station term (s)", fontsize=7)
    fig.suptitle("Dotted: residual expected if a conversion 1 s after P were picked as S", fontsize=8)
    fig.tight_layout()
    out2 = outdir / "summary_Sres_vs_dist.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    return [out, out2]


def near_terms(f_ph, terms_df, f_b1picks, dists=(40, 60, 100), nmin=10):
    """Distance-limited S statics at all OBS: median no-term iter1 S residual of picks within D km
    (n >= nmin), next to the Track A term (all distances). Also the proposed D-B010 revision."""
    ph = pd.read_parquet(f_ph, columns=["sta", "phase", "res", "tcorr", "w", "dist"])
    ph = ph[(ph.w > 0) & (ph.phase.astype(str) == "S")]
    ph = ph.assign(sta=ph.sta.astype(str), res0=ph.res + ph.tcorr)
    o = terms_df[terms_df.is_obs][["station", "zsrce_km", "term_P", "term_S"]].copy()
    for D in dists:
        g = ph[ph.dist <= D].groupby("sta").res0.agg(["median", "size"])
        o[f"near{D}"] = o.station.map(g["median"].where(g["size"] >= nmin))
        o[f"n{D}"] = o.station.map(g["size"]).fillna(0).astype(int)
    md = ph.groupby("sta").dist.median()
    o["dist_med_S"] = o.station.map(md)
    b = pd.read_csv(f_b1picks, usecols=["station", "phase", "is_obs"])
    o["nS_b1"] = o.station.map(b[(b.phase == "S") & b.is_obs].groupby("station").size()).fillna(0).astype(int)
    # proposed rule: near-60 km static, else near-100 km, else none
    o["S_static_proposed"] = o.near60.fillna(o.near100)
    o["static_source"] = np.where(o.near60.notna(), "near60",
                                  np.where(o.near100.notna(), "near100", "none"))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    P = load_params()
    base = Path(P["paths"]["base"])
    b1 = base / "b1_dataset"
    rawdir = base / "obs_check" / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    outdir = REPO / "data" / "tomo_v1" / "obs_check"
    outdir.mkdir(parents=True, exist_ok=True)
    run = Path(P["paths"]["reloc_v5"]) / "runs" / P["paths"]["trackA_run"]
    f_ph = b1 / "cache" / f"phases_{P['paths']['trackA_run']}.parquet"
    f_loc = run / "locations.csv"
    f_terms = b1 / "station_terms_analysis.csv"

    terms_df = pd.read_csv(f_terms)
    terms = {r.station: (r.term_P, r.term_S) for r in terms_df.itertuples()}
    land = terms_df[(~terms_df.is_obs) & (terms_df.term_S.abs() < 0.1) & (terms_df.term_P.abs() < 0.1)
                    & (terms_df.lon < -123.3)].sort_values("i0_n_S", ascending=False)
    GROUPS["ctrl_land"] = land.station.head(3).tolist()

    loc = pd.read_csv(f_loc, usecols=["ID", "time", "lat", "lon", "depth"])
    ph = pd.read_parquet(f_ph)
    wanted = [s for g in GROUPS.values() for s in g]
    ph = ph[ph.sta.astype(str).isin(wanted)]
    ph = pick_table(ph, loc)

    stats, allS, evrows, figs = [], [], [], []
    for g, stas in GROUPS.items():
        for sta in stas:
            r, s = station_stats(ph, sta, terms)
            r["group"] = g
            stats.append(r)
            s["group"] = g
            allS.append(s)
            if len(s) == 0:
                continue
            sel = choose_events(s)
            plot_station(sta, g, sel, rawdir, outdir, terms, not a.no_download, figs)
            evrows += measure_station(sta, g, choose_events(s, N_MEAS), rawdir, terms, not a.no_download)
            print(sta, g, len(s), "picks;", len(sel), "events plotted", flush=True)
    allS = pd.concat(allS)
    figs += plot_summary(allS, stats, outdir)
    stats = pd.DataFrame(stats)
    ev = pd.DataFrame(evrows)
    if "tH_minus_pick" in ev:
        ev["tH_minus_tZ"] = ev.tH_minus_pick - ev.tZ_minus_pick
        m = ev.groupby("station").agg(n_wf=("tH_minus_pick", "count"),
                                      tZ_pick_med=("tZ_minus_pick", "median"),
                                      tH_pick_med=("tH_minus_pick", "median"),
                                      tH_tZ_med=("tH_minus_tZ", "median"),
                                      HZ01_med=("HZ_0_1s", "median"), HZ12_med=("HZ_1_2s", "median"),
                                      HZpre_med=("HZ_pre", "median"))
        stats = stats.merge(m, left_on="station", right_index=True, how="left")
    f_stats = outdir / "obs_check_station_stats.csv"
    stats.round(3).to_csv(f_stats, index=False)
    f_ev = outdir / "obs_check_events.csv"
    ev.round(3).to_csv(f_ev, index=False)
    f_b1 = b1 / "picks.csv"
    nt = near_terms(f_ph, terms_df, f_b1)
    f_nt = outdir / "obs_near_station_S_terms.csv"
    nt.round(3).to_csv(f_nt, index=False)
    nb = nt[nt.nS_b1 > 0]
    cnt = {f"b1_obs_S_picks_static_{k}": int(v) for k, v in nb.groupby("static_source").nS_b1.sum().items()}
    cnt["b1_obs_S_picks_trackA_term"] = int(nb[nb.term_S.notna()].nS_b1.sum())
    cnt["obs_trackA_term_lt_-0.5"] = int((nb.term_S < -0.5).sum())
    cnt["obs_proposed_static_lt_-0.5"] = int((nb.S_static_proposed < -0.5).sum())
    print(cnt)
    write_manifest("b1_obs_waveform_check", outdir, inputs=[f_ph, f_loc, f_terms, f_b1],
                   outputs=[f_stats, f_ev, f_nt, *figs],
                   params={"vpvs_model": VPVS_MODEL, "band_hz": BAND, "n_ev": N_EV, "n_meas": N_MEAS, "groups": GROUPS},
                   counts={**cnt, "stations": len(stats), "waveform_windows": len(evrows),
                           "windows_with_data": int(sum(r["ntr"] > 0 for r in evrows))})


if __name__ == "__main__":
    main()
