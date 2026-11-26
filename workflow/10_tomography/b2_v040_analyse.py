"""v0.4.0 report: what changed when v0.3.1 was re-inverted with the re-timed picks (D-B072/D-B075).

Compares every v0.4.0 run with its v0.3.1 counterpart (M0 margin_m0_v3 vs margin_m0_v2, tiles <t>_m0t vs <t>_m0) and
the merged models v0.4.0 vs v0.3.1, against the numerical noise floor (the v0.4.0 chain rerun with N(0, 1e-6 s) on
every time: margin_m0_v3nf, <t>_m0tnf, merged dry run noise/merged_v040nf.nc).

Sections (tables in <out>/report/*.csv, figures *.png, report.html; small files copied to data/tomo_v1/v040_retimed/):
  1 picks re-timed in each inversion, by phase, station and size; B1 QC (MAD) cuts before/after (b1t_qc_check.py);
    tomoDD's own residual cuts (WRCT 8 / 6 x MAD) before/after: data in the inputs vs data left in tomoDD.res
  2 residuals of the final iteration: rms by region, phase and station class; re-timed stations: station means and
    the residual against time of day (the drift made picks early by up to ~0.5 s at the end of the UTC day)
  3 merged model differences dVp, dVs (%), d(Vp/Vs): maps, sections, share of resolved nodes moved > 1 % / 2 %,
    against the noise floor
  4 hypocentre shifts of the tomography events (tomoDD final), by region; noise-floor shifts for scale
  5 rule checks: M0 and tile smoothing scans with the re-timed picks (rule of D-B046) against v0.3.1's choice
  6 P-4 and input verification, run times and resources

  python b2_v040_analyse.py
"""
import base64
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402

P = load_params()
T, RT = P["tomodd"], P["retime"]
RUNS = Path(T["runs"])
BASE = Path(P["paths"]["base"])
OUT = Path(RT["out"]) / "report"
GIT = REPO / "data" / "tomo_v1" / "v040_retimed"
STORE = Path(T["models"]["store"]) / T["models"]["name"]
TILES = {"mendocino_gorda_c": 480, "blanco": 20, "oregon": 480, "washington": 480}
RUNSET = {"M0": ("margin_m0_v2", "margin_m0_v3", "margin_m0_v3nf", 480)}
RUNSET.update({t: (f"{t}_m0", f"{t}_m0t", f"{t}_m0tnf", s) for t, s in TILES.items()})
SIZES = [0.02, 0.05, 0.1, 0.3]
TXT = []          # markdown-ish lines for the report


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def run_dir(region, sm):
    return RUNS / region / f"d200_s{sm}"


# ------------------------------------------------------------------ readers
def read_res(run, ini=False):
    f = run / ("tomoDD-SE.res.ini" if ini else "Output_Files/tomoDD.res")
    r = pd.read_csv(f, sep=r"\s+", skiprows=1, header=None,
                    names=["code", "dt", "c1", "c2", "idx", "qual", "res_ms", "wt", "offs"], dtype={"res_ms": str})
    r["res_ms"] = pd.to_numeric(r.res_ms, errors="coerce")       # Fortran writes ***** when a value overflows
    nbad = int(r.res_ms.isna().sum())
    if nbad:
        print(f"{f}: {nbad} residuals unreadable (overflowed format), left out")
    return r.dropna(subset=["res_ms"])


def abs_res(run, ini=False):
    r = read_res(run, ini)
    a = r[r.c1 == r.c2].copy()
    a["phase"] = np.where(a.idx == 3, "P", "S")
    a = a.rename(columns={"c1": "event_ID"})
    a["res_s"] = a.res_ms / 1000.0
    return a[["event_ID", "code", "phase", "res_s", "wt", "qual"]], r[r.c1 != r.c2]


def reloc(run):
    f = run / "Output_Files" / "tomoDD.reloc_geo"
    d = pd.read_csv(f, sep=r"\s+", header=None)
    d = d.rename(columns={0: "ID", 1: "lat", 2: "lon", 3: "depth", 10: "yr", 11: "mo", 12: "dy", 13: "hr", 14: "mi",
                          15: "sc"})
    t = pd.to_datetime(dict(year=d.yr, month=d.mo, day=d.dy, hour=d.hr, minute=d.mi)) + \
        pd.to_timedelta(d.sc, unit="s")
    d["t"] = t
    return d[["ID", "lat", "lon", "depth", "t"]]


def parse_log(run):
    txt = (run / "tomoDD-SE.log").read_text(errors="replace")
    wt = [float(x) for x in re.findall(r"weighted ct rms \[s\][^=]*=\s+([-\d.]+)", txt)]
    ab = [float(x) for x in re.findall(r"absolute ct rms \[s\] =\s+([-\d.]+)", txt)]
    cut_abs = re.findall(r"abs ct res/dist cutoff \[s\]:\s+([-\d.]+)s/\s*([-\d.]+)km \(\s*([\d.]+)%\)", txt)
    cut_dt = re.findall(r"diff ct res/dist cutoff \[s\]:\s+([-\d.]+)s/\s*([-\d.]+)km \(\s*([\d.]+)%\)", txt)
    ev = [int(x) for x in re.findall(r"# events =\s+(\d+)", txt)]
    aq = [int(x) for x in re.findall(r"Number of air quakes \(AQ\) =\s+(\d+)", txt)]
    tm = (run / "time.txt").read_text().split()
    return dict(rms_wt_start=wt[0], rms_wt_final=wt[-1], rms_abs_start=ab[0], rms_abs_final=ab[-1],
                abs_cut_last_s=float(cut_abs[-1][0]), abs_cut_last_pct=float(cut_abs[-1][2]),
                abs_cut_pct_sum=round(sum(float(c[2]) for c in cut_abs), 2),
                dt_cut_last_s=float(cut_dt[-1][0]), dt_cut_pct_sum=round(sum(float(c[2]) for c in cut_dt), 2),
                events_final=ev[-1] if ev else np.nan, aq_total=sum(aq),
                wall_min=round(float(tm[-6]) / 60, 1), user_min=round(float(tm[-4]) / 60, 1),
                maxrss_gb=round(float(tm[-2]) / 1024 ** 2, 2))


# ------------------------------------------------------------------ 1 picks
def picks_tables():
    ds = pd.read_csv(BASE / RT["datasets"]["b1b"][1] / "picks.csv",
                     usecols=["event_ID", "station", "phase", "pick_time", "timing_correction_s", "timing_corrected",
                              "retime_epoch_class", "is_obs", "network"])
    rows, st_rows = [], []
    for name, (old, new, nf, sm) in RUNSET.items():
        pu = pd.read_csv(RUNS / new / "inputs" / "picks_used.csv", usecols=["event_ID", "station", "code", "phase",
                                                                           "is_obs"])
        m = pu.merge(ds[["event_ID", "station", "phase", "timing_correction_s", "retime_epoch_class"]],
                     on=["event_ID", "station", "phase"], how="left")
        a = m.timing_correction_s.abs()
        for ph in ("P", "S", "all"):
            s = m if ph == "all" else m[m.phase == ph]
            aa = s.timing_correction_s.abs()
            r = dict(region=name, phase=ph, picks=len(s), retimed=int((aa > 0).sum()),
                     retimed_pct=round(100 * (aa > 0).mean(), 2),
                     listed=int((s.retime_epoch_class == "listed").sum()),
                     found=int((s.retime_epoch_class == "found").sum()),
                     stations_retimed=int(s[aa > 0].station.nunique()), obs_retimed=int(((aa > 0) & s.is_obs).sum()))
            for z in SIZES:
                r[f"gt_{z:g}s"] = int((aa > z).sum())
            r["median_abs_s"] = round(float(aa[aa > 0].median()), 4) if (aa > 0).any() else 0.0
            r["max_abs_s"] = round(float(aa.max()), 4)
            rows.append(r)
        g = m[a > 0].groupby("station")
        st = pd.DataFrame({"region": name, "retimed": g.size(), "P": g.phase.apply(lambda x: int((x == "P").sum())),
                           "S": g.phase.apply(lambda x: int((x == "S").sum())),
                           "median_corr_s": g.timing_correction_s.median().round(4),
                           "max_abs_corr_s": g.timing_correction_s.apply(lambda x: x.abs().max()).round(4),
                           "class": g.retime_epoch_class.agg(lambda x: "/".join(sorted(set(x)))),
                           "picks_at_station": m.groupby("station").size().reindex(g.size().index)})
        st_rows.append(st.reset_index())
    t1 = pd.DataFrame(rows)
    t2 = pd.concat(st_rows).sort_values(["region", "retimed"], ascending=[True, False])
    # unique tomography picks (B1b, every pick once)
    aa = ds.timing_correction_s.abs()
    u = dict(b1b_picks=len(ds), retimed=int((aa > 0).sum()), stations=int(ds[aa > 0].station.nunique()),
             obs=int(((aa > 0) & ds.is_obs).sum()), **{f"gt_{z:g}s": int((aa > z).sum()) for z in SIZES},
             listed=int((ds.retime_epoch_class == "listed").sum()), found=int((ds.retime_epoch_class == "found").sum()))
    return t1, t2, u, ds


def mad_cuts():
    """tomoDD's residual cuts: absolute and dt data in the inputs vs in the final tomoDD.res, events kept."""
    rows = []
    for name, (old, new, nf, sm) in RUNSET.items():
        for lab, reg in (("v0.3.1", old), ("v0.4.0", new)):
            run = run_dir(reg, sm)
            a, d = abs_res(run)
            pu = pd.read_csv(RUNS / reg / "inputs" / "picks_used.csv", usecols=["event_ID"])
            ev_final = set(a.event_ID)
            rl = reloc(run)
            kept_ev = set(rl.ID)
            n_in_kept = int(pu.event_ID.isin(kept_ev).sum())
            ndt_in = sum(1 for line in open(RUNS / reg / "inputs" / "dt.ct") if line.strip() and line[0] != "#")
            L = parse_log(run)
            rows.append(dict(region=name, version=lab, abs_in=len(pu), abs_final=len(a),
                             events_in=int(pu.event_ID.nunique()), events_final=len(kept_ev),
                             abs_of_kept_events_in=n_in_kept, abs_cut=n_in_kept - len(a),
                             abs_cut_pct=round(100 * (n_in_kept - len(a)) / n_in_kept, 3),
                             dt_in=ndt_in, dt_final=len(d), abs_cut_last_s=L["abs_cut_last_s"],
                             dt_cut_last_s=L["dt_cut_last_s"], aq_total=L["aq_total"],
                             rms_wt_start=L["rms_wt_start"], rms_wt_final=L["rms_wt_final"],
                             rms_abs_start=L["rms_abs_start"], rms_abs_final=L["rms_abs_final"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2 residuals
def residuals(ds):
    rt_sta = set(ds.loc[ds.timing_correction_s != 0, "station"])
    rows, per = [], []
    for name, (old, new, nf, sm) in RUNSET.items():
        codes = pd.read_csv(RUNS / new / "inputs" / "station_codes.csv")[["station", "code", "is_obs"]]
        both = {}
        for lab, reg, ini in (("v0.3.1", old, False), ("v0.4.0", new, False), ("v0.3.1 start", old, True),
                              ("v0.4.0 start", new, True)):
            a, _ = abs_res(run_dir(reg, sm), ini=ini)
            a = a.merge(codes, on="code", how="left")
            a = a.merge(ds[["event_ID", "station", "phase", "timing_correction_s", "pick_time"]],
                        on=["event_ID", "station", "phase"], how="left")
            a["cls"] = np.where(a.is_obs, "OBS", np.where(a.station.isin(rt_sta), "land, re-timed stations",
                                                          "land, other"))
            both[lab] = a
            for (ph, cls), g in a.groupby(["phase", "cls"]):
                rows.append(dict(region=name, version=lab, phase=ph, cls=cls, n=len(g),
                                 rms_s=round(float(np.sqrt((g.res_s ** 2).mean())), 4),
                                 mean_s=round(float(g.res_s.mean()), 4)))
            for ph, g in a.groupby("phase"):
                rows.append(dict(region=name, version=lab, phase=ph, cls="all", n=len(g),
                                 rms_s=round(float(np.sqrt((g.res_s ** 2).mean())), 4),
                                 mean_s=round(float(g.res_s.mean()), 4)))
            if not ini:
                per.append(a.assign(region=name, version=lab))
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["region", "phase", "cls"], columns="version", values="rms_s").reset_index()
    nn = t[t.version == "v0.4.0"].set_index(["region", "phase", "cls"]).n
    piv["n"] = [nn.get((r.region, r.phase, r.cls), np.nan) for r in piv.itertuples()]
    return t, piv, pd.concat(per, ignore_index=True), rt_sta


def daytime(per, ds):
    """Re-timed stations: station-mean residual and the time-of-day trend, before / after (final iteration, all
    inversions pooled; picks of re-timed station-days only, i.e. timing_correction_s != 0)."""
    a = per[per.timing_correction_s.fillna(0) != 0].copy()
    t = pd.to_datetime(a.pick_time)
    a["hod"] = (t - t.dt.normalize()).dt.total_seconds() / 86400.0       # fraction of the UTC day
    cls = ds.groupby(["event_ID", "station", "phase"]).retime_epoch_class.first()
    a["epoch"] = [cls.get((e, s, p), "") for e, s, p in zip(a.event_ID, a.station, a.phase)]
    rows = []
    for (sta, ver), g in a.groupby(["station", "version"]):
        if len(g) < 30:
            continue
        sl = np.polyfit(g.hod, g.res_s, 1)[0] if g.hod.std() > 0.05 else np.nan
        sc = np.polyfit(g.hod, -g.timing_correction_s, 1)[0] if g.hod.std() > 0.05 else np.nan
        rows.append(dict(station=sta, version=ver, n=len(g), mean_res_s=round(float(g.res_s.mean()), 4),
                         median_res_s=round(float(g.res_s.median()), 4),
                         slope_s_per_day=round(float(sl), 4), minus_corr_slope_s_per_day=round(float(sc), 4),
                         mean_corr_s=round(float(g.timing_correction_s.mean()), 4),
                         epochs="/".join(sorted(set(g.epoch)))))
    st = pd.DataFrame(rows)
    sp = st.pivot_table(index="station", columns="version",
                        values=["n", "mean_res_s", "slope_s_per_day", "mean_corr_s"]).round(4)
    sp.columns = [f"{a_}_{b}" for a_, b in sp.columns]
    # binned: 2-h bins, listed and found epochs
    a["bin"] = np.floor(a.hod * 12) / 12 + 1 / 24
    bins = a.groupby(["epoch", "version", "bin"]).agg(n=("res_s", "size"), median_res_s=("res_s", "median"),
                                                      mean_corr_s=("timing_correction_s", "mean")).reset_index()
    return st, sp.reset_index(), bins, a


# ------------------------------------------------------------------ 3 model differences
def model_diff():
    import xarray as xr
    old = xr.open_dataset(STORE / "v0.3.1.nc")
    new = xr.open_dataset(STORE / f"v{RT['version']}.nc")
    nf = xr.open_dataset(Path(RT["out"]) / "noise" / "merged_v040nf.nc")
    D = {}
    for k in ("Vp", "Vs"):
        D[f"d{k}"] = 100 * (new[k].values / old[k].values - 1)
        D[f"n{k}"] = 100 * (nf[k].values / new[k].values - 1)
    D["dVpVs"] = new.VpVs.values - old.VpVs.values
    D["nVpVs"] = nf.VpVs.values - new.VpVs.values
    res_p = new.resolution_p.values > 0
    res_s = new.resolution_s.values > 0
    cov_p = new.tile_weight_p.values > 0
    cov_s = new.tile_weight_s.values > 0
    same_mask = bool((old.resolution_p.values == new.resolution_p.values).all() and
                     (old.resolution_s.values == new.resolution_s.values).all())

    def local_max(x):
        from scipy.ndimage import maximum_filter
        return maximum_filter(np.abs(x), size=3, mode="nearest")

    rows = []
    for q, msk_name, msk in (("Vp", "resolved P (resolution_p > 0)", res_p), ("Vs", "resolved S (resolution_s > 0)", res_s),
                             ("Vp", "tile-covered P (tile_weight_p > 0)", cov_p),
                             ("Vs", "tile-covered S (tile_weight_s > 0)", cov_s),
                             ("Vp", "all nodes", np.ones_like(res_p)), ("Vs", "all nodes", np.ones_like(res_p)),
                             ("VpVs", "resolved P and S", res_p & res_s)):
        d, n = D[f"d{q}"][msk], D[f"n{q}"][msk]
        if q == "VpVs":    # as percent of Vp/Vs
            d = 100 * d / old.VpVs.values[msk]
            n = 100 * n / new.VpVs.values[msk]
        nl = local_max(D[f"n{q}"] if q != "VpVs" else 100 * D["nVpVs"] / new.VpVs.values)[msk]
        rows.append(dict(quantity=q, nodes=msk_name, n=int(msk.sum()),
                         gt1_pct=round(100 * float((np.abs(d) > 1).mean()), 2),
                         gt2_pct=round(100 * float((np.abs(d) > 2).mean()), 2),
                         rms_pct=round(float(np.sqrt((d ** 2).mean())), 3), max_abs_pct=round(float(np.abs(d).max()), 2),
                         noise_rms_pct=round(float(np.sqrt((n ** 2).mean())), 4),
                         noise_p99_pct=round(float(np.percentile(np.abs(n), 99)), 3),
                         noise_max_pct=round(float(np.abs(n).max()), 3),
                         noise_gt1_pct=round(100 * float((np.abs(n) > 1).mean()), 2),
                         above_noise_pct=round(100 * float((np.abs(d) > 2 * nl).mean()), 2),
                         gt1_and_above_noise_pct=round(100 * float(((np.abs(d) > 1) & (np.abs(d) > 2 * nl)).mean()), 2)))
    t = pd.DataFrame(rows)
    # by depth (resolved)
    z = new.depth.values
    dz = []
    for k, q, msk in (("Vp", "dVp", res_p), ("Vs", "dVs", res_s)):
        for i, zz in enumerate(z):
            m = msk[i]
            if m.sum() == 0:
                continue
            d, n = D[q][i][m], D["n" + k][i][m]
            dz.append(dict(quantity=k, depth_km=zz, resolved_nodes=int(m.sum()),
                           gt1_pct=round(100 * float((np.abs(d) > 1).mean()), 1),
                           gt2_pct=round(100 * float((np.abs(d) > 2).mean()), 1),
                           rms_pct=round(float(np.sqrt((d ** 2).mean())), 3),
                           noise_rms_pct=round(float(np.sqrt((n ** 2).mean())), 4)))
    return t, pd.DataFrame(dz), D, new, old, nf, same_mask


def regions_of_change(D, new):
    """Where: resolved nodes moved > 1 % (Vp or Vs), by geographic area and depth band."""
    lon, lat, z = new.longitude.values, new.latitude.values, new.depth.values
    Z, LA, LO = np.meshgrid(z, lat, lon, indexing="ij")
    rows = []
    for k, msk in (("Vp", new.resolution_p.values > 0), ("Vs", new.resolution_s.values > 0)):
        big = msk & (np.abs(D[f"d{k}"]) > 1)
        if not big.any():
            continue
        df = pd.DataFrame(dict(lon=LO[big], lat=LA[big], z=Z[big], d=D[f"d{k}"][big]))
        df["area"] = pd.cut(df.lat, [38, 41, 43.5, 46, 48, 51],
                            labels=["39-41N", "41-43.5N", "43.5-46N", "46-48N", "48-51N"]).astype(str) + \
            np.where(df.lon < -125.5, " west of 125.5W", np.where(df.lon < -123.5, " 125.5-123.5W", " east of 123.5W"))
        df["zband"] = pd.cut(df.z, [-4, 6, 16, 30, 130], labels=["<6 km", "6-16 km", "16-30 km", ">30 km"]).astype(str)
        g = df.groupby(["area", "zband"]).agg(nodes=("d", "size"), median_pct=("d", "median"),
                                              max_abs_pct=("d", lambda x: x.abs().max())).reset_index()
        g.insert(0, "quantity", k)
        rows.append(g)
    return pd.concat(rows).sort_values(["quantity", "nodes"], ascending=[True, False]) if rows else pd.DataFrame()


# ------------------------------------------------------------------ 4 hypocentres
def hypo_shifts():
    rows, allsh = [], []
    for name, (old, new, nf, sm) in RUNSET.items():
        ev = pd.read_csv(RUNS / new / "inputs" / "events.csv")[["ID", "region"]]
        a, b, c = reloc(run_dir(old, sm)), reloc(run_dir(new, sm)), reloc(run_dir(nf, sm))
        for lab, x, y in (("v0.4.0 - v0.3.1", b, a), ("noise floor (nf - v0.4.0)", c, b)):
            m = x.merge(y, on="ID", suffixes=("", "_0")).merge(ev, on="ID", how="left")
            m["dh_km"] = np.hypot((m.lat - m.lat_0) * 111.19, (m.lon - m.lon_0) * 111.19 * np.cos(np.radians(m.lat)))
            m["dz_km"] = m.depth - m.depth_0
            m["dt_s"] = (m.t - m.t_0).dt.total_seconds()
            m["comparison"], m["inversion"] = lab, name
            allsh.append(m[["inversion", "comparison", "ID", "region", "dh_km", "dz_km", "dt_s"]])
            for reg, g in list(m.groupby("region")) + [("all", m)]:
                rows.append(dict(inversion=name, comparison=lab, region=reg, events=len(g),
                                 only_in_one=len(set(x.ID) ^ set(y.ID)) if reg == "all" else np.nan,
                                 dh_median_km=round(float(g.dh_km.median()), 3),
                                 dh_p90_km=round(float(g.dh_km.quantile(0.9)), 3),
                                 dz_median_km=round(float(g.dz_km.median()), 3),
                                 dz_absmedian_km=round(float(g.dz_km.abs().median()), 3),
                                 dz_p90abs_km=round(float(g.dz_km.abs().quantile(0.9)), 3),
                                 dt_median_s=round(float(g.dt_s.median()), 4),
                                 gt1km_3d_pct=round(100 * float((np.hypot(g.dh_km, g.dz_km) > 1).mean()), 1)))
    return pd.DataFrame(rows), pd.concat(allsh, ignore_index=True)


# ------------------------------------------------------------------ 5 rule checks
def rule_checks():
    tol = float(T["merge"]["smooth_rule_tol"])
    out = []
    for name, (old, new, nf, sm) in RUNSET.items():
        for lab, reg in (("v0.3.1", old), ("v0.4.0", new)):
            f = RUNS / reg / "summary_runs.csv"
            if not f.exists():
                continue
            s = pd.read_csv(f)
            s = s[s.tag.str.match(r"d200_s\d+$")]
            if name != "M0":
                s = s[s.smooth_x.isin([x[0] for x in T["merge"]["tile_smooth_scan"]])]
            best = s.rms_wt_final.min()
            ok = s[s.rms_wt_final <= (1 + tol) * best]
            pick = int(ok.smooth_x.max())
            for r in s.sort_values("smooth_x").itertuples():
                out.append(dict(inversion=name, version=lab, smooth=int(r.smooth_x), rms_wt_final=r.rms_wt_final,
                                rel_to_best_pct=round(100 * (r.rms_wt_final / best - 1), 2),
                                within_tol=bool(r.rms_wt_final <= (1 + tol) * best), rule_choice=pick,
                                kept_choice=sm))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 6 resources
def resources():
    rows = []
    for d in sorted(RUNS.glob("margin_m0_v3*/d200_s*")) + sorted(RUNS.glob("*_m0t*/d200_s*")) + \
            sorted(RUNS.glob("margin_m0_v3*/check01")) + sorted(RUNS.glob("*_m0t*/check01")):
        if ".partial" in d.name or not (d / "time.txt").exists():
            continue
        tm = (d / "time.txt").read_text().split()
        try:
            rows.append(dict(run=f"{d.parent.name}/{d.name}", wall_min=round(float(tm[-6]) / 60, 1),
                             user_min=round(float(tm[-4]) / 60, 1), maxrss_gb=round(float(tm[-2]) / 1024 ** 2, 2)))
        except (ValueError, IndexError):
            rows.append(dict(run=f"{d.parent.name}/{d.name}", wall_min=np.nan, user_min=np.nan, maxrss_gb=np.nan))
    st = Path(RT["out"]) / "chain.state"
    js = pd.read_csv(st, sep="\t", header=None, names=["job", "start", "end", "seconds"]) if st.exists() else None
    return pd.DataFrame(rows), js


# ------------------------------------------------------------------ figures
def coast():
    try:
        import pygmt
        rel = pygmt.datasets.load_earth_relief("05m", region=[-133, -119, 38.5, 52], registration="gridline")
        return rel.lon.values, rel.lat.values, rel.values
    except Exception as e:  # noqa: BLE001
        print("no coastline:", e)
        return None


def fig_maps(D, new, CO, which="d", fname="maps_diff.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon, lat, z = new.longitude.values, new.latitude.values, new.depth.values
    depths = [3, 9, 15, 21, 29, 40]
    fig, axs = plt.subplots(3, len(depths), figsize=(2.6 * len(depths), 9.6), sharex=True, sharey=True)
    specs = [("Vp", "dVp/Vp (%)", 2.0), ("Vs", "dVs/Vs (%)", 2.0), ("VpVs", "d(Vp/Vs)", 0.03)]
    for i, (k, lab, v) in enumerate(specs):
        for j, dd in enumerate(depths):
            ax = axs[i, j]
            iz = int(np.argmin(np.abs(z - dd)))
            a = D[f"{which}{k}"][iz]
            im = ax.pcolormesh(lon, lat, a, cmap="RdBu_r" if k != "VpVs" else "PuOr_r", vmin=-v, vmax=v,
                               shading="nearest", rasterized=True)
            rp = (new.resolution_p.values[iz] > 0) if k != "Vs" else (new.resolution_s.values[iz] > 0)
            if rp.any():
                ax.contour(lon, lat, rp.astype(float), levels=[0.5], colors="k", linewidths=0.6)
            if CO is not None:
                ax.contour(CO[0], CO[1], CO[2], levels=[0], colors="0.5", linewidths=0.4)
            ax.set_xlim(lon.min(), lon.max())
            ax.set_ylim(lat.min(), lat.max())
            ax.set_aspect(1 / np.cos(np.radians(45)))
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(f"{z[iz]:g} km", fontsize=8)
        fig.colorbar(im, ax=axs[i, :], shrink=0.8, label=lab, pad=0.01)
    t = "v0.4.0 - v0.3.1" if which == "d" else "noise floor: v0.4.0 (1e-6 s perturbed) - v0.4.0"
    fig.suptitle(f"{t}; black contour = checkerboard-resolved nodes (resolution > 0)", fontsize=9)
    fig.savefig(OUT / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


SECTIONS = {"A: 41.0 N": ("lat", 41.0), "B: 44.0 N": ("lat", 44.0), "C: 47.0 N": ("lat", 47.0),
            "D: 124.5 W": ("lon", -124.5)}


def fig_sections(D, new, which="d", fname="sections_diff.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon, lat, z = new.longitude.values, new.latitude.values, new.depth.values
    zm = z <= 55
    fig, axs = plt.subplots(len(SECTIONS), 3, figsize=(13, 2.4 * len(SECTIONS)))
    specs = [("Vp", "dVp/Vp (%)", 2.0), ("Vs", "dVs/Vs (%)", 2.0), ("VpVs", "d(Vp/Vs)", 0.03)]
    for i, (name, (ax_, val)) in enumerate(SECTIONS.items()):
        for j, (k, lab, v) in enumerate(specs):
            a = D[f"{which}{k}"]
            r = new.resolution_p.values if k != "Vs" else new.resolution_s.values
            if ax_ == "lat":
                ii = int(np.argmin(np.abs(lat - val)))
                sec, rr, x = a[:, ii, :], r[:, ii, :], lon
            else:
                ii = int(np.argmin(np.abs(lon - val)))
                sec, rr, x = a[:, :, ii], r[:, :, ii], lat
            ax = axs[i, j]
            im = ax.pcolormesh(x, z[zm], sec[zm], cmap="RdBu_r" if k != "VpVs" else "PuOr_r", vmin=-v, vmax=v,
                               shading="nearest", rasterized=True)
            if (rr[zm] > 0).any():
                ax.contour(x, z[zm], (rr[zm] > 0).astype(float), levels=[0.5], colors="k", linewidths=0.6)
            ax.set_ylim(55, -3)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{name}: {lab}", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(("v0.4.0 - v0.3.1" if which == "d" else "noise floor") + "; black = resolved; depth km bsl",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_daytime(bins, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, ep in zip(axs[:2], ("listed", "found")):
        for ver, c in (("v0.3.1", "tab:red"), ("v0.4.0", "tab:blue")):
            g = bins[(bins.epoch == ep) & (bins.version == ver)]
            ax.plot(24 * g.bin, g.median_res_s, "o-", c=c, ms=3, label=f"{ver} final residual (median)")
        g = bins[(bins.epoch == ep) & (bins.version == "v0.4.0")]
        ax.plot(24 * g.bin, -g.mean_corr_s, "k--", lw=0.8, label="- correction (mean)")
        ax.axhline(0, c="0.7", lw=0.5)
        ax.set_xlabel("UTC hour of the pick")
        ax.set_ylabel("s")
        ax.set_title(f"re-timed picks, epoch class '{ep}' (all inversions pooled)", fontsize=8)
        ax.legend(fontsize=6)
    p = st.pivot_table(index="station", columns="version", values="slope_s_per_day")
    if {"v0.3.1", "v0.4.0"} <= set(p.columns):
        axs[2].scatter(p["v0.3.1"], p["v0.4.0"], s=10)
        lim = np.nanmax(np.abs(p.values)) * 1.05
        axs[2].plot([-lim, lim], [-lim, lim], c="0.7", lw=0.5)
        axs[2].axhline(0, c="0.7", lw=0.5)
        axs[2].axvline(0, c="0.7", lw=0.5)
        axs[2].set_xlabel("v0.3.1 residual slope vs time of day (s/day)")
        axs[2].set_ylabel("v0.4.0 (s/day)")
        axs[2].set_title("re-timed stations (>= 30 re-timed picks)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "daytime_trend.png", dpi=120)
    plt.close(fig)


def fig_hypo(allsh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, col, lab in ((axs[0], "dh_km", "horizontal shift (km)"), (axs[1], "dz_km", "depth shift (km, + deeper)")):
        for (cmp_, c) in (("v0.4.0 - v0.3.1", "tab:red"), ("noise floor (nf - v0.4.0)", "0.4")):
            x = allsh[allsh.comparison == cmp_][col]
            ax.hist(x.clip(-3, 3) if col == "dz_km" else x.clip(0, 3), bins=120, histtype="step", color=c,
                    label=f"{cmp_} (n {len(x)})", log=True)
        ax.set_xlabel(lab + ", clipped at 3 km")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "hypocentre_shifts.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ html
def html_table(df, n=None):
    d = df if n is None else df.head(n)
    return d.to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:.4g}", na_rep="")


def img(name):
    b = base64.b64encode((OUT / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{name}">'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GIT.mkdir(parents=True, exist_ok=True)
    t1, t2, u, ds = picks_tables()
    t1.to_csv(OUT / "picks_retimed_by_inversion.csv", index=False)
    t2.to_csv(OUT / "picks_retimed_by_station.csv", index=False)
    qc = json.loads((Path(RT["out"]) / "qc_check" / "qc_summary.json").read_text())
    qcc = pd.read_csv(Path(RT["out"]) / "qc_check" / "qc_counts.csv")
    mc = mad_cuts()
    mc.to_csv(OUT / "tomodd_residual_cuts.csv", index=False)
    rt, rpiv, per, rt_sta = residuals(ds)
    rt.to_csv(OUT / "residual_rms_long.csv", index=False)
    rpiv.to_csv(OUT / "residual_rms.csv", index=False)
    st, sp, bins, _ = daytime(per, ds)
    st.to_csv(OUT / "retimed_station_residuals_long.csv", index=False)
    sp.to_csv(OUT / "retimed_station_residuals.csv", index=False)
    bins.to_csv(OUT / "daytime_bins.csv", index=False)
    fig_daytime(bins, st)
    md, mdz, D, new, old, nf, same_mask = model_diff()
    md.to_csv(OUT / "model_diff_shares.csv", index=False)
    mdz.to_csv(OUT / "model_diff_by_depth.csv", index=False)
    where = regions_of_change(D, new)
    where.to_csv(OUT / "model_diff_where.csv", index=False)
    CO = coast()
    fig_maps(D, new, CO, "d", "maps_diff.png")
    fig_maps(D, new, CO, "n", "maps_noise.png")
    fig_sections(D, new, "d", "sections_diff.png")
    fig_sections(D, new, "n", "sections_noise.png")
    hs, allsh = hypo_shifts()
    hs.to_csv(OUT / "hypocentre_shifts.csv", index=False)
    allsh.to_csv(OUT / "hypocentre_shifts_events.csv.gz", index=False, compression="gzip")
    fig_hypo(allsh)
    rc = rule_checks()
    rc.to_csv(OUT / "rule_checks_smoothing.csv", index=False)
    res, js = resources()
    res.to_csv(OUT / "run_resources.csv", index=False)
    ver = {}
    for name, (old_, new_, nf_, sm) in RUNSET.items():
        f = RUNS / new_ / "inputs" / f"verify_vs_{old_}.json"
        ver[name] = json.loads(f.read_text()) if f.exists() else None
    (OUT / "input_verification.json").write_text(json.dumps(ver, indent=1))
    attrs_new = {k: new.attrs.get(k) for k in ("reference_fallback_fraction_p", "reference_fallback_fraction_s",
                                                "vs_reverted_nodes", "vs_capped_nodes", "gated_tiles",
                                                "excluded_tiles", "covered_nodes_p", "provenance_git_commit")}
    attrs_old = {k: old.attrs.get(k) for k in attrs_new}
    summary = dict(unique_picks=u, qc=dict(kept_v031=qc["kept_v031"], kept_retimed=qc["kept_retimed"],
                                            changed=qc["changed"], transitions=qc["transitions"]),
                   merge_attrs_v040=attrs_new, merge_attrs_v031=attrs_old, masks_identical=same_mask)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # ---- html
    css = ("body{font-family:system-ui,sans-serif;max-width:1150px;margin:auto;padding:0 16px;color:#222;"
           "background:#fff}h1{font-size:1.4em}h2{font-size:1.15em;margin-top:1.6em;border-bottom:1px solid #ccc}"
           "table.t{border-collapse:collapse;font-size:12px;margin:6px 0}table.t td,table.t th{border:1px solid "
           "#ddd;padding:2px 6px;text-align:right}img{max-width:100%}p,li{font-size:14px;line-height:1.45}"
           ".note{color:#555;font-size:13px}")
    H = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
         f"initial-scale=1'><title>Model v0.4.0 report</title><style>{css}</style></head><body>",
         "<h1>Velocity model v0.4.0: v0.3.1 re-inverted with the re-timed picks</h1>",
         f"<p class='note'>{RT['description']}. Generated by workflow/10_tomography/b2_v040_analyse.py "
         f"(branch tomo/v040-retimed). Same events, picks, weights, OBS S statics, P-4 rule, damping 200, smoothing "
         "and merge rules as v0.3.1; only the pick times differ (D-B072/D-B075). ridges_north was not rerun: it is "
         "out of the merge (D-B050).</p>",
         "<h2>1. Picks re-timed in the tomography</h2>",
         f"<p>Unique B1b tomography picks: {u['b1b_picks']:,}; re-timed {u['retimed']:,} "
         f"({100 * u['retimed'] / u['b1b_picks']:.1f} %) on {u['stations']} stations ({u['obs']} on OBS); "
         f"|correction| &gt; 0.02 s: {u['gt_0.02s']:,}, &gt; 0.05 s: {u['gt_0.05s']:,}, &gt; 0.1 s: {u['gt_0.1s']:,}, "
         f"&gt; 0.3 s: {u['gt_0.3s']:,}; epoch class listed {u['listed']:,}, found {u['found']:,}.</p>",
         html_table(t1),
         "<p>Stations with re-timed picks (per inversion, largest counts first; full table "
         "picks_retimed_by_station.csv):</p>", html_table(t2[t2.region != "M0"], 40),
         "<h3>MAD cuts</h3>",
         "<p>B1 pick QC (D-B009: |res - median| &gt; 4 sigma_MAD; S-P 5 sigma_MAD), v0.3.1 choice kept; the "
         f"rule rerun with the re-timed picks changes {qc['changed']} of {qc['candidates']:,} candidate picks "
         f"(kept {qc['kept_v031']:,} -> {qc['kept_retimed']:,}).</p>", html_table(qcc),
         html_table(pd.DataFrame(qc["transitions"])),
         "<p>tomoDD's own cuts (WRCT 8 then 6 x MAD, WDCT 10 km): absolute picks of the kept events that are "
         "missing from the final tomoDD.res, and the last cutoff values.</p>", html_table(mc),
         "<h2>2. Residuals (final iteration; 'start' = first iteration in the starting model)</h2>",
         "<p>rms (s) by inversion, phase and station class. Re-timed stations = stations with at least one "
         "re-timed tomography pick.</p>", html_table(rpiv),
         "<h3>Re-timed stations: time-of-day trend</h3>", img("daytime_trend.png"),
         html_table(sp.sort_values("n_v0.4.0", ascending=False) if "n_v0.4.0" in sp else sp, 50),
         "<h2>3. Model differences v0.4.0 - v0.3.1 (merged models)</h2>",
         f"<p>Resolution masks identical in both models: {same_mask}. Noise floor = the v0.4.0 chain rerun with "
         "N(0, 1e-6 s) added to every observed time (M0 and tiles), merged with the same rules. 'above noise' = "
         "|difference| &gt; 2 x the largest |noise difference| among the node and its 26 neighbours.</p>",
         html_table(md), html_table(mdz), img("maps_diff.png"), img("maps_noise.png"), img("sections_diff.png"),
         img("sections_noise.png"),
         "<p>Where resolved nodes moved by more than 1 %:</p>", html_table(where, 40),
         "<h2>4. Hypocentre shifts of the tomography events (tomoDD final)</h2>", img("hypocentre_shifts.png"),
         html_table(hs),
         "<h2>5. Rule checks (smoothing, D-B046 rule: largest within 1.5 % of the best weighted rms)</h2>",
         html_table(rc),
         "<h2>6. Inputs, P-4 and resources</h2>",
         "<pre style='font-size:11px;white-space:pre-wrap'>" + json.dumps(ver, indent=1)[:6000] + "</pre>",
         html_table(res), html_table(js) if js is not None else "",
         "<p class='note'>Merge attributes v0.4.0: " + json.dumps(attrs_new, default=str) + "<br>v0.3.1: " +
         json.dumps(attrs_old, default=str) + "</p>",
         "</body></html>"]
    (OUT / "report.html").write_text("\n".join(H))
    for f in OUT.iterdir():
        if f.is_file() and f.stat().st_size < 5 * 1024 ** 2 and not f.name.endswith(".DONE"):
            shutil.copy2(f, GIT / f.name)
    write_manifest("b2_v040_analyse", OUT, outputs=sorted(p for p in OUT.iterdir() if p.is_file()),
                   params={"retime": RT}, counts=summary["unique_picks"])
    (OUT / "ANALYSE.DONE").write_text("done\n")
    print("report:", OUT / "report.html")


if __name__ == "__main__":
    main()
