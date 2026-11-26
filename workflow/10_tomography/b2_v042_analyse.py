"""v0.4.2 report (D-B099): v0.4.1 without self-paired differential times, re-inverted with the 2026-11-26 tomoDD freeze.

Sections (tables <v042.out>/report/*.csv, figures *.png; the self-contained report goes to <v042.staging>/report_v0.4.2.html;
small files copied to data/tomo_v1/v042/):
  1 freeze identity: v0.4.1 deterministic runs rerun with the 2026-11-26 binaries (full: M0, oregon, blanco; first
    iterations: mendocino_gorda_c, washington)
  2 inputs: rebuilt tiles = v0.4.1's minus exactly the self-pair blocks (b2_v042_verify_inputs.py)
  3 model v0.4.2 - v0.4.1 (merged inversion values) on resolved / tile-covered nodes against the numerical noise floors
    of both chains (N(0, 1e-6 s) on every observed time), by depth; maps and sections of Vp, Vs, Vp/Vs
  4 the 8 self-paired events and their co-located twins: start, v0.4.1 and v0.4.2 final hypocentres
  5 floor stops and events above the true surface (must be 0) per stage
  6 hypocentre shifts by stage and region (+ noise floors)
  7 residual rms by phase and station class
  8 product (staged netCDF: read-back checks, sha256, md5) and run times
  python b2_v042_analyse.py            (default env; the merges and the staged product must exist)
"""
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
import b2_v040_analyse as A  # noqa: E402
import b2_v041_analyse as F41  # noqa: E402  (floor_log, final_positions, reloc_w, html_table)

P = load_params()
T, V, V41 = P["tomodd"], P["v042"], P["v041"]
RUNS = Path(T["runs"])
VOUT = Path(V["out"])
OUT = VOUT / "report"
STAGING = Path(V["staging"])
GIT = REPO / "data" / "tomo_v1" / "v042"
STORE = Path(T["models"]["store"]) / T["models"]["name"]
SM = {"M0": 480, "mendocino_gorda_c": 480, "blanco": 20, "oregon": 480, "washington": 480}
REBUILT = ("mendocino_gorda_c", "blanco", "washington")
# stage: (v0.4.1 region, v0.4.2 region, v0.4.2 noise floor, v0.4.1 noise floor)
STAGES = {"M0": ("margin_m0_v4", "margin_m0_v4", "margin_m0_v4nf", "margin_m0_v4nf")}
STAGES.update({t: (f"{t}_m0f", f"{t}_m0g" if t in REBUILT else f"{t}_m0f", f"{t}_m0gnf" if t in REBUILT else f"{t}_m0fnf",
                   f"{t}_m0fnf") for t in ("mendocino_gorda_c", "blanco", "oregon", "washington")})
SELF = {"washington": [28514, 40700, 53148, 55170, 80094, 94266], "mendocino_gorda_c": [156844], "blanco": [81324]}
TXT = []


def rd(region, stage):
    return RUNS / region / f"d200_s{SM[stage]}"


def fmt(x):
    return "dropped" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:g}"


def img(name):
    return f'<img src="data:image/png;base64,{base64.b64encode((OUT / name).read_bytes()).decode()}" alt="{name}">'


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 1 identity
def identity():
    rows = []
    for d in sorted((VOUT / "identity").glob("*_d200_s*")):
        c = (d / "compare.txt").read_text().splitlines() if (d / "compare.txt").exists() else []
        tm = (d / "time.txt").read_text().split() if (d / "time.txt").exists() else []
        b = [x for x in c if x.startswith("binary")]
        rows.append(dict(run=d.name, binary=b[0].split()[1].rsplit("/", 1)[-1] if b else "",
                         binary_md5=b[0].split()[-1][:12] if b else "",
                         exit=(d / "exit.txt").read_text().strip() if (d / "exit.txt").exists() else "running",
                         files_compared=int(c[-1].split()[1]) if c and c[-1].startswith("files") else np.nan,
                         different=[x.split()[1] for x in c if x.startswith("DIFFERENT") and "(text" not in x],
                         log_text_differs=any(x.startswith("DIFFERENT tomoDD-SE.log") for x in c),
                         wall_min=round(float(tm[0]) / 60, 1) if len(tm) > 2 and tm[1] == "s" else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2 inputs
def input_checks():
    rows = []
    for t in REBUILT:
        for new, old in ((f"{t}_m0g", f"{t}_m0f"), (f"{t}_m0gnf", f"{t}_m0fnf")):
            f = RUNS / new / "inputs" / f"verify_vs_{old}.json"
            if not f.exists():
                rows.append(dict(new=new, old=old, checked=False))
                continue
            j = json.loads(f.read_text())
            dt = j["files"].get("dt_unperturbed.ct") or j["files"].get("dt.ct")
            rows.append(dict(new=new, old=old, checked=True, problems=j["problems"],
                             identical_files=sum(1 for v in j["files"].values() if v.get("identical") is True),
                             dt_equals_old_minus_self_pairs=dt["equals_old_minus_self_pairs"],
                             self_pairs_removed=dt["self_pairs_removed"], data_lines_removed=dt["self_pair_data_lines_removed"],
                             pairs_old_new=[dt["pairs_old"], dt["pairs_new"]], data_old_new=[dt["data_old"], dt["data_new"]],
                             self_pairs_left=dt["self_pairs_left"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3 models
def model_diff():
    import xarray as xr
    new = xr.open_dataset(VOUT / "product" / "merged_v042.nc")
    old = xr.open_dataset(Path(V41["out"]) / "product" / "merged_v041.nc")
    nf = xr.open_dataset(VOUT / "noise" / "merged_v042nf.nc")
    onf = xr.open_dataset(Path(V41["out"]) / "noise" / "merged_v041nf.nc")
    D = {}
    for k in ("Vp", "Vs"):
        D[f"d{k}"] = 100 * (new[k].values / old[k].values - 1)
        D[f"n{k}"] = 100 * (nf[k].values / new[k].values - 1)
        D[f"o{k}"] = 100 * (onf[k].values / old[k].values - 1)
    D["dVpVs"] = new.Vp.values / new.Vs.values - old.Vp.values / old.Vs.values
    D["nVpVs"] = nf.Vp.values / nf.Vs.values - new.Vp.values / new.Vs.values
    D["oVpVs"] = onf.Vp.values / onf.Vs.values - old.Vp.values / old.Vs.values
    from scipy.ndimage import maximum_filter
    z = new.depth.values
    res = {"Vp": new.resolution_p.values > 0, "Vs": new.resolution_s.values > 0}
    res["VpVs"] = res["Vp"] & res["Vs"]
    cov = {"Vp": new.tile_weight_p.values > 0, "Vs": new.tile_weight_s.values > 0}
    cov["VpVs"] = cov["Vp"] & cov["Vs"]
    same_mask = bool((old.resolution_p.values == new.resolution_p.values).all() and
                     (old.resolution_s.values == new.resolution_s.values).all())
    unit = {"Vp": "pct", "Vs": "pct", "VpVs": "abs"}
    rows, tot, above = [], [], {}
    for nodes, MS in (("resolved", res), ("tile-covered", cov)):
        for k in ("Vp", "Vs", "VpVs"):
            nl = maximum_filter(np.maximum(np.abs(D[f"n{k}"]), np.abs(D[f"o{k}"])), size=3, mode="nearest")
            above[(nodes, k)] = (np.abs(D[f"d{k}"]) > 2 * nl) & MS[k]
            for i, zz in enumerate(list(z) + ["all"]):
                m = MS[k][i] if zz != "all" else MS[k]
                if m.sum() == 0:
                    continue
                sel = (lambda a: a[i][m]) if zz != "all" else (lambda a: a[m])
                d, n, o, L = sel(D[f"d{k}"]), sel(D[f"n{k}"]), sel(D[f"o{k}"]), sel(nl)
                r = dict(quantity=k, unit=unit[k], nodes=nodes, depth_km=zz, n_nodes=int(m.sum()),
                         rms=round(float(np.sqrt((d ** 2).mean())), 4), max_abs=round(float(np.abs(d).max()), 4),
                         changed_nodes=int((d != 0).sum()),
                         noise_v042_rms=round(float(np.sqrt((n ** 2).mean())), 4),
                         noise_v041_rms=round(float(np.sqrt((o ** 2).mean())), 4),
                         noise_max_abs=round(float(max(np.abs(n).max(), np.abs(o).max())), 4),
                         above_noise_nodes=int((np.abs(d) > 2 * L).sum()),
                         above_noise_pct=round(100 * float((np.abs(d) > 2 * L).mean()), 2))
                (tot if zz == "all" else rows).append(r)
    lon, lat = new.longitude.values, new.latitude.values
    ev = pd.concat([pd.read_csv(RUNS / STAGES[st][1] / "inputs" / "events.csv").query("ID in @ids")
                    for st, ids in SELF.items()])
    big = []
    for (nodes, k), msk in above.items():
        iz, iy, ix = np.nonzero(msk)
        for a, b_, c in zip(iz, iy, ix):
            dh = np.hypot((ev.lat.values - lat[b_]) * 111.19, (ev.lon.values - lon[c]) * 111.19 * np.cos(np.radians(lat[b_])))
            vo = (old.Vp.values / old.Vs.values if k == "VpVs" else old[k].values)[a, b_, c]
            vn = (new.Vp.values / new.Vs.values if k == "VpVs" else new[k].values)[a, b_, c]
            big.append(dict(nodes=nodes, quantity=k, depth_km=float(z[a]), lat=round(float(lat[b_]), 2),
                            lon=round(float(lon[c]), 3), km_to_nearest_self_paired_event=round(float(dh.min()), 1),
                            v041=round(float(vo), 4), v042=round(float(vn), 4), diff=round(float(D[f"d{k}"][a, b_, c]), 4)))
    big = pd.DataFrame(big)
    if len(big):
        big = big.reindex(big["diff"].abs().sort_values(ascending=False).index)
    return pd.DataFrame(rows), pd.DataFrame(tot), big, D, new, old, same_mask


def fig_maps(D, new, CO, fname, which, ev=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon, lat, z = new.longitude.values, new.latitude.values, new.depth.values
    depths = [1, 5, 9, 15, 21, 29]
    specs = [("Vp", "dVp/Vp (%)", 1.0, "RdBu_r"), ("Vs", "dVs/Vs (%)", 1.0, "RdBu_r"), ("VpVs", "d(Vp/Vs)", 0.02, "PuOr_r")]
    fig, axs = plt.subplots(3, len(depths), figsize=(2.6 * len(depths), 10.2), sharex=True, sharey=True)
    for i, (k, lab, v, cm) in enumerate(specs):
        for j, dd in enumerate(depths):
            ax = axs[i, j]
            iz = int(np.argmin(np.abs(z - dd)))
            im = ax.pcolormesh(lon, lat, D[f"{which}{k}"][iz], cmap=cm, vmin=-v, vmax=v, shading="nearest", rasterized=True)
            rr = (new.resolution_s.values if k == "Vs" else new.resolution_p.values)[iz] > 0
            if rr.any():
                ax.contour(lon, lat, rr.astype(float), levels=[0.5], colors="k", linewidths=0.5)
            if CO is not None:
                ax.contour(CO[0], CO[1], CO[2], levels=[0], colors="0.5", linewidths=0.4)
            if ev is not None:
                ax.scatter(ev.lon, ev.lat, s=10, marker="x", c="m", linewidths=0.8)
            ax.set_aspect(1 / np.cos(np.radians(45)))
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(f"{z[iz]:g} km", fontsize=8)
        fig.colorbar(im, ax=axs[i, :], shrink=0.8, label=lab, pad=0.01)
    t = "v0.4.2 - v0.4.1" if which == "d" else "noise floor of v0.4.2 (1e-6 s perturbed chain - v0.4.2)"
    fig.suptitle(f"{t} (merged inversion values); colour saturates at the bar ends; black = checkerboard-resolved; "
                 "magenta x = the 8 self-paired events (v0.4.1 start)", fontsize=9)
    fig.savefig(OUT / fname, dpi=105, bbox_inches="tight")
    plt.close(fig)


SECTIONS = {"Vancouver Island (lat 49.7)": ("lat", 49.7), "Puget Sound (lat 48.2)": ("lat", 48.2),
            "Blanco (lat 44.0)": ("lat", 44.0), "Mendocino (lat 40.6)": ("lat", 40.6), "lon -124.5": ("lon", -124.5)}


def fig_sections(D, new, fname, which, ev=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon, lat, z = new.longitude.values, new.latitude.values, new.depth.values
    zm = z <= 55
    specs = [("Vp", "dVp/Vp (%)", 2.0, "RdBu_r"), ("Vs", "dVs/Vs (%)", 2.0, "RdBu_r"), ("VpVs", "d(Vp/Vs)", 0.03, "PuOr_r")]
    fig, axs = plt.subplots(len(SECTIONS), 3, figsize=(13, 2.5 * len(SECTIONS)))
    for i, (name, (ax_, val)) in enumerate(SECTIONS.items()):
        for j, (k, lab, v, cm) in enumerate(specs):
            a = D[f"{which}{k}"]
            r = new.resolution_s.values if k == "Vs" else new.resolution_p.values
            if ax_ == "lat":
                ii = int(np.argmin(np.abs(lat - val)))
                sec, rr, x = a[:, ii, :], r[:, ii, :], lon
            else:
                ii = int(np.argmin(np.abs(lon - val)))
                sec, rr, x = a[:, :, ii], r[:, :, ii], lat
            ax = axs[i, j]
            im = ax.pcolormesh(x, z[zm], sec[zm], cmap=cm, vmin=-v, vmax=v, shading="nearest", rasterized=True)
            if (rr[zm] > 0).any():
                ax.contour(x, z[zm], (rr[zm] > 0).astype(float), levels=[0.5], colors="k", linewidths=0.6)
            if ev is not None:
                e = ev[np.abs((ev.lat if ax_ == "lat" else ev.lon) - val) < (0.15 if ax_ == "lat" else 0.2)]
                ax.scatter(e.lon if ax_ == "lat" else e.lat, e.depth, s=12, marker="x", c="m", linewidths=0.8)
            ax.set_ylim(55, -3)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{name}: {lab}", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(("v0.4.2 - v0.4.1" if which == "d" else "noise floor of v0.4.2") +
                 "; black = resolved; magenta x = self-paired events near the section; depth km bsl", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=105, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 4 self-paired events
def self_pairs():
    """The 8 events and their co-located twins (same start hypocentre): start, v0.4.1 and v0.4.2 final hypocentres."""
    rows, pts = [], []
    for st, ids in SELF.items():
        old, new = STAGES[st][0], STAGES[st][1]
        ev = pd.read_csv(RUNS / new / "inputs" / "events.csv")
        key = ev.set_index(["lat", "lon", "depth_bsl_km"])
        g0, g1 = A.reloc(rd(old, st)).set_index("ID"), A.reloc(rd(new, st)).set_index("ID")
        for i in ids:
            r = ev[ev.ID == i].iloc[0]
            twins = [int(x) for x in key.loc[(r.lat, r.lon, r.depth_bsl_km)].ID if x != i]
            for role, j in [("self-paired", i)] + [("twin", t) for t in twins]:
                d = dict(stage=st, event=j, role=role, of=i, region=r.region, start_lat=r.lat, start_lon=r.lon,
                         start_depth=r.depth_bsl_km)
                for lab, g in (("v041", g0), ("v042", g1)):
                    if j in g.index:
                        x = g.loc[j]
                        dx = (x.lon - r.lon) * 111.19 * np.cos(np.radians(r.lat))
                        dy = (x.lat - r.lat) * 111.19
                        d[f"{lab}_lat"] = round(float(x.lat), 5)
                        d[f"{lab}_lon"] = round(float(x.lon), 5)
                        d[f"{lab}_depth"] = round(float(x.depth), 3)
                        d[f"{lab}_from_start_km"] = round(float(np.sqrt(dx ** 2 + dy ** 2 + (x.depth - r.depth_bsl_km) ** 2)), 2)
                    else:
                        d[f"{lab}_depth"] = "dropped"
                rows.append(d)
                pts.append(dict(lat=r.lat, lon=r.lon, depth=r.depth_bsl_km))
    t = pd.DataFrame(rows)
    sep = []
    for (st, i), g in t.groupby(["stage", "of"]):
        s, tw = g[g.role == "self-paired"].iloc[0], g[g.role == "twin"]
        for _, w in tw.iterrows():
            row = dict(stage=st, self_paired=i, twin=w.event)
            for lab in ("v041", "v042"):
                if isinstance(s.get(f"{lab}_depth"), float) and isinstance(w.get(f"{lab}_depth"), float):
                    row[f"{lab}_separation_km"] = round(float(np.sqrt(((s[f"{lab}_lat"] - w[f"{lab}_lat"]) * 111.19) ** 2 + (
                        (s[f"{lab}_lon"] - w[f"{lab}_lon"]) * 111.19 * np.cos(np.radians(s.start_lat))) ** 2 +
                        (s[f"{lab}_depth"] - w[f"{lab}_depth"]) ** 2)), 2)
            sep.append(row)
    return t, pd.DataFrame(sep), pd.DataFrame(pts).drop_duplicates()


# ------------------------------------------------------------------ 5 floor / above surface
def floor_table():
    rows = []
    for st, (old, new, nf, onf) in STAGES.items():
        if not (rd(new, st) / "Output_Files" / "tomoDD.reloc_geo").exists():
            continue
        f1 = F41.final_positions(new, SM[st], True)
        f0 = F41.final_positions(old, SM[st], True)
        h1, aq1, nm1, it1 = F41.floor_log(rd(new, st))
        h0, aq0, nm0, it0 = F41.floor_log(rd(old, st))
        l0, l1 = A.parse_log(rd(old, st)), A.parse_log(rd(new, st))
        rows.append(dict(stage=st, run=f"{new}/d200_s{SM[st]}", rerun=new != old, events_start=sum(
            1 for _ in open(rd(new, st) / "Input_Files" / "event.dat")),
            events_final_v041=len(f0), events_final_v042=len(f1), floor_lines_v041=len(h0), floor_lines_v042=len(h1),
            events_hit_v041=int(h0.ID.nunique()), events_hit_v042=int(h1.ID.nunique()), floor_not_met_v042=len(nm1),
            at_floor_final_v041=int(f0.at_floor.sum()), at_floor_final_v042=int(f1.at_floor.sum()),
            above_floor_by_more_than_2m_v042=int((f1.above_floor_km > F41.EPS_KM).sum()),
            air_quakes_v042=len(aq1), iterations_v042=it1,
            above_true_surface_final_v041=int((f0.below_surface_km < 0).sum()),
            above_true_surface_final_v042=int((f1.below_surface_km < 0).sum()),
            rms_abs_final_v041=l0["rms_abs_final"], rms_abs_final_v042=l1["rms_abs_final"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6 hypocentres
def hypo_shifts():
    rows = []
    for st, (old, new, nf, onf) in STAGES.items():
        if new == old or not (rd(new, st) / "Output_Files" / "tomoDD.reloc_geo").exists():
            continue
        ev = pd.read_csv(RUNS / new / "inputs" / "events.csv")[["ID", "region"]]
        pairs = [("v0.4.2 - v0.4.1", rd(new, st), rd(old, st))]
        if (rd(nf, st) / "Output_Files" / "tomoDD.reloc_geo").exists():
            pairs.append(("noise floor v0.4.2 (nf - v0.4.2)", rd(nf, st), rd(new, st)))
        if (rd(onf, st) / "Output_Files" / "tomoDD.reloc_geo").exists():
            pairs.append(("noise floor v0.4.1 (nf - v0.4.1)", rd(onf, st), rd(old, st)))
        for lab, x_, y_ in pairs:
            x, y = A.reloc(x_), A.reloc(y_)
            m = x.merge(y, on="ID", suffixes=("", "_0")).merge(ev, on="ID", how="left")
            m["dh_km"] = np.hypot((m.lat - m.lat_0) * 111.19, (m.lon - m.lon_0) * 111.19 * np.cos(np.radians(m.lat)))
            m["dz_km"] = m.depth - m.depth_0
            sp = m.ID.isin(SELF[st])
            for reg, g in list(m.groupby("region")) + [("all", m), ("all but the self-paired events", m[~sp])]:
                rows.append(dict(stage=st, comparison=lab, region=reg, events=len(g),
                                 dh_median_km=round(float(g.dh_km.median()), 4), dh_p90_km=round(float(g.dh_km.quantile(0.9)), 3),
                                 dz_absmedian_km=round(float(g.dz_km.abs().median()), 4),
                                 dz_p90abs_km=round(float(g.dz_km.abs().quantile(0.9)), 3),
                                 max_3d_km=round(float(np.hypot(g.dh_km, g.dz_km).max()), 2),
                                 gt1km_3d=int((np.hypot(g.dh_km, g.dz_km) > 1).sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 7 residuals
def residuals():
    rows = []
    for st, (old, new, nf, onf) in STAGES.items():
        if new == old or not (rd(new, st) / "Output_Files" / "tomoDD.res").exists():
            continue
        codes = pd.read_csv(RUNS / new / "inputs" / "station_codes.csv")[["code", "is_obs"]]
        for lab, reg in (("v0.4.1", old), ("v0.4.2", new)):
            r = A.read_res(rd(reg, st))
            selfline = (r.c1 == r.c2) & r.c1.isin(SELF[st]) & (r.dt == 0)
            n_self = int(selfline.sum())
            a, dtr = A.abs_res(rd(reg, st))
            if n_self:            # the self-pair lines: tomoDD's 'absolute' times of 0 s, not picks
                rs = r[selfline]
                bad = set(zip(rs.c1, rs.code, np.where(rs.idx == 3, "P", "S"), (rs.res_ms / 1000).round(6)))
                a = a[[(e, c, p, round(x, 6)) not in bad for e, c, p, x in zip(a.event_ID, a.code, a.phase, a.res_s)]]
            a = a.merge(codes, on="code", how="left")
            a["cls"] = np.where(a.is_obs, "OBS", "land")
            for (ph, cls), g in list(a.groupby(["phase", "cls"])) + [((ph, "all"), g) for ph, g in a.groupby("phase")]:
                rows.append(dict(stage=st, version=lab, phase=ph, cls=cls, n=len(g),
                                 rms_s=round(float(np.sqrt((g.res_s ** 2).mean())), 4), self_pair_lines_left_out=n_self))
            rows.append(dict(stage=st, version=lab, phase="dt (catalog)", cls="all", n=len(dtr),
                             rms_s=round(float(np.sqrt(((dtr.res_ms / 1000) ** 2).mean())), 4), self_pair_lines_left_out=n_self))
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["stage", "phase", "cls"], columns="version", values="rms_s").reset_index()
    nn = t.pivot_table(index=["stage", "phase", "cls"], columns="version", values="n").reset_index()
    piv["n_v0.4.1"], piv["n_v0.4.2"] = nn["v0.4.1"].values, nn["v0.4.2"].values
    piv["change_pct"] = (100 * (piv["v0.4.2"] / piv["v0.4.1"] - 1)).round(2)
    return piv


# ------------------------------------------------------------------ 8 product, times
def product():
    nw = VOUT / "product" / "nodes_written.json"
    if not nw.exists():
        return {}
    j = json.loads(nw.read_text())
    j["md5"] = {k: md5(v["path"]) for k, v in j["files"].items() if Path(v["path"]).is_file()}
    return j


def run_times():
    rows = []
    for st, (old, new, nf, onf) in STAGES.items():
        for lab, reg in (("v0.4.1", old), ("v0.4.2", new), ("v0.4.2 noise floor", nf)):
            f = rd(reg, st) / "time.txt"
            if f.exists() and (lab == "v0.4.1" or reg != old):
                tm = f.read_text().split()
                rows.append(dict(stage=st, version=lab, run=f"{reg}/d200_s{SM[st]}", wall_min=round(float(tm[-6]) / 60, 1),
                                 maxrss_gb=round(float(tm[-2]) / 1024 ** 2, 2)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    GIT.mkdir(parents=True, exist_ok=True)
    H = F41.html_table
    idt = identity()
    idt.to_csv(OUT / "identity.csv", index=False)
    inp = input_checks()
    inp.to_csv(OUT / "input_checks.csv", index=False)
    md, mdt, big, D, new, old, same_mask = model_diff()
    md.to_csv(OUT / "model_diff_by_depth.csv", index=False)
    mdt.to_csv(OUT / "model_diff_total.csv", index=False)
    big.to_csv(OUT / "nodes_above_noise.csv", index=False)
    near = (big.assign(within_30km_of_a_self_paired_event=big.km_to_nearest_self_paired_event <= 30)
            .groupby(["nodes", "quantity", "within_30km_of_a_self_paired_event"]).size().rename("nodes_above_noise")
            .reset_index()) if len(big) else pd.DataFrame()
    near.to_csv(OUT / "nodes_above_noise_by_distance.csv", index=False)
    sp, sep, pts = self_pairs()
    sp.to_csv(OUT / "self_paired_events.csv", index=False)
    sep.to_csv(OUT / "self_paired_separation.csv", index=False)
    CO = A.coast()
    fig_maps(D, new, CO, "maps_diff.png", "d", pts)
    fig_maps(D, new, CO, "maps_noise.png", "n", pts)
    fig_sections(D, new, "sections_diff.png", "d", pts)
    fig_sections(D, new, "sections_noise.png", "n", pts)
    fl = floor_table()
    fl.to_csv(OUT / "floor_by_stage.csv", index=False)
    hs = hypo_shifts()
    hs.to_csv(OUT / "hypocentre_shifts.csv", index=False)
    rs = residuals()
    rs.to_csv(OUT / "residual_rms.csv", index=False)
    rt = run_times()
    rt.to_csv(OUT / "run_times.csv", index=False)
    pr = product()
    summary = dict(identity=idt.to_dict("records"), inputs=inp.to_dict("records"), model_total=mdt.to_dict("records"),
                   masks_identical=same_mask, nodes_above_noise=near.to_dict("records"), self_pair_separation=sep.to_dict("records"),
                   floor=fl.to_dict("records"), product=pr)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    css = ("body{font-family:system-ui,sans-serif;max-width:1150px;margin:auto;padding:0 16px;color:#222;"
           "background:#fff}h1{font-size:1.4em}h2{font-size:1.15em;margin-top:1.6em;border-bottom:1px solid #ccc}"
           "table.t{border-collapse:collapse;font-size:12px;margin:6px 0;display:block;overflow-x:auto}"
           "table.t td,table.t th{border:1px solid #ddd;padding:2px 6px;text-align:right}img{max-width:100%}"
           "p,li{font-size:14px;line-height:1.45}.note{color:#555;font-size:13px}pre{font-size:11px;"
           "white-space:pre-wrap}@media (prefers-color-scheme: dark){body{background:#111;color:#ddd}"
           "table.t td,table.t th{border-color:#444}.note{color:#aaa}img{background:#fff}}")
    above_tot = int(fl.above_true_surface_final_v042.sum()) if len(fl) else None
    tv = mdt[mdt.nodes == "resolved"] if len(mdt) else mdt
    Hh = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
          f"initial-scale=1'><title>Model v0.4.2 report</title><style>{css}</style></head><body>",
          "<h1>Velocity model v0.4.2: v0.4.1 without self-paired differential times (D-B099)</h1>",
          f"<p class='note'>{V['description']}. Generated by workflow/10_tomography/b2_v042_analyse.py (branch "
          "tomo/ensemble-fix). Staged, not registered.</p>",
          "<h2>Summary</h2><ul>",
          f"<li>Freeze 2026-11-26 ({V['binaries']}): the v0.4.1 runs below reproduced byte for byte "
          f"({int((idt.different.str.len() == 0).sum()) if len(idt) else 0} of {len(idt)} runs with no differing output file).</li>",
          f"<li>Inputs: the three rebuilt tiles equal v0.4.1's minus exactly the self-pair blocks "
          f"({int(inp.dt_equals_old_minus_self_pairs.sum()) if 'dt_equals_old_minus_self_pairs' in inp else 0} of {len(inp)} "
          "datasets incl. the noise-floor copies); every other input byte-identical.</li>",
          f"<li>Model: resolved-node rms change " + ", ".join(
              f"{r.quantity} {r.rms:g} {'%' if r.unit == 'pct' else ''} (noise floors {r.noise_v042_rms:g} / "
              f"{r.noise_v041_rms:g}), {r.above_noise_nodes} nodes above 2x the local noise"
              for r in tv.itertuples()) + ".</li>",
          f"<li>Self-paired events: separation from their co-located twins v0.4.1 -> v0.4.2 (km): "
          + ", ".join(f"{int(r.self_paired)}/{int(r.twin)} {fmt(r.get('v041_separation_km'))} -> "
                      f"{fmt(r.get('v042_separation_km'))}" for _, r in sep.iterrows()) + ".</li>",
          "<li>Events kept: " + ", ".join(f"{r.stage} {r.events_final_v041} -> {r.events_final_v042} of {r.events_start}"
                                          for r in fl[fl.rerun].itertuples()) + " (v0.4.1 had dropped 4 of the self-paired "
          "events).</li>",
          f"<li>Final hypocentres above the true surface in v0.4.2: {above_tot}; air quakes "
          f"{int(fl.air_quakes_v042.sum()) if len(fl) else None}.</li></ul>",
          "<h2>1. Freeze identity</h2>",
          "<p>v0.4.1 deterministic runs rerun with the 2026-11-26 binary of the same size class, same inputs, "
          "tomoDD-SE.inp, MOD and surface.dat; Output_Files compared with cmp (partial runs: tomoDD.loc, the first "
          "iterations' tomoDD.reloc.001.00N and tomoDD-SE.res.ini). tomoDD-SE.log differs in text only (timestamps).</p>",
          H(idt), "<h2>2. Inputs of the rebuilt tiles</h2>", H(inp),
          "<h2>3. Model v0.4.2 - v0.4.1 (merged inversion values)</h2>",
          f"<p>Resolved = checkerboard-resolved nodes (D-B049 masks; identical in both: {same_mask}); tile-covered = "
          "inside a merged tile's box. Vp, Vs in percent, Vp/Vs absolute. Noise floors: each chain rerun with "
          "N(0, 1e-6 s) on every observed time and merged with the same rules (v0.4.2nf: M0 and oregon noise runs of "
          "v0.4.1, the three rebuilt tiles rerun). above_noise = |difference| &gt; 2 x the largest |noise difference| "
          "(either chain) at the node and its 26 neighbours.</p>", H(mdt), H(md),
          "<p>Nodes above the noise, by distance to the nearest self-paired event (start position):</p>", H(near),
          "<p>Resolved nodes above the noise, largest first:</p>", H(big[big.nodes == "resolved"], 40),
          "<p>Tile-covered nodes above the noise, largest first:</p>", H(big[big.nodes == "tile-covered"], 40),
          img("maps_diff.png"), img("sections_diff.png"), img("maps_noise.png"), img("sections_noise.png"),
          "<h2>4. The self-paired events and their co-located twins</h2>",
          "<p>Each self-paired event had a twin with the identical start hypocentre (v4-NLL3D); in v0.4.1 the event's "
          "zero-time 'absolute' data pulled it away. Final hypocentres from tomoDD.reloc_geo (true frame).</p>",
          H(sep), H(sp),
          "<h2>5. Floor stops, air quakes, events above the true surface</h2>", H(fl),
          "<h2>6. Hypocentre shifts v0.4.2 - v0.4.1 (rebuilt tiles)</h2>", H(hs),
          "<h2>7. Residual rms (final iteration, tomoDD.res)</h2>",
          "<p>tomoDD.res holds the last iteration's linearised residuals. v0.4.1's self-pair lines (absolute times "
          "of 0 s) are left out of the absolute classes.</p>", H(rs),
          "<h2>8. Product (staged) and run times</h2>",
          "<pre>" + json.dumps(pr, indent=1, default=str) + "</pre>", H(rt), "</body></html>"]
    (OUT / "report.html").write_text("\n".join(Hh))
    shutil.copy2(OUT / "report.html", STAGING / "report_v0.4.2.html")
    for f in OUT.iterdir():
        if f.is_file() and f.stat().st_size < 5 * 1024 ** 2:
            shutil.copy2(f, GIT / f.name)
    write_manifest("b2_v042_analyse", OUT, outputs=sorted(p for p in OUT.iterdir() if p.is_file()),
                   params={"v042": V}, counts={"stages": len(fl)})
    print("report:", STAGING / "report_v0.4.2.html")


if __name__ == "__main__":
    main()
