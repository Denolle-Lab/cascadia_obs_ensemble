"""v0.4.1 report (D-B083): what the D-B081 floor in the inversion changed against v0.4.0.

Sections (tables <out>/report/*.csv, figures *.png, report.html self-contained; small files copied to
data/tomo_v1/v041_floor/):
  1 binary identity (step 1): v0.4.0 runs rerun with the 2026-11-25 inversion binaries without surface.dat
  2 surface.dat writer (step 2): per stage, start hypocentres against the true surface and tomoDD's floor
  3 test (step 3): v0.4.0 blanco tile with the floor (hits, air quakes, what changed) and with an unreachable floor
    (byte identity)
  4 production: floor hits per stage, air quakes before (v0.4.0) and after (v0.4.1), events kept
  5 events that ended at the floor, by stage and region; events above the true surface at the end (errors)
  6 v0.4.1 - v0.4.0 merged models on resolved nodes against the noise floors, by depth (+ maps)
  7 hypocentre shifts by stage and region (+ noise floors)
  8 residual rms by phase and station class (final iteration; start)
  9 run times
  python b2_v041_analyse.py
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
import b2_v040_analyse as A  # noqa: E402  (readers: read_res, abs_res, reloc, parse_log, coast)
import b2_tomodd_floor as FLR  # noqa: E402

P = load_params()
T, V, RT = P["tomodd"], P["v041"], P["retime"]
RUNS = Path(T["runs"])
VOUT = Path(V["out"])
OUT = VOUT / "report"
GIT = REPO / "data" / "tomo_v1" / "v041_floor"
STORE = Path(T["models"]["store"]) / T["models"]["name"]
TILES = {"mendocino_gorda_c": 480, "blanco": 20, "oregon": 480, "washington": 480}
# stage: (v0.4.0 region, v0.4.1 region, v0.4.1 noise floor, smoothing, v0.4.0 noise floor)
STAGES = {"M0": ("margin_m0_v3", "margin_m0_v4", "margin_m0_v4nf", 480, "margin_m0_v3nf")}
STAGES.update({t: (f"{t}_m0t", f"{t}_m0f", f"{t}_m0fnf", s, f"{t}_m0tnf") for t, s in TILES.items()})
AT_FLOOR_KM = 0.005          # "at the floor": final depth within 5 m below tomoDD's floor (it stops 1 m under it)
TXT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def rd(region, sm):
    return RUNS / region / f"d200_s{sm}"


def log_text(run):
    return (Path(run) / "tomoDD-SE.log").read_text(errors="replace")


def floor_log(run):
    """>>>Floor lines by iteration ('Iteration N finished' closes an iteration), air-quake lines, not-met lines."""
    it, hits, aq, notmet, niter = 1, [], [], [], 0
    for ln in log_text(run).splitlines():
        if ln.startswith(">>>Floor: "):
            t = ln.split()
            hits.append(dict(iteration=it, ID=int(t[1]), depth=float(t[2]), floor=float(t[3])))
        elif ln.startswith(">>>Floor not met"):
            notmet.append(int(ln.split()[-1]))
        elif ln.startswith(">>>Warning: negative depth"):
            aq.append(dict(iteration=it, ID=int(ln.split()[-1])))
        else:
            m = re.match(r"Iteration\s+(\d+) finished", ln)
            if m:
                niter = int(m.group(1))
                it = niter + 1
    return pd.DataFrame(hits, columns=["iteration", "ID", "depth", "floor"]), \
        pd.DataFrame(aq, columns=["iteration", "ID"]), notmet, niter


def reloc_w(run):
    """tomoDD.reloc in tomoDD's own (rescaled) frame: ID, lat, lon, depth."""
    d = pd.read_csv(Path(run) / "Output_Files" / "tomoDD.reloc", sep=r"\s+", header=None)
    return d.rename(columns={0: "ID", 1: "lat_w", 2: "lon_w", 3: "depth_w"})[["ID", "lat_w", "lon_w", "depth_w"]]


def surface_depth(lat, lon):
    from b4_surface import surface_m
    return -surface_m(np.asarray(lon), np.asarray(lat)) / 1000.0


def final_positions(region, sm, floored):
    """Final hypocentres (true frame from reloc_geo, tomoDD frame from reloc), the exact surface depth, and for
    floored runs tomoDD's floor at the final position."""
    run = rd(region, sm)
    g = A.reloc(run)
    w = reloc_w(run)
    d = g.merge(w, on="ID", validate="1:1")
    d["surface_km"] = surface_depth(d.lat.values, d.lon.values)
    d["below_surface_km"] = d.depth - d.surface_km
    if floored:
        S = FLR.read_surface_dat(run / "surface.dat")
        d["floor_km"] = FLR.floordep(S, d.lat_w.values, d.lon_w.values)[0]
        d["above_floor_km"] = d.floor_km - d.depth_w
        d["at_floor"] = (d.depth_w - d.floor_km >= -1e-6) & (d.depth_w - d.floor_km < AT_FLOOR_KM)
    ev = pd.read_csv(RUNS / region / "inputs" / "events.csv")[["ID", "region"]]
    return d.merge(ev, on="ID", how="left")


def html_table(df, n=None):
    d = df if n is None else df.head(n)
    return d.to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:.4g}", na_rep="")


def img(name):
    b = base64.b64encode((OUT / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{name}">'


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
                         output_files=int(c[-1].split()[1]) if c and c[-1].startswith("files") else np.nan,
                         different=[x.split()[1] for x in c if x.startswith("DIFFERENT")],
                         log_identical=any(x == "IDENTICAL tomoDD-SE.log" for x in c),
                         wall_min=round(float(tm[0]) / 60, 1) if tm else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2 writer
def writer_checks():
    rows, bad = [], []
    for name, (old, new, nf, sm, oldnf) in STAGES.items():
        for reg in (new, nf):
            inp = RUNS / reg / "inputs"
            if not (inp / "floor_check.json").exists():
                continue
            f, c = json.loads((inp / "floor.json").read_text()), json.loads((inp / "floor_check.json").read_text())
            rows.append(dict(stage=name, region=reg, spacing_s=f["spacing_s"], nodes=f"{f['nodes'][0]} x {f['nodes'][1]}",
                             events=c["events"], below_true_surface=c["below_true_surface"],
                             above_true_surface_at_start=c["above_true_surface"],
                             below_surface_but_above_floor=c["below_surface_but_above_floor"],
                             floor_below_surface_m_p50_p90_p99_max=c["floor_minus_surface_m_p50_p90_p99_max"],
                             floor_below_surface_min_m=c["floor_minus_surface_min_m"],
                             node_mismatch=c["floor_node_tomodd_vs_true_mismatch"],
                             float32_edge_ties=c["floor_node_edge_ties"]))
            if reg == new:
                e = pd.read_csv(inp / "floor_check_events.csv")
                bad.append(e.assign(stage=name))
    return pd.DataFrame(rows), (pd.concat(bad, ignore_index=True) if bad else pd.DataFrame())


# ------------------------------------------------------------------ 3 test
def test_section():
    t = VOUT / "test"
    out = {}
    v0 = RUNS / "blanco_m0t" / "d200_s20"
    fr = t / "blanco_floor" / "run"
    if not (t / "blanco_floor" / "DONE").exists():
        return None
    h, aq, nm, nit = floor_log(fr)
    h0, aq0, _, nit0 = floor_log(v0)
    out["floor_lines"] = len(h)
    out["events_hit"] = int(h.ID.nunique())
    out["hits_by_iteration"] = h.groupby("iteration").size().to_dict()
    out["floor_not_met"] = len(nm)
    out["height_above_floor_m_p50_p90_max"] = ([round(float(np.percentile(1000 * (h.floor - h.depth), p)), 1)
                                                for p in (50, 90, 100)] if len(h) else None)
    out["air_quakes_v040"] = len(aq0)
    out["air_quakes_test"] = len(aq)
    out["iterations_v040_incl_aq_repeats"] = nit0
    out["iterations_test"] = nit
    ev0 = sum(1 for _ in open(v0 / "Input_Files" / "event.dat"))
    out["events_start"] = ev0
    out["events_final_v040"] = sum(1 for _ in open(v0 / "Output_Files" / "tomoDD.reloc"))
    out["events_final_test"] = sum(1 for _ in open(fr / "Output_Files" / "tomoDD.reloc"))
    a, b = reloc_w(v0), reloc_w(fr)
    m = a.merge(b, on="ID", suffixes=("_0", "_1"))
    m["dz"] = m.depth_w_1 - m.depth_w_0
    m["dh"] = np.hypot((m.lat_w_1 - m.lat_w_0) * 111.19, (m.lon_w_1 - m.lon_w_0) * 111.19 * np.cos(np.radians(m.lat_w_0)))
    hit = m.ID.isin(h.ID)
    for lab, g in (("hit", m[hit]), ("not_hit", m[~hit])):
        out[f"shift_{lab}"] = dict(n=len(g), dz_median_km=round(float(g.dz.median()), 4) if len(g) else None,
                                   dz_absp90_km=round(float(g.dz.abs().quantile(0.9)), 4) if len(g) else None,
                                   dh_median_km=round(float(g.dh.median()), 4) if len(g) else None,
                                   moved_gt_1m=int(((g.dz.abs() > 0.001) | (g.dh > 0.001)).sum()))
    S = FLR.read_surface_dat(fr / "surface.dat")
    fz = FLR.floordep(S, b.lat_w.values, b.lon_w.values)[0]
    out["final_at_floor"] = int(((b.depth_w - fz >= -1e-6) & (b.depth_w - fz < AT_FLOOR_KM)).sum())
    out["final_above_floor"] = int((b.depth_w - fz < -1e-6).sum())
    from b2_tomodd_backtransform import inverse            # the test run was not back-transformed: same map here
    fw = json.loads((RUNS / "blanco_m0t" / "inputs" / "forward.json").read_text())
    for lab, g in (("v040", a), ("test", b)):
        la, lo = inverse(g.lat_w.values, g.lon_w.values, fw)
        out[f"final_above_true_surface_{lab}"] = int((g.depth_w.values < surface_depth(la, lo)).sum())
    vp0 = np.loadtxt(v0 / "Output_Files" / "Vp_model.dat")
    vp1 = np.loadtxt(fr / "Output_Files" / "Vp_model.dat")
    vs0 = np.loadtxt(v0 / "Output_Files" / "Vs_model.dat")
    vs1 = np.loadtxt(fr / "Output_Files" / "Vs_model.dat")
    for k, x0, x1 in (("Vp", vp0, vp1), ("Vs", vs0, vs1)):
        d = 100 * (x1 / x0 - 1)
        out[f"d{k}_pct_rms_max_changed_nodes"] = [round(float(np.sqrt((d ** 2).mean())), 4),
                                                  round(float(np.abs(d).max()), 3), int((d != 0).sum()), int(d.size)]
    l0, l1 = A.parse_log(v0), A.parse_log(fr)
    out["rms_abs_final_v040_test"] = [l0["rms_abs_final"], l1["rms_abs_final"]]
    out["rms_wt_final_v040_test"] = [l0["rms_wt_final"], l1["rms_wt_final"]]
    out["wall_min_v040_test"] = [l0["wall_min"], l1["wall_min"]]
    c = (t / "blanco_nohit" / "compare.txt").read_text().splitlines() if (t / "blanco_nohit" / "compare.txt").exists() else []
    out["nohit_files_compared"] = len(c)
    out["nohit_different"] = [x.split()[1] for x in c if x.startswith("DIFFERENT")]
    nh, _, _, _ = floor_log(t / "blanco_nohit" / "run") if (t / "blanco_nohit" / "run" / "tomoDD-SE.log").exists() \
        else (pd.DataFrame(), None, None, None)
    out["nohit_floor_lines"] = len(nh)
    return out


# ------------------------------------------------------------------ 4/5 production
def production():
    rows, reg_rows, above, finals = [], [], [], {}
    for name, (old, new, nf, sm, oldnf) in STAGES.items():
        if not (rd(new, sm) / "Output_Files" / "tomoDD.reloc_geo").exists():
            continue
        h, aq, nm, nit = floor_log(rd(new, sm))
        h0, aq0, _, nit0 = floor_log(rd(old, sm))
        f1 = final_positions(new, sm, True)
        f0 = final_positions(old, sm, False)
        finals[name] = (f0, f1)
        ev = pd.read_csv(RUNS / new / "inputs" / "events.csv")[["ID", "region"]]
        l0, l1 = A.parse_log(rd(old, sm)), A.parse_log(rd(new, sm))
        rows.append(dict(stage=name, run=f"{new}/d200_s{sm}", events_start=len(ev),
                         floor_lines=len(h), events_hit=int(h.ID.nunique()), floor_not_met=len(nm),
                         hit_height_above_floor_m_p50_max=([round(float(np.percentile(1000 * (h.floor - h.depth), p)), 1)
                                                            for p in (50, 100)] if len(h) else None),
                         events_at_floor_final=int(f1.at_floor.sum()),
                         events_above_floor_final=int((f1.above_floor_km > 1e-6).sum()),
                         air_quakes_v040=len(aq0), air_quakes_v041=len(aq),
                         iterations_v040=nit0, iterations_v041=nit,
                         events_final_v040=len(f0), events_final_v041=len(f1),
                         above_true_surface_final_v040=int((f0.below_surface_km < 0).sum()),
                         above_true_surface_final_v041=int((f1.below_surface_km < 0).sum()),
                         rms_abs_final_v040=l0["rms_abs_final"], rms_abs_final_v041=l1["rms_abs_final"],
                         rms_wt_final_v040=l0["rms_wt_final"], rms_wt_final_v041=l1["rms_wt_final"]))
        hit = set(h.ID)
        for r, g in ev.groupby("region"):
            g1 = f1[f1.region == r]
            g0 = f0[f0.region == r]
            reg_rows.append(dict(stage=name, region=r, events=len(g), hit_during_inversion=int(g.ID.isin(hit).sum()),
                                 at_floor_final=int(g1.at_floor.sum()),
                                 at_floor_pct=round(100 * float(g1.at_floor.mean()), 2) if len(g1) else np.nan,
                                 air_quakes_v040=int(aq0.ID.isin(g.ID).sum()), air_quakes_v041=int(aq.ID.isin(g.ID).sum()),
                                 above_surface_final_v040=int((g0.below_surface_km < 0).sum()),
                                 within_100m_of_surface_v041=int((g1.below_surface_km < 0.1).sum())))
        a1 = f1[f1.below_surface_km < 0]
        if len(a1):
            above.append(a1.assign(stage=name, version="v0.4.1"))
    return pd.DataFrame(rows), pd.DataFrame(reg_rows), (pd.concat(above) if above else pd.DataFrame()), finals


# ------------------------------------------------------------------ 6 models
def model_diff():
    import xarray as xr
    old = xr.open_dataset(STORE / "v0.4.0.nc")
    new = xr.open_dataset(VOUT / "product" / "merged_v041.nc")
    nf = xr.open_dataset(VOUT / "noise" / "merged_v041nf.nc")
    onf = xr.open_dataset(Path(RT["out"]) / "noise" / "merged_v040nf.nc")
    D = {}
    for k in ("Vp", "Vs"):
        D[f"d{k}"] = 100 * (new[k].values / old[k].values - 1)
        D[f"n{k}"] = 100 * (nf[k].values / new[k].values - 1)
        D[f"o{k}"] = 100 * (onf[k].values / old[k].values - 1)
    from scipy.ndimage import maximum_filter
    z = new.depth.values
    res = {"Vp": new.resolution_p.values > 0, "Vs": new.resolution_s.values > 0}
    same_mask = bool((old.resolution_p.values == new.resolution_p.values).all() and
                     (old.resolution_s.values == new.resolution_s.values).all())
    rows, tot = [], []
    for k in ("Vp", "Vs"):
        nl = maximum_filter(np.maximum(np.abs(D[f"n{k}"]), np.abs(D[f"o{k}"])), size=3, mode="nearest")
        for i, zz in enumerate(list(z) + ["all"]):
            m = res[k][i] if zz != "all" else res[k]
            if m.sum() == 0:
                continue
            d = D[f"d{k}"][i][m] if zz != "all" else D[f"d{k}"][m]
            n = D[f"n{k}"][i][m] if zz != "all" else D[f"n{k}"][m]
            o = D[f"o{k}"][i][m] if zz != "all" else D[f"o{k}"][m]
            L = nl[i][m] if zz != "all" else nl[m]
            r = dict(quantity=k, depth_km=zz, resolved_nodes=int(m.sum()),
                     rms_pct=round(float(np.sqrt((d ** 2).mean())), 4), max_abs_pct=round(float(np.abs(d).max()), 3),
                     gt1_pct=round(100 * float((np.abs(d) > 1).mean()), 2),
                     noise_v041_rms_pct=round(float(np.sqrt((n ** 2).mean())), 4),
                     noise_v040_rms_pct=round(float(np.sqrt((o ** 2).mean())), 4),
                     noise_max_abs_pct=round(float(max(np.abs(n).max(), np.abs(o).max())), 3),
                     above_noise_pct=round(100 * float((np.abs(d) > 2 * L).mean()), 2))
            (tot if zz == "all" else rows).append(r)
    return pd.DataFrame(rows), pd.DataFrame(tot), D, new, old, same_mask


def fig_maps(D, new, CO, fname, which="d"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon, lat, z = new.longitude.values, new.latitude.values, new.depth.values
    depths = [-1, 1, 3, 5, 9, 15]
    fig, axs = plt.subplots(2, len(depths), figsize=(2.6 * len(depths), 6.8), sharex=True, sharey=True)
    for i, (k, lab) in enumerate((("Vp", "dVp/Vp (%)"), ("Vs", "dVs/Vs (%)"))):
        for j, dd in enumerate(depths):
            ax = axs[i, j]
            iz = int(np.argmin(np.abs(z - dd)))
            im = ax.pcolormesh(lon, lat, D[f"{which}{k}"][iz], cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest",
                               rasterized=True)
            rr = (new.resolution_p.values if k == "Vp" else new.resolution_s.values)[iz] > 0
            if rr.any():
                ax.contour(lon, lat, rr.astype(float), levels=[0.5], colors="k", linewidths=0.6)
            if CO is not None:
                ax.contour(CO[0], CO[1], CO[2], levels=[0], colors="0.5", linewidths=0.4)
            ax.set_aspect(1 / np.cos(np.radians(45)))
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(f"{z[iz]:g} km", fontsize=8)
        fig.colorbar(im, ax=axs[i, :], shrink=0.8, label=lab, pad=0.01)
    t = "v0.4.1 - v0.4.0" if which == "d" else "noise floor of v0.4.1 (1e-6 s perturbed chain - v0.4.1)"
    fig.suptitle(f"{t} (inversion values); black contour = checkerboard-resolved nodes", fontsize=9)
    fig.savefig(OUT / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_floor_map(finals, CO):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(finals), figsize=(3.2 * len(finals), 4.2))
    for ax, (name, (f0, f1)) in zip(np.atleast_1d(axs), finals.items()):
        ax.scatter(f1.lon, f1.lat, s=0.3, c="0.75", rasterized=True)
        a = f1[f1.at_floor]
        ax.scatter(a.lon, a.lat, s=3, c="tab:red", label=f"at floor ({len(a)})")
        if CO is not None:
            ax.contour(CO[0], CO[1], CO[2], levels=[0], colors="0.4", linewidths=0.4)
        ax.set_xlim(f1.lon.min() - 0.2, f1.lon.max() + 0.2)
        ax.set_ylim(f1.lat.min() - 0.2, f1.lat.max() + 0.2)
        ax.set_aspect(1 / np.cos(np.radians(45)))
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("v0.4.1 final hypocentres (grey) and those ending on tomoDD's floor (red)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "floor_events_map.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ 7 hypocentres
def hypo_shifts():
    rows = []
    for name, (old, new, nf, sm, oldnf) in STAGES.items():
        if not (rd(new, sm) / "Output_Files" / "tomoDD.reloc_geo").exists():
            continue
        ev = pd.read_csv(RUNS / new / "inputs" / "events.csv")[["ID", "region"]]
        pairs = [("v0.4.1 - v0.4.0", rd(new, sm), rd(old, sm))]
        if (rd(nf, sm) / "Output_Files" / "tomoDD.reloc_geo").exists():
            pairs.append(("noise floor v0.4.1 (nf - v0.4.1)", rd(nf, sm), rd(new, sm)))
        if (rd(oldnf, sm) / "Output_Files" / "tomoDD.reloc_geo").exists():
            pairs.append(("noise floor v0.4.0 (nf - v0.4.0)", rd(oldnf, sm), rd(old, sm)))
        for lab, x_, y_ in pairs:
            x, y = A.reloc(x_), A.reloc(y_)
            m = x.merge(y, on="ID", suffixes=("", "_0")).merge(ev, on="ID", how="left")
            m["dh_km"] = np.hypot((m.lat - m.lat_0) * 111.19, (m.lon - m.lon_0) * 111.19 * np.cos(np.radians(m.lat)))
            m["dz_km"] = m.depth - m.depth_0
            for reg, g in list(m.groupby("region")) + [("all", m)]:
                rows.append(dict(stage=name, comparison=lab, region=reg, events=len(g),
                                 dh_median_km=round(float(g.dh_km.median()), 3),
                                 dh_p90_km=round(float(g.dh_km.quantile(0.9)), 3),
                                 dz_median_km=round(float(g.dz_km.median()), 3),
                                 dz_absmedian_km=round(float(g.dz_km.abs().median()), 3),
                                 dz_p90abs_km=round(float(g.dz_km.abs().quantile(0.9)), 3),
                                 gt1km_3d_pct=round(100 * float((np.hypot(g.dh_km, g.dz_km) > 1).mean()), 1)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 8 residuals
def residuals():
    rows = []
    for name, (old, new, nf, sm, oldnf) in STAGES.items():
        if not (rd(new, sm) / "Output_Files" / "tomoDD.res").exists():
            continue
        codes = pd.read_csv(RUNS / new / "inputs" / "station_codes.csv")[["code", "is_obs"]]
        for lab, reg, ini in (("start", new, True), ("v0.4.0", old, False), ("v0.4.1", new, False)):
            a, dtr = A.abs_res(rd(reg, sm), ini=ini)
            a = a.merge(codes, on="code", how="left")
            a["cls"] = np.where(a.is_obs, "OBS", "land")
            for (ph, cls), g in list(a.groupby(["phase", "cls"])) + [((ph, "all"), g) for ph, g in a.groupby("phase")]:
                rows.append(dict(stage=name, version=lab, phase=ph, cls=cls, n=len(g),
                                 rms_s=round(float(np.sqrt((g.res_s ** 2).mean())), 4),
                                 mean_s=round(float(g.res_s.mean()), 4)))
            rows.append(dict(stage=name, version=lab, phase="dt (catalog)", cls="all", n=len(dtr),
                             rms_s=round(float(np.sqrt(((dtr.res_ms / 1000) ** 2).mean())), 4),
                             mean_s=round(float((dtr.res_ms / 1000).mean()), 4)))
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["stage", "phase", "cls"], columns="version", values="rms_s").reset_index()
    nn = t[t.version == "v0.4.1"].set_index(["stage", "phase", "cls"]).n
    piv["n_v0.4.1"] = [nn.get((r.stage, r.phase, r.cls), np.nan) for r in piv.itertuples()]
    if {"v0.4.0", "v0.4.1"} <= set(piv.columns):
        piv["change_pct"] = (100 * (piv["v0.4.1"] / piv["v0.4.0"] - 1)).round(2)
    return t, piv


# ------------------------------------------------------------------ 9 run times
def run_times():
    rows = []
    for name, (old, new, nf, sm, oldnf) in STAGES.items():
        for lab, reg in (("v0.4.0", old), ("v0.4.1", new), ("v0.4.1 noise floor", nf)):
            f = rd(reg, sm) / "time.txt"
            if f.exists():
                tm = f.read_text().split()
                rows.append(dict(stage=name, version=lab, run=f"{reg}/d200_s{sm}", wall_min=round(float(tm[-6]) / 60, 1),
                                 user_min=round(float(tm[-4]) / 60, 1), maxrss_gb=round(float(tm[-2]) / 1024 ** 2, 2)))
    st = VOUT / "chain.state"
    js = pd.read_csv(st, sep="\t", header=None, names=["job", "start", "end", "seconds"]) if st.exists() else None
    return pd.DataFrame(rows), js


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GIT.mkdir(parents=True, exist_ok=True)
    idt = identity()
    idt.to_csv(OUT / "identity.csv", index=False)
    wc, wbad = writer_checks()
    wc.to_csv(OUT / "writer_check.csv", index=False)
    wbad.to_csv(OUT / "writer_check_events.csv", index=False)
    tst = test_section()
    (OUT / "test.json").write_text(json.dumps(tst, indent=1, default=str))
    prod, preg, pabove, finals = production()
    prod.to_csv(OUT / "floor_by_stage.csv", index=False)
    preg.to_csv(OUT / "floor_by_region.csv", index=False)
    pabove.to_csv(OUT / "above_surface_final_v041.csv", index=False)
    md, mdt, D, new, old, same_mask = model_diff()
    md.to_csv(OUT / "model_diff_by_depth.csv", index=False)
    mdt.to_csv(OUT / "model_diff_total.csv", index=False)
    CO = A.coast()
    fig_maps(D, new, CO, "maps_diff.png", "d")
    fig_maps(D, new, CO, "maps_noise.png", "n")
    fig_floor_map(finals, CO)
    hs = hypo_shifts()
    hs.to_csv(OUT / "hypocentre_shifts.csv", index=False)
    rt, rpiv = residuals()
    rt.to_csv(OUT / "residual_rms_long.csv", index=False)
    rpiv.to_csv(OUT / "residual_rms.csv", index=False)
    rtm, js = run_times()
    rtm.to_csv(OUT / "run_times.csv", index=False)
    reg = pd.read_csv(STORE / "registry.csv", dtype={"version": str})
    regrow = reg[reg.version == str(V["version"])]
    nodes = json.loads((VOUT / "product" / "nodes_written.json").read_text()) \
        if (VOUT / "product" / "nodes_written.json").exists() else {}
    nll = json.loads((VOUT / "product" / "nll_written.json").read_text()) \
        if (VOUT / "product" / "nll_written.json").exists() else {}
    summary = dict(identity=idt.to_dict("records"), test=tst, floor_by_stage=prod.to_dict("records"),
                   model_total=mdt.to_dict("records"), masks_identical=same_mask,
                   registration=regrow.to_dict("records"), product_info=nodes.get("info"),
                   product_checks=nodes.get("checks"), nll_md5=nll.get("md5"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    css = ("body{font-family:system-ui,sans-serif;max-width:1150px;margin:auto;padding:0 16px;color:#222;"
           "background:#fff}h1{font-size:1.4em}h2{font-size:1.15em;margin-top:1.6em;border-bottom:1px solid #ccc}"
           "table.t{border-collapse:collapse;font-size:12px;margin:6px 0;display:block;overflow-x:auto}"
           "table.t td,table.t th{border:1px solid #ddd;padding:2px 6px;text-align:right}img{max-width:100%}"
           "p,li{font-size:14px;line-height:1.45}.note{color:#555;font-size:13px}pre{font-size:11px;"
           "white-space:pre-wrap}@media (prefers-color-scheme: dark){body{background:#111;color:#ddd}"
           "table.t td,table.t th{border-color:#444}.note{color:#aaa}img{background:#fff}}")
    T_ = tst or {}
    H = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
         f"initial-scale=1'><title>Model v0.4.1 report</title><style>{css}</style></head><body>",
         "<h1>Velocity model v0.4.1: v0.4.0 re-inverted with the tomoDD floor (D-B083)</h1>",
         f"<p class='note'>{V['description']}. Generated by workflow/10_tomography/b2_v041_analyse.py (branch "
         "tomo/v041-floor). Floor = D-B081 (tomoDD-SE 2026-11-25 freeze): an event update that would put the event "
         "above the floor depth of the nearest surface.dat node stops 1 m below it; the event is kept and logged. "
         "surface.dat = conservative floor of the D-B068 merged surface (exact cell minimum - 1 m; M0 45 s, tiles "
         "30 s, the 1001-node limit), in each stage's rescaled frame (D-B028 P-2). Air_dep -1000 km. Velocities "
         "above the surface stay rock-continued in the inversion (D-B083). Everything else as v0.4.0: same inputs, "
         "damping 200, smoothing (M0 480, blanco 20, others 480), merge rules D-B049 (v0.3.1 d200 masks), D-B050, "
         "D-B051.</p>",
         "<h2>1. Binary identity without surface.dat (step 1)</h2>",
         "<p>Each v0.4.0 run below was rerun with the 2026-11-25 inversion binary of its size class, same inputs, "
         "tomoDD-SE.inp and MOD, no surface.dat; every Output_Files file compared byte for byte (cmp).</p>",
         html_table(idt),
         "<h2>2. surface.dat writer (step 2)</h2>",
         "<p>Start hypocentres (event.dat, the v0.4.0 starts) against the exact merged surface at their true position "
         "and against the floor tomoDD reads at their rescaled position (float32 nearest-node rule reproduced). "
         "'below surface but above floor': below the true surface, but in the cell-minimum gap, so their first "
         "upward update stops at the floor. Node choice: tomoDD's node vs NonLinLoc's rule at the true position.</p>",
         html_table(wc),
         "<p>Listed start hypocentres (above the true surface at start, or below it and above the floor):</p>",
         html_table(wbad, 60),
         "<h2>3. Test on the v0.4.0 blanco tile (step 3)</h2>",
         f"<p>Floored run (30 s floor, Air_dep -1000 km) against v0.4.0 blanco_m0t d200_s20: {T_.get('floor_lines')} "
         f"floor stops on {T_.get('events_hit')} events (by iteration {T_.get('hits_by_iteration')}), not met "
         f"{T_.get('floor_not_met')}; air quakes {T_.get('air_quakes_v040')} (v0.4.0) -> {T_.get('air_quakes_test')}; "
         f"events kept {T_.get('events_final_v040')} -> {T_.get('events_final_test')} of {T_.get('events_start')}; "
         f"final positions at the floor {T_.get('final_at_floor')}, above the floor {T_.get('final_above_floor')}, "
         f"above the true surface {T_.get('final_above_true_surface_v040')} (v0.4.0) -> "
         f"{T_.get('final_above_true_surface_test')}. Unreachable floor (all nodes 50 km above sea level, Air_dep "
         f"as v0.4.0): {T_.get('nohit_files_compared')} output files compared, different: "
         f"{T_.get('nohit_different')}.</p>",
         "<pre>" + json.dumps(tst, indent=1, default=str) + "</pre>",
         "<h2>4. Production: floor stops and air quakes per stage</h2>",
         "<p>floor_lines = '>>>Floor' log lines (one per event and iteration where the update was stopped); "
         f"at floor = final depth within {1000 * AT_FLOOR_KM:g} m below tomoDD's floor; air quakes = events tomoDD "
         "dropped ('negative depth', each forcing a repeated iteration). above_true_surface = final hypocentres above "
         "the exact merged surface (errors; must be 0 for v0.4.1).</p>", html_table(prod),
         "<h2>5. Events at the floor, by region</h2>", html_table(preg), img("floor_events_map.png"),
         ("<p><b>Errors: v0.4.1 final hypocentres above the true surface:</b></p>" + html_table(pabove, 50))
         if len(pabove) else "<p>No v0.4.1 final hypocentre is above the true surface.</p>",
         "<h2>6. Model v0.4.1 - v0.4.0 on resolved nodes (inversion values, before the product rules)</h2>",
         f"<p>Resolution masks identical: {same_mask}. Noise floors: the v0.4.1 chain and the v0.4.0 chain rerun "
         "with N(0, 1e-6 s) on every observed time, merged with the same rules. above_noise = |difference| &gt; 2 x "
         "the largest |noise difference| (either chain) at the node and its 26 neighbours.</p>",
         html_table(mdt), html_table(md), img("maps_diff.png"), img("maps_noise.png"),
         "<h2>7. Hypocentre shifts (tomoDD final), by stage and region</h2>", html_table(hs),
         "<h2>8. Residual rms by phase and station class</h2>",
         "<p>Absolute residuals of the final iteration (tomoDD.res; 'start' = v0.4.1's first iteration, same data "
         "and start as v0.4.0's); catalog dt residuals in the last row of each stage.</p>", html_table(rpiv),
         "<h2>9. Run times</h2>", html_table(rtm), html_table(js) if js is not None else "",
         "<h2>10. Product and registration</h2>",
         "<pre>" + json.dumps(dict(registration=regrow.to_dict("records"), info=nodes.get("info"),
                                   checks=nodes.get("checks"), nll=nll.get("md5")), indent=1, default=str) + "</pre>",
         "</body></html>"]
    (OUT / "report.html").write_text("\n".join(H))
    for f in OUT.iterdir():
        if f.is_file() and f.stat().st_size < 5 * 1024 ** 2 and not f.name.endswith(".DONE"):
            shutil.copy2(f, GIT / f.name)
    write_manifest("b2_v041_analyse", OUT, outputs=sorted(p for p in OUT.iterdir() if p.is_file()),
                   params={"v041": V, "tomodd.floor": T["floor"]}, counts={"stages": len(prod)})
    (OUT / "ANALYSE.DONE").write_text("done\n")
    print("report:", OUT / "report.html")


if __name__ == "__main__":
    main()
