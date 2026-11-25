"""Track B, B2: checkerboard resolution tests on the real geometry (D-B018), Mendocino-Gorda v2.

True model: +-A Vp on interior nodes (Vp/Vs unchanged, so Vs also +-A), lateral cells of L km in geographic
km from the region centre (per-node cos(lat)), sign alternating cell to cell and between the depth bands
`bands_km` (node depth in [b_i, b_i+1)); nodes outside the bands and the boundary nodes keep the background.
Phase 1 shifts the pattern by `phase_shift` cells in x and y. Synthetics with tomoDD-SE_syn on the real events
(true = v4-NLL3D start locations), stations, pick list, catalogue pairs and weights (b2_smooth_spike_test helpers).
Variants of the inversion data:
  clean       noise-free, inversion from the true locations
  noise       Gaussian pick noise, sigma = D-B008 per pick (picks_used.csv sigma_s, S = 2x P); one draw
              (noise_seed) shared by all patterns; the same draw enters the absolute pick and every
              differential time built from it
  noise_ploc  as noise, inversion from perturbed locations: per event N(0, h_max / loc_h_from_68) per
              horizontal component (h_max = 68% semi-major axis) and N(0, sig_z) in depth (v4-NLL3D iter1
              uncertainties, events.csv); depth not moved above min(true depth, 0)
Inversions: b2_tomodd_run.py <region>/res_tests/chk_<PAT>_<VARIANT> inv_d<D>_s<W1> --damp D --smooth W1 W2 W3,
from the background MOD (the starting model), production iteration sets.
Metrics (per node of each layer, lateral square window of side window_cells x L): local Pearson correlation
of recovered and true dlnV, and amplitude = sum(rec true) / sum(true^2) (regression slope through 0).
Resolved = corr >= 0.7 and amp >= 0.5 (D-B018).

  python b2_res_checker.py prep                       # true models + forward dirs for every pattern
  python b2_res_checker.py syn PAT                    # forward run of one pattern (e.g. L060_p0)
  python b2_res_checker.py inputs PAT VARIANT         # inversion inputs for one variant
  python b2_res_checker.py analyse [--mask-smooth W1] # metrics, masks, table, figures -> data/tomo_v1/b2_res_tests
Run from the repo root with LD_LIBRARY_PATH=.pixi/envs/default/lib.
"""
import json
import shutil
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
import b2_smooth_spike_test as sp  # noqa: E402
from b2_tomodd_backtransform import inverse  # noqa: E402

T = load_params()["tomodd"]
R = sp.res_cfg()                 # tile from env TOMO_RES_REGION (default res_tests.region)
C = R["checker"]
sp.use("res_tests")
BASE = Path(T["runs"]) / R["region"]
INP = BASE / "inputs"
TEST = BASE / R["subdir"]
OUT = REPO / "data" / "tomo_v1" / "b2_res_tests" / R["region"]
if os.environ.get("TOMO_RES_DAMP"):          # damping override: results in <tile>/d<damp>/
    OUT = OUT / f"d{R['damp']:g}"
KM = 111.195
VARIANTS = ("clean", "noise", "noise_ploc")


def patterns():
    return [f"L{L:03d}_p{ph}" for L in C["cell_km"] for ph in (0, 1)]


def parse_pat(pat):
    L, ph = pat.split("_")
    return int(L[1:]), int(ph[1:])


def geo_grid(m):
    """Geographic lon/lat of the MOD axes (separable map, b2_tomodd_backtransform)."""
    fw = json.loads((INP / "forward.json").read_text())
    c = fw["center"]
    lat = inverse(m["y"], np.full_like(m["y"], c[1]), fw)[0]
    lon = inverse(np.full_like(m["x"], c[0]), m["x"], fw)[1]
    return lon, lat, c


def true_pattern(m, L, phase):
    """dlnV [nz, ny, nx] of the checkerboard (before rounding)."""
    lon, lat, (clat, clon) = geo_grid(m)
    LO, LA = np.meshgrid(lon, lat)                      # [ny, nx]
    xk = (LO - clon) * KM * np.cos(np.radians(LA))
    yk = (LA - clat) * KM
    sh = C["phase_shift"] * phase
    s2 = (-1.0) ** (np.floor(xk / L + sh) + np.floor(yk / L + sh))
    bands = np.array(C["bands_km"], float)
    dv = np.zeros_like(m["vp"])
    for k, z in enumerate(m["z"]):
        b = np.searchsorted(bands, z, side="right") - 1
        if 0 <= b < len(bands) - 1:
            dv[k] = C["amp"] * s2 * (-1.0) ** b
    dv[[0, -1]] = 0
    dv[:, [0, -1]] = 0
    dv[:, :, [0, -1]] = 0
    return dv


def prep():
    m = sp.read_mod(INP / "MOD")
    for pat in patterns():
        L, ph = parse_pat(pat)
        vp = np.round(m["vp"] * (1 + true_pattern(m, L, ph)), 3)
        d = TEST / f"chk_{pat}"
        sp.prep_syn_dir(d, lambda path: sp.write_mod(path, m, vp, exact=True), inp=INP)
        shutil.rmtree(d / "inputs")                    # inversion inputs live in the variant dirs
        np.save(d / "true_dvp.npy", vp / m["vp"] - 1)
        print(pat, f"{(vp != m['vp']).sum()} nodes perturbed")
    write_manifest("b2_res_checker_prep", TEST, inputs=[INP / "MOD", INP / "forward.json"],
                   outputs=[TEST / f"chk_{p}" / "syn" / "MOD" for p in patterns()], params=C,
                   counts={"patterns": len(patterns())})


def syn(pat):
    sd = TEST / f"chk_{pat}" / "syn"
    sp.run_syn(sd, pat)
    print(pat, "forward done")


def noise_draw():
    p = pd.read_csv(INP / "picks_used.csv", usecols=["event_ID", "code", "phase", "sigma_s"])
    rng = np.random.default_rng(C["noise_seed"])
    e = rng.standard_normal(len(p)) * p["sigma_s"].to_numpy()
    return dict(zip(zip(p["event_ID"].astype(int), p["code"], p["phase"]), e)), float(np.sqrt(np.mean(e ** 2)))


def perturbed_events(path_out):
    ev = pd.read_csv(INP / "events.csv", usecols=["ID", "h_max", "sig_z"]).set_index("ID")
    rng = np.random.default_rng(C["noise_seed"] + 1)
    lines, sh, sz = [], [], []
    for line in (INP / "event.dat").read_text().splitlines():
        t = line.split()
        eid = int(t[9])
        sh_ = ev.loc[eid, "h_max"] / C["loc_h_from_68"]
        dx, dy = rng.standard_normal(2) * sh_
        dz = rng.standard_normal() * ev.loc[eid, "sig_z"]
        lat, lon, dep = float(t[2]), float(t[3]), float(t[4])
        lat2 = lat + dy / KM
        lon2 = lon + dx / (KM * np.cos(np.radians(lat)))
        dep2 = max(dep + dz, min(dep, 0.0))
        sh.append(np.hypot(dx, dy))
        sz.append(dep2 - dep)
        lines.append(f"{t[0]}  {t[1]}  {lat2:9.6f} {lon2:11.6f} {dep2:8.3f} {' '.join(t[5:9])} {eid:10d} {t[10]}")
    path_out.write_text("\n".join(lines) + "\n")
    return {"loc_shift_h_med_km": float(np.median(sh)), "loc_shift_z_absmed_km": float(np.median(np.abs(sz))),
            "loc_shift_h_p90_km": float(np.percentile(sh, 90)), "loc_shift_z_abs_p90_km": float(np.percentile(np.abs(sz), 90))}


def make_inputs(pat, variant):
    d = TEST / f"chk_{pat}_{variant}"
    out = d / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    sd = TEST / f"chk_{pat}" / "syn"
    for f in ("MOD", "summary.json", "forward.json"):
        shutil.copy(INP / f, out / f)
    if not (out / "station.dat").exists():
        (out / "station.dat").symlink_to(INP / "station.dat")
    noise, counts = None, {}
    if variant.startswith("noise"):
        noise, rms = noise_draw()
        counts["noise_rms_s"] = rms
    counts.update(sp.syn_to_inputs(sd, out, inp=INP, noise=noise))
    if variant == "noise_ploc":
        if (out / "event.dat").exists():
            (out / "event.dat").unlink()
        counts.update(perturbed_events(out / "event.dat"))
    elif not (out / "event.dat").exists():
        (out / "event.dat").symlink_to(INP / "event.dat")
    print(pat, variant, counts)
    write_manifest(f"b2_res_checker_inputs_{pat}_{variant}", d,
                   inputs=[sd / "MOD", sd / "absolute.syn", sd / "dt.syn", INP / "absolute.dat", INP / "dt.ct"],
                   outputs=[out / "absolute.dat", out / "dt.ct", out / "event.dat"],
                   params={"pattern": pat, "variant": variant, **C}, counts=counts)


# ---------------------------------------------------------------- analysis
def read_model(path, shape):
    return np.loadtxt(path).reshape(shape)


def box_sum(a, hy, hx):
    """Sum of a [ny, nx] over the (2hy+1) x (2hx+1) window around every node (zero outside)."""
    c = np.cumsum(np.cumsum(np.pad(a, ((hy + 1, hy), (hx + 1, hx))), 0), 1)
    ny, nx = a.shape
    return (c[2 * hy + 1:2 * hy + 1 + ny, 2 * hx + 1:2 * hx + 1 + nx] - c[:ny, 2 * hx + 1:2 * hx + 1 + nx]
            - c[2 * hy + 1:2 * hy + 1 + ny, :nx] + c[:ny, :nx])


def local_metrics(t, r, hy, hx):
    """Per-node local correlation and amplitude of r against t (2D layers [ny, nx], interior only)."""
    w = np.zeros_like(t)
    w[1:-1, 1:-1] = 1
    t, r = t * w, r * w
    n = box_sum(w, hy, hx)
    st, sr = box_sum(t, hy, hx), box_sum(r, hy, hx)
    stt, srr, str_ = box_sum(t * t, hy, hx), box_sum(r * r, hy, hx), box_sum(t * r, hy, hx)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = str_ - st * sr / n
        vt = stt - st ** 2 / n
        vr = srr - sr ** 2 / n
        corr = np.where(vr > 1e-12, cov / np.sqrt(vt * vr), 0.0)
        amp = np.where(stt > 0, str_ / stt, np.nan)
    return corr, amp


def analyse(mask_smooth):
    import matplotlib
    matplotlib.use("Agg")
    m = sp.read_mod(INP / "MOD")
    nz, ny, nx = m["vp"].shape
    vp0 = m["vp"]
    vs0 = vp0 / m["vpvs"]
    lon, lat, (clat, clon) = geo_grid(m)
    z = m["z"]
    dx = 0.125 * KM * np.cos(np.radians(clat))       # node spacing (km), interior
    dy = 0.1 * KM
    reg = T["regions"][R["region"]]
    in_box = np.zeros((ny, nx), bool)
    jj = (lat >= reg["events_lat"][0]) & (lat <= reg["events_lat"][1])
    ii = (lon >= reg["events_lon"][0]) & (lon <= reg["events_lon"][1])
    in_box[np.ix_(jj, ii)] = True
    area = dx * dy * np.cos(np.radians(lat))[:, None] / np.cos(np.radians(clat)) * np.ones((1, nx))
    rows, fields = [], {}
    runs = sorted(TEST.glob(f"chk_L*_p*_*/inv_d{R['damp']:g}_s*"))   # this damping only
    for run in runs:
        o = run / "Output_Files"
        if not (o / "Vs_model.dat").exists():
            continue
        case = run.parent.name[4:]                    # L060_p0_noise
        L, ph = parse_pat("_".join(case.split("_")[:2]))
        variant = "_".join(case.split("_")[2:])
        tag = run.name
        tdv = np.load(TEST / f"chk_L{L:03d}_p{ph}" / "true_dvp.npy")
        hx = int(round(C["window_cells"] * L / 2 / dx))
        hy = int(round(C["window_cells"] * L / 2 / dy))
        for wave, f, ref in (("P", "Vp_model.dat", vp0), ("S", "Vs_model.dat", vs0)):
            rdv = read_model(o / f, (nz, ny, nx)) / ref - 1
            corr = np.full((nz, ny, nx), np.nan)
            amp = np.full((nz, ny, nx), np.nan)
            for k in range(1, nz - 1):
                if not np.any(tdv[k]):
                    continue
                corr[k], amp[k] = local_metrics(tdv[k], rdv[k], hy, hx)
            resolved = (corr >= C["resolved"]["corr"]) & (amp >= C["resolved"]["amp"])
            fields[(tag, case, wave)] = dict(rec=rdv, corr=corr, amp=amp, res=resolved, true=tdv)
            for k in range(1, nz - 1):
                if not np.any(tdv[k]):
                    continue
                sel = in_box
                rows.append(dict(inv=tag, smooth=int(tag.split("_s")[1]), L=L, phase=ph, variant=variant, wave=wave,
                                 z=z[k], n_box=int(sel.sum()), n_res_box=int(resolved[k][sel].sum()),
                                 frac_res_box=float(resolved[k][sel].mean()),
                                 area_res_km2=float(area[resolved[k] & np.ones_like(sel)].sum()),
                                 corr_med_box=float(np.nanmedian(corr[k][sel])), amp_med_box=float(np.nanmedian(amp[k][sel])),
                                 rec_rms_box=float(np.sqrt(np.mean(rdv[k][sel] ** 2)) / C["amp"])))
        loc = loc_errors(run)
        if loc:
            rows.append(dict(inv=tag, smooth=int(tag.split("_s")[1]), L=L, phase=ph, variant=variant, wave="loc", **loc))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "checker_by_depth.csv", index=False, float_format="%.4g")
    masks = build_masks(fields, mask_smooth, lon, lat, z, nz, ny, nx)
    np.savez_compressed(OUT / "resolution_mask.npz", **masks)
    summary_table(df, masks, area, in_box, lon, lat, z)
    plot_all(fields, masks, mask_smooth, lon, lat, z)
    write_manifest("b2_res_checker_analyse", OUT, inputs=[r / "Output_Files" / "Vp_model.dat" for r in runs],
                   outputs=sorted(OUT.glob("*")), params={**C, "mask_smooth": mask_smooth},
                   counts={"runs": len(fields) // 2})
    return df


def loc_errors(run):
    """Event location error against the true (event.dat of the region inputs), start and final (MOD coordinates)."""
    f = run / "Output_Files" / "tomoDD.reloc"
    if not f.exists():
        return None
    tru = pd.read_csv(INP / "event.dat", sep=r"\s+", header=None).set_index(9)[[2, 3, 4]]
    st = pd.read_csv(run / "Input_Files" / "event.dat", sep=r"\s+", header=None).set_index(9)[[2, 3, 4]]
    rel = pd.read_csv(f, sep=r"\s+", header=None).set_index(0)[[1, 2, 3]]
    rel.columns = st.columns
    out = {}
    for name, a in (("start", st), ("final", rel)):
        b = tru.reindex(a.index)
        dh = np.hypot((a[2] - b[2]) * KM, (a[3] - b[3]) * KM * np.cos(np.radians(b[2])))
        dz = a[4] - b[4]
        out.update({f"{name}_h_med_km": float(np.median(dh)), f"{name}_h_p90_km": float(np.percentile(dh, 90)),
                    f"{name}_z_absmed_km": float(np.median(np.abs(dz))), f"{name}_z_med_km": float(np.median(dz)),
                    f"{name}_z_abs_p90_km": float(np.percentile(np.abs(dz), 90))})
    out["n_events"] = len(rel)
    return out


def build_masks(fields, mask_smooth, lon, lat, z, nz, ny, nx):
    """Finest resolved cell size per node (km; 0 = not resolved at 100 km), per wave, for smoothing mask_smooth.
    A scale counts as resolved at a node if the noisy runs of both pattern phases (p0, p1) resolve it; where
    only p0 exists, p0 alone."""
    out = dict(lon=lon, lat=lat, z=z, cell_km=np.array(C["cell_km"]))
    tagw = f"_s{mask_smooth}"
    for wave in ("P", "S"):
        best = np.zeros((nz, ny, nx))
        for L in sorted(C["cell_km"], reverse=True):
            got = [v["res"] for (tag, case, w), v in fields.items()
                   if tag.endswith(tagw) and w == wave and case.startswith(f"L{L:03d}_") and case.endswith("_noise")]
            if not got:
                continue
            ok = np.logical_and.reduce(got)
            best[ok] = L
        out[f"finest_{wave.lower()}"] = best.transpose(2, 1, 0)       # [nx, ny, nz] like model_geo.npz
    return out


def summary_table(df, masks, area, in_box, lon, lat, z):
    """Markdown tables: per depth and smoothing, fraction of event-box nodes resolved (noise, p0) for each L."""
    d = df[(df.wave != "loc") & (df.variant == "noise") & (df.phase == 0)]
    piv = d.pivot_table(index="z", columns=["wave", "L", "smooth"], values="frac_res_box")
    lines = ["# Checkerboard recovery: fraction of event-box nodes resolved (noise, phase 0)", ""]
    for wave in ("P", "S"):
        if wave not in piv.columns.get_level_values(0):
            continue
        p = piv[wave]
        cols = list(p.columns)
        lines += [f"## V{wave.lower()}", "", "| z (km) | " + " | ".join(f"L{L} s{s}" for L, s in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for zz, r in p.iterrows():
            lines.append(f"| {zz:g} | " + " | ".join("" if np.isnan(v) else f"{100 * v:.0f}%" for v in r) + " |")
        lines.append("")
    # mask: resolved area per depth for each finest scale
    lines += ["# Resolution mask (finest resolved cell, noisy runs, both phases)", "",
              "| z (km) | Vp 40 km area (km2) | Vp <=60 | Vp <=100 | Vs 40 | Vs <=60 | Vs <=100 |", "|---|---|---|---|---|---|---|"]
    a = area.T                                                    # [nx, ny]
    for k, zz in enumerate(z[1:-1], start=1):
        r = []
        for wave in ("p", "s"):
            f = masks[f"finest_{wave}"][:, :, k]
            r += [a[(f > 0) & (f <= L)].sum() for L in (40, 60, 100)]
        if any(r):
            lines.append(f"| {zz:g} | " + " | ".join(f"{v:,.0f}" for v in r) + " |")
    lo = df[df.wave == "loc"]
    if len(lo):
        lo = lo.drop(columns=[c for c in lo.columns if lo[c].isna().all()])
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731 (no tabulate in the env)
        lines += ["", "# Location recovery (km; MOD coordinates vs true)", "",
                  "| " + " | ".join(lo.columns) + " |", "|" + "---|" * len(lo.columns)]
        lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in lo.itertuples(index=False)]
    (OUT / "summary_tables.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- figures
SLICES = [3, 7, 12, 21, 34]
XS = {k: tuple(map(tuple, v)) for k, v in R["sections"].items()}   # params res_tests(.regions.<tile>).sections


def coast():
    f = OUT / "coast_i.txt"
    if not f.exists():
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:          # gmt writes gmt.history in its cwd
            reg = T["regions"][R["region"]]
            rr = f"-R{reg['node_lon'][0] - 1}/{reg['node_lon'][1] + 1}/{reg['node_lat'][0] - 0.5}/{reg['node_lat'][1] + 0.5}"
            txt = subprocess.run([str(Path(sys.executable).parent / "gmt"), "coast", rr, "-Di", "-W", "-M"],
                                 capture_output=True, text=True, cwd=td).stdout
        f.write_text(txt)
    segs, cur = [], []
    for line in f.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                segs.append(np.array(cur))
            cur = []
        else:
            cur.append([float(v) for v in line.split()[:2]])
    if cur:
        segs.append(np.array(cur))
    return segs


def section(field, lon, lat, z, p0, p1, n=200, method="linear"):
    from scipy.interpolate import RegularGridInterpolator
    g = RegularGridInterpolator((z, lat, lon), field, bounds_error=False, fill_value=np.nan, method=method)
    la = np.linspace(p0[0], p1[0], n)
    lo = np.linspace(p0[1], p1[1], n)
    dist = np.hypot((la - la[0]) * KM, (lo - lo[0]) * KM * np.cos(np.radians(la)))
    zz = np.linspace(-3, 50, 107)
    Z, I = np.meshgrid(zz, np.arange(n), indexing="ij")
    v = g(np.stack([Z.ravel(), la[I.ravel()], lo[I.ravel()]], 1)).reshape(Z.shape)
    return dist, zz, v


MTJ_BOX = R.get("highlight_box")   # Mendocino-Gorda: MTJ shallow-anomaly box (b2_mtj_common.MTJ); other tiles none


def category(f):
    """finest resolved cell (0, 40, 60, 100) -> 0..3"""
    return np.select([f == 40, f == 60, f == 100], [1, 2, 3], 0).astype(float)


def plot_all(fields, masks, mask_smooth, lon, lat, z):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    segs = coast()
    reg = T["regions"][R["region"]]
    ink2 = "#52514e"
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
    ext = (lon[1], lon[-2], lat[1], lat[-2])
    ks = [int(np.argmin(np.abs(z - s))) for s in SLICES]
    cmap = ListedColormap(["#f0efec", "#1f5fae", "#5f9fd9", "#b7d5f0"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    labels = ["none", "40 km", "60 km", "100 km"]

    def deco(a, xs=True):
        for sg in segs:
            a.plot(sg[:, 0], sg[:, 1], color="k", lw=0.4)
        bx, by = reg["events_lon"], reg["events_lat"]
        a.plot([bx[0], bx[1], bx[1], bx[0], bx[0]], [by[0], by[0], by[1], by[1], by[0]], color=ink2, lw=0.5, ls="--")
        if MTJ_BOX:
            bx, by = MTJ_BOX["lon"], MTJ_BOX["lat"]
            a.plot([bx[0], bx[1], bx[1], bx[0], bx[0]], [by[0], by[0], by[1], by[1], by[0]], color="#e0a500", lw=0.8)
        if xs:
            for name, (p0, p1) in XS.items():
                a.plot([p0[1], p1[1]], [p0[0], p1[0]], color="#7a3fb0", lw=0.6)
                a.text(p0[1], p0[0], name[0], color="#7a3fb0", fontsize=6, va="top")
        a.set_xlim(ext[0], ext[1])
        a.set_ylim(ext[2], ext[3])
        a.set_aspect(1 / np.cos(np.radians(np.mean(reg["events_lat"]))))

    # 1. recovered checkerboards (noise, p0) at the mask smoothing: per wave, rows = depth, cols = L
    Ls = C["cell_km"]
    for wave in ("P", "S"):
        fig, ax = plt.subplots(len(SLICES), len(Ls), figsize=(6.4, 9.4), sharex=True, sharey=True,
                               constrained_layout=True)
        im = None
        for r, k in enumerate(ks):
            for c, L in enumerate(Ls):
                key = (f"inv_d{R['damp']}_s{mask_smooth}", f"L{L:03d}_p0_noise", wave)
                a = ax[r, c]
                if key not in fields:
                    a.set_axis_off()
                    continue
                fv = fields[key]
                im = a.pcolormesh(lon, lat, 100 * fv["rec"][k], cmap="RdBu", vmin=-8, vmax=8, shading="nearest")
                a.contour(lon, lat, fv["true"][k], levels=[0], colors=ink2, linewidths=0.3, linestyles=":")
                a.contour(lon, lat, fv["res"][k].astype(float), levels=[0.5], colors="k", linewidths=0.8)
                deco(a, xs=False)
                if r == 0:
                    a.set_title(f"L = {L} km", fontsize=7)
                if c == 0:
                    a.set_ylabel(f"z = {z[k]:g} km")
        if im is not None:
            fig.colorbar(im, ax=ax, shrink=0.35, label=f"recovered dln{'Vp' if wave == 'P' else 'Vs'} (%), true +-8%")
        fig.suptitle(f"Checkerboard {'Vp' if wave == 'P' else 'Vs'}, damp {R['damp']}, smooth {mask_smooth}, pick noise. "
                     "Black: resolved (corr >= 0.7, amp >= 0.5); dotted: true cell edges" + ("; yellow: MTJ box" if MTJ_BOX else ""), fontsize=6.5)
        fig.savefig(OUT / f"checker_maps_{wave.lower()}_s{mask_smooth}.png", dpi=200)
        plt.close(fig)

    # 2. resolution mask maps (finest resolved cell size)
    fig, ax = plt.subplots(2, len(SLICES), figsize=(7.2, 3.3), sharex=True, sharey=True, constrained_layout=True)
    for r, wave in enumerate(("p", "s")):
        for c, k in enumerate(ks):
            a = ax[r, c]
            im = a.pcolormesh(lon, lat, category(masks[f"finest_{wave}"][:, :, k].T), cmap=cmap, norm=norm,
                              shading="nearest")
            deco(a)
            if r == 0:
                a.set_title(f"z = {z[k]:g} km", fontsize=7)
            if c == 0:
                a.set_ylabel("Vp" if wave == "p" else "Vs")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(labels)
    cb.set_label("finest resolved checker")
    fig.suptitle(f"Resolution mask, damp {R['damp']}, smooth {mask_smooth} (pick noise, both pattern phases)", fontsize=7)
    fig.savefig(OUT / f"resolution_mask_maps_s{mask_smooth}.png", dpi=200)
    plt.close(fig)

    # 3. cross-sections: recovered (L 60, noise p0) and mask, Vp and Vs
    fig, ax = plt.subplots(len(XS), 4, figsize=(7.2, 4.4), constrained_layout=True)
    im = im2 = None
    for r, (name, (p0, p1)) in enumerate(XS.items()):
        for c, (wave, kind) in enumerate((("P", "rec60"), ("S", "rec60"), ("P", "mask"), ("S", "mask"))):
            a = ax[r, c]
            if kind == "mask":
                f = category(masks[f"finest_{wave.lower()}"]).transpose(2, 1, 0)
                dist, zz, v = section(f, lon, lat, z, p0, p1, method="nearest")
                im2 = a.pcolormesh(dist, zz, v, cmap=cmap, norm=norm, shading="nearest")
                a.set_title(f"{name[:1]}: V{wave.lower()} mask", fontsize=7)
            else:
                key = (f"inv_d{R['damp']}_s{mask_smooth}", "L060_p0_noise", wave)
                if key not in fields:
                    a.set_axis_off()
                    continue
                dist, zz, v = section(100 * fields[key]["rec"], lon, lat, z, p0, p1)
                _, _, vt = section(100 * fields[key]["true"], lon, lat, z, p0, p1)
                im = a.pcolormesh(dist, zz, v, cmap="RdBu", vmin=-8, vmax=8, shading="nearest")
                a.contour(dist, zz, vt, levels=[0], colors=ink2, linewidths=0.4, linestyles=":")
                a.set_title(f"{name[:1]}: V{wave.lower()} recovered, L 60", fontsize=7)
            a.set_ylim(50, -3)
            a.set_xlabel(f"km from {abs(p0[0]):.1f}N {abs(p0[1]):.1f}W", fontsize=6)
            if c == 0:
                a.set_ylabel(f"{name}\ndepth (km)", fontsize=6)
    if im is not None:
        fig.colorbar(im, ax=ax[:, :2], shrink=0.5, label="dlnV (%), true +-8%", location="bottom")
    cb = fig.colorbar(im2, ax=ax[:, 2:], shrink=0.5, location="bottom", ticks=[0, 1, 2, 3])
    cb.ax.set_xticklabels(labels)
    cb.set_label("finest resolved checker")
    fig.savefig(OUT / f"resolution_sections_s{mask_smooth}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "syn":
        syn(sys.argv[2])
    elif cmd == "inputs":
        make_inputs(sys.argv[2], sys.argv[3])
    elif cmd == "analyse":
        ms = int(sys.argv[sys.argv.index("--mask-smooth") + 1]) if "--mask-smooth" in sys.argv else 20
        analyse(ms)
