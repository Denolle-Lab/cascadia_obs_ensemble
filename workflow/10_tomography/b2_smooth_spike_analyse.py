"""Track B, B2: analyse the smoothing spike test (b2_smooth_spike_test.py).

For every spike: recovered peak (fraction of the true +A%), half-width at half maximum (HWHM) of the
recovered anomaly along x, y and up/down in z, from node values with linear interpolation between nodes.
The true spike is a single node, so its HWHM is half the node spacing in each direction; the broadening
factor is recovered / true HWHM. The null run (synthetics from the background model) is subtracted first.
Recovery is bimodal (rays reach a node or the node is fixed below the DWS threshold), so spikes with recovered
peak < `min_rec` of the true amplitude count as unresolved; widths and recovery are summarised over the resolved
spikes, and the fraction resolved is reported separately.
Also writes the per-layer spacing table: dz, lateral spacing, the gradient-equivalent vertical/lateral
smoothing ratio (w3 dz)/(w1 dx), and the mean P DWS inside the event box from the check01 run.

  python b2_smooth_spike_analyse.py   ->  data/tomo_v1/b2_smooth_test/{spikes.csv, by_depth.csv, layers.csv, psf.png}
  python b2_smooth_spike_analyse.py --cfg res_tests --inv inv_d1000_s20
                                      ->  data/tomo_v1/b2_res_tests/spike_inv_d1000_s20/{same files}
  (damping and smoothing of the run are read from its tomoDD-SE.inp)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
import b2_smooth_spike_test as st  # noqa: E402
from b2_smooth_spike_test import T, read_mod  # noqa: E402

OUT = REPO / "data" / "tomo_v1" / "b2_smooth_test"
INV = "inv"
MIN_REC = 0.5
KM_DEG = 111.195


def read_model(path, shape):
    return np.loadtxt(path).reshape(shape)


def hwhm_1d(vals, pos, c):
    """Half-width at half max on each side of index c of profile vals at positions pos (km)."""
    half = vals[c] / 2
    out = []
    for step in (-1, 1):
        i = c
        w = np.nan
        while 0 <= i + step < len(vals):
            a, b = vals[i], vals[i + step]
            if b <= half:
                f = (a - half) / (a - b)
                w = abs(pos[i] + f * (pos[i + step] - pos[i]) - pos[c])
                break
            i += step
        out.append(w)
    return out  # [toward lower index, toward higher index]


def dws_by_layer(m):
    # DWS of the first iteration (background-model rays): check01 when it has it, else this test's null run
    # (same events, stations and settings; mendocino_gorda_v2/check01 was written without the DWS block)
    for vel in (st.BASE / "check01" / "Output_Files" / "tomoDD.vel", st.zdir("null") / INV / "Output_Files" / "tomoDD.vel"):
        lines = vel.read_text().split("\n") if vel.exists() else []
        i0 = next((i for i, l in enumerate(lines) if "DWS for P-wave" in l), None)
        if i0 is not None:
            break
    else:
        raise SystemExit("no DWS block in check01 or the null run")
    nz, ny, nx = m["vp"].shape
    d = np.array([lines[i0 + 1 + r].split() for r in range(ny * nz)], float).reshape(nz, ny, nx)
    lo, la = T["regions"][st.REG]["events_lon"], T["regions"][st.REG]["events_lat"]
    ii = (m["x"] >= lo[0]) & (m["x"] <= lo[1])
    jj = (m["y"] >= la[0]) & (m["y"] <= la[1])
    return d[:, jj][:, :, ii]


def run_settings():
    """(damp, [w1, w2, w3]) of the null inversion (tomoDD-SE.inp)."""
    lines = (st.zdir("null") / INV / "tomoDD-SE.inp").read_text().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("* weight1")][0]
    sm = [float(v) for v in lines[i + 1].split()[:3]]
    j = [k for k, l in enumerate(lines) if l.startswith("* NITER")][0]
    return float(lines[j + 1].split()[10]), sm


def main():
    S = st.S
    damp, smooth = run_settings()
    w1, w2, w3 = smooth
    m, sp = st.spikes()
    nz, ny, nx = m["vp"].shape
    x, y, z = m["x"], m["y"], m["z"]
    vs0 = m["vp"] / m["vpvs"]
    null = st.zdir("null") / INV / "Output_Files"
    nvp = read_model(null / "Vp_model.dat", (nz, ny, nx)) / m["vp"] - 1
    nvs = read_model(null / "Vs_model.dat", (nz, ny, nx)) / vs0 - 1
    rows = []
    for ztag, k, nodes in sp:
        o = st.zdir(ztag) / INV / "Output_Files"
        if not (o / "Vp_model.dat").exists():
            print("missing", ztag)
            continue
        for wave, f, ref, nul in (("P", "Vp_model.dat", m["vp"], nvp), ("S", "Vs_model.dat", vs0, nvs)):
            dv = read_model(o / f, (nz, ny, nx)) / ref - 1 - nul
            for j, i in nodes:
                kx = np.cos(np.radians(y[j])) * KM_DEG
                pk = dv[k, j, i]
                win = dv[max(k - 2, 1):k + 3, j - 2:j + 3, i - 2:i + 3]
                r = dict(ztag=ztag, z=z[k], wave=wave, lon=x[i], lat=y[j], rec=pk / S["amp"],
                         rec_max_win=win.max() / S["amp"], peak_at_node=bool(pk >= win.max() - 1e-9),
                         dz_up=z[k] - z[k - 1], dz_dn=z[k + 1] - z[k], dx_km=0.125 * kx, dy_km=0.1 * KM_DEG,
                         nul_rms=float(np.sqrt(np.mean(nul[k, j - 2:j + 3, i - 2:i + 3] ** 2))) / S["amp"])
                if pk / S["amp"] >= MIN_REC:
                    hx = hwhm_1d(dv[k, j, :], x * kx, i)
                    hy = hwhm_1d(dv[k, :, i], y * KM_DEG, j)
                    hz = hwhm_1d(dv[:, j, i], z, k)
                    r.update(hw_x=np.nanmean(hx), hw_y=np.nanmean(hy), hw_up=hz[0], hw_dn=hz[1])
                    r.update(bx=r["hw_x"] / (r["dx_km"] / 2), by=r["hw_y"] / (r["dy_km"] / 2),
                             bup=r["hw_up"] / (r["dz_up"] / 2), bdn=r["hw_dn"] / (r["dz_dn"] / 2))
                rows.append(r)
    sp_df = pd.DataFrame(rows)
    for c in ("hw_x", "hw_y", "hw_up", "hw_dn", "bx", "by", "bup", "bdn"):   # no spike above MIN_REC (e.g. ridges_north)
        if c not in sp_df:
            sp_df[c] = np.nan
    OUT.mkdir(parents=True, exist_ok=True)
    sp_df.to_csv(OUT / "spikes.csv", index=False, float_format="%.4g")

    agg = {"rec": "median", "hw_x": "median", "hw_y": "median", "hw_up": "median", "hw_dn": "median",
           "bx": "median", "by": "median", "bup": "median", "bdn": "median", "dz_up": "first", "dz_dn": "first",
           "nul_rms": "median"}
    res = sp_df[sp_df["rec"] >= MIN_REC]
    by = res.groupby(["wave", "z"]).agg(agg).reindex(sp_df.groupby(["wave", "z"]).size().index)
    by["n_resolved"] = res.groupby(["wave", "z"]).size().reindex(by.index).fillna(0).astype(int)
    by["n"] = sp_df.groupby(["wave", "z"]).size()
    by.to_csv(OUT / "by_depth.csv", float_format="%.3g")

    # per-layer spacing table
    dws = dws_by_layer(m)
    dx = 0.125 * np.cos(np.radians(41.3)) * KM_DEG
    lay = []
    for k in range(1, nz - 1):
        dzu, dzd = z[k] - z[k - 1], z[k + 1] - z[k]
        lay.append(dict(z=z[k], dz_up=dzu, dz_dn=dzd, dx_km=dx, dy_km=0.1 * KM_DEG,
                        grad_ratio_zx=(w3 * dzd) / (w1 * dx), dws_p_mean=dws[k].mean(),
                        dws_p_per_km=dws[k].mean() / ((dzu + dzd) / 2), frac_nodes_dws_gt_thres=(dws[k] > T["inp"]["thres"]).mean()))
    lay = pd.DataFrame(lay)
    lay.to_csv(OUT / "layers.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(lay.to_string(index=False))
    print(by.round(2).to_string())
    plot(by, OUT / "psf.png", damp, smooth)
    write_manifest("b2_smooth_spike_analyse", OUT,
                   inputs=[st.zdir(t) / INV / "Output_Files" / "Vp_model.dat" for t, _, _ in sp] + [null / "Vp_model.dat"],
                   outputs=[OUT / f for f in ("spikes.csv", "by_depth.csv", "layers.csv", "psf.png")],
                   params={"min_rec": MIN_REC, **S, "smooth": smooth, "damp": damp, "cfg": st.CFG["name"], "inv": INV},
                   counts={"spikes": len(sp_df)})


def plot(by, path, damp, smooth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, ink2, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 6.2), sharey=True, constrained_layout=True)
    for r, wave in enumerate(("P", "S")):
        d = by.loc[wave].reset_index()
        a = ax[r, 0]
        a.plot(100 * d["rec"], d["z"], "-o", color=blue, lw=2, ms=5, label="median peak of resolved spikes")
        a.plot(100 * d["n_resolved"] / d["n"], d["z"], "-o", color=orange, lw=2, ms=5, label="spikes resolved (peak >= 50%)")
        a.axvline(100, color=ink2, lw=0.8, ls=":")
        a.legend(frameon=False, loc="lower left")
        a.set_xlabel(f"% ({'Vp' if wave == 'P' else 'Vs'}, 25 spikes per depth)")
        a.set_ylabel("spike depth (km)")
        a.set_xlim(0, 110)
        a = ax[r, 1]
        a.plot(d["hw_x"], d["z"], "-o", color=blue, lw=2, ms=5, label="lateral (x)")
        a.plot((d["hw_up"] + d["hw_dn"]) / 2, d["z"], "-o", color=orange, lw=2, ms=5, label="vertical (mean up/down)")
        a.plot((d["dz_up"] + d["dz_dn"]) / 4, d["z"], "--", color=orange, lw=1, alpha=0.7)
        a.axvline(0.125 * np.cos(np.radians(41.3)) * KM_DEG / 2, color=blue, lw=1, ls="--", alpha=0.7)
        a.set_xlabel("half-width at half max (km); dashed = true spike")
        a.legend(frameon=False, loc="lower right")
        for aa in ax[r]:
            aa.grid(color=grid, lw=0.6)
            aa.set_axisbelow(True)
    ax[0, 0].invert_yaxis()
    fig.suptitle(f"tomoDD spike test, {st.REG} grid, smooth {[f'{v:g}' for v in smooth]}, damp {damp:g}",
                 fontsize=9, color=ink)
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--cfg" in a:
        st.use(a[a.index("--cfg") + 1])
        if st.CFG["name"] != "smooth_test":
            INV = a[a.index("--inv") + 1] if "--inv" in a else st.CFG["inv_tags"][0]
            OUT = REPO / "data" / "tomo_v1" / "b2_res_tests" / st.REG / f"spike_{INV}"
    main()
