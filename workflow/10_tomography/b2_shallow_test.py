"""Track B, B2: does adding shallow events (b2_shallow_events.py) resolve the upper crust? Spike test, v2 vs v2s.

Spikes (+amp Vp, Vp/Vs unchanged) at the MOD node nearest each station inside the event box (stations taken in
order of their pick count, a node kept if it is >= min_sep_nodes from all kept nodes in i and j), one run per depth
(depths_km), the same spike set for both regions (same MOD and stations; only the event set differs). Noise-free
synthetics (tomoDD-SE_syn, e13k build) on each region's own picks, pairs and weights; inversion from the
background MOD and true locations with damp/smooth of params `tomodd.shallow_test` and the e13k binary for both.
A null run per region (synthetics from the background MOD) is subtracted.

  python b2_shallow_test.py prep                  # dirs <runs>/<region>/<subdir>/{z..,null}/{syn,inputs}
  python b2_shallow_test.py syn REGION ZTAG       # forward run + synthetic inputs
  python b2_shallow_test.py inv REGION ZTAG       # inversion (b2_tomodd_run.py, tag inv)
  python b2_shallow_test.py analyse               # -> data/tomo_v1/b2_shallow_test/
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b2_smooth_spike_test import prep_syn_dir, read_mod, syn_to_inputs, write_mod  # noqa: E402

P = load_params()
T = P["tomodd"]
C = T["shallow_test"]
OUT = REPO / "data" / "tomo_v1" / "b2_shallow_test"
PY = sys.executable


def rdir(region):
    return Path(T["runs"]) / region / C["subdir"]


def ztag(z):
    return f"z{z:+04.0f}".replace("+", "p").replace("-", "m")


def spike_nodes():
    """[(j, i, station)] from the first region's inputs (station.dat in MOD coordinates)."""
    inp = Path(T["runs"]) / C["regions"][0] / "inputs"
    m = read_mod(inp / "MOD")
    st = pd.read_csv(inp / "station.dat", sep=r"\s+", header=None, names=["code", "lat", "lon", "elev"])
    codes = pd.read_csv(inp / "station_codes.csv")
    st = st.merge(codes[["code", "station", "sta_lat", "sta_lon"]], on="code")
    R = T["regions"][C["regions"][0]]
    st = st[st.sta_lat.between(*R["events_lat"]) & st.sta_lon.between(*R["events_lon"])]
    npk = pd.read_csv(inp / "picks_used.csv", usecols=["code"]).code.value_counts()
    st = st.assign(n=st.code.map(npk).fillna(0)).sort_values(["n", "code"], ascending=[False, True])
    out = []
    for r in st.itertuples():
        i = int(np.argmin(np.abs(m["x"] - r.lon)))
        j = int(np.argmin(np.abs(m["y"] - r.lat)))
        if all(max(abs(i - i2), abs(j - j2)) >= C["min_sep_nodes"] for j2, i2, _ in out):
            out.append((j, i, r.station))
    return m, out


def prep():
    m, nodes = spike_nodes()
    pd.DataFrame(nodes, columns=["j", "i", "station"]).to_csv(OUT.parent / "b2_shallow_test_nodes.csv", index=False) \
        if OUT.parent.exists() else None
    for region in C["regions"]:
        inp = Path(T["runs"]) / region / "inputs"
        for z in C["depths_km"] + [None]:
            tag = "null" if z is None else ztag(z)
            vp = m["vp"].copy()
            if z is not None:
                k = int(np.argmin(np.abs(m["z"] - z)))
                assert abs(m["z"][k] - z) < 1e-6
                for j, i, _ in nodes:
                    vp[k, j, i] *= 1 + C["amp"]
            vp = np.round(vp, 3)
            d = rdir(region) / tag
            prep_syn_dir(d, lambda path: write_mod(path, m, vp, exact=True), inp=inp)
            np.save(d / "true_dvp.npy", vp / m["vp"] - 1)
        print(region, len(nodes), "spike nodes per depth")
    write_manifest("b2_shallow_test_prep", rdir(C["regions"][0]), params=C,
                   inputs=[Path(T["runs"]) / r / "inputs" / f for r in C["regions"] for f in ("MOD", "absolute.dat")],
                   counts={"spikes_per_depth": len(nodes), "nodes": [list(map(str, n)) for n in nodes]})


def syn(region, tag):
    d = rdir(region) / tag
    sd = d / "syn"
    with open(sd / "stdout.txt", "w") as so:
        rc = subprocess.run([T["bin_syn_e13k"], "tomoDD-SE_syn.inp"], cwd=sd, stdout=so, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f"{region}/{tag}: tomoDD-SE_syn exit {rc}")
    inp = Path(T["runs"]) / region / "inputs"
    counts = syn_to_inputs(sd, d / "inputs", inp=inp)
    print(region, tag, counts)
    write_manifest(f"b2_shallow_test_syn_{tag}", d, inputs=[sd / "MOD", T["bin_syn_e13k"], inp / "absolute.dat", inp / "dt.ct"],
                   outputs=[d / "inputs" / "absolute.dat", d / "inputs" / "dt.ct"], params={"region": region, "tag": tag, **C},
                   counts=counts)


def inv(region, tag):
    rel = f"{region}/{C['subdir']}/{tag}"
    cmd = [PY, str(HERE / "b2_tomodd_run.py"), rel, "inv", "--damp", str(C["damp"]),
           "--smooth", *map(str, C["smooth"]), "--bin", T["bin_e13k"]]
    sys.exit(subprocess.run(cmd).returncode)


def last_dws(vel, shape):
    lines = Path(vel).read_text().split("\n")
    i0 = max(i for i, l in enumerate(lines) if "DWS for P-wave" in l)
    nz, ny, nx = shape
    return np.array([lines[i0 + 1 + r].split() for r in range(ny * nz)], float).reshape(nz, ny, nx)


def analyse():
    m, nodes = spike_nodes()
    nz, ny, nx = m["vp"].shape
    z = m["z"]
    vs0 = m["vp"] / m["vpvs"]
    rows, lay = [], []
    for region in C["regions"]:
        base = rdir(region)
        nul = {w: np.loadtxt(base / "null/inv/Output_Files" / f).reshape(nz, ny, nx) / r0 - 1
               for w, f, r0 in (("P", "Vp_model.dat", m["vp"]), ("S", "Vs_model.dat", vs0))}
        dws = last_dws(base / "null/inv/Output_Files/tomoDD.vel", (nz, ny, nx))
        for k in range(1, nz - 1):
            lay.append(dict(region=region, z=z[k], active_frac=float((dws[k, 1:-1, 1:-1] > T["inp"]["thres"]).mean()),
                            dws_median_active=float(np.median(dws[k][dws[k] > T["inp"]["thres"]])) if (dws[k] > T["inp"]["thres"]).any() else 0.0,
                            dws_at_spikes_median=float(np.median([dws[k, j, i] for j, i, _ in nodes]))))
        for zd in C["depths_km"]:
            tag = ztag(zd)
            k = int(np.argmin(np.abs(z - zd)))
            o = base / tag / "inv" / "Output_Files"
            for w, f, r0 in (("P", "Vp_model.dat", m["vp"]), ("S", "Vs_model.dat", vs0)):
                dv = (np.loadtxt(o / f).reshape(nz, ny, nx) / r0 - 1 - nul[w]) / C["amp"]
                for j, i, sta in nodes:
                    col = dv[1:-1, j, i]
                    rows.append(dict(region=region, z=zd, wave=w, station=sta, j=j, i=i, rec=dv[k, j, i],
                                     col_sum=float(col.sum()), z_of_max=float(z[1:-1][np.argmax(col)]),
                                     lat_max=float(dv[k, j - 1:j + 2, i - 1:i + 2].max()), dws=float(dws[k, j, i])))
    d = pd.DataFrame(rows)
    lay = pd.DataFrame(lay)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "spikes.csv", index=False, float_format="%.4g")
    lay.to_csv(OUT / "layers.csv", index=False, float_format="%.4g")
    g = d.groupby(["wave", "z", "region"])
    s = pd.DataFrame({"n": g.size(), "resolved_ge50": g.rec.apply(lambda x: int((x >= 0.5).sum())),
                      "resolved_ge25": g.rec.apply(lambda x: int((x >= 0.25).sum())),
                      "rec_median": g.rec.median(), "rec_p75": g.rec.quantile(0.75),
                      "col_sum_median": g.col_sum.median(), "z_of_max_median": g.z_of_max.median()})
    s.to_csv(OUT / "by_depth.csv", float_format="%.3g")
    pd.set_option("display.width", 200)
    print(s.round(2).to_string())
    print(lay[lay.z <= 15].pivot(index="z", columns="region", values=["active_frac", "dws_at_spikes_median"]).round(3).to_string())
    plot(d, s, OUT / "shallow_test.png")
    write_manifest("b2_shallow_test_analyse", OUT, params=C,
                   inputs=[rdir(r) / ztag(zd) / "inv/Output_Files/Vp_model.dat" for r in C["regions"] for zd in C["depths_km"]],
                   outputs=[OUT / f for f in ("spikes.csv", "layers.csv", "by_depth.csv", "shallow_test.png")],
                   counts={"spikes": len(d)})


def plot(d, s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {C["regions"][0]: "#2a78d6", C["regions"][1]: "#eb6834"}
    lab = {C["regions"][0]: "v2 (B1b events)", C["regions"][1]: "v2s (+2,387 shallow events)"}
    ink2, grid = "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True, constrained_layout=True)
    for a, w in zip(ax, ("P", "S")):
        for r in C["regions"]:
            x = d[(d.wave == w) & (d.region == r)]
            jit = 0.25 if r == C["regions"][1] else -0.25
            a.scatter(100 * x.rec.clip(-0.2, 1.2), x.z + jit, s=9, color=col[r], alpha=0.5, lw=0)
            q = s.loc[w].xs(r, level="region")
            a.plot(100 * q.rec_median, q.index, "-o", color=col[r], lw=2, ms=5, label=lab[r])
        a.axvline(100, color=ink2, lw=0.8, ls=":")
        a.axvline(50, color=ink2, lw=0.6, ls="--")
        a.set_xlabel(f"recovered spike peak, % of true ({'Vp' if w == 'P' else 'Vs'}); dots = spikes, line = median")
        a.grid(color=grid, lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_ylabel("spike depth (km)")
    ax[0].invert_yaxis()
    ax[0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Station-centred spikes, Mendocino-Gorda, damp {C['damp']}, smooth {C['smooth']}", fontsize=9)
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"prep": lambda: prep(), "syn": lambda: syn(a[1], a[2]), "inv": lambda: inv(a[1], a[2]),
     "analyse": lambda: analyse()}[a[0]]()
