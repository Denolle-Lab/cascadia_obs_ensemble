"""Track B: tiles tested against each other where they overlap (notes/merge_design.md, tests T1 and T2).

For tiles A and B and one real-data run of each (tags such as d300_s20):
  * common grid: the merge lattice (params tomodd.merge.grid) inside the intersection of the two events boxes,
    depths = interior node depths down to merge.max_depth_km;
  * each tile's starting model (run/MOD) and final model (Output_Files/model_geo.npz) evaluated there with
    tomoDD's trilinear lookup on that tile's geographic nodes. The D-B028 back-transform is affine per axis, so
    trilinear in geographic coordinates is tomoDD's own interpolation;
  * dlnV = V / V_start - 1 for each tile, compared depth by depth in three classes of the tiles' checkerboard
    masks (b2_res_tests/<tile>[/d<D>]/resolution_mask.npz, finest cell > 0, nearest node): resolved in both,
    in one, in neither. A tile without a mask for that damping puts every point in class "no mask";
  * T1 statistics: start-model difference (should be ~0), rms of each dlnV, median and robust spread
    (1.4826 MAD) of dlnV_A - dlnV_B, correlation, amplitude ratio B/A (least squares through 0), Vp/Vs
    difference;
  * T2: events relocated in both runs (same catalog ID in both tomoDD.reloc_geo): horizontal and depth
    difference of the two final locations, and each run's shift from the common start (inputs/events.csv).
Outputs: data/tomo_v1/b2_merge/overlap/<A>_<tagA>__<B>_<tagB>/ (depth_stats.csv, events.csv, summary.json,
overlap.png) and data/tomo_v1/b2_merge/overlap/summary.md (all pairs).

  python b2_tile_overlap.py                       # every pair and tag combination in tomodd.merge.pairs
  python b2_tile_overlap.py A tagA B tagB         # one combination
Run from the repo root with LD_LIBRARY_PATH=.pixi/envs/default/lib.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b2_fwd_common import read_mod, mod_velocity  # noqa: E402

T = load_params()["tomodd"]
M = T["merge"]
RUNS = Path(T["runs"])
RES = REPO / "data" / "tomo_v1" / "b2_res_tests"
OUT = REPO / "data" / "tomo_v1" / "b2_merge" / "overlap"
CLASSES = ("both", "one", "neither", "no mask")


def damp_of(tag):
    return float(tag.split("_")[0][1:])


def load_run(tile, tag):
    run = RUNS / tile / tag
    geo = np.load(run / "Output_Files" / "model_geo.npz")
    start = read_mod(run / "MOD")
    lon, lat, z = geo["lon"], geo["lat"], geo["z"]
    m0 = dict(lon=lon, lat=lat, z=z, vp=start["vp"], vs=start["vs"])
    m1 = dict(lon=lon, lat=lat, z=z, vp=geo["vp"], vs=geo["vs"])
    return run, m0, m1


def load_mask(tile, tag):
    d = damp_of(tag)
    f = RES / tile / f"d{d:g}" / "resolution_mask.npz"
    if not f.exists() and d == M["mask_default_damp"]:
        f = RES / tile / "resolution_mask.npz"
    if not f.exists():
        return None, None
    return np.load(f), f


def mask_at(mk, phase, lon, lat, z):
    """Nearest-node value of the resolved mask (True where the finest resolved cell > 0)."""
    a = mk["finest_p" if phase == "P" else "finest_s"]
    ix = np.abs(mk["lon"][None, :] - lon[:, None]).argmin(1)
    iy = np.abs(mk["lat"][None, :] - lat[:, None]).argmin(1)
    iz = np.abs(mk["z"][None, :] - z[:, None]).argmin(1)
    return a[ix, iy, iz] > 0


def box(tile):
    r = T["regions"][tile]
    return r["events_lat"], r["events_lon"]


def common_grid(ta, tb, z_nodes):
    (la, oa), (lb, ob) = box(ta), box(tb)
    lat0, lat1 = max(la[0], lb[0]), min(la[1], lb[1])
    lon0, lon1 = max(oa[0], ob[0]), min(oa[1], ob[1])
    if lat0 >= lat1 or lon0 >= lon1:
        return None
    g = M["grid"]
    lons = np.arange(np.ceil(lon0 / g["lon_step"] - 1e-9), np.floor(lon1 / g["lon_step"] + 1e-9) + 1) * g["lon_step"]
    lats = np.arange(np.ceil(lat0 / g["lat_step"] - 1e-9), np.floor(lat1 / g["lat_step"] + 1e-9) + 1) * g["lat_step"]
    zs = z_nodes[(z_nodes > z_nodes.min()) & (z_nodes <= M["max_depth_km"])]
    LO, LA, Z = np.meshgrid(lons, lats, zs, indexing="ij")
    return dict(lon=LO.ravel(), lat=LA.ravel(), z=Z.ravel(), box=[lat0, lat1, lon0, lon1],
                n=(len(lons), len(lats), len(zs)))


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def stats(a, b):
    if len(a) < 3:
        return dict(n=len(a))
    d = a - b
    return dict(n=len(a), rms_a=np.sqrt(np.mean(a ** 2)), rms_b=np.sqrt(np.mean(b ** 2)),
                diff_mean=np.mean(d), diff_med=np.median(d), diff_rstd=robust_std(d), diff_rms=np.sqrt(np.mean(d ** 2)),
                corr=np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan,
                amp_b_over_a=np.sum(a * b) / np.sum(a * a) if np.sum(a * a) > 0 else np.nan)


def read_reloc(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return pd.DataFrame({"ID": [int(r[0]) for r in rows], "lat": [float(r[1]) for r in rows],
                         "lon": [float(r[2]) for r in rows], "depth": [float(r[3]) for r in rows]}).set_index("ID")


def km(lat1, lon1, lat2, lon2):
    y = (lat1 - lat2) * 111.195
    x = (lon1 - lon2) * 111.195 * np.cos(np.radians(0.5 * (lat1 + lat2)))
    return np.hypot(x, y)


def events_compare(ta, ra, tb, rb, bx):
    a = read_reloc(ra / "Output_Files" / "tomoDD.reloc_geo")
    b = read_reloc(rb / "Output_Files" / "tomoDD.reloc_geo")
    s = pd.read_csv(RUNS / ta / "inputs" / "events.csv", usecols=["ID", "lat", "lon", "depth_bsl_km"]).set_index("ID")
    ids = a.index.intersection(b.index).intersection(s.index)
    if not len(ids):
        return pd.DataFrame(), {"n_events": 0}
    a, b, s = a.loc[ids], b.loc[ids], s.loc[ids]
    df = pd.DataFrame({"ID": ids, "lat_start": s["lat"].values, "lon_start": s["lon"].values,
                       "z_start": s["depth_bsl_km"].values,
                       "dh_ab_km": km(a["lat"].values, a["lon"].values, b["lat"].values, b["lon"].values),
                       "dz_ab_km": a["depth"].values - b["depth"].values,
                       "dh_a_km": km(a["lat"].values, a["lon"].values, s["lat"].values, s["lon"].values),
                       "dz_a_km": a["depth"].values - s["depth_bsl_km"].values,
                       "dh_b_km": km(b["lat"].values, b["lon"].values, s["lat"].values, s["lon"].values),
                       "dz_b_km": b["depth"].values - s["depth_bsl_km"].values})
    inside = ((df.lat_start >= bx[0]) & (df.lat_start <= bx[1]) & (df.lon_start >= bx[2]) & (df.lon_start <= bx[3]))
    df["in_overlap_box"] = inside.values
    q = df[df.in_overlap_box]
    summ = {"n_events": int(len(df)), "n_events_overlap_box": int(len(q))}
    for k in ("dh_ab_km", "dh_a_km", "dh_b_km"):
        summ[f"{k}_med"] = float(q[k].median()) if len(q) else np.nan
        summ[f"{k}_p90"] = float(q[k].quantile(0.9)) if len(q) else np.nan
    for k in ("dz_ab_km", "dz_a_km", "dz_b_km"):
        summ[f"{k}_med"] = float(q[k].median()) if len(q) else np.nan
        summ[f"abs_{k}_med"] = float(q[k].abs().median()) if len(q) else np.nan
        summ[f"abs_{k}_p90"] = float(q[k].abs().quantile(0.9)) if len(q) else np.nan
    return df, summ


def compare(ta, tag_a, tb, tag_b, plot=True):
    ra, a0, a1 = load_run(ta, tag_a)
    rb, b0, b1 = load_run(tb, tag_b)
    g = common_grid(ta, tb, a0["z"])
    name = f"{ta}_{tag_a}__{tb}_{tag_b}"
    if g is None:
        print(f"{name}: events boxes do not overlap")
        return None
    o = OUT / name
    o.mkdir(parents=True, exist_ok=True)
    mka, fa = load_mask(ta, tag_a)
    mkb, fb = load_mask(tb, tag_b)
    rows, fields = [], {}
    for ph in ("P", "S"):
        va0 = mod_velocity(a0, ph, g["lon"], g["lat"], g["z"], clamp_surface=False)
        va1 = mod_velocity(a1, ph, g["lon"], g["lat"], g["z"], clamp_surface=False)
        vb0 = mod_velocity(b0, ph, g["lon"], g["lat"], g["z"], clamp_surface=False)
        vb1 = mod_velocity(b1, ph, g["lon"], g["lat"], g["z"], clamp_surface=False)
        da, db = va1 / va0 - 1, vb1 / vb0 - 1
        start_diff = vb0 / va0 - 1
        if mka is not None and mkb is not None:
            res_a, res_b = mask_at(mka, ph, g["lon"], g["lat"], g["z"]), mask_at(mkb, ph, g["lon"], g["lat"], g["z"])
            cls = np.where(res_a & res_b, "both", np.where(res_a | res_b, "one", "neither"))
        else:
            cls = np.full(len(da), "no mask")
        fields[ph] = dict(da=da, db=db, cls=cls, va1=va1, vb1=vb1)
        for z in np.unique(g["z"]):
            for c in CLASSES:
                sel = (g["z"] == z) & (cls == c)
                if not sel.any():
                    continue
                rows.append(dict(phase=ph, z=z, cls=c, start_absmax=float(np.abs(start_diff[sel]).max()),
                                 start_absmed=float(np.median(np.abs(start_diff[sel]))), **stats(da[sel], db[sel])))
        for c in CLASSES:
            sel = (cls == c) & (g["z"] >= 1) & (g["z"] <= 34)
            if sel.any():
                rows.append(dict(phase=ph, z="1-34", cls=c, start_absmax=float(np.abs(start_diff[sel]).max()),
                                 start_absmed=float(np.median(np.abs(start_diff[sel]))), **stats(da[sel], db[sel])))
    # Vp/Vs where both phases are resolved in both tiles
    rpa = fields["P"]["va1"] / fields["S"]["va1"]
    rpb = fields["P"]["vb1"] / fields["S"]["vb1"]
    both_ps = (fields["P"]["cls"] == "both") & (fields["S"]["cls"] == "both")
    vpvs = {"n_both_PS": int(both_ps.sum())}
    if both_ps.sum() >= 3:
        d = rpa[both_ps] - rpb[both_ps]
        vpvs.update(vpvs_diff_med=float(np.median(d)), vpvs_diff_rstd=float(robust_std(d)))
    df = pd.DataFrame(rows)
    df.to_csv(o / "depth_stats.csv", index=False, float_format="%.5g")
    ev, ev_summ = events_compare(ta, ra, tb, rb, g["box"])
    if len(ev):
        ev.to_csv(o / "events.csv", index=False, float_format="%.5g")
    tot = df[df.z == "1-34"].drop(columns="z").set_index(["phase", "cls"])
    summ = dict(tile_a=ta, tag_a=tag_a, tile_b=tb, tag_b=tag_b, damp_a=damp_of(tag_a), damp_b=damp_of(tag_b),
                overlap_box=g["box"], grid_n=g["n"], mask_a=str(fa) if fa else None, mask_b=str(fb) if fb else None,
                pooled_1_34km={f"{p}_{c}": {k: (None if pd.isna(v) else float(v)) for k, v in r.items()}
                               for (p, c), r in tot.iterrows()},
                vpvs=vpvs, events=ev_summ)
    (o / "summary.json").write_text(json.dumps(summ, indent=1))
    if plot:
        figure(o, name, g, fields, ev)
    write_manifest("b2_tile_overlap", o,
                   inputs=[ra / "Output_Files" / "model_geo.npz", rb / "Output_Files" / "model_geo.npz",
                           ra / "MOD", rb / "MOD"] + [f for f in (fa, fb) if f],
                   params={"merge": M}, counts={"grid_points": int(len(g["lon"]))})
    print(f"{name}: grid {g['n']}, events {ev_summ.get('n_events_overlap_box', 0)} in the overlap box")
    return summ


def figure(o, name, g, fields, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    col = {"both": "#1b6ca8", "one": "#e08a1e", "neither": "#9a9a9a", "no mask": "#9a9a9a"}
    f = fields["P"]
    sel = (g["z"] >= 1) & (g["z"] <= 34)
    for c in ("neither", "no mask", "one", "both"):
        s = sel & (f["cls"] == c)
        if s.any():
            ax[0].scatter(100 * f["da"][s], 100 * f["db"][s], s=3, color=col[c], label=f"{c} ({s.sum()})", alpha=0.6)
    lim = np.nanpercentile(np.abs(100 * np.r_[f["da"][sel], f["db"][sel]]), 99.5) if sel.any() else 5
    ax[0].plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
    ax[0].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="dlnVp tile A (%)", ylabel="dlnVp tile B (%)",
              title="Vp change from start, 1-34 km")
    ax[0].legend(fontsize=7, markerscale=3, loc="upper left")
    zs = np.unique(g["z"])
    for c in ("both", "one", "neither", "no mask"):
        med, sp, zz = [], [], []
        for z in zs:
            s = (g["z"] == z) & (f["cls"] == c)
            if s.sum() >= 3:
                d = 100 * (f["da"][s] - f["db"][s])
                med.append(np.median(d)); sp.append(robust_std(d)); zz.append(z)
        if zz:
            ax[1].errorbar(med, zz, xerr=sp, color=col[c], marker="o", ms=3, lw=1, capsize=2, label=c)
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="dlnVp A - B (%), median +- robust std", ylabel="depth (km)", title="Disagreement by depth")
    ax[1].legend(fontsize=7)
    if len(ev):
        q = ev[ev.in_overlap_box]
        bins = np.linspace(0, max(1.0, np.nanpercentile(q.dh_ab_km, 98) if len(q) else 1), 30)
        ax[2].hist(q.dh_ab_km, bins=bins, color="#1b6ca8", alpha=0.7, label="horizontal A-B")
        ax[2].hist(q.dz_ab_km.abs(), bins=bins, color="#e08a1e", alpha=0.6, label="|depth A-B|")
        ax[2].set(xlabel="km", ylabel="events", title=f"Same events, two tiles (n={len(q)})")
        ax[2].legend(fontsize=7)
    fig.suptitle(name.replace("__", "  vs  "), fontsize=9)
    fig.tight_layout()
    fig.savefig(o / "overlap.png", dpi=150)
    plt.close(fig)


def summary_md(summs):
    lines = ["# Tiles compared in their overlaps (b2_tile_overlap.py; notes/merge_design.md T1, T2)", "",
             "Pooled over 1-34 km. dlnV = change from each tile's own starting model. 'both' = resolved in both "
             "tiles' checkerboard masks at their damping (smoothing 20); 'one' = in one only; 'no mask' = a mask is "
             "missing for that damping. diff = dlnV_A - dlnV_B (mean and rms, %); amp = B/A amplitude ratio (least "
             "squares through 0); corr = Pearson; start = |V0_B / V0_A - 1| of the two starting models (median / max, %).",
             "", "| A | B | class | n | rms A % | rms B % | diff mean % | diff rms % | corr | amp B/A | start med / max % |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summs:
        for k, r in s["pooled_1_34km"].items():
            if not k.startswith("P_") or r.get("n", 0) < 3:
                continue
            lines.append(f"| {s['tile_a']} {s['tag_a']} | {s['tile_b']} {s['tag_b']} | {k[2:]} | {int(r['n'])} | "
                         f"{100 * r['rms_a']:.2f} | {100 * r['rms_b']:.2f} | {100 * r['diff_mean']:+.2f} | "
                         f"{100 * r['diff_rms']:.2f} | {r['corr']:.2f} | {r['amp_b_over_a']:.2f} | "
                         f"{100 * r['start_absmed']:.2f} / {100 * r['start_absmax']:.2f} |")
    lines += ["", "## Same events relocated in two tiles (events inside the overlap box)", "",
              "| A | B | n | horiz A-B med / p90 km | depth A-B med km | abs depth A-B med / p90 km | "
              "horiz shift A / B med km | abs depth shift A / B med km |", "|---|---|---|---|---|---|---|---|"]
    for s in summs:
        e = s["events"]
        if not e.get("n_events_overlap_box"):
            continue
        lines.append(f"| {s['tile_a']} {s['tag_a']} | {s['tile_b']} {s['tag_b']} | {e['n_events_overlap_box']} | "
                     f"{e['dh_ab_km_med']:.2f} / {e['dh_ab_km_p90']:.2f} | {e['dz_ab_km_med']:+.2f} | "
                     f"{e['abs_dz_ab_km_med']:.2f} / {e['abs_dz_ab_km_p90']:.2f} | "
                     f"{e['dh_a_km_med']:.2f} / {e['dh_b_km_med']:.2f} | "
                     f"{e['abs_dz_a_km_med']:.2f} / {e['abs_dz_b_km_med']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) == 5:
        compare(*sys.argv[1:5])
        return
    summs = []
    for ta, tb, tags in M["pairs"]:
        for tag_a, tag_b in tags:
            if not ((RUNS / ta / tag_a / "Output_Files" / "model_geo.npz").exists()
                    and (RUNS / tb / tag_b / "Output_Files" / "model_geo.npz").exists()):
                print(f"skip {ta} {tag_a} / {tb} {tag_b}: run missing")
                continue
            s = compare(ta, tag_a, tb, tag_b)
            if s:
                summs.append(s)
    summary_md(summs)


if __name__ == "__main__":
    main()
