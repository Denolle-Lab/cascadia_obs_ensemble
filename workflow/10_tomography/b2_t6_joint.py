"""Track B, test T6 (notes/merge_design.md): one joint inversion of two tiles against the stitched model of the
two tile inversions (Lin et al. 2010 did this check for the California statewide model).

  joint    J = run of the joint tile (params tomodd.merge.t6.joint, e.g. blanco_oregon_v2 d200_s20)
  stitched S = the two tile runs blended with the proposed merge weights (merge_design.md M3):
           w_k = t_k u_k / sum_k t_k u_k, t_k = sin^2(pi/2 min(1, d_k / W)) from the edges of tile k's events box
           that lie inside the other box (d_k in km, W = overlap width), 0 outside box k; u_k = resolution window
           (1 on tile k's checkerboard mask, cosine taper over taper_km, floor r outside), per phase.
           dlnV_S = sum_k w_k dlnV_k, each tile's change from its own starting model; where no tile box covers a
           point, dlnV_S = 0.
Compared on the merge lattice (tomodd.merge.grid) inside the union of the two events boxes, depth nodes down to
merge.max_depth_km, as dlnV (joint change from the joint start against the blended change), by zone:
  A only / B only (inside one box, more than W from the other box), near seam A / B (within W of the other box),
  overlap (inside both boxes); and by mask class (resolved by a covering tile / unresolved).
Yardsticks, same zones: tile A at the chosen damping against tile A at the yardstick damping (how much one
damping step moves the model), and tile B on the region weight scale against the dataset scale (D-B045).
Events: the same event in J and in each tile (horizontal and depth difference), by zone.

Outputs: data/tomo_v1/b2_merge/t6/<name>/ zone_stats.csv, events.csv, summary.json, summary.md, maps.png

  python b2_t6_joint.py
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
from b2_fwd_common import mod_velocity  # noqa: E402
from b2_tile_overlap import km, load_run, read_reloc, stats  # noqa: E402

T = load_params()["tomodd"]
M = T["merge"]
C = M["t6"]
RUNS = Path(T["runs"])
RES = REPO / "data" / "tomo_v1" / "b2_res_tests"
OUT = REPO / "data" / "tomo_v1" / "b2_merge" / "t6"
KM_DEG = 111.195


def box(tile):
    r = T["regions"][tile]
    return [*r["events_lat"], *r["events_lon"]]


def inside(b, lat, lon):
    return (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3])


def facing_edge_points(b, other, step=0.01):
    """Points on the edges of box b that lie inside box `other` (the edges that face the other tile)."""
    la0, la1, lo0, lo1 = b
    lats, lons = np.arange(la0, la1 + step / 2, step), np.arange(lo0, lo1 + step / 2, step)
    e = np.r_[np.c_[lats, np.full_like(lats, lo0)], np.c_[lats, np.full_like(lats, lo1)],
              np.c_[np.full_like(lons, la0), lons], np.c_[np.full_like(lons, la1), lons]]
    o = other
    keep = (e[:, 0] > o[0]) & (e[:, 0] < o[1]) & (e[:, 1] > o[2]) & (e[:, 1] < o[3])
    return e[keep]


def dist_km(lat, lon, pts):
    if not len(pts):
        return np.full(len(lat), np.inf)
    d = np.full(len(lat), np.inf)
    for c in range(0, len(pts), 500):
        p = pts[c:c + 500]
        dy = (lat[:, None] - p[None, :, 0]) * KM_DEG
        dx = (lon[:, None] - p[None, :, 1]) * KM_DEG * np.cos(np.radians(lat[:, None]))
        d = np.minimum(d, np.hypot(dx, dy).min(1))
    return d


def overlap_width_km(a, b):
    lat0, lat1, lon0, lon1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    w_lat = (lat1 - lat0) * KM_DEG
    w_lon = (lon1 - lon0) * KM_DEG * np.cos(np.radians(0.5 * (lat0 + lat1)))
    return float(min(w_lat, w_lon))     # the narrow side of the intersection is the seam's width


def box_taper(bk, bo, lat, lon, W):
    t = np.sin(0.5 * np.pi * np.minimum(1.0, dist_km(lat, lon, facing_edge_points(bk, bo)) / W)) ** 2
    return np.where(inside(bk, lat, lon), t, 0.0)


def load_mask(tile, damp):
    f = RES / tile / f"d{damp:g}" / "resolution_mask.npz"
    if not f.exists() and damp == M["mask_default_damp"]:
        f = RES / tile / "resolution_mask.npz"
    return (np.load(f), f) if f.exists() else (None, None)


def res_window(mk, phase, lat, lon, z, taper_km, floor):
    """1 on the resolved nodes (finest cell > 0), cosine taper over taper_km, floor outside; same depth node."""
    if mk is None:
        return np.full(len(lat), floor), np.zeros(len(lat), bool)
    a = mk["finest_p" if phase == "P" else "finest_s"] > 0
    ix = np.abs(mk["lon"][None, :] - lon[:, None]).argmin(1)       # nearest node, as b2_tile_overlap.mask_at
    iy = np.abs(mk["lat"][None, :] - lat[:, None]).argmin(1)
    iz = np.abs(mk["z"][None, :] - z[:, None]).argmin(1)
    res = a[ix, iy, iz]
    LO, LA = np.meshgrid(mk["lon"], mk["lat"], indexing="ij")
    u = np.where(res, 1.0, floor)
    for k in np.unique(iz):
        s = (iz == k) & ~res
        r = a[:, :, k]
        if not r.any() or not s.any():
            continue
        d = dist_km(lat[s], lon[s], np.c_[LA[r], LO[r]])
        u[s] = floor + (1 - floor) * np.cos(0.5 * np.pi * np.minimum(1.0, d / taper_km)) ** 2
    return u, res


def grid(ba, bb, z_nodes):
    g = M["grid"]
    lat0, lat1, lon0, lon1 = min(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), max(ba[3], bb[3])
    lons = np.arange(np.ceil(lon0 / g["lon_step"] - 1e-9), np.floor(lon1 / g["lon_step"] + 1e-9) + 1) * g["lon_step"]
    lats = np.arange(np.ceil(lat0 / g["lat_step"] - 1e-9), np.floor(lat1 / g["lat_step"] + 1e-9) + 1) * g["lat_step"]
    zs = z_nodes[(z_nodes > z_nodes.min()) & (z_nodes <= M["max_depth_km"])]
    LO, LA, Z = np.meshgrid(lons, lats, zs, indexing="ij")
    lon, lat, z = LO.ravel(), LA.ravel(), Z.ravel()
    keep = inside(ba, lat, lon) | inside(bb, lat, lon)
    return dict(lon=lon[keep], lat=lat[keep], z=z[keep], lons=lons, lats=lats, zs=zs)


def zones(ba, bb, lat, lon, W, na, nb):
    ia, ib = inside(ba, lat, lon), inside(bb, lat, lon)
    da = dist_km(lat, lon, facing_edge_points(bb, ba))   # distance to B's edges inside A (seam on A's side)
    db = dist_km(lat, lon, facing_edge_points(ba, bb))
    z = np.full(len(lat), "", object)
    z[ia & ib] = "overlap"
    z[ia & ~ib & (da <= W)] = f"near seam {na}"
    z[ia & ~ib & (da > W)] = f"{na} only"
    z[ib & ~ia & (db <= W)] = f"near seam {nb}"
    z[ib & ~ia & (db > W)] = f"{nb} only"
    return z


def dlnv(run_m0, run_m1, ph, g):
    v0 = mod_velocity(run_m0, ph, g["lon"], g["lat"], g["z"], clamp_surface=False)
    v1 = mod_velocity(run_m1, ph, g["lon"], g["lat"], g["z"], clamp_surface=False)
    return v1 / v0 - 1


def events(tj, rj, tiles, ba, bb, W, na, nb):
    j = read_reloc(rj / "Output_Files" / "tomoDD.reloc_geo")
    s = pd.read_csv(RUNS / tj / "inputs" / "events.csv", usecols=["ID", "lat", "lon", "depth_bsl_km"]).set_index("ID")
    out = []
    for name, (tile, run) in tiles.items():
        t = read_reloc(run / "Output_Files" / "tomoDD.reloc_geo")
        ids = j.index.intersection(t.index).intersection(s.index)
        a, b, s0 = j.loc[ids], t.loc[ids], s.loc[ids]
        out.append(pd.DataFrame({"ID": ids, "tile": name, "lat_start": s0["lat"].values, "lon_start": s0["lon"].values,
                                 "dh_joint_tile_km": km(a["lat"].values, a["lon"].values, b["lat"].values, b["lon"].values),
                                 "dz_joint_tile_km": a["depth"].values - b["depth"].values,
                                 "dh_tile_shift_km": km(b["lat"].values, b["lon"].values, s0["lat"].values, s0["lon"].values),
                                 "dz_tile_shift_km": b["depth"].values - s0["depth_bsl_km"].values}))
    ev = pd.concat(out, ignore_index=True)
    ev["zone"] = zones(ba, bb, ev.lat_start.values, ev.lon_start.values, W, na, nb)
    lost = {name: int(len(s.index.difference(j.index).intersection(read_reloc(run / "Output_Files" / "tomoDD.reloc_geo").index)))
            for name, (tile, run) in tiles.items()}
    return ev, lost


def main():
    (tj, tagj), (ta, taga), (tb, tagb) = C["joint"], C["tile_a"], C["tile_b"]
    na, nb = ta.split("_")[0], tb.split("_")[0]
    name = f"{tj}_{tagj}"
    o = OUT / name
    o.mkdir(parents=True, exist_ok=True)
    rj, j0, j1 = load_run(tj, tagj)
    ra, a0, a1 = load_run(ta, taga)
    rb, b0, b1 = load_run(tb, tagb)
    ba, bb = box(ta), box(tb)
    W = overlap_width_km(ba, bb)
    g = grid(ba, bb, j0["z"])
    zone = zones(ba, bb, g["lat"], g["lon"], W, na, nb)
    ta_w = box_taper(ba, bb, g["lat"], g["lon"], W)
    tb_w = box_taper(bb, ba, g["lat"], g["lon"], W)
    damp = float(taga.split("_")[0][1:])
    mka, fa = load_mask(C["mask_tiles"][0], damp)
    mkb, fb = load_mask(C["mask_tiles"][1], damp)
    rows, F = [], {}
    yard = {}                                        # evaluated inside the first run's events box only
    for key, (t1, g1, t2, g2) in C["yardsticks"].items():
        yard[key] = (box(t1), load_run(t1, g1), load_run(t2, g2))
    for ph in ("P", "S"):
        dj, da, db = dlnv(j0, j1, ph, g), dlnv(a0, a1, ph, g), dlnv(b0, b1, ph, g)
        ua, resa = res_window(mka, ph, g["lat"], g["lon"], g["z"], C["taper_km"], C["floor"])
        ub, resb = res_window(mkb, ph, g["lat"], g["lon"], g["z"], C["taper_km"], C["floor"])
        wa, wb = ta_w * ua, tb_w * ub
        s = wa + wb
        ds = np.where(s > 0, (wa * np.nan_to_num(da) + wb * np.nan_to_num(db)) / np.where(s > 0, s, 1), 0.0)
        cover_res = (resa & (ta_w > 0)) | (resb & (tb_w > 0))
        cls = np.where(cover_res, "resolved", "unresolved")
        F[ph] = dict(dj=dj, ds=ds, da=da, db=db, cls=cls, wa=np.where(s > 0, wa / np.where(s > 0, s, 1), 0))
        ys = {}
        for key, (yb, (_, y0a, y1a), (_, y0b, y1b)) in yard.items():
            inb = inside(yb, g["lat"], g["lon"])
            ys[key] = (np.where(inb, dlnv(y0a, y1a, ph, g), np.nan), np.where(inb, dlnv(y0b, y1b, ph, g), np.nan))
        sel_z = (g["z"] >= 1) & (g["z"] <= 34)
        for zn in [f"{na} only", f"near seam {na}", "overlap", f"near seam {nb}", f"{nb} only"]:
            for c in ("resolved", "unresolved", "all"):
                base = (zone == zn) & sel_z & ((cls == c) if c != "all" else True)
                if base.sum() < 3:
                    continue
                r = dict(phase=ph, zone=zn, cls=c, **{f"joint_vs_stitched_{k}": v for k, v in stats(dj[base], ds[base]).items()})
                for key, (y1, y2) in ys.items():
                    ok = base & np.isfinite(y1) & np.isfinite(y2)
                    if ok.sum() >= 3:
                        st = stats(y1[ok], y2[ok])
                        r[f"{key}_diff_rms"], r[f"{key}_n"] = st.get("diff_rms"), st.get("n")
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(o / "zone_stats.csv", index=False, float_format="%.5g")
    ev, lost = events(tj, rj, {na: (ta, ra), nb: (tb, rb)}, ba, bb, W, na, nb)
    ev.to_csv(o / "events.csv", index=False, float_format="%.5g")
    ev_summ = (ev.groupby(["tile", "zone"])
               .agg(n=("ID", "size"), dh_med=("dh_joint_tile_km", "median"),
                    dh_p90=("dh_joint_tile_km", lambda x: x.quantile(0.9)),
                    dz_med=("dz_joint_tile_km", "median"),
                    absdz_med=("dz_joint_tile_km", lambda x: x.abs().median()),
                    absdz_p90=("dz_joint_tile_km", lambda x: x.abs().quantile(0.9)),
                    tile_shift_h_med=("dh_tile_shift_km", "median"),
                    tile_shift_absz_med=("dz_tile_shift_km", lambda x: x.abs().median())).reset_index())
    ev_summ.to_csv(o / "events_by_zone.csv", index=False, float_format="%.4g")
    summ = dict(joint=[tj, tagj], tile_a=[ta, taga], tile_b=[tb, tagb], overlap_width_km=W,
                masks=[str(fa) if fa else None, str(fb) if fb else None], taper_km=C["taper_km"], floor=C["floor"],
                grid_points=int(len(g["lon"])), events_in_tile_not_in_joint=lost,
                yardsticks={k: list(v) for k, v in C["yardsticks"].items()})
    (o / "summary.json").write_text(json.dumps(summ, indent=1))
    summary_md(o, name, df, ev_summ, summ)
    figure(o, g, F, ba, bb, name)
    write_manifest("b2_t6_joint", o, inputs=[rj / "Output_Files" / "model_geo.npz", ra / "Output_Files" / "model_geo.npz",
                                             rb / "Output_Files" / "model_geo.npz"] + [f for f in (fa, fb) if f],
                   params={"merge": M}, counts={"grid_points": int(len(g["lon"])), "events": int(len(ev))})
    print((o / "summary.md").read_text())


def summary_md(o, name, df, ev, summ):
    ykeys = list(summ["yardsticks"])
    L = [f"# T6 joint against stitched: {name}", "",
         f"Joint {' '.join(summ['joint'])}; stitched = {' '.join(summ['tile_a'])} + {' '.join(summ['tile_b'])} with "
         f"box taper x resolution window (taper {summ['taper_km']} km, floor {summ['floor']}); overlap width "
         f"{summ['overlap_width_km']:.0f} km. dlnV pooled over 1-34 km, in %. Yardsticks: diff rms of "
         + "; ".join(f"{k} = {' '.join(v[0:2])} vs {' '.join(v[2:4])}" for k, v in summ["yardsticks"].items()) + ".",
         f"Masks: {summ['masks']}.", "",
         "| phase | zone | class | n | rms joint | rms stitched | diff mean | diff rms | corr | amp S/J | "
         + " | ".join(f"{k} diff rms" for k in ykeys) + " |",
         "|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(ykeys)]
    for r in df.itertuples():
        g = r._asdict()
        if g.get("joint_vs_stitched_n", 0) < 3 or pd.isna(g.get("joint_vs_stitched_rms_a")):
            continue
        ys = " | ".join("" if pd.isna(g.get(f"{k}_diff_rms")) else f"{100 * g[f'{k}_diff_rms']:.2f}" for k in ykeys)
        L.append(f"| {r.phase} | {r.zone} | {r.cls} | {int(g['joint_vs_stitched_n'])} | "
                 f"{100 * g['joint_vs_stitched_rms_a']:.2f} | {100 * g['joint_vs_stitched_rms_b']:.2f} | "
                 f"{100 * g['joint_vs_stitched_diff_mean']:+.2f} | {100 * g['joint_vs_stitched_diff_rms']:.2f} | "
                 f"{g['joint_vs_stitched_corr']:.2f} | {g['joint_vs_stitched_amp_b_over_a']:.2f} | {ys} |")
    L += ["", "## Same events: joint against each tile", "",
          f"Events in a tile run but not relocated in the joint run: {summ['events_in_tile_not_in_joint']}.", "",
          "| tile | zone | n | horiz med / p90 km | depth med km | abs depth med / p90 km | tile's own shift h / abs z med km |",
          "|---|---|---|---|---|---|---|"]
    for r in ev.itertuples():
        L.append(f"| {r.tile} | {r.zone} | {r.n} | {r.dh_med:.2f} / {r.dh_p90:.2f} | {r.dz_med:+.2f} | "
                 f"{r.absdz_med:.2f} / {r.absdz_p90:.2f} | {r.tile_shift_h_med:.2f} / {r.tile_shift_absz_med:.2f} |")
    (o / "summary.md").write_text("\n".join(L) + "\n")


def figure(o, g, F, ba, bb, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depths = [z for z in (3, 7, 12, 21) if z in g["zs"]]
    f = F["P"]
    fig, ax = plt.subplots(len(depths), 4, figsize=(13, 2.6 * len(depths)), squeeze=False)
    for i, zd in enumerate(depths):
        s = g["z"] == zd
        for k, (fld, title) in enumerate([("dj", "joint"), ("ds", "stitched"), ("diff", "joint - stitched"),
                                          ("wa", "weight of tile A")]):
            a = ax[i, k]
            v = (f["dj"] - f["ds"])[s] if fld == "diff" else f[fld][s]
            v = v if fld == "wa" else 100 * v
            vm = (0, 1) if fld == "wa" else (-5, 5)
            sc = a.scatter(g["lon"][s], g["lat"][s], c=v, s=9, marker="s", cmap="viridis" if fld == "wa" else "RdBu",
                           vmin=vm[0], vmax=vm[1], linewidths=0)
            un = s & (f["cls"] == "unresolved")
            a.scatter(g["lon"][un], g["lat"][un], s=0.6, c="k", alpha=0.25, linewidths=0)
            for b, c in ((ba, "k"), (bb, "0.4")):
                a.plot([b[2], b[3], b[3], b[2], b[2]], [b[0], b[0], b[1], b[1], b[0]], color=c, lw=0.8)
            a.set_aspect(1 / np.cos(np.radians(44.5)))
            a.set_title(f"{title}, Vp, {zd} km", fontsize=8)
            a.tick_params(labelsize=7)
            if i == len(depths) - 1:
                fig.colorbar(sc, ax=a, orientation="horizontal", fraction=0.05, pad=0.15,
                             label="weight" if fld == "wa" else "dlnVp (%)")
    fig.suptitle(f"T6 {name}: dots = no covering tile resolves the node", fontsize=9)
    fig.tight_layout()
    fig.savefig(o / "maps.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
