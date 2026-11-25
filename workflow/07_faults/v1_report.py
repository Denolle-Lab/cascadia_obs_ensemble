"""HTML report of fault surfaces v1 (catalog v5.1) and its comparison with v0.

Reads the outputs of v1_build.py (data/catalog_v5/v5.1_analysis/faults/ and the large files on /wd1) and the v0
product (data/faults/v0/); writes fault_surfaces_v1_report.html (figures embedded) and the PNG figures next to it.

    python workflow/07_faults/v1_report.py
"""
from __future__ import annotations

import base64
import io
import json
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from scipy.stats import norm  # noqa: E402

import v1_common as C  # noqa: E402

O = C.OUT
S = "fault_surfaces_v1"
REG = {"BL": "Blanco", "MT": "Mendocino triple junction", "GO": "Gorda", "PU": "Puget", "WA": "Washington offshore",
       "OR": "Oregon offshore", "EN": "Endeavour-Explorer", "ON": "Onshore and other", "AX": "Axial"}
VIEWS = [("Whole margin", [-131, -119.5, 39, 51]), ("Blanco transform", [-130.6, -126.6, 42.6, 45.3]),
         ("Mendocino and Gorda", [-128.0, -123.2, 39.8, 42.5]), ("Puget and Mount St. Helens", [-124.0, -121.4, 45.9, 49.2])]
FIGS = {}


def png(fig, name):
    fig.savefig(O / f"{S}_{name}.png", dpi=150)
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110)
    plt.close(fig)
    FIGS[name] = base64.b64encode(b.getvalue()).decode()


def img(name, cap):
    return f'<figure><img src="data:image/png;base64,{FIGS[name]}" alt="{escape(cap)}"><figcaption>{cap}</figcaption></figure>'


def table(df, cols=None, names=None, fmt=None):
    df = df if cols is None else df[cols]
    names = names or list(df.columns)
    h = "".join(f"<th>{escape(str(n))}</th>" for n in names)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for k, v in r.items():
            if isinstance(v, float):
                v = "" if np.isnan(v) else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")
            cells.append(f"<td>{escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def outlines_v0():
    v = pd.read_csv(C.V0DIR / "fault_surfaces_v0_vertices.csv", usecols=["surface_id", "lon", "lat", "depth_km"])
    out = {}
    for sid, g in v.groupby("surface_id"):
        xy = g[["lon", "lat"]].to_numpy()
        try:
            out[sid] = (xy[ConvexHull(xy).vertices], g.depth_km.mean())
        except Exception:
            out[sid] = (xy[[0, -1]], g.depth_km.mean())
    return out


def outlines_v1():
    g = json.load(open(O / f"{S}_rectangles.geojson"))
    out = {}
    for f in g["features"]:
        r = np.array(f["geometry"]["coordinates"][0])
        out[f["properties"]["surface_id"]] = (r[:, :2], -r[:4, 2].mean() / 1000)
    return out


def fig_maps(cat, asg, o0, o1):
    fig, axs = plt.subplots(2, 4, figsize=(20, 11), constrained_layout=True)
    loc = cat[cat.located & ~cat.nonquake]
    for col, (title, b) in enumerate(VIEWS):
        sel = loc.longitude.between(b[0], b[1]) & loc.latitude.between(b[2], b[3])
        for row, (o, lab) in enumerate([(o0, "v0 (catalog v4)"), (o1, "v1 (catalog v5.1)")]):
            ax = axs[row, col]
            ax.scatter(loc.longitude[sel], loc.latitude[sel], s=0.15, c="0.82", lw=0, rasterized=True)
            polys = [p for p, _ in o.values()]
            vals = [d for _, d in o.values()]
            pc = PolyCollection(polys, array=np.array(vals), cmap="viridis_r", clim=(0, 40), edgecolors="k",
                                linewidths=0.25, alpha=0.75)
            ax.add_collection(pc)
            if col > 0:
                for sid, (poly, _) in o.items():
                    cx, cy = poly[:-1].mean(0) if len(poly) > 2 else poly.mean(0)
                    if b[0] < cx < b[1] and b[2] < cy < b[3]:
                        ax.text(cx, cy, sid.replace("v1_", ""), fontsize=4.5, ha="center", va="center", color="k")
            ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3])
            ax.set_aspect(1 / np.cos(np.radians((b[2] + b[3]) / 2)))
            ax.set_title(f"{title}: {lab}, {len(o)} surfaces", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.colorbar(pc, ax=axs, label="Mean depth of the surface (km; v1 below sea level, v0 catalog v4 datum)", shrink=0.5)
    png(fig, "map_v0_v1")


def fig_assign(cat, asg):
    a = asg.merge(cat[["event_id", "longitude", "latitude", "z"]], on="event_id")
    fig, axs = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
    rng = np.random.default_rng(3)
    ids = sorted(a.surface_id.dropna().unique())
    ids = [i for i in ids if i]
    col = {i: plt.cm.tab20(rng.integers(0, 20)) for i in ids}
    for ax, (title, b) in zip(axs, VIEWS):
        s = a[a.longitude.between(b[0], b[1]) & a.latitude.between(b[2], b[3])]
        nf = s[s.assignment.str.startswith("not fitted")]
        ax.scatter(nf.longitude, nf.latitude, s=0.2, c="#e8d9b5", lw=0, rasterized=True)
        n = s[s.assignment == "none"]
        ax.scatter(n.longitude, n.latitude, s=0.2, c="0.75", lw=0, rasterized=True)
        for cls, size, alpha in (("assigned", 0.3, 0.35), ("fit member", 0.35, 1.0)):
            m = s[s.assignment == cls]
            ax.scatter(m.longitude, m.latitude, s=size, c=[col[x] for x in m.surface_id], lw=0, alpha=alpha, rasterized=True)
        q = s[s.assignment == "not an earthquake"]
        ax.scatter(q.longitude, q.latitude, s=2, c="r", marker="x", lw=0.4, rasterized=True)
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect(1 / np.cos(np.radians((b[2] + b[3]) / 2)))
        ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle("Located events of catalog v5.1 by surface (colour), assigned only (faint colour), none (grey), "
                 "Axial, not fitted (beige), non-earthquakes (red x)", fontsize=10)
    png(fig, "assignment_map")


def fig_compare(pairs):
    m = pairs[pairs.change == "matched"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    axs[0].hist(m.strike_change, bins=np.arange(0, 181, 10), color="#3d6fa8")
    axs[0].set_xlabel("Strike change (deg; modulo 180 when both dips > 75)")
    axs[1].hist(m.dip_change, bins=np.arange(-50, 51, 5), color="#3d6fa8")
    axs[1].set_xlabel("Dip change v1 - v0 (deg)")
    axs[2].scatter(m.v0_members_median_shift_z_km, m.centroid_shift_z_km, s=12, c="#3d6fa8")
    lim = [-5, 20]
    axs[2].plot(lim, lim, "k:", lw=0.8)
    axs[2].set_xlabel("Median v4 -> v5.1 depth shift of the v0 members (km)")
    axs[2].set_ylabel("Centroid depth change v1 - v0 (km)")
    axs[3].hist(m.centroid_shift_h_km, bins=np.arange(0, 31, 1), color="#3d6fa8")
    axs[3].set_xlabel("Horizontal centroid shift (km)")
    for ax in axs[[0, 1, 3]]:
        ax.set_ylabel("Matched surfaces")
    fig.suptitle(f"Matched surfaces (one v0 to one v1, n = {len(m)})", fontsize=10)
    png(fig, "matched_changes")


def fig_calibration(res, summ):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    r = np.abs(res.resid_km) / res.sigma_n_km
    bins = np.linspace(0, 3, 61)
    for t, c_ in zip("ABC", ("#1b7837", "#3d6fa8", "#b2182b")):
        axs[0].hist(r[res.tier == t], bins=bins, histtype="step", density=True, color=c_, label=f"tier {t}")
    x = np.linspace(0, 3, 100)
    axs[0].plot(x, 2 * np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k:", label="|N(0,1)|")
    axs[0].legend(fontsize=8); axs[0].set_xlabel("|normal residual| / event error along the normal")
    axs[1].hist(summ.scatter_to_sigma_n, bins=np.linspace(0, 2, 41), color="#3d6fa8")
    axs[1].set_xlabel("RMS normal residual / RMS event error along the normal, per surface")
    axs[1].set_ylabel("Surfaces")
    axs[2].scatter(summ.n_events, summ.sigma_total_median_km, s=10, c="#3d6fa8")
    axs[2].set_xscale("log"); axs[2].set_xlabel("Events on the surface"); axs[2].set_ylabel("Median sigma_total (km)")
    png(fig, "calibration")


def fig_regions(reg0, reg1):
    m = reg1.merge(reg0, on="region", how="outer", suffixes=("", "_v0"))
    m = m[m.region != "AX"]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    x = np.arange(len(m))
    ax.bar(x - 0.27, m.fraction_v0, 0.27, label="v0: v4 fit set on a fault", color="0.6")
    ax.bar(x, m.frac_fit_set_on_a_fault, 0.27, label="v1: v5.1 fit set, fit members", color="#3d6fa8")
    ax.bar(x + 0.27, m.frac_earthquakes_on_a_fault, 0.27, label="v1: all located earthquakes, members + assigned", color="#9ecae1")
    ax.set_xticks(x); ax.set_xticklabels([REG.get(r, r) for r in m.region], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Share on a fault"); ax.legend(fontsize=8); ax.set_ylim(0, 1)
    png(fig, "regions")


def main():
    summ = pd.read_csv(O / f"{S}_summary.csv")
    pairs = pd.read_csv(O / f"{S}_v0_pairs.csv")
    st0 = pd.read_csv(O / f"{S}_v0_status.csv")
    cmp1 = pd.read_csv(O / f"{S}_v0_comparison.csv")
    rem = pd.read_csv(O / f"{S}_removed.csv") if (O / f"{S}_removed.csv").stat().st_size > 2 else pd.DataFrame()
    sep = pd.read_csv(O / f"{S}_strand_separation.csv") if (O / f"{S}_strand_separation.csv").stat().st_size > 2 else pd.DataFrame()
    mlog = pd.read_csv(O / f"{S}_strand_merges.csv") if (O / f"{S}_strand_merges.csv").stat().st_size > 2 else pd.DataFrame()
    sel = pd.read_csv(O / f"{S}_selection.csv")
    reg1 = pd.read_csv(O / f"{S}_assignment_by_region.csv")
    reg0 = pd.read_csv(C.V0DIR / "fault_surfaces_v0_assignment.csv").rename(columns={"events": "events_v0", "assigned": "assigned_v0", "fraction": "fraction_v0"})
    asg = pd.read_csv(O / f"{S}_assignment.csv.gz", keep_default_na=False, na_values=[""])
    asg["surface_id"] = asg.surface_id.fillna("")
    res = pd.read_csv(C.BIG / "work" / "member_residuals.csv.gz")
    cat, _ = C.load()
    o0, o1 = outlines_v0(), outlines_v1()
    fig_maps(cat, asg, o0, o1)
    fig_assign(cat, asg)
    fig_compare(pairs)
    fig_calibration(res, summ)
    fig_regions(reg0, reg1)
    ctx = checks(cat, summ, st0, cmp1, pairs, mlog, sep, res)
    ctx.update(summ=summ, pairs=pairs, st0=st0, cmp1=cmp1, rem=rem, sep=sep, mlog=mlog, sel=sel, reg0=reg0, reg1=reg1,
               asg=asg, res=res, cat=cat)
    rem_t_ids = rem.removed_id[rem.reason.str.startswith("non-earthquake")] if len(rem) else []
    ctx["n_rem_timing"] = int(asg.removed_surface.fillna("").isin(rem_t_ids).sum())
    ctx["n_blast"] = int(asg.assignment.str.startswith("non-earthquake timing").sum())
    ctx["n_rem_other"] = ctx["n_rem_timing"] - ctx["n_blast"]
    html = build_html(ctx)
    (O / f"{S}_report.html").write_text(html)
    print("written", O / f"{S}_report.html", f"{len(html) / 1e6:.1f} MB")


def checks(cat, summ, st0, cmp1, pairs, mlog, sep, res):
    """Numbers quoted in the findings, computed here so that a rerun updates them."""
    x = {}
    try:
        x["run"] = json.load(open(O / f"{S}_run.json"))
    except Exception:
        x["run"] = {}
    loc = cat[cat.located]
    m0 = pd.read_csv(C.V0DIR / "fault_surfaces_v0_members.csv")
    ci = cat.set_index("event_id")
    # first v1 run without the dip rule in the reassignment
    try:
        x["nodip"] = json.load(open(C.WORK / "nodipfix" / "detect_main.json"))["msg"]
    except Exception:
        x["nodip"] = "n/a"
    # Mount St. Helens: events deeper than 30 km, weekly timing
    msh = loc[loc.latitude.between(46.05, 46.30) & loc.longitude.between(-122.35, -122.10) & (loc.loc_source == "nll3d")]
    d = msh[msh.z > 30]
    t = d.t
    x["msh"] = dict(n=len(msh), deep=len(d), tiers=d.tier.value_counts().to_dict(), sigz=float(d.sz.median()),
                    zmin=float(d.z.min()), zmax=float(d.z.max()),
                    wed=int(((t.dt.dayofweek == 2) & (t.dt.hour == 19) & t.dt.minute.between(14, 20)).sum()),
                    tue=int(((t.dt.dayofweek == 1) & (t.dt.hour == 19) & t.dt.minute.between(14, 20)).sum()))
    # Cascadia Depression clusters of v0: map elongation in v4 and v5.1

    def az(lon, lat):
        kx = 111.195 * np.cos(np.radians(np.mean(lat)))
        X = np.c_[(lon - lon.mean()) * kx, (lat - lat.mean()) * 111.195]
        w, V = np.linalg.eigh(np.cov(X.T))
        return float(np.degrees(np.arctan2(V[0, 1], V[1, 1])) % 180), float(np.sqrt(w[1] / w[0]))
    cas = []
    for sid in ("BL005", "BL008", "BL012"):
        e = ci.loc[ci.index.intersection(m0[m0.surface_id == sid].event_id)]
        e = e[e.located]
        a4, a5 = az(e.v4_lon.values, e.v4_lat.values), az(e.longitude.values, e.latitude.values)
        top = st0.set_index("v0_surface").top_v1_surfaces.get(sid, "")
        cas.append(dict(v0_surface=sid, events=len(e), v4_azimuth=round(a4[0]), v4_elongation=round(a4[1], 1),
                        v51_azimuth=round(a5[0]), v51_elongation=round(a5[1], 1), median_shift_h_km=round(e.shift_h_km.median(), 1),
                        v1_surfaces=top))
    x["cas"] = pd.DataFrame(cas)
    # Blanco Ridge (v0 BL003a): depth change of its events
    e = ci.loc[ci.index.intersection(m0[m0.surface_id == "BL003a"].event_id)]
    x["br"] = dict(n=int(e.located.sum()), v4=float(e.v4_depth.median()), v51=float(e.z.median()))
    # Brookings 2012 swarm: v1 surface holding most of v0 GO020's events
    g = pairs[pairs.v0_surface == "GO020"]
    mem = pd.read_csv(O / f"{S}_members.csv")
    cand = summ[(summ.region == "GO") & (summ.lon > -125.0) & (summ.lat.between(42.0, 42.4))]
    if len(cand):
        sid = cand.sort_values("n_events").iloc[-1].surface_id
        e = ci.loc[mem[mem.surface_id == sid].event_id]
        x["brook"] = dict(sid=sid, n=len(e), z=float(e.z.median()), zmin=float(e.z.min()), zmax=float(e.z.max()),
                          slab=float(np.nanmedian(C.v0.slab()(e.longitude.values, e.latitude.values))),
                          v4=float(e.v4_depth.median()), sigz=float(e.sz.median()), dmin=float(e.dist_min.median()),
                          y2012=int((e.year == 2012).sum()), ok=float((e.depth_quality == "ok").mean()))
    # tier calibration. Members were selected within the detection tolerance of their plane, so their residuals are a
    # truncated sample: expected median |r|/sigma_n under N(0, sigma_n) truncated at +-tolerance, member by member
    r = np.abs(res.resid_km) / res.sigma_n_km
    t = (res.surface_id.map(summ.set_index("surface_id").tolerance_km) / res.sigma_n_km).to_numpy()
    rng = np.random.default_rng(3)
    E = np.array([norm.ppf((1 + rng.uniform(0, 1, len(t)) * (2 * norm.cdf(t) - 1)) / 2) for _ in range(20)])
    exp_tr = {tier: float(np.mean(np.median(E[:, (res.tier == tier).to_numpy()], axis=1))) for tier in "ABC"}
    cal = res.assign(r=r).groupby("tier").agg(members=("r", "size"), median_abs_resid_over_sigma=("r", "median"),
                                             median_sigma_n_km=("sigma_n_km", "median"),
                                             rms_resid_km=("resid_km", lambda v: float(np.sqrt(np.mean(v ** 2))))).reset_index()
    cal.insert(3, "expected_untruncated", 0.67)
    cal.insert(4, "expected_with_truncation", cal.tier.map(exp_tr))
    x["cal"] = cal.round(2)
    # detection-level spread: builder's four reruns and the verifier's three reruns (WORK/verifier_reruns)
    ns = summ[~summ.strand]
    rec = {}
    for name in ("sub80", "perturbed", "tierAB"):
        col = ns[f"recovery_J_{name}"]
        rec[name] = dict(n=int(col.notna().sum()), rec=int((col >= 0.5).sum()),
                         stable=f"{int((col[ns.stable] >= 0.5).sum())}/{int(col[ns.stable].notna().sum())}",
                         median_J=float(col.median()))
    both = (ns.recovery_J_sub80 >= 0.5) & (ns.recovery_J_perturbed >= 0.5)
    allr = both & (ns.recovery_J_tierAB >= 0.5)
    # strike/dip ranges over the builder's four reruns exist only for non-strand surfaces matched in at least one rerun
    # (strands are not rerun); the counts quoted with the medians are those of that sample
    sw, dw = summ.strike_settings_range_width, summ.dip_settings_range_width
    assert (sw.notna() == dw.notna()).all(), "strike and dip range samples differ"
    has = sw.notna()
    uns, stb = has & ~summ.stable, has & summ.stable
    x["det"] = dict(rec=rec, both=int(both.sum()), both_ids=list(ns.surface_id[both]), all3=int(allr.sum()),
                    all3_ids=list(ns.surface_id[allr]), n_ns=len(ns),
                    sw_unstable=float(sw[uns].median()), sw_stable=float(sw[stb].median()),
                    dw_unstable=float(dw[uns].median()), dw_stable=float(dw[stb].median()),
                    n_unstable=int(uns.sum()), n_stable=int(stb.sum()), n_sw=int(has.sum()),
                    n_strand=int(summ.strand.sum()),
                    nomatch_ids=list(ns.surface_id[ns.strike_settings_range_width.isna()]))
    x["shift_flag"] = C.SHIFT_FLAG
    # cost of the dip rule: v0 surfaces flatter than 30 deg and what became of them
    s0 = pd.read_csv(C.V0DIR / "fault_surfaces_v0_summary.csv")
    x["flat"] = st0[st0.v0_surface.isin(s0.surface_id[s0.dip < 30])]
    return x


def build_html(x):
    from v1_report_text import text
    return text(x, FIGS, img, table, REG)


if __name__ == "__main__":
    main()
