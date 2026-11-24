"""Track B, step B4: catalog from the NLLoc relocation in a registered model, depths back from the Earth-flattened
frame, and a comparison with v4-NLL3D (Track A iter1).

Input: Track A's catalog of the run (a2_build_catalog.py RUN -> /wd1/mdenolle_data/reloc_v5/catalog/
catalog_v5_trackA_<RUN>.csv: every v4 event, NLLoc location or v4 kept (tier U), tiers A-D/U). That file is written
with '%.5g' (lon to 0.01 deg, up to 0.4 km; a2_build_catalog.py of the reloc/track-a worktree), so lat, lon, depth and
exp_depth of the located events are taken again from runs/<RUN>/locations.csv, and the v4 coordinates (tier U and
v4_*) from the v4.0 catalog, both at full precision; the output is written with '%.8g' (D-B013). The time grids of
b4_time_grids.py use flattened depth z'; here depth, exp_depth and shift_z_km of the located events (loc_source
nll3d) are mapped back with z = Re (1 - exp(-z' / Re)) (0.07 km at 30 km, 0.2 km at 50 km). Tiers are kept (their
depth conditions move by less than the change).
Flags (rule 4 of D-B049..D-B051, PI 2026-11-24; /wd1/mdenolle_data/tomo_v1/depth_literature/depth_literature.html),
parameters in params b4.depth_flags and b4.quarry:
* depth_quality: "unconstrained" for every event of the Endeavour, Explorer and JdF ridge/plate regions and for Blanco
  events outside the 2012-09-20 to 2013-10-05 OBS deployment (UTC dates, inclusive); "biased_deep" for Gorda events
  with dist_min > 15 km; "ok" otherwise (tier U has no dist_min, so the Gorda rule does not apply to it);
* event_flag: "quarry_blast_likely" for events in the Texada Island box (lat 49.40-49.85, lon -124.75 to -124.25)
  on a weekday between 07:00 and 19:00 America/Vancouver time, "" otherwise.
Surface rule (D-B055, PI 2026-11-24): no hypocentre above the solid surface (land or seafloor). surface_km (km,
positive up) is GMT earth_relief 01m (gridline), bilinear, at the final epicentre of every event (tier U included: it
keeps the v4 position, 23 % of which are above the surface); depth_raw is the depth before the rule, depth and
exp_depth are max(value, -surface_km), and above_surface marks the events moved up to the surface. NLLoc masks the
same grid (LOCTOPO_SURFACE) but by nearest node, which leaves ~0.7 % of the located events up to 0.4 km above the
bilinear surface; tier U events can be km above it (v4 depths).
Outputs: /wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_<RUN>.csv, and data/tomo_v1/b4/<RUN>/ (compare.md,
compare_by_region.csv, depth_hist.png).

  python b4_catalog.py RUN [--ref iter1_vpvs176]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402

B4 = load_params().get("b4") or {}
# defaults = the values agreed with the PI (2026-11-24); params b4.depth_flags / b4.quarry override them key by key
DF = {"unconstrained_regions": ["Endeavour", "Explorer", "JdF ridge/plate"],
      "blanco_window": ["2012-09-20", "2013-10-05"], "gorda_biased_dist_km": 15.0}
QB = {"name": "Texada Island", "lat": [49.40, 49.85], "lon": [-124.75, -124.25], "weekday_hours_local": [7, 19],
      "tz": "America/Vancouver"}
for _d, _k in ((DF, "depth_flags"), (QB, "quarry")):
    assert set(B4.get(_k) or {}) <= set(_d), f"params b4.{_k}: unknown keys {set(B4[_k]) - set(_d)}"   # else ignored
    _d.update(B4.get(_k) or {})
RE = 6371.0
CAT = Path("/wd1/mdenolle_data/reloc_v5/catalog")
RUNS = Path("/wd1/mdenolle_data/reloc_v5/runs")
# a2_build_catalog.py reads the symlink ~/wt_reloc/data/catalog_v4/Cascadia_relocated_catalog_v4.csv, which points
# here (11-23). If it is repointed (e.g. to v4.1), change V4 too: the check in main() stops the run only when the
# coordinates differ beyond the 5th digit
V4 = Path("/wd1/mdenolle_data/catalog_v4_0_backup/Cascadia_relocated_catalog_v4.csv")


def unflatten(z):
    return RE * (1 - np.exp(-np.asarray(z, float) / RE))


def surface_km(lat, lon):
    """Solid-surface elevation (km, positive up): GMT earth_relief 01m, gridline, bilinear (as a2_parse_hyp.py)."""
    import pygmt
    rel = pygmt.datasets.load_earth_relief("01m", region=[-134, -118, 37.5, 52.5], registration="gridline")
    e = rel.interp(lat=("p", np.asarray(lat, float)), lon=("p", np.asarray(lon, float))).values / 1e3
    assert np.isfinite(e).all(), f"{int((~np.isfinite(e)).sum())} epicentres outside the earth_relief window"
    return e


def km_h(a, b):
    y = (a.lat.values - b.lat.values) * 111.195
    x = (a.lon.values - b.lon.values) * 111.195 * np.cos(np.radians(a.lat.values))
    return np.hypot(x, y)


def flags(c):
    """depth_quality and event_flag of catalog c (docstring; final lat, lon, time in UTC, region, dist_min)."""
    t = pd.to_datetime(c.time.str.rstrip("Z"), format="ISO8601").dt.tz_localize("UTC")
    w0, w1 = (pd.Timestamp(d, tz="UTC") for d in DF["blanco_window"])
    q = np.full(len(c), "ok", dtype=object)
    q[(c.region == "Gorda").to_numpy() & (c.dist_min > DF["gorda_biased_dist_km"]).to_numpy()] = "biased_deep"
    q[c.region.isin(DF["unconstrained_regions"]).to_numpy()] = "unconstrained"
    q[((c.region == "Blanco") & ((t < w0) | (t >= w1 + pd.Timedelta(days=1)))).to_numpy()] = "unconstrained"
    tl = t.dt.tz_convert(QB["tz"])
    qb = (c.lat.between(*QB["lat"]) & c.lon.between(*QB["lon"]) & (tl.dt.weekday < 5)
          & (tl.dt.hour >= QB["weekday_hours_local"][0]) & (tl.dt.hour < QB["weekday_hours_local"][1]))
    return q, np.where(qb, "quarry_blast_likely", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default="iter1_vpvs176")
    a = ap.parse_args()
    c = pd.read_csv(CAT / f"catalog_v5_trackA_{a.run}.csv")
    nll = c.loc_source == "nll3d"
    # full-precision coordinates in place of the '%.5g' ones (selection as in a2_build_catalog.py)
    loc = pd.read_csv(RUNS / a.run / "locations.csv")
    loc = loc[loc.status == "LOCATED"].drop_duplicates("ID").set_index("ID")
    assert set(loc.index) == set(c.ID[nll]), "locations.csv and the Track A catalog disagree on the located events"
    v4 = pd.read_csv(V4, skipinitialspace=True).set_index("ID")
    V = (("lat", "latitude"), ("lon", "longitude"), ("depth", "depth(km)"))
    for m, col, src in ([(nll, k, loc[k]) for k in ("lat", "lon", "depth", "exp_depth")] +
                        [(np.ones(len(c), bool), f"v4_{k}", v4[s]) for k, s in V] +
                        [(c.loc_source == "v4", k, v4[s]) for k, s in V]):
        new, old = src.reindex(c.ID[m]).to_numpy(float), c.loc[m, col].to_numpy(float)
        # more than the 5th digit apart (or missing) = run and catalog do not belong together, not rounding
        assert np.all(np.abs(new - old) <= 1e-4 * np.abs(old) + 1e-6), (col, np.nanmax(np.abs(new - old)))
        c.loc[m, col] = new
    for col in ("depth", "exp_depth"):
        c.loc[nll, col] = unflatten(c.loc[nll, col])
    # D-B055: no hypocentre above the solid surface (docstring)
    c["surface_km"] = surface_km(c.lat, c.lon)
    c["depth_raw"] = c.depth
    c["above_surface"] = c.depth < -c.surface_km
    c["depth"] = np.maximum(c.depth, -c.surface_km)
    c["exp_depth"] = np.maximum(c.exp_depth, -c.surface_km)   # NaN (tier U) stays NaN
    assert (c.depth >= -c.surface_km).all()
    c.loc[nll, "shift_z_km"] = c.loc[nll, "depth"] - c.loc[nll, "v4_depth"]
    c["depth_frame"] = np.where(nll, "true depth (EFT undone, b4_catalog.py)", "v4")
    c["depth_quality"], c["event_flag"] = flags(c)
    out = CAT / f"catalog_b4_{a.run}.csv"
    c.to_csv(out, index=False, float_format="%.8g")   # D-B013: %.5g keeps lon to 0.01 deg (0.4 km)
    r = pd.read_csv(CAT / f"catalog_v5_trackA_{a.ref}.csv").set_index("ID")
    c = c.set_index("ID")
    both = c.index[(c.loc_source == "nll3d")].intersection(r.index[r.loc_source == "nll3d"])
    cb, rb = c.loc[both], r.loc[both]
    d = pd.DataFrame({"region": cb.region, "tier_b4": cb.tier, "tier_ref": rb.tier,
                      "dz": cb.depth - rb.depth, "dh": km_h(cb, rb),
                      "dt": (pd.to_datetime(cb.time) - pd.to_datetime(rb.time)).dt.total_seconds(),
                      "rms_b4": cb.rms, "rms_ref": rb.rms, "above_b4": cb.above_surface, "above_ref": rb.above_surface})
    od = REPO / "data" / "tomo_v1" / "b4" / a.run
    od.mkdir(parents=True, exist_ok=True)
    g = d.groupby("region").agg(n=("dz", "size"), dz_med=("dz", "median"), absdz_med=("dz", lambda x: x.abs().median()),
                                absdz_p90=("dz", lambda x: x.abs().quantile(0.9)), dh_med=("dh", "median"),
                                dh_p90=("dh", lambda x: x.quantile(0.9)), dt_med=("dt", "median"),
                                rms_b4=("rms_b4", "median"), rms_ref=("rms_ref", "median"))
    g.loc["all"] = [len(d), d.dz.median(), d.dz.abs().median(), d.dz.abs().quantile(0.9), d.dh.median(),
                    d.dh.quantile(0.9), d.dt.median(), d.rms_b4.median(), d.rms_ref.median()]
    g.round(3).to_csv(od / "compare_by_region.csv")
    tiers = pd.DataFrame({"b4": c.tier.value_counts(), "ref": r.tier.value_counts()}).fillna(0).astype(int)
    L = [f"# B4 relocation {a.run} against {a.ref}", "",
         f"Catalog: {out} ({len(c)} events, {int(nll.sum())} located in 3D). Depths in km below sea level, "
         "EFT undone. Differences B4 minus reference, events located in both.", "",
         "## Tiers", "", "| tier | B4 | reference |", "|---|---|---|"]
    L += [f"| {t} | {row.b4} | {row.ref} |" for t, row in tiers.sort_index().iterrows()]
    L += ["", "## By region", "", "| region | n | dz med | abs dz med / p90 | dh med / p90 | dt med s | rms B4 / ref |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {int(v.n)} | {v.dz_med:+.2f} | {v.absdz_med:.2f} / {v.absdz_p90:.2f} | {v.dh_med:.2f} / "
          f"{v.dh_p90:.2f} | {v.dt_med:+.3f} | {v.rms_b4:.3f} / {v.rms_ref:.3f} |" for k, v in g.iterrows()]
    na, nu = int(c.above_surface.sum()), int((c.above_surface & (c.loc_source != "nll3d")).sum())
    L += ["", f"Above the surface before the D-B055 rule (located in both): B4 {100 * d.above_b4.mean():.1f} %, "
          f"reference {100 * d.above_ref.mean():.1f} % (reference: a2_parse_hyp.py check, no EFT).",
          f"D-B055: {na} events moved up to the surface ({na - nu} located in 3D, {nu} tier U at the v4 position).",
          "The ANSS check per tier is in the tier summary of a2_build_catalog.py (copied next to this file)."]
    fq = pd.crosstab(c.region, c.depth_quality, margins=True, margins_name="all")
    nq = int((c.event_flag == "quarry_blast_likely").sum())
    L += ["", "## Flags (rule 4 of D-B049..D-B051)", "", "| region | " + " | ".join(fq.columns) + " |",
          "|---" * (len(fq.columns) + 1) + "|"]
    L += [f"| {k} | " + " | ".join(str(v) for v in row) + " |" for k, row in fq.iterrows()]
    L += ["", f"event_flag quarry_blast_likely: {nq} events ({QB['name']} box, weekdays "
          f"{QB['weekday_hours_local'][0]:02d}:00-{QB['weekday_hours_local'][1]:02d}:00 {QB['tz']})."]
    (od / "compare.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regs = list(g.index[:-1])
    fig, ax = plt.subplots(1, len(regs), figsize=(2.6 * len(regs), 3.4), sharey=True, squeeze=False)
    for i, rg in enumerate(regs):
        s = d.region == rg
        bins = np.arange(-5, 61, 2)
        ax[0, i].hist(rb.depth[s.values], bins=bins, orientation="horizontal", histtype="step", color="0.5", label="ref")
        ax[0, i].hist(cb.depth[s.values], bins=bins, orientation="horizontal", histtype="step", color="C3", label="B4")
        ax[0, i].set_title(f"{rg} ({int(s.sum())})", fontsize=8)
        ax[0, i].tick_params(labelsize=7)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel("depth (km bsl)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(od / "depth_hist.png", dpi=130)
    write_manifest(f"b4_catalog_{a.run}", od, inputs=[CAT / f"catalog_v5_trackA_{a.run}.csv",
                                                      CAT / f"catalog_v5_trackA_{a.ref}.csv",
                                                      RUNS / a.run / "locations.csv", V4],
                   outputs=[out, od / "compare.md", od / "compare_by_region.csv", od / "depth_hist.png"],
                   params={"depth_flags": DF, "quarry": QB},
                   counts={"events": len(c), "nll3d": int(nll.sum()), "both": len(both),
                           "depth_quality": {k: int(v) for k, v in c.depth_quality.value_counts().items()},
                           "quarry_blast_likely": nq, "moved_to_surface": na, "moved_to_surface_tier_U": nu})
    print((od / "compare.md").read_text())


if __name__ == "__main__":
    main()
