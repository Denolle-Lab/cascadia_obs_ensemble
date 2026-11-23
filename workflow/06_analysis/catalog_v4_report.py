#!/usr/bin/env python3
"""Catalog v4 report: the main-paper figures redrawn with catalog v4, next to catalog v3.

Catalog v4 (data/catalog_v4/, utils/fix_catalog_v4.py) is I. McBrearty's GENIE + GraphDD
catalog of the ver6 picks, 2010-2016. It has no travel-time residuals, so the v3 QC cut
(P >= 4, S >= 4, RMS < 2.5 s) becomes P >= 4, S >= 4 here, and the RMS panel of Fig. 3 is
replaced by the GraphDD spatial uncertainty (`spatial_sigma`). Its `magnitude` column is used
as M (type to confirm with I. McBrearty).

Writes to data/catalog_v4/report/:
  v4_catalog.csv, v4_qc.csv         adapters for phase5_pygmt_map.py (event_id, evla, evlo, evdp,
                                    M, M_type / orid, nass)
  v4_events_gap.csv                 events with nass, azimuthal gap, qc flag
  fig01_stations_v4.png             stations with v4 picks (new vs v3)
  fig03_qc_v4.png                   QC histograms, v3 vs v4
  fig05a/05b/06a-f/07 *_v4.png      phase5_pygmt_map.py on v4 (same presets as the paper)
  fig08_axial_endeavour_v4.png      Axial/Endeavour events coloured by number of picks
  figX_yearly_v4.png, figX_mfd_v4.png, figX_depth_v4.png
  summary.json                      numbers quoted in the report

    pixi run python workflow/06_analysis/catalog_v4_report.py [--skip-maps]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "utils"))
try:
    import paper_style  # noqa: F401  (rcParams of the paper figures)
except Exception:
    pass

V4 = ROOT / "data" / "catalog_v4"
OUT = V4 / "report"
V3 = ROOT / "data" / "datasets_all_regions" / "origin_2010_2015_reloc_cog_ver3_cc.csv"
V3_STATIONS = ROOT / "data" / "datasets_all_regions" / "all_stations_2010_2015_ver3.csv"
STATIONS = Path("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
PB = ROOT / "data" / "datasets_all_regions" / "PB2002_boundaries.dig"
C3, C4 = "#7570b3", "#d95f02"            # v3, v4

MAPS = [  # (report name, phase5 args) -- the presets of paper/export_figures.py
    ("fig05a_catalog_map_v4.png", ["--region", "full"]),
    ("fig05b_context_map_v4.png", ["--region", "full", "--layer", "context"]),
    ("fig06a_endeavour_v4.png", ["--region", "endeavour", "--gmrt", "med"]),
    ("fig06b_wa_margin_v4.png", ["--region", "wa_margin", "--gmrt", "med",
                                 "--coupling-transparency", "65"]),
    ("fig06c_or_margin_v4.png", ["--region", "or_margin", "--gmrt", "med"]),
    ("fig06d_blanco_v4.png", ["--region", "blanco", "--gmrt", "med"]),
    ("fig06e_gorda_v4.png", ["--region", "gorda", "--gmrt", "med"]),
    ("fig06f_mendocino_v4.png", ["--region", "mendocino", "--gmrt", "med"]),
    ("fig07_puget_slab_v4.png", ["--region", "puget", "--slab-contours", "--legend",
                                 "--legend-pos", "TR", "--gmrt", "med"]),
]


def load():
    ev = pd.read_csv(V4 / "Cascadia_relocated_catalog_v4.csv", skipinitialspace=True)
    pk = pd.read_csv(V4 / "Cascadia_picks_catalog_v4.csv", skipinitialspace=True,
                     usecols=["event_ID", "station", "phase"])
    ev["time"] = pd.to_datetime(ev.datetime, utc=True)
    ev["nass"] = ev.Num_p_picks + ev.Num_s_picks
    ev["qc"] = (ev.Num_p_picks >= 4) & (ev.Num_s_picks >= 4)
    v3 = pd.read_csv(V3, index_col=0)
    v3["time_utc"] = pd.to_datetime(v3.time, unit="s", utc=True)
    v3["qc"] = (v3.p_picks >= 4) & (v3.s_picks >= 4) & (v3.rms < 2.5)
    return ev, pk, v3


def azimuthal_gap(ev, pk):
    st = pd.read_csv(STATIONS).drop_duplicates(["network", "station"])
    st["key"] = st.station + "." + st.network
    xy = st.set_index("key")[["lat", "lon"]]
    p = pk[["event_ID", "station"]].drop_duplicates().join(xy, on="station")
    miss = p.lat.isna().sum()
    p = p.dropna().join(ev.set_index("ID")[["latitude", "longitude"]], on="event_ID")
    la1, lo1, la2, lo2 = map(np.radians, (p.latitude, p.longitude, p.lat, p.lon))
    az = np.degrees(np.arctan2(np.sin(lo2 - lo1) * np.cos(la2),
                               np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1))) % 360
    p = p.assign(az=az.values).sort_values(["event_ID", "az"])
    g = p.groupby("event_ID").az
    inner = p.az.diff().where(p.event_ID.eq(p.event_ID.shift())).groupby(p.event_ID).max()
    wrap = 360 - g.max() + g.min()                       # across north; 360 for one station
    return np.fmax(inner.reindex(wrap.index), wrap), int(miss)


def fig_stations(pk, out):
    import pygmt
    st = pd.read_csv(STATIONS).drop_duplicates(["network", "station"])
    st["key"] = st.station + "." + st.network
    n = pk.groupby("station").size().rename("n")
    st = st.join(n, on="key").dropna(subset=["n"])
    v3s = pd.read_csv(V3_STATIONS)
    st["in_v3"] = [(a, b) in set(zip(v3s.network, v3s.station)) for a, b in zip(st.network, st.station)]
    fig = pygmt.Figure()
    region = [-131.5, -119.5, 39, 51.5]
    pygmt.config(FONT_ANNOT_PRIMARY="7p", FONT_LABEL="7p", MAP_FRAME_TYPE="plain")
    grid = pygmt.datasets.load_earth_relief(resolution="02m", region=region)
    fig.basemap(region=region, projection="M8.4c", frame=["WSne", "xa2f1", "ya2f1"])
    fig.grdimage(grid=grid, cmap="gray", shading=True, transparency=55)
    fig.coast(shorelines="0.25p,black", borders=["1/0.3p", "2/0.2p"], resolution="i")
    size = 0.06 + 0.05 * np.log10(st.n)
    for sub, fill, lab in ((st[st.in_v3 & (st.elev_m < 0)], "dodgerblue3", "seafloor - in v3"),
                           (st[st.in_v3 & (st.elev_m >= 0)], "gray30", "land - in v3"),
                           (st[~st.in_v3 & (st.elev_m < 0)], "cyan", "seafloor - new in v4"),
                           (st[~st.in_v3 & (st.elev_m >= 0)], "orange", "land - new in v4")):
        if len(sub):
            fig.plot(x=sub.lon, y=sub.lat, size=size[sub.index], style="tc", fill=fill,
                     pen="0.15p,black", label=f"{lab} ({len(sub)})+S0.15c")
    fig.legend(position="jBL+o0.1c", box="+gwhite@20+p0.3p")
    fig.savefig(out, dpi=300)
    return {"stations_v4": int(len(st)), "stations_new_vs_v3": int((~st.in_v3).sum()),
            "stations_v3_list": int(len(v3s))}


def fig_qc(ev, v3, out):
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for a, (c3, c4, xl, bins) in zip(ax.flat, [
            ("p_picks", "Num_p_picks", "P picks per event", np.arange(0, 81, 2)),
            ("s_picks", "Num_s_picks", "S picks per event", np.arange(0, 81, 2)),
            ("gap", "gap", "Azimuthal gap (deg)", np.arange(0, 361, 10)),
            (None, "spatial_sigma(m)", "v4: GraphDD spatial uncertainty (km)", np.arange(0, 51, 1))]):
        q3, q4 = v3[v3.qc], ev[ev.qc]
        if c3:
            a.hist(q3[c3], bins=bins, color=C3, histtype="step", lw=1.4, label=f"v3 QC ({len(q3):,})", zorder=3)
            a.hist(q4[c4], bins=bins, color=C4, alpha=0.55, label=f"v4 QC ({len(q4):,})")
        else:
            a.hist(q4[c4] / 1000, bins=bins, color=C4, alpha=0.7, label=f"v4 QC ({len(q4):,})")
        a.set_xlabel(xl)
        a.set_ylabel("Events")
        a.legend(frameon=False, fontsize=7)
    for a, l in zip(ax.flat, "abcd"):
        a.text(0.97, 0.95, f"({l})", transform=a.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_axial(ev, out):
    import pygmt
    d = ev[ev.qc].sort_values("nass")
    region = [-130, -126, 46, 50.5]
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="7p,Helvetica,black", FONT_LABEL="7p,Helvetica,black",
                 MAP_FRAME_TYPE="plain")
    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)
    shade = pygmt.grdgradient(grid=grid, azimuth="0/90", normalize="t1")
    fig.basemap(region=region, projection="M6.9c", frame=["WSne", "xa1f0.5", "ya1f0.5"])
    fig.grdimage(grid=grid, cmap="blue", shading=shade, transparency=50)
    fig.coast(shorelines="0.25p,black", borders=["1/0.3p", "2/0.2p"], resolution="f")
    if PB.exists():
        fig.plot(data=str(PB), pen="0.8p,dodgerblue1", transparency=50)
    d = d[d.longitude.between(*region[:2]) & d.latitude.between(*region[2:])]
    pygmt.makecpt(cmap="viridis", series=[d.nass.min(), d.nass.max()])
    fig.plot(x=d.longitude, y=d.latitude, style="c0.07c", fill=d.nass, cmap=True,
             pen="0.05p,black", transparency=35)
    fig.colorbar(position="JBC+o0c/0.7c+h+w5c/0.18c",
                 frame=["xa50f25+lNumber of P and S picks per event"])
    fig.savefig(out, dpi=300)
    return int(len(d))


def fig_yearly(ev, v3, out):
    y4, y3 = ev.time.dt.year.value_counts().sort_index(), v3.time_utc.dt.year.value_counts().sort_index()
    q4 = ev[ev.qc].time.dt.year.value_counts().sort_index()
    q3 = v3[v3.qc].time_utc.dt.year.value_counts().sort_index()
    yrs = np.arange(2010, 2017)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    w = 0.2
    for k, (s, c, lab, h) in enumerate([(y3, C3, "v3 all", None), (q3, C3, "v3 QC", "//"),
                                        (y4, C4, "v4 all", None), (q4, C4, "v4 QC", "//")]):
        ax.bar(yrs + (k - 1.5) * w, s.reindex(yrs, fill_value=0), w, color=c,
               alpha=0.9 if h is None else 0.45, hatch=h, label=lab)
    ax.set_xticks(yrs)
    ax.set_ylabel("Events per year")
    ax.legend(frameon=False, ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_mfd(ev, out):
    m3 = pd.read_csv(ROOT / "data" / "magnitude" / "cascadia_catalog_M_routeA.csv", usecols=["M"]).M.dropna()
    m4 = ev.magnitude.dropna()
    bins = np.arange(-2, 7.01, 0.1)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for m, c, lab in ((m3, C3, f"v3 M (Route A, {len(m3):,})"), (m4, C4, f"v4 magnitude ({len(m4):,})")):
        h, e = np.histogram(m, bins)
        ax.semilogy(e[:-1], h, "o", ms=2.5, color=c, alpha=0.7)
        ax.semilogy(e[:-1], h[::-1].cumsum()[::-1], "-", color=c, label=lab)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Events (dots: per 0.1 bin; line: cumulative)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {"mag_v4_median": float(m4.median()), "mag_v3_median": float(m3.median()),
            "mag_v4_max": float(m4.max())}


def fig_depth(ev, v3, out):
    bins = np.arange(-12, 61, 1)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(v3[v3.qc].depth, bins=bins, color=C3, histtype="step", lw=1.4, label="v3 QC", zorder=3)
    ax.hist(ev[ev.qc]["depth(km)"], bins=bins, color=C4, alpha=0.6, label="v4 QC")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Depth (km; datum to confirm for v4)")
    ax.set_ylabel("Events")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run_maps(cat, qc):
    def one(job):
        name, args = job
        cmd = [sys.executable, "phase5_pygmt_map.py", "--compact-sizes", "--catalog", str(cat), "--qc-catalog", str(qc),
               "--out", str(OUT / name)] + args
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        return name, r.returncode, (r.stdout + r.stderr)[-400:]
    with ThreadPoolExecutor(5) as ex:
        return list(ex.map(one, MAPS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-maps", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ev, pk, v3 = load()
    gap, miss = azimuthal_gap(ev, pk)
    ev["gap"] = ev.ID.map(gap)
    ev.drop(columns="time").to_csv(OUT / "v4_events_gap.csv", index=False)

    cat = OUT / "v4_catalog.csv"
    pd.DataFrame({"event_id": ev.ID, "otime": ev.datetime, "evla": ev.latitude, "evlo": ev.longitude,
                  "evdp": ev["depth(km)"], "M": ev.magnitude, "M_type": "ML"}).to_csv(cat, index=False)
    qc = OUT / "v4_qc.csv"
    pd.DataFrame({"orid": ev.ID, "nass": ev.nass}).to_csv(qc, index=False)

    s = {"events_v4": int(len(ev)), "events_v4_qc": int(ev.qc.sum()), "events_v3": int(len(v3)),
         "events_v3_qc": int(v3.qc.sum()), "picks_v4": int(len(pk)),
         "picks_without_station_coords": miss,
         "events_v4_2010_2015": int((ev.time.dt.year <= 2015).sum()),
         "events_v4_qc_2010_2015": int((ev.qc & (ev.time.dt.year <= 2015)).sum()),
         "gap_median_v4_qc": float(ev[ev.qc].gap.median()), "gap_median_v3_qc": float(v3[v3.qc].gap.median()),
         "nass_median_v4": float(ev.nass.median()), "nass_median_v3": float(v3.nass.median()),
         "sigma_km_median_v4_qc": float(ev[ev.qc]["spatial_sigma(m)"].median() / 1000)}
    s.update(fig_stations(pk, OUT / "fig01_stations_v4.png"))
    fig_qc(ev, v3, OUT / "fig03_qc_v4.png")
    s["events_axial_endeavour_v4_qc"] = fig_axial(ev, OUT / "fig08_axial_endeavour_v4.png")
    fig_yearly(ev, v3, OUT / "figX_yearly_v4.png")
    s.update(fig_mfd(ev, OUT / "figX_mfd_v4.png"))
    fig_depth(ev, v3, OUT / "figX_depth_v4.png")
    if not a.skip_maps:
        res = run_maps(cat, qc)
        s["maps"] = {n: rc for n, rc, _ in res}
        for n, rc, log in res:
            if rc:
                print(f"map {n} failed:\n{log}")
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
