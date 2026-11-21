#!/usr/bin/env python3
"""Ridge-transform report: Blanco transform, Axial Seamount and Endeavour in catalogs v3 and v4.

Catalog v4 (data/catalog_v4/) is built on the ver6 picks, which add to the ver3 picks of the
paper the region runs left out of ver3 (ver4), the offshore gap fill (ver5: 7A, 7D, NV, OO,
X9, Z5, incl. the OOI cabled array at Axial) and the 122-123°W strip (ver6). This script
measures what those added picks bring to the three ridge-transform areas.

For each area: events and QC events in v3 and v4, v3 events recovered by v4 (3 s, 30 km),
events new in v4, share of the v4 picks that are not in ver3 (by pick table version and
network), monthly event rates, a depth section, and maps (wide and tight zooms, v3 and v4,
phase5_pygmt_map.py with compact marker sizes). QC: v3 >= 4 P, >= 4 S, RMS < 2.5 s;
v4 >= 4 P, >= 4 S (v4 has no RMS). Depths are as delivered (v4 datum to confirm).

Writes data/catalog_v4/ridge_report/ (figures, summary.json) and the self-contained
data/catalog_v4/ridge_report/ridge_transform_v4_report.html.

    pixi run python workflow/06_analysis/ridge_transform_v4_report.py [--skip-maps]
"""
import argparse
import base64
import io
import json
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
V4 = ROOT / "data" / "catalog_v4"
REP = V4 / "report"                       # adapters written by catalog_v4_report.py
OUT = V4 / "ridge_report"
V3 = ROOT / "data" / "datasets_all_regions" / "origin_2010_2015_reloc_cog_ver3_cc.csv"
V3_M = ROOT / "data" / "magnitude" / "cascadia_catalog_M_routeA.csv"
VER6 = Path("/wd1/mdenolle_data/picks_v6/all_picks_all_regions_2010_2016_ver6.csv.gz")
N_VER3, N_VER4, N_VER5 = 39_597_551, 49_541_718, 60_918_847     # first pick_id of each addition
C3, C4 = "#6a64ab", "#c45a12"
ERUPTION = pd.Timestamp("2015-04-24", tz="UTC")                  # Axial eruption onset

# area box (lon0, lon1, lat0, lat1), map presets (wide, tight), depth section
AREAS = {
    "Blanco transform": dict(box=(-130.6, -126.6, 42.6, 45.3), maps=("blanco", "blanco_central"),
                             section=("line", (44.45, -130.35), (43.05, -126.75), 25.0)),
    "Axial Seamount": dict(box=(-130.35, -129.75, 45.75, 46.2), maps=("axial",),
                           section=("lon", 45.95)),
    "Endeavour": dict(box=(-130.4, -127.2, 47.4, 49.8), maps=("endeavour", "endeavour_north"),
                      section=("lat_window", (-129.7, -127.9, 48.55, 49.4))),
}


def load():
    ev = pd.read_csv(V4 / "Cascadia_relocated_catalog_v4.csv", skipinitialspace=True)
    ev["time"] = pd.to_datetime(ev.datetime, utc=True)
    ev["qc"] = (ev.Num_p_picks >= 4) & (ev.Num_s_picks >= 4)
    ev = ev.rename(columns={"latitude": "lat", "longitude": "lon", "depth(km)": "depth",
                            "magnitude": "M"})
    v3 = pd.read_csv(V3, index_col=0)
    v3["time"] = pd.to_datetime(v3.time, unit="s", utc=True)
    v3["qc"] = (v3.p_picks >= 4) & (v3.s_picks >= 4) & (v3.rms < 2.5)
    v3 = v3.merge(pd.read_csv(V3_M, usecols=["event_id", "M"]), left_on="orid",
                  right_on="event_id", how="left")
    return ev, v3


def pick_ids():
    """v4 pick -> ver6 pick_id (same station, same time); cached."""
    cache = OUT / "v4_pick_ids.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import pyarrow as pa
    import pyarrow.csv as pc
    pk = pd.read_csv(V4 / "Cascadia_picks_catalog_v4.csv", skipinitialspace=True,
                     usecols=["event_ID", "pick_datetime", "station", "phase"])
    pk["t"] = pd.to_datetime(pk.pick_datetime, utc=True).astype("int64")
    tb = pc.read_csv(VER6, convert_options=pc.ConvertOptions(
        include_columns=["network", "station", "pick_time", "pick_id"],
        column_types={c: pa.string() for c in ("network", "station", "pick_time")} | {"pick_id": pa.int64()}))
    v = tb.to_pandas()
    v["sta"] = v.station + "." + v.network
    v = v[v.sta.isin(set(pk.station))]
    v["t"] = pd.to_datetime(v.pick_time, format="ISO8601", utc=True).astype("int64")
    m = pd.merge_asof(pk.sort_values("t"), v.sort_values("t")[["t", "sta", "pick_id"]], on="t",
                      left_by="station", right_by="sta", direction="nearest", tolerance=5_000_000)
    m = m[["event_ID", "station", "phase", "pick_id"]]
    m.to_parquet(cache)
    return m


def inbox(d, box):
    return d[d.lon.between(box[0], box[1]) & d.lat.between(box[2], box[3])]


def match_v3(a3, a4):
    a = a3.sort_values("time")[["orid", "time", "lat", "lon"]].assign(t=lambda x: x.time.astype("int64") / 1e9)
    b = a4.sort_values("time")[["ID", "time", "lat", "lon"]].assign(t=lambda x: x.time.astype("int64") / 1e9)
    m = pd.merge_asof(a.drop(columns="time"), b.drop(columns="time").rename(columns={"lat": "la4", "lon": "lo4"}),
                      on="t", direction="nearest", tolerance=3.0)
    d = np.hypot(m.lat - m.la4, (m.lon - m.lo4) * np.cos(np.radians(m.lat))) * 111.2
    return m.ID.where(d < 30)


def provenance(p):
    v = pd.cut(p.pick_id, [-1, N_VER3 - 1, N_VER4 - 1, N_VER5 - 1, np.inf],
               labels=["ver3", "ver4", "ver5", "ver6"])
    return v


def fig_rates(a3, a4, name, out):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bins = pd.date_range("2010-01-01", "2017-01-01", freq="MS", tz="UTC")
    for d, c, lab in ((a3, C3, "v3"), (a4, C4, "v4")):
        for q, ls in ((d, "-"), (d[d.qc], ":")):
            h = pd.cut(q.time, bins).value_counts(sort=False)
            ax.step(bins[:-1], h.values, where="post", color=c, ls=ls, lw=1.2,
                    label=f"{lab} {'all' if q is d else 'QC'}")
    if name == "Axial Seamount":
        ax.axvline(ERUPTION, color="k", lw=0.6)
        ax.text(ERUPTION, ax.get_ylim()[1] * 0.92, " eruption 2015-04-24", fontsize=7, va="top")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Events per month")
    ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def section_xy(d, sec):
    lat0 = np.radians(d.lat.mean() if len(d) else 45)
    if sec[0] == "line":
        (la1, lo1), (la2, lo2), half = sec[1], sec[2], sec[3]
        x1, y1 = lo1 * 111.2 * np.cos(lat0), la1 * 111.2
        x2, y2 = lo2 * 111.2 * np.cos(lat0), la2 * 111.2
        ux, uy = (x2 - x1), (y2 - y1)
        L = np.hypot(ux, uy)
        ux, uy = ux / L, uy / L
        px, py = d.lon * 111.2 * np.cos(lat0) - x1, d.lat * 111.2 - y1
        along, across = px * ux + py * uy, -px * uy + py * ux
        keep = (across.abs() <= half) & along.between(0, L)
        return along[keep], d.depth[keep], "Distance along the transform from its NW end (km)"
    if sec[0] == "lon":
        x = (d.lon + 130.0) * 111.2 * np.cos(lat0)
        return x, d.depth, "Distance east of 130°W (km)"
    b = sec[1]
    k = d.lon.between(b[0], b[1]) & d.lat.between(b[2], b[3])
    return (d.lat[k] - b[2]) * 111.2, d.depth[k], f"Distance north of {b[2]}°N (km), {-b[1]}-{-b[0]}°W"


def fig_section(a3, a4, sec, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, sharex=True)
    for ax, d, c, lab in ((axs[0], a3[a3.qc], C3, "v3 QC"), (axs[1], a4[a4.qc], C4, "v4 QC")):
        x, z, xl = section_xy(d, sec)
        ax.scatter(x, z, s=1.2, c=c, alpha=0.35, lw=0)
        ax.set_title(f"{lab} ({len(x):,})", fontsize=8)
        ax.set_xlabel(xl, fontsize=7)
        ax.axhline(0, color="k", lw=0.4)
    axs[0].set_ylabel("Depth (km)")
    axs[0].set_ylim(30, -6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_sources(p, name, out):
    t = p.groupby(["net", "ver"], observed=True).size().unstack(fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index[:10]]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    left = np.zeros(len(t))
    cols = {"ver3": "#9aa3b5", "ver4": "#5b8fb9", "ver5": C4, "ver6": "#7a4b9c"}
    for v in [c for c in ["ver3", "ver4", "ver5", "ver6"] if c in t.columns]:
        ax.barh(t.index, t[v], left=left, color=cols[v], label=v)
        left += t[v].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("Picks of v4 events in the area")
    ax.legend(frameon=False, fontsize=7, title="pick table", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run_maps(jobs):
    def one(j):
        name, preset, cat, qc = j
        cmd = [sys.executable, "phase5_pygmt_map.py", "--region", preset, "--compact-sizes",
               "--gmrt", "high" if preset == "axial" else "med",
               "--out", str(OUT / name)]
        if preset == "axial":
            cmd += ["--width-cm", "8.4", "--size-scale", "0.9"]
        if cat:
            cmd += ["--catalog", str(cat), "--qc-catalog", str(qc)]
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        return name, r.returncode, (r.stdout + r.stderr)[-300:]
    with ThreadPoolExecutor(5) as ex:
        return list(ex.map(one, jobs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-maps", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ev, v3 = load()
    pk = pick_ids()
    pk["net"] = pk.station.str.split(".").str[1]
    pk["ver"] = provenance(pk)
    s = {}
    jobs = []
    for name, cfg in AREAS.items():
        key = name.split()[0].lower()
        a3, a4 = inbox(v3, cfg["box"]), inbox(ev, cfg["box"])
        rec = match_v3(a3, a4)
        p = pk[pk.event_ID.isin(a4.ID)]
        p_qc = pk[pk.event_ID.isin(a4[a4.qc].ID)]
        added = (p.ver != "ver3").mean()
        only_new = p.groupby("event_ID").ver.apply(lambda x: (x != "ver3").all())
        st = {"v3_events": int(len(a3)), "v3_qc": int(a3.qc.sum()),
              "v4_events": int(len(a4)), "v4_qc": int(a4.qc.sum()),
              "v4_events_2010_2015": int((a4.time.dt.year <= 2015).sum()),
              "v3_recovered_frac": float(rec.notna().mean()) if len(rec) else None,
              "v3_qc_recovered_frac": float(rec[a3.qc.to_numpy()].notna().mean()) if a3.qc.any() else None,
              "v4_events_not_in_v3": int(len(a4) - rec.dropna().nunique()),
              "picks": int(len(p)), "picks_added_after_ver3_frac": float(added),
              "picks_by_version": p.ver.value_counts().sort_index().astype(int).to_dict(),
              "events_only_added_picks": int(only_new.sum()),
              "picks_per_event_median_v4": float(a4.Num_p_picks.add(a4.Num_s_picks).median()),
              "picks_per_event_median_v3": float(a3.nass.median()) if len(a3) else None,
              "depth_median_v4_qc": float(a4[a4.qc].depth.median()),
              "depth_median_v3_qc": float(a3[a3.qc].depth.median()) if a3.qc.any() else None,
              "top_networks": p.net.value_counts().head(6).astype(int).to_dict()}
        if key == "axial":
            pre = a4.time < ERUPTION
            st["v4_events_2015_04_24_to_05_31"] = int(a4.time.between(ERUPTION, pd.Timestamp("2015-06-01", tz="UTC")).sum())
            st["v3_events_2015_04_24_to_05_31"] = int(a3.time.between(ERUPTION, pd.Timestamp("2015-06-01", tz="UTC")).sum())
            st["v4_events_before_eruption"] = int(pre.sum())
        s[name] = st
        fig_rates(a3, a4, name, OUT / f"{key}_rates.png")
        fig_section(a3, a4, cfg["section"], OUT / f"{key}_section.png")
        fig_sources(p_qc if len(p_qc) else p, name, OUT / f"{key}_sources.png")
        for preset in cfg["maps"]:
            jobs += [(f"{key}_{preset}_v3.png", preset, None, None),
                     (f"{key}_{preset}_v4.png", preset, REP / "v4_catalog.csv", REP / "v4_qc.csv")]
    if not a.skip_maps:
        res = run_maps(jobs)
        s["maps"] = {n: rc for n, rc, _ in res}
        for n, rc, log in res:
            if rc:
                print(f"map {n} failed:\n{log}")
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
