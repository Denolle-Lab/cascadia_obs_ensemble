"""Magnitude of completeness and b-values for catalog v5.1 (v5.1 hypocentres and magnitudes).

Same estimators, grid and regional boxes as v4_mc_maps.py (imported; MAXC+0.2, MBS,
GFT90/95, the Woessner & Wiemer 2005 "best" rule, 200 bootstrap resamples, Aki-Utsu b
with the Shi & Bolt error), run on catalog v5.1:

  * events: located in v5.1 (loc_source nll3d or comcat; the 623 rows without a
    hypocentre are left out), with a magnitude M, and earthquakes only (rows with an
    explosion / quarry-blast / other / sonic-boom flag are left out);
  * epicentre and year from the v5.1 hypocentre;
  * M: the v5.1 preferred magnitude (cascadia_catalog_v5.1_magnitudes.csv);
  * QC variant: at least 4 P and 4 S picks in the v4.1t pick file (the v4 rule).

Regional tables, three versions so that the magnitude change can be told apart from
the relocation and selection change:
  A  v4.1 M, v4 epicentres, every v4.1 event with M (what v4_mc_maps.py did);
  B  v4.1 M, v5.1 epicentres, the v5.1 selection above;
  C  v5.1 M, v5.1 epicentres, the v5.1 selection above.
B vs C isolates the new magnitudes; A vs B the relocation and selection.
b uncertainty: Shi & Bolt (1982) at Mc_best, and the bootstrap standard deviation of b
when Mc_best is re-estimated in each of the 200 resamples (b_sigma_boot).
Also per v5.1 catalog region label (column region).

Stage "paired" (paired_b.csv): the test for a change of b. On the events that have
both a v5.1 and a v4.1 M (located earthquakes), b(v5.1 M) - b(v4.1 M) at one common Mc
per box (Mc_best of C, and also Mc_best of B where it differs), with a paired bootstrap
(the same resampled events for both magnitudes, Mc held fixed, 500 resamples). z is the
difference over the paired bootstrap std; the Bonferroni threshold is for the boxes
tested at Mc_best of C. The b_sigma_boot of the regional table is not a test of the
change: it treats the two estimates as independent and lets Mc move between them.
Extra rows: the onshore box without Mount St. Helens (lat 46.0-46.4, lon -122.4 to
-122.0) and Mount St. Helens alone.

Usage:
  python v51_mc_b.py --mags /wd1/.../mag/cascadia_catalog_v5.1_magnitudes.csv \
      --out <worktree>/data/catalog_v5/v5.1_analysis/mag/mc_b [--stage compute|plot|paired|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4_mc_maps as mm  # noqa: E402

PICKS = "/wd1/mdenolle_data/catalog_v5/v5.1/Cascadia_picks_catalog_v4.1t.csv.gz"
V4CAT = "/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/Cascadia_relocated_catalog_v4.csv"
_MAGS = None
_PS = None


def _regions(lon, lat):
    reg = np.full(len(lon), "ON", dtype=object)
    for key, _, (a, b, c, d) in mm.REGIONS[::-1]:
        s = (lon >= a) & (lon <= b) & (lat >= c) & (lat <= d)
        reg[np.asarray(s)] = key
    return reg


def _pick_counts():
    global _PS
    if _PS is None:
        p = pd.read_csv(PICKS, usecols=[0, 5], skipinitialspace=True)
        p.columns = ["event_ID", "phase"]
        _PS = p.assign(phase=p.phase.str.strip()).groupby(["event_ID", "phase"]).size().unstack(fill_value=0)
    return _PS


def _finish(df, mcol, qc):
    if qc:
        ps = _pick_counts()
        ok = (ps.get("P", 0) >= 4) & (ps.get("S", 0) >= 4)
        df = df[df.event_id.map(ok).fillna(False).astype(bool)]
    df = df[df[mcol].notna()].copy()
    df["M"] = df[mcol]
    df["ibin"] = np.clip(np.round((df.M.values - mm.MLO) / mm.DM).astype(int), 0, len(mm.MBINS) - 1)
    df["region"] = _regions(df.lon.values, df.lat.values)
    return df.reset_index(drop=True)


def load_v51(qc: bool = False, mcol: str = "M") -> pd.DataFrame:
    """Version C (mcol M) or B (mcol M_v4): v5.1 epicentres, located earthquakes."""
    d = _MAGS.copy()
    d = d[d.loc_source.isin(["nll3d", "comcat"]) & d.is_earthquake.astype(bool)]
    d = d.rename(columns={"ID": "event_id"})
    d["year"] = pd.to_datetime(d.time, utc=True, format="ISO8601").dt.year
    return _finish(d, mcol, qc)


def load_v4(qc: bool = False) -> pd.DataFrame:
    """Version A: v4.1 M on the v4 epicentres, every event with M."""
    d = _MAGS.rename(columns={"ID": "event_id"})[["event_id", "M_v4", "catalog_region"]].copy()
    cat = pd.read_csv(V4CAT, skipinitialspace=True)
    cat.columns = [c.strip() for c in cat.columns]
    cat = cat.rename(columns={"ID": "event_id", "latitude": "lat", "longitude": "lon"})
    cat["year"] = pd.to_datetime(cat["datetime"].str.strip(), utc=True, format="ISO8601").dt.year
    v41 = _MAGS[_MAGS.ID >= 900001][["ID", "time", "lat", "lon"]].rename(columns={"ID": "event_id"})
    v41["year"] = pd.to_datetime(v41.time, utc=True, format="ISO8601").dt.year
    cat = pd.concat([cat[["event_id", "lat", "lon", "year"]], v41[["event_id", "lat", "lon", "year"]]])
    d = d.merge(cat, on="event_id", how="inner")
    return _finish(d, "M_v4", qc)


def b_boot(ibin, nboot, seed):
    """Point Mc/b + bootstrap sigma of b with Mc re-estimated per resample."""
    r = mm.mc_with_bootstrap(ibin, nboot, seed)
    h = np.bincount(ibin, minlength=len(mm.MBINS))
    nz = np.nonzero(h)[0]
    lo, hi = nz[0], max(nz[-1] + 1, nz[0] + mm.WIN + 1)
    h, m = h[lo:hi], mm.MBINS[lo:hi]
    rng = np.random.default_rng(seed)
    C = np.vstack([h, rng.multinomial(int(h.sum()), h / h.sum(), size=nboot)])
    rr = mm.mc_counts(C, m)
    bb = rr["b"][1:]
    r["b_sigma_boot"] = float(np.nanstd(bb)) if np.isfinite(bb).sum() > 10 else np.nan
    # b above a fixed MAXC+0.2 cutoff (the most conservative of the three at most nodes)
    j = int(np.clip(np.round((r["mc_maxc"] - m[0]) / mm.DM), 0, len(m) - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.cumsum(C[:, ::-1], 1)[:, ::-1][:, j]
        s1 = np.cumsum((C * m)[:, ::-1], 1)[:, ::-1][:, j]
        bm = 1.0 / (mm.LN10 * (s1 / n - (m[j] - mm.DM / 2)))
    bm[n < mm.NMIN_B] = np.nan
    r["b_maxc"] = float(bm[0])
    r["b_maxc_sigma_boot"] = float(np.nanstd(bm[1:])) if np.isfinite(bm[1:]).sum() > 10 else np.nan
    r["n_maxc"] = int(n[0])
    return r


def regional(df, nboot, key="region", order=None, periods=True):
    rows = []
    pers = [("all", (2010, 2016))] + (list(mm.PERIODS.items()) if periods else [])
    for per, (y0, y1) in pers:
        dp = df[df.year.between(y0, y1)]
        regs = (order or sorted(dp[key].dropna().unique())) + (["OFF", "ALL"] if key == "region" else ["ALL"])
        for reg in regs:
            if reg == "ALL":
                d = dp
            elif reg == "OFF":
                d = dp[dp[key].isin(mm.OFFSHORE)]
            else:
                d = dp[dp[key] == reg]
            row = dict(region=reg, period=per, n=len(d))
            if len(d) >= 100:
                row |= b_boot(d.ibin.values, nboot, seed=zlib.crc32(f"{reg}{per}".encode()))
            rows.append(row)
    return pd.DataFrame(rows)


def compute(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mm.OUT = out
    meta = dict(n_events=a.n, n_events_period=a.n_period, rmax_km=a.rmax, nboot=a.nboot,
                step_deg=a.step, nmin_b=mm.NMIN_B, maxc_corr=mm.CORR, mbs_window_bins=mm.WIN,
                dm=mm.DM, periods=mm.PERIODS, mags=a.mags)
    C = load_v51(False, "M")
    print(f"C: v5.1 M, located earthquakes with M: {len(C):,}", flush=True)
    g = mm.mc_grid(C, a.n, a.rmax, a.nboot, a.step, a.nproc)
    g.to_csv(out / "mc_grid.csv", index=False, float_format="%.4f")
    print("grid all:", int(g.mc_best.notna().sum()), "nodes", flush=True)
    Cq = load_v51(True, "M")
    gq = mm.mc_grid(Cq, a.n, a.rmax, a.nboot, a.step, a.nproc)
    gq.to_csv(out / "mc_grid_qc.csv", index=False, float_format="%.4f")
    summ = [mm.node_summary(g, "all")]
    for per, (y0, y1) in mm.PERIODS.items():
        gp = mm.mc_grid(C[C.year.between(y0, y1)].reset_index(drop=True), a.n_period, a.rmax,
                        a.nboot, a.step, a.nproc)
        gp.to_csv(out / f"mc_grid_{per}.csv", index=False, float_format="%.4f")
        summ.append(mm.node_summary(gp, per))
        print(f"grid {per}:", int(gp.mc_best.notna().sum()), "nodes", flush=True)
    pd.concat(summ).to_csv(out / "mc_nodes_by_region.csv", index=False, float_format="%.3f")
    # same grid on version B (v4.1 M, v5.1 epicentres) for a like-for-like difference map
    B = load_v51(False, "M_v4")
    gB = mm.mc_grid(B, a.n, a.rmax, a.nboot, a.step, a.nproc)
    gB.to_csv(out / "mc_grid_v4M_v51loc.csv", index=False, float_format="%.4f")
    A = load_v4(False)
    print(f"A: {len(A):,}  B: {len(B):,}  C: {len(C):,} events", flush=True)

    tabs = []
    for ver, d in (("A_v4M_v4loc", A), ("B_v4M_v51loc", B), ("C_v51M_v51loc", C)):
        t = regional(d, a.nboot, order=mm.REG_ORDER)
        t.insert(0, "version", ver)
        tabs.append(t)
    tq = regional(Cq, a.nboot, order=mm.REG_ORDER, periods=False)
    tq.insert(0, "version", "C_qc")
    tabs.append(tq)
    pd.concat(tabs).to_csv(out / "mc_b_regional.csv", index=False, float_format="%.4f")
    tabs = []
    for ver, d in (("B_v4M_v51loc", B), ("C_v51M_v51loc", C)):
        t = regional(d, a.nboot, key="catalog_region", periods=False)
        t.insert(0, "version", ver)
        tabs.append(t)
    pd.concat(tabs).to_csv(out / "mc_b_catalog_region.csv", index=False, float_format="%.4f")
    (out / "run_params.json").write_text(json.dumps(meta, indent=2))


MSH_BOX = (-122.4, -122.0, 46.0, 46.4)   # lon0, lon1, lat0, lat1


def _b_fixed(mag_bin, mc):
    """Aki-Utsu b above a fixed Mc on 0.1-binned magnitudes (same formula as mc_counts)."""
    x = mag_bin[mag_bin >= mc - 1e-9]
    if len(x) < mm.NMIN_B:
        return np.nan, len(x)
    return 1.0 / (mm.LN10 * (x.mean() - (mc - mm.DM / 2))), len(x)


def paired(a, nboot=500):
    from scipy.stats import norm
    out = Path(a.out)
    reg = pd.read_csv(out / "mc_b_regional.csv")
    reg = reg[reg.period == "all"]
    mcC = reg[reg.version == "C_v51M_v51loc"].set_index("region").mc_best
    mcB = reg[reg.version == "B_v4M_v51loc"].set_index("region").mc_best
    d = _MAGS[_MAGS.loc_source.isin(["nll3d", "comcat"]) & _MAGS.is_earthquake.astype(bool)
              & _MAGS.M.notna() & _MAGS.M_v4.notna()].copy()
    d["box"] = _regions(d.lon.values, d.lat.values)
    d["msh"] = d.lon.between(MSH_BOX[0], MSH_BOX[1]) & d.lat.between(MSH_BOX[2], MSH_BOX[3])
    m5 = np.round(d.M.to_numpy() / mm.DM) * mm.DM
    m4 = np.round(d.M_v4.to_numpy() / mm.DM) * mm.DM
    groups = [(r, d.box.eq(r).to_numpy(), r) for r in mm.REG_ORDER]
    groups += [("OFF", d.box.isin(mm.OFFSHORE).to_numpy(), "OFF"), ("ALL", np.ones(len(d), bool), "ALL"),
               ("ON without MSH", (d.box.eq("ON") & ~d.msh).to_numpy(), "ON"),
               ("MSH only", (d.box.eq("ON") & d.msh).to_numpy(), None)]
    rng = np.random.default_rng(20261125)
    rows = []
    for name, sel, mc_from in groups:
        x5, x4 = m5[sel], m4[sel]
        if mc_from is None:   # Mc_best of C for this subset
            ib = np.clip(np.round((x5 - mm.MLO) / mm.DM).astype(int), 0, len(mm.MBINS) - 1)
            mcs = [("C", round(mm.mc_with_bootstrap(ib, 0, 0)["mc_best"], 1))]
        else:
            mcs = [("C", round(float(mcC[mc_from]), 1))]
            if abs(mcB[mc_from] - mcC[mc_from]) > 0.05:
                mcs.append(("B", round(float(mcB[mc_from]), 1)))
        for src, mc in mcs:
            b4, n4 = _b_fixed(x4, mc)
            b5, n5 = _b_fixed(x5, mc)
            db = []
            for _ in range(nboot):
                i = rng.integers(0, len(x5), len(x5))
                db.append(_b_fixed(x5[i], mc)[0] - _b_fixed(x4[i], mc)[0])
            sd = float(np.nanstd(db))
            z = (b5 - b4) / sd if sd > 0 else np.nan
            rows.append(dict(box=name, mc_source=f"Mc_best {src}", mc=mc, n_events=int(sel.sum()),
                             n_above_v41=n4, n_above_v51=n5, b_v41=b4, b_v51=b5, db=b5 - b4,
                             sd_paired=sd, z=z, p_two_sided=2 * norm.sf(abs(z))))
    r = pd.DataFrame(rows)
    main = r.box.isin(mm.REG_ORDER + ["OFF", "ALL"]) & r.mc_source.eq("Mc_best C")
    ntest = int(main.sum())
    zcrit = float(norm.isf(0.05 / 2 / ntest))
    r["bonferroni_n_tests"] = ntest
    r["bonferroni_z_crit"] = zcrit
    r["significant_bonferroni"] = r.z.abs() > zcrit
    r.to_csv(out / "paired_b.csv", index=False, float_format="%.4f")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def plot(a):
    import pygmt
    import map_context as mcx
    out = Path(a.out)
    mm.OUT = out
    mcx.PB2002 = str(mm.ROOT / "data/tectonics/PB2002_steps.json")
    pygmt.config(FONT_ANNOT_PRIMARY="6p", FONT_LABEL="7p", MAP_FRAME_TYPE="plain",
                 MAP_TICK_LENGTH_PRIMARY="0.06c", MAP_FRAME_PEN="0.5p", FORMAT_GEO_MAP="ddd")
    g = pd.read_csv(out / "mc_grid.csv")
    gB = pd.read_csv(out / "mc_grid_v4M_v51loc.csv")
    sta = mm._stations()
    cpt = out / "mc.cpt"
    pygmt.makecpt(cmap="batlow", series=[0.0, 3.0, 0.1], continuous=True, output=str(cpt))

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=200)
        print("wrote", name)

    fig = pygmt.Figure()
    panels = [("mc_maxc", "a) MAXC + 0.2"), ("mc_mbs", "b) MBS"), ("mc_best", "c) GFT 95/90 (W&W 2005 rule)")]
    for i, (col, t) in enumerate(panels):
        if i:
            fig.shift_origin(xshift="5.5c")
        mm._map_panel(fig, mm._grid_da(g, col), str(cpt), t, sta, frame_left=(i == 0))
    fig.colorbar(cmap=str(cpt), position="JBC+w8c/0.22c+h+o-5.5c/0.7c", frame=["xa0.5f0.1+lMc (v5.1 M)"])
    save(fig, "fig1_mc_three_methods_v51")

    fig = pygmt.Figure()
    bcpt = out / "b.cpt"
    pygmt.makecpt(cmap="vik", series=[0.6, 1.4, 0.05], continuous=True, output=str(bcpt))
    mm._map_panel(fig, mm._grid_da(g, "b"), str(bcpt), "a) b above Mc (v5.1)", sta)
    fig.colorbar(cmap=str(bcpt), position="JBC+w4.5c/0.2c+h+o0/0.7c", frame=["xa0.2f0.1+lb"])
    fig.shift_origin(xshift="5.9c")
    dcpt = out / "dmc.cpt"
    pygmt.makecpt(cmap="vik", series=[-0.5, 0.5, 0.05], continuous=True, output=str(dcpt))
    gd = g[["lon", "lat"]].assign(d=g.mc_best.values - gB.mc_best.values)
    mm._map_panel(fig, mm._grid_da(gd, "d"), str(dcpt), "b) Mc(v5.1 M) - Mc(v4.1 M), v5.1 epicentres",
                  sta, frame_left=False)
    fig.colorbar(cmap=str(dcpt), position="JBC+w4.5c/0.2c+h+o0/0.7c", frame=["xa0.2f0.1+ldelta Mc"])
    fig.shift_origin(xshift="5.9c")
    gb = g[["lon", "lat"]].assign(d=g.b.values - gB.b.values)
    mm._map_panel(fig, mm._grid_da(gb, "d"), str(dcpt), "c) b(v5.1 M) - b(v4.1 M)", sta, frame_left=False)
    fig.colorbar(cmap=str(dcpt), position="JBC+w4.5c/0.2c+h+o0/0.7c", frame=["xa0.2f0.1+ldelta b"])
    save(fig, "fig2_b_and_change_v51")

    fig = pygmt.Figure()
    for i, p in enumerate(mm.PERIODS):
        if i:
            fig.shift_origin(xshift="5.5c")
        gp = pd.read_csv(out / f"mc_grid_{p}.csv")
        mm._map_panel(fig, mm._grid_da(gp, "mc_best"), str(cpt), p, mm._stations(p), frame_left=(i == 0))
    fig.colorbar(cmap=str(cpt), position="JBC+w8c/0.22c+h+o-8.25c/0.7c",
                 frame=["xa0.5f0.1+lMc best (GFT95/90, else MAXC)"])
    save(fig, "fig3_mc_periods_v51")


def main():
    global _MAGS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mags", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=["compute", "plot", "paired", "all"], default="all")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--n-period", type=int, default=150)
    p.add_argument("--rmax", type=float, default=50.0)
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--nproc", type=int, default=14)
    a = p.parse_args()
    _MAGS = pd.read_csv(a.mags, low_memory=False).rename(columns={"region": "catalog_region"})
    if a.stage in ("compute", "all"):
        compute(a)
    if a.stage in ("plot", "all"):
        plot(a)
    if a.stage in ("paired", "all"):
        paired(a)


if __name__ == "__main__":
    main()
