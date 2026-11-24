"""Track B, step B1b: balanced travel-time dataset from well-recorded stations (D-B022).

Input: the B1 dataset (<base>/b1_dataset: picks.csv, events.csv, stations.csv). No pick is re-located or
re-measured; B1b only selects stations and assigns weights. Parameters: params.yaml `balance`.

1. Core stations. A station is kept if its B1 P picks
     - number at least min_P_land (land) or min_P_obs (OBS: deployments of ~1 year), and
     - cover at least min_sectors (land) / min_sectors_obs (OBS) back-azimuth sectors of sector_deg with >= min_per_bin picks each, and
     - cover at least min_dist_bins of the distance bins dist_edges_km with >= min_per_bin picks each,
   or if its S picks pass the same three tests (some OBS have many more S than P picks).
   Picks at other stations are dropped (reason `station`). Both phases of a core station are kept.
   The `fail` column reports the P-test failure.
2. Path-redundancy weight. Per station and phase, picks are binned by epicentral distance (bin_dist_km)
   and back-azimuth (bin_az_deg); w_bin = min(1, K / n_bin). Nothing is dropped: a dense bin keeps all
   its picks for the relocation but counts as K picks in the misfit.
3. Region weight. Each event region (events.csv `region`) gets w_region = target share / share of the
   sum of w_bin, with equal target shares over the regions listed in `balance.regions`, clipped to
   region_w_clip, then rescaled so the mean pick weight is 1. Regions not listed keep w_region = 1.
4. Final balance weight w_bal = w_bin * w_region, multiplied later by the pick-quality weight
   (D-B008: (0.05/sigma)^2) when the tomoDD phase file is written.
5. Events: an event is kept if it still has >= post_min_picks picks and gap <= post_max_gap_deg on
   core stations.
6. Ridge regions (D-B023; events selected in B1 by the softer ridge rule, events.csv `rule`): a station
   with >= ridge.min_picks picks of ridge events is kept for those picks only (an OBS for all its picks if
   ridge.obs_keep_all, D-B035), with no coverage test;
   ridge events need ridge.post_min_picks picks and gap <= ridge.post_max_gap_deg; the three ridge regions
   share one weight group (`region_groups`).

Outputs in <base>/b1b_balanced: picks.csv (B1 columns + baz_deg, w_bin, w_region, w_bal), events.csv,
stations_core.csv (all B1 stations with coverage numbers and keep flag), picks_dropped.csv,
balance_metrics.csv; figures and b1b_tables.md in data/tomo_v1/b1b_balanced.

Run: LD_LIBRARY_PATH=.pixi/envs/default/lib .pixi/envs/default/bin/python workflow/10_tomography/b1b_balance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params, write_manifest  # noqa: E402
from b1_select_events import gap_deg, md_table, rounded  # noqa: E402

P = load_params()
B = P["balance"]
BASE = Path(P["paths"]["base"])
IN = BASE / P["paths"].get("b1_dir", "b1_dataset")           # D-B035: versioned (b1_dataset_c, b1b_balanced_c)
OUT = BASE / P["paths"].get("b1b_dir", "b1b_balanced")
FIG = REPO / "data/tomo_v1" / P["paths"].get("b1b_dir", "b1b_balanced")
LOG = []
GROUP = {r: g for g, rs in B["region_groups"].items() for r in rs}   # region -> weight group (D-B023)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() > 0 else np.nan


def back_azimuth(elat, elon, slat, slon):
    """Azimuth from station to event, degrees clockwise from north."""
    la1, lo1, la2, lo2 = map(np.radians, (elat, elon, slat, slon))
    return np.mod(np.degrees(np.arctan2(np.sin(lo1 - lo2) * np.cos(la1),
                                        np.cos(la2) * np.sin(la1) - np.sin(la2) * np.cos(la1) * np.cos(lo1 - lo2))),
                  360)


def load():
    pk = pd.read_csv(IN / "picks.csv")
    ev = pd.read_csv(IN / "events.csv")
    st = pd.read_csv(IN / "stations.csv")
    if "rule" not in ev:                                  # B1 built before D-B023
        ev["rule"] = "main"
    pk = pk.merge(ev[["ID", "lat", "lon", "region", "rule"]], left_on="event_ID", right_on="ID", how="left")
    pk["group"] = pk.region.map(GROUP).fillna(pk.region)
    assert pk.ID.notna().all()
    pk["sta"] = pk.network + "." + pk.station
    st["sta"] = st.network + "." + st.station
    pk["baz_deg"] = back_azimuth(pk.lat, pk.lon, pk.sta_lat, pk.sta_lon)
    return pk, ev, st


def core_stations(pk, st):
    """Coverage test per phase on main-rule picks; a station is core if its P picks or its S picks pass.
    n_P / n_S are overwritten with main-rule counts."""
    reg = pk.groupby(["sta", "region"]).size().unstack(fill_value=0)
    s = st.set_index("sta")
    ok = {}
    pm = pk[pk.rule == "main"]                           # main test on main-rule picks only (D-B023)
    for ph, n_col in (("P", "n_P"), ("S", "n_S")):
        p = pm[pm.phase == ph]
        s[n_col] = p.groupby("sta").size().reindex(s.index).fillna(0).astype(int)
        sec = (p.baz_deg // B["sector_deg"]).astype(int)
        dbin = np.digitize(p.epi_dist_km, B["dist_edges_km"])
        s[f"n_sectors_{ph}"] = (p.groupby(["sta", sec]).size() >= B["min_per_bin"]).groupby(level=0).sum() \
            .reindex(s.index).fillna(0).astype(int)
        s[f"n_dist_bins_{ph}"] = (p.groupby(["sta", dbin]).size() >= B["min_per_bin"]).groupby(level=0).sum() \
            .reindex(s.index).fillna(0).astype(int)
        minN = np.where(s.is_obs, B["min_P_obs"], B["min_P_land"])
        minSec = np.where(s.is_obs, B["min_sectors_obs"], B["min_sectors"])
        ok[ph] = (s[n_col] >= minN) & (s[f"n_sectors_{ph}"] >= minSec) & \
            (s[f"n_dist_bins_{ph}"] >= B["min_dist_bins"])
        s[f"pass_{ph}"] = ok[ph]
    s["main_region"] = reg.idxmax(1).reindex(s.index)
    s["main_region_frac"] = (reg.max(1) / reg.sum(1)).reindex(s.index).round(3)
    s["core"] = ok["P"] | ok["S"]
    minP = np.where(s.is_obs, B["min_P_obs"], B["min_P_land"])
    s["n_sectors"], s["n_dist_bins"] = s.n_sectors_P, s.n_dist_bins_P
    s["fail"] = ""
    s.loc[s.n_P < minP, "fail"] += "nP "
    s.loc[s.n_sectors < np.where(s.is_obs, B["min_sectors_obs"], B["min_sectors"]), "fail"] += "azimuth "
    s.loc[s.n_dist_bins < B["min_dist_bins"], "fail"] += "distance"
    s["fail"] = s.fail.str.strip()
    # stations kept only because of the lower OBS bar (would fail the land bar)
    s["obs_bar_only"] = s.core & s.is_obs & (np.maximum(s.n_P, s.n_S) < B["min_P_land"])
    s["S_only"] = s.core & ~s.pass_P
    # D-B023: softer bar on ridge-event picks only; a station that passes only this keeps only its ridge picks
    nr = pk[pk.rule == "ridge"].groupby("sta").size().reindex(s.index).fillna(0).astype(int)
    s["n_ridge_picks"] = nr
    s["ridge_core"] = nr >= B["ridge"]["min_picks"]
    s["ridge_only"] = s.ridge_core & ~s.core
    return s.reset_index()


def weights(k):
    k = k.copy()
    k["rbin"] = (k.epi_dist_km // B["bin_dist_km"]).astype(int)
    k["abin"] = (k.baz_deg // B["bin_az_deg"]).astype(int)
    n = k.groupby(["sta", "phase", "rbin", "abin"]).ID.transform("size")
    k["n_bin"] = n
    k["w_bin"] = np.minimum(1.0, B["K_per_bin"] / n)
    share = k.groupby("group").w_bin.sum() / k.w_bin.sum()
    groups = list(B["regions"]) + list(B["region_groups"])
    tgt = 1.0 / len(groups)
    wr = pd.Series(1.0, index=share.index)
    for r in groups:
        if r in share.index:
            wr[r] = tgt / share[r]
    lo, hi = B["region_w_clip"]
    wr = wr.clip(lo, hi)
    k["w_region"] = k.group.map(wr)
    k["w_bal"] = k.w_bin * k.w_region
    k["w_bal"] /= k.w_bal.mean()
    return k.drop(columns=["rbin", "abin"]), wr


def metrics(name, d, w):
    c = d.assign(w=w).groupby("sta").w.sum()
    r = d.assign(w=w).groupby("group").w.sum() / w.sum() * 100
    row = {"set": name, "picks": len(d), "kish_n_eff": round(float(w.sum() ** 2 / (w ** 2).sum())),
           "stations": d.sta.nunique(), "events": d.ID.nunique(),
           "gini_station": round(gini(c), 3), "top50_station_share": round(c.nlargest(50).sum() / c.sum(), 3),
           "obs_share": round(float((w * d.is_obs).sum() / w.sum()), 3),
           "S_share": round(float((w * (d.phase == "S")).sum() / w.sum()), 3)}
    row.update({f"pct_{k}": round(v, 1) for k, v in r.items()})
    return row


def event_table(ev, k):
    g = k.groupby("ID")
    n = k.groupby("ID").phase.value_counts().unstack(fill_value=0)
    e = ev.set_index("ID")
    e["nP_b"] = n.get("P", pd.Series(dtype=int)).reindex(e.index).fillna(0).astype(int)
    e["nS_b"] = n.get("S", pd.Series(dtype=int)).reindex(e.index).fillna(0).astype(int)
    e["gap_b"] = pd.Series({i: gap_deg(e.at[i, "lat"], e.at[i, "lon"], d.sta_lat.values, d.sta_lon.values)
                            for i, d in g}).reindex(e.index).fillna(360.0)
    e["w_event"] = g.w_bal.sum().reindex(e.index).fillna(0).round(3)
    rr = e.rule == "ridge"
    e["keep"] = ((e.nP_b + e.nS_b) >= np.where(rr, B["ridge"]["post_min_picks"], B["post_min_picks"])) & \
        (e.gap_b <= np.where(rr, B["ridge"]["post_max_gap_deg"], B["post_max_gap_deg"]))
    return e.reset_index()


def figures(st, k0, k, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. map of stations: core / soft (ridge) bar / dropped, sized by effective picks. Dropped = no pick left in
    # the dataset; ~core alone also counted the soft-bar stations, which keep picks (all of them for OBS, D-B035)
    eff = k.groupby("sta").w_bal.sum()
    kept = st.sta.isin(k.sta)
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    ax.scatter(ev.lon, ev.lat, s=0.3, c="0.75", rasterized=True, label="B1 events")
    d = st[~kept]
    ax.scatter(d.lon, d.lat, s=12, marker="x", c="tab:red", lw=0.8, label=f"dropped, no picks ({len(d)})")
    obs_soft = "all picks" if B["ridge"].get("obs_keep_all") else "ridge picks"
    for cls, ob, m, col, lab in (("core", False, "^", "tab:blue", "core land"),
                                 ("core", True, "v", "tab:green", "core OBS"),
                                 ("ridge_only", False, "^", "tab:purple", "soft bar land (ridge picks)"),
                                 ("ridge_only", True, "v", "tab:orange", f"soft bar OBS ({obs_soft})")):
        c = st[st[cls] & kept & (st.is_obs == ob)]
        ax.scatter(c.lon, c.lat, s=4 + 60 * np.sqrt(eff.reindex(c.sta).fillna(0) / eff.max()), marker=m,
                   c=col, ec="k", lw=0.3, label=f"{lab} ({len(c)})")
    ax.set_aspect(1 / np.cos(np.radians(45)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("B1b stations (size: effective picks after weighting)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "b1b_stations.png", dpi=150)
    plt.close(fig)

    # 2. picks per station, raw vs effective (Lorenz curves)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for lab, c in (("B1 (all stations)", k0.groupby("sta").size()),
                   ("B1b core, unweighted", k.groupby("sta").size()),
                   ("B1b core, w_bin", k.groupby("sta").w_bin.sum()),
                   ("B1b core, w_bal", k.groupby("sta").w_bal.sum())):
        x = np.sort(c.values)[::-1]
        axs[0].plot(np.arange(1, len(x) + 1), x, label=f"{lab} (Gini {gini(x):.2f})", lw=1)
        y = np.cumsum(np.sort(c.values)) / c.sum()
        axs[1].plot(np.linspace(0, 1, len(y)), y, lw=1, label=lab)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("station rank")
    axs[0].set_ylabel("picks (or summed weight)")
    axs[0].legend(fontsize=7)
    axs[1].plot([0, 1], [0, 1], "k:", lw=0.6)
    axs[1].set_xlabel("fraction of stations")
    axs[1].set_ylabel("fraction of picks / weight")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "b1b_station_balance.png", dpi=150)
    plt.close(fig)

    # 3. region shares
    groups = list(B["regions"]) + list(B["region_groups"])
    regs = groups + ["other"]
    def shares(d, w):
        s = d.assign(w=w, r=d.group.where(d.group.isin(groups), "other")).groupby("r").w.sum()
        return (s / s.sum() * 100).reindex(regs).fillna(0)
    sets = {"B1": shares(k0, np.ones(len(k0))), "core": shares(k, np.ones(len(k))),
            "w_bin": shares(k, k.w_bin), "w_bal": shares(k, k.w_bal)}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(regs))
    for i, (lab, s) in enumerate(sets.items()):
        ax.bar(x + (i - 1.5) * 0.2, s.values, width=0.2, label=lab)
    ax.axhline(100 / len(groups), c="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(regs, fontsize=7, rotation=20)
    ax.set_ylabel("% of picks / weight")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "b1b_region_shares.png", dpi=150)
    plt.close(fig)


def main():
    pk, ev, st = load()
    log(f"B1 input: {len(pk)} picks, {pk.ID.nunique()} events, {pk.sta.nunique()} stations")
    s = core_stations(pk, st)
    core = set(s.loc[s.core, "sta"])
    rcore = set(s.loc[s.ridge_only, "sta"])
    log(f"core stations: {len(core)} of {len(s)} (OBS {int((s.core & s.is_obs).sum())}, "
        f"land {int((s.core & ~s.is_obs).sum())}); admitted only by the OBS bar: {int(s.obs_bar_only.sum())}; "
        f"only by the S test: {int(s.S_only.sum())}")
    use = pk.sta.isin(core) | (pk.sta.isin(rcore) & (pk.rule == "ridge"))
    if B["ridge"].get("obs_keep_all"):   # D-B035: an OBS admitted by the soft bar keeps all its picks, not only soft-rule ones
        robs = set(s.loc[s.ridge_only & s.is_obs, "sta"])
        use |= pk.sta.isin(robs)
        log(f"D-B035: {len(robs)} soft-bar OBS keep all their picks (+{int((pk.sta.isin(robs) & (pk.rule != 'ridge')).sum())})")
    log(f"ridge-only stations (>= {B['ridge']['min_picks']} ridge picks, no coverage test): {len(rcore)}")
    k = pk[use]
    dropped = pk[~use].assign(reason="station")
    k, wr = weights(k)
    log("region weights:", {r: round(v, 3) for r, v in wr.items()})
    e = event_table(ev, k)
    keep_ev = set(e.loc[e.keep, "ID"])
    dropped = pd.concat([dropped, k[~k.ID.isin(keep_ev)].assign(reason="event")])
    k = k[k.ID.isin(keep_ev)].copy()
    k["w_bal"] /= k.w_bal.mean()
    log(f"events kept: {len(keep_ev)} of {len(ev)}; picks kept {len(k)}; dropped {len(dropped)} "
        f"(station {int((dropped.reason == 'station').sum())}, event {int((dropped.reason == 'event').sum())})")

    m = pd.DataFrame([metrics("B1", pk, pd.Series(1.0, index=pk.index)),
                      metrics("B1b core, unweighted", k, pd.Series(1.0, index=k.index)),
                      metrics("B1b w_bin", k, k.w_bin),
                      metrics("B1b w_bal", k, k.w_bal)])

    # what the lower OBS bar admits
    ob = s[s.obs_bar_only].groupby("main_region").agg(stations=("sta", "size"), P=("n_P", "sum"), S=("n_S", "sum"))
    # dropped = no pick left in the dataset (~core also counted the soft-bar stations, which keep picks)
    kept = s.sta.isin(k.sta)
    soft = s[s.ridge_only & kept].assign(picks=lambda d: d.sta.map(k.sta.value_counts()))
    soft = soft.groupby(["main_region", "is_obs"]).agg(stations=("sta", "size"), picks_kept=("picks", "sum"))
    fails = s[~kept].groupby(["is_obs", "fail"]).size().rename("stations")
    lost = s[~kept].groupby("main_region").agg(stations=("sta", "size"), P=("n_P", "sum"), S=("n_S", "sum"))
    # events by region
    er = e.groupby(["region", "rule"]).agg(B1=("ID", "size"), kept=("keep", "sum"))
    log(md_table(m, index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    outs = {"picks.csv": rounded(k.drop(columns=["ID", "lat", "lon"])),
            "events.csv": rounded(e[e.keep]),
            "stations_core.csv": rounded(s),
            "picks_dropped.csv": rounded(dropped.drop(columns=["ID", "lat", "lon"])),
            "balance_metrics.csv": m}
    for f, d in outs.items():
        d.to_csv(OUT / f, index=False)
    figures(s, pk, k, ev)
    md = ["# B1b balanced dataset: tables", "", "Written by workflow/10_tomography/b1b_balance.py.", "",
          "## Metrics", "", md_table(m, index=False), "",
          "## Region weights", "", md_table(wr.round(3).rename("w_region").to_frame()), "",
          "## Stations admitted only by the OBS bar (main region of their picks)", "", md_table(ob), "",
          "## Stations kept by the soft (ridge) bar only (OBS: all picks if balance.ridge.obs_keep_all, D-B035)", "",
          md_table(soft), "",
          "## Dropped stations (no picks in the dataset) by reason", "", md_table(fails.to_frame()), "",
          "## Dropped stations (no picks in the dataset) by main region", "", md_table(lost), "",
          "## Events by region", "", md_table(er), "", "## Log", "", "```", *LOG, "```"]
    (FIG / "b1b_tables.md").write_text("\n".join(md))
    write_manifest("b1b_balance", OUT,
                   inputs=[IN / "picks.csv", IN / "events.csv", IN / "stations.csv"],
                   outputs=[OUT / f for f in outs] + [FIG / "b1b_tables.md"],
                   params=B, counts={"core_stations": len(core), "events": len(keep_ev), "picks": len(k)})


if __name__ == "__main__":
    main()
