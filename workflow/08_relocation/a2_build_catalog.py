"""Catalog v5 (Track A) from an NLLoc run: every event of the catalog v4.1 event list, with a location-quality tier.

Events NLLoc located get the 3D maximum-likelihood location (loc_source = "nll3d"; with a deep pass, D-B067, the row
a2_parse_hyp.py --deep chose). Events it could not locate have no hypocentre here (loc_source = "none", tier U; time,
lat, lon, depth empty): a v4 hypocentre is never copied into the catalog columns (D-B055, D-B057: v4 depths have no
surface constraint). Their v4 values stay in the v4_* columns, and u_reason says why NLLoc did not locate them
(N = LOCMETH minimum number of phases of the run's own nlloc.in files; picks counted in the run's events.obs files):
  "rejected: <NLLoc comment>"           NLLoc REJECTED the location (e.g. maximum likelihood on the box edge; other
                                        NLLoc statuses the same way, e.g. "aborted: ...")
  "not in the run's observation files"  the event has no block in any events.obs of the run
  "fewer than N picks"                  the event's observation block has fewer than N picks
  "too few usable arrivals"             N or more picks, but fewer than N usable (e.g. outside the time grids)
region is the box label of the v4 epicentre (a2_parse_hyp.region), for every row, located or not.

Tiers (first one met; z* = max(shallower of 3D and v4 depth, below surface, 5 km); dist_min = nearest
station used, km):
  A  nphs >= 10, gap <= 180, h_max <= 3 km, sig_z <= 3 km, dist_min <= 1 z*, rms <= 0.3 s
  B  nphs >= 8,  gap <= 250, h_max <= 6 km, sig_z <= 6 km, dist_min <= 2 z*, rms <= 0.5 s
  C  h_max <= 15 km, gap <= 300: epicentre usable, depth not reliable
  D  located, but none of the above; also every event located with 5 phases or fewer (one degree of freedom:
     D-B066, five-pick events published as tier D)
  U  not located in 3D: no hypocentre (see u_reason)
h_max is the 68% horizontal semi-major axis; sig_z is sqrt(CovZZ) of the NLLoc PDF. Formal errors leave out
velocity-model error; the dist_min / z* condition is there because depths with no station within
2 z* were 3-11 km deeper than ANSS (iter1 check), and at Mount Hood (nearest station 29 km) three velocity
models gave -1, 8 and 17 km with the same RMS: such depths follow the model, not the data.
Also: depth_constrained = tier in (A, B).

Usage: python a2_build_catalog.py RUN_NAME   (after a2_parse_hyp.py RUN_NAME [--deep DEEP_RUN])
  -> /wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_RUN_NAME.csv and a tier summary
     data/reloc_v5/catalog_v5_trackA_RUN_NAME_tiers.md (incl. ANSS check per tier and the tier U reasons)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from a2_parse_hyp import read_v4, region  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
B = Path("/wd1/mdenolle_data/reloc_v5")
OUT = B / "catalog"                      # catalog CSV
MD = REPO / "data/reloc_v5"              # tier summary
TIERS = {
    "A": dict(nphs=10, gap=180, h_max=3, sig_z=3, ratio=1, rms=0.3),
    "B": dict(nphs=8, gap=250, h_max=6, sig_z=6, ratio=2, rms=0.5),
}


def md_table(df):
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    df = df.reset_index()
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def tier(df):
    # shallower of the 3D and v4 depths (below surface): a depth pushed deep by the model must not
    # make its own nearest-station test pass (Mount Hood: 3D 16.7 km, v4 3.3, ANSS 4.3, nearest station 29 km)
    zstar = np.maximum(np.minimum(df.depth, df.v4_depth) + df.surface_km, 5.0)
    ratio = df.dist_min / zstar
    t = np.full(len(df), "D", dtype=object)
    c = (df.h_max <= 15) & (df.gap <= 300)
    t[c] = "C"
    for name in ("B", "A"):
        p = TIERS[name]
        m = ((df.nphs >= p["nphs"]) & (df.gap <= p["gap"]) & (df.h_max <= p["h_max"]) &
             (df.sig_z <= p["sig_z"]) & (ratio <= p["ratio"]) & (df.rms <= p["rms"]))
        t[m] = name
    t[(df.nphs <= 5).to_numpy()] = "D"            # one degree of freedom (D-B066)
    return t, ratio


def main_workdirs(rd):
    """Workdirs of the run (one nlloc.in each)."""
    w = sorted(f.parent for f in rd.glob("*/nlloc.in"))
    assert w, f"no */nlloc.in under {rd}"
    return w


def min_phases(rd):
    """LOCMETH minimum number of phases (3rd field) of the run's own nlloc.in files (one value for the run)."""
    n = set()
    for w in main_workdirs(rd):
        lm = [l.split() for l in (w / "nlloc.in").read_text().splitlines() if l.startswith("LOCMETH")]
        assert len(lm) == 1, f"{w}/nlloc.in: {len(lm)} LOCMETH lines"
        n.add(int(lm[0][3]))
    assert len(n) == 1, f"{rd}: LOCMETH minimum differs between workdirs: {sorted(n)}"
    return n.pop()


def obs_picks(rd):
    """Picks per event in the run's events.obs files (event blocks 'PUBLIC_ID v4_<ID>', then arrivals, blank line)."""
    cnt, cur = {}, None
    for w in main_workdirs(rd):
        for line in open(w / "events.obs"):
            s = line.strip()
            if s.startswith("PUBLIC_ID"):
                cur = int(s.split()[1].split("_")[1])
                cnt[cur] = cnt.get(cur, 0)
            elif not s:
                cur = None
            elif cur is not None and not s.startswith("#"):
                cnt[cur] += 1
    return pd.Series(cnt, dtype=int)


def main(run):
    rd = B / "runs" / run
    allrows = pd.read_csv(rd / "locations.csv")
    assert allrows.ID.is_unique, f"{rd}/locations.csv: several rows per event (rerun a2_parse_hyp.py)"
    loc = allrows[allrows.status == "LOCATED"].copy()
    loc["tier"], loc["dist_over_depth"] = tier(loc)
    loc["loc_source"] = "nll3d"
    v4 = read_v4()
    cols = ["ID", "time", "lat", "lon", "depth", "exp_depth", "loc_source", "tier", "nphs", "n_sta", "gap",
            "rms", "h_min", "h_max", "sig_z", "ell_len3", "dist_min", "dist_med", "dist_max",
            "dist_over_depth", "surface_km", "above_surface", "shift_h_km", "shift_z_km", "tile"]
    cat = v4[["ID", "raw_ID", "v4_time", "v4_lat", "v4_lon", "v4_depth", "v4_npick", "v4_sigma_m",
              "magnitude"]].merge(loc[cols], on="ID", how="left")
    out_list = sorted(set(loc.ID) - set(cat.ID))
    if out_list:
        print(f"{len(out_list)} located events are not in the v4.1 event list and are left out: {out_list[:10]}")
    u = cat.loc_source.isna()
    cat.loc[u, "loc_source"] = "none"          # no hypocentre: never the v4 one (docstring)
    cat.loc[u, "tier"] = "U"
    # why not located (docstring)
    n = min_phases(rd)
    npk = obs_picks(rd)
    nl = allrows[allrows.status != "LOCATED"].set_index("ID")
    com = nl.status.str.lower() + ": " + nl.comment.fillna("").astype(str)   # e.g. "rejected: WARNING: max prob ..."
    k = cat.ID.map(npk)
    why = np.select([cat.ID.isin(com.index), k.isna(), k < n],
                    [cat.ID.map(com).fillna(""), "not in the run's observation files", f"fewer than {n} picks"],
                    "too few usable arrivals")
    cat["u_reason"] = np.where(u, why, "")
    cat["region"] = [region(a, b) for a, b in zip(cat.v4_lat, cat.v4_lon)]   # v4 epicentre, every row
    cat = cat.rename(columns={"magnitude": "v4_magnitude"})
    OUT.mkdir(exist_ok=True)
    out = OUT / f"catalog_v5_trackA_{run}.csv"
    cat.to_csv(out, index=False, float_format="%.5g")

    # summary + ANSS check per tier
    lines = [f"# Catalog v5 Track A ({run}): location-quality tiers", "",
             f"{len(cat)} events (catalog v4.1 event list). File: `{out}`", "",
             "Tier definitions: see `workflow/08_relocation/a2_build_catalog.py`.", ""]
    tab = cat.groupby("tier").agg(n=("ID", "size"), rms=("rms", "median"), nphs=("nphs", "median"),
                                  h_max=("h_max", "median"), sig_z=("sig_z", "median"),
                                  depth=("depth", "median"))
    tab["share"] = (tab.n / len(cat) * 100).round(1)
    lines += ["## Tiers", "", md_table(tab.round(2)), ""]
    lines += ["## Tiers by region (counts)", "",
              md_table(pd.crosstab(cat.region, cat.tier, margins=True)), ""]
    lines += [f"## Not located (tier U): why (LOCMETH minimum {n} phases, from the run's nlloc.in)", "",
              md_table(cat[u].u_reason.str.replace(r"rejected: .*boundary (\d+).*", r"rejected: max likelihood on box "
                                                   r"face \1", regex=True).value_counts().rename("n").to_frame()), ""]
    anss = rd / "anss_matched.csv"
    if anss.exists():
        m = pd.read_csv(anss)[["ID", "a_lat", "a_lon", "a_depth", "net"]].merge(cat, on="ID")
        m["dz"] = m.depth - m.a_depth
        m["dz_v4"] = m.v4_depth - m.a_depth
        m["dh"] = np.hypot((m.lat - m.a_lat) * 111.2, (m.lon - m.a_lon) * 111.2 * np.cos(np.radians(m.lat)))
        on = m[m.region.isin(["Cascades", "Puget/Georgia", "Other onshore", "Mendocino", "Offshore forearc"])]
        a = on.groupby("tier").agg(n=("ID", "size"), dz_median=("dz", "median"),
                                   abs_dz=("dz", lambda s: s.abs().median()),
                                   abs_dz_v4=("dz_v4", lambda s: s.abs().median()),
                                   dh_median=("dh", "median"))
        lines += ["## Check against ANSS/ComCat (onshore, Mendocino, offshore forearc)", "",
                  "dz = this catalog minus ANSS depth (km); abs_dz_v4 = same for catalog v4.", "",
                  md_table(a.round(2)), ""]
    md = MD / f"catalog_v5_trackA_{run}_tiers.md"
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
