"""Catalog v5 (Track A) from an NLLoc run: every catalog v4 event, with a location-quality tier.

Events NLLoc located get the 3D maximum-likelihood location (loc_source = "nll3d"). Events it could not
locate (fewer than 6 usable phases once stations beyond ~150-175 km drop out, or rejected) keep their v4
location (loc_source = "v4") and tier U.

Tiers (first one met; z* = max(shallower of 3D and v4 depth, below surface, 5 km); dist_min = nearest
station used, km):
  A  nphs >= 10, gap <= 180, h_max <= 3 km, sig_z <= 3 km, dist_min <= 1 z*, rms <= 0.3 s
  B  nphs >= 8,  gap <= 250, h_max <= 6 km, sig_z <= 6 km, dist_min <= 2 z*, rms <= 0.5 s
  C  h_max <= 15 km, gap <= 300: epicentre usable, depth not reliable
  D  located, but none of the above
  U  not relocated in 3D; v4 location kept
h_max is the 68% horizontal semi-major axis; sig_z is sqrt(CovZZ) of the NLLoc PDF. Formal errors leave out
velocity-model error; the dist_min / z* condition is there because depths with no station within
2 z* were 3-11 km deeper than ANSS (iter1 check), and at Mount Hood (nearest station 29 km) three velocity
models gave -1, 8 and 17 km with the same RMS: such depths follow the model, not the data.
Also: depth_constrained = tier in (A, B).

Usage: python a2_build_catalog.py RUN_NAME
  -> /wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_RUN_NAME.csv and a tier summary
     data/reloc_v5/catalog_v5_trackA_RUN_NAME_tiers.md (incl. ANSS check per tier)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
B = Path("/wd1/mdenolle_data/reloc_v5")
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
    return t, ratio


def main(run):
    rd = B / "runs" / run
    loc = pd.read_csv(rd / "locations.csv")
    loc = loc[loc.status == "LOCATED"].drop_duplicates("ID")
    loc["tier"], loc["dist_over_depth"] = tier(loc)
    loc["loc_source"] = "nll3d"
    v4 = pd.read_csv(REPO / "data/catalog_v4/Cascadia_relocated_catalog_v4.csv", skipinitialspace=True)
    v4 = v4.rename(columns={"datetime": "v4_time", "latitude": "v4_lat", "longitude": "v4_lon",
                            "depth(km)": "v4_depth", "spatial_sigma(m)": "v4_sigma_m"})
    v4["v4_npick"] = v4.Num_p_picks + v4.Num_s_picks
    cols = ["ID", "time", "lat", "lon", "depth", "exp_depth", "loc_source", "tier", "nphs", "n_sta", "gap",
            "rms", "h_min", "h_max", "sig_z", "ell_len3", "dist_min", "dist_med", "dist_max",
            "dist_over_depth", "surface_km", "above_surface", "shift_h_km", "shift_z_km", "tile"]
    cat = v4[["ID", "raw_ID", "v4_time", "v4_lat", "v4_lon", "v4_depth", "v4_npick", "v4_sigma_m",
              "magnitude"]].merge(loc[cols], on="ID", how="left")
    u = cat.loc_source.isna()
    cat.loc[u, "loc_source"] = "v4"
    cat.loc[u, "tier"] = "U"
    cat.loc[u, "time"] = cat.loc[u, "v4_time"].str.rstrip("Z")
    for a, b in (("lat", "v4_lat"), ("lon", "v4_lon"), ("depth", "v4_depth")):
        cat.loc[u, a] = cat.loc[u, b]
    # region label (same boxes as the parser)
    sys.path.insert(0, str(Path(__file__).parent))
    from a2_parse_hyp import region
    cat["region"] = [region(a, b) for a, b in zip(cat.v4_lat, cat.v4_lon)]
    cat = cat.rename(columns={"magnitude": "v4_magnitude"})
    (B / "catalog").mkdir(exist_ok=True)
    out = B / f"catalog/catalog_v5_trackA_{run}.csv"
    cat.to_csv(out, index=False, float_format="%.8g")

    # summary + ANSS check per tier
    lines = [f"# Catalog v5 Track A ({run}): location-quality tiers", "",
             f"{len(cat)} events (all of catalog v4). File: `{out}`", "",
             "Tier definitions: see `workflow/08_relocation/a2_build_catalog.py`.", ""]
    tab = cat.groupby("tier").agg(n=("ID", "size"), rms=("rms", "median"), nphs=("nphs", "median"),
                                  h_max=("h_max", "median"), sig_z=("sig_z", "median"),
                                  depth=("depth", "median"))
    tab["share"] = (tab.n / len(cat) * 100).round(1)
    lines += ["## Tiers", "", md_table(tab.round(2)), ""]
    lines += ["## Tiers by region (counts)", "",
              md_table(pd.crosstab(cat.region, cat.tier, margins=True)), ""]
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
    md = REPO / f"data/reloc_v5/catalog_v5_trackA_{run}_tiers.md"
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
