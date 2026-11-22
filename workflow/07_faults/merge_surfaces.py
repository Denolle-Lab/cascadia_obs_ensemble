"""Merge fault surfaces into one (refit as a single rectangle) and update every v0 file.

    python workflow/07_faults/merge_surfaces.py BL003a BL013a --reason "deeper band 2 km south: depth-dependent location offset"

The first id is kept; the others are removed (fault_surfaces_v0_removed.csv, with the reason). The kept
surface is refit from the union of the events with the same rectangle fit and uncertainty steps as
fault_surfaces_planes.py; its literature fields are kept and the merge is noted in lit_evidence.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_surfaces_v0 as v0  # noqa: E402
import fault_surfaces_planes as fp  # noqa: E402
from split_strands import trace_row  # noqa: E402

V0 = v0.D / "faults" / "v0"
STEM = "fault_surfaces_v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keep")
    ap.add_argument("merge", nargs="+")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--draws", type=int, default=100)
    a = ap.parse_args()
    ids = [a.keep] + a.merge
    fa = argparse.Namespace(tol_h=1.5, tol_z=3.0, tol_k=1.5, pad=0.5, grid=1.0, draws=a.draws)
    summ = pd.read_csv(V0 / f"{STEM}_summary.csv"); mem = pd.read_csv(V0 / f"{STEM}_members.csv")
    lit = pd.read_csv(V0 / f"{STEM}_literature.csv")
    old = summ.set_index("surface_id").loc[ids]
    c, _ = v0.load(1.0, 1e9)
    c = c[c.region != "AX"].reset_index(drop=True)
    ev = set(mem[mem.surface_id.isin(ids)].event_id)
    idx = np.flatnonzero(c.event_id.isin(ev).to_numpy())
    s = fp.surface_rect(c, idx, fa, v0.cfm(), v0.slab(), v0.relief(), np.random.default_rng(0))
    p = s["props"]
    k0 = old.loc[a.keep]
    p.update(surface_id=a.keep, kind="steep zone (fault candidate)" if p["dip"] >= 45 else "dipping zone",
             cluster_jaccard_median=float(np.nanmin(old.cluster_jaccard_median.astype(float))),
             merged_from=";".join(ids))
    p["stable"] = bool(p["cluster_jaccard_median"] >= 0.5)
    p = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in p.items()}
    for col in ("split_from", "strand_azimuth", "lit_match", "lit_confidence", "lit_class", "lit_new_candidate"):
        if col in old.columns:
            p[col] = k0[col]
    props = dict(version="v0", catalog="Cascadia catalog v4", **p)
    props["normal_xyz"] = [round(float(x), 4) for x in s["normal"]]
    tri = [[[[round(float(s["lon"][i]), 5), round(float(s["lat"][i]), 5), round(float(-s["dep"][i] * 1000), 1)]
             for i in (t0, t1, t2, t0)]] for t0, t1, t2 in s["tris"]]
    feat = dict(type="Feature", properties=props, geometry=dict(type="MultiPolygon", coordinates=tri))
    row = {k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in props.items()}
    # files
    rem = summ[summ.surface_id.isin(a.merge)][["surface_id", "n_events", "lon", "lat", "strike", "dip", "lit_class", "lit_match"]].assign(
        reason=f"merged into {a.keep}: {a.reason}")
    pd.concat([pd.read_csv(V0 / f"{STEM}_removed.csv"), rem]).to_csv(V0 / f"{STEM}_removed.csv", index=False)
    summ = pd.concat([summ[~summ.surface_id.isin(ids)], pd.DataFrame([row])], ignore_index=True)
    summ.to_csv(V0 / f"{STEM}_summary.csv", index=False)
    v = pd.DataFrame({"surface_id": a.keep, "vertex": np.arange(len(s["lon"])), "lon": np.round(s["lon"], 5), "lat": np.round(s["lat"], 5),
                      "depth_km": np.round(s["dep"], 3), "sigma_boot_km": np.round(s["sb"], 3), "sigma_loc_km": np.round(s["sl"], 3),
                      "sigma_method_km": np.round(s["sm"], 3), "sigma_total_km": np.round(s["st"], 3),
                      "constrained": s["constrained"].astype(int), "surface_elev_km": np.round(s["surface_elev_km"], 3)})
    vx = pd.read_csv(V0 / f"{STEM}_vertices.csv")
    pd.concat([vx[~vx.surface_id.isin(ids)], v]).to_csv(V0 / f"{STEM}_vertices.csv", index=False)
    W = np.round(s["Wt"], 3)
    en = pd.read_csv(V0 / f"{STEM}_ensemble.csv.gz")
    pd.concat([en[~en.surface_id.isin(ids)], pd.DataFrame({"surface_id": a.keep, "vertex": np.tile(np.arange(W.shape[1]), W.shape[0]),
              "draw": np.repeat(np.arange(W.shape[0]), W.shape[1]), "normal_offset_km": W.ravel()})]
              ).to_csv(V0 / f"{STEM}_ensemble.csv.gz", index=False, compression="gzip")
    mem.loc[mem.surface_id.isin(ids), "surface_id"] = a.keep
    mem.drop_duplicates("event_id").to_csv(V0 / f"{STEM}_members.csv", index=False)
    tr = pd.read_csv(V0 / f"{STEM}_traces.csv")
    trow = dict(surface_id=a.keep, region=a.keep[:2], n_events=p["n_events"], lon=p["lon"], lat=p["lat"], depth_km=p["depth_km"],
                top_km=p["top_km"], bottom_km=p["bottom_km"], strike=p["strike"], dip=p["dip"], dip_68=";".join(map(str, p["dip_68"])),
                length_km=p["length_km"], width_km=p["width_km"], stable=p["stable"], kind=p["kind"], **trace_row(p, v),
                nearest_cfm=p["nearest_cfm"], nearest_cfm_km=p["nearest_cfm_km"], nearest_cfm_dip=p["nearest_cfm_dip"],
                slab2_minus_depth_km=p["slab2_minus_depth_km"])
    pd.concat([tr[~tr.surface_id.isin(ids)], pd.DataFrame([trow])]).to_csv(V0 / f"{STEM}_traces.csv", index=False)
    li = lit.set_index("surface_id")
    li.loc[a.keep, "lit_evidence"] = f"{li.loc[a.keep, 'lit_evidence']} Merged with {', '.join(a.merge)}: {a.reason}."
    li = li.drop(index=[m for m in a.merge if m in li.index])
    li.reset_index().to_csv(V0 / f"{STEM}_literature.csv", index=False)
    g = json.load(open(V0 / f"{STEM}.geojson"))
    g["features"] = [f for f in g["features"] if f["properties"]["surface_id"] not in ids] + [feat]
    (V0 / f"{STEM}.geojson").write_text(json.dumps(g, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    sp = V0 / f"{STEM}_strand_separation.csv"
    if sp.exists():
        d = pd.read_csv(sp)
        d[~(d.a.isin(ids) | d.b.isin(ids))].to_csv(sp, index=False)
    asg = pd.read_csv(V0 / f"{STEM}_assignment.csv")
    m2 = pd.read_csv(V0 / f"{STEM}_members.csv")
    asg["assigned"] = asg.region.map(m2.surface_id.str[:2].value_counts()).fillna(0).astype(int)
    asg["fraction"] = (asg.assigned / asg.events).round(3)
    asg.to_csv(V0 / f"{STEM}_assignment.csv", index=False)
    print(f"{a.keep} refit from {len(idx):,} events: strike {p['strike']}, dip {p['dip']}, {p['length_km']} x {p['width_km']} km; "
          f"{len(g['features'])} surfaces")


if __name__ == "__main__":
    main()
