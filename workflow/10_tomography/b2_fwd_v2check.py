"""D-B028 end-to-end check: tomoDD initial residuals of a region's inputs against P-1 reference residuals.

tomoDD residual r_t = absolute.dat time - tomoDD time (tomoDD-SE.res.ini of a 1-iteration run, e.g. the --check run);
reference residual r_p1 = (pick - origin - OBS S static) - P-1 time (no station correction). Difference
d = r_t - r_p1 per pick; for a station-corrected build d = t_p1 - t_tomoDD + c(sta, phase). Tables by phase x
station type x distance (as notes/forward_accuracy.md). Also the first build (check01) for contrast.

  python b2_fwd_v2check.py mendocino_gorda_v2 check01 [--old mendocino_gorda/check01]
Output: <runs>/<region>/<tag>/v2check_{tables.md,summary.csv}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from b2_fwd_common import bias_table, md_table, summary_row
from manifest import load_params

T = load_params()["tomodd"]


def res_ini(run):
    r = pd.read_csv(run / "tomoDD-SE.res.ini", sep=r"\s+")
    r = r[r.C1 == r.C2]
    return pd.DataFrame({"event_ID": r.C1.values, "code": r.STA.values,
                         "phase": np.where(r.IDX.values == 3, "P", "S"), "res_t": r["RES"].values / 1000.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("tag")
    ap.add_argument("--old", default="mendocino_gorda/check01")
    a = ap.parse_args()
    base = Path(T["runs"]) / a.region
    fw = (base / "inputs" / "forward.json").read_text() if (base / "inputs" / "forward.json").exists() else ""
    ref_dir = Path(T["runs"]) / "mendocino_gorda_v2" / "p1_reference"
    import json
    if fw:
        ref_dir = Path(T["runs"]) / json.loads(fw)["p1_reference"]
    ref = pd.read_parquet(ref_dir / "region_picks_p1.parquet")
    rows, md = [], []
    for label, run_dir, inp in (("v2 (" + a.region + "/" + a.tag + ")", base / a.tag, base / "inputs"),
                                ("first build (" + a.old + ")", Path(T["runs"]) / a.old,
                                 Path(T["runs"]) / a.old.split("/")[0] / "inputs")):
        pk = pd.read_csv(inp / "picks_used.csv")
        pk["stype"] = np.where(pk.is_obs, "obs", "land")
        d = pk.merge(res_ini(run_dir), on=["event_ID", "code", "phase"], how="inner")
        d = d.merge(ref[["event_ID", "station", "phase", "t_p1"]], on=["event_ID", "station", "phase"], how="left")
        d["res_p1"] = d.tt - d.t_p1
        d["d"] = d.res_t - d.res_p1
        rows.append({"build": label, "n": int(d.d.notna().sum()), **summary_row(d, "d")})
        md += [f"#### {label}: tomoDD initial residual - P-1 reference residual", "",
               md_table(bias_table(d, "d").round(3).reset_index()), ""]
        if label.startswith("v2"):
            d[["event_ID", "station", "phase", "res_t", "res_p1", "d"]].to_parquet(run_dir / "v2check_picks.parquet")
    s = pd.DataFrame(rows)
    out = base / a.tag
    s.to_csv(out / "v2check_summary.csv", index=False)
    (out / "v2check_tables.md").write_text("\n".join(md))
    pd.set_option("display.width", 250)
    print(s.set_index("build").T.to_string())


if __name__ == "__main__":
    main()
