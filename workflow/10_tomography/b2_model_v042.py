"""Track B: model v0.4.2 (D-B099) = v0.4.1 with the self-paired differential times removed from the mendocino_gorda_c,
blanco and washington tiles, those tiles re-inverted with the 2026-11-26 tomoDD freeze (M0 and oregon are v0.4.1's runs,
reproduced byte for byte by the new freeze), exported with exactly v0.4.1's code and rules (b2_model_v041.build_nodes:
medium 0 rock / 1 water / 2 air at the D-B068 merged surface, Vp water 1.5 km/s, air NaN, D-B073 shallow_crust_node rule,
Vp_tomography / Vs_tomography = the inversion values, merge variables as written by b2_merge.py --product product_v042).

Written to the staging folder params v042.staging (store layout <staging>/<models.name>/v0.4.2.{nc,zarr} and
v0.4.2_surface_lattice.nc), NOT to the model store and NOT registered: the PI's session registers it (append-only).
No NonLinLoc grids (catalog v6.0 stays on v0.4.1, D-B099).

  python b2_model_v042.py nodes      # staging files + read-back checks -> <v042.out>/product/nodes_written.json
Run with the models env python.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_model_v041 as M  # noqa: E402
import b2_model_export as EX  # noqa: E402

V = M.P["v042"]
STAGING = Path(V["staging"])


def configure():
    """Point b2_model_v041's module state at v0.4.2 (same functions, v0.4.2 inputs and outputs)."""
    M.V = V
    M.VERSION = str(V["version"])
    M.DESC = V["description"]
    M.WORK = Path(V["out"]) / "product"
    M.MERGED = M.WORK / "merged_v042.nc"
    M.FLOOR_STAGES = {k: tuple(v.split("/")) for k, v in V["stages"].items()}
    M.BINARIES = V["binaries"]
    M.FLOOR_DECISIONS = ("D-B081 (tomoDD floor), D-B083 (floor in the inversion); D-B099: tiles mendocino_gorda_c, "
                         "blanco, washington re-inverted without self-paired differential times with the 2026-11-26 "
                         "freeze (M0 and oregon: the 2026-11-25 freeze runs of v0.4.1, reproduced byte for byte)")
    M.SOURCE_NOTE = "b2_model_v042.py (b2_model_v041 code) from the v0.4.2 merge (product_v042)"
    EX.STORE = STAGING                                  # EX.write -> <staging>/<name>/v0.4.2.*
    M.STORE = STAGING / EX.X["name"]


def main():
    if sys.argv[1:] != ["nodes"]:
        raise SystemExit(__doc__)
    configure()
    M.WORK.mkdir(parents=True, exist_ok=True)
    M.STORE.mkdir(parents=True, exist_ok=True)
    M.cmd_nodes(None)
    nw = json.loads((M.WORK / "nodes_written.json").read_text())
    print("staged:", json.dumps({k: v["path"] for k, v in nw["files"].items()}))


if __name__ == "__main__":
    main()
