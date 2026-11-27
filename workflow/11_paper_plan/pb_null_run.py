#!/usr/bin/env python3
"""Run the null-truth members (D-B112) with the ensemble scheduler of b2_ens_run.py, then score them.

  B2_ENS_CONFIG=v0.4.2 B2_ENS_ROOT=/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null \
      setsid nohup nice -n 10 python pb_null_run.py [N_MEMBERS] >> ROOT/run.log 2>&1 &

Members s000..s(N-1) (default 20), the same member seeds as the calibration members; at most 6 tomoDD processes;
MemAvailable throttle raised from the configuration's 30 GB to 40 GB (coordinator limit) for this run only. Resumable:
rerunning skips merged members and finished stages. When every member has merged (or failed), runs
pb_null_score.py on catalog v6.1 (results only; nothing is committed).
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "10_tomography"))
NULL_ROOT = Path("/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null")

if Path(os.environ.get("B2_ENS_ROOT", "")).resolve() != NULL_ROOT or os.environ.get("B2_ENS_CONFIG") != "v0.4.2":
    raise SystemExit(f"set B2_ENS_CONFIG=v0.4.2 and B2_ENS_ROOT={NULL_ROOT}")
import b2_ens_run as R  # noqa: E402

assert R.ROOT.resolve() == NULL_ROOT
R.E["min_avail_gb"] = 40.0
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
names = [f"s{i:03d}" for i in range(n)]
R.guarded(R.run_members, "synth", names, 6)
score = HERE / "pb_null_score.py"
if score.exists():
    env = {**os.environ, "PB_CATALOG": "/wd1/mdenolle_data/catalog_v6/v6.1/cascadia_catalog_v6.1.csv", "PB_TAG": "v61",
           "PB_LABEL": "v6.1", "PB_OUT": "/wd1/mdenolle_data/paper_plan/v6.1/tables",
           "PB_BIG": "/wd1/mdenolle_data/paper_plan/v6.1"}
    subprocess.run([sys.executable, str(score)], env=env, cwd=HERE)
