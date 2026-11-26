#!/bin/bash
# v0.4.2 (D-B099) step 2: deterministic inputs of the m0g tiles (fixed diff_times) and their noise-floor copies, one at a time.
set -u
cd /home/mdenolle/wt_ensfix
PY="env LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib nice -n 10 /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
W=workflow/10_tomography
L=/wd1/mdenolle_data/tomo_v1/v042/logs
for r in washington_m0g mendocino_gorda_c_m0g blanco_m0g washington_m0gnf mendocino_gorda_c_m0gnf blanco_m0gnf; do
  echo "$(date '+%F %T') start inputs $r"
  if [ -e /wd1/mdenolle_data/tomo_v1/b2_tomodd/$r/inputs/summary.json ]; then echo "  exists, skipped"; continue; fi
  $PY $W/b2_tomodd_inputs.py $r > $L/inp_$r.log 2>&1 || { echo "  FAILED inputs $r"; continue; }
  case $r in *nf) $PY $W/b2_v040_noise.py $r >> $L/inp_$r.log 2>&1 || echo "  FAILED noise $r";; esac
  echo "$(date '+%F %T') done inputs $r"
done
