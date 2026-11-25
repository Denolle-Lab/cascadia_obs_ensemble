#!/bin/bash
# Final CC run with the chosen parameters (b5_cc_params.yaml) and outputs (rerun-safe: overwrites own outputs).
set -e
cd /home/mdenolle/wt_ccdt/workflow/10_tomography
export LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
PY="nice -n 10 /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
for s in ${SETS:-jdf axial blanco msh}; do
  $PY b5_cc_measure.py --set $s --params b5_cc_params.yaml --out /wd1/mdenolle_data/tomo_v1/cc_dt/$s/cc_raw.parquet --workers ${WORKERS:-14}
  $PY b5_cc_write.py --set $s --params b5_cc_params.yaml > /wd1/mdenolle_data/tomo_v1/cc_dt/logs/write_$s.log
done
# station clock epochs (MH07.XD, MG07.XD free-running clocks 2016-02..05): adds clock_* and straddles_clock_epoch
# columns to every cc_<set>.parquet and writes qc/clock_epochs.csv (needs clock/nll_residuals.parquet from
# b5_cc_clock_res.py and the reset times from b5_cc_clock_hdr.py)
$PY b5_cc_clock.py --screen --fit MH07.XD:2016-05-12T05:09:09,MG07.XD:2016-05-08T07:57:28 --apply > /wd1/mdenolle_data/tomo_v1/cc_dt/logs/clock_fit.log
echo "run_final.sh done $(date)"
