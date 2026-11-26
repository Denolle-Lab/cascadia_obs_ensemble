#!/bin/bash
# All-family CC differential times (D-B093): stages in order; every stage is resumable (rerun skips finished work).
#   STAGES="snippets ncedc" b5_ccall_run.sh      (I/O stages; run in parallel as two calls)
#   STAGES="measure write clock final" b5_ccall_run.sh
# Output /wd1/mdenolle_data/tomo_v1/cc_dt_all (CCALL_OUT overrides; snippet caches are fixed). State: state/*.json.
set -e
cd /home/mdenolle/wt_ccall/workflow/10_tomography
export LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
OUT=${CCALL_OUT:-/wd1/mdenolle_data/tomo_v1/cc_dt_all}
mkdir -p $OUT/logs
for s in ${STAGES:-tasks snippets ncedc measure write clock final}; do
  echo "$(date '+%F %T') stage $s start" >> $OUT/logs/run.log
  case $s in
    tasks)    $PY b5_ccall_tasks.py >> $OUT/logs/tasks.log 2>&1 ;;
    snippets) $PY b5_ccall_snippets.py --workers ${WORKERS:-24} --max-workers ${MAX_WORKERS:-48} --chunk 12 >> $OUT/logs/snippets.out 2>&1 ;;
    ncedc)    $PY b5_ccall_ncedc.py --threads ${THREADS:-8} >> $OUT/logs/ncedc.out 2>&1 ;;
    measure)  $PY b5_ccall_measure.py --workers ${WORKERS:-24} --max-workers ${MAX_WORKERS:-48} >> $OUT/logs/measure.out 2>&1 ;;
    write)    # three block groups side by side (largest blocks apart; peak memory ~10 GB for gorda)
              $PY b5_ccall_write.py --blocks gorda,jdf,endeavour,explorer >> $OUT/logs/write_a.out 2>&1 &
              $PY b5_ccall_write.py --blocks blanco,mendocino,onshore >> $OUT/logs/write_b.out 2>&1 &
              $PY b5_ccall_write.py --blocks forearc,axial,msh,cascades,puget >> $OUT/logs/write_c.out 2>&1 &
              wait ;;
    clock)    $PY b5_ccall_clock.py >> $OUT/logs/clock.out 2>&1 ;;
    final)    $PY b5_ccall_final.py >> $OUT/logs/final.out 2>&1 ;;
  esac
  echo "$(date '+%F %T') stage $s done" >> $OUT/logs/run.log
done
