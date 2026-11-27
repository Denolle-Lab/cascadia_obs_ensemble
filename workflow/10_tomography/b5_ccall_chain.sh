#!/bin/bash
# Waits for the two read stages (state/snippets.json, state/ncedc.json: done true), reruns any of them that ended
# with missing events (once), then runs measure, write, clock, final (b5_ccall_run.sh). Detached-safe and resumable.
OUT=${CCALL_OUT:-/wd1/mdenolle_data/tomo_v1/cc_dt_all}
cd "${CCALL_CODE:-$(dirname "$(readlink -f "$0")")}"   # this script's folder (was the worktree ~/wt_ccall; D-B110)
st() { python3 -c "import json,sys; s=json.load(open('$OUT/state/$1.json')); print('done' if s.get('done') else ('ended' if 'missing' in s else 'running'))" 2>/dev/null || echo running; }
echo "$(date '+%F %T') chain: waiting for snippets and ncedc" >> $OUT/logs/run.log
for k in 1 2; do
  while [ "$(st snippets)" = running ] || [ "$(st ncedc)" = running ]; do sleep 60; done
  if [ "$(st snippets)" = done ] && [ "$(st ncedc)" = done ]; then break; fi
  [ "$(st snippets)" = ended ] && STAGES=snippets ./b5_ccall_run.sh
  [ "$(st ncedc)" = ended ] && STAGES=ncedc ./b5_ccall_run.sh
done
echo "$(date '+%F %T') chain: read stages snippets=$(st snippets) ncedc=$(st ncedc); starting measure" >> $OUT/logs/run.log
STAGES="measure write clock final" ./b5_ccall_run.sh
echo "$(date '+%F %T') chain: finished" >> $OUT/logs/run.log
