#!/bin/bash
# D-B099 freeze identity check: rerun one v0.4.1 deterministic run (with its surface.dat: D-B083 floor) with the
# 2026-11-26 binary of the same size class, same inputs, tomoDD-SE.inp, MOD and surface.dat; cmp every Output_Files file.
# usage: run_identity.sh REGION TAG CLASS [ITER]   ITER: stop after this many complete iterations (partial check:
#        tomoDD.loc, tomoDD-SE.res.ini and tomoDD.reloc.001.001..ITER); default: the full run.
set -u
REG=$1; TAG=$2; CLS=$3; IT=${4:-0}
SRC=/wd1/mdenolle_data/tomo_v1/b2_tomodd/$REG/$TAG
BIN=/wd1/mdenolle_data/tomo_v1/src/frozen_2026-11-26_inv/bin/tomoDD-SE_$CLS
W=/wd1/mdenolle_data/tomo_v1/v042/identity/${REG}_${TAG}
if [ "${COMPARE_ONLY:-0}" = 1 ]; then          # redo the comparison of a finished run (no tomoDD)
  cd $W; files="tomoDD.loc $(for i in $(seq 1 $IT); do printf "tomoDD.reloc.001.%03d " $i; done)"
  [ "$IT" -gt 0 ] || files=$(cd $SRC/Output_Files && ls | grep -v -e _geo -e model_geo.npz)
else
[ -e $W/DONE ] && exit 0
rm -rf $W; mkdir -p $W/Input_Files $W/Output_Files
for f in absolute.dat dt.ct event.dat station.dat; do ln -s $(readlink -f $SRC/Input_Files/$f) $W/Input_Files/$f; done
cp $SRC/MOD $SRC/tomoDD-SE.inp $SRC/ak135.15.SKS $SRC/layer-16.dat $SRC/surface.dat $W/
cd $W
if [ "$IT" -gt 0 ]; then
  nice -n 10 $BIN tomoDD-SE.inp > stdout.txt 2> time.txt &
  pid=$!
  nxt=$(printf "Output_Files/tomoDD.reloc.001.%03d" $((IT+1)))
  while kill -0 $pid 2>/dev/null && [ ! -e $nxt ]; do sleep 5; done
  kill $pid 2>/dev/null; wait $pid; rc=$?
  echo "stopped after $IT iterations (rc $rc)" > exit.txt
  files="tomoDD.loc $(for i in $(seq 1 $IT); do printf "tomoDD.reloc.001.%03d " $i; done)"
else
  /usr/bin/time -f "%e s %U user %M KB" nice -n 10 $BIN tomoDD-SE.inp > stdout.txt 2> time.txt
  echo "exit $?" > exit.txt
  files=$(cd $SRC/Output_Files && ls | grep -v -e _geo -e model_geo.npz)
fi
fi
{
  echo "binary $BIN md5 $(md5sum < $BIN | cut -c1-32)"
  echo "reference $SRC (v0.4.1, binary $(grep -o '"binary": "[^"]*' $SRC/b2_tomodd_run_*.manifest.json | cut -d'"' -f4))"
  n=0; nd=0
  for f in $files; do
    n=$((n+1))
    if [ "$IT" -gt 0 ] && [ "$f" = tomoDD.loc ]; then   # stopped run: tomoDD.loc is closed at the end, its buffer lost
      sz=$(stat -c %s $W/Output_Files/$f)
      if [ "$sz" -gt 0 ] && cmp -s -n $sz $SRC/Output_Files/$f $W/Output_Files/$f; then
        echo "IDENTICAL $f (byte prefix: $sz of $(stat -c %s $SRC/Output_Files/$f) bytes flushed before the stop)"
      else echo "DIFFERENT $f"; nd=$((nd+1)); fi
      continue
    fi
    if cmp -s $SRC/Output_Files/$f $W/Output_Files/$f; then echo "IDENTICAL $f"; else echo "DIFFERENT $f"; nd=$((nd+1)); fi
  done
  n=$((n+1)); if cmp -s $SRC/tomoDD-SE.res.ini $W/tomoDD-SE.res.ini; then echo "IDENTICAL tomoDD-SE.res.ini"; else echo "DIFFERENT tomoDD-SE.res.ini"; nd=$((nd+1)); fi
  for f in tomoDD-SE.log stdout.txt; do cmp -s $SRC/$f $W/$f && echo "IDENTICAL $f" || echo "DIFFERENT $f (text; see diff)"; done
  echo "files $n different $nd"
} > compare.txt
diff $SRC/tomoDD-SE.log $W/tomoDD-SE.log > log.diff
touch DONE
