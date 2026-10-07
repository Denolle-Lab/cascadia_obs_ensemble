#!/usr/bin/env bash
# Route A end-to-end driver (runs in workflow/05_magnitude; OUT and DATA are relative to it): sharded amplitude measurement -> merge -> steps 2-3.
#
# Detach it from the terminal so it survives logout (no screen/tmux needed):
#   setsid nohup bash scale/run_route_a.sh > workflow/05_magnitude/rerun_v2/driver.log 2>&1 < /dev/null &
#
# Re-running the same command resumes: each shard restarts after its last complete
# row and finished shards are skipped; merge and stage 3 (minutes) always rerun.
# Status: rerun_v2/STATUS (one line per stage),
# rerun_v2/DONE or rerun_v2/FAILED at the end. Stop everything: kill -- -<PGID>
# (the PGID is in rerun_v2/driver.pid).
#
# A variant run must not touch the paper's outputs: give it its own OUT, DATA and
# SUFFIX, e.g. the 0.5 Hz high-pass revision test:
#   OUT=rerun_hp05 DATA=../../data/magnitude_hp05 SUFFIX=_routeA_hp05 HIGHPASS=0.5 \
#     setsid nohup bash scale/run_route_a.sh > workflow/05_magnitude/rerun_hp05/driver.log 2>&1 < /dev/null &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../workflow/05_magnitude" && pwd)"   # runs in the magnitude stage
cd "$HERE"
ENV="${ENV:-$HERE/../../.pixi/envs/amplitude}"
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ENV/lib"          # system libstdc++ is too old for pandas

PICKS="${PICKS:-/wd1/hbito_data/data/datasets_all_regions/Cascadia_updated_catalog_picks_assignment_ver_3.csv}"
INV="${INV:-station_inventory_v2_slim.xml}"   # slim_inventory.py: ~0.6 GB/process vs 11.7 GB
ANSS="${ANSS:-/wd1/hbito_data/data/datasets_anss/anss_2010-15.csv}"
QC_CAT="${QC_CAT:-/wd1/hbito_data/data/datasets_all_regions/origin_2010_2015_reloc_cog_ver3_cc.csv}"
MAPENV="${MAPENV:-$HERE/../../.pixi/envs/default}"   # pygmt lives in the default env
NSHARD="${NSHARD:-20}"
TRIES="${TRIES:-3}"
OUT="${OUT:-rerun_v2}"
DATA="${DATA:-../../data/magnitude}"
SUFFIX="${SUFFIX:-_routeA}"               # stage-3 file suffix
HIGHPASS="${HIGHPASS:-1.0}"               # post-Wood-Anderson high-pass (Hz)
mkdir -p "$OUT" "$DATA"
echo "$(ps -o pgid= $$ | tr -d ' ')" > "$OUT/driver.pid"
rm -f "$OUT/DONE" "$OUT/FAILED"

status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/STATUS"; }
fail()   { status "FAILED: $*"; touch "$OUT/FAILED"; exit 1; }

TOTAL=$(( $(wc -l < "$PICKS") - 1 ))
CHUNK=$(( (TOTAL + NSHARD - 1) / NSHARD ))
status "start: $TOTAL picks, $NSHARD shards of $CHUNK, inventory $INV, highpass $HIGHPASS Hz, -> $DATA/*$SUFFIX*"

# rows already written to a shard (drops a trailing partial line left by a crash)
shard_done() {
    local f="$1"
    [ -s "$f" ] || { echo 0; return; }
    if [ "$(tail -c1 "$f" | od -An -c | tr -d ' ')" != '\n' ]; then
        head -n "$(( $(wc -l < "$f") ))" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    fi
    echo $(( $(wc -l < "$f") - 1 ))
}

# ---- stage 1: amplitudes, sharded, with retries --------------------------------
for try in $(seq 1 "$TRIES"); do
    pending=0
    for i in $(seq 0 $(( NSHARD - 1 ))); do
        start=$(( i * CHUNK )); n=$(( TOTAL - start < CHUNK ? TOTAL - start : CHUNK ))
        f=$(printf "%s/shard_%02d.csv" "$OUT" "$i")
        done_n=$(shard_done "$f")
        [ "$done_n" -ge "$n" ] && continue
        pending=$(( pending + 1 ))
        "$PY" route_a_wa_amplitudes.py --picks "$PICKS" --inventory "$INV" \
            --source pnwstore --out "$f" --highpass "$HIGHPASS" \
            --start-index $(( start + done_n )) --limit $(( n - done_n )) \
            >> "$(printf "%s/shard_%02d.log" "$OUT" "$i")" 2>&1 &
    done
    [ "$pending" -eq 0 ] && break
    status "stage1 try $try: $pending shard(s) running"
    wait
done
for i in $(seq 0 $(( NSHARD - 1 ))); do
    start=$(( i * CHUNK )); n=$(( TOTAL - start < CHUNK ? TOTAL - start : CHUNK ))
    d=$(shard_done "$(printf "%s/shard_%02d.csv" "$OUT" "$i")")
    [ "$d" -ge "$n" ] || fail "shard $i incomplete ($d/$n) after $TRIES tries; see its log"
done
status "stage1 done"

# ---- stage 2: merge shards -----------------------------------------------------
"$PY" - "$OUT" "$TOTAL" <<'EOF' || fail "merge"
import glob, sys, pandas as pd
out, total = sys.argv[1], int(sys.argv[2])
df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False)
                for f in sorted(glob.glob(f"{out}/shard_*.csv"))], ignore_index=True)
df = df.drop_duplicates("arid", keep="last")
df = df.assign(_a=df.arid.astype(int)).sort_values("_a").drop(columns="_a")
assert len(df) == total, f"{len(df)} unique arids, expected {total}"
df.to_csv(f"{out}/raw_wa_amplitudes_v2.csv", index=False)
print(df.reason.str.split(":").str[0].value_counts().to_string())
print(df[df.reason == "ok"].sensor.value_counts().head(12).to_string())
EOF
status "stage2 merged -> $OUT/raw_wa_amplitudes_v2.csv"

# ---- stage 3: dataset, inversion, anchoring, QC, calibration --------------------
run() { status "run: $*"; "$@" || fail "$*"; }
run "$PY" route_a_build_dataset.py --raw "$OUT/raw_wa_amplitudes_v2.csv" \
    --out "$DATA/amp_distance_dataset$SUFFIX.csv" --min-snr 3 --epoch-station
# amplitudes are in mm, so the counts-era floor --min-log10a 0 would drop everything;
# noise is handled by the SNR gate above
run "$PY" phase3_route_b_relative_magnitude.py \
    --dataset "$DATA/amp_distance_dataset$SUFFIX.csv" --outdir "$DATA" \
    --fix-n 1.0 --min-log10a -99 --suffix "$SUFFIX"
run "$PY" phase2_anchor_comcat_ml.py \
    --events "$DATA/route_b_event_relative_mag$SUFFIX.csv" \
    --catalog ../../data/Cascadia_relocated_catalog_ver_3.csv \
    --outdir "$DATA" --suffix "$SUFFIX"
run "$PY" phase4_qc_and_gr.py --catalog "$DATA/cascadia_catalog_ML$SUFFIX.csv" \
    --tag "${SUFFIX#_}" --outdir "$DATA"
for tgt in ml mw; do
    run "$PY" phase16_calibrate_magnitude.py --catalog "$DATA/cascadia_catalog_ML$SUFFIX.csv" \
        --target "$tgt" --anss "$ANSS" \
        --out "$DATA/cascadia_catalog${SUFFIX}_calibrated_${tgt}.csv"
done
# map in the default env (pygmt); its own lib dir first, as for the amplitude env
LD_LIBRARY_PATH="$MAPENV/lib" "$MAPENV/bin/python" ../06_analysis/phase5_pygmt_map.py \
    --catalog "$DATA/cascadia_catalog_ML$SUFFIX.csv" --qc-catalog "$QC_CAT" \
    --out "$DATA/cascadia_ML_map$SUFFIX.png" \
    || status "map failed (non-fatal): see driver.log"

status "ALL DONE"
touch "$OUT/DONE"
