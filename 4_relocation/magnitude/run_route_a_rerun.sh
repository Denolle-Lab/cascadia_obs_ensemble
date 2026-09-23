#!/usr/bin/env bash
# Route A end-to-end driver: sharded amplitude measurement -> merge -> steps 2-3.
#
# Detach it from the terminal so it survives logout (no screen/tmux needed):
#   setsid nohup bash run_route_a_rerun.sh > rerun_v2/driver.log 2>&1 < /dev/null &
#
# Re-running the same command resumes: each shard restarts after its last complete
# row and finished shards are skipped; merge and stage 3 (minutes) always rerun.
# Status: rerun_v2/STATUS (one line per stage),
# rerun_v2/DONE or rerun_v2/FAILED at the end. Stop everything: kill -- -<PGID>
# (the PGID is in rerun_v2/driver.pid).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
ENV="${ENV:-$HERE/../../.pixi/envs/amplitude}"
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ENV/lib"          # system libstdc++ is too old for pandas

PICKS="${PICKS:-/wd1/hbito_data/data/datasets_all_regions/Cascadia_updated_catalog_picks_assignment_ver_3.csv}"
INV="${INV:-station_inventory_v2.xml}"
ANSS="${ANSS:-/wd1/hbito_data/data/datasets_anss/anss_2010-15.csv}"
NSHARD="${NSHARD:-20}"
TRIES="${TRIES:-3}"
OUT="${OUT:-rerun_v2}"
DATA="${DATA:-../../data/magnitude}"
mkdir -p "$OUT" "$DATA"
echo "$(ps -o pgid= $$ | tr -d ' ')" > "$OUT/driver.pid"
rm -f "$OUT/DONE" "$OUT/FAILED"

status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/STATUS"; }
fail()   { status "FAILED: $*"; touch "$OUT/FAILED"; exit 1; }

TOTAL=$(( $(wc -l < "$PICKS") - 1 ))
CHUNK=$(( (TOTAL + NSHARD - 1) / NSHARD ))
status "start: $TOTAL picks, $NSHARD shards of $CHUNK, inventory $INV"

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
            --source pnwstore --out "$f" \
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
    --out "$DATA/amp_distance_dataset_routeA.csv" --min-snr 3 --epoch-station
# amplitudes are in mm, so the counts-era floor --min-log10a 0 would drop everything;
# noise is handled by the SNR gate above
run "$PY" phase3_route_b_relative_magnitude.py \
    --dataset "$DATA/amp_distance_dataset_routeA.csv" --outdir "$DATA" \
    --fix-n 1.0 --min-log10a -99 --suffix _routeA
run "$PY" phase2_anchor_comcat_ml.py \
    --events "$DATA/route_b_event_relative_mag_routeA.csv" \
    --catalog ../../data/Cascadia_relocated_catalog_ver_3.csv \
    --outdir "$DATA" --suffix _routeA
run "$PY" phase4_qc_and_gr.py --catalog "$DATA/cascadia_catalog_ML_routeA.csv" \
    --tag routeA --outdir "$DATA"
for tgt in ml mw; do
    run "$PY" phase16_calibrate_magnitude.py --catalog "$DATA/cascadia_catalog_ML_routeA.csv" \
        --target "$tgt" --anss "$ANSS" \
        --out "$DATA/cascadia_catalog_routeA_calibrated_${tgt}.csv"
done
"$PY" phase5_pygmt_map.py --catalog "$DATA/cascadia_catalog_ML_routeA.csv" \
    --out "$DATA/cascadia_ML_map_routeA.png" \
    || status "map skipped (phase5 needs pygmt; not in the amplitude env)"

status "ALL DONE"
touch "$OUT/DONE"
