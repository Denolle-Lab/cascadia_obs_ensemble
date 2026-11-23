#!/usr/bin/env bash
# Route A magnitudes for catalog v4 (Ian's final catalog on the ver6 picks).
# Same chain and options as stage 3 of scale/run_route_a.sh (the v3 run), with the
# amplitudes taken from the per-station-day tables (ver5 r2 + ver6) instead of the
# per-pick run, then the phase18/19 preferred magnitude.
#   1. v4_build_arrivals.py  catalog v4 -> arrivals_v4.csv (pick_id, coordinates)
#   2. route_a_day_amplitudes.py select  -> raw_wa_amplitudes_v4.csv (Route A window)
#   3. route_a_build_dataset -> phase3 -> phase2 -> phase4 -> phase16 (ml, mw)
#   4. phase18 (ComCat tensors) -> phase19 (preferred M)
# Outputs in data/magnitude_v4/, suffix _routeA. Run from the repo root:
#   bash scale/run_magnitude_v4.sh            (SKIP_SELECT=1 reuses step 1-2 outputs)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/workflow/05_magnitude"
ENV="${ENV:-$ROOT/.pixi/envs/default}"
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ENV/lib"          # system libstdc++ is too old for pandas
export PYTHONPATH="$PWD"
DATA="${DATA:-../../data/magnitude_v4}"
SUFFIX="${SUFFIX:-_routeA}"
AMPS="${AMPS:-/wd1/mdenolle_data/amps_v6_all}"   # symlinks: amps_v5_* + amps_v6_*
ANSS="$DATA/anss_2010-16.csv"                   # anss_2010-15.csv + USGS 2016
MATCHED="$DATA/comcat_mt_matched_v4.csv"
mkdir -p "$DATA"
run() { echo "[$(date +%F' '%T)] $*"; "$@"; }

if [ -z "${SKIP_SELECT:-}" ]; then
  run "$PY" v4_build_arrivals.py --outdir "$DATA"
  run "$PY" route_a_day_amplitudes.py select --amps "$AMPS" \
      --arrivals "$DATA/arrivals_v4.csv" --out "$DATA/raw_wa_amplitudes_v4.csv"
fi
run "$PY" route_a_build_dataset.py --raw "$DATA/raw_wa_amplitudes_v4.csv" \
    --out "$DATA/amp_distance_dataset$SUFFIX.csv" --min-snr 3 --epoch-station
run "$PY" phase3_route_b_relative_magnitude.py \
    --dataset "$DATA/amp_distance_dataset$SUFFIX.csv" --outdir "$DATA" \
    --fix-n 1.0 --min-log10a -99 --suffix "$SUFFIX"
run "$PY" phase2_anchor_comcat_ml.py \
    --events "$DATA/route_b_event_relative_mag$SUFFIX.csv" \
    --catalog "$DATA/catalog_v4_for_phase2.csv" --outdir "$DATA" --suffix "$SUFFIX"
run "$PY" phase4_qc_and_gr.py --catalog "$DATA/cascadia_catalog_ML$SUFFIX.csv" \
    --tag "v4${SUFFIX}" --outdir "$DATA"
for tgt in ml mw; do
  run "$PY" phase16_calibrate_magnitude.py --catalog "$DATA/cascadia_catalog_ML$SUFFIX.csv" \
      --target "$tgt" --anss "$ANSS" --out "$DATA/cascadia_catalog${SUFFIX}_calibrated_${tgt}.csv"
done
run "$PY" phase18_moment_tensor_match.py --data "$DATA" --suffix "$SUFFIX" --matched "$MATCHED"
run "$PY" phase19_preferred_magnitude.py --data "$DATA" --suffix "$SUFFIX" --matched "$MATCHED"
echo "[$(date +%F' '%T)] done -> $DATA/cascadia_catalog_M$SUFFIX.csv"
