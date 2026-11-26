#!/usr/bin/env bash
# Route A magnitudes for catalog v5.1 (v4.1 events relocated with NonLinLoc in model
# v0.3.1, re-timed picks). Same chain and options as scale/run_magnitude_v4.sh; what
# changes is the hypocentres (distances, distance-scaled windows, the 150 km pick
# selection of phase19) and the event list (v4.1, rows without a hypocentre left out).
# The amplitudes are the same per-station-day tables (amps_v6_all), keyed by pick_id:
# they are not re-measured (see v51_build_arrivals.py and the report).
#   1. v51_build_arrivals.py  catalog v5.1 -> arrivals_v51.csv (+ fallback file)
#   2. route_a_day_amplitudes.py select  -> raw_wa_amplitudes_v51.csv (window per r)
#   3. route_a_build_dataset -> phase3 -> phase2 -> phase4 -> phase16 (ml, mw)
#   4. phase18 (ComCat tensors) -> phase19 (preferred M)
#   5. v51_magnitude_table.py: per-event table joined to the v5.1 flags, fallback
#      ML_v4hyp for rows without a hypocentre
# Large outputs in $DATA (default $CASCADIA_WD1/catalog_v5/v5.1_analysis/mag), suffix _v51.
# Resumable: steps whose output exists are skipped (FORCE=1 reruns all).
# Paths: the pixi env and the v4 run (gitignored CSVs) are taken from the main checkout,
# found from the git common directory so that the script also runs from a worktree
# (CASCADIA_MAIN overrides it; ENV, V4 override each one). CASCADIA_WD1 is the project
# data disk (default /wd1/mdenolle_data). Same defaults as workflow/05_magnitude/v51_paths.py.
#   bash scale/run_magnitude_v51.sh
# Another catalog of the v4.1 events (catalog v6.1, D-B105: magnitudes on its own hypocentres) runs the same chain
# with CATALOG=<catalog csv> DATA=<its folder> SUFFIX=<_v61> NAME=<6.1>: arrivals from that catalog's hypocentres
# (file names arrivals_v51*.csv kept inside DATA), table cascadia_catalog_v<NAME>_magnitudes.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/workflow/05_magnitude"
if [ -z "${CASCADIA_MAIN:-}" ]; then
  COMMON="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$ROOT/.git")"
  CASCADIA_MAIN="$(dirname "$COMMON")"
fi
export CASCADIA_MAIN
export CASCADIA_WD1="${CASCADIA_WD1:-/wd1/mdenolle_data}"
ENV="${ENV:-$CASCADIA_MAIN/.pixi/envs/default}"
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ENV/lib"
export PYTHONPATH="$PWD"
DATA="${DATA:-$CASCADIA_WD1/catalog_v5/v5.1_analysis/mag}"
SUFFIX="${SUFFIX:-_v51}"
AMPS="${AMPS:-$CASCADIA_WD1/amps_v6_all}"
V4="${V4:-$CASCADIA_MAIN/data/magnitude_v4}"
ANSS="$DATA/anss_2010-16.csv"
MATCHED="$DATA/comcat_mt_matched$SUFFIX.csv"
CATALOG="${CATALOG:-}"          # default: catalog v5.1 (v51_paths.V51_CATALOG)
NAME="${NAME:-5.1}"             # table name cascadia_catalog_v<NAME>_magnitudes.csv
mkdir -p "$DATA"
# same ComCat inputs as the v4 run (anchors comparable; no refetch)
[ -s "$ANSS" ] || cp "$V4/anss_2010-16.csv" "$ANSS"
[ -s "$DATA/comcat_ml_events.csv" ] || cp "$V4/comcat_ml_events.csv" "$DATA/comcat_ml_events.csv"
run() { echo "[$(date +%F' '%T)] $*"; "$@"; }
need() { [ -n "${FORCE:-}" ] || [ ! -s "$1" ]; }

need "$DATA/arrivals_v51.csv" && run "$PY" v51_build_arrivals.py --outdir "$DATA" ${CATALOG:+--catalog "$CATALOG"}
need "$DATA/raw_wa_amplitudes_v51.csv" && run "$PY" route_a_day_amplitudes.py select \
    --amps "$AMPS" --arrivals "$DATA/arrivals_v51.csv" --out "$DATA/raw_wa_amplitudes_v51.csv"
need "$DATA/raw_wa_amplitudes_v51_nohypo_v4hyp.csv" && run "$PY" route_a_day_amplitudes.py select \
    --amps "$AMPS" --arrivals "$DATA/arrivals_v51_nohypo_v4hyp.csv" \
    --out "$DATA/raw_wa_amplitudes_v51_nohypo_v4hyp.csv"
need "$DATA/amp_distance_dataset$SUFFIX.csv" && run "$PY" route_a_build_dataset.py \
    --raw "$DATA/raw_wa_amplitudes_v51.csv" \
    --out "$DATA/amp_distance_dataset$SUFFIX.csv" --min-snr 3 --epoch-station
need "$DATA/amp_distance_dataset_nohypo_v4hyp.csv" && run "$PY" route_a_build_dataset.py \
    --raw "$DATA/raw_wa_amplitudes_v51_nohypo_v4hyp.csv" \
    --out "$DATA/amp_distance_dataset_nohypo_v4hyp.csv" --min-snr 3 --epoch-station
need "$DATA/route_b_event_relative_mag$SUFFIX.csv" && run "$PY" phase3_route_b_relative_magnitude.py \
    --dataset "$DATA/amp_distance_dataset$SUFFIX.csv" --outdir "$DATA" \
    --fix-n 1.0 --min-log10a -99 --suffix "$SUFFIX"
need "$DATA/cascadia_catalog_ML$SUFFIX.csv" && run "$PY" phase2_anchor_comcat_ml.py \
    --events "$DATA/route_b_event_relative_mag$SUFFIX.csv" \
    --catalog "$DATA/catalog_v51_for_phase2.csv" --outdir "$DATA" --suffix "$SUFFIX"
need "$DATA/v51${SUFFIX}_qc.png" && run "$PY" phase4_qc_and_gr.py \
    --catalog "$DATA/cascadia_catalog_ML$SUFFIX.csv" --tag "v51${SUFFIX}" --outdir "$DATA"
for tgt in ml mw; do
  need "$DATA/cascadia_catalog${SUFFIX}_calibrated_${tgt}.csv" && run "$PY" phase16_calibrate_magnitude.py \
      --catalog "$DATA/cascadia_catalog_ML$SUFFIX.csv" --target "$tgt" --anss "$ANSS" \
      --out "$DATA/cascadia_catalog${SUFFIX}_calibrated_${tgt}.csv"
done
need "$MATCHED" && run "$PY" phase18_moment_tensor_match.py --data "$DATA" --suffix "$SUFFIX" --matched "$MATCHED"
need "$DATA/cascadia_catalog_M$SUFFIX.csv" && run "$PY" phase19_preferred_magnitude.py \
    --data "$DATA" --suffix "$SUFFIX" --matched "$MATCHED"
run "$PY" v51_magnitude_table.py --data "$DATA" --suffix "$SUFFIX" ${CATALOG:+--catalog "$CATALOG"} --name "$NAME"
echo "[$(date +%F' '%T)] done -> $DATA/cascadia_catalog_v${NAME}_magnitudes.csv"
