#!/usr/bin/env bash
# Download the figure/pipeline input data from the lab server into ./data via rsync.
#
# The figure notebooks (figures/fig*.ipynb) read catalogs/tables that live on the
# server under /wd1/hbito_data/data/{datasets_all_regions,datasets_anss} -- these are
# NOT in the repo (only data/ds01.csv is). This script fetches the small data files
# (catalogs, station lists, ANSS, map-overlay files) into data/datasets_all_regions/
# and data/datasets_anss/, mirroring the server layout, so the notebooks can run
# locally once their paths point at ../data (a one-line DATA_DIR per notebook).
#
# Images (*.png/*.pdf) are skipped, and the large per-pick tables (arrival_*/assoc_*,
# which only fig3 needs) are skipped unless you pass --with-picks.
#
# Usage:
#   ./download_data.sh                                   # default host, small data
#   ./download_data.sh mdenolle@psound.ess.washington.edu
#   ./download_data.sh <user@host> --with-picks          # also arrival/assoc (large; fig3)
#   REMOTE_BASE=/some/other/path ./download_data.sh <user@host>
#   PORT=27531 ./download_data.sh <user@host>             # non-default SSH port
set -euo pipefail

REMOTE="${1:-mdenolle@psound.ess.washington.edu}"
REMOTE_BASE="${REMOTE_BASE:-/wd1/hbito_data/data}"
PORT="${PORT:-27531}"                                       # SSH port (override: PORT=2222 ...)
WITH_PICKS=0
[[ "${2:-}" == "--with-picks" || "${1:-}" == "--with-picks" ]] && WITH_PICKS=1
[[ "${1:-}" == "--with-picks" ]] && REMOTE="mdenolle@psound.ess.washington.edu"

DEST="$(cd "$(dirname "$0")" && pwd)/data"
echo "Source : ${REMOTE}:${REMOTE_BASE}"
echo "Dest   : ${DEST}/{datasets_all_regions,datasets_anss}"
echo "Picks  : $([[ $WITH_PICKS -eq 1 ]] && echo 'included (arrival/assoc, large)' || echo 'skipped (use --with-picks for fig3)')"
echo

# rsync filter: keep small data files, drop images; drop the big pick tables unless asked.
FILTERS=(--include='*/')
[[ $WITH_PICKS -eq 0 ]] && FILTERS+=(--exclude='arrival_*' --exclude='assoc_*')
FILTERS+=(--include='*.csv' --include='*.txt' --include='*.geojson'
          --include='*.json' --include='*.npy' --include='*.npz' --exclude='*')

for sub in datasets_all_regions datasets_anss; do
  echo "==> $sub"
  mkdir -p "${DEST}/${sub}"
  rsync -avh --progress -e "ssh -p ${PORT}" "${FILTERS[@]}" \
    "${REMOTE}:${REMOTE_BASE}/${sub}/" "${DEST}/${sub}/"
done

echo
echo "Done."
echo "Next: point the figure notebooks at ../data (set DATA_DIR at the top of each"
echo "figures/fig*.ipynb, replacing the /wd1/hbito_data/data prefix), then: make figs"
[[ $WITH_PICKS -eq 0 ]] && echo "(fig3 needs the pick tables -- re-run with --with-picks to fetch arrival_*/assoc_*.)"
