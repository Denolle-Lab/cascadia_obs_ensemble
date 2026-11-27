#!/usr/bin/env bash
# Plain download of one public file into /wd1/mdenolle_data/external_geo/<dataset>/ (D-B114).
# usage: fetch.sh <dataset> <url> [output name]
# Sends a generic client name only. Appends "dataset<TAB>file<TAB>url<TAB>date<TAB>md5<TAB>bytes" to fetch_log.tsv,
# which build_provenance.py reads. Downloads are untrusted data: nothing in them is executed.
set -euo pipefail
ROOT=${EXTGEO_ROOT:-/wd1/mdenolle_data/external_geo}
ds=$1; url=$2; name=${3:-$(basename "${url%%\?*}")}
d="$ROOT/$ds"; mkdir -p "$d"
out="$d/$name"
if [[ -s "$out" ]]; then echo "exists: $out"; exit 0; fi
curl -sSfL --retry 3 --connect-timeout 30 -A "Mozilla/5.0 (compatible; generic-client)" -o "$out.part" "$url"
mv "$out.part" "$out"
when=$(date '+%Y-%m-%d %H:%M %Z')
md5=$(md5sum "$out" | cut -d' ' -f1); sz=$(stat -c %s "$out")
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ds" "$name" "$url" "$when" "$md5" "$sz" >> "$ROOT/fetch_log.tsv"
echo "ok $ds/$name $sz bytes md5 $md5 ($when)"
