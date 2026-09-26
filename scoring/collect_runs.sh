#!/bin/bash
# Pull the reported runs' artefacts into one local directory that build_table.py can read.
#
# Each run is a prefix $BUCKET/<run_id>/ holding results.jsonl and run.log, plus job.json for
# MolmoAct2 runs (the only place MOLMO_NUM_STEPS is recorded). pi0.5 runs need no job.json and
# some have none; that is expected.
#
# Usage:  BUCKET=s3://my-bucket/yam-bench bash collect_runs.sh OUT_DIR [RUNS_LIST]
#         RUNS_LIST defaults to benchmark_runs.txt beside this script. Pass "all" to pull every
#         prefix in the bucket instead, which is for exploring, not for the reported table.
# Then:   python3 build_table.py --runs OUT_DIR --real real_reference_pertrial.csv \
#             --runs-list benchmark_runs.txt --json-out table.json

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?usage: collect_runs.sh OUT_DIR [RUNS_LIST|all]}"
LIST="${2:-$HERE/benchmark_runs.txt}"
BUCKET="${BUCKET:?set BUCKET, e.g. s3://my-bucket/yam-bench}"
REGION="${REGION:-eu-west-2}"
export AWS_PAGER=""

mkdir -p "$OUT"

if [ "$LIST" = "all" ]; then
  RUNS=$(aws s3 ls --region "$REGION" "$BUCKET/" | awk '/PRE /{print $2}' | tr -d '/')
else
  RUNS=$(sed 's/#.*//' "$LIST" | tr '+' '\n' | tr -d ' ' | grep -v '^$')
fi
echo "== pulling $(echo "$RUNS" | wc -l | tr -d ' ') runs from $BUCKET"

for j in $RUNS; do
  mkdir -p "$OUT/$j"
  for f in results.jsonl run.log job.json; do
    aws s3 cp --region "$REGION" "$BUCKET/$j/$f" "$OUT/$j/$f" --only-show-errors 2>/dev/null \
      || { [ "$f" = job.json ] || echo "   missing $j/$f"; }
  done
  # a prefix with no results is a media-only or aborted job
  [ -s "$OUT/$j/results.jsonl" ] || rm -rf "$OUT/$j"
done
echo "   $(ls "$OUT" | wc -l | tr -d ' ') runs with results"

echo
echo "done. now run:"
echo "  python3 $HERE/build_table.py --runs $OUT --real $HERE/real_reference_pertrial.csv \\"
echo "      --runs-list $HERE/benchmark_runs.txt --json-out table.json"
