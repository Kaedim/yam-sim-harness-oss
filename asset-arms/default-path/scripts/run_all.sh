#!/bin/bash
# Run every asset through gen_trellis.py, sequentially, one log per asset. Skips assets whose GLB already exists.
#
#   bash run_all.sh INPUTS_DIR OUT_DIR [LOG_DIR]
#
#   INPUTS_DIR  one folder per asset, named by asset id, holding the single photo: INPUTS_DIR/<asset_id>/Photo_*.HEIC
#   OUT_DIR     writes OUT_DIR/<asset_id>/<asset_id>.glb plus provenance
#   LOG_DIR     per-asset logs (default OUT_DIR/logs)
#
# Set TRELLIS_DIR to your microsoft/TRELLIS checkout if trellis is not importable on its own.
set -u
IN=${1:?usage: run_all.sh INPUTS_DIR OUT_DIR [LOG_DIR]}
OUT=${2:?usage: run_all.sh INPUTS_DIR OUT_DIR [LOG_DIR]}
LOGS=${3:-$OUT/logs}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT" "$LOGS"
for d in "$IN"/*/; do
  a=$(basename "$d"); img=$(ls "$d"/Photo_*.HEIC 2>/dev/null | head -1)
  if [ -z "$img" ]; then echo "[skip] $a (no Photo_*.HEIC)"; continue; fi
  if [ -f "$OUT/$a/$a.glb" ]; then echo "[skip] $a"; continue; fi
  echo "[run ] $a  <- $(basename "$img")"
  python "$HERE/gen_trellis.py" --image "$img" --name "$a" --out "$OUT" > "$LOGS/$a.log" 2>&1 || echo "[FAIL] $a (see $LOGS/$a.log)"
done
echo ALL_DONE
