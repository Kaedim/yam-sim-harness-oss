#!/bin/bash
# GLB -> USD for every generated asset, sizes from a by-eye scale file, into ASSETS_DIR/<asset_id>/<asset_id>.usd
#
#   bash usd_all.sh GLB_DIR ASSETS_DIR [SCALES_JSON]
#
#   GLB_DIR      output of run_all.sh: GLB_DIR/<asset_id>/<asset_id>.glb
#   ASSETS_DIR   USD asset root, e.g. $HARNESS_ROOT/assets_polaris
#   SCALES_JSON  {asset_id: largest dimension in metres, typed by eye} (default ../scales_by_eye.json)
set -u
GLB=${1:?usage: usd_all.sh GLB_DIR ASSETS_DIR [SCALES_JSON]}
DST=${2:?usage: usd_all.sh GLB_DIR ASSETS_DIR [SCALES_JSON]}
HERE=$(cd "$(dirname "$0")" && pwd)
SCALES=${3:-$HERE/../scales_by_eye.json}
for g in "$GLB"/*/*.glb; do
  a=$(basename "$g" .glb)
  sz=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$SCALES" "$a") || { echo "[FAIL] $a has no size in $SCALES"; continue; }
  python "$HERE/glb_to_usd.py" --glb "$g" --name "$a" --size "$sz" --out "$DST" 2>&1 | grep "^\[usd\]"
done
echo USD_ALL_DONE
