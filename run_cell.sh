#!/bin/bash
# Run one benchmark cell, both asset arms, exactly as published.
#
#   ./run_cell.sh configs/bottles_pi05.env kaedim
#   ./run_cell.sh configs/bottles_pi05.env polaris
#
# The config file holds every value that differs between cells. Everything else is
# the runner's default, which is what the published runs used.
set -eu
CFG=${1:?usage: run_cell.sh configs/<task>_<policy>.env <arm>}
ARM=${2:?usage: run_cell.sh configs/<task>_<policy>.env <arm>}
HARNESS_ROOT=${HARNESS_ROOT:-/opt/isaacwork}

set -a; . "$CFG"; set +a

export ASSET_ARM=$ARM
case "$ARM" in
  kaedim)  export ASSET_ROOT=${ASSET_ROOT:-$HARNESS_ROOT/assets} ;;
  polaris) export ASSET_ROOT=${ASSET_ROOT:-$HARNESS_ROOT/assets_polaris} ;;
  *)       : "${ASSET_ROOT:?set ASSET_ROOT for a custom arm}" ;;
esac
# ASSET_ARM also selects the matching scene capture and its fitted transform.

echo "cell   $(basename "$CFG" .env)"
echo "arm    $ARM   root $ASSET_ROOT"
echo "budget $ITERS x $CHUNK_STEPS steps, $TRIALS trials, TOP_H $TOP_H"
exec bash "$(dirname "$0")/harness/run_trials.sh"
