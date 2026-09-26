#!/bin/bash
# One worker per GPU. Claims the oldest queued job atomically, runs it, files it under done/ or failed/.
set -u
GPU=${1:-0}; B=/opt/bench; Q=$B/queue
mkdir -p $Q/queued $Q/running $Q/done $Q/failed $B/jobs
$B/notify.py "worker gpu$GPU up"
while true; do
  NEXT=$(ls $Q/queued 2>/dev/null | sort | head -1)
  if [ -z "$NEXT" ]; then sleep 60; continue; fi
  # atomic claim: mv fails for the loser when two workers race
  mv "$Q/queued/$NEXT" "$Q/running/gpu${GPU}_$NEXT" 2>/dev/null || continue
  JOB="$Q/running/gpu${GPU}_$NEXT"
  if bash $B/run_job.sh "$JOB" "$GPU"; then mv "$JOB" "$Q/done/$NEXT"; else mv "$JOB" "$Q/failed/$NEXT"; fi
  sleep 5
done
