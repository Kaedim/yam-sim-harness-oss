#!/bin/bash
# enqueue.sh <id> <taskset> <policy> <trials> <iters> [KEY=VAL ...]   e.g. enqueue.sh 001_bottles_pi05 bottles pi05 20 400
set -eu
ID=$1; TASKSET=$2; POLICY=$3; TRIALS=$4; ITERS=$5; shift 5
ENVJSON=$(python3 -c "import json,sys; print(json.dumps(dict(a.split('=',1) for a in sys.argv[1:])))" "$@")
mkdir -p /opt/bench/queue/queued
python3 -c "import json; json.dump({'id':'$ID','taskset':'$TASKSET','policy':'$POLICY','trials':$TRIALS,'iters':$ITERS,'env':$ENVJSON}, open('/opt/bench/queue/queued/$ID.json','w'), indent=1)"
echo "queued /opt/bench/queue/queued/$ID.json"
