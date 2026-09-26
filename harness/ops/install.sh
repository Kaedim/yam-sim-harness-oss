#!/bin/bash
# Install the per-GPU queue on a box that already has $HARNESS_ROOT (runner, assets, splats),
# the policy checkpoint, and docker with the Isaac image pulled.
set -eu
B=${BENCH_ROOT:-/opt/bench}; mkdir -p $B/queue/{queued,running,done,failed} $B/jobs
HERE=$(cd "$(dirname "$0")" && pwd)
cp $HERE/{notify.py,sync.sh,stats.py,run_job.sh,worker.sh,enqueue.sh} $B/
cp $HERE/../run_trials.sh $HERE/../isaac_trials.py $B/    # the launcher runs the runner beside it
chmod +x $B/*.sh $B/*.py
[ -f $B/env ] || printf 'SLACK_WEBHOOK=\n# S3_BUCKET=\n# S3_PREFIX=yam-bench\n# S3_REGION=\n' > $B/env
cp $HERE/yam-worker@.service /etc/systemd/system/
systemctl daemon-reload
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for i in $(seq 0 $((N-1))); do systemctl enable --now yam-worker@$i; done
systemctl --no-pager --type=service | grep yam-worker || true
