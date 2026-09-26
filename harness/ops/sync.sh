#!/bin/bash
# Mirror a job directory to S3 when the box role allows it (S3_BUCKET/S3_PREFIX in /opt/bench/env).
# Silent no-op otherwise, so the pipeline never blocks on storage.
set -u
JOB=${1:?jobdir}
source /opt/bench/env 2>/dev/null || true
[ -n "${S3_BUCKET:-}" ] || exit 0
/usr/local/bin/aws s3 sync "$JOB" "s3://$S3_BUCKET/${S3_PREFIX:-yam-bench}/$(basename "$JOB")/" \
  --region "${S3_REGION:-eu-west-2}" --exclude "t??/*" --exclude "t??_cams/*" --only-show-errors 2>>/opt/bench/sync.err || true
