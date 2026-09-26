#!/bin/bash
# Run one job file on one GPU, hands-off. Resumable: the runner skips trials already in results.jsonl.
# job json: {"id": "...", "taskset": "bottles", "policy": "pi05", "trials": 20, "iters": 400, "env": {...}}
set -u
JOBFILE=${1:?job json}; GPU=${2:-0}
B=/opt/bench; source $B/env 2>/dev/null || true
# GPU is the worker SLOT (drives the port), GPU_DEVICE (from $B/env, sourced above) is the CUDA/docker
# device. On a single-GPU box set GPU_DEVICE=0 and run yam-worker@0 and yam-worker@1 side by side: Isaac uses ~3 GB
# of the 46 GB and the GPU idles between control steps, so two concurrent jobs run at ~1.6x the throughput of one.
DEV=${GPU_DEVICE:-$GPU}
J() { python3 -c "import json,sys; j=json.load(open('$JOBFILE')); v=j$1; print(v if not isinstance(v,(dict,list)) else json.dumps(v))" 2>/dev/null; }
ID=$(J "['id']"); TASKSET=$(J "['taskset']"); POLICY=$(J "['policy']"); TRIALS=$(J "['trials']"); ITERS=$(J "['iters']")

# --- JOBENV_EXPORTED ---
# Export the job's env HERE, before the policy server / bridge start block reads it, so e.g.
# MOLMO_NUM_STEPS from a job file reaches the bridge and not only the container launch below.
while IFS= read -r _kv; do [ -n "$_kv" ] && export "$_kv"; done < <(
  python3 -c "import json,sys; [print('%s=%s'%(k,v)) for k,v in json.load(open(sys.argv[1])).get('env',{}).items()]" "$JOBFILE")
echo "[jobenv] exported: MOLMO_NUM_STEPS=${MOLMO_NUM_STEPS:-unset} MOLMO_CHUNK=${MOLMO_CHUNK:-unset} SPLAT_DZ=${SPLAT_DZ:-unset}"
# --- end JOBENV_EXPORTED ---
OUT=$B/jobs/$ID; mkdir -p $OUT/clips; chown -R 1234:1234 $OUT
LOG=$OUT/run.log
# One port range per policy so a server left running from the previous job can never answer for the
# wrong policy (a molmoact2 job finding a pi0.5 server on its port would skip the bridge and run pi0.5).
if [ "$POLICY" = "molmoact2" ]; then PORT=$((5577 + GPU)); else PORT=$((5566 + GPU)); fi
CHUNK=16; [ "$POLICY" = "molmoact2" ] && CHUNK=${MOLMO_CHUNK:-30}   # pi0.5 chunks are 16 actions, MolmoAct2 30
# CHUNK is the action horizon (MolmoAct2 always returns 30), NOT MOLMO_NUM_STEPS, which is the flow-matching
# denoising step count (client default 10). Exported so the container computes its step budget from the same
# chunk the policy actually serves, rather than from its own default of 16.
export CHUNK_STEPS=$CHUNK
# Top-camera height is a property of the POLICY: Robocurve mount it at ~88 cm for MolmoAct2 and 72 cm for pi0.5.
# The launcher's flat TOP_H=0.72 default cannot know which policy is behind the socket, so set it here from the
# policy; a job that names TOP_H explicitly still wins, because $EXTRA is applied after this.
if [ "$POLICY" = "molmoact2" ]; then export TOP_H=${TOP_H:-0.88}; else export TOP_H=${TOP_H:-0.72}; fi
# valid (non-voided) trial count, the only number that decides whether a job is finished
OK() { python3 -c "import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if l.strip() and not json.loads(l).get(\"voided\")))
except Exception: print(0)" "$1" 2>/dev/null || echo 0; }
$B/notify.py "job $ID START on gpu$GPU: $POLICY $TASKSET, $TRIALS trials x $((ITERS*CHUNK)) steps $( [ -s $OUT/results.jsonl ] && echo "(resuming, $(wc -l < $OUT/results.jsonl) done)" )"
# policy server for this GPU
if [ "$POLICY" = "pi05" ]; then
  if ! ss -ltn | grep -q ":$PORT "; then
    (cd /opt/isaacwork && CUDA_VISIBLE_DEVICES=$DEV PORT=$PORT nohup /opt/openpi/.venv/bin/python pi05_server.py > $B/pi05_server_gpu$GPU.log 2>&1 &)
    for i in $(seq 1 120); do ss -ltn | grep -q ":$PORT " && break; sleep 5; done
    ss -ltn | grep -q ":$PORT " || { $B/notify.py "job $ID FAILED: pi05 server on port $PORT did not come up"; exit 2; }
  fi
elif [ "$POLICY" = "molmoact2" ]; then
  # MolmoAct2 ships an HTTP server (allenai/molmoact2 examples/yam/host_server_yam.py, :8202). The
  # bridge (molmoact2_bridge.py) speaks the trial's socket protocol on $PORT and forwards to it, so the trial is unchanged.
  # Both run in the molmoact2 uv venv (/opt/molmoact2). One server per box:
  # it holds ~15 GB of GPU memory; the bridge is per GPU/port.
  if ! curl -sf http://127.0.0.1:8202/act >/dev/null 2>&1 && ! ss -ltn | grep -q ":8202 "; then
    (cd /opt/molmoact2 && HF_HOME=/opt/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$DEV nohup .venv/bin/python -u examples/yam/host_server_yam.py --host 127.0.0.1 --port 8202 --dtype bfloat16 > $B/molmo_server.log 2>&1 &)
    for i in $(seq 1 120); do ss -ltn | grep -q ":8202 " && break; sleep 5; done
    ss -ltn | grep -q ":8202 " || { $B/notify.py "job $ID FAILED: MolmoAct2 server on :8202 did not come up (see molmo_server.log)"; exit 2; }
  fi
  # whatever holds the bridge port must be the bridge; kill anything else, then (re)start it.
  # One bridge PER PORT (two slots on one box each need their own).
  if ss -ltn | grep -q ":$PORT " && ! pgrep -f "molmoact2_bridge.py" >/dev/null; then fuser -k $PORT/tcp 2>/dev/null; sleep 1; fi
  # --- BRIDGE_NUMSTEPS_GUARD ---
  # A bridge left running from an earlier job serves this one at ITS num_steps, not ours.
  _want=${MOLMO_NUM_STEPS:-10}   # 10 is the real rig default (config.py num_steps)
  for _bp in $(pgrep -f "molmoact2_bridge.py"); do
    _bport=$(tr "\0" "\n" < /proc/$_bp/environ 2>/dev/null | sed -n "s/^PORT=//p")
    _bns=$(tr "\0" "\n" < /proc/$_bp/environ 2>/dev/null | sed -n "s/^MOLMO_NUM_STEPS=//p")
    if [ "$_bport" = "$PORT" ] && [ "${_bns:-10}" != "$_want" ]; then
      echo "[bridge] :$PORT running num_steps=${_bns:-30}, job wants $_want -> restarting"
      kill $_bp 2>/dev/null; sleep 2
    fi
  done
  # --- end BRIDGE_NUMSTEPS_GUARD ---
  if ! ss -ltn | grep -q ":$PORT "; then
    (cd /opt/isaacwork && PORT=$PORT MOLMO_URL=http://127.0.0.1:8202/act MOLMO_NUM_STEPS=${MOLMO_NUM_STEPS:-10} nohup /opt/molmoact2/.venv/bin/python -u molmoact2_bridge.py > $B/molmo_bridge_gpu$GPU.log 2>&1 &)
    for i in $(seq 1 60); do ss -ltn | grep -q ":$PORT " && break; sleep 2; done
    ss -ltn | grep -q ":$PORT " || { $B/notify.py "job $ID FAILED: MolmoAct2 bridge on port $PORT did not come up: $(tail -1 $B/molmo_bridge_gpu$GPU.log)"; exit 2; }
  fi
  $B/notify.py "job $ID: MolmoAct2 bridge on :$PORT -> server :8202, $(grep -c . $B/molmo_bridge_gpu$GPU.log) log lines"
else
  $B/notify.py "job $ID FAILED: policy $POLICY not wired yet"; exit 2
fi
# per-trial clips + notify lines, in the background for this job
(cd /opt/isaacwork && nohup python3 incremental_clips.py $OUT $LOG > $OUT/clips.log 2>&1 &)
# Mirror finished trials to S3 every 10 min while the job runs, so a wiped box loses at most
# one trial. The final sync below still runs at the end.
(while sleep 600; do $B/sync.sh $OUT; done) & SYNC_LOOP=$!
# the trial runner (extra env from the job file)
# One k=v per line into an array, so values containing spaces survive (an unquoted $EXTRA made `env` exec the 2nd word).
mapfile -t EXTRA < <(python3 -c "import json; [print('%s=%s'%(k,v)) for k,v in json.load(open('$JOBFILE')).get('env',{}).items()]")
for attempt in 1 2 3; do
  env SAVE_CAMS=1 TOP_H=$TOP_H "${EXTRA[@]}" GPU=$DEV TASKSET=$TASKSET TRIALS=$TRIALS ITERS=$ITERS PORT=$PORT OUTDIR=$OUT CHUNK_STEPS=$CHUNK bash ${RUN_TRIALS:-$B/run_trials.sh} >> $LOG 2>&1
  # Count NON-VOIDED rows. results.jsonl has one line per ATTEMPTED trial, voided included, so a raw
  # wc -l would reach TRIALS with voids in it and skip the retry that re-runs them.
  if grep -aq TRIALSDONE $LOG && [ "$(OK $OUT/results.jsonl)" -ge "$TRIALS" ]; then break; fi
  $B/notify.py "job $ID: runner exited early (attempt $attempt), $(OK $OUT/results.jsonl)/$TRIALS valid trials done, restarting"
  sleep 30
done
# Let the live cutter finish its last clip (its ffmpeg child would otherwise keep writing the same
# file compose_all is about to reuse, and the clip ships corrupt).
for i in $(seq 1 60); do pgrep -f "incremental_clips.py $OUT" >/dev/null || break; grep -q INCDONE $OUT/clips.log 2>/dev/null && break; sleep 5; done
pkill -f "incremental_clips.py $OUT" 2>/dev/null
while pgrep -f "ffmpeg.*$OUT/t[0-9][0-9]/" >/dev/null; do sleep 3; done
kill $SYNC_LOOP 2>/dev/null
DONE=$(OK $OUT/results.jsonl)
[ "$DONE" -ge "$TRIALS" ] || { $B/notify.py "job $ID FAILED after 3 attempts: $DONE/$TRIALS trials"; $B/sync.sh $OUT; exit 1; }
# videos + stats
(cd /opt/isaacwork && python3 compose_all.py $OUT > $OUT/compose.log 2>&1) || $B/notify.py "job $ID: video compose failed, see compose.log"
SUMMARY=$(/opt/openpi/.venv/bin/python $B/stats.py $OUT $TASKSET 2>&1 | tail -1)
$B/sync.sh $OUT
$B/notify.py "job $ID DONE: $SUMMARY"
exit 0
