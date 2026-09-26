#!/bin/bash
# pi0.5 bottles-in-bin trial with the settings Robocurve published for their real rig
# (robocurve/inspect-robots-yam issue #124 and the serve_pi05_yam.py gist linked from it):
#   GPU=N selects the GPU (default 0). 30 Hz, 640x480 capture squashed to 640x360, whole 16-step chunk, 6400 steps (Robocurve's per-trial data: bottles trials ran to max_steps 6400).
set -u
# HARNESS_ROOT holds assets, splats and trial output. BENCH_ROOT holds the job queue.
# Both default to the paths the published runs used, so setting nothing reproduces them.
HARNESS_ROOT=${HARNESS_ROOT:-/opt/isaacwork}
# RUNNER is the trial runner to execute: this repo's copy by default. The published runs executed
# an identical copy placed at $HARNESS_ROOT/isaac_trials.py; RUNNER=$HARNESS_ROOT/isaac_trials.py
# reproduces that layout exactly.
RUNNER=${RUNNER:-$(cd "$(dirname "$0")" && pwd)/isaac_trials.py}
[ -f "$RUNNER" ] || { echo "runner not found: $RUNNER" >&2; exit 2; }
RUNNER_DIR=$(cd "$(dirname "$RUNNER")" && pwd)
BENCH_ROOT=${BENCH_ROOT:-/opt/bench}
# SPLAT_ENV=1 (default) needs the 6.0.1 container and its own Omniverse cache (driver-specific shader cache,
# never shared between images or machines). SPLAT_ENV=0 uses the 5.1.0 image.
if [ "${SPLAT_ENV:-1}" != "0" ]; then
  ISAAC_IMAGE=${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}; OVCACHE=${OVCACHE:-/opt/ovcache601}
else
  ISAAC_IMAGE=${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}; OVCACHE=${OVCACHE:-/opt/ovcache}
fi
mkdir -p $BENCH_ROOT/jobs $OVCACHE/{kit,cache,nvomni,data}; chown -R 1234:1234 $OVCACHE $HARNESS_ROOT $BENCH_ROOT/jobs
docker run --rm --gpus "\"device=${GPU:-0}\"" --network host \
  --entrypoint /isaac-sim/python.sh \
  -e "TASKSET=${TASKSET:-bottles}" -e "TASK=${TASK:-}" \
  -e "CONTROL_HZ=${CONTROL_HZ:-30}" -e "CAM_W=${CAM_W:-640}" -e "CAM_H=${CAM_H:-360}" -e "CROP_H=${CROP_H:-0}" \
  -e "ITERS=${ITERS:-400}" -e "TRIALS=${TRIALS:-1}" -e "PORT=${PORT:-5566}" \
  -e "OUTDIR=${OUTDIR:-$HARNESS_ROOT/trials}" -e HARNESS_ROOT -e "SAVE_CAMS=${SAVE_CAMS:-1}" \
  -e "TOP_H=${TOP_H:-0.72}" -e TOP_TILT_DEG -e TOP_X -e TOP_HFOV_DEG -e TOP_RENDER_H -e SPECT_POS -e SPECT_TGT -e OBJ_DX -e BIN_YAW_DEG -e "JITTER=${JITTER:-0.01}" -e RIG -e ARM_X -e ARM_Y -e BACKDROP \
  -e SPLAT_ENV -e SPLAT_USD -e SPLAT_XFORM -e SPLAT_WARMUP -e SPLAT_TABLE -e TABLE_X \
  -e TABLE_MU_S -e TABLE_MU_D -e TABLE_RESTITUTION -e TABLE_R -e TABLE_G -e TABLE_B -e SUN -e DOME \
  -e SPLAT_DZ -e FSD -e STRICT_STEPS -e VIS_GATE -e VIS_CHECK -e USD_POSES -e GRAPES_MASS_KG -e SCORE_RULE -e CHUNK_STEPS -e MOLMO_NUM_STEPS -e RENDER_SETTLE -e CONVEX_OBJS -e WRIST_BACK_M -e CAM_GAIN_R -e CAM_GAIN_G -e CAM_GAIN_B -e CAM_GAMMA_R -e CAM_GAMMA_G -e CAM_GAMMA_B -e CAM_SUPERSAMPLE -e LAYOUT -e REPLAY_JSONL -e TABLE_TEX -e TABLE_TEX_SCALE -e TABLE_TEX_GAIN \
  -e FINGER_MAXFORCE -e FINGER_KP -e FINGER_KD -e WELD_ON_CONTACT -e WELD_HYST -e WELD_OPEN_M -e FINGER_MU_S -e FINGER_MU_D -e FRICTION_COMBINE -e JOINT_FRICTION -e WRIST_KP -e WRIST_KD -e GRIPPER_STROKE_S -e ARM_GRAVITY \
  -e CONTACT_OFFSET_M -e OBJ_MU_S -e OBJ_MU_D -e TOPCAM_PP -e SPRAY_YAW_DEG -e ALLOW_TOPH_MISMATCH -e GOPEN_M -e PIPELINE_PHYS -e TOPCAM_RIG -e GRAPES_COLLIDER -e CURTAIN -e RENDERER -e SPP -e EXPORT_USD \
  -e GRIP_DEBUG -e BOTTLE_ORDER -e ASSET_ROOT -e ASSET_ARM -e FILTER_FINGERS -e CLIP_JOINTS -e SPLAT_XFORM_FILE \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" -e "OMNI_KIT_ACCEPT_EULA=YES" \
  -v $OVCACHE/kit:/isaac-sim/kit/cache:rw \
  -v $OVCACHE/cache:/isaac-sim/.cache:rw \
  -v $OVCACHE/nvomni:/isaac-sim/.nvidia-omniverse:rw \
  -v $OVCACHE/data:/isaac-sim/.local/share/ov/data:rw \
  -v $HARNESS_ROOT:$HARNESS_ROOT:rw \
  -v $BENCH_ROOT:$BENCH_ROOT:rw \
  -v $RUNNER_DIR:/runner:ro \
  $ISAAC_IMAGE \
  /runner/$(basename "$RUNNER")
