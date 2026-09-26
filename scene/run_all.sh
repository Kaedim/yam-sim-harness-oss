#!/bin/bash
# Build a scene splat from one video, PolaRiS recipe: frames -> COLMAP -> 2DGS (30k iters) -> TSDF mesh -> pad to
# 3DGS layout -> automatic table-plateau fit. Each stage skips itself if its output exists, so re-running resumes.
#
#   bash run_all.sh VIDEO WORK_DIR 2DGS_REPO
#
#   VIDEO      phone video of the cell (e.g. a 2-5 min walk-around .MOV)
#   WORK_DIR   everything is written here; deliverables end up in WORK_DIR/out
#   2DGS_REPO  a working checkout of https://github.com/hbb1/2d-gaussian-splatting (its env must be active)
#
# Table geometry for the fit is passed through FIT_ARGS, e.g. FIT_ARGS="--table-long 1.20 --table-short 0.60".
set -u
VIDEO=${1:?usage: run_all.sh VIDEO WORK_DIR 2DGS_REPO}
R=$(mkdir -p "${2:?usage: run_all.sh VIDEO WORK_DIR 2DGS_REPO}" && cd "$2" && pwd)
TWODGS=$(cd "${3:?usage: run_all.sh VIDEO WORK_DIR 2DGS_REPO}" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
FIT_ARGS=${FIT_ARGS:-}
# OpenBLAS with more than ~24 threads corrupted the heap right after COLMAP matching on a 96-core machine; cap it.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-16} OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
L=$R/logs; mkdir -p "$L" "$R/out" "$R/data"; cd "$R"
st(){ echo "$(date +%FT%T) $1" | tee -a "$R/STATUS"; }
st "run_all start pid $$"
st "frames start"
[ -f data/images/00000.jpg ] || python "$HERE/extract_frames.py" "$VIDEO" data/images > "$L/frames.log" 2>&1 || { st "frames FAILED"; exit 1; }
st "frames done: $(tail -1 "$L/frames.log")"
st "colmap start"
[ -f data/colmap/undistorted/sparse/0/cameras.bin ] || python "$HERE/run_colmap.py" data/images data/colmap > "$L/colmap.log" 2>&1 || { st "colmap FAILED"; exit 1; }
st "colmap done: $(grep COLMAP_DONE "$L/colmap.log" | tail -1)"
st "train start (30k iters)"
cd "$TWODGS"
[ -f "$R/model/point_cloud/iteration_30000/point_cloud.ply" ] || python train.py -s "$R/data/colmap/undistorted" -m "$R/model" \
   --depth_ratio 0 --lambda_normal 0.05 --lambda_dist 0 --iterations 30000 > "$L/train.log" 2>&1 || { st "train FAILED"; exit 1; }
st "train done"
st "mesh start"
# --mesh_res must be a multiple of 512 with --unbounded
[ -f "$R/model/train/ours_30000/fuse_post.ply" ] || python render.py -m "$R/model" -s "$R/data/colmap/undistorted" --skip_train --skip_test \
   --unbounded --mesh_res 1024 > "$L/mesh.log" 2>&1 || st "mesh FAILED (non-fatal, splat still usable)"
cd "$R"
st "pad start"
python "$HERE/pad_scale2.py" model/point_cloud/iteration_30000/point_cloud.ply out/scene_2dgs_as3dgs.ply > "$L/pad.log" 2>&1 || { st "pad FAILED"; exit 1; }
st "pad done: $(tail -1 "$L/pad.log")"
st "fit start"
# shellcheck disable=SC2086
python "$HERE/fit_transform_auto.py" out/scene_2dgs_as3dgs.ply out/fit $FIT_ARGS > "$L/fit.log" 2>&1 || st "fit FAILED (non-fatal)"
cp model/point_cloud/iteration_30000/point_cloud.ply out/scene_2dgs_raw.ply
cp model/train/ours_30000/fuse_post.ply out/scene_mesh_fuse_post.ply 2>/dev/null
cp model/cfg_args out/ 2>/dev/null; cp -r data/colmap/undistorted/sparse out/colmap_sparse 2>/dev/null
cp "$L"/*.log out/ 2>/dev/null
( cd out && ls -la ) > "$L/out_listing.txt"
st "DONE  ->  $R/out"
