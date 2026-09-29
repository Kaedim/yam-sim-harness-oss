# Scene: a Gaussian splat of the real cell

The environment behind the robot is a Gaussian splat of the real room, not an authored backdrop. Appearance
comes from the splat. Physics comes from authored colliders (table, ground) that are hidden from every camera.
The scene is one of the two things that differ between the arms (the other is the objects). Both arms render
through the same Isaac Sim RTX renderer.

| | authored arm | default arm |
| --- | --- | --- |
| splat | 3DGS of the evaluator's room, 796k gaussians (training tool not recorded) | COLMAP + [2DGS](https://github.com/hbb1/2d-gaussian-splatting), 30k iterations, same video |
| format fix | none, already 3DGS layout | `pad_scale2.py`: 2DGS has two scales, Isaac's converter needs three |
| alignment to the sim | table-plane fit (`fit_splat_transform.py`, plateau height set by hand), then a hand correction of the translation | same alignment and cut as the authored arm: registered onto the aligned authored splat (`register_to_reference.py`), so it inherits that alignment |
| cut | `cut_splat_to_environment.py` | the same filters via `cut_generic.py`, which reads the registration and has two 2DGS switches (`CUT_NEEDLE=0`, `CUT_KEEP_FLAT=1`) |
| renderer | Isaac RTX, via the shipped gsplat converter | same |

The default arm follows a PolaRiS-inspired scene recipe ([arXiv 2512.16881](https://arxiv.org/abs/2512.16881) sec. 4
and App. B, and `docs/custom_environments.md` in `arhanjain/PolaRiS`), with the departures listed below.

## What is in this folder

Build the splat from a video:

| script | does |
| --- | --- |
| `extract_frames.py` | video to sharp JPEG frames (keeps the sharpest of every N) |
| `run_colmap.py` | pycolmap SfM, undistorted to PINHOLE, laid out for the 2DGS/3DGS loaders |
| `run_all.sh` | frames, COLMAP, 2DGS training, TSDF mesh, pad, fit, in one resumable run |
| `pad_scale2.py` | 2DGS PLY (2 scales) to 3DGS-layout PLY (3 scales) |
| `ply_to_usd.py` | PLY to USD with Isaac Sim's own gsplat converter |

Align it to the sim world:

| script | does |
| --- | --- |
| `fit_transform_auto.py` | automatic table-plateau fit, writes both 180 deg yaw candidates. Use this for a new capture. |
| `fit_splat_transform.py` | the same fit with the plateau height given by hand. This is what produced the authored-arm transform. |
| `register_to_reference.py` | register a second splat of the same room onto an aligned one (scale from table height, yaw and shift from the table band). This is what produced the default-arm transform. |
| `bake_transform_into_ply.py` | apply a 4x4 to every gaussian (positions, rotations, scales) |

Cut it to the environment:

| script | does |
| --- | --- |
| `cut_splat_to_environment.py` | remove the captured tabletop contents and the gaussians that streak |
| `cut_generic.py` | the same cut, transform read from `register_to_reference.py`'s `fit.json`; used for the default-arm scene |
| `filter_ply.py` | one filter at a time on a raw cloud (far tail, or large gaussians), for diagnosing render problems |

## Requirements

- A phone video of the cell: a 2 to 5 minute slow walk-around, the table and the space behind it in view.
- The dimensions of your real table (long edge, short edge, top height). Scale comes from the table, not a board.
- Reconstruction (any Linux CUDA machine, 24 GB GPU is enough):
  - `pycolmap`, `opencv-python`, `imageio-ffmpeg`, `numpy`, `plyfile`
  - a working checkout of `hbb1/2d-gaussian-splatting`, set up per its own README
- Conversion and trials: the Isaac Sim **6.0.1 container** (`nvcr.io/nvidia/isaac-sim:6.0.1`). See Known issues.

## From a video to a splat the harness can load

Paths below are examples. `$W` is your work directory.

**1. Build the splat** (frames, COLMAP, 2DGS 30k, mesh, pad, fit):

```bash
FIT_ARGS="--table-long 1.60 --table-short 0.80 --table-centre 0.42 0 0.750" \
  bash scene/run_all.sh my_cell.mov $W /path/to/2d-gaussian-splatting
```

Or stage by stage:

```bash
python scene/extract_frames.py my_cell.mov $W/data/images            # --fps 3 --keep-group 2
python scene/run_colmap.py $W/data/images $W/data/colmap             # --threads 32
cd /path/to/2d-gaussian-splatting
python train.py -s $W/data/colmap/undistorted -m $W/model \
  --depth_ratio 0 --lambda_normal 0.05 --lambda_dist 0 --iterations 30000
python render.py -m $W/model -s $W/data/colmap/undistorted --skip_train --skip_test --unbounded --mesh_res 1024   # optional mesh
cd -
python scene/pad_scale2.py $W/model/point_cloud/iteration_30000/point_cloud.ply $W/out/scene_2dgs_as3dgs.ply
```

If you trained a standard 3DGS instead, skip `pad_scale2.py`.

**2. Fit the splat to the sim world.** `--table-centre` is where your sim table's top centre is, in metres.

```bash
python scene/fit_transform_auto.py $W/out/scene_2dgs_as3dgs.ply $W/out/fit \
  --table-long 1.60 --table-short 0.80 --table-centre 0.42 0 0.750
```

- Read `fit_report.json`: `picked` should show a short edge near your real one and a table height near your
  real one. These two numbers are not used by the fit, so they are a real check.
- Look at `plateau_topdown.png`: the table outline should be a clean rectangle.
- Pick yaw candidate A or B. `auto_pick` is a heuristic tuned on one room (see Known issues). Confirm by rendering.

**3. Cut the tabletop contents out.** The output stays in the splat's own frame; the transform only decides what
to cut. Size the slab to your table.

```bash
python scene/cut_splat_to_environment.py $W/out/scene_2dgs_as3dgs.ply $W/out/scene_env.ply \
  --matrix-file $W/out/fit/M_math_A.txt \
  --slab-x -0.17 0.73 --slab-y -0.85 0.85 --slab-z 0.750 1.20
```

**4. Convert to USD**, inside the Isaac Sim 6.0.1 container:

```bash
/isaac-sim/python.sh scene/ply_to_usd.py $W/out/scene_env.ply $W/out/scene_env.usd
```

**5. Point the runner at it** (`harness/isaac_trials.py` reads these):

```bash
export SPLAT_USD=$W/out/scene_env.usd
export SPLAT_XFORM_FILE=$W/out/fit/splat_xform_A.txt
export SPLAT_DZ=0.0
```

| variable | meaning |
| --- | --- |
| `SPLAT_ENV` | `1` (default) renders the splat; `0` runs the same trial with no splat, the control. |
| `SPLAT_USD` | the splat USD, referenced onto `/World/splat`. Unset, it follows `ASSET_ARM`: `$HARNESS_ROOT/splat_env2_k080.usd` (authored) or `$HARNESS_ROOT/polaris_splat/scene/polaris_env_final.usd` (default). |
| `SPLAT_XFORM` | splat to sim transform, 16 floats in USD `Matrix4d` row-major layout (translation in slots 12, 13, 14). Default is the authored-arm transform. |
| `SPLAT_XFORM_FILE` | a text file holding the same 16 numbers. Wins over `SPLAT_XFORM`. Use it for your own splat: 16 space-separated numbers do not survive some job queues. |
| `SPLAT_DZ` | metres added to the transform's z translation, applied last. A single number, for fine height trims. |
| `SPLAT_TABLE` | which table the cameras see: `splat` (default, the authored table is collider only), `wood`, or anything else for the authored table painted grey. |

- The `SPLAT_XFORM_FILE` and `SPLAT_DZ` defaults (`-0.005` authored, `-0.0105` default arm) are keyed on the two
  published USD paths. For any other `SPLAT_USD`, the file default is empty, `SPLAT_DZ` defaults to 0 and the
  transform falls back to the authored-arm one. So always set `SPLAT_XFORM_FILE` with your own splat.
- If the runner's transform cannot be changed, bake instead: `bake_transform_into_ply.py` with your `M_fit`,
  then again with the runner's transform and `--inverse`. The runner's default then lands the cloud where
  `M_fit` puts it.

**6. Check it.** Run one trial and look at the first frame from every camera. The captured table should sit
under the sim objects, and the room should not be mirrored.

## Rebuilding the default-arm scene

The default-arm scene was registered onto the aligned authored splat rather than fitted from scratch, then cut
with the 2DGS switches on:

```bash
CAMS_JSON=colmap_cam_centers.json \
  python scene/register_to_reference.py scene_2dgs_as3dgs.ply fit/ authored_splat.ply   # writes fit.json, splat_xform.txt
CUT_NEEDLE=0 CUT_KEEP_FLAT=1 \
  python scene/cut_generic.py scene_2dgs_as3dgs.ply fit/fit.json scene_env.ply
/isaac-sim/python.sh scene/ply_to_usd.py scene_env.ply polaris_env_final.usd
```

`colmap_cam_centers.json` holds the COLMAP camera centres in the splat frame; it fixes which way is up.
The runner reads `fit/splat_xform.txt` through `SPLAT_XFORM_FILE` for the default arm.

## Departures from PolaRiS

All forced by the inputs or the runner, and the same for both arms.

| PolaRiS | here | why |
| --- | --- | --- |
| ChArUco board in the video for scale and up | table-plateau fit (`fit_transform_auto.py`) | no board in the scene capture |
| own surfel rasteriser, mask-composited over Isaac | Isaac RTX SPG renderer via the gsplat converter | our runner, same renderer for both arms |
| TSDF mesh is the scene collider | authored hidden table collider, mesh saved but unused | keeps both arms on the same physics; using the mesh is a separate decision |
| GPU COLMAP, sequential + vocab tree | pycolmap CPU SIFT, exhaustive | the wheel has no CUDA; the vocab-tree auto-download aborts the process |

## Known issues

- **Use the Isaac Sim 6.0.1 container.** It is the only build we found with both the splat extensions and
  working offscreen capture:

  | build | gsplat / spg / nurec extensions | `Camera.get_rgba()` |
  | --- | --- | --- |
  | `nvcr.io/nvidia/isaac-sim:5.1.0` container | none | works |
  | Isaac Sim 6.0.1 host install | all three | returns a 0-dim array |
  | `nvcr.io/nvidia/isaac-sim:6.0.1` container | all three | works |

- **Launch args.** The splat needs these in the `SimulationApp` launch config's `extra_args` (the runner sets
  them when `SPLAT_ENV=1`):
  `--enable omni.rtx.spg --enable isaacsim.replicator.nurec_utils --/renderer/multiGpu/enabled=false`
- **Reference the splat onto a typeless prim.** `add_reference_to_stage` defines a typed `Xform`, whose local type
  overrides the referenced `ParticleField3DGaussianSplat`, and then nothing treats it as a splat. The runner uses
  `stage.DefinePrim(path)` with no type.
- **Warm up the renderer.** RTPT accumulates; with only a few ticks `get_rgba()` has no frame yet. The runner
  steps 1200 times by default (`SPLAT_WARMUP`). The first run in a fresh container also compiles shaders for
  several minutes at low GPU use, which looks like a hang and is not one.
- **180 deg yaw ambiguity.** PCA of the table cannot tell the two ends apart. `fit_transform_auto.py` writes both
  and auto-picks by which side has more above-table mass behind the table. That rule was calibrated on one room
  and is not geometry, so confirm by rendering: with the wrong candidate the room is mirrored.
- **Cut thin, and never cut on opacity.** A large cut exposes volume the capture never saw, which renders as
  smear. Culling low-opacity gaussians removes surfaces, not haze. Cap gaussian size instead. Details are in
  `cut_splat_to_environment.py`.
- **The cut leaves a hole where the captured tabletop was.** The trial covers it with the authored table
  (`SPLAT_TABLE`).
- **Residual alignment offset on the authored arm.** The plateau fit got the scale right but the placement off
  by 0.43 m in x and 0.11 m in z. The published transform includes a hand correction made in the Isaac GUI.
  Check your own fit against landmarks the same way.
- **2DGS `render.py --unbounded`** needs `--mesh_res` to be a multiple of 512.
- **pycolmap threads.** On a 96-core machine OpenBLAS with more than about 24 threads corrupted the heap right
  after matching. `run_all.sh` caps `OPENBLAS_NUM_THREADS=16`. Feature extraction is skipped if
  `database.db` exists, so a crashed run resumes at mapping.
- **Pinhole cameras only** for 3DGS in Isaac Sim.
