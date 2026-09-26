# Default asset path (PolaRiS-style)

The object assets of the default arm. The two arms differ in their objects and their scene splat
(`scene/README.md`); the table, cameras, layout, policy, instruction and rubric are the same.

It follows what PolaRiS publishes, and uses only the information a PolaRiS user would have.

## What PolaRiS does, and where that is written down

| step | PolaRiS | source |
|---|---|---|
| geometry | multi-view photos, SAM2 segmentation, TRELLIS image-to-3D, appearance baked onto the mesh | arXiv 2512.16881 sec. 4.1 |
| metric scale | typed by a human in their scene-composition GUI, against the metric splat | published scenes carry hand-set uniform `xformOp:scale` of 0.28 / 0.13 / 0.06 |
| collider | `convexDecomposition` | `owhan/PolaRiS-Hub` `block_stack_kitchen/scene.usda` |
| mass, friction, restitution | nothing authored | same file: zero occurrences. "we set standard physics parameters for contact dynamics using default values from IsaacSim"; "Object mass parameters can be estimated by the users" |

Unauthored means PhysX defaults apply: density 1000 kg/m³ over the collider volume, friction 0.5, restitution
0.0. A collider with no material bound gets PhysX's default material (friction 0.5), not USD's schema fallback
of 0.0; we measured this in Isaac Sim (an unbound floor and a 0.5 floor give identical slides).

The PolaRiS code repo contains no asset-generation code. It is the Isaac Lab environment, the splat renderer,
the rubrics and an openpi client. The mesh step is ours to run either way.

## The recipe, exactly

- **Input:** one photo per object, the shot **without** the ChArUco board, taken for this purpose.
  `photo_to_asset_mapping.json` maps each photo to its asset id. The board shots feed the authored arm's scale
  stage and are not used here.
- **Not used anywhere in this arm:** the ChArUco board, the measurement sheet, the bill of materials, the authored
  arm's per-asset metadata, and the camera setup photos.
- **Geometry:** original `microsoft/TRELLIS`, model `TRELLIS-image-large`, single-image mode, shipped defaults:
  built-in rembg (u2net) background removal, seed 1, default sampler params, `to_glb(simplify=0.95,
  texture_size=1024)` as in TRELLIS `example.py`.
- **Scale:** `scales_by_eye.json`, one number per object: the largest dimension in metres, typed by eye from the
  photo and everyday familiarity, never checked against a tape measure. A PolaRiS user sizes the object against
  the metric room splat in their GUI, so this arm has slightly less information than theirs, not more.
- **USD:** GLB (Y up) to a Z-up USD, re-based to stand on z=0 centred in xy (the runner spawns each asset at its
  origin on the table top). Uniform `xformOp:scale` from the by-eye size. `PhysicsRigidBodyAPI` +
  `PhysxRigidBodyAPI` on the root, `PhysicsCollisionAPI` + `MeshCollisionAPI` with `convexDecomposition` on the
  mesh. No mass, no density, no physics material.

## Running it

On a Linux CUDA machine with at least 24 GB of GPU memory. Known-good: torch 2.4.1+cu124, python 3.11, nvcc 12.4.

```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS
TORCH_CUDA_ARCH_LIST=8.6 bash scripts/setup_trellis.sh TRELLIS       # then apply install gotcha 2 below

export TRELLIS_DIR=$PWD/TRELLIS
bash scripts/run_all.sh inputs out          # inputs/<asset_id>/Photo_*.HEIC -> out/<asset_id>/<asset_id>.glb
bash scripts/usd_all.sh out $HARNESS_ROOT/assets_polaris scales_by_eye.json
```

- `run_all.sh` skips objects whose GLB already exists, so it can be re-run to resume.
- `usd_all.sh` only rescales and rewrites USD. To change a size, edit `scales_by_eye.json` and re-run it; the
  geometry does not need regenerating.
- For a single object: `python scripts/gen_trellis.py --image Photo_x.HEIC --name <asset_id> --out out`, then
  `python scripts/glb_to_usd.py --glb out/<asset_id>/<asset_id>.glb --name <asset_id> --size 0.23 --out assets_polaris`.

Folder names must match the authored arm's asset ids exactly, so the runner swaps arms with one argument:

```bash
./run_cell.sh configs/bottles_pi05.env polaris          # ASSET_ROOT defaults to $HARNESS_ROOT/assets_polaris
```

`ASSET_ARM=polaris` also disables the two authored-arm spawn overrides in `harness/isaac_trials.py` (the grapes
mass override and the baseball bounding-sphere collider). Both are authored knowledge this arm should not have.

## Install gotchas

1. `xformers`: TRELLIS `setup.sh` has no row for torch 2.4.1 + cu124. `0.0.28.post1` from the cu124 index works.
2. TRELLIS calls `xops.fmha.BlockDiagonalMask`, which has moved. Patch the three files under
   `trellis/modules/sparse/attention/` to `xops.fmha.attn_bias.BlockDiagonalMask`.
3. `nvdiffrast` and `diff-gaussian-rasterization` need `pip install --no-build-isolation`, or they build in an
   env with no torch and fail.
4. `pip install --ignore-installed blinker` first, or every install aborts on a distutils-owned package.
   `setup_trellis.sh` does this.
5. `kaolin` 0.17.0 from the `torch-2.4.1_cu124` wheel index.
6. `PhysxSchema` is NVIDIA-only and absent from `usd-core`, so `glb_to_usd.py` writes the `Physx*` API schema
   names straight into the layer's `apiSchemas` list-op. They resolve inside Isaac Sim.

## Provenance

Every asset carries two json files:

- `out/<id>/provenance.json`: input image, pipeline, seed, sampler params, face count.
- `assets_polaris/<id>/<id>.json`: scale, method, bbox after scale, and the explicit statement that mass and
  material are unauthored.

`trellis_input_518.png` is the exact 518x518 RGBA image TRELLIS received.
