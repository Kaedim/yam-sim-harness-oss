"""N policy trials in ONE Isaac session, with a per-trial verdict and frames for a montage.

One container launch per trial would cost ~40 s of Isaac startup for ~90 s of work,
so over hundreds of trials most of the time would be startup. Here the
app boots once and the world is reset between trials, with the bottle placement
re-jittered each time so repeats are independent attempts rather than the same run.

Physics carries the settings that took arm-vs-bottle penetration from 7.63 mm to
0.14 mm: contactOffset 6 mm and restOffset 0 on every collider (they were unset),
240 Hz, solver 64/16. Stabilization is OFF -- it damps jitter but freezes bodies in
non-physical rest poses, which is what made a bottle stand tilted.
"""
import os, socket, struct, pickle, json, numpy as np
from isaacsim import SimulationApp
_REND = os.environ.get("RENDERER", "RaytracedLighting")
_cfg = {"headless": True, "renderer": _REND, "width": 640, "height": 480}
# SPLAT_ENV=1 (default) renders Robocurve's room as a gaussian splat (scene/README.md). It needs
# the 6.0.1 CONTAINER (5.1.0 ships no gsplat/nurec/spg; the 6.0.1 host install cannot capture), the
# SPG + NuRec extensions enabled at launch, and the multi-GPU renderer off. It defaults on so a job
# cannot silently run a bare grey box instead of the real cell. Everything else about the trial is
# identical, so SPLAT_ENV=0 on the same image is the control arm of the A/B.
SPLAT_ENV = os.environ.get("SPLAT_ENV", "1") not in ("0", "", "no")
if SPLAT_ENV:
    _cfg["extra_args"] = ["--enable", "omni.rtx.spg", "--enable", "isaacsim.replicator.nurec_utils",
                          "--/renderer/multiGpu/enabled=false"]
# FSD: Fabric Scene Delegate. With the splat, rigid bodies can render at the wrong body's transform
# (e.g. the Rubik's cube drawn on the left gripper in policy frames). Hydra-through-Fabric is the
# suspect. FSD=0 makes Hydra read USD, which PhysX writes every step under USD_POSES. FSD=1 forces it
# on. Unset leaves the container default.
_FSD = os.environ.get("FSD")
if _FSD is not None:
    _cfg.setdefault("extra_args", [])
    if _FSD not in ("0", "", "no"):
        _cfg["extra_args"] += ["--/app/useFabricSceneDelegate=true",
                               "--/app/usdrt/scene_delegate/enableProxyCubes=false",
                               "--/app/usdrt/scene_delegate/geometryStreaming/enabled=false"]
    else:
        _cfg["extra_args"] += ["--/app/useFabricSceneDelegate=false"]
if _REND == "PathTracing":
    _cfg["samples_per_pixel_per_frame"] = int(os.environ.get("SPP", "16"))
app = SimulationApp(_cfg)
print("[app ] extra_args %s" % _cfg.get("extra_args", []), flush=True)

from pxr import UsdGeom, UsdLux, UsdPhysics, PhysxSchema, Gf, Usd, PhysicsSchemaTools
# Objects can render at a pose that is neither their physics pose nor their USD pose (the Rubik's
# cube drew 30 cm to the left of where PhysX and USD both put it; the spray bottle and baseball drew
# off-table with the raw splat). The renderer takes rigid-body transforms from Fabric, and that
# copy went stale for bodies at rest. Route poses through USD instead: PhysX writes them there every
# step and the renderer reads them. A dozen bodies, so the cost is nil. USD_POSES=0 restores Fabric.
import carb.settings as _carb_settings
if os.environ.get("USD_POSES", "1") not in ("0", "", "no"):
    _cs = _carb_settings.get_settings()
    _cs.set("/physics/updateToUsd", True)
    _cs.set("/physics/updateVelocitiesToUsd", False)
    _cs.set("/physics/fabricUpdateTransformations", False)
    print("[phys] rigid-body poses -> USD (updateToUsd on, Fabric transform updates off)", flush=True)
from omni.physx import get_physx_simulation_interface
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices as q2r
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# WHICH VARIABLES MATTER
#
# This file reads about 100 environment variables. The reported runs set 7 to 12 of them,
# 14 distinct across the set: the launcher passes nine on every run and the job record adds
# a few more. Everything else was left at its default.
#
# The benchmark configuration is the twelve variables listed in configs/*.env, one file
# per reported cell, with values read back out of the published runs' own logs. To
# reproduce a cell, use those files rather than setting anything by hand:
#
#     ./run_cell.sh configs/bottles_pi05.env kaedim
#
# Every other variable here is development surface: physics sweeps, camera colour
# matching, gripper welding, alternate layouts, ablation toggles. They default to off or
# to the published value. They are kept because this is the code that produced the
# numbers, and because a reader reproducing an ablation needs them. Setting one means
# leaving the benchmark configuration, and scoring/build_table.py will reject the run.
# ---------------------------------------------------------------------------

# HARNESS_ROOT is where the harness expects to find assets, splats and scratch space. Everything
# below hangs off it, so a different machine needs this one variable and nothing else. The default
# is the path the published runs used, so setting nothing reproduces them exactly.
HARNESS_ROOT = os.environ.get("HARNESS_ROOT", "/opt/isaacwork").rstrip("/")

# ASSET_ROOT selects the asset arm. Folder names are identical across arms, so nothing else changes.
#   $HARNESS_ROOT/assets          authored: single-photo image-to-3D mesh scaled from a ChArUco clip,
#                                 pipeline-chosen colliders, authored mass/friction/restitution (default)
#   $HARNESS_ROOT/assets_polaris  default path: single-image mesh, scale by eye, convexDecomposition,
#                                 NO mass/material authored (PhysX defaults). ASSET_ARM=polaris also
#                                 disables the two authored-arm spawn overrides below (grapes mass,
#                                 baseball bounding sphere).
# Centre of mass and inertia are engine-derived in both arms.
A_ROBOT = HARNESS_ROOT + "/assets"         # the YAM USD, always from here: the robot is not the variable
A = os.environ.get("ASSET_ROOT", A_ROBOT)
ASSET_ARM = os.environ.get("ASSET_ARM", "kaedim")
TT = 0.75   # table top surface, measured at Robocurve (0.735 to the underside, 0.75 to the top)
print("[assets] arm=%s root=%s" % (ASSET_ARM, A), flush=True)
# Which physical rig to model. See the camera block below for what differs.
#   yamlab    our asset's own calibrated workstation
#   molmoact2 the BimanualYAM rig from the allenai/molmoact2 issue threads (default)
# The evaluator's cell follows the MolmoAct2 setup, so that is the default. yamlab stays authoritative for the
# ROBOT only (gains, gripper travel, fingertip keypoints), never for the cell.
RIG = os.environ.get("RIG", "molmoact2")
TASKSET = os.environ.get("TASKSET", "bottles")
# Instruction strings are the evaluator's own, verbatim, as published on their results page
# (kaedim.pages.dev/real-world-evals, "Tasks and rubrics"); all twenty reported run logs print
# exactly these. clear_table really is "blue bin" in their instruction.
_DEFAULT_TASK = {"bottles":     "put 2 glass bottles in the blue bin",
                 "latte":       "move the steel latte art cup onto the cutting board",
                 "clear_table": "put all items on the table inside the blue bin",
                 "blocks":      "stack the red block on the blue block",
                 "bowls":       "stack the bowls on the sides on top of the center bowl"}[TASKSET]
# Default chunk counts at pi0.5's 16 steps per chunk. The reported runs never rely on these: every
# configs/*.env sets ITERS to the real budget for its policy (_REAL_STEPS below), and the blocks and
# bowls defaults here (400 chunks, 6400 steps) disagree with the real 3600, so the budget check below
# refuses to start those tasks without a config file. ITERS from the environment wins.
_DEFAULT_ITERS = {"bottles": 400, "latte": 225, "clear_table": 800, "blocks": 400, "bowls": 400}[TASKSET]
TASK = os.environ.get("TASK") or _DEFAULT_TASK
TRIALS = int(os.environ.get("TRIALS", "20"))
# pi0.5's chunk is 16, from the checkpoint's own model card ("open-loop action MSE
# (16-step chunks, n=20), 0.00206"). MolmoAct2's is 30; they are different policies.
# The evaluator's per-trial data: every bottles-in-bin trial ran to max_steps = 6400 at 30 Hz
# (213 s of motion, 295-560 s wall), termination_reason max_steps on all 20.
# 400 chunks x 16 = 6400 steps. (clear-table 12800, latte 3600)
ITERS = int(os.environ.get("ITERS", str(_DEFAULT_ITERS)))
# Real per-task budgets, counted from the evaluator's per-trial recorded actions (actions.jsonl):
# latte 3600, bowls 3600, blocks 3600 (both policies), bottles 6400 (MolmoAct2 run07), clear_table
# 12800 (pi0.5 run15, MolmoAct2 run14). Their results page says "3600 for all"; the trajectory files
# disagree and win. A run at the wrong budget is refused unless STRICT_STEPS=0.
_REAL_STEPS = {"bottles": 6400, "latte": 3600, "clear_table": 12800, "bowls": 3600, "blocks": 3600}
_CHUNK = int(os.environ.get("CHUNK_STEPS", os.environ.get("MOLMO_NUM_STEPS", "16")))
# The runner is never told which policy is behind the socket, but the chunk length is unambiguous:
# MolmoAct2 replans every 30 steps, pi0.5 every 16 (measured on Robocurve's own recorded trajectories).
_IS_MOLMO = (_CHUNK == 30) or os.environ.get("IS_MOLMO", "") in ("1", "yes")
# Executed action horizon. Separate from the solver step count; see the note at the ask() call site.
_HORIZON = int(os.environ.get("MOLMO_HORIZON", "0"))
if TASKSET in _REAL_STEPS and os.environ.get("STRICT_STEPS", "1") not in ("0", "", "no"):
    if abs(ITERS * _CHUNK - _REAL_STEPS[TASKSET]) > 0.02 * _REAL_STEPS[TASKSET]:
        raise SystemExit("[budget] ITERS %d x %d steps/chunk = %d, real max_steps for %s is %d. "
                         "Fix ITERS or set STRICT_STEPS=0 for a deliberate off-budget run."
                         % (ITERS, _CHUNK, ITERS * _CHUNK, TASKSET, _REAL_STEPS[TASKSET]))
print("[budget] %s: %d chunks x %d = %d steps (real %s)" % (TASKSET, ITERS, _CHUNK, ITERS * _CHUNK, _REAL_STEPS.get(TASKSET, "n/a")), flush=True)
PORT = int(os.environ.get("PORT", "5566"))
OUT = os.environ.get("OUTDIR", HARNESS_ROOT + "/trials")
# Control rate. robocurve/inspect-robots-yam (public), src/inspect_robots_yam/config.py, defaults
# to control_hz 10.0, but their published pi0.5 recipe (issue #124) passes -E control_hz=30, and
# issue #145 confirms the Kaedim pi0.5 trials stored frames at 30 Hz. The same config.py gives
# gripper_stroke_s 1.0 and gripper_open 1.0 / gripper_closed 0.0 (1 = open).
CONTROL_HZ = float(os.environ.get("CONTROL_HZ", "30.0"))
GRIPPER_STROKE_S = 1.0
# Seconds for the jaws to travel their full stroke. 0 restores the old instantaneous step (kept only for A/B).
_GSTROKE = float(os.environ.get("GRIPPER_STROKE_S", str(GRIPPER_STROKE_S)))
_GRIP_DBG = os.environ.get("GRIP_DEBUG", "") in ("1", "yes")
_GCMD = {}          # per-arm commanded gripper target, rate-limited; reset at the start of every trial
_DOF_LIM = {}       # per-arm (lower, upper) runtime joint limits, read after Articulation.initialize()
# What the policy is served. Issue #124 passes -P cam_height=360 -P cam_width=640. Their embodiment
# captures 640x480 and cv2.resize()s it to 640x360 (embodiment.py:627), a plain squash, not a centre
# crop (the intrinsics are rescaled independently in x and y). So we render the full 4:3 field at the
# true intrinsics and squash on the way out. openpi then letterboxes the 640x360 to 224x224 itself
# (model.py:166). Never serve a square.
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "360"))
# The manufacturer's URDF (i2rt yam_station_linear_4310_d405) gives the finger prismatic joint
# upper limit as 0.04695 m.
GOPEN = -float(os.environ.get("GOPEN_M", "0.04695"))
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0, physics_dt=1/240.0,
              rendering_dt=1.0/CONTROL_HZ)
world.scene.add(GroundPlane(prim_path="/World/ground", size=8.0))
# Cell geometry from Robocurve's dimensioned drawing: table 160 x 80 cm, arms
# mounted on a 6 cm extrusion along the BACK edge with the origin at the central
# camera mount.
world.scene.add(FixedCuboid(prim_path="/World/extrusion", name="extrusion",
                            position=np.array([-0.03, 0.0, TT-0.02]),
                            scale=np.array([0.06, 0.80, 0.04]),
                            color=np.array([0.22, 0.22, 0.24])))
# yam.yaml table: top surface at [0.6505, 0.0, 0.7517], corners [0.2995, 0.55] to
# [1.0015, -0.55], so 0.702 m deep by 1.10 m wide.
if RIG == "yamlab":
    _TBL_POS = np.array([0.6505, 0.0, TT-0.0075])
    _TBL_SCALE = np.array([1.0015-0.2995, 1.10, 0.015])
else:
    # Robocurve's top_cam photo shows wood under and behind the arm rail, so the 80 cm deep table
    # runs from 12 cm behind the extrusion to 68 cm in front of it, not from the extrusion forward.
    _TBL_POS = np.array([float(os.environ.get("TABLE_X", "0.28")), 0.0, TT-0.0075])
    _TBL_SCALE = np.array([0.80, 1.60, 0.015])
world.scene.add(FixedCuboid(prim_path="/World/table", name="table",
                            position=_TBL_POS,
                            scale=_TBL_SCALE,
                            color=np.array([float(os.environ.get("TABLE_R", "0.19")), float(os.environ.get("TABLE_G", "0.115")), float(os.environ.get("TABLE_B", "0.075"))])))
stage = omni.usd.get_context().get_stage()
# Robocurve's real top-camera frame shows a dim, softly lit
# dark-wood table. At sun 2500 / dome 700 the dark-brown table material rendered pale tan.
_SUN = float(os.environ.get("SUN", "1600.0")); _DOME = float(os.environ.get("DOME", "700.0"))
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(_SUN)
_dome = UsdLux.DomeLight.Define(stage, "/World/dome")
_dome.CreateIntensityAttr(_DOME)
print("[light] sun %.0f dome %.0f" % (_SUN, _DOME), flush=True)

# The cell had no backdrop, so beyond the table the top camera saw an empty white void:
# measured 55.8% of the frame was near-white with no information in it. That is the
# policy's most important input, half wasted. The photographs of the real rig show a
# black curtain enclosing it, so put one in. Dark, so it also stops the dome light
# washing the scene out.
if SPLAT_ENV:
    # Referenced onto a TYPELESS prim: add_reference_to_stage() defines a typed Xform whose local
    # type opinion overrides the referenced ParticleField3DGaussianSplat, after which nothing
    # recognises it as a splat. Transform = the table-plateau fit (scale 0.8072 m/unit, gravity
    # from a RANSAC plane) with a manual alignment on top:
    # z 1.327877 -> 1.221677 puts their table top at our 0.750, x 0.243351 -> -0.186649 puts the
    # arm mounts on their table's near edge. SPLAT_XFORM (16 floats, row-major USD layout) overrides.
    _sp = stage.DefinePrim("/World/splat")
    # WHICH splat is derived from the asset arm: an authored-arm run gets our capture, a PolaRiS
    # run gets theirs with its own fitted transform and dz. Explicit SPLAT_USD / SPLAT_DZ /
    # SPLAT_XFORM_FILE still win, so a run can name the splat it wants (e.g. PolaRiS assets in OUR
    # splat). The dz and the fitted transform follow THE SPLAT ACTUALLY LOADED, never the arm:
    # pairing our capture with PolaRiS's transform would misplace the scene.
    _ARM_SPLAT = {"kaedim":  HARNESS_ROOT + "/splat_env2_k080.usd",
                  "polaris": HARNESS_ROOT + "/polaris_splat/scene/polaris_env_final.usd"}
    _SPLAT_FIT = {
        _ARM_SPLAT["kaedim"]:  ("-0.005", ""),
        _ARM_SPLAT["polaris"]: ("-0.0105",
                                HARNESS_ROOT + "/polaris_splat/scene/fit/splat_xform.txt"),
    }
    _SPLAT_USD = os.environ.get("SPLAT_USD") or _ARM_SPLAT.get(ASSET_ARM, _ARM_SPLAT["kaedim"])
    _d_dz, _d_xf = _SPLAT_FIT.get(_SPLAT_USD, ("0.0", ""))
    print("[splat] arm=%s -> %s (%s)" % (ASSET_ARM, _SPLAT_USD,
          "SPLAT_USD set explicitly" if os.environ.get("SPLAT_USD") else "defaulted from ASSET_ARM"), flush=True)
    # SPLAT_USD, when set, overrides the per-arm default above.
    _sp.GetReferences().AddReference(_SPLAT_USD)
    _mv = [float(v) for v in os.environ.get("SPLAT_XFORM",
        "-0.079868 -0.803229 0.007717 0  0.752607 -0.077538 -0.281401 0  "
        "0.280749 -0.020648 0.756551 0  -0.186649 0.144665 1.221677 1").split()]
    # A second splat (PolaRiS-style capture) needs its own fitted transform, and 16 space-separated
    # numbers do not survive the job queue's env passing. SPLAT_XFORM_FILE points at a text file holding the same
    # 16 numbers (row-major USD layout) and wins over SPLAT_XFORM when set.
    _xf_file = os.environ.get("SPLAT_XFORM_FILE") or _d_xf
    if _xf_file:
        _mv = [float(v) for v in open(_xf_file).read().split()]
        print("[splat] SPLAT_XFORM_FILE %s (%s)" % (_xf_file,
              "explicit" if os.environ.get("SPLAT_XFORM_FILE") else "defaulted from ASSET_ARM"), flush=True)
    assert len(_mv) == 16, "SPLAT_XFORM needs 16 numbers"
    # SPLAT_DZ: raise/lower the whole splat by this many metres (a single number, so it survives the queue's
    # unquoted env passing; SPLAT_XFORM's 16 numbers do not). Defaults per splat from _SPLAT_FIT.
    _dz = float(os.environ.get("SPLAT_DZ") or _d_dz or "0.0")
    _mv[14] += _dz
    print("[splat] SPLAT_DZ %+.3f m -> translation z %.4f (%s)" % (_dz, _mv[14],
          "explicit" if os.environ.get("SPLAT_DZ") else "defaulted from ASSET_ARM"), flush=True)
    _sx = UsdGeom.Xformable(_sp); _sx.ClearXformOpOrder()
    _sx.AddTransformOp().Set(Gf.Matrix4d(*_mv))
    print("[splat] %s referenced as /World/splat, type %r, transform translation %s"
          % (_SPLAT_USD, _sp.GetTypeName(),
             [round(v, 3) for v in _mv[12:15]]), flush=True)

# With the splat the room IS the backdrop; the authored curtain would occlude it.
if os.environ.get("BACKDROP", "0" if SPLAT_ENV else "1") not in ("0", "", "no"):
    _cur = float(os.environ.get("CURTAIN", "0.045"))     # near-black curtain fabric
    for _nm, _pos, _scl in (
            ("back",  (2.10, 0.0, 1.30), (0.04, 3.20, 2.60)),   # far side, beyond table
            ("left",  (0.80, 1.55, 1.30), (2.80, 0.04, 2.60)),
            ("right", (0.80, -1.55, 1.30), (2.80, 0.04, 2.60)),
            ("behind", (-1.20, 0.0, 1.30), (0.04, 3.20, 2.60)),  # behind the arms
    ):
        world.scene.add(FixedCuboid(
            prim_path=f"/World/curtain_{_nm}", name=f"curtain_{_nm}",
            position=np.array(_pos), scale=np.array(_scl),
            color=np.array([_cur, _cur, _cur*1.05])))
    print("[cell] black-curtain backdrop added (set BACKDROP=0 to remove)", flush=True)

# Arm base placement. allenai/molmoact2#24 has the nominal values reconstructed from
# the official layout diagram in #2 and checked against the real videos:
#     left base x=-0.25 y=-0.03, right base x=+0.25 y=-0.03, base_rz 90 deg
# Their table frame is x+ right / y+ forward; ours is x+ forward / y+ left, so the
# change of basis is Rz(-90). That maps their (x,y) to our (y,x) with our_y = -their_x,
# and it makes the base rotation Rz(-90) @ Rz(90) = identity in our frame -- so placing
# the arms unrotated is correct.
# yam.yaml arms: left [0.2525, 0.305, 0.76], right [0.2525, -0.305, 0.76], both with
# identity quaternion. molmoact2 issue 24 gives -0.03 / +-0.25 instead, for their rig.
if RIG == "yamlab":
    _ARM_X_D, _ARM_Y_D = "0.2525", "0.305"
else:
    _ARM_X_D, _ARM_Y_D = "-0.03", "0.25"
ARM_Y = float(os.environ.get("ARM_Y", _ARM_Y_D))   # half-spacing
ARM_X = float(os.environ.get("ARM_X", _ARM_X_D))
# Their grasp offset is 0.1347 m along link6 local +Z, but that is for the
# ETHRoboticsClub yam.urdf, whose link6 frame is NOT ours: measured on our USD the
# gripper mesh spans local Z -0.011 .. +0.064, so 0.1347 lands 7.1 cm past our jaws.
# Use our own jaw centre for anything physical, and keep theirs only as a note.
for side, y in (("left", ARM_Y), ("right", -ARM_Y)):
    pp = f"/World/{side}_arm"
    # The robot is not part of the asset arm, so it always comes from the base root, never from
    # ASSET_ROOT. A PolaRiS-style root holds only the 20 task objects.
    add_reference_to_stage(usd_path=f"{A_ROBOT}/yam_robot/arm/yam.usd", prim_path=pp)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(pp)); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(ARM_X, y, 0.76))   # on the back extrusion
    UsdPhysics.FixedJoint.Define(stage, f"{pp}/base_fix").CreateBody1Rel().SetTargets([f"{pp}/arm/arm"])

# Robocurve's setup: two bottles of each type (Pellegrino, Sapporo, Heineken) with the
# blue bin behind them. Their setup photo shows the bin centre-back close to the arms
# and the six bottles in front of and beside it.
BLU = "sm_block_woodblue_d8b"
GRN = "sm_block_woodgreen_9d5"
RED = "sm_block_woodreda_e0a"
SAP = "sm_beerbottle_glasssapporo_d06"
HEI = "sm_bottle_glassheineken_8d6"
PEL = "sm_bottle_glasspellegrino_d5d"
# Task assets for clear-table, move-latte-cup and stack-bowls, from our asset team.
PIT   = "sm_pitcher_metalbrushed_046"        # the "steel latte art cup" (11.5 x 7.6 x 9.2 cm, sdf)
BRD   = "sm_cuttingboard_bamboofunctional_c14"  # 23.6 x 35.2 x 2.4 cm, convexHull (flat board, fine)
CAN   = "sm_cokecan_aluminumcylindrical_5ba"
SPR   = "sm_cleaningbottle_plasticlabel_3fc"
CBIN  = "sm_storagebin_plasticclear"         # clear/white bin, 44 x 26 x 17 cm; unused: Robocurve's setup photo shows the BLUE bin
BASE  = "sm_baseball_leatherstitched_713"
# Robocurve's grapes are RED fake plastic ones (their setup photo). The red asset,
# sm_grapes_thinplastichollow_a7d, is plastic class, 0.217 kg authored against the BOM's
# 3 oz (85 g); the mass is fixed in the asset, not here (the runner does not edit assets).
GRP   = "sm_grapes_thinplastichollow_a7d"
MUG   = "sm_mug_plasticred_02c"
CUBE  = "sm_rubikscube_plasticpuzzle_64e"
PLATE = "sm_plate_plasticspeckled_3b0"       # small purple plate
BOWL  = "sm_bowl_plasticribbed_07f"
# Re-laid inside the real table (arms at x=0, table x 0.02..0.82), bin back-centre
# with the bottles in front of and beside it, as in their setup photo.
# STATIC objects are posed once in USD before physics starts and never move (see the spawn loop):
# the bins and the cutting board. Everything else is a dynamic rigid body the policy can grasp.
# Optional 4th tuple element is a yaw in degrees for static objects (the blue bin reads BIN_YAW_DEG).
STATIC = set()
if TASKSET == "blocks":
    # 40-44 mm wooden blocks, 40-45 g, convexHull colliders, spread in front of the arms with no
    # overlaps. Robocurve's real setup (72 cm, pi0.5 rig) uses TWO blocks: red left of centre, blue
    # right, about 20 cm apart, ~37 cm from the arm bases, back-projected from their setup photo.
    # Their blocks were letter cubes; ours are plain red and blue at the same positions.
    OBJS = [("block_red",   RED, (0.393,  0.026)),     # 70 deg back-projection
            ("block_blue",  BLU, (0.414, -0.177))]
    LYING = set()
elif TASKSET == "latte":
    # Layout read off Robocurve's top-camera setup photo, mapped through the bottles
    # frame from the same camera (bin at x 0.40 sits at 21% of image height, the front bottles at
    # x 0.16 sit at 55%). Board: centred left-right, near edge just past the gripper tips, long
    # side (35 cm) running left-right, so local y along world y = yaw 0. Cup, coke can and spray
    # bottle stand in a row directly behind the board's far edge: cup left of centre, can on the
    # centre line, spray bottle right of centre.
    STATIC = {"board"}
    # Refit: each object's contact pixel in Robocurve's photo vs the same pixel in our render,
    # back-projected through the top camera at the 70 deg pitch read off the arms, rig-6 intrinsics
    # (fx 913, cy 381): board centre from its four corners, objects from their base contact pixels.
    # The asset board is ~10% larger than the real one, so its centre pixel misleads. Spray bottle
    # yawed 90 so its wide side faces the camera as in the photo. The board's far edge is at
    # x 0.291 + 0.118 = 0.409. Base-contact pixels put the can at 0.424 and the spray at 0.427, which
    # leaves their footprints (radius 3.2 / 5.4 cm) straddling the 24 mm board edge, so both tipped
    # flat at spawn. Pushed back so each clears the edge by ~2 cm; the real photo shows them just
    # behind the board, not on it.
    OBJS = [("board", BRD, (0.291, -0.014), 0.0),
            ("cup",   PIT, (0.465,  0.115)),
            ("can",   CAN, (0.460, -0.016)),
            ("spray", SPR, (0.485, -0.143), 90.0)]
    LYING = set()
elif TASKSET == "clear_table":
    # Layout read off Robocurve's top-camera setup photo, same mapping as latte.
    # The photo shows the BLUE bin (matching their instruction string). Bin back-centre, low open side to the
    # arms (yaw 90, as bottles). Items on both sides: Rubik's cube (far left), baseball (left,
    # beside the bin), red grapes (front left); red mug (far right), purple plate (front right).
    STATIC = {"bin"}
    # Refit against the real photo (same method as latte) at 70 deg, rig-2 intrinsics (fx 907, cy 376). The bin's
    # near bottom edge back-projects to x 0.247; its footprint is 0.267 deep at yaw 90, so the centre is ~0.37.
    OBJS = [("bin",      "sm_storagebin_plasticopen_f43", (0.370, -0.041), 90.0),
            ("cube",     CUBE,  (0.434,  0.305)),
            ("baseball", BASE,  (0.372,  0.182)),
            ("grapes",   GRP,   (0.278,  0.298)),
            # Mug set 2 cm back. At 0.389 its base sat mug-radius + plate-radius (11.5 cm) from the plate
            # centre, so with 1 cm jitter it landed on the plate rim and tipped in ~55% of trials.
            # The mug's base CENTRE in the photo is ~2 cm beyond its visible near edge anyway.
            ("mug",      MUG,   (0.410, -0.341)),
            ("plate",    PLATE, (0.274, -0.342))]
    LYING = set()
elif TASKSET == "bowls":
    # Robocurve's real setup (72 cm, pi0.5 rig): three cream bowls in a row ~31 cm from the arm
    # bases, the centre bowl slightly right of the midline (spacing 31 cm left-centre, 24 cm
    # centre-right). bowl_c is the CENTRE bowl the instruction names as the target.
    OBJS = [("bowl_l", BOWL, (0.347,  0.249)),     # 70 deg back-projection
            ("bowl_c", BOWL, (0.343, -0.062)),
            ("bowl_r", BOWL, (0.345, -0.303))]
    LYING = set()
else:
    # Layout from Robocurve's real top-camera frame: ALL SIX bottles stand
    # upright; left to right across the frame they are Heineken, Sapporo, Pellegrino, Sapporo,
    # Heineken, Pellegrino. World +y is the LEFT of the top-camera image (the left arm sits at
    # y=+0.25 and appears bottom-left). Bin slightly left of centre, open front toward the arms.
    # Refit at 70 deg, rig-1 intrinsics (fx 910, cy 384), from the bottle base contact pixels in their
    # setup photo. Bin centre = near edge at 0.30 + half depth 0.136 = 0.436.
    OBJS = [("bin",          "sm_storagebin_plasticopen_f43", (0.436,  0.004)),
            ("heineken_a",   HEI, (0.533,  0.309)),   # back left, beside the bin
            ("sapporo_a",    SAP, (0.300,  0.245)),   # left, mid
            ("pellegrino_a", PEL, (0.184,  0.081)),   # front, left of centre
            ("sapporo_b",    SAP, (0.193, -0.099)),   # front right
            ("heineken_b",   HEI, (0.342, -0.223)),   # right, mid
            ("pellegrino_b", PEL, (0.531, -0.263))]   # back right, beside the bin
    # BOTTLE_ORDER=real swaps the ASSET on the three 'a' slots only -- names and v3 positions are
    # untouched, so the layout refit still applies by name. Counts stay at 2 of each type.
    # Default for MOLMOACT2 ONLY: four real MolmoAct2 top frames (top 4, top 5, run12, run19) show
    # the three 'a' slots holding sapporo, pellegrino, heineken, and the stock order is wrong on all
    # three. pi0.5 ran a DIFFERENT real arrangement in which the pellegrinos were correctly
    # stationed, so pi0.5 keeps the stock order. _IS_MOLMO is inferred from the 30-step chunk (see
    # above). BOTTLE_ORDER=real or =stock overrides either way.
    if os.environ.get("BOTTLE_ORDER", "real" if _IS_MOLMO else "stock") == "real":
        _SWAP = {"heineken_a": SAP, "sapporo_a": PEL, "pellegrino_a": HEI}
        OBJS = [(o[0], _SWAP.get(o[0], o[1])) + tuple(o[2:]) for o in OBJS]
        print("[layout] BOTTLE_ORDER=real -> %s" % {o[0]: o[1] for o in OBJS if o[0] != "bin"}, flush=True)
    LYING = set()   # nothing starts on its side
    STATIC = {"bin"}
OBJS = [o if len(o) == 4 else (o[0], o[1], o[2], 0.0) for o in OBJS]
# LAYOUT=v3 (default): refit from Robocurve's per-trial policy-camera start frames (start_top.jpg,
# 640x480, the SAME camera the policy saw), not the 1280x720 setup photos the v2 layouts used.
# Method: object centroid pixel in the real frame vs the segmentation centroid in our render,
# both back-projected through the runner's own top-camera model; the delta
# is added to the v2 position. Deltas: clear_table bin -8.4 cm (closer to the arms), cube -7.5, plate -6.9/+5.8,
# baseball -4.3; bowls +2.5..+3.9 and 7 cm narrower on the left; blocks -6.6 and 6 cm closer together; latte spray
# +5.0, board -1.8/-2.7, cup -0.7/-2.5; bottles: rigid shift of the bin delta (-5.0, -3.7) so the matched layout keeps
# its shape. v2 was fitted through a generic camera model against 1280x720 photos. Measured against
# the real blocks frame: v2 is 5.2 cm out, v3 is 2.0 cm out. LAYOUT=v2 still available for the A/B.
_LAYOUT = os.environ.get("LAYOUT", "v3")
_V3 = {
    # pi0.5 latte ran in rig-6's cell at 72 cm; MolmoAct2 latte ran in rig-1's cell at 88 cm, and the two are not the
    # same physical setup -- back-projecting each real start frame through its OWN camera puts the board 4.6 cm further
    # out and 4.1 cm across, and the cup 5.3 cm further out. One layout cannot serve
    # both, so latte carries a per-cell entry; _V3 is selected by policy below.
    "latte":       {"board": (0.273, -0.041), "cup": (0.458, 0.090), "can": (0.476, -0.038), "spray": (0.479, -0.134)},   # spray: contact-point refit
    "latte_molmo": {"board": (0.319, -0.082), "cup": (0.511, 0.042), "can": (0.506, -0.057), "spray": (0.508, -0.204)},   # spray: contact-point refit
    "clear_table": {"bin": (0.286, -0.032), "cube": (0.359, 0.302), "baseball": (0.329, 0.158), "grapes": (0.243, 0.332),
                    "mug": (0.388, -0.316), "plate": (0.205, -0.284)},
    "bowls":       {"bowl_l": (0.296, 0.175), "bowl_c": (0.292, -0.013), "bowl_r": (0.283, -0.202)},
    # bowls: contact-point refit from their MolmoAct2 run07 start frame. Their three bowls are evenly
    # spaced 18.8 and 18.9 cm apart and sit ~6 cm closer to the arms than the centroid fit put them.
    # Same rig-6 cell for both policies, so one entry serves both.
    "blocks":      {"block_red": (0.327, 0.062), "block_blue": (0.349, -0.203)},
    # blocks has the same rig split latte has -- rig-1 at 72 cm for pi0.5, rig-2 at 88 cm for
    # MolmoAct2 -- so the MolmoAct2 cell has its own entry, refit from their MolmoAct2 run11 start
    # frame through the 88 cm camera. The method was validated by refitting the pi0.5 frame through
    # the 72 cm camera: it reproduces the pi0.5 entry to within 1.5 cm on both cubes.
    # Their MolmoAct2 cubes sit 23.8 cm apart against 26.5 cm for
    # pi0.5, i.e. ~4 cm closer together, and at equal depth rather than offset.
    "blocks_molmo": {"block_red": (0.340, 0.074), "block_blue": (0.340, -0.164)},
    "bottles":     {"bin": (0.386, -0.033), "heineken_a": (0.483, 0.272), "sapporo_a": (0.250, 0.208), "pellegrino_a": (0.134, 0.044),
                    "sapporo_b": (0.143, -0.136), "heineken_b": (0.292, -0.260), "pellegrino_b": (0.481, -0.300)},
}
# LAYOUT=v4: a SINGLE-BOTTLE correction, opt-in so the shipped cell is untouched.
# Measured from Robocurve's own run07 frame 0 through the rig-1 top camera: their Pellegrino base
# back-projects 3.5 cm further forward and 4.7 cm further from the left arm than v3 puts it.
# Why it matters: at v3 the Pellegrino sits 0.1025 m from the left wrist at rest, and contact with
# the gripper body starts at 0.090 m (half-width 0.051 + bottle radius 0.039), i.e. 1.3 cm of
# clearance on the ONE bottle in the scene that is also taller than the gripper. Real has 6.0 cm.
# Our arm flattens it inside the first 25 s of every replay and then reaches for a bottle lying down.
# Only sapporo_a is moved; the other five are unchanged because only this one could be located
# unambiguously in the real frame. The back-projection carries about 2.7 cm of absolute model error,
# which cancels in the difference but not in the absolute position -- so this is a test, not a fit.
_V4 = dict(_V3["bottles"])
_V4["sapporo_a"]    = (0.285, 0.161)   # the Pellegrino, back-projected from run07 frame 0
# The two bottles nearest the arms are spread far wider than real. Measured from body CENTROIDS,
# which need no base-ellipse guess: ours are 142 px apart in the top camera, real's are 90 px.
# At the same local scale that is 0.180 m against 0.114 m, so ours are 6.6 cm too far apart.
# Their midpoint is already right (341 px real vs 346 px ours), so hold the midpoint and close
# the gap. Forward position (x) left untouched: it cannot be read reliably from these frames.
_V4["pellegrino_a"] = (0.134,  0.011)  # small green, was +0.044
_V4["sapporo_b"]    = (0.143, -0.103)  # small brown, was -0.136
_V3_KEY = TASKSET + "_molmo" if (_IS_MOLMO and (TASKSET + "_molmo") in _V3) else TASKSET
if _LAYOUT == "v4" and TASKSET == "bottles":
    OBJS = [(o[0], o[1], _V4.get(o[0], o[2]), o[3]) for o in OBJS]
    print("[layout] v4 (bottles, sapporo_a refit from run07 frame 0): %s" % {o[0]: o[2] for o in OBJS}, flush=True)
elif _LAYOUT == "v3" and _V3_KEY in _V3:
    OBJS = [(o[0], o[1], _V3[_V3_KEY].get(o[0], o[2]), o[3]) for o in OBJS]
    print("[layout] v3 (%s, real start-frame refit): %s" % (_V3_KEY, {o[0]: o[2] for o in OBJS}), flush=True)
else:
    print("[layout] %s: %s" % (_LAYOUT, {o[0]: o[2] for o in OBJS}), flush=True)
# +-0.03 m of jitter pushed bottles into the bin wall; the nominal layout
# only has 16-24 mm of clearance on three of them.
JITTER = float(os.environ.get("JITTER", "0.01"))
OBJ_DX = float(os.environ.get("OBJ_DX", "0.0"))   # forward shift of every object; applied at spawn AND at every per-trial placement
BOTTLES = [n for n, _, _, _ in OBJS if n not in STATIC]   # the graspable objects (name kept from the bottles task)
def _set_translate(xf, pr, pos):
    """Add a translate op matching the precision of any xformOp:translate the prim already has.
    ClearXformOpOrder only clears the ORDER; the attribute stays, and AddTranslateOp with a
    different precision raises (e.g. the latte cup asset)."""
    a = pr.GetAttribute("xformOp:translate")
    if a and str(a.GetTypeName()) == "float3":
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*[float(v) for v in pos]))
    else:
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) for v in pos]))

def _set_orient(xf, pr, q_wxyz):
    a = pr.GetAttribute("xformOp:orient")
    q = [float(v) for v in q_wxyz]
    if a and str(a.GetTypeName()) == "quatd":
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], q[1], q[2], q[3]))
    else:
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(q[0], q[1], q[2], q[3]))

# PIPELINE_PHYS=1 re-binds the friction in each asset's <folder>.json onto its colliders. The authored
# assets ALREADY carry their pipeline physics: checked through the USD API (a byte grep cannot read
# binary USDC attribute names and gives a false negative), every asset has a
# /<asset>/Physics/PhysicsMaterial prim with the json's own friction, BOUND to every collider --
# pitcher 0.60/0.45, bowl 0.35/0.28, sapporo 0.55/0.40, can 0.45/0.35. This re-binder is therefore
# OFF by default; it exists only to A/B a deliberate override, and it must never be the default or
# it would overwrite the asset's own values.
_PIPE_PHYS = os.environ.get("PIPELINE_PHYS", "0") not in ("0", "", "no")
_pipe_bound = []

def _bind_pipeline_physics(prim_path, folder):
    """Bind the asset's own pipeline-authored friction to its colliders, from <folder>.json."""
    import json as _json, glob as _glob
    from pxr import UsdShade as _US
    _js = _glob.glob(os.path.join(A, folder, "*.json"))
    if not _js:
        return None
    try:
        _phys = _json.load(open(_js[0])).get("physics", {})
    except Exception:
        return None
    _mus, _mud = _phys.get("static_friction"), _phys.get("dynamic_friction")
    if _mus is None:
        return None
    _mp = "/World/Physics_Materials/pipe_" + folder
    if not stage.GetPrimAtPath(_mp).IsValid():
        _m = _US.Material.Define(stage, _mp)
        UsdPhysics.MaterialAPI.Apply(_m.GetPrim())
        _api = UsdPhysics.MaterialAPI(_m.GetPrim())
        _api.CreateStaticFrictionAttr().Set(float(_mus))
        _api.CreateDynamicFrictionAttr().Set(float(_mud if _mud is not None else _mus))
        _api.CreateRestitutionAttr().Set(float(_phys.get("restitution", 0.0) or 0.0))
    _m = _US.Material.Get(stage, _mp)
    _n = 0
    for _q in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if _q.HasAPI(UsdPhysics.CollisionAPI):
            _US.MaterialBindingAPI.Apply(_q)
            _US.MaterialBindingAPI(_q).Bind(_m, _US.Tokens.weakerThanDescendants, "physics")
            _n += 1
    return (_mus, _mud, _n) if _n else None

for name, folder, (x, y), yaw0 in OBJS:
    pp = f"/World/{name}"
    add_reference_to_stage(usd_path=f"{A}/{folder}/{folder}.usd", prim_path=pp)
    if _PIPE_PHYS:
        _r = _bind_pipeline_physics(pp, folder)
        if _r:
            _pipe_bound.append((name, _r[0], _r[1], _r[2]))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(pp)); xf.ClearXformOpOrder()
    # placements were authored for arms at x=-0.03; OBJ_DX carries them forward to
    # wherever the bases actually are (0.16 puts the blocks under the top camera's aim)
    _set_translate(xf, stage.GetPrimAtPath(pp), (x + OBJ_DX, y, TT + 0.002))
    if name == "grapes" and ASSET_ARM != "polaris":
        # GRAPES_MASS_KG: spawn-time mass override on the scene's copy; the asset file is not modified.
        # Default OFF: the headline arm runs the asset's pipeline-authored 0.217 kg (the BOM says 3 oz,
        # 0.085 kg). Any other value, e.g. 0.046 kg (an earlier green plastic asset's authored mass), is
        # an ABLATION and has to be asked for explicitly.
        _gm_env = os.environ.get("GRAPES_MASS_KG", "").strip()
        if _gm_env:
            _gm = float(_gm_env)
            for q in Usd.PrimRange(stage.GetPrimAtPath(pp)):
                if q.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.MassAPI.Apply(q).CreateMassAttr().Set(_gm)
                    print("[grapes] mass OVERRIDDEN to %.3f kg at spawn (asset authored 0.217 kg)" % _gm, flush=True)
        else:
            print("[grapes] pipeline-authored mass kept (set GRAPES_MASS_KG to override)", flush=True)
    if name == "baseball" and ASSET_ARM != "polaris":
        # The pipeline picks convexHull / convexDecomposition / sdf and
        # gave the baseball sdf; for a sphere a bounding-sphere collider is the right choice. Set on
        # the scene's referenced prims only; the asset file is not modified.
        _nsph = 0
        for q in Usd.PrimRange(stage.GetPrimAtPath(pp)):
            if q.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(q).GetApproximationAttr().Set(UsdPhysics.Tokens.boundingSphere)
                _nsph += 1
        print("[baseball] collision approximation -> boundingSphere on %d mesh prim(s)" % _nsph, flush=True)
    if name == "grapes" and os.environ.get("GRAPES_COLLIDER", "convexDecomposition") not in ("", "asset"):
        # The bunch is a thin, complex silhouette; its SDF collider jitters at rest
        # and when dropped. A convex decomposition is stable and close enough for a placed object.
        # Same in-scene mechanism as the baseball above; the asset file is not modified.
        _tok = getattr(UsdPhysics.Tokens, os.environ.get("GRAPES_COLLIDER", "convexDecomposition"))
        _ng = 0
        for q in Usd.PrimRange(stage.GetPrimAtPath(pp)):
            if q.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(q).GetApproximationAttr().Set(_tok); _ng += 1
        print("[grapes] collision approximation -> %s on %d mesh prim(s)" % (_tok, _ng), flush=True)
    # The latte cup is the only graspable object still on an SDF collider. It is a thin steel
    # shell (34 cm3 of metal); SDF contact on a thin wall gives the finger pads inconsistent normals, so the
    # gripper closes, "holds" for up to 25 steps and the cup slides out under 1 cm of lift
    # (best lift 0.01 m across 40+ trials). The bottles lift 13-15 cm on their CoACD convex decompositions and the
    # grapes were moved off SDF for the same reason. CONVEX_OBJS lists objects to switch at spawn (scene copy only).
    if name in [o for o in os.environ.get("CONVEX_OBJS", "").split(",") if o] and name != "grapes":
        _nc = 0
        for q in Usd.PrimRange(stage.GetPrimAtPath(pp)):
            if q.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(q).GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition); _nc += 1
        print("[%s] collision approximation -> convexDecomposition on %d mesh prim(s) (CONVEX_OBJS)" % (name, _nc), flush=True)
    if name in STATIC:
        # The bin is a STATIC collider, posed once here in USD before physics starts. As a dynamic
        # rigid body re-posed through the physics API every trial, the renderer only sometimes
        # picked that pose up: the same logged yaw produced different pictures run to run.
        # The real bin does not move during a trial, so nothing is lost. With the
        # RigidBodyAPI removed, physics, USD and the picture all read this one transform.
        # SPRAY_YAW_DEG makes the cleaning bottle's yaw testable. Their bottle presents its WIDE face
        # to the top camera (roughly 270 px across in the real frame); ours renders narrow (~140 px), which is a
        # 90-degree difference in how it is turned, not a position error.
        if name == "bin":
            BIN_YAW_DEG = float(os.environ.get("BIN_YAW_DEG", "90.0"))
        elif name == "spray":
            BIN_YAW_DEG = float(os.environ.get("SPRAY_YAW_DEG", str(yaw0)))
        else:
            BIN_YAW_DEG = float(yaw0)
        _hy = np.radians(BIN_YAW_DEG) / 2
        _set_orient(xf, stage.GetPrimAtPath(pp), (np.cos(_hy), 0.0, 0.0, np.sin(_hy)))
        for q in Usd.PrimRange(stage.GetPrimAtPath(pp)):
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                q.RemoveAPI(UsdPhysics.RigidBodyAPI)
                for _api in ("PhysxRigidBodyAPI", "PhysicsMassAPI"):
                    if _api in [str(t) for t in q.GetAppliedSchemas()]:
                        q.RemoveAppliedSchema(_api)
                print("[bin ] RigidBodyAPI removed from %s; bin is a static collider at yaw %.0f deg" % (q.GetPath(), BIN_YAW_DEG), flush=True)

def look_quat(eye, tgt):
    d = np.asarray(tgt, float) - np.asarray(eye, float); d /= np.linalg.norm(d)
    return rot_utils.euler_angles_to_quats(
        np.array([0.0, np.degrees(np.arcsin(-d[2])), np.degrees(np.arctan2(d[1], d[0]))]), degrees=True)

# RIG=molmoact2 calibration, from allenai/molmoact2 issue #2, measured on the actual cell, not fitted:
#   top camera   Intel D435, fx 601.4896 fy 601.1243 cx 320.0818 cy 253.5846 @ 640x480
#                -> fovy 43.53 deg, fovx 56.03 deg
#   extrinsic    T_midpoint_from_camera, midpoint = the point exactly between the two
#                arms on the table next to the extrusion. Camera sits 0.8896 m above
#                it, essentially straight up, looking 72.0 deg below horizontal.
#   wrist cams   Intel D405, fx 388.4646 fy 387.9068 -> fovy 63.49 deg
# Cameras render 640x480 at the true intrinsics and _serve() resizes to CAM_W x CAM_H (640x360 by
# default), a plain squash as in their embodiment (embodiment.py:627). CROP_H > 0 centre-crops
# first instead, the "center_crop" mode of the renderer in molmoact2#24; it is off by default.
#
# Two different rigs exist and they are not interchangeable:
#   RIG=molmoact2  the BimanualYAM rig described in the allenai/molmoact2 issue
#                  threads, which is what pi0.5 and MolmoAct2 were trained on (default).
#   RIG=yamlab     our asset's own workstation, calibrated per robot in yamlab's
#                  configs/robot/yam.yaml, whose header calls the contents
#                  "HARDWARE FACTS measured per robot + workstation".
# They have different cameras: yamlab's top lens is fy 391.72, MolmoAct2's D435 is fy 601.12.
#
# yamlab applies its camera entries as
#     TiledCameraCfg.OffsetCfg(pos=..., rot=quaternion_opengl, convention="opengl")
# relative to the prim's parent, with intrinsics calibrated at 640x480 and the render
# resolution decoupled (yam_bimanual_scene.py). So fovy is recomputed at our render
# height, fovx is unchanged.
RENDER_W, RENDER_H = 640, 480
TOP_RENDER_H = int(os.environ.get("TOP_RENDER_H", str(RENDER_H)))   # top camera only
# CAM_SUPERSAMPLE renders the cameras at N x the pixel count and lets _serve's
# existing resize down to CAM_W x CAM_H do the filtering, which is plain supersampling: sharper
# edges, no FOV change. It multiplies ONLY the render-product resolution. RENDER_W/RENDER_H are
# deliberately left alone because they feed _fov(); scaling them would silently change the field
# of view. Motivation: our wrist view reads sharpness 121 / edge 3.5%% against real's
# 469 / 12.1%% at the same arm pose, and the bottle labels are illegible where real's are not.
_SS = float(os.environ.get("CAM_SUPERSAMPLE", "1"))
def _ssres(w, h):
    return (int(round(w*_SS)), int(round(h*_SS)))
# No crop by default: Robocurve squash 480 rows to 360, they do not crop (embodiment.py:627).
CROP_H = int(os.environ.get("CROP_H", "0"))

def _fov(fx, fy, w, h):
    return (2*np.degrees(np.arctan((h/2)/fy)), 2*np.degrees(np.arctan((w/2)/fx)))

if RIG == "yamlab":
    # yam.yaml cameras: intrinsic_resolution [640, 480]
    FOVY_TOP, FOVX_TOP = _fov(392.195617675781, 391.722351074219, RENDER_W, RENDER_H)
    FOVY_WRIST, FOVX_WRIST = _fov(390.666, 390.162, RENDER_W, RENDER_H)
    # top camera, world frame, quaternion_opengl wxyz
    _TOP_POS = np.array([0.08600512146949768, -0.008999999612569809, 1.7043205499649048])
    _TOP_QUAT = np.array([0.68301, 0.18301, -0.18301, -0.68301])
    # wrist cameras, offset from link_6, quaternion_opengl wxyz
    _WRIST_POS = {"left": np.array([-0.0004, 0.069638, 0.073063]),
                  "right": np.array([0.0, 0.069638, 0.072])}
    _WRIST_QUAT = np.array([-0.003227, 0.002817, 0.975619, 0.219430])
else:
    # molmoact2 issue 2: D435 top, D405 wrists, 640x480 intrinsics
    FOVY_TOP, FOVX_TOP = _fov(601.4896240234375, 601.124267578125, RENDER_W, RENDER_H)
    FOVY_WRIST, FOVX_WRIST = _fov(388.464599609375, 387.9068298339844, RENDER_W, RENDER_H)
    # Diagnostic: MolmoAct2's authors evaluate this robot with a NATIVE 640x360 top camera
    # at hfov 69.4 deg (allenai/molmoact2 sim_eval/robots/bimanual_yam.py), not the D435 4:3 mode
    # squashed. TOP_HFOV_DEG + TOP_RENDER_H=360 reproduce that camera; both default off.
    if os.environ.get("TOP_HFOV_DEG"):
        FOVX_TOP = float(os.environ["TOP_HFOV_DEG"])
        FOVY_TOP = float(np.degrees(2*np.arctan(np.tan(np.radians(FOVX_TOP)/2) * TOP_RENDER_H / RENDER_W)))
    # Robocurve mount the top camera ~88 cm above the table for MolmoAct2 and 72 cm for
    # pi0.5, in the same reference frame as the issue-2 extrinsic, i.e. above the
    # midpoint on the table.
    _TOP_H = float(os.environ.get("TOP_H", "0.72"))
    # Guard: MolmoAct2 is evaluated at 88 cm and pi0.5 at 72 cm. The 30-step chunk is the tell for
    # MolmoAct2 behind the socket. Fail loudly rather than ship a number taken 16 cm too low.
    if _CHUNK == 30 and abs(_TOP_H - 0.72) < 1e-6 and os.environ.get("ALLOW_TOPH_MISMATCH", "") not in ("1", "yes"):
        raise SystemExit("[cam ] REFUSING TO RUN: 30-step chunks means MolmoAct2, which is evaluated at "
                         "TOP_H=0.88, but TOP_H is 0.72. Set TOP_H, or ALLOW_TOPH_MISMATCH=1 for a deliberate test.")
    _TOP_POS = np.array([float(os.environ.get("TOP_X", "-0.0075")), -0.0139, TT + _TOP_H])   # TOP_X: forward offset from the mount midpoint
    _TOP_QUAT = None      # aim by look-at instead, see below
    # The D405 wrist mount for this rig is not published, so the
    # only measured mount pose anywhere is yam.yaml's. Their wrist intrinsics agree to
    # within 0.6% (yam.yaml fx 390.67 vs molmoact2 fx 388.46, both D405 at 640x480),
    # i.e. the same camera on the same link_6, so carry the bracket across rather than
    # fall back on a hand-picked sweep. Flag it as a cross-rig substitution, not a
    # source for this rig.
    _WRIST_POS = {"left": np.array([-0.0004, 0.069638, 0.073063]),
                  "right": np.array([0.0, 0.069638, 0.072])}
    _WRIST_QUAT = np.array([-0.003227, 0.002817, 0.975619, 0.219430])

_bp = _TOP_POS
if _TOP_QUAT is None:
    # Pitch below horizontal, default 70.0. 72.0 is the issue-#2 rotation (0.3098, -0.9508) measured
    # at 88 cm. At 72 cm, 70 is read off the ARMS, whose world position (CAD, issue #24) and rest pose
    # (dataset frame-0 states, all joints ~0) are known: the right gripper's camera housing sits at row
    # 285-290 (of 480) in every real 72 cm photo (rigs 2, 6, bowls, blocks) vs 265 in our 72 deg render
    # and 375 at 61 deg -> 69-70 deg, +-2. A board-corner fit that said ~61.5 was the outlier.
    # TOP_TILT_DEG overrides.
    _tilt = np.radians(float(os.environ.get("TOP_TILT_DEG", "70.0")))
    _bdir = np.array([np.cos(_tilt), 0.003, -np.sin(_tilt)]); _bdir /= np.linalg.norm(_bdir)
else:
    # OpenGL/USD camera convention: the camera views along its own local -Z
    _bdir = -np.asarray(rot_utils.quats_to_rot_matrices(
        _TOP_QUAT.reshape(1, 4)))[0][:, 2]
_bl = _bp + _bdir*((TT - _bp[2])/_bdir[2]) if abs(_bdir[2]) > 1e-6 else _bp + _bdir
base_cam = Camera(prim_path="/World/base_view", position=_bp, orientation=look_quat(_bp, _bl),
                  frequency=int(CONTROL_HZ), resolution=_ssres(RENDER_W, TOP_RENDER_H))
# recording view: pulled back behind the extrusion so the arms stay in shot
if RIG == "yamlab":
    # operator's eye: in front of the table looking back at both arms. Arms at
    # x 0.2525, y +-0.305; table x 0.2995..1.0015, y -0.55..0.55. The old pose was
    # aimed at x 0.42, which framed one arm and half the table.
    _hp, _ht = np.array([1.75, -0.95, 1.55]), np.array([0.50, 0.0, 0.82])
elif SPLAT_ENV:
    # Picked from two render sweeps on the cut splat: high and just behind the
    # arms, looking steeply down the table. The capture is a close orbit of the cell with nothing past
    # 2.5 m, so any view toward the far room renders unobserved volume as white smear; this one sees
    # mostly table, floor and arms, which the capture covers, and keeps every bottle, the bin and both
    # arms in frame. Side and front poses smear; a high pose over the arms sits inside the cloud.
    _hp, _ht = np.array([-0.6, 0.0, 1.6]), np.array([0.45, 0.0, 0.78])   # "behind-mid" pose
else:
    _hp, _ht = np.array([1.45, -0.95, 1.50]), np.array([0.28, 0.0, 0.82])
# SPECT_POS / SPECT_TGT (x,y,z metres) override the recording camera. Recording only: the policy never
# sees this view and nothing scored depends on it.
if os.environ.get("SPECT_POS"):
    _hp = np.array([float(v) for v in os.environ["SPECT_POS"].split(",")])
    _ht = np.array([float(v) for v in os.environ.get("SPECT_TGT", "0.42,0,0.85").split(",")])
    print("[cam ] spectator override: pos %s -> target %s" % (np.round(_hp, 2), np.round(_ht, 2)), flush=True)
behind_cam = Camera(prim_path="/World/behind", position=_hp, orientation=look_quat(_hp, _ht),
                    frequency=int(CONTROL_HZ), resolution=_ssres(RENDER_W, RENDER_H))
# close-up of the bin from between the arms, so the open front is unambiguous in a still
_bsp, _bst = np.array([0.08, -0.02, 0.98]), np.array([0.40, -0.02, 0.84])
binshot_cam = Camera(prim_path="/World/binshot", position=_bsp, orientation=look_quat(_bsp, _bst),
                     frequency=int(CONTROL_HZ), resolution=_ssres(RENDER_W, RENDER_H))
wrist_cams = {}   # built after world.reset(), from link_6's actual pose

# physics: the settings that fixed penetration; stabilization deliberately off
for p in stage.Traverse():
    if p.GetTypeName() == "PhysicsScene":
        sa = PhysxSchema.PhysxSceneAPI.Apply(p)
        sa.CreateTimeStepsPerSecondAttr().Set(240)
        sa.CreateEnableCCDAttr().Set(True)
        sa.CreateEnableStabilizationAttr().Set(False)
nc = 0
for root in ["/World/left_arm", "/World/right_arm"] + [f"/World/{n}" for n, *_ in OBJS]:
    pr = stage.GetPrimAtPath(root)
    if not (pr and pr.IsValid()): continue
    for q in Usd.PrimRange(pr):
        if not UsdPhysics.CollisionAPI(q): continue
        ca = PhysxSchema.PhysxCollisionAPI.Apply(q)
        # CONTACT_OFFSET_M makes this tunable. The bowl's wall is ~2 mm and the cup's is thinner;
        # a 6 mm contact shell is three times the wall, so the pads can register contact through the wall
        # without ever developing a squeeze. Measured symptom: 682 steps of two-pad contact, 3 cm of lift.
        ca.CreateContactOffsetAttr().Set(float(os.environ.get("CONTACT_OFFSET_M", "0.006")))
        ca.CreateRestOffsetAttr().Set(0.0)
        nc += 1
# FILTER_FINGERS (default on). The two finger links of each gripper collide with each other and
# stop the jaws at joint -0.022, i.e. 47% closed, in an EMPTY scene with no object and 100 N of force
# (20x force buys 1 mm; the runtime joint limit is 0.0 and the joint tracks its target perfectly down
# to -0.030 before hitting the wall). Robocurve's own recorded actions command the gripper to 0.034
# normalised, i.e. 3.4% -- essentially closed -- so the real robot routinely closes far past where
# an unfiltered pair can. Filtering the finger pair releases it to exactly 0.0.
# FILTER_FINGERS=0 restores the unfiltered behaviour.
_FFILT = os.environ.get("FILTER_FINGERS", "1") not in ("0", "", "no")
if _FFILT:
    _nfilt = 0
    for _side in ("left", "right"):
        _fl = {}
        _r = stage.GetPrimAtPath("/World/%s_arm" % _side)
        if not (_r and _r.IsValid()):
            continue
        for _q in Usd.PrimRange(_r):
            if _q.GetName() in ("left_finger", "right_finger") and _q.HasAPI(UsdPhysics.RigidBodyAPI):
                _fl[_q.GetName()] = _q
        if len(_fl) == 2:
            _a, _b = _fl["left_finger"], _fl["right_finger"]
            UsdPhysics.FilteredPairsAPI.Apply(_a).CreateFilteredPairsRel().AddTarget(_b.GetPath())
            UsdPhysics.FilteredPairsAPI.Apply(_b).CreateFilteredPairsRel().AddTarget(_a.GetPath())
            _nfilt += 1
    print("[phys] FILTER_FINGERS on: finger<->finger collision filtered on %d gripper(s); "
          "jaws can now reach joint 0.0 instead of stalling at -0.022" % _nfilt, flush=True)

for root in [f"/World/{n}" for n, *_ in OBJS] + ["/World/left_arm/arm/arm", "/World/right_arm/arm/arm"]:
    pr = stage.GetPrimAtPath(root)
    if pr and pr.IsValid():
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(pr)
        rb.CreateSolverPositionIterationCountAttr().Set(64)
        rb.CreateSolverVelocityIterationCountAttr().Set(16)
        if not root.endswith("/arm"):
            # PhysX default is unbounded: a bottle squeezed between finger colliders was ejected
            # at tens of m/s (in one trial: 5.1 m displacement, one bottle at z 3.8 m at the next
            # reset). 3 m/s is faster than anything a real bottle does on this table.
            rb.CreateMaxDepenetrationVelocityAttr().Set(3.0)
# Reference physics + drive setup, all from yamlab/robot/yam/yam.py, the Isaac config
# for this exact robot:
#   :52  disable_gravity=True on the arm rigid bodies, re-asserted :114 for high-PD
#        ("used in sim"). With gravity on, the arm carries its own weight against gains
#        meant for a weightless one. Measured: disabling it drops the standing tracking
#        error on joints 1-4 from 0.008-0.011 rad to ~0.0001.
#   :53  max_depenetration_velocity 100.0
#   :57-58  solver_position_iteration_count 16, solver_velocity_iteration_count 1
#   yam_gripper effort_limit_sim 100.0 -> the finger drives ship maxForce=0.0, and
#        set_gains() writes only stiffness/damping, so without an explicit maxForce the
#        fingers have NO force authority: commanded shut, the left finger does not move.
_NOGRAV = os.environ.get("ARM_GRAVITY", "0") in ("0", "", "no")
_ng = _nsolv = 0
for _side in ("left", "right"):
    _root = stage.GetPrimAtPath("/World/%s_arm" % _side)
    if not (_root and _root.IsValid()):
        continue
    for _q in Usd.PrimRange(_root):
        if UsdPhysics.RigidBodyAPI(_q):
            _rb = PhysxSchema.PhysxRigidBodyAPI.Apply(_q)
            if _NOGRAV:
                _rb.CreateDisableGravityAttr().Set(True); _ng += 1
            _rb.CreateSolverPositionIterationCountAttr().Set(64)
            _rb.CreateSolverVelocityIterationCountAttr().Set(16)
            _nsolv += 1
_FMAX = float(os.environ.get("FINGER_MAXFORCE", "100.0"))
_nf = 0
for _side in ("left", "right"):
    _root = stage.GetPrimAtPath("/World/%s_arm" % _side)
    if not (_root and _root.IsValid()):
        continue
    for _q in Usd.PrimRange(_root):
        if _q.GetName() in ("left_finger", "right_finger"):
            _d = UsdPhysics.DriveAPI.Get(_q, "linear")
            if _d:
                _d.CreateMaxForceAttr().Set(_FMAX); _nf += 1
# JOINT_FRICTION. The vendor's own motor config (i2rt/robots/config/yam_v1.yml) gives
# coulomb_friction [0.3, 0.3, 0.3, 0.06, 0.06, 0.06] Nm on every arm joint; ours are frictionless
# by default. Takes a comma list (per joint, in path order) or one value for all. Unset = off.
_JFRIC = os.environ.get("JOINT_FRICTION", "").strip()
if _JFRIC:
    _jv = [float(v) for v in _JFRIC.split(",")]
    _njf = 0; _jshow = []
    for _side in ("left", "right"):
        _r = stage.GetPrimAtPath("/World/%s_arm" % _side)
        if not (_r and _r.IsValid()):
            continue
        _js = [q for q in Usd.PrimRange(_r) if q.IsA(UsdPhysics.RevoluteJoint)]
        _js.sort(key=lambda q: q.GetPath().pathString)
        for _i, _q in enumerate(_js):
            _v = _jv[_i] if _i < len(_jv) else _jv[-1]
            PhysxSchema.PhysxJointAPI.Apply(_q).CreateJointFrictionAttr().Set(_v)
            _njf += 1
            if _side == "left":
                _jshow.append((_q.GetName(), _v))
    # print the NAMES, so the per-joint assignment is checkable from the log and never assumed
    print("[phys] JOINT_FRICTION on %d joint(s); left arm in path order: %s" % (_njf, _jshow), flush=True)
else:
    print("[phys] JOINT_FRICTION not set: arm joints are frictionless (real is 0.3/0.06 Nm, yam_v1.yml)", flush=True)

if _PIPE_PHYS:
    print("[phys] pipeline friction BOUND on %d/%d objects: %s"
          % (len(_pipe_bound), len(OBJS),
             {n: "mu_s %.2f mu_d %.2f x%d" % (a, b, c) for n, a, b, c in _pipe_bound}), flush=True)
else:
    if ASSET_ARM == "polaris":
        print("[phys] object friction: NO physics material bound on any task object (PolaRiS-style arm) -> PhysX default 0.5", flush=True)
    else:
        print("[phys] object friction: the assets' own bound pipeline materials (verified present), nothing overridden", flush=True)
print("[phys] 240 Hz, CCD on, stabilization OFF, %d colliders contactOffset %.0f mm" % (nc, 1000*float(os.environ.get("CONTACT_OFFSET_M", "0.006"))),
      flush=True)
print("[phys] arm gravity %s (%d bodies), solver 64/16 on %d bodies, finger maxForce "
      "%.1f on %d drives" % ("DISABLED" if _NOGRAV else "on", _ng, _nsolv, _FMAX, _nf),
      flush=True)
print("[ctrl] %.1f Hz, absolute joint targets clipped to %s, no per-step delta limit on the arm; gripper TARGET rate-limited to a "
      "%.2f s full stroke (GRIPPER_STROKE_S, fitted to the 1/30-per-step cap in their recorded actions; 0 disables), serving %dx%d"
      % (CONTROL_HZ,
         "each joint's own limits (Robocurve, 8 Sep 2026: 'each predicted joint target is clipped to the joint range')"
         if os.environ.get("CLIP_JOINTS", "1") not in ("0", "", "no") else "[-pi, pi] only (CLIP_JOINTS=0, A/B)",
         _GSTROKE, CAM_W, CAM_H), flush=True)
print("[task] %s | instruction %r" % (TASKSET, TASK), flush=True)
print("[cell] RIG=%s | table %.0fx%.0f cm at x=%.4f | arm bases x=%.4f, half-spacing "
      "%.4f m" % (RIG, _TBL_SCALE[0]*100, _TBL_SCALE[1]*100, _TBL_POS[0], ARM_X, ARM_Y),
      flush=True)
print("[cam ] top D435 at %s -> %s | %.2f m above table, %.1f deg down, fovy %.2f"
      % (np.round(_bp, 4), np.round(_bl, 4), _bp[2]-TT,
         np.degrees(np.arctan2(-_bdir[2], np.hypot(_bdir[0], _bdir[1]))), FOVY_TOP), flush=True)
print("[cam ] top camera %.4f m above the table (TOP_H)"
      % (_bp[2] - TT), flush=True)
print("[cam ] rendering %dx%d, serving the policy %dx%d (fovy top %.2f, wrist %.2f)"
      % (RENDER_W, RENDER_H, CAM_W, CAM_H, FOVY_TOP, FOVY_WRIST), flush=True)
print("[cam ] CAM_SUPERSAMPLE=%.2f -> render products %s (FOV unchanged)"
      % (_SS, (_ssres(RENDER_W, RENDER_H),)), flush=True)

for n, *_ in OBJS:
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(f"/World/{n}")).CreateThresholdAttr().Set(0.0)

def _kinematic(name, on):
    """Weld helper: a kinematic body is moved by set_world_poses and is NOT integrated by the
    solver, so re-imposing its pose every control step cannot accumulate velocity and fling it
    on release, which is what the dynamic version did (bottles landed 1.0 m off the table)."""
    _p = stage.GetPrimAtPath("/World/%s" % name)
    _rb = UsdPhysics.RigidBodyAPI(_p)
    if _rb:
        _a = _rb.GetKinematicEnabledAttr()
        if not _a: _a = _rb.CreateKinematicEnabledAttr()
        _a.Set(bool(on))

PEN = {}
_WELD_N = int(os.environ.get("WELD_ON_CONTACT", "0"))
_WELDED = {}
_WELD_REL = {}
_WELD_OPEN = -float(os.environ.get("WELD_OPEN_M", "0.020"))
_WELD_HYST = int(os.environ.get("WELD_HYST", "3"))
_WOPENRUN = {}      # per-object run of consecutive steps with the gripper commanded open
_WSTAT = {}         # per-trial weld bookkeeping, written into results.jsonl
TOUCH = set()          # (object, side, finger) contacts seen since the last clear
def on_contact(headers, data):
    for h in headers:
        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        both = a0 + a1
        who = next((n for n, *_ in OBJS if n in both), None)
        if who is None: continue
        for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
            sep = float(data[i].separation)
            d = PEN.setdefault(who, [0, 0.0])
            d[0] += 1
            if sep < d[1]: d[1] = sep
        # yam.yaml's grasp test is a contact test, not a height-and-proximity guess:
        # a finger pad must be in contact with the target. Record which finger, so a
        # hold can be defined as BOTH fingers of one arm touching the same object.
        for _side in ("left", "right"):
            for _f in ("left_finger", "right_finger"):
                if ("/World/%s_arm/arm/%s" % (_side, _f)) in (a0, a1):
                    TOUCH.add((who, _side, _f))
sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

# FixedCuboid(color=...) does not take: the table authored at dark brown
# (0.19, 0.115, 0.075) and the extrusion at dark grey (0.22) both render pale, while
# the gripper's own asset material renders correctly black. So bind a real
# UsdPreviewSurface to each authored box instead of trusting the color argument.
def _paint(prim_path, rgb, rough=0.65):
    from pxr import UsdShade, Sdf
    mat_path = "/World/Looks/mat_" + prim_path.strip("/").replace("/", "_")
    mat = UsdShade.Material.Define(stage, mat_path)
    sh = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    _tex = os.environ.get("TABLE_TEX", "") if prim_path == "/World/table" else ""
    if _tex and os.path.exists(_tex):
        # TABLE_TEX: real wood grain instead of a flat colour. The image is an orthographic texture of
        # Robocurve's table built from their top-camera video frame (colour
        # matched to the frame mean 96/105/103). UV: the cuboid's st runs 0..1 over its local x/y; TABLE_TEX_SCALE
        # tiles it. Falls back to the flat colour if the file is missing.
        _st = UsdShade.Shader.Define(stage, mat_path + "/st"); _st.CreateIdAttr("UsdPrimvarReader_float2")
        _st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        _st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        _tx = UsdShade.Shader.Define(stage, mat_path + "/tex"); _tx.CreateIdAttr("UsdUVTexture")
        _tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(_tex)
        _tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat"); _tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        _sc = float(os.environ.get("TABLE_TEX_SCALE", "1.0"))
        _tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[float(os.environ.get("TABLE_TEX_GAIN", "1.0"))]*3, 1.0))
        _tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(_st.ConnectableAPI(), "result")
        _tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(_tx.ConnectableAPI(), "rgb")
        print("[cell] table textured from %s (TABLE_TEX)" % _tex, flush=True)
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    pr = stage.GetPrimAtPath(prim_path)
    if pr and pr.IsValid():
        UsdShade.MaterialBindingAPI(pr).Bind(mat)
        return True
    return False

_TBL_RGB = (float(os.environ.get("TABLE_R", "0.19")),
            float(os.environ.get("TABLE_G", "0.115")),
            float(os.environ.get("TABLE_B", "0.075")))
_painted = []
if _paint("/World/table", _TBL_RGB, 0.92):   # matte: the real wood shows no reflections
    _painted.append("table")
if _paint("/World/extrusion", (0.10, 0.10, 0.11), 0.4):
    _painted.append("extrusion")
for _nm in ("back", "left", "right", "behind"):
    if _paint(f"/World/curtain_{_nm}", (0.045, 0.045, 0.047), 0.9):
        _painted.append("curtain_" + _nm)
print("[cell] painted %s (table rgb %s)" % (_painted, np.round(_TBL_RGB, 3)), flush=True)

# Table surface physics material: mu_s 0.55, mu_d 0.45, restitution 0.02 bound to /World/table.
# On by default under SPLAT_ENV (see _TABLE_MU_S below); without the splat it is off unless
# TABLE_MU_S is set.
# The gripper PADS have no physics material bound by default. An unbound collider gets PhysX's
# default material (friction 0.5) and the two surfaces are COMBINED (default: average), so a grasp
# averages the object's authored friction with that engine default. FINGER_MU_S/_D bind a real pad
# material. Default is EMPTY, so this changes nothing until it is asked for -- it is a knob to test
# with, not a silent physics change.
_FINGER_MU_S = os.environ.get("FINGER_MU_S", "").strip()
if _FINGER_MU_S:
    from pxr import UsdShade as _US
    _fmat = _US.Material.Define(stage, "/World/Physics_Materials/finger_pad")
    UsdPhysics.MaterialAPI.Apply(_fmat.GetPrim())
    _fapi = UsdPhysics.MaterialAPI(_fmat.GetPrim())
    _fapi.CreateStaticFrictionAttr().Set(float(_FINGER_MU_S))
    _fapi.CreateDynamicFrictionAttr().Set(float(os.environ.get("FINGER_MU_D", _FINGER_MU_S)))
    _fapi.CreateRestitutionAttr().Set(0.0)
    # FRICTION_COMBINE. PhysX combines the two surfaces' friction, default "average",
    # so a pad at mu 1.45 against the bottle's authored 0.55 realises only 1.0 and anything above
    # that is unreachable. PhysX resolves a disagreement by taking the HIGHER combine mode
    # (average < min < multiply < max), so setting it on the pad alone is enough. Unset = average.
    _fcm = os.environ.get("FRICTION_COMBINE", "").strip()
    if _fcm:
        _pmat = PhysxSchema.PhysxMaterialAPI.Apply(_fmat.GetPrim())
        _pmat.CreateFrictionCombineModeAttr().Set(_fcm)
    _nfp = 0
    for _side in ("left", "right"):
        _root = stage.GetPrimAtPath("/World/%s_arm" % _side)
        if not (_root and _root.IsValid()):
            continue
        for _q in Usd.PrimRange(_root):
            # Match on the finger SUBTREE, not on prims NAMED left_finger/right_finger: in yam.usd
            # /arm/left_finger is a bare Xform and the actual colliders are 8 meshes per side at
            # /arm/left_finger/collisions/left_finger_col_tip_left_N/..., so a name match binds 0 prims.
            # The subtree match picks up all 16 pad colliders.
            if not UsdPhysics.CollisionAPI(_q):
                continue
            _pp = _q.GetPath().pathString
            if "/left_finger/" in _pp or "/right_finger/" in _pp:
                _US.MaterialBindingAPI.Apply(_q)
                _US.MaterialBindingAPI(_q).Bind(_fmat, _US.Tokens.weakerThanDescendants, "physics")
                _nfp += 1
    print("[phys] finger pads mu_s=%s mu_d=%s bound to %d prim(s); frictionCombineMode=%s -> realised mu_s %s"
          % (_FINGER_MU_S, os.environ.get("FINGER_MU_D", _FINGER_MU_S), _nfp,
             _fcm or "average (default)",
             ("%.3f" % max(float(_FINGER_MU_S), 0.55)) if _fcm == "max"
             else ("%.3f" % ((float(_FINGER_MU_S) + 0.55) / 2.0))), flush=True)
else:
    print("[phys] finger pads: NO physics material bound (PhysX default 0.5, averaged with the object's own)", flush=True)

# OBJ_MU_S/_D override the friction of the DYNAMIC TASK OBJECTS only.
# Why this exists: every asset's json tags its friction `friction_convention: pair_value_on_steel_dry`, i.e. the value
# is for that material against STEEL. The gripper pad is not steel, and PhysX averages the object's value with the
# pad's (unbound, so PhysX's own 0.5). So a bottle grasp runs a glass-on-steel number averaged with an engine default,
# and neither describes glass against a rubber pad. This knob tests a corrected counter-surface value.
# It binds ONLY to the task objects, never to the pads or the table, so a bottles job cannot disturb any other cell.
# Default EMPTY = the assets' own pipeline values, unchanged.
_OBJ_MU_S = os.environ.get("OBJ_MU_S", "").strip()
if _OBJ_MU_S:
    from pxr import UsdShade as _US2
    _omat = _US2.Material.Define(stage, "/World/Physics_Materials/task_object")
    UsdPhysics.MaterialAPI.Apply(_omat.GetPrim())
    _oapi = UsdPhysics.MaterialAPI(_omat.GetPrim())
    _oapi.CreateStaticFrictionAttr().Set(float(_OBJ_MU_S))
    _oapi.CreateDynamicFrictionAttr().Set(float(os.environ.get("OBJ_MU_D", _OBJ_MU_S)))
    _nob, _obnames = 0, []
    for _o in OBJS:
        _nm = _o[0]
        if _nm in STATIC:
            continue
        _r = stage.GetPrimAtPath("/World/%s" % _nm)
        if not (_r and _r.IsValid()):
            continue
        _hit = 0
        for _q in Usd.PrimRange(_r):
            if UsdPhysics.CollisionAPI(_q):
                _US2.MaterialBindingAPI.Apply(_q)
                _US2.MaterialBindingAPI(_q).Bind(_omat, _US2.Tokens.strongerThanDescendants, "physics")
                _hit += 1
        if _hit:
            _nob += _hit; _obnames.append(_nm)
    print("[phys] task-object friction OVERRIDDEN mu_s=%s mu_d=%s on %d collider(s): %s"
          % (_OBJ_MU_S, os.environ.get("OBJ_MU_D", _OBJ_MU_S), _nob, _obnames), flush=True)

_TABLE_MU_S = os.environ.get("TABLE_MU_S", "0.55" if SPLAT_ENV else "")   # splat default: 0.55/0.45/0.02
if _TABLE_MU_S:
    from pxr import UsdShade
    _mat = UsdShade.Material.Define(stage, "/World/Physics_Materials/table_surface")
    UsdPhysics.MaterialAPI.Apply(_mat.GetPrim())
    _mapi = UsdPhysics.MaterialAPI(_mat.GetPrim())
    _mapi.CreateStaticFrictionAttr().Set(float(_TABLE_MU_S))
    _mapi.CreateDynamicFrictionAttr().Set(float(os.environ.get("TABLE_MU_D", "0.45")))
    _mapi.CreateRestitutionAttr().Set(float(os.environ.get("TABLE_RESTITUTION", "0.02")))
    _tbl_prim = stage.GetPrimAtPath("/World/table")
    if _tbl_prim and _tbl_prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(_tbl_prim)
        UsdShade.MaterialBindingAPI(_tbl_prim).Bind(_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        print("[phys] table surface mu_s=%.2f mu_d=%.2f restitution=%.2f" % (
            _mapi.GetStaticFrictionAttr().Get(), _mapi.GetDynamicFrictionAttr().Get(),
            _mapi.GetRestitutionAttr().Get()), flush=True)

if SPLAT_ENV:
    # Which table the cameras see. Their table is cut out of the splat (a second bin and
    # bottles would otherwise be in the policy's view) and the hole behind it renders as smear:
    #   authored  ours stays visible, recoloured to the splat's own steel-grey table
    #             (opacity-weighted mean of 556 tabletop gaussians). The configuration with a
    #             measured result: served top view mean 105.4 vs the real 90.4, from 137 white.
    #   splat     ours becomes a collider only and the splat's table is the surface. What
    #             splat_align.usda intends; whether the cut leaves a visible hole under the new
    #             alignment is the open question, so it is a switch.
    # Ground, extrusion and curtains always hide: the splat has the real floor and rig.
    # purpose = guide hides a prim from every camera and keeps its collider live.
    SPLAT_TABLE = os.environ.get("SPLAT_TABLE", "splat").lower()   # default: their table is the visible surface
    _guide = ["/World/ground", "/World/extrusion", "/World/curtain_back", "/World/curtain_left",
              "/World/curtain_right", "/World/curtain_behind"]
    #   wood      ours stays visible in the plain runner's dark-wood paint, the colour tuned against
    #             Robocurve's real top-camera frame. The raw splat leaves their clutter as grey smears and
    #             black blobs on the tabletop; the cut splat removes them and this
    #             plug covers the hole in the colour the real table actually has.
    if SPLAT_TABLE == "splat":
        _guide.append("/World/table")
    elif SPLAT_TABLE == "wood":
        pass   # keep the paint applied above
    else:
        _paint("/World/table", (0.2023, 0.1815, 0.2141), 0.55)
    _hidden = []
    for _p in _guide:
        _pr = stage.GetPrimAtPath(_p)
        if _pr and _pr.IsValid():
            UsdGeom.Imageable(_pr).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            _hidden.append(_p.split("/")[-1])
    print("[splat] hidden, colliders kept: %s | visible table: %s"
          % (_hidden, {"splat": "splat's own", "wood": "authored, dark wood"}.get(SPLAT_TABLE, "authored, steel grey")), flush=True)
    # nurec_config.yaml reads its /rtx/rtpt/gaussian/* overrides once at the first Hydra sync and
    # warns that setting them after that first update() has no effect, so before reset().
    try:
        from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering, classify_stage
        _k = classify_stage(stage)
        _ok, _nu, _spg, _pb = setup_for_rendering(stage)
        print("[splat] particle_field=%s spg=%s | setup ok=%s nurec=%s problems=%s"
              % (_k.particle_field, _k.spg, _ok, _nu, _pb), flush=True)
    except Exception as _e:
        print("[splat] nurec setup unavailable: %r" % (_e,), flush=True)

world.reset()
arms = {}
for side in ("left", "right"):
    a_ = Articulation(prim_paths_expr=f"/World/{side}_arm/arm/arm"); a_.initialize()
    n_ = a_.num_dof
    # yam.yaml controller.high_pd, per actuator group, with the joint groupings from
    # yam.py: yam_shoulder joint[1-3] 800/50, yam_elbow joint4 800/50,
    # yam_wrist joint[5-6] 30/5, yam_gripper fingers 2000/100.
    # Applying 800/50 to joints 5 and 6 as well would make the wrist 26x stiffer than the
    # real robot.
    kp = np.full((1, n_), 800.0, np.float32); kd = np.full((1, n_), 50.0, np.float32)
    # The REAL arm's gain SHAPE, from the hardware vendor's own config
    # (i2rt/robots/config/yam_v1.yml): kp [80,80,80,10,10,10], kd [5,5,5,1.5,1.5,1.5], i.e. 8:1
    # shoulder-to-elbow/wrist, plus coulomb_friction [0.3,0.3,0.3,0.06,0.06,0.06] Nm on every joint.
    # Ours is 800/800/800/800/30/30, so our ELBOW is relatively ~80x stiffer than the real one and our
    # joints are frictionless. Those are DM MIT-mode gains against Isaac drive gains, so the absolute
    # numbers are not comparable; the SHAPE is. ARM_KP/ARM_KD take a comma list to test it.
    # Unset = the high_pd values above.
    _akp = os.environ.get("ARM_KP", "").strip()
    _akd = os.environ.get("ARM_KD", "").strip()
    if _akp:
        _v = [float(x) for x in _akp.split(",")]
        for _i, _x in enumerate(_v[:min(6, n_)]): kp[0, _i] = _x
    if _akd:
        _v = [float(x) for x in _akd.split(",")]
        for _i, _x in enumerate(_v[:min(6, n_)]): kd[0, _i] = _x
    if not _akp:
        kp[0, 4:6] = float(os.environ.get("WRIST_KP", "30.0"))
    if not _akd:
        kd[0, 4:6] = float(os.environ.get("WRIST_KD", "5.0"))
    if _akp or _akd:
        print("[ctrl] %s arm gains OVERRIDDEN kp=%s kd=%s" % (side, kp[0, :6].tolist(), kd[0, :6].tolist()), flush=True)
    kp[0, 6:] = float(os.environ.get("FINGER_KP", "2000.0"))
    kd[0, 6:] = float(os.environ.get("FINGER_KD", "100.0"))
    if os.environ.get("FINGER_KP") or os.environ.get("FINGER_KD"):
        print("[ctrl] %s finger drive kp=%.1f kd=%.1f (default 2000/100)"
              % (side, kp[0, 6], kd[0, 6]), flush=True)
    a_.set_gains(kps=kp, kds=kd)
    arms[side] = a_
    try:
        _l = np.asarray(a_.get_dof_limits())[0]          # (num_dof, 2) lower/upper, runtime values
        _DOF_LIM[side] = (_l[:, 0].astype(np.float32), _l[:, 1].astype(np.float32))
    except Exception as _e:
        print("[ctrl] %s arm: could not read DOF limits (%s); targets clipped to +-pi" % (side, _e), flush=True)
def _m4(path):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)) \
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m).T

# The D405s are BOLTED TO THE WRIST on real hardware, so they have to be children of
# link_6, not world prims. Planting them in world space at the home pose would freeze
# two of the policy's three image inputs: the arm moves and the wrist views never change,
# so the policy gets no evidence its own motion did anything.
def _mount_wrist(side, ysign):
    """Mount the wrist camera where yam.yaml says it is.

    yam.yaml puts it at `{ENV}/LeftArm/arm/link_6/wrist_camera` with
        left  position [-0.0004, 0.069638, 0.073063]
        right position [ 0.0,    0.069638, 0.072   ]
        both  quaternion_opengl [-0.003227, 0.002817, 0.975619, 0.219430]
    applied by yamlab as OffsetCfg(..., convention="opengl") relative to link_6.

    Note Z 0.073 is PAST the jaw tips, which sit at Z 0.064 (measured on our USD). A mount
    short of that stares into the gripper instead of at the table.

    OpenGL and USD camera conventions agree on the view direction: the camera looks
    along its own local -Z with +Y up. So the quaternion can be used as the camera
    prim's local rotation directly. The printed world forward vector is the check --
    it must point away from the arm and downward, not back into the robot.
    """
    l6 = f"/World/{side}_arm/arm/link_6"
    path = f"{l6}/{side}_wrist_view"
    cam = Camera(prim_path=path, frequency=int(CONTROL_HZ),
                 resolution=_ssres(RENDER_W, RENDER_H))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    if _WRIST_POS is not None:
        pos = np.array(_WRIST_POS[side], dtype=float)
        R = np.asarray(rot_utils.quats_to_rot_matrices(_WRIST_QUAT.reshape(1, 4)))[0]
        # WRIST_BACK_M: slide the D405 back along its own optical axis. The yam.yaml mount is a cross-rig
        # guess; in the real BimanualYAM dataset frames the fingertips span 0.68 of the image width at rest, in ours
        # 0.79, so our camera sits ~16% closer to the fingertips. 0.018 m undoes that for a ~11 cm tip distance.
        _wb = float(os.environ.get("WRIST_BACK_M", "0.0"))
        if _wb:
            pos = pos + _wb * R[:, 2]          # camera looks along local -Z, so +Z is backwards
            print("[wristcam] %s moved back %.3f m along its optical axis (WRIST_BACK_M)" % (side, _wb), flush=True)
        Tl = np.eye(4); Tl[:3, :3] = R; Tl[:3, 3] = pos
        xf.AddTransformOp().Set(Gf.Matrix4d(*Tl.T.flatten().tolist()))
        src = "yam.yaml"
    T = _m4(path)
    fwd = -T[:3, 2]              # a USD/OpenGL camera views along its local -Z
    print("[wristcam] %-5s (%s) world pos %s forward %s"
          % (side, src, np.round(T[:3, 3], 4), np.round(fwd, 3)), flush=True)
    return cam

for side, ysign in (("left", 1.0), ("right", -1.0)):
    wrist_cams[side] = _mount_wrist(side, ysign)

def _body_path(root):
    """The prim under `root` that carries UsdPhysics.RigidBodyAPI. PhysX writes the pose of THAT
    prim every step, so set_world_poses on the parent Xform is silently overwritten (the bin would
    never rotate or reset between trials)."""
    pr = stage.GetPrimAtPath(root)
    for q in Usd.PrimRange(pr):
        if q.HasAPI(UsdPhysics.RigidBodyAPI):
            return q.GetPath().pathString
    return root
_BODY = {n: _body_path(f"/World/{n}") for n, _, _, _ in OBJS}
print("[phys] rigid-body prims: %s" % {n: _BODY[n].replace("/World/", "") for n in _BODY}, flush=True)
# Realised mass/density per rigid body, read off the LOADED stage. Print-only.
# An asset variant that claims to retune mass cannot otherwise be distinguished from stock:
# the sidecar .json is metadata and is not what PhysX reads, and the .usd is binary USDC.
def _authored_mass(_p):
    q = stage.GetPrimAtPath(_p)
    if not q or not q.HasAPI(UsdPhysics.MassAPI):
        return None
    _m = UsdPhysics.MassAPI(q)
    return (_m.GetMassAttr().Get(), _m.GetDensityAttr().Get())
print("[phys] authored (mass_kg, density) per body: %s"
      % {n: _authored_mass(_BODY[n]) for n in _BODY}, flush=True)
RP = {n: RigidPrim(prim_paths_expr=_BODY[n]) for n, _, _, _ in OBJS if n not in STATIC}   # static objects have no rigid body
for r in RP.values(): r.initialize()
# Height of each object's origin above its own bottom, measured at its first placement (the spawn
# pose). The bottles' origin is at the base so 4 mm above the table worked; the spray bottle's
# origin is mid-body, so 4 mm up put half of it inside the table and PhysX ejected it.
_ZOFF = {}
# Never let the objects sleep. The renderer reads physics transforms through Fabric, which is
# only written for AWAKE bodies. A bin that had come to rest before its pose was set kept
# rendering at its old pose while physics and USD both reported the new one: identical logs,
# different pictures, run to run. Sleep threshold 0 keeps every object's render
# in sync with its physics pose at all times.
for r in RP.values():
    try:
        r.set_sleep_thresholds(np.zeros(1, dtype=np.float32))
    except Exception as _e:
        print("[phys] set_sleep_thresholds failed: %s" % _e, flush=True)
# The ACTUAL per-rig top-camera intrinsics, from Robocurve's three rs-enumerate-devices -c dumps
# (spec/robocurve_calibration/). Colour stream at 640x480, which is the mode the eval captures.
# Which rig ran which cell is on their results page; see _TASK_RIG_BY_POLICY below.
# Note cy: 250.7 to 256.1, i.e. 10 to 16 px BELOW the 240 centre on every rig. A camera model with a centred
# principal point aims about 1.0 to 1.5 degrees too high, which is ~2 cm on the table at 0.72 m.
_RIG_TOPCAM = {
    "rig1": (606.6367, 606.1340, 320.8224, 256.1416),
    "rig2": (604.6842, 603.4332, 323.2669, 250.6949),
    "rig6": (608.7092, 607.8015, 324.8832, 254.1929),
}
# Which rig each cell actually ran on (their results page), keyed on task AND policy: two cells
# disagree with their task's other policy: latte is rig-6 for pi0.5 but rig-1 for MolmoAct2, and
# blocks is rig-1 for pi0.5 but rig-2 for MolmoAct2. The policy is not passed to the runner, but
# the chunk length is the tell: 30 = MolmoAct2, 16 = pi0.5.
_TASK_RIG_BY_POLICY = {
    ("latte", False): "rig6",  ("latte", True): "rig1",
    ("bowls", False): "rig6",  ("bowls", True): "rig6",
    ("clear_table", False): "rig2", ("clear_table", True): "rig2",
    ("bottles", False): "rig1", ("bottles", True): "rig1",
    ("blocks", False): "rig1", ("blocks", True): "rig2",
}
_TASK_RIG = {t: r for (t, m), r in _TASK_RIG_BY_POLICY.items() if m == _IS_MOLMO}

def _set_intrinsics(cam, fovy, fovx, focal=24.0, cx=None, cy=None, w=None, h=None):
    gc = UsdGeom.Camera(stage.GetPrimAtPath(cam.prim_path))
    gc.GetFocalLengthAttr().Set(focal)
    _va = 2*focal*np.tan(np.radians(fovy)/2); _ha = 2*focal*np.tan(np.radians(fovx)/2)
    gc.GetVerticalApertureAttr().Set(_va)
    gc.GetHorizontalApertureAttr().Set(_ha)
    # Optionally model the PRINCIPAL POINT. The real D435s sit 10-16 px
    # low (see _RIG_TOPCAM), which tilts the modelled view ~1.3 deg up relative to the real one and biases every
    # position we back-project out of their frames by ~2 cm. USD expresses it as an aperture offset in the same
    # units as the aperture: offset = (principal - centre)/resolution * aperture. USD +y is UP, image +y is DOWN.
    if cx is not None and w:
        gc.GetHorizontalApertureOffsetAttr().Set(float(-(cx - w/2.0)/w*_ha))
    if cy is not None and h:
        # Sign: USD shifts the frustum WINDOW by +offset, so the optical axis moves the OPPOSITE way.
        # Verified with Gf.Camera.frustum.ComputeProjectionMatrix on rig1: the axis lands on pixel
        # y 256.14 against a true cy of 256.1.
        gc.GetVerticalApertureOffsetAttr().Set(float((cy - h/2.0)/h*_va))
    gc.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

for c in (base_cam, behind_cam, binshot_cam, *wrist_cams.values()):
    c.initialize()
# Instance-id segmentation on the top camera feeds the per-trial visibility gate. Attached here, before
# the splat warmup, because the annotator carries no data for its first few hundred frames.
try:
    base_cam.add_instance_id_segmentation_to_frame(); base_cam._vg_seg_on = True
    print("[vgate] instance-id segmentation annotator attached to the top camera", flush=True)
except Exception as _e:
    print("[vgate] could not attach the segmentation annotator: %r" % (_e,), flush=True)
# A USD camera prim draws as a large white frustum stand-in in every OTHER camera's
# render. /World/base_view hangs directly over the table and /World/behind sits beside
# it, so both wrist views came back mostly filled by a white slab. purpose=guide stops
# the stand-in geometry rendering and leaves the camera fully usable.
# Isaac's built-in /OmniverseKit_* are deliberately NOT touched: /OmniverseKit_Persp is
# what the default render product binds to, and marking it guide aborts the render
# thread with Mutex.h(158) "unlock() called by non-owning thread".
_hid = []
for _p in stage.Traverse():
    if _p.GetTypeName() == "Camera" and _p.GetPath().pathString.startswith("/World"):
        UsdGeom.Imageable(_p).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        _hid.append(_p.GetPath().pathString.split("/")[-1])
print("[cam ] purpose=guide, gizmos hidden: %s" % _hid, flush=True)
# Use the real per-rig intrinsics for the top camera when we know which rig this task ran on.
# TOPCAM_RIG overrides; TOPCAM_RIG=none uses the single generic issue-#2 set for an A/B.
_trig = os.environ.get("TOPCAM_RIG", _TASK_RIG.get(TASKSET, ""))
if os.environ.get("TOP_HFOV_DEG"):
    # honour the explicit override instead of silently recomputing over it
    _set_intrinsics(base_cam, FOVY_TOP, FOVX_TOP)
    print("[cam ] top camera: TOP_HFOV_DEG override in force, fovx %.3f fovy %.3f" % (FOVX_TOP, FOVY_TOP), flush=True)
elif _trig in _RIG_TOPCAM:
    _fx, _fy, _cx, _cy = _RIG_TOPCAM[_trig]
    FOVY_TOP, FOVX_TOP = _fov(_fx, _fy, RENDER_W, TOP_RENDER_H if TOP_RENDER_H else RENDER_H)
    _pp = os.environ.get("TOPCAM_PP", "0") not in ("0", "", "no")
    # The v3 layout was back-projected through a CENTRED model (principal point at (320, 240)).
    # Rendering through the same centred model makes sim objects land on the same pixels as
    # the real ones, which is the whole point of the refit. Modelling the measured principal point here
    # without refitting the layout would reintroduce a 16 px fit-vs-render mismatch, so it is OFF by
    # default and TOPCAM_PP=1 turns it on for an A/B. Per-rig FOCAL LENGTH is used either way.
    _set_intrinsics(base_cam, FOVY_TOP, FOVX_TOP,
                    cx=(_cx if _pp else None), cy=(_cy if _pp else None),
                    w=RENDER_W, h=(TOP_RENDER_H if TOP_RENDER_H else RENDER_H))
    print("[cam ] top camera: MEASURED %s intrinsics fx %.2f fy %.2f cx %.2f cy %.2f -> fovx %.3f fovy %.3f, "
          "principal point %s" % (_trig, _fx, _fy, _cx, _cy, FOVX_TOP, FOVY_TOP,
           "MODELLED (TOPCAM_PP=1)" if _pp else "at image centre, matching the layout fit"), flush=True)
else:
    _set_intrinsics(base_cam, FOVY_TOP, FOVX_TOP)      # D435, generic issue-#2 numbers
    print("[cam ] top camera: generic intrinsics, principal point NOT modelled (TOPCAM_RIG=%r)" % (_trig,), flush=True)
for c in wrist_cams.values():
    _set_intrinsics(c, FOVY_WRIST, FOVX_WRIST)         # D405
_set_intrinsics(behind_cam, 55.0, 70.0)                # recording only
_set_intrinsics(binshot_cam, 45.0, 58.0)               # bin close-up still only
if SPLAT_ENV:
    # RTPT accumulates: with 10 ticks Camera.get_rgba() is still 0-dimensional (no frame yet).
    _WARM = int(os.environ.get("SPLAT_WARMUP", "1200"))   # render ticks before the first frame
    for _i in range(_WARM):
        world.step(render=True)
        if _i in (9, 49, _WARM - 1):
            _a = np.asarray(base_cam.get_rgba())
            print("[splat] after %4d render ticks: get_rgba shape %s%s" % (_i + 1, _a.shape,
                  "  mean %.1f" % _a[:, :, :3].mean() if _a.ndim == 3 else "  (no frame)"), flush=True)
else:
    for _ in range(10): world.step(render=True)

if os.environ.get("EXPORT_USD"):
    for _ in range(30): world.step(render=True)
    out = os.environ["EXPORT_USD"]
    stage.Export(out)
    print("[out] stage exported to", out, flush=True)
    app.close()
    raise SystemExit(0)

# REPLAY_JSONL: real2sim check. Instead of asking the policy, feed Robocurve's recorded joint targets
# (per-trial actions.jsonl: header row, then {"t": n, "action": [14]}) chunk by chunk, so the
# sim arm retraces the real trial. Comparing our frames with their top/left/right.mp4 at the same t tests the robot
# model, the joint zero conventions and the camera mounts with no policy in the loop.
_REPLAY = os.environ.get("REPLAY_JSONL", "")
_RA = None; _RI = [0]
if _REPLAY:
    _rows = [json.loads(_l) for _l in open(_REPLAY) if _l.strip()]
    _RA = np.array([_r["action"] for _r in _rows if "action" in _r], np.float32)
    print("[replay] %s: %d recorded steps, served %d per chunk, no policy server used" % (_REPLAY, len(_RA), _CHUNK), flush=True)
    sock = None
else:
    sock = socket.create_connection(("127.0.0.1", PORT), timeout=900)
    print("[net] connected to policy server on %d" % PORT, flush=True)
def enc(a):
    a = np.ascontiguousarray(a); return (a.tobytes(), list(a.shape), a.dtype.str)
def dec(t):
    b, sh, ds = t; return np.frombuffer(b, dtype=np.dtype(ds)).reshape(sh)
def ask(obs, st):
    if _REPLAY:
        i = _RI[0]; a = _RA[i:i + _CHUNK]
        if len(a) < _CHUNK:
            a = np.vstack([a, np.repeat(_RA[-1:], _CHUNK - len(a), 0)]) if len(a) else np.repeat(_RA[-1:], _CHUNK, 0)
        _RI[0] += _CHUNK
        return a.astype(np.float32)
    p = pickle.dumps({"images": {k: enc(v) for k, v in obs.items()},
                      "state": enc(st.astype(np.float32)), "task": TASK}, protocol=4)
    sock.sendall(struct.pack(">I", len(p)) + p)
    hdr = b""
    while len(hdr) < 4: hdr += sock.recv(4-len(hdr))
    ln = struct.unpack(">I", hdr)[0]; buf = b""
    while len(buf) < ln: buf += sock.recv(ln-len(buf))
    r = pickle.loads(buf)
    if "error" in r: raise RuntimeError(r["error"])
    return dec(r["action"])

def _usd_pose(path, pos, q_wxyz):
    """Write translate + orient on the USD prim too. PhysX reports a body's pose through the tensor
    API immediately, but a SLEEPING body's transform is never written back to the stage, so the
    render (and BBoxCache) kept showing the bin where the reference placed it: four yaws rendered
    pixel-identical while the physics read-back said 0/90/180/270."""
    pr = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(pr); xf.ClearXformOpOrder()
    _set_translate(xf, pr, pos)
    _set_orient(xf, pr, q_wxyz)


# Objects whose "upright" has no meaning for the start check (grapes roll into a pose of their own).
NO_TILT_CHECK = {"grapes"}

def _scene_problems():
    """Bottles must be upright (unless meant to lie), on the table, and clear of each other."""
    bad = []
    cs = {n: obj_bbox(n)[0] for n in BOTTLES}
    for n in BOTTLES:
        c = cs[n]
        if n not in LYING and n not in NO_TILT_CHECK and tilt(n) > 20.0: bad.append("%s tilted %.0f deg" % (n, tilt(n)))
        if not (0.02 < c[0] < 0.82 and -0.80 < c[1] < 0.80): bad.append("%s off the table at (%.3f, %.3f)" % (n, c[0], c[1]))
        if c[2] > TT + 0.30: bad.append("%s airborne at z %.3f" % (n, c[2]))
    for i, a in enumerate(BOTTLES):
        for b in BOTTLES[i+1:]:
            if float(np.linalg.norm(cs[a][:2] - cs[b][:2])) < 0.055: bad.append("%s and %s only close" % (a, b))
    for _sn, _, (_sx, _sy), _ in OBJS:
        if _sn not in STATIC: continue
        bc = obj_bbox(_sn)[0]
        if abs(bc[0] - _sx) > 0.03 or abs(bc[1] - _sy) > 0.03: bad.append("%s displaced to (%.3f, %.3f)" % (_sn, bc[0], bc[1]))
    return bad

def home_and_place(seed):
    """Reset arms and objects, then verify; re-place up to 6 times if anything is off."""
    for attempt in range(1, 7):
        _place_once(seed + 100 * (attempt - 1))
        bad = _scene_problems()
        if not bad:
            if attempt > 1: print("[reset] valid after %d attempts" % attempt, flush=True)
            return
        print("[reset] attempt %d invalid: %s" % (attempt, "; ".join(bad)), flush=True)
    print("[reset] WARNING: scene still invalid after 6 attempts", flush=True)

def _place_once(seed):
    rs = np.random.default_rng(seed)
    for side, a_ in arms.items():
        q = np.zeros((1, a_.num_dof), np.float32); q[0, 6] = q[0, 7] = GOPEN
        a_.set_joint_positions(q); a_.set_joint_velocities(np.zeros_like(q))
        a_.set_joint_position_targets(q)
    for n, _, (x0, y0), _yaw0 in OBJS:
        if n in STATIC:
            continue   # static, posed in USD at spawn (see the spawn loop)
        elif n in LYING:
            # on its side: 90 deg about x, with a random yaw so it is not identical
            yaw = rs.uniform(0, 2*np.pi)
            qz = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])
            qx = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0])
            w0, x0q, y0q, z0q = qz; w1, x1, y1, z1 = qx
            q = np.array([[w0*w1 - x0q*x1 - y0q*y1 - z0q*z1,
                           w0*x1 + x0q*w1 + y0q*z1 - z0q*y1,
                           w0*y1 - x0q*z1 + y0q*w1 + z0q*x1,
                           w0*z1 + x0q*y1 - y0q*x1 + z0q*w1]])
            p = np.array([[x0 + OBJ_DX + rs.uniform(-JITTER, JITTER), y0 + rs.uniform(-JITTER, JITTER), TT + 0.041]])
            RP[n].set_world_poses(positions=p, orientations=q)
            _usd_pose(_BODY[n], p[0], q[0])
        else:
            if n not in _ZOFF:
                _o = float(np.asarray(RP[n].get_world_poses()[0])[0][2]); _bmin = float(obj_bbox(n)[1][2])
                _ZOFF[n] = max(0.0, _o - _bmin) if np.isfinite(_bmin) else 0.0
                print("[phys] %s origin sits %.3f m above its bottom" % (n, _ZOFF[n]), flush=True)
            p = np.array([[x0 + OBJ_DX + rs.uniform(-JITTER, JITTER), y0 + rs.uniform(-JITTER, JITTER), TT + 0.004 + _ZOFF[n]]])
            # Apply the OBJS layout yaw (4th tuple element) to upright dynamic objects too, not only to the
            # static bin and board; the spray bottle carries yaw 90.
            _y = float(os.environ.get("SPRAY_YAW_DEG", str(_yaw0))) if n == "spray" else float(_yaw0)
            _h = np.radians(_y) / 2.0
            _q = np.array([[np.cos(_h), 0.0, 0.0, np.sin(_h)]])
            RP[n].set_world_poses(positions=p, orientations=_q)
            _usd_pose(_BODY[n], p[0], _q[0])
        RP[n].set_velocities(np.zeros((1, 6), np.float32))   # bottles only; the bin is static
    for _ in range(360): world.step(render=False)
    # Refresh every camera AFTER placement. Without this the first observation of a trial was
    # the last rendered frame: the pre-placement scene on trial 1 and the END of the previous
    # trial on later ones. Under the splat + Fabric Scene Delegate, 3 ticks are not enough -- an object can
    # render at a stale pose in the visibility check and in the first observation, then correctly a few
    # seconds later. RENDER_SETTLE ticks (default 30) let the renderer catch up first.
    for _ in range(int(os.environ.get("RENDER_SETTLE", "30"))): world.render()
    for _sn in [o[0] for o in OBJS if o[0] in STATIC]:
        _c, _mn, _mx = obj_bbox(_sn)
        _Musd = np.array(UsdGeom.Xformable(stage.GetPrimAtPath("/World/%s" % _sn)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        _yaw2 = np.degrees(np.arctan2(_Musd[1, 0], _Musd[0, 0]))
        print("[%-5s] static USD transform (the only pose there is): yaw %.1f deg  centre %s  extent x %.3f y %.3f z %.3f"
              % (_sn, _yaw2, np.round(_c, 3), _mx[0]-_mn[0], _mx[1]-_mn[1], _mx[2]-_mn[2]), flush=True)
        if _sn == "bin":
            _low2 = _Musd[:3, :3] @ np.array([0.0, 0.136, 0.087])
            print("[bin ] low wall dx %+.3f dy %+.3f -> %s" % (_low2[0], _low2[1],
                  "ARMS" if _low2[0] < -0.09 else "AWAY" if _low2[0] > 0.09 else "LEFT" if _low2[1] > 0.09 else "RIGHT"), flush=True)
    for _n in BOTTLES:
        _qb = np.asarray(RP[_n].get_world_poses()[1])[0]
        print("[obj ] %-12s tilt %5.1f deg  centre %s" % (_n, tilt(_n), np.round(obj_bbox(_n)[0], 3)), flush=True)


# yam.yaml `fingers:` gives pad keypoints in each finger link's own frame, [tip,
# base1, base2], described there as the single source of truth for finger naming and
# geometry. Keypoint 0 is the exact fingertip. These supersede every link_6
# extrapolation: measuring reach from link_6 (its body centre, or a ray along its +Z)
# reported 0.105-0.122 m while both fingers were in contact with a block.
YAM_TIPS = {"left_finger":  np.array([-0.088,  0.025, -0.045]),
            "right_finger": np.array([-0.025, -0.088, -0.045])}

def _tip_world(side, finger):
    """World position of one fingertip, from yam.yaml's keypoint in the link frame."""
    pth = "/World/%s_arm/arm/%s" % (side, finger)
    pr = stage.GetPrimAtPath(pth)
    if not (pr and pr.IsValid()):
        return None
    T = np.array(UsdGeom.Xformable(pr)
                 .ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    return T[:3, :3] @ YAM_TIPS[finger] + T[:3, 3]

def fingertips():
    """Every fingertip on both arms, world frame."""
    out = []
    for side in ("left", "right"):
        for finger in ("left_finger", "right_finger"):
            p = _tip_world(side, finger)
            if p is not None:
                out.append(p)
    return out



_BBC = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

def obj_bbox(name):
    # One cache, cleared once per control step (see the scoring block), instead of a fresh BBoxCache per call.
    c = _BBC
    c.Clear()
    b = c.ComputeWorldBound(stage.GetPrimAtPath(f"/World/{name}")).ComputeAlignedRange()
    mn, mx = np.array(list(b.GetMin())), np.array(list(b.GetMax()))
    return (mn+mx)/2, mn, mx

def state14():
    o = []
    for side in ("left", "right"):
        q = np.asarray(arms[side].get_joint_positions())[0]
        o.extend(list(q[:6]) + [float(np.clip(q[6]/GOPEN, 0, 1))])
    return np.asarray(o, np.float32)

def tilt(name):
    # Signed, not abs(): with abs() a 180-degree flip reports tilt 0.0 and an UPSIDE-DOWN object would
    # score the "upright" rung (10 instead of 8).
    q = np.asarray(RP[name].get_world_poses()[1])[0]
    return float(np.degrees(np.arccos(np.clip(q2r(q.reshape(1, 4))[0][:, 2][2], -1.0, 1.0))))

# Task maximum and the score Robocurve's own report calls a success (bottles 16: they scored 0/20
# upright in twenty real runs; latte 10 and clear_table 70 are full marks in their report).
MAX_SCORE  = {"bottles": 20, "latte": 10, "clear_table": 70, "blocks": 30, "bowls": 50}[TASKSET]
SUCCESS_AT = {"bottles": 16, "latte": 10, "clear_table": 70, "blocks": 30, "bowls": 50}[TASKSET]
rows = []
_JSONL = os.path.join(OUT, "results.jsonl")
_DONE = set()
if os.path.exists(_JSONL):
    for _ln in open(_JSONL):
        _ln = _ln.strip()
        if _ln:
            try:
                _r = json.loads(_ln); rows.append(_r)
                # a VOIDED trial is not done: on resume it is run again (same seed 1000+trial; the policy is stochastic), so a job that lost trials to an
                # invalid start scene still ends with TRIALS valid ones
                if not _r.get("voided"): _DONE.add(int(_r["trial"]))
            except Exception:
                pass
    if _DONE:
        print("[resume] %d trials already in %s, skipping them: %s" % (len(_DONE), _JSONL, sorted(_DONE)), flush=True)
# every joint target we actually command, so the same trial can be replayed inside the
# Isaac Sim GUI at full render quality instead of only existing as a headless mp4
_WANT_DBG = {}
SAVE_CAMS = os.environ.get("SAVE_CAMS", "1") not in ("0", "", "no")
_CAM_GAIN = None
if any(os.environ.get(k) for k in ("CAM_GAIN_R", "CAM_GAIN_G", "CAM_GAIN_B")):
    _CAM_GAIN = np.array([float(os.environ.get("CAM_GAIN_R", "1")), float(os.environ.get("CAM_GAIN_G", "1")),
                          float(os.environ.get("CAM_GAIN_B", "1"))], np.float32)
    print("[cam ] per-channel gain on policy images: %s" % _CAM_GAIN.tolist(), flush=True)
_CAM_GAMMA = None
if any(os.environ.get(k) for k in ("CAM_GAMMA_R", "CAM_GAMMA_G", "CAM_GAMMA_B")):
    _CAM_GAMMA = np.array([float(os.environ.get("CAM_GAMMA_R", "1")), float(os.environ.get("CAM_GAMMA_G", "1")),
                           float(os.environ.get("CAM_GAMMA_B", "1"))], np.float32)
    print("[cam ] per-channel gamma on policy images: %s" % _CAM_GAMMA.tolist(), flush=True)
_REC = {}
for trial in range(1, TRIALS+1):
    if trial in _DONE:
        continue
    PEN.clear()
    home_and_place(1000 + trial)
    rest = {n: obj_bbox(n)[0].copy() for n, _, _, _ in OBJS}
    # start-of-trial sanity: everything upright, on the table, and not interpenetrating
    _bad = []
    for _n in BOTTLES:
        _t = tilt(_n)
        _c = rest[_n]
        if _n not in LYING and _t > 20.0:
            _bad.append("%s tilted %.0f deg" % (_n, _t))
        if not (0.02 < _c[0] < 0.82 and -0.80 < _c[1] < 0.80):
            _bad.append("%s off the table at (%.3f, %.3f)" % (_n, _c[0], _c[1]))
        if _c[2] > TT + 0.30:
            _bad.append("%s airborne at z %.3f" % (_n, _c[2]))
    for _i, _a in enumerate(BOTTLES):
        for _b in BOTTLES[_i+1:]:
            _d = float(np.linalg.norm(rest[_a][:2] - rest[_b][:2]))
            if _d < 0.055:
                _bad.append("%s and %s only %.3f m apart" % (_a, _b, _d))
    print("[start] %s" % ("scene OK: all %d bottles upright, on the table, clear of each "
          "other" % len(BOTTLES) if not _bad else "SCENE INVALID: " + "; ".join(_bad)),
          flush=True)
    _SCENE_BAD = bool(_bad)   # scored trials must start from a valid scene; see _void below
    tdir = os.path.join(OUT, "t%02d" % trial); os.makedirs(tdir, exist_ok=True)
    cdir = os.path.join(OUT, "t%02d_cams" % trial)
    os.makedirs(cdir, exist_ok=True)
    fi = 0; near_min = 1e9
    LIFTMAX = {n: 0.0 for n in BOTTLES}
    WAS_IN_BIN = {}      # clear_table: object -> True once it has been inside the bin at any point
    RED_OVER_EVER = [False]   # blocks: True once the red block has been carried above the blue one
    _GCMD.clear()        # the jaws start wherever the reset put them, not where the last trial left them
    for _n in list(_WELD_REL):   # never start a trial with a bottle still pinned, or still kinematic
        _kinematic(_n, False)
    _WELD_REL.clear(); _WOPENRUN.clear()
    _WSTAT.clear(); _WSTAT.update(contact=0, closed=0, pinned=0, pins=0, releases=0)
    HELD = {n: 0 for n in BOTTLES}       # longest run of "elevated and in the gripper"
    FIRST_SCORE = {}                     # bottle -> first logged step it was held>=5 steps or inside the bin
    RUN = {n: 0 for n in BOTTLES}
    for it in range(ITERS):
        if fi == 0:
            _W0 = {k: _m4(c.prim_path)[:3, 3].copy() for k, c in wrist_cams.items()}
        def _serve(cam):
            a = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)
            if _CAM_GAMMA is not None:
                # Gain alone matches the MEAN of the real frame but crushes the highlights,
                # so small bright objects (the blocks) wash into the table and the task gets harder than
                # real. Gamma first, then gain, matches mean, p50 and p95 of the real capture.
                a = np.clip(255.0 * np.power(np.clip(a.astype(np.float32) / 255.0, 0.0, 1.0), _CAM_GAMMA),
                            0, 255).astype(np.uint8)
            if _CAM_GAIN is not None:
                # camera colour model: Robocurve's D435 start frames carry a green/blue cast and are darker
                # than our render (board 133/154/115 real vs 189/165/123 sim, latte run5). Per-channel gain
                # on every camera the policy sees. CAM_GAIN_R/G/B, default off.
                a = np.clip(a.astype(np.float32) * _CAM_GAIN, 0, 255).astype(np.uint8)
            # optionally centre-crop the rendered rows to CROP_H (0 = no crop, the default),
            # then resize to what the policy is served
            if CROP_H and a.shape[0] > CROP_H:
                y0 = (a.shape[0] - CROP_H) // 2
                a = a[y0:y0 + CROP_H]
            if (a.shape[1], a.shape[0]) != (CAM_W, CAM_H):
                a = np.asarray(Image.fromarray(a).resize((CAM_W, CAM_H)), np.uint8)
            return a
        obs = {"base_view": _serve(base_cam),
               "left_wrist_view": _serve(wrist_cams["left"]),
               "right_wrist_view": _serve(wrist_cams["right"])}
        if trial == 1 and it == 0:
            for _c, _v in obs.items():
                Image.fromarray(np.asarray(_v).astype(np.uint8)).save(
                    os.path.join(OUT, f"polcam_{TASKSET}_{_c}.png"))
            Image.fromarray(np.asarray(binshot_cam.get_rgba())[:, :, :3].astype(np.uint8)).save(
                os.path.join(OUT, "polcam_%s_binshot.png" % TASKSET))
            Image.fromarray(np.asarray(behind_cam.get_rgba())[:, :, :3].astype(np.uint8)).save(
                os.path.join(OUT, "polcam_%s_behind.png" % TASKSET))
            print("[out] saved the policy's own camera inputs + bin close-up", flush=True)
            # Per-object visibility check. Under the splat renderer a mesh can silently
            # vanish from every camera while physics reports it upright on the table: the spray bottle
            # with the raw splat, the Rubik's cube at (0.41, 0.29) with the cut one. Hide each object in
            # turn and count the top-camera pixels that change. A few hundred is normal; near zero means
            # the policy never sees it and the run is worthless. VIS_CHECK=0 skips it.
            # This hide/show check is UNRELIABLE under the Fabric Scene Delegate (it has reported the cube at
            # 0 blocks while the saved top-camera frame shows 627 cube pixels). The
            # segmentation gate below replaces it. Off by default; VIS_CHECK=1 re-enables it for comparison.
            if os.environ.get("VIS_CHECK", "0") not in ("0", "", "no"):
                # The splat render is sampled, so single pixels differ frame to frame everywhere; compare
                # 8x8 block means instead, and report the no-change baseline alongside.
                def _blocks():
                    _a = np.asarray(base_cam.get_rgba())[:, :, :3].astype(np.float32)
                    _h, _w = (_a.shape[0] // 8) * 8, (_a.shape[1] // 8) * 8
                    return _a[:_h, :_w].reshape(_h // 8, 8, _w // 8, 8, 3).mean(axis=(1, 3))
                def _changed(_x, _y): return int((np.abs(_x - _y).max(axis=2) > 15).sum())
                for _ in range(4): world.render()
                _ref = _blocks()
                for _ in range(4): world.render()
                _vis = {"_baseline_nothing_hidden": _changed(_blocks(), _ref)}
                for _n, _, _, _ in OBJS:
                    if _n in STATIC: continue   # purpose toggles on static prims do not reach the renderer in time; they read 0
                    _pr = stage.GetPrimAtPath("/World/%s" % _n)
                    if not _pr or not _pr.IsValid(): continue
                    _im = UsdGeom.Imageable(_pr); _im.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
                    for _ in range(4): world.render()
                    _vis[_n] = _changed(_blocks(), _ref)
                    _im.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
                for _ in range(4): world.render()
                # Where does each object RENDER vs where physics says it is? Print physics / USD / Fabric.
                try:
                    import usdrt
                    _rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
                except Exception as _e:
                    _rt = None; print("[pose] usdrt unavailable: %r" % (_e,), flush=True)
                for _n, _, _, _ in OBJS:
                    try:
                        _ph = np.asarray(RP[_n].get_world_poses()[0])[0] if _n in RP else None
                        _us = np.array(omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(_BODY[_n])).ExtractTranslation())
                        _fb = None
                        if _rt is not None:
                            _rp = _rt.GetPrimAtPath(_BODY[_n])
                            if _rp:
                                _names = [str(_a.GetName()) for _a in _rp.GetAttributes()]
                                _tn = [_x for _x in _names if any(_k in _x.lower() for _k in ("world", "xform", "matrix", "position"))]
                                if _n == OBJS[0][0] or _n == "cube": print("[pose] fabric attrs on %s: %s" % (_n, _tn), flush=True)
                                for _cand in ("_worldPosition", "omni:fabric:worldPosition"):
                                    if _rp.HasAttribute(_cand): _fb = np.array(_rp.GetAttribute(_cand).Get()); break
                                if _fb is None:
                                    for _cand in ("_worldMatrix", "omni:fabric:worldMatrix"):
                                        if _rp.HasAttribute(_cand):
                                            _M = np.array(_rp.GetAttribute(_cand).Get()); _fb = np.array(_M).reshape(4, 4)[3, :3]; break
                        _fmt = lambda v: "-" if v is None else "(%.3f %.3f %.3f)" % tuple(np.asarray(v)[:3])
                        print("[pose] %-9s physics %-22s usd %-22s fabric %-22s%s" % (_n, _fmt(_ph), _fmt(_us), _fmt(_fb),
                              "" if (_ph is None or _fb is None or np.linalg.norm(np.asarray(_ph)[:3] - _fb[:3]) < 0.02) else "   <-- RENDERS %.2f m FROM ITS PHYSICS POSE" % np.linalg.norm(np.asarray(_ph)[:3] - _fb[:3])), flush=True)
                    except Exception as _e:
                        print("[pose] %s: %r" % (_n, _e), flush=True)
                _b0 = _vis["_baseline_nothing_hidden"]
                print("[vis ] baseline (nothing hidden): %d of %d blocks differ" % (_b0, _ref.shape[0] * _ref.shape[1]), flush=True)
                for _n, _px in _vis.items():
                    if _n.startswith("_"): continue
                    print("[vis ] %-9s %5d blocks (8x8 px) change when hidden%s" % (_n, _px, "" if _px >= _b0 + 6 else "   <-- INVISIBLE TO THE POLICY"), flush=True)
                with open(os.path.join(OUT, "visibility.json"), "w") as _f:
                    json.dump(_vis, _f, indent=1)
        # Visibility GATE, every trial: under the splat renderer a mesh can silently fail to render
        # while physics has it on the table. Hide each dynamic
        # object in turn and count top-camera 8x8 blocks that change. Invisible -> re-poke its pose through USD,
        # toggle visibility, re-check, 3 attempts. Still invisible -> the trial is VOIDED, never scored.
        if it == 0 and os.environ.get("VIS_GATE", "1") not in ("0", "", "no"):
            # Preferred instrument: the renderer's own instance-id segmentation of the top camera. It reports
            # exactly which prim each pixel came from, so it needs no hide/show toggling (toggles reach the FSD
            # renderer late and give false readings). Falls back to the toggle method if the
            # annotator is unavailable in this build.
            def _seg_counts():
                try:
                    if not getattr(base_cam, "_vg_seg_on", False):
                        base_cam.add_instance_id_segmentation_to_frame(); base_cam._vg_seg_on = True
                    _sg = None
                    for _try in range(10):          # the annotator needs a few frames before it carries data
                        for _ in range(4): world.render()
                        _fr = base_cam.get_current_frame(); _sg = _fr.get("instance_id_segmentation")
                        if _sg and _sg.get("data") is not None and np.asarray(_sg["data"]).size: break
                        _sg = None
                    if not _sg:
                        print("[vgate] segmentation annotator returned no data after 40 ticks, using hide/show toggles", flush=True); return None
                    _dat = np.asarray(_sg["data"]); _lab = _sg["info"]["idToLabels"]
                    _out = {}
                    for _n, _, _, _ in OBJS:
                        if _n in STATIC: continue
                        _ids = [int(k) for k, v in _lab.items() if str(v).startswith("/World/%s" % _n)]
                        _m = np.isin(_dat, _ids) if _ids else np.zeros(_dat.shape, bool)
                        _ys, _xs = np.where(_m)
                        _out[_n] = (int(_m.sum()), (int(_xs.mean()), int(_ys.mean())) if _m.sum() else None)
                    return _out
                except Exception as _e:
                    print("[vgate] segmentation unavailable (%r), using hide/show toggles" % (_e,), flush=True); return None
            _sc = _seg_counts()
            if _sc is not None:
                _invis = []
                for _n, (_cnt, _cen) in _sc.items():
                    print("[vgate] trial %d: %-9s %5d px in the top camera%s" % (trial, _n, _cnt, ("  centroid %s" % (_cen,)) if _cen else "   <-- INVISIBLE"), flush=True)
                for _att in range(3):
                    _bad_n = [n for n, (c, _) in _sc.items() if c < 20]
                    if not _bad_n: break
                    print("[vgate] trial %d: %s invisible, repair attempt %d (re-pose + 30 render ticks)" % (trial, _bad_n, _att + 1), flush=True)
                    for _n in _bad_n:
                        if _n in RP:
                            _pp, _qq = RP[_n].get_world_poses(); _usd_pose(_BODY[_n], np.asarray(_pp)[0], np.asarray(_qq)[0])
                    for _ in range(30): world.render()
                    _sc = _seg_counts() or _sc
                _invis = [n for n, (c, _) in _sc.items() if c < 20]
                if _invis:
                    print("[vgate] trial %d: %s STILL INVISIBLE after repair -> trial VOIDED" % (trial, _invis), flush=True)
                    _SCENE_BAD = True
                else:
                    print("[vgate] trial %d: all %d dynamic objects visible (segmentation)" % (trial, len(_sc)), flush=True)
            else:
              def _vg_blocks():
                  _a = np.asarray(base_cam.get_rgba())[:, :, :3].astype(np.float32)
                  _h, _w = (_a.shape[0] // 8) * 8, (_a.shape[1] // 8) * 8
                  return _a[:_h, :_w].reshape(_h // 8, 8, _w // 8, 8, 3).mean(axis=(1, 3))
              def _vg_changed(_x, _y): return int((np.abs(_x - _y).max(axis=2) > 15).sum())
              for _ in range(4): world.render()
              _vref = _vg_blocks()
              for _ in range(4): world.render()
              _vb0 = _vg_changed(_vg_blocks(), _vref)
              _invis = []
              for _n, _, _, _ in OBJS:
                  if _n in STATIC: continue
                  _pr = stage.GetPrimAtPath("/World/%s" % _n)
                  if not _pr or not _pr.IsValid(): continue
                  _im = UsdGeom.Imageable(_pr); _ok = False
                  for _att in range(3):
                      _im.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
                      for _ in range(4): world.render()
                      _px = _vg_changed(_vg_blocks(), _vref)
                      _im.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
                      for _ in range(4): world.render()
                      if _px >= _vb0 + 6:
                          _ok = True; break
                      print("[vgate] trial %d: %s INVISIBLE (%d blocks, baseline %d), repair attempt %d"
                            % (trial, _n, _px, _vb0, _att + 1), flush=True)
                      _im.MakeInvisible()
                      for _ in range(3): world.render()
                      _im.MakeVisible()
                      if _n in RP:
                          _pp, _qq = RP[_n].get_world_poses()
                          _usd_pose(_BODY[_n], np.asarray(_pp)[0], np.asarray(_qq)[0])
                      for _ in range(12): world.render()
                      _vref = _vg_blocks()
                      for _ in range(4): world.render()
                      _vb0 = _vg_changed(_vg_blocks(), _vref)
                  if not _ok: _invis.append(_n)
              if _invis:
                  print("[vgate] trial %d: %s flagged by the hide/show fallback. ADVISORY ONLY: this method misreads under the Fabric Scene Delegate (job 129g), not voiding." % (trial, _invis), flush=True)
              else:
                  print("[vgate] trial %d: all %d dynamic objects visible to the top camera" % (trial, len(BOTTLES)), flush=True)
        _st = state14()
        act = ask(obs, _st)
        # MOLMO_NUM_STEPS is the flow-matching SOLVER step count, NOT the action horizon.
        # Probed against the live MolmoAct2-BimanualYAM server: num_steps 5/10/16/25/30 all return a
        # (30, 14) chunk and only dt_ms moves (548 -> 2304 ms). Robocurve's own stack says the same
        # (inspect-robots-yam config.py ActServerConfig: "does NOT control how many actions come back";
        # action_horizon is metadata and "the actual length is always taken from the server's response").
        # Executing all 30 is therefore CORRECT for this rig, and it is what the real robot did: the
        # per-step jump in Robocurve's recorded actions concentrates at period 30, phase 29 (2.4x the mean
        # on bottles run07, 4.4x bowls run07, 3.6x blocks run11), against period 16 for pi0.5.
        # allenai/molmoact2's own examples/yam client hardcodes action_horizon = 25, but that is not the
        # stack Robocurve ran. MOLMO_HORIZON exists only to A/B that upstream value; unset = execute the
        # whole returned chunk, which matches the real cadence.
        if _HORIZON and act.shape[0] > _HORIZON:
            act = act[:_HORIZON]
        if trial == 1:
            _a = np.asarray(act, float)
            _d0 = _a[0, :] - _st                     # first target minus current state
            _dl = _a[-1, :] - _st                    # last target minus current state
            _span = _a.max(0) - _a.min(0)            # spread within the chunk
            print("[chunk %02d] shape %s" % (it, tuple(_a.shape)), flush=True)
            print("           state  L %s | R %s"
                  % (np.round(_st[:7], 4), np.round(_st[7:], 4)), flush=True)
            print("           act[0] L %s | R %s"
                  % (np.round(_a[0, :7], 4), np.round(_a[0, 7:], 4)), flush=True)
            print("           act[-1]L %s | R %s"
                  % (np.round(_a[-1, :7], 4), np.round(_a[-1, 7:], 4)), flush=True)
            print("           |act[0]-state| max %.5f mean %.5f   "
                  "|act[-1]-state| max %.5f mean %.5f   in-chunk span max %.5f"
                  % (np.abs(_d0).max(), np.abs(_d0).mean(),
                     np.abs(_dl).max(), np.abs(_dl).mean(), _span.max()), flush=True)
            print("           gripper: state L %.3f R %.3f -> commanded L %.3f R %.3f"
                  % (_st[6], _st[13], _a[-1, 6], _a[-1, 13]), flush=True)
        for t in range(act.shape[0]):
            tgt = act[t]
            for side, off in (("left", 0), ("right", 7)):
                a_ = arms[side]
                cur = np.asarray(a_.get_joint_positions())[0].copy()
                want = cur.copy()
                # absolute target, as their _send() does, with no per-step delta limit on the arm.
                # Clipped to each joint's OWN limits, as Robocurve's controller does; with a +-pi clip
                # the policy can command e.g. joint4 below its -1.571 limit, so the drive pushes into the
                # stop instead of resting on it. Falls back to +-pi if limits are unavailable.
                # CLIP_JOINTS=0 restores the +-pi clip for an A/B.
                _lim = _DOF_LIM.get(side) if os.environ.get("CLIP_JOINTS", "1") not in ("0", "", "no") else None
                if _lim is not None:
                    want[:6] = np.clip(tgt[off:off+6], _lim[0][:6], _lim[1][:6])
                else:
                    want[:6] = np.clip(tgt[off:off+6], -np.pi, np.pi)
                # Rate-limit the COMMANDED gripper target to the hardware's stroke speed. Measured across
                # all 9 of Robocurve's recorded trials (43,121 steps): the per-step change in the gripper channel never
                # exceeds 1/30 = 0.03333, and the 99.9th percentile EQUALS the max, so it is a hard cap, not a habit.
                # 1/30 per step at 30 Hz is a full stroke in exactly 1.000 s -> GRIPPER_STROKE_S = 1.0. (A saturated
                # close is 25 steps for 0.9 -> 0.1, which is 0.8 of range at the cap; a slower non-saturated close
                # is not the cap.) Stepping the whole stroke in ONE control step makes first contact an impact, not
                # a squeeze, and knocks the object out of the jaws.
                # Limit the target, never the measured position, so a blocked finger still builds force.
                _gdes = float(np.clip(tgt[off+6], 0.0, 1.0))*GOPEN
                if _GSTROKE > 0.0:
                    _gprev = _GCMD.get(side)
                    if _gprev is None:
                        _gprev = float(cur[6])
                    _gstep = abs(GOPEN)/(_GSTROKE*CONTROL_HZ)
                    _gdes = float(np.clip(_gdes, _gprev - _gstep, _gprev + _gstep))
                _GCMD[side] = _gdes
                # GRIP_DEBUG=1: per-STEP gripper trace for trial 1. The [grip] print below
                # only fires at chunk boundaries (t == 0), so a close that happens inside a 30-step chunk is
                # invisible there.
                if _GRIP_DBG and trial == 1:
                    _cur6 = float(np.asarray(arms[side].get_joint_positions())[0][6])
                    print("[gripstep] it %4d t %2d %-5s norm_policy %.4f  target %.5f  achieved %.5f  gap %.5f"
                          % (it, t, side, float(np.clip(tgt[off+6], 0.0, 1.0)), _gdes, _cur6, _gdes - _cur6),
                          flush=True)
                want[6] = want[7] = _gdes
                a_.set_joint_position_targets(want.reshape(1, -1).astype(np.float32))
                if trial == 1:
                    _WANT_DBG[side] = want.copy()
                _REC.setdefault(side, []).append(want.astype(np.float32).copy())
            if trial == 1 and t == 0 and (it % 40) == 0:
                # Is the gripper actually closing, or stalling against the object? Print the commanded
                # finger target next to the achieved position. A large persistent gap means the drive is blocked
                # (a real squeeze); a small gap with the jaws still open means it never reached the object.
                for _sd in ("left", "right"):
                    _cur = np.asarray(arms[_sd].get_joint_positions())[0]
                    print("[grip] it %3d %s: commanded %.4f  achieved %.4f  gap %.4f m"
                          % (it, _sd, _GCMD.get(_sd, float("nan")), _cur[6], _GCMD.get(_sd, 0.0) - _cur[6]), flush=True)
            TOUCH.clear()
            for _ in range(max(1, int(round((1.0/CONTROL_HZ)/(1/240.0))))):
                world.step(render=False)
            if trial == 1 and t == act.shape[0] - 1:
                for sd in ("left", "right"):
                    _got = np.asarray(arms[sd].get_joint_positions())[0]
                    _err = _got[:6] - _WANT_DBG[sd][:6]
                    print("           %-5s arm tracking: max|commanded-reached| %.5f rad "
                          "(%.3f deg), fingers commanded %s reached %s"
                          % (sd, np.abs(_err).max(), np.degrees(np.abs(_err).max()),
                             np.round(_WANT_DBG[sd][6:8], 4), np.round(_got[6:8], 4)),
                          flush=True)
            # Render WITHOUT stepping physics. isaacsim.core.api simulation_context.py:704-710:
            # step(render=True) with a non-zero rendering_dt calls app.update(), which advances
            # the timeline by another rendering_dt of physics. The loop above already stepped
            # exactly one control period, so step(render=True) here doubled every action's
            # duration (halving the effective control rate). render() flips /app/player/playSimulations
            # off around app.update() (same file :744-746), so it only refreshes the cameras.
            world.render()
            if t % 2 == 0:
                im = np.asarray(behind_cam.get_rgba())[:, :, :3].astype(np.uint8)
                Image.fromarray(im).resize((480, 360)).save(os.path.join(tdir, "%04d.png" % fi))
                # also keep the three frames the policy itself was given this step,
                # tiled, so the whole trial can be watched from the policy's point of
                # view rather than from a spectator camera
                if SAVE_CAMS:
                    tw, th = 320, 180
                    tile = Image.new("RGB", (tw*3, th), (12, 12, 12))
                    for _i, _k in enumerate(("base_view", "left_wrist_view",
                                             "right_wrist_view")):
                        _a = obs.get(_k)
                        if _a is None:
                            continue
                        tile.paste(Image.fromarray(np.asarray(_a).astype(np.uint8))
                                   .resize((tw, th)), (_i*tw, 0))
                    _d = ImageDraw.Draw(tile)
                    for _i, _lab in enumerate(("base / top D435", "left wrist D405",
                                               "right wrist D405")):
                        _d.rectangle([_i*tw, 0, _i*tw+tw, 13], fill=(0, 0, 0))
                        _d.text((_i*tw+4, 2), _lab, fill=(255, 220, 90))
                    _d.text((4, th-13), "t=%.1fs" % (t/CONTROL_HZ), fill=(255, 220, 90))
                    tile.save(os.path.join(cdir, "%04d.png" % fi))
                fi += 1
            # This bookkeeping runs every CONTROL STEP, not once per chunk: TOUCH is cleared every step, so a
            # per-chunk read would see only the last step of each chunk, make HELD count chunks instead of
            # steps, and miss a lift-and-drop inside one chunk.
            # Reach is measured from the fingertips yam.yaml specifies, not from link_6, whose body
            # centre sits behind the pads: it reads 0.105-0.122 m in trials where
            # the contact report shows 29,000 contacts per finger against a block.
            tips = fingertips()
            for n in BOTTLES:
                cz = obj_bbox(n)[0]
                up = float(cz[2] - rest[n][2])
                LIFTMAX[n] = max(LIFTMAX[n], up)
                if tips:
                    near_min = min(near_min,
                                   min(float(np.linalg.norm(t - cz)) for t in tips))
                # A hold is both fingers of one arm in contact with the same object, which
                # is yam.yaml's check_pad criterion. No lift precondition, so a firm
                # grasp that never leaves the table still registers.
                gripped_now = any(
                    {(n, side, "left_finger"), (n, side, "right_finger")} <= TOUCH
                    for side in ("left", "right"))
                if gripped_now:
                    RUN[n] += 1; HELD[n] = max(HELD[n], RUN[n])
                else:
                    RUN[n] = 0
                if _WELD_N > 0:
                    _wside = next((_s for _s in ("left", "right")
                                   if {(n, _s, "left_finger"), (n, _s, "right_finger")} <= TOUCH), None)
                    # GOPEN is negative and _gdes = norm*GOPEN, so _gnow >= _WELD_OPEN means "commanded shut".
                    _gnow = _GCMD.get(_wside, GOPEN) if _wside else GOPEN
                    if _wside:
                        _WSTAT["contact"] = _WSTAT.get("contact", 0) + 1
                        if _gnow >= _WELD_OPEN:
                            _WSTAT["closed"] = _WSTAT.get("closed", 0) + 1
                    if (_wside and n not in _WELD_REL and RUN[n] >= _WELD_N
                            and _gnow >= _WELD_OPEN):
                        _T6 = _m4("/World/%s_arm/arm/link_6" % _wside)
                        _p, _q = RP[n].get_world_poses()
                        _To = np.eye(4); _To[:3, 3] = np.asarray(_p)[0]
                        _To[:3, :3] = q2r(np.asarray(_q).reshape(1, 4))[0]
                        _WELD_REL[n] = (_wside, np.linalg.inv(_T6) @ _To)
                        _WOPENRUN[n] = 0
                        _kinematic(n, True)
                        _WSTAT["pins"] = _WSTAT.get("pins", 0) + 1
                        print("[weld] %s pinned to %s link_6 after %d held steps" % (n, _wside, RUN[n]), flush=True)
                    # Release only after the gripper has been commanded open for WELD_HYST steps running,
                    # so a one-step dip across the threshold cannot drop the bottle.
                    if n in _WELD_REL:
                        if _GCMD.get(_WELD_REL[n][0], GOPEN) < _WELD_OPEN:
                            _WOPENRUN[n] = _WOPENRUN.get(n, 0) + 1
                        else:
                            _WOPENRUN[n] = 0
                        if _WOPENRUN.get(n, 0) >= _WELD_HYST:
                            _WELD_REL.pop(n, None); _WOPENRUN.pop(n, None)
                            _kinematic(n, False)
                            try: RP[n].set_velocities(np.zeros((1, 6), np.float32))
                            except Exception: pass
                            _WSTAT["releases"] = _WSTAT.get("releases", 0) + 1
                            print("[weld] %s RELEASED (gripper open %d steps)" % (n, _WELD_HYST), flush=True)
                    if n in _WELD_REL:
                        _ws, _rel = _WELD_REL[n]
                        _Tw = _m4("/World/%s_arm/arm/link_6" % _ws) @ _rel
                        _qq = rot_utils.rot_matrices_to_quats(_Tw[:3, :3].reshape(1, 3, 3))
                        RP[n].set_world_poses(positions=_Tw[:3, 3].reshape(1, 3).astype(np.float32),
                                              orientations=np.asarray(_qq).reshape(1, 4).astype(np.float32))
                        try: RP[n].set_velocities(np.zeros((1, 6), np.float32))
                        except Exception: pass
                        _WSTAT["pinned"] = _WSTAT.get("pinned", 0) + 1
                if TASKSET == "blocks" and n == "block_red" and not RED_OVER_EVER[0]:
                    _cb = obj_bbox("block_blue")[0]
                    if (float(np.linalg.norm(cz[:2] - _cb[:2])) < 0.05 and cz[2] - _cb[2] > 0.02
                            and HELD.get("block_red", 0) >= 5):
                        RED_OVER_EVER[0] = True
                if TASKSET == "clear_table" and not WAS_IN_BIN.get(n):
                    _bc2, _bmn2, _bmx2 = obj_bbox("bin")
                    if (_bmn2[0] < cz[0] < _bmx2[0] and _bmn2[1] < cz[1] < _bmx2[1]
                            and _bmn2[2] < cz[2] < _bmx2[2] + 0.05):
                        WAS_IN_BIN[n] = True
                if n not in FIRST_SCORE and TASKSET not in ("blocks", "latte", "clear_table", "bowls"):
                    _bc, _bmn, _bmx = obj_bbox("bin")
                    _inb_now = (_bmn[0] < cz[0] < _bmx[0] and _bmn[1] < cz[1] < _bmx[1] and _bmn[2] < cz[2] < _bmx[2] + 0.05)
                    if RUN[n] >= 5 or _inb_now:
                        FIRST_SCORE[n] = t
    # Bottles, per Robocurve's published rubric (kaedim.pages.dev/real-world-evals, "Tasks and
    # rubrics"): score per bottle, at most two bottles counted, further movement scores 0 incremental.
    #   in bin and upright      10
    #   in bin, not upright      8
    #   picked up                5
    #   never picked up          0
    per = {}
    if TASKSET == "blocks":
        # Robocurve's Stack Blocks rubric (published, see above), a ladder, final state:
        #   arm moves towards the red block and attempts a pick        5
        #   red block gripped and lifted clear of the table            10
        #   red block positioned over the blue block (carried above)   20
        #   red block resting on the blue block at the end             30
        #   blue block resting on the red block at the end             15
        # "carried above" needs history we do not keep per step, so it is read as: red was lifted clear
        # and ended within 5 cm (xy) of blue without resting on it.
        cr, rmn, rmx = obj_bbox("block_red"); cb, bmn, bmx = obj_bbox("block_blue")
        dxy = float(np.linalg.norm(cr[:2] - cb[:2]))
        red_on_blue  = dxy < 0.03 and 0.025 < (cr[2] - cb[2]) < 0.06
        blue_on_red  = dxy < 0.03 and 0.025 < (cb[2] - cr[2]) < 0.06
        red_lifted   = HELD.get("block_red", 0) >= 5 and LIFTMAX.get("block_red", 0.0) > 0.03
        # "carried above it, released or not" is an event, so use the running flag set during
        # the trial (RED_OVER_EVER) rather than the final pose only.
        red_over     = (red_lifted and dxy < 0.05 and not red_on_blue) or (RED_OVER_EVER[0] and not red_on_blue)
        attempted    = HELD.get("block_red", 0) >= 2 or near_min < 0.03
        per = {"block_red": 0, "block_blue": 0}
        if red_on_blue:  score = 30
        elif red_over:   score = 20
        elif blue_on_red: score = 15
        elif red_lifted: score = 10
        elif attempted:  score = 5
        else:            score = 0
        per["block_red"] = score
        in_bin = red_on_blue
    elif TASKSET == "latte":
        # Robocurve's published move-latte-cup rubric:
        #   cup on cutting board and upright 10 / on board, not upright 8 / gripped or moved 5 / else 0
        _, bmn, bmx = obj_bbox("board")
        c, cmn, _ = obj_bbox("cup")
        # Ceiling: without an upper bound a cup held in the gripper 30 cm above the board would score
        # the full 10. 5 cm of headroom matches the clear_table test.
        on_board = (bmn[0] < c[0] < bmx[0] and bmn[1] < c[1] < bmx[1]
                    and bmx[2] - 0.015 < cmn[2] < bmx[2] + 0.05)
        up = tilt("cup") < 30.0
        # "gripped / moved": held in the gripper, displaced 3 cm, OR lifted 2 cm and set back down
        # (a cup lifted 4 cm and dropped in place would otherwise score 0 where a human gives 5).
        moved = (HELD.get("cup", 0) >= 5 or float(np.linalg.norm(obj_bbox("cup")[0][:2] - rest["cup"][:2])) > 0.03
                 or LIFTMAX.get("cup", 0.0) > 0.02)
        per["cup"] = 10 if (on_board and up) else 8 if on_board else 5 if moved else 0
        for n in BOTTLES:
            per.setdefault(n, 0)
        score = int(per["cup"])
        in_bin = bool(on_board)
    elif TASKSET == "clear_table":
        # Robocurve's published clear-table rubric, per item, summed, max 70:
        #   item        moved  lifted  placed in box
        #   cube          5      10      15
        #   baseball      5      10      15
        #   plate        10      17      20
        #   mug           5       8      10
        #   grapes        5       8      10
        # "Zero for any item that is moved out of the box before the run ends" is the final-state read.
        RUNGS = {"cube": (5, 10, 15), "baseball": (5, 10, 15), "plate": (10, 17, 20),
                 "mug": (5, 8, 10), "grapes": (5, 8, 10)}
        _, bmn, bmx = obj_bbox("bin")
        for n in BOTTLES:
            c = obj_bbox(n)[0]
            placed = (bmn[0] < c[0] < bmx[0] and bmn[1] < c[1] < bmx[1] and bmn[2] < c[2] < bmx[2] + 0.05)
            lifted_ = HELD.get(n, 0) >= 5 and LIFTMAX.get(n, 0.0) > 0.03
            moved_ = HELD.get(n, 0) >= 3 or float(np.linalg.norm(c[:2] - rest[n][:2])) > 0.02
            r = RUNGS[n]
            # Their rubric says "Zero for any item moved out of the box before the run ends": an item placed and
            # then pulled back out scores 0, not one rung lower. WAS_IN_BIN is set per control step by the scoring loop.
            if WAS_IN_BIN.get(n) and not placed:
                per[n] = 0
            else:
                per[n] = r[2] if placed else r[1] if lifted_ else r[0] if moved_ else 0
        score = int(sum(per.values()))
        in_bin = any(per[n] == RUNGS[n][2] for n in BOTTLES)
    elif TASKSET == "bowls":
        # Robocurve's published Stack Bowls rubric, final state, a ladder with lifts below it:
        #   a bowl is lifted (once, per bowl)                                   5 each
        #   2 bowls almost stacked: one bowl in overlapping contact with another 20
        #   2 bowls stacked                                                     30
        #   3 bowls almost stacked: 3rd bowl overlapping a stack of 2           40
        #   3 bowls stacked                                                     50
        # stacked = nested: centres within 4 cm (xy) and the upper bowl 1-7 cm higher. almost = rims overlap
        # (xy within BOWL_ALMOST_M), upper bowl at least 1 cm higher, not nested.
        # "almost" = Robocurve's "one bowl in overlapping contact with another". That wording
        # carries no number. Rims overlap when centres are within one rim DIAMETER.
        # Our bowl measures 0.1439 x 0.1441 x 0.0712 m in the USD; the real bowl is 5.8 in x 2.9 in
        # = 0.1473 x 0.0737 m (manufacturer spec), so ours is the real bowl to within 2.3%.
        # A 0.10 m threshold would demand overlap to 68% of the bowl's own diameter, well past "overlapping".
        _ALMOST_M = float(os.environ.get("BOWL_ALMOST_M", "0.144"))
        cs = {n: obj_bbox(n)[0] for n in BOTTLES}
        def _rel(a, b):
            d = float(np.linalg.norm(cs[a][:2] - cs[b][:2])); dz = float(cs[a][2] - cs[b][2])
            if d < 0.04 and 0.01 < dz < 0.07: return "stacked"
            if d < _ALMOST_M and dz > 0.01: return "almost"
            return None
        rels = {(a, b): _rel(a, b) for a in BOTTLES for b in BOTTLES if a != b}
        n_stacked = sum(1 for v in rels.values() if v == "stacked")
        n_almost  = sum(1 for v in rels.values() if v == "almost")
        lifted_bowls = [n for n in BOTTLES if HELD.get(n, 0) >= 5 and LIFTMAX.get(n, 0.0) > 0.03]
        # The 50 rung is "3 bowls stacked, UPRIGHT" in their rubric, so a toppled pile must not score
        # full marks. A bowl past 40 deg is not stacked upright.
        _all_upright = all(tilt(n) < 40.0 for n in BOTTLES)
        if n_stacked >= 2 and _all_upright:   score = 50
        elif n_stacked >= 2:                  score = 40   # three bowls together but not upright: cap at "almost"
        elif n_stacked == 1 and n_almost >= 1: score = 40
        elif n_stacked == 1:                  score = 30
        elif n_almost >= 1:                   score = 20
        else:                                 score = 5 * len(lifted_bowls)
        per = {n: (5 if n in lifted_bowls else 0) for n in BOTTLES}
        for (a, b), v in rels.items():
            if v: print("[bowls] %s %s on %s" % (a, v, b), flush=True)
        in_bin = n_stacked >= 1
    else:
        bc, bmn, bmx = obj_bbox("bin")
        for n in BOTTLES:
            c = obj_bbox(n)[0]
            # Ceiling, same reason as on_board -- a bottle dangling above the bin must not score
            # "in bin" without ever being released into it.
            inb = (bmn[0] < c[0] < bmx[0] and bmn[1] < c[1] < bmx[1] and bmn[2] < c[2] < bmx[2] + 0.05)
            up = tilt(n) < 30.0
            # Robocurve's "picked up" rung is a lift, so require both pads in contact for
            # 5 steps AND the bottle rising at least 3 cm off its rest height; a bottle merely
            # knocked airborne does not count.
            picked = HELD.get(n, 0) >= 5 and LIFTMAX.get(n, 0.0) > 0.03
            per[n] = 10 if (inb and up) else 8 if inb else 5 if picked else 0
        # Two readings of "up to 2 bottles". best2: the two highest-scoring bottles. first2: the two
        # bottles that scored first in time, each at its final rung, which is how Robocurve's report reads
        # (the first two bottles scored, not specific bottles). On Robocurve's 20 real
        # runs the two rules give identical totals (no run mixes an 8 with a 5), so this only matters in
        # sim. SCORE_RULE picks which one is `score`; both are written to results.
        score_best2 = int(sum(sorted(per.values(), reverse=True)[:2]))
        _order = sorted((n for n in BOTTLES if per[n] > 0), key=lambda n: FIRST_SCORE.get(n, 10**9))
        score_first2 = int(sum(per[n] for n in _order[:2]))
        score = score_first2 if os.environ.get("SCORE_RULE", "best2") == "first2" else score_best2
        in_bin = any(v >= 8 for v in per.values())
    lifted = max(LIFTMAX.get(n, 0.0) for n in BOTTLES)
    # A peak lift above 0.40 m is an ejection, not a result, and the
    # bbox in_bin test would happily score it as placed if it flew through the bin.
    _void = lifted > 0.40 or _SCENE_BAD
    if _SCENE_BAD:
        print("[VOID] trial %d: start scene was INVALID (objects off the table or overlapping). Score discarded." % trial, flush=True)
    if lifted > 0.40:
        print("[VOID] trial %d: peak lift %.3f m is physically impossible -- object was "
              "ejected, not placed. Score discarded." % (trial, lifted), flush=True)
    moved = max(float(np.linalg.norm(obj_bbox(n)[0] - rest[n])) for n in BOTTLES)
    # how far each wrist camera travelled this trial. If this is ~0 the cameras are not
    # attached to the wrist and the policy is flying blind on 2 of its 3 inputs.
    _wtrav = {k: float(np.linalg.norm(_m4(c.prim_path)[:3, 3] - _W0[k]))
              for k, c in wrist_cams.items()} if "_W0" in dir() else {}
    print("[wristcam] travel this trial: %s" % {k: round(v, 4) for k, v in _wtrav.items()},
          flush=True)
    row = dict(trial=trial, frames=fi, near=round(near_min, 4), moved=round(moved, 4),
               wrist_cam_travel={k: round(v, 4) for k, v in _wtrav.items()},
               lifted=round(lifted, 4), in_bin=bool(in_bin),
               voided=bool(_void),
               score=(0 if _void else score), max_score=MAX_SCORE,
               per_bottle=per, taskset=TASKSET,
               held_steps={n: HELD[n] for n in BOTTLES},
               score_best2=locals().get("score_best2"), score_first2=locals().get("score_first2"),
               first_score_step={n: FIRST_SCORE.get(n) for n in BOTTLES if n in FIRST_SCORE},
               success=bool(score >= SUCCESS_AT),
               # 16, not 20: Robocurve scored 0/20 bottles upright in twenty
               # real runs and call 16 a success in their own report

               # Final world centre + tilt of every graspable object. Without this,
               # changing any scoring threshold forces a full re-run: the rest of the row keeps only the
               # resulting rungs, and the [obj ] log lines are RESET poses for the NEXT trial, not
               # this trial's end state. With it, a rubric change is an offline rescore.
               final_pose={n: [round(float(v), 4) for v in obj_bbox(n)[0]] + [round(float(tilt(n)), 1)]
                           for n in BOTTLES},
               tilt={n: round(tilt(n), 1) for n in BOTTLES},
               pen_mm={k: round(v[1]*1000, 2) for k, v in PEN.items()},
               weld=dict(_WSTAT))
    rows.append(row)
    with open(_JSONL, "a") as _f:
        _f.write(json.dumps(row) + "\n")
    print("[trial %2d/%d] score %2d/%d  near %.3f  best lift %.3f  held %s  in_bin %s   scored %s"
          % (trial, TRIALS, row["score"], row["max_score"], row["near"], row["lifted"],
             {k: v for k, v in row["held_steps"].items() if v > 0} or "none",
             row["in_bin"], {k: v for k, v in row["per_bottle"].items() if v > 0} or "all 0"), flush=True)
    if _WELD_N > 0:
        _c = max(1, _WSTAT.get("contact", 0))
        print("[weld] trial %d: contact %d steps, commanded-closed %d (%.1f%%), pinned %d steps, pins %d releases %d"
              % (trial, _WSTAT.get("contact", 0), _WSTAT.get("closed", 0),
                 100.0*_WSTAT.get("closed", 0)/_c, _WSTAT.get("pinned", 0),
                 _WSTAT.get("pins", 0), _WSTAT.get("releases", 0)), flush=True)

sc = [r["score"] for r in rows]
print("\n[RESULT] mean score %.2f / 20   median %.1f   any pickup: %d/%d trials"
      % (float(np.mean(sc)), float(np.median(sc)),
         sum(1 for r in rows if r["lifted"] > 0.05), len(rows)), flush=True)
print("[RESULT] best lift %.3f m   best approach %.3f m   max displacement %.3f m"
      % (max(r["lifted"] for r in rows), min(r["near"] for r in rows), max(r["moved"] for r in rows)), flush=True)
for _sd, _seq in _REC.items():
    np.save(os.path.join(OUT, "cmd_%s.npy" % _sd), np.asarray(_seq, np.float32))
print("[out] commanded targets saved: %s"
      % {k: np.asarray(v).shape for k, v in _REC.items()}, flush=True)
rows.sort(key=lambda r: r["trial"])
json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("[out] %s/results.json and t01..t%02d frame dirs" % (OUT, TRIALS), flush=True)
# The marker the job script waits for. It must come BEFORE app.close(): Kit's shutdown ends the
# process and a line after it never prints, so a finished job would look like an early exit and
# be restarted for nothing.
print("TRIALSDONE", flush=True)
app.close()
