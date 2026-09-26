"""Bimanual YAM + Kaedim bottles + bin in Isaac Sim, and prove a camera sees it.

Geometry is YAMLab's measured cell (yamlab/configs/robot/yam.yaml), so all three
camera poses and both arm bases are mutually consistent rather than half-fitted:
  arms   x 0.2525, y +/-0.305, z 0.76
  table  0.6505, 0.0, 0.7517
  top cam 0.0860, -0.0090, 1.7043, fovy from fy=391.72 at 480 px
Assets are the authored-arm (projectsim) USDs, loaded natively with their authored mass and
SDF colliders intact (a convex hull would make the bin a solid block).
"""
import os, numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                     "width": 640, "height": 360})

from pxr import UsdGeom, UsdLux, UsdPhysics, Gf, Usd
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

A = "/opt/isaacwork/assets"
TABLE_TOP = 0.7517
world = World(stage_units_in_meters=1.0, physics_dt=1/120.0, rendering_dt=1/30.0)
world.scene.add(GroundPlane(prim_path="/World/ground", size=8.0))
world.scene.add(FixedCuboid(prim_path="/World/table", name="table",
                            position=np.array([0.6505, 0.0, TABLE_TOP-0.02]),
                            scale=np.array([0.80, 1.40, 0.04]),
                            color=np.array([0.35, 0.22, 0.14])))
stage = omni.usd.get_context().get_stage()
d = UsdLux.DistantLight.Define(stage, "/World/sun"); d.CreateIntensityAttr(2500.0)
dome = UsdLux.DomeLight.Define(stage, "/World/dome"); dome.CreateIntensityAttr(700.0)

# --- two YAM arms ---
for side, y in (("left", 0.305), ("right", -0.305)):
    pp = f"/World/{side}_arm"
    add_reference_to_stage(usd_path=f"{A}/yam_robot/arm/yam.usd", prim_path=pp)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(pp))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.2525, y, 0.76))
    # The YAM USD has no fixed base, so once physics steps the whole arm falls off
    # the table. Anchor its root link to the world.
    fj = UsdPhysics.FixedJoint.Define(stage, f"{pp}/base_fix")
    fj.CreateBody1Rel().SetTargets([f"{pp}/arm/arm"])
    print(f"[add] {side} arm anchored", flush=True)

# --- Kaedim assets ---
OBJS = [("bin",      "sm_storagebin_plasticopen_f43",  (0.86,  0.00)),
        ("sapporo",  "sm_beerbottle_glasssapporo_d06", (0.56,  0.17)),
        ("heineken", "sm_bottle_glassheineken_8d6",    (0.56, -0.17))]
for name, folder, (x, y) in OBJS:
    pp = f"/World/{name}"
    add_reference_to_stage(usd_path=f"{A}/{folder}/{folder}.usd", prim_path=pp)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(pp))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, TABLE_TOP + 0.002))
    print(f"[add] {name} at ({x}, {y})", flush=True)

# --- cameras: YAMLab measured top pose; fovy from their intrinsics ---
fovy = 2*np.degrees(np.arctan(240.0/391.722351074219))
# YAMLab's measured top pose frames THEIR cell and leaves our arms out of shot.
# These come from a least-squares fit of four scene landmarks (far table edge, bin
# top, bottle top, arm base) onto the rows they occupy in the real MolmoAct2 top
# frame, rms 20.4 px.
cam_pos = np.array([0.2000, 0.0, 1.5012])
yaw, pitch = 0.0, 60.15
base_cam = Camera(prim_path="/World/base_view", position=cam_pos,
                  frequency=30, resolution=(640, 360),
                  orientation=rot_utils.euler_angles_to_quats(
                      np.array([0.0, pitch, yaw]), degrees=True))
print("[cam] base_view fovy %.2f deg  yaw %.1f pitch %.1f" % (fovy, yaw, pitch), flush=True)

ov_pos = np.array([-0.9, -1.5, 1.75])
ov_look = np.array([0.62, 0.0, 0.85])
f2 = ov_look - ov_pos; f2 /= np.linalg.norm(f2)
ov_cam = Camera(prim_path="/World/overview", position=ov_pos, frequency=30,
                resolution=(640, 360),
                orientation=rot_utils.euler_angles_to_quats(
                    np.array([0.0, np.degrees(np.arcsin(-f2[2])),
                              np.degrees(np.arctan2(f2[1], f2[0]))]), degrees=True))

world.reset()
ov_cam.initialize()
base_cam.initialize()
base_cam.set_focal_length(1.0)
# match vertical FOV: aperture = 2*f*tan(fovy/2)
base_cam.set_vertical_aperture(2.0*1.0*np.tan(np.radians(fovy)/2))
base_cam.set_horizontal_aperture(2.0*1.0*np.tan(np.radians(fovy)/2)*640/360)

# report the articulations so the policy can be wired to real joint names
from isaacsim.core.prims import Articulation
try:
    art = Articulation(prim_paths_expr="/World/left_arm.*")
    art.initialize()
    print("[art] left dof", art.num_dof, flush=True)
    print("[art] joints:", art.dof_names, flush=True)
except Exception as e:
    print("[art] Articulation wrap failed:", type(e).__name__, e, flush=True)

from pxr import UsdGeom as _UG
def _armz():
    c = _UG.BBoxCache(Usd.TimeCode.Default(), [_UG.Tokens.default_, _UG.Tokens.render])
    b = c.ComputeWorldBound(stage.GetPrimAtPath("/World/left_arm")).ComputeAlignedRange()
    return None if b.IsEmpty() else (round(b.GetMin()[2], 3), round(b.GetMax()[2], 3))
print("[arm z] before stepping:", _armz(), flush=True)
for i in range(90):
    world.step(render=True)
print("[arm z] after 90 steps:", _armz(), flush=True)
img = base_cam.get_rgba()
a = np.asarray(img)
print("[render] shape", a.shape, flush=True)
if a.size:
    rgb = a[:, :, :3].astype(np.uint8)
    print("[render] min %d max %d mean %.1f" % (rgb.min(), rgb.max(), rgb.mean()), flush=True)
    from PIL import Image
    Image.fromarray(rgb).save("/opt/isaacwork/scene_base_view.png")
    o = np.asarray(ov_cam.get_rgba())
    if o.size:
        Image.fromarray(o[:, :, :3].astype(np.uint8)).save("/opt/isaacwork/scene_overview.png")
    print("[out] scene_base_view.png scene_overview.png", flush=True)
    print("SCENE_OK", flush=True)
# Save the stage so it can be opened in the Isaac Sim GUI on the console display.
OUT_USD = "/opt/isaacwork/yam_bottles_bin.usd"
stage.Export(OUT_USD)
print("[out] stage saved:", OUT_USD, flush=True)
app.close()
print("DONE", flush=True)
