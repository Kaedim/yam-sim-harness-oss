"""GLB -> USD asset with PolaRiS-style physics, i.e. what owhan/PolaRiS-Hub scene.usda files contain and nothing more:
PhysicsRigidBodyAPI + PhysxRigidBodyAPI on the root, PhysicsCollisionAPI + PhysicsMeshCollisionAPI approximation
convexDecomposition on the mesh, a uniform xformOp:scale, NO mass, NO density, NO material bound (PhysX defaults:
density 1000 kg/m3, friction 0.5, restitution 0). Geometry is re-based so the object stands on z=0 centred in xy,
because the runner spawns each asset at its origin on the table top. Y-up GLB -> Z-up stage.

  python glb_to_usd.py --glb out/<name>/<name>.glb --name <name> --size 0.23 --out assets_polaris
"""
import os, shutil, argparse, json

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--glb", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--scale", type=float, default=None, help="uniform scale, metres per TRELLIS unit")
ap.add_argument("--size", type=float, default=None, help="target largest dimension in metres, set by eye (PolaRiS compose-GUI style); converted to a uniform scale")
ap.add_argument("--out", required=True, help="output root; writes <out>/<name>/<name>.usd")
a = ap.parse_args()

import numpy as np, trimesh
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf, Vt
# PhysxSchema is NVIDIA-only (not in usd-core); the Physx API schemas are recorded by name and resolve inside Isaac Sim.

sc = trimesh.load(a.glb, force="scene")
geoms = list(sc.geometry.values()); assert len(geoms) >= 1
m = trimesh.util.concatenate([g.copy() for g in geoms]) if len(geoms) > 1 else geoms[0].copy()
V = np.asarray(m.vertices, dtype=np.float64)
V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)              # glTF Y-up -> Z-up
V[:, :2] -= (V[:, :2].max(0) + V[:, :2].min(0)) / 2.0             # centre in xy
V[:, 2] -= V[:, 2].min()                                           # stand on z=0
F = np.asarray(m.faces, dtype=np.int32)
ext0 = V.max(0) - V.min(0)
if a.scale is None:
    assert a.size is not None, "give --size (metres, by eye) or --scale"
    a.scale = float(a.size / ext0.max())
uv = None; tex = None
if isinstance(m.visual, trimesh.visual.TextureVisuals) and m.visual.uv is not None:
    uv = np.asarray(m.visual.uv, dtype=np.float32)
    mat = m.visual.material
    tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)

od = os.path.join(a.out, a.name); os.makedirs(os.path.join(od, "textures"), exist_ok=True)
usd_path = os.path.join(od, a.name + ".usd")
stage = Usd.Stage.CreateNew(usd_path)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/" + a.name); stage.SetDefaultPrim(root.GetPrim())
root.AddScaleOp().Set(Gf.Vec3f(a.scale, a.scale, a.scale))
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
def _add_api(prim, names):
    # Physx* API schemas are not registered in usd-core, so write the apiSchemas listOp directly; Isaac resolves them.
    spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
    cur = list(spec.GetInfo("apiSchemas").GetAppliedItems()) if spec.HasInfo("apiSchemas") else []
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(cur + [n for n in names if n not in cur]))
_add_api(root.GetPrim(), ["PhysxRigidBodyAPI"])

mesh = UsdGeom.Mesh.Define(stage, "/%s/mesh" % a.name)
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, dtype=np.int32)))
mesh.CreateSubdivisionSchemeAttr("none")
ext = np.stack([V.min(0), V.max(0)]).astype(np.float32); mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(ext))
if uv is not None:
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(uv))
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()); mc.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
_add_api(mesh.GetPrim(), ["PhysxCollisionAPI", "PhysxConvexHullCollisionAPI", "PhysxConvexDecompositionCollisionAPI"])

if tex is not None:
    tex_rel = "textures/%s_basecolor.png" % a.name; tex.save(os.path.join(od, tex_rel))
    matp = UsdShade.Material.Define(stage, "/%s/Looks/mat" % a.name)
    sh = UsdShade.Shader.Define(stage, "/%s/Looks/mat/pbr" % a.name); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5); sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    st = UsdShade.Shader.Define(stage, "/%s/Looks/mat/st" % a.name); st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st"); st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tx = UsdShade.Shader.Define(stage, "/%s/Looks/mat/tex" % a.name); tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./" + tex_rel)
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    matp.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(matp)
stage.GetRootLayer().Save()
size = (V.max(0) - V.min(0)) * a.scale
json.dump({"name": a.name, "source_glb": os.path.basename(a.glb), "scale_uniform": a.scale, "scale_method": "largest dimension typed by eye (PolaRiS compose-GUI style), no measurement used", "size_by_eye_m": a.size,
           "bbox_m_after_scale": [round(float(s), 4) for s in size], "faces": int(len(F)),
           "physics": {"rigid_body": True, "collider": "convexDecomposition", "mass": "not authored (PhysX default density 1000 kg/m3)",
                       "material": "not bound (PhysX default friction 0.5, restitution 0.0)"}}, open(os.path.join(od, a.name + ".json"), "w"), indent=1)
print("[usd]", usd_path, "bbox m", np.round(size, 3), "faces", len(F))
