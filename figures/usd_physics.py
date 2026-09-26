#!/usr/bin/env python3
"""Read what each asset USD actually authors: collider approximation, mass/density, bound physics
material (friction, restitution, combine modes), root scale. Primary source for Table 1 / Figure 2.

usage: usd_physics.py ROOT [ROOT ...]  -> prints a table and writes physics_<basename>.json beside it

ROOT is an asset arm directory holding one folder per object (<id>/<id>.usd).
"""
import glob, json, os, sys
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

def attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None

def inspect(path):
    st = Usd.Stage.Open(path)
    out = dict(file=os.path.basename(path), colliders=[], mass=None, density=None, materials=[], scale=None,
               meters_per_unit=UsdGeom.GetStageMetersPerUnit(st))
    root = st.GetDefaultPrim() or st.GetPseudoRoot().GetChildren()[0]
    xf = UsdGeom.Xformable(root)
    if xf:
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                out["scale"] = list(op.Get())
    for p in st.Traverse():
        if p.HasAPI(UsdPhysics.MassAPI):
            m = attr(p, "physics:mass"); d = attr(p, "physics:density")
            if m is not None: out["mass"] = float(m)
            if d is not None: out["density"] = float(d)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            out["colliders"].append(dict(prim=p.GetPath().pathString.split("/")[-1],
                                         approximation=attr(p, "physics:approximation"),
                                         mesh=p.IsA(UsdGeom.Mesh)))
            rel = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel("physics")
            targets = rel.GetTargets() if rel else []
            for t in targets:
                mp = st.GetPrimAtPath(t)
                if mp and mp.HasAPI(UsdPhysics.MaterialAPI):
                    out["materials"].append(dict(prim=t.pathString.split("/")[-1],
                        static_friction=attr(mp, "physics:staticFriction"),
                        dynamic_friction=attr(mp, "physics:dynamicFriction"),
                        restitution=attr(mp, "physics:restitution"),
                        density=attr(mp, "physics:density"),
                        friction_combine=attr(mp, "physxMaterial:frictionCombineMode"),
                        restitution_combine=attr(mp, "physxMaterial:restitutionCombineMode")))
    # material bound at the root rather than the collider
    if not out["materials"]:
        for p in st.Traverse():
            if p.HasAPI(UsdPhysics.MaterialAPI):
                out["materials"].append(dict(prim=p.GetPath().pathString.split("/")[-1], unbound_lookup=True,
                    static_friction=attr(p, "physics:staticFriction"), dynamic_friction=attr(p, "physics:dynamicFriction"),
                    restitution=attr(p, "physics:restitution"), density=attr(p, "physics:density"),
                    friction_combine=attr(p, "physxMaterial:frictionCombineMode"),
                    restitution_combine=attr(p, "physxMaterial:restitutionCombineMode")))
    return out

def main(roots):
    for root in roots:
        res = {}
        for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
            usds = glob.glob(os.path.join(d, "*.usd")) + glob.glob(os.path.join(d, "*.usda"))
            if not usds: continue
            res[os.path.basename(d)] = inspect(usds[0])
        json.dump(res, open(os.path.join(os.path.dirname(root.rstrip("/")), "physics_%s.json" % os.path.basename(root.rstrip("/"))), "w"), indent=1)
        print("==", root, len(res), "assets")
        print("%-40s %-34s %8s %8s %6s %6s %5s %-8s %s" % ("asset", "collider approx (n prims)", "mass", "density", "sf", "df", "rest", "fcomb", "scale"))
        for k, v in res.items():
            approx = {}
            for c in v["colliders"]: approx[c["approximation"] or "none"] = approx.get(c["approximation"] or "none", 0) + 1
            m = v["materials"][0] if v["materials"] else {}
            f = lambda x: "-" if x is None else ("%.3g" % x if isinstance(x, float) else str(x))
            print("%-40s %-34s %8s %8s %6s %6s %5s %-8s %s" % (k[:40], ", ".join("%s x%d" % kv for kv in approx.items()),
                  f(v["mass"]), f(v["density"]), f(m.get("static_friction")), f(m.get("dynamic_friction")), f(m.get("restitution")),
                  f(m.get("friction_combine")), "%.3g" % v["scale"][0] if v["scale"] else "-"))


if __name__ == "__main__":
    main(sys.argv[1:])
