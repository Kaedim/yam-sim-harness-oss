#!/usr/bin/env python3
"""Figure 1: one object in both asset arms, drawn from the USDs themselves.

    python3 fig1_objects.py sm_bowl_plasticribbed_07f "ribbed bowl" \
        --a-root ASSETS/authored --b-root ASSETS/default [--photo photo.png] --out fig1_bowl

Columns: [input photograph] | textured render | mesh (flat shaded) | collision geometry | the
physical values authored in the USD. Without --photo only version A (no photo column) is written;
with it, version B as well.

Every value in the right-hand column is read from the USD by usd_physics.inspect. Nothing is typed
in: an arm that authors no mass or physics material is shown with the PhysX default it falls back
to. Textures are sampled per face from the basecolor PNG through the mesh UVs, because the USD
materials do not bind textures outside Omniverse.

Collision panels: an SDF collider (physics:approximation = sdf) collides with the mesh surface, so
the panel is the mesh. convexDecomposition is decomposed by PhysX at load and the pieces are not
stored, so the panel is a CoACD decomposition (at most 32 hulls, seed 0) as a stand-in.

Assets are laid out as <root>/<id>/<id>.usd (or .usda). The default-path arm may carry
<id>.json with "size_by_eye_m", the largest dimension typed by eye, which is shown if present.

Needs pxr (usd-core), numpy, matplotlib, pillow, coacd.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from pxr import Usd, UsdGeom, Gf
import coacd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from usd_physics import inspect   # noqa: E402

# PhysX falls back to these when nothing is authored or bound.
PHYSX_DEFAULT_DENSITY, PHYSX_DEFAULT_FRICTION, PHYSX_DEFAULT_RESTITUTION = 1000, 0.5, 0.0


def tri(counts, idx):
    out, k = [], 0
    for c in counts:
        for j in range(1, c - 1):
            out.append((idx[k], idx[k + j], idx[k + j + 1]))
        k += c
    return np.array(out, np.int64)


def find_usd(root, obj):
    for ext in (".usd", ".usda", ".usdc"):
        p = os.path.join(root, obj, obj + ext)
        if os.path.exists(p):
            return p
    sys.exit("no USD for %s under %s" % (obj, root))


def load(path):
    """The largest mesh in the stage, in metres: (points, triangles, per-triangle UVs or None,
    basecolor texture as float RGB or None)."""
    st = Usd.Stage.Open(path)
    xc = UsdGeom.XformCache()
    mpu = UsdGeom.GetStageMetersPerUnit(st) or 1.0
    best = None
    for p in st.Traverse():
        if not p.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(p)
        pts, counts = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        counts, idx = list(counts), list(idx)
        M = xc.GetLocalToWorldTransform(p)
        P = np.array([M.Transform(Gf.Vec3d(*q)) for q in pts]) * mpu
        F = tri(counts, idx)
        uvt = None
        pv = UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
        if pv and pv.Get():
            U = np.array(pv.Get(), np.float64)
            ind = pv.GetIndices()
            if ind:
                U = U[np.array(ind)]
            if pv.GetInterpolation() == "faceVarying":
                uvt = U[tri(counts, list(range(len(idx))))]
            elif len(U) == len(P):
                uvt = U[F]
        if best is None or len(F) > len(best[1]):
            best = (P, F, uvt)
    if best is None:
        sys.exit("no mesh in %s" % path)
    tex = None
    for p in st.Traverse():
        if p.GetTypeName() != "Shader":
            continue
        for a in p.GetAttributes():
            v = a.Get() if a.GetTypeName() == "asset" else None
            if v and "basecolor" in str(v.path).lower():
                tp = v.resolvedPath or os.path.join(os.path.dirname(path), v.path.lstrip("./"))
                if os.path.exists(tp):
                    tex = np.asarray(Image.open(tp).convert("RGB"), np.float32) / 255.0
    return best + (tex,)


def face_colors(F, uvt, tex):
    """basecolor at each triangle's UV centroid; flat grey without UVs or texture"""
    if uvt is None or tex is None:
        return np.tile(np.array([[0.66, 0.66, 0.66]]), (len(F), 1))
    H, W, _ = tex.shape
    c = uvt.mean(1)
    u = np.clip((c[:, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
    v = np.clip((1 - (c[:, 1] % 1.0)) * (H - 1), 0, H - 1).astype(int)
    return tex[v, u]


def decompose(P, F):
    parts = coacd.run_coacd(coacd.Mesh(P.astype(np.float64), F.astype(np.int64)),
                            threshold=0.05, max_convex_hull=32, seed=0)
    palette = plt.get_cmap("tab20").colors
    return [(np.array(v), np.array(f, np.int64), palette[i % 20]) for i, (v, f) in enumerate(parts)]


def draw(ax, pieces, azim=-55, elev=22):
    allP = np.concatenate([p for p, _, _ in pieces])
    c = allP.mean(0)
    r = (allP.max(0) - allP.min(0)).max() / 2
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    for P, F, col in pieces:
        tris = P[F]
        percol = isinstance(col, np.ndarray) and col.ndim == 2
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        keep = (n @ view) > 0
        base = col if percol else np.array(matplotlib.colors.to_rgb(col))[None]
        if keep.sum() > 0.2 * len(keep):   # back-face cull, unless the winding is inconsistent
            tris, n = tris[keep], n[keep]
            if percol:
                base = base[keep]
        shade = np.clip(0.45 + 0.55 * np.clip(n @ np.array([0.4, -0.6, 0.7]), 0, 1), 0, 1)
        cols = base * shade[:, None]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=np.c_[cols, np.ones(len(cols))],
                                             edgecolors="none", linewidths=0))
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))


def describe(phys, P, F, geometry, scale):
    """The right-hand column, from what the USD authors."""
    approx = {c.get("approximation") for c in phys["colliders"]}
    if "sdf" in approx:
        collider, pieces = "collider: signed distance field of the mesh", None
    else:
        pieces = decompose(P, F)
        collider = "collider: convex decomposition, %d hulls shown (CoACD stand-in)" % len(pieces)
    m = phys["materials"][0] if phys["materials"] else {}
    val = lambda k: m.get(k)
    if phys["mass"] is not None:
        mass = "mass: %.3f kg, authored" % phys["mass"]
    elif phys["density"] is not None:
        mass = "mass: density %g kg/m3, authored" % phys["density"]
    else:
        mass = "mass: not authored, PhysX default density %d kg/m3" % PHYSX_DEFAULT_DENSITY
    if val("static_friction") is not None or val("dynamic_friction") is not None:
        fr = "friction: static %.2f, dynamic %.2f, authored" % (val("static_friction") or 0.0,
                                                               val("dynamic_friction") or 0.0)
    else:
        fr = "friction: not bound, PhysX default %.1f" % PHYSX_DEFAULT_FRICTION
    if val("restitution") is not None:
        rest = "restitution: %.2f, authored" % val("restitution")
    else:
        rest = "restitution: not bound, PhysX default %.1f" % PHYSX_DEFAULT_RESTITUTION
    comb = "combine mode: %s" % (val("friction_combine") or "PhysX default (average)")
    dims = tuple((P.max(0) - P.min(0)) * 100)
    lines = ["geometry: %s, %d faces" % (geometry, len(F)),
             "size: %.1f x %.1f x %.1f cm, %s" % (dims + (scale,)), collider, mass, fr, rest, comb]
    return lines, pieces


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("obj", help="object folder name, e.g. sm_bowl_plasticribbed_07f")
    ap.add_argument("nice", help="display name (unused in the panels, kept for file naming)")
    ap.add_argument("--a-root", required=True, help="first arm's asset directory (authored)")
    ap.add_argument("--b-root", required=True, help="second arm's asset directory (default path)")
    ap.add_argument("--a-title", default="authored reconstruction")
    ap.add_argument("--b-title", default="default reconstruction (TRELLIS + defaults)")
    ap.add_argument("--a-geometry", default="reconstructed from the photograph")
    ap.add_argument("--b-geometry", default="TRELLIS from the same photograph")
    ap.add_argument("--a-scale", default="scale from ChArUco board")
    ap.add_argument("--b-scale", default="scale typed by eye")
    ap.add_argument("--photo", help="input photograph (PNG/JPG); adds the version-B figure")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>_A.png [and <out>_B.png]")
    args = ap.parse_args()
    coacd.set_log_level("error")

    rows = []
    for root, title, geometry, scale in ((args.a_root, args.a_title, args.a_geometry, args.a_scale),
                                         (args.b_root, args.b_title, args.b_geometry, args.b_scale)):
        usd = find_usd(root, args.obj)
        P, F, uvt, tex = load(usd)
        side = os.path.join(root, args.obj, args.obj + ".json")
        by_eye = json.load(open(side)).get("size_by_eye_m") if os.path.exists(side) else None
        if by_eye is not None and scale == "scale typed by eye":
            scale = "scale typed by eye (%.2f m)" % by_eye
        lines, hulls = describe(inspect(usd), P, F, geometry, scale)
        rows.append((title, [(P, F, face_colors(F, uvt, tex))], [(P, F, "#a8a8a8")],
                     hulls or [(P, F, "#6f86d6")], lines))

    photo = None
    if args.photo:
        im = Image.open(args.photo).convert("RGB")
        w, h = im.size
        s = min(w, h)
        cx, cy = w // 2, int(h * 0.52)
        photo = im.crop((cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2))

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
                         "text.color": "#1a1a1a", "figure.dpi": 200, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "savefig.facecolor": "white"})

    def build(with_photo, out):
        widths = ([1.15] if with_photo else []) + [1, 1, 1, 1.55]
        fig = plt.figure(figsize=(2.05 * sum(widths) + 0.4, 4.4))
        gs = fig.add_gridspec(2, len(widths), width_ratios=widths, hspace=0.0, wspace=0.02)
        off = 0
        if with_photo:
            ax = fig.add_subplot(gs[:, 0]); ax.imshow(photo); ax.set_axis_off()
            ax.set_title("input photograph", fontsize=9, loc="left")
            off = 1
        for r, (title, tex, mesh, coll, lines) in enumerate(rows):
            a0 = fig.add_subplot(gs[r, off + 0], projection="3d"); draw(a0, tex)
            a1 = fig.add_subplot(gs[r, off + 1], projection="3d"); draw(a1, mesh)
            a2 = fig.add_subplot(gs[r, off + 2], projection="3d"); draw(a2, coll)
            a3 = fig.add_subplot(gs[r, off + 3]); a3.set_axis_off()
            a3.text(0.0, 0.92, title, fontsize=9.2, weight="bold", va="top", transform=a3.transAxes)
            a3.text(0.0, 0.80, "\n".join(lines), fontsize=7.6, va="top", linespacing=1.55,
                    transform=a3.transAxes, color="#2b2b2b")
            if r == 0:
                a0.set_title("textured render", fontsize=9, loc="left")
                a1.set_title("mesh (flat shaded)", fontsize=9, loc="left")
                a2.set_title("collision geometry", fontsize=9, loc="left")
                a3.set_title("physical values (read from the USD)", fontsize=9, loc="left")
        fig.savefig(out); plt.close(fig); print("wrote", out)

    build(False, args.out + "_A.png")
    if photo is not None:
        build(True, args.out + "_B.png")


if __name__ == "__main__":
    main()
