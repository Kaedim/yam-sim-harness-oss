"""Single minimal filters on a RAW capture, one at a time. A diagnostic, not a pipeline step: use it to find which
part of a raw cloud causes a render problem (for example meshes vanishing in the splat composite) while leaving
everything else exactly as captured.

  --mode far    drop gaussians farther than 2.5 m (sim world) from the point (0.5, 0, 0.75)
  --mode big    drop gaussians whose largest axis exceeds 5 cm (sim world)

The splat -> world transform defaults to the runner's default SPLAT_XFORM (the authored-arm scene); pass --xform-file
with your own 16 numbers (USD row-major layout, same format as SPLAT_XFORM_FILE) for another capture.

  python filter_ply.py --mode far IN.ply OUT.ply [--xform-file splat_xform.txt]
"""
import argparse, os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mode", required=True, choices=["far", "big"])
ap.add_argument("src", help="input 3DGS-layout binary little-endian PLY")
ap.add_argument("dst", help="output PLY")
ap.add_argument("--xform-file", help="16 floats, USD Matrix4d row-major layout (default: the authored-arm scene transform)")
a = ap.parse_args()

import numpy as np

mode, src, dst = a.mode, a.src, a.dst
with open(src, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"): hdr += f.readline()
    names = [l.split()[2].decode() for l in hdr.split(b"\n") if l.startswith(b"property")]
    n = int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[2])
    data = np.fromfile(f, dtype=np.dtype([(nm, "<f4") for nm in names]), count=n)
xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
M = np.array([-0.079868, -0.803229, 0.007717, 0, 0.752607, -0.077538, -0.281401, 0, 0.280749, -0.020648, 0.756551, 0, -0.186649, 0.144665, 1.221677, 1]).reshape(4, 4)
SCALE = 0.8072                      # metres per splat unit for the default transform
if a.xform_file:
    M = np.array([float(x) for x in open(a.xform_file).read().split()]).reshape(4, 4)
    SCALE = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
w = xyz @ M[:3, :3] + M[3, :3]
if mode == "far":
    keep = np.linalg.norm(w - np.array([0.5, 0.0, 0.75]), axis=1) < 2.5
else:
    smax = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], 1)).max(1) * SCALE
    keep = smax <= 0.05
out = data[keep]
newhdr = hdr.replace(("element vertex %d" % n).encode(), ("element vertex %d" % len(out)).encode())
with open(dst, "wb") as f:
    f.write(newhdr); out.tofile(f)
print("MODE=%s kept %d of %d gaussians -> %s" % (mode, len(out), n, dst), flush=True)
