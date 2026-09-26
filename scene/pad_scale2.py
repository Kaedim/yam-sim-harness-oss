"""2DGS surfel PLY (scale_0, scale_1) -> 3DGS-layout PLY (scale_0..2), so omni.kit.converter.gsplat and Isaac's
SPG renderer accept it.

The third axis (the surfel normal, column 2 of R(q)) gets a small constant log-scale (--log-thick, default -12,
i.e. exp(-12) ~ 6e-6 splat units), so each disc stays a disc. Everything else is copied verbatim.

  python pad_scale2.py IN_2DGS.ply OUT_3DGS_LAYOUT.ply [--log-thick -12.0]
"""
import argparse, os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="2DGS point_cloud.ply (has scale_0, scale_1, no scale_2)")
ap.add_argument("dst", help="output PLY in 3DGS layout")
ap.add_argument("--log-thick", type=float, default=float(os.environ.get("SURFEL_LOG_THICK", "-12.0")),
                help="log-scale written to scale_2 (default -12.0)")
a = ap.parse_args()

import numpy as np
from plyfile import PlyData, PlyElement

SRC, DST, THICK = a.src, a.dst, a.log_thick
v = PlyData.read(SRC)["vertex"]; names = [p.name for p in v.properties]
assert "scale_1" in names and "scale_2" not in names, names
order = ["x", "y", "z", "nx", "ny", "nz"] + [n for n in names if n.startswith("f_dc_")] + \
        sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1])) + \
        ["opacity", "scale_0", "scale_1", "scale_2"] + [n for n in names if n.startswith("rot_")]
n = v.count; out = np.empty(n, dtype=[(k, "<f4") for k in order])
for k in order:
    out[k] = v[k] if k != "scale_2" else np.full(n, THICK, "<f4")
if "nx" not in names:
    for k in ("nx", "ny", "nz"): out[k] = 0
PlyData([PlyElement.describe(out, "vertex")], text=False, byte_order="<").write(DST)
s0, s1 = np.exp(v["scale_0"]), np.exp(v["scale_1"]); al = 1 / (1 + np.exp(-v["opacity"]))
print("wrote %s: %d gaussians, %d props; median disc radii %.4f/%.4f units, median opacity %.3f, %d with opacity>0.5"
      % (DST, n, len(order), np.median(s0), np.median(s1), np.median(al), (al > 0.5).sum()))
