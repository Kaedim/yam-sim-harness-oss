"""Apply a 4x4 uniform-scale rigid transform to every gaussian in a 3DGS-layout PLY: positions, normals, rotation
quaternions and log-scales.

Two uses:
  - put a cloud into the sim world frame (bake M_fit), so it can be cut with `cut_splat_to_environment.py
    --matrix identity` and loaded with an identity SPLAT_XFORM;
  - pre-compensate for a runner transform you cannot change: baking M_runner^-1 . M_fit means the runner's own
    SPLAT_XFORM (M_runner) lands the cloud where M_fit puts it. Bake M_fit, then bake M_runner with --inverse.

  python bake_transform_into_ply.py IN.ply OUT.ply "16 floats"            # row-major MATH layout (rows of M)
  python bake_transform_into_ply.py IN.ply OUT.ply "16 floats" --usd      # USD Matrix4d layout (translation in 12,13,14)
  python bake_transform_into_ply.py IN.ply OUT.ply "16 floats" --inverse  # apply M^-1 instead of M
"""
import argparse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="input 3DGS-layout PLY")
ap.add_argument("dst", help="output PLY")
ap.add_argument("matrix", help="16 space-separated floats, quoted as one argument")
ap.add_argument("--usd", action="store_true", help="matrix is in USD Matrix4d layout (transpose of the math layout)")
ap.add_argument("--inverse", action="store_true", help="apply the inverse of the given matrix")
args = ap.parse_args()

import numpy as np
from plyfile import PlyData, PlyElement

inv, usd = args.inverse, args.usd
SRC, DST, MS = args.src, args.dst, [float(x) for x in args.matrix.split()]
M = np.array(MS).reshape(4, 4); M = M.T if usd else M
if inv: M = np.linalg.inv(M)
A = M[:3, :3]; t = M[:3, 3]; s = np.cbrt(abs(np.linalg.det(A))); R = A / s
assert np.allclose(R @ R.T, np.eye(3), atol=1e-4), "not a uniform-scale rigid transform"
def quat_from_R(R):  # w,x,y,z
    tr = np.trace(R)
    if tr > 0: S = np.sqrt(tr + 1) * 2; return np.array([0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])
    i = np.argmax(np.diag(R)); j, k = (i + 1) % 3, (i + 2) % 3
    S = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / S; q[i + 1] = 0.25 * S; q[j + 1] = (R[j, i] + R[i, j]) / S; q[k + 1] = (R[k, i] + R[i, k]) / S
    return q
def qmul(a, b):  # (N,4) wxyz
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)
ply = PlyData.read(SRC); v = ply["vertex"]; d = v.data.copy()
P = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64); P2 = P @ A.T + t
d["x"], d["y"], d["z"] = P2[:, 0], P2[:, 1], P2[:, 2]
if "nx" in d.dtype.names:
    N = np.stack([d["nx"], d["ny"], d["nz"]], 1).astype(np.float64) @ R.T; d["nx"], d["ny"], d["nz"] = N.T
qR = quat_from_R(R); Q = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], 1).astype(np.float64)
Q2 = qmul(np.tile(qR, (len(Q), 1)), Q); Q2 /= np.linalg.norm(Q2, axis=1, keepdims=True)
for i in range(4): d["rot_%d" % i] = Q2[:, i]
for k in ("scale_0", "scale_1", "scale_2"): d[k] = d[k] + np.log(s)
PlyData([PlyElement.describe(d, "vertex")], text=False, byte_order="<").write(DST)
print("baked scale %.5f, rotation quat %s, translation %s into %d gaussians -> %s" % (s, np.round(qR, 4), np.round(t, 4), len(d), DST))
