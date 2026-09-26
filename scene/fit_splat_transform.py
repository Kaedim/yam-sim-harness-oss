"""Splat -> sim-world 4x4 from the table plateau at a KNOWN height (used for the authored-arm scene).

The authored-arm capture's plateau was located once by hand at --plateau-z 0.934 splat units above the floor
(aspect 1.87 against the true 2.00). For a new capture, run fit_transform_auto.py instead, or read a plateau
height off its fit_report.json candidates ("zc_units") and pass it here.

Method: RANSAC the dominant plane for gravity and rotate it to +Z; zero the floor at the 0.5th height percentile;
take the band within --half-band of the plateau; PCA of the band puts the long axis on sim +Y; scale =
--table-long / measured long edge; translate the band centre onto --table-centre. The 180 deg PCA yaw ambiguity
is not resolved here: if the room renders mirrored, add 180 deg of yaw (or use fit_transform_auto.py's B candidate).

Prints the scale, the 4x4 in row-major math layout, the same matrix in USD Matrix4d layout, and where the table
centre lands. On the authored-arm capture this gives 0.8072 m per unit; the fit's two independent checks were
short edge 0.848 m against a true 0.80, and table height above the floor 0.754 m against a measured 0.750.

  python fit_splat_transform.py SPLAT.ply [--plateau-z 0.934] [--half-band 0.03] [--table-long 1.60] [--table-centre 0.42 0 0.750]
"""
import argparse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="3DGS-layout binary little-endian PLY in the splat's own frame")
ap.add_argument("--plateau-z", type=float, default=0.934, help="table plateau height above the floor, splat units (default 0.934)")
ap.add_argument("--half-band", type=float, default=0.03, help="half-thickness of the plateau band, splat units (default 0.03)")
ap.add_argument("--table-long", type=float, default=1.60, help="true table long edge in metres, placed on sim +Y (default 1.60)")
ap.add_argument("--table-centre", type=float, nargs=3, default=[0.42, 0.0, 0.750], metavar=("X", "Y", "Z"),
                help="sim-world position of the table-top centre in metres (default 0.42 0 0.750)")
args = ap.parse_args()

import numpy as np

PLY = args.src
with open(PLY, "rb") as fh:
    props, n = [], None
    while True:
        l = fh.readline().decode("ascii", "replace").strip()
        if l.startswith("element vertex"): n = int(l.split()[-1])
        elif l.startswith("property"): props.append(l.split()[-1])
        elif l == "end_header": break
    raw = np.frombuffer(fh.read(n*len(props)*4), dtype="<f4").reshape(n, len(props))
P = np.stack([raw[:, props.index(c)] for c in "xyz"], 1).astype(np.float64)
al = 1/(1+np.exp(-raw[:, props.index("opacity")].astype(np.float64)))
P = P[np.isfinite(P).all(1) & (al > 0.05)]
lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
P = P[((P > lo) & (P < hi)).all(1)]

rng = np.random.default_rng(0)
S = P[rng.choice(len(P), min(200000, len(P)), replace=False)]
tol = 0.004*np.ptp(S, 0).max(); best = (0, None, None)
for _ in range(3000):
    a, b, c = S[rng.choice(len(S), 3, replace=False)]
    nr = np.cross(b-a, c-a); ln = np.linalg.norm(nr)
    if ln < 1e-9: continue
    nr /= ln
    k = int((np.abs((S-a) @ nr) < tol).sum())
    if k > best[0]: best = (k, nr, a)
_, up, _ = best
if up[2] < 0: up = -up
v = np.cross(up, [0, 0, 1.0]); s = np.linalg.norm(v); c = float(up @ [0, 0, 1.0])
vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
R1 = np.eye(3) + vx + vx @ vx * ((1-c)/s**2)
Q = P @ R1.T
z0 = np.percentile(Q[:, 2], 0.5); Q[:, 2] -= z0

# the table plateau, in floor-zeroed splat units
zc, halfband = args.plateau_z, args.half_band
band = Q[np.abs(Q[:, 2] - zc) < halfband]
uv = band[:, :2] - band[:, :2].mean(0)
w, V = np.linalg.eigh(np.cov(uv.T)); V = V[:, ::-1]
ax = uv @ V
L = np.percentile(ax[:, 0], 98) - np.percentile(ax[:, 0], 2)
SCALE = args.table_long / L
yaw = np.arctan2(V[1, 0], V[0, 0])
# put the long axis on sim +Y (the sim table is short edge in x, long edge in y)
dz = np.pi/2 - yaw
R2 = np.array([[np.cos(dz), -np.sin(dz), 0], [np.sin(dz), np.cos(dz), 0], [0, 0, 1.0]])
R = R2 @ R1
M = np.eye(4); M[:3, :3] = R * SCALE
# translation: band centre (with the floor-zeroing folded back in) onto the sim table-top centre
M[:3, 3] = np.array(args.table_centre) - (R * SCALE) @ (np.append(band[:,:2].mean(0), zc + z0))

print("scale            %.6f m per unit" % SCALE)
print("gravity rotation R1 =\n%s" % np.round(R1, 6))
print("yaw about Z      %.3f deg" % np.degrees(dz))
print("\n4x4 (row-major, splat -> our world):")
for r in M:
    print("  [%12.6f %12.6f %12.6f %12.6f]" % tuple(r))
print("\nUSD matrix4d (column-major, as USD wants it):")
Mu = M.T
print("( (%.6f, %.6f, %.6f, %.6f)," % tuple(Mu[0]))
print("  (%.6f, %.6f, %.6f, %.6f)," % tuple(Mu[1]))
print("  (%.6f, %.6f, %.6f, %.6f)," % tuple(Mu[2]))
print("  (%.6f, %.6f, %.6f, %.6f) )" % tuple(Mu[3]))
# sanity: where does the table centre land?
chk = (M[:3, :3] @ np.append(band[:, :2].mean(0), zc + z0)) + M[:3, 3]
print("\ncheck: table centre maps to %s   (want %s)" % (np.round(chk, 4), args.table_centre))
