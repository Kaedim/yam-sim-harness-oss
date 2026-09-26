"""Cut a scene splat down to the environment: remove the captured tabletop contents and the gaussians that streak.

The capture contains the evaluator's own table contents (bin, bottles, cup, plate) and arms. Rendered whole, the
policy sees a second copy of everything it manipulates. The output PLY stays in the splat's OWN frame (the runner
applies the transform), so the transform given here is used only to decide what to cut.

Filters, applied in the sim world frame:

  slab cut             tabletop contents. A THIN SLAB over the sim table footprint, only as tall as what sits on
                       the table (default x -0.17..0.73, y -0.85..0.85, z 0.750..1.20 m: the sim table spans
                       x -0.12..0.68, y -0.80..0.80 with its top at 0.750, plus 5 cm margins; 0.45 m of height
                       clears a 0.23 m bottle and the bin). Size it to YOUR table.
  opacity > 0.02       deliberately permissive, see DO NOT FILTER ON OPACITY
  largest sigma < 5 cm size cap: large gaussians are what streak
  largest/middle < 4   needles
  r < 3.0 m of cell    the sparse tail, nothing renderable in it
  >= 3 per 10 cm voxel isolated floaters

WHY THE SLAB IS THIN. Rendering the raw cloud and a filtered cut from identical poses with the same transform and
renderer showed the raw cloud clean and the first cut (a column with no upper bound, removing 52.1% of the cloud
including the backdrop behind the table) as smear. A cut that large exposes volume the capture never observed from
any angle, and unobserved volume renders as smear. The thin slab still removes about a third of the cloud because
the tabletop is where the capture is densest, so the trial plugs the hole with the authored table, kept visible.

DO NOT FILTER ON OPACITY. A size-aware opacity ramp (al > 0.015 + 8*hi) turned the table white: served top view mean
84.8 -> 130.3 against the real top camera's 90.4. A 3DGS surface IS a stack of overlapping low-opacity gaussians,
so culling translucent ones removes the surface and lets the background through.

What streaks is SIZE. On the authored-arm capture (796,079 gaussians, 0.8072 m per unit): 9,648 have sigma > 5 cm,
with 2-sigma spans up to 1.24 m at 0.25 mean opacity; 7,209 are needles; 36.1% are zero-thickness discs with
near-isotropic normals, so no viewpoint avoids seeing some edge-on and moving the camera is not the lever.

  python cut_splat_to_environment.py SRC.ply DST.ply                              # authored-arm transform (default)
  python cut_splat_to_environment.py SRC.ply DST.ply --matrix-file fit/M_math_A.txt
  python cut_splat_to_environment.py SRC_in_world.ply DST.ply --matrix identity   # cloud already baked into the sim frame
"""
import argparse, os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="3DGS-layout binary little-endian PLY, in the splat's own frame")
ap.add_argument("dst", help="output PLY (same frame as the input)")
g = ap.add_mutually_exclusive_group()
g.add_argument("--matrix", help='splat -> sim-world transform: 16 floats, row-major MATH layout (rows of M), quoted; '
                                'or "identity". Default: the authored-arm scene transform.')
g.add_argument("--matrix-file", help="text file holding the same 16 floats (e.g. fit_transform_auto.py's M_math_A.txt)")
ap.add_argument("--slab-x", type=float, nargs=2, default=[-0.17, 0.73], metavar=("MIN", "MAX"), help="slab cut x range, m")
ap.add_argument("--slab-y", type=float, nargs=2, default=[-0.85, 0.85], metavar=("MIN", "MAX"), help="slab cut y range, m")
ap.add_argument("--slab-z", type=float, nargs=2, default=[0.750, 1.20], metavar=("MIN", "MAX"), help="slab cut z range, m")
ap.add_argument("--cell-centre", type=float, nargs=3, default=[0.42, 0.0, 0.85], metavar=("X", "Y", "Z"),
                help="centre for the far-tail radius cut, m (default 0.42 0 0.85)")
args = ap.parse_args()

import numpy as np

SRC, DST = args.src, args.dst
# Default transform: the authored-arm scene. The table-plateau fit (fit_splat_transform.py) got the scale right but
# the placement wrong in two places, corrected by hand-aligning in the Isaac GUI:
#   z 1.327877 -> 1.221677 (-0.1062) puts the captured table top at the sim's 0.750
#   x 0.243351 -> -0.186649 (-0.43)  brings the captured table's near edge to x -0.10, so the arm mounts sit on it
# Rotation and scale are the fit's. This is the runner's default SPLAT_XFORM, transposed to math layout.
M = np.array([[-0.079868, 0.752607, 0.280749, -0.186649],
              [-0.803229,-0.077538,-0.020648,  0.144665],
              [ 0.007717,-0.281401, 0.756551,  1.221677],
              [ 0.0,0.0,0.0,1.0]])
SCALE = 0.8072                      # metres per splat unit, from fit_splat_transform.py
_mat = args.matrix
if args.matrix_file:
    _mat = open(args.matrix_file).read()
if _mat:
    M = np.eye(4) if _mat.strip() == "identity" else np.array([float(x) for x in _mat.split()]).reshape(4, 4)
    SCALE = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    print("matrix override: scale %.4f m/unit" % SCALE)
ALPHA, HIMAX, NEEDLE, RMAX = 0.02, 0.05, 4.0, 3.0
VOX, MINPTS = 0.10, 3
SLAB = dict(x=tuple(args.slab_x), y=tuple(args.slab_y), z=tuple(args.slab_z))

with open(SRC, "rb") as fh:
    header, props, n = [], [], None
    while True:
        rl = fh.readline(); t = rl.decode("ascii", "replace").strip(); header.append(rl)
        if t.startswith("element vertex"): n = int(t.split()[-1])
        elif t.startswith("property"): props.append(t.split()[-1])
        elif t == "end_header": break
    body = np.frombuffer(fh.read(n*len(props)*4), dtype="<f4").reshape(n, len(props))
ix = {p: i for i, p in enumerate(props)}
P = body[:, [ix["x"], ix["y"], ix["z"]]].astype(np.float64)
al = 1/(1+np.exp(-body[:, ix["opacity"]].astype(np.float64)))
sig = np.sort(np.exp(body[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]].astype(np.float64)), 1)
sig *= SCALE                        # smallest, middle, largest sigma, in metres
mid, hi = sig[:, 1], sig[:, 2]
W = P @ M[:3, :3].T + M[:3, 3]

fin  = np.isfinite(P).all(1)
opac = al > ALPHA
size = hi < HIMAX
thin = (hi / np.maximum(mid, 1e-12)) < NEEDLE
cut  = ((W[:, 0] > SLAB["x"][0]) & (W[:, 0] < SLAB["x"][1])
        & (W[:, 1] > SLAB["y"][0]) & (W[:, 1] < SLAB["y"][1])
        & (W[:, 2] > SLAB["z"][0]) & (W[:, 2] < SLAB["z"][1]))
CELL = np.array(args.cell_centre)
rad  = np.linalg.norm(W - CELL, axis=1)
far  = rad > RMAX
keep = fin & opac & size & thin & ~cut & ~far
print("%d total" % n)
print("  opacity > %.2f keeps      %7d" % (ALPHA, opac.sum()))
print("  size cap hi<%.2f m drops  %7d" % (HIMAX, (~size).sum()))
print("  needle cut hi/mid>%.0f drops %6d" % (NEEDLE, (~thin).sum()))
print("  slab cut drops           %7d (%.1f%%)   beyond %.1f m of the cell drops %d"
      % (cut.sum(), 100*cut.mean(), RMAX, far.sum()))
print("  all of the above keep    %7d" % keep.sum())

idx = np.flatnonzero(keep)
vox = np.floor(W[idx]/VOX).astype(np.int64)
key = (vox[:, 0] << 42) ^ (vox[:, 1] << 21) ^ vox[:, 2]
uq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
keep2 = np.zeros(len(P), bool); keep2[idx[cnt[inv] >= MINPTS]] = True
print("  >=%d per %.2f m voxel      %7d  <- final" % (MINPTS, VOX, keep2.sum()))

Wk = W[keep2]
r = np.linalg.norm(Wk - CELL, axis=1)
print("kept extent (sim frame): x %+.2f..%+.2f  y %+.2f..%+.2f  z %+.2f..%+.2f"
      % (Wk[:, 0].min(), Wk[:, 0].max(), Wk[:, 1].min(), Wk[:, 1].max(),
         Wk[:, 2].min(), Wk[:, 2].max()))
print("distance from the cell centre: %.1f%% within 1.5 m, p50 %.2f m, p99 %.2f m."
      % (100*(r < 1.5).mean(), np.percentile(r, 50), np.percentile(r, 99)))
print("largest surviving sigma %.4f m, needles %d, beyond %.1f m %d"
      % (hi[keep2].max(), (hi[keep2]/np.maximum(mid[keep2], 1e-12) > NEEDLE).sum(), RMAX, (r > RMAX).sum()))

out = body[keep2]
with open(DST, "wb") as fh:
    for rl in header:
        t = rl.decode("ascii", "replace")
        fh.write(("element vertex %d\n" % len(out)).encode()
                 if t.startswith("element vertex") else rl)
    fh.write(np.ascontiguousarray(out, dtype="<f4").tobytes())
print("wrote %s (%d splats, %.1f MB)" % (DST, len(out), os.path.getsize(DST)/1e6))
