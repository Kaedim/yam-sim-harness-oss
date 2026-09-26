"""Environment-only cut of a fitted splat (used for the default-arm scene): the filters of
cut_splat_to_environment.py, with the transform and scale read from register_to_reference.py's fit.json.

  slab cut      their tabletop contents (bin, bottles, cup...) so the policy is not shown a second set of the
                objects it manipulates; a THIN slab just above the table (see the original for why thin)
  size cap      largest sigma < 5 cm (what actually streaks)
  needle cut    largest/middle sigma < 4
  radius        nothing beyond 3.0 m of the cell
  voxel         >= 3 gaussians per 10 cm voxel
usage: CUT_NEEDLE=0 CUT_KEEP_FLAT=1 python cut_generic.py <in.ply> <fit.json> <out.ply>

CUT_NEEDLE=0 disables the needle filter (2DGS surfels are flat by construction), CUT_KEEP_FLAT=1 spares the
table-top surfels inside the slab, CUT_SLAB_Z0 moves the slab floor (default 0.750 m).
"""
import json, os, sys
import numpy as np

SRC, FIT, DST = sys.argv[1], sys.argv[2], sys.argv[3]
fit = json.load(open(FIT))
M = np.array(fit["M_rowmajor_math"]); SCALE = float(fit["scale_m_per_unit"])
ALPHA, HIMAX, NEEDLE, RMAX = 0.02, 0.05, 4.0, 3.0
# 2DGS surfels are flat discs by construction: the needle filter would drop a third of the cloud. NEEDLE=0 disables it.
NEEDLE = float(os.environ.get("CUT_NEEDLE", NEEDLE)) or 1e9; HIMAX = float(os.environ.get("CUT_HIMAX", HIMAX))
VOX, MINPTS = 0.10, 3
SLAB = dict(x=(-0.17, 0.73), y=(-0.85, 0.85), z=(float(os.environ.get("CUT_SLAB_Z0", "0.750")), 1.20))   # CUT_SLAB_Z0 raises the slab floor

with open(SRC, "rb") as fh:
    header, props, n = [], [], None
    while True:
        rl = fh.readline(); t = rl.decode("ascii", "replace").strip(); header.append(rl)
        if t.startswith("element vertex"): n = int(t.split()[-1])
        elif t.startswith("property"): props.append(t.split()[-1])
        elif t == "end_header": break
    body = np.frombuffer(fh.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))
ix = {p: i for i, p in enumerate(props)}
P = body[:, [ix["x"], ix["y"], ix["z"]]].astype(np.float64)
al = 1 / (1 + np.exp(-body[:, ix["opacity"]].astype(np.float64)))
sig = np.sort(np.exp(body[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]].astype(np.float64)), 1) * SCALE
mid, hi = sig[:, 1], sig[:, 2]
W = P @ M[:3, :3].T + M[:3, 3]
fin = np.isfinite(P).all(1); opac = al > ALPHA; size = hi < HIMAX
thin = (hi / np.maximum(mid, 1e-12)) < NEEDLE
cut = ((W[:, 0] > SLAB["x"][0]) & (W[:, 0] < SLAB["x"][1]) & (W[:, 1] > SLAB["y"][0]) & (W[:, 1] < SLAB["y"][1])
       & (W[:, 2] > SLAB["z"][0]) & (W[:, 2] < SLAB["z"][1]))
# 2DGS surfels carry normals: inside the slab, keep the TABLE TOP (near-vertical normal, within 3 cm of the top) and
# drop only the objects standing on it (their sides have horizontal normals). CUT_KEEP_FLAT=1 enables it.
if os.environ.get("CUT_KEEP_FLAT", "0") == "1" and "rot_0" in ix:
    # surfel normal = the axis of the smallest scale (scale_2 after padding), i.e. column 2 of R(q); the PLY's nx/ny/nz
    # are zeros in 2DGS exports, so build it from the quaternion (w, x, y, z)
    q = body[:, [ix["rot_0"], ix["rot_1"], ix["rot_2"], ix["rot_3"]]].astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9); w, x, y, z = q.T
    Nl = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    Nw = Nl @ M[:3, :3].T
    Nw /= np.maximum(np.linalg.norm(Nw, axis=1, keepdims=True), 1e-9)
    flat = (np.abs(Nw[:, 2]) > 0.8) & (W[:, 2] < SLAB["z"][0] + 0.03)
    print("keep-flat: %d table-top surfels spared inside the slab" % (cut & flat).sum(), flush=True)
    cut = cut & ~flat
CELL = np.array([0.42, 0.0, 0.85])
far = np.linalg.norm(W - CELL, axis=1) > RMAX
keep = fin & opac & size & thin & ~cut & ~far
idx = np.flatnonzero(keep)
vox = np.floor(W[idx] / VOX).astype(np.int64)
key = (vox[:, 0] << 42) ^ (vox[:, 1] << 21) ^ vox[:, 2]
uq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
keep2 = np.zeros(len(P), bool); keep2[idx[cnt[inv] >= MINPTS]] = True
out = body[keep2]
with open(DST, "wb") as fh:
    for rl in header:
        t = rl.decode("ascii", "replace")
        fh.write(("element vertex %d\n" % len(out)).encode() if t.startswith("element vertex") else rl)
    fh.write(np.ascontiguousarray(out, dtype="<f4").tobytes())
print("CUT_OK %d -> %d gaussians (slab removed %d = %.1f%%, size cap %d, needles %d, far %d) -> %s"
      % (n, len(out), cut.sum(), 100 * cut.mean(), (~size).sum(), (~thin).sum(), far.sum(), DST), flush=True)
if len(out) < 0.3 * n:
    print("CUT_WARN kept under 30%% of the cloud; expect smear", flush=True)
