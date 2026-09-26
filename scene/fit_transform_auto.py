"""Automatic splat -> sim-world transform from the table plateau (used for the default-arm scene).

Same method as fit_splat_transform.py, but the table plateau height is found automatically instead of hardcoded:

  1. gravity: RANSAC the dominant plane, rotate its normal to +Z
  2. floor: 0.5th height percentile; candidate plateaus = local maxima of the height histogram above the floor band
  3. pick the plateau whose footprint looks like the table (short edge within 25% of --table-short at the implied
     scale, aspect closest to long/short); scale = --table-long / measured long edge
  4. yaw: PCA of the plateau band puts the long axis on sim +Y. PCA cannot tell +Y from -Y, so BOTH candidates
     A and B (180 deg apart) are written.

Outputs in OUT_DIR:
  fit_report.json          all candidates, both transforms, and an automatic A/B pick (see below)
  splat_xform_{A,B}.txt    16 floats, USD Matrix4d row-major layout -> the runner's SPLAT_XFORM_FILE
  M_math_{A,B}.txt         16 floats, row-major MATH layout (rows of M) -> cut_splat_to_environment.py --matrix-file
  plateau_topdown.png      top-down density of the plateau band, to eyeball the table outline

The automatic A/B pick is a heuristic calibrated on ONE room: the candidate with more above-table mass in the far
strip behind the table (x 0.82..1.54 m: lights, camera stand, curtain rig) wins. It is not derived from geometry.
Always confirm the pick by rendering: with the wrong candidate the room is mirrored (the capture's rig lands in
front of the table instead of behind it).

  python fit_transform_auto.py SPLAT.ply OUT_DIR [--table-long 1.60] [--table-short 0.80] [--table-centre 0.42 0 0.750]
"""
import argparse, os, json

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="3DGS-layout PLY in the splat's own frame (for 2DGS, the output of pad_scale2.py)")
ap.add_argument("out", help="output directory (created)")
ap.add_argument("--table-long", type=float, default=1.60, help="true table long edge in metres, placed on sim +Y (default 1.60)")
ap.add_argument("--table-short", type=float, default=0.80, help="true table short edge in metres, on sim X (default 0.80)")
ap.add_argument("--table-centre", type=float, nargs=3, default=[0.42, 0.0, 0.750], metavar=("X", "Y", "Z"),
                help="sim-world position of the table-top centre in metres (default 0.42 0 0.750)")
a = ap.parse_args()

import numpy as np, cv2
from plyfile import PlyData

SRC, OUT = a.src, a.out; os.makedirs(OUT, exist_ok=True)
TL, TS, CTR = a.table_long, a.table_short, np.array(a.table_centre)
v = PlyData.read(SRC)["vertex"]
P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
al = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
P = P[np.isfinite(P).all(1) & (al > 0.05)]
lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0); P = P[((P > lo) & (P < hi)).all(1)]
rng = np.random.default_rng(0); S = P[rng.choice(len(P), min(200000, len(P)), replace=False)]
tol = 0.004 * np.ptp(S, 0).max(); best = (0, None)
for _ in range(3000):
    a_, b, c = S[rng.choice(len(S), 3, replace=False)]; nr = np.cross(b - a_, c - a_); ln = np.linalg.norm(nr)
    if ln < 1e-9: continue
    nr /= ln; k = int((np.abs((S - a_) @ nr) < tol).sum())
    if k > best[0]: best = (k, nr)
up = best[1]
if up[2] < 0: up = -up
vv = np.cross(up, [0, 0, 1.0]); s = np.linalg.norm(vv); c = float(up @ [0, 0, 1.0])
vx = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
R1 = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2) if s > 1e-9 else np.eye(3)
Q = P @ R1.T; z0 = np.percentile(Q[:, 2], 0.5); Q[:, 2] -= z0
H = np.ptp(Q[:, 2]); nb = 200; hist, edges = np.histogram(Q[:, 2], bins=nb); zc_all = 0.5 * (edges[1:] + edges[:-1])
# candidate plateaus: local maxima of the height histogram above the floor band
cands = []
for i in range(3, nb - 3):
    if hist[i] == hist[i - 3:i + 4].max() and zc_all[i] > 0.15 * H and hist[i] > 0.2 * hist[3:].max():
        cands.append(i)
report = {"gravity_tilt_deg": float(np.degrees(np.arccos(np.clip(c, -1, 1)))), "floor_z0_units": float(z0),
          "height_range_units": float(H), "candidates": []}
def fit_plateau(zc):
    band = Q[np.abs(Q[:, 2] - zc) < 0.032 * zc]
    uv = band[:, :2] - band[:, :2].mean(0); w, V = np.linalg.eigh(np.cov(uv.T)); V = V[:, ::-1]; ax = uv @ V
    L = np.percentile(ax[:, 0], 98) - np.percentile(ax[:, 0], 2); W = np.percentile(ax[:, 1], 98) - np.percentile(ax[:, 1], 2)
    return band, V, L, W
for i in cands:
    band, V, L, W = fit_plateau(zc_all[i]); scale = TL / L
    report["candidates"].append({"zc_units": float(zc_all[i]), "count": int(hist[i]), "long_units": float(L), "short_units": float(W),
        "aspect": float(L / W), "scale_m_per_unit": float(scale), "short_edge_m_at_scale": float(W * scale),
        "table_height_m_at_scale": float(zc_all[i] * scale)})
# pick: aspect closest to long/short with the short edge within 25% of its true length; else the highest-count candidate
ok = [cd for cd in report["candidates"] if abs(cd["short_edge_m_at_scale"] - TS) < 0.25 * TS]
pick = min(ok, key=lambda cd: abs(cd["aspect"] - TL / TS)) if ok else (max(report["candidates"], key=lambda cd: cd["count"]) if report["candidates"] else None)
if pick is None: json.dump(report, open(f"{OUT}/fit_report.json", "w"), indent=1); raise SystemExit("no plateau found")
band, V, L, W = fit_plateau(pick["zc_units"]); SCALE = TL / L; yaw = np.arctan2(V[1, 0], V[0, 0])
ctr_units = np.append(band[:, :2].mean(0), pick["zc_units"] + z0)
report["picked"] = pick; report["transforms"] = {}
for tag, dz in (("A", np.pi / 2 - yaw), ("B", np.pi / 2 - yaw + np.pi)):
    R2 = np.array([[np.cos(dz), -np.sin(dz), 0], [np.sin(dz), np.cos(dz), 0], [0, 0, 1.0]]); R = R2 @ R1
    M = np.eye(4); M[:3, :3] = R * SCALE; M[:3, 3] = CTR - (R * SCALE) @ ctr_units
    chk = M[:3, :3] @ ctr_units + M[:3, 3]
    usd = M.T.reshape(-1).tolist()   # row-major USD Matrix4d layout == 16 floats for SPLAT_XFORM
    report["transforms"][tag] = {"yaw_deg": float(np.degrees(dz)), "M_rowmajor_math": M.tolist(), "SPLAT_XFORM_16": " ".join("%.6f" % x for x in usd),
                                 "check_table_centre": [round(float(x), 4) for x in chk]}
# top-down density of the plateau band, for a human to see the table outline and where the rig is
img = np.zeros((600, 600), np.float32); uv = (band[:, :2] - band[:, :2].mean(0)) * SCALE
ij = np.clip(((uv + 1.5) / 3.0 * 599).astype(int), 0, 599); np.add.at(img, (599 - ij[:, 1], ij[:, 0]), 1)
cv2.imwrite(f"{OUT}/plateau_topdown.png", np.clip(img / max(img.max(), 1) * 4 * 255, 0, 255).astype(np.uint8))
# Resolve the 180 deg yaw ambiguity EMPIRICALLY. Calibrated on the authored-arm capture, where the correct answer is
# known (candidate A there reproduces fit_splat_transform.py to 0.001 in every matrix entry): with the correct candidate
# the above-table mass in the far strip (x 0.82..1.54: lights, camera stand, curtain rig) is 1.64x the near strip
# (x -0.70..0.02). The strip bounds are specific to that room. NOT derived from geometry: check by rendering.
Ps = P[rng.choice(len(P), min(400000, len(P)), replace=False)]
for tag in ("A", "B"):
    Mm = np.array(report["transforms"][tag]["M_rowmajor_math"]); Wp = Ps @ Mm[:3, :3].T + Mm[:3, 3]
    slab = Wp[(Wp[:, 2] > 0.76) & (Wp[:, 2] < 1.40) & (np.abs(Wp[:, 1]) < 0.9)]
    near = int(((slab[:, 0] > -0.70) & (slab[:, 0] < 0.02)).sum()); far = int(((slab[:, 0] > 0.82) & (slab[:, 0] < 1.54)).sum())
    report["transforms"][tag]["rig_side_counts"] = {"near_x<0.02": near, "far_x>0.82": far}
ratioA = report["transforms"]["A"]["rig_side_counts"]["near_x<0.02"] / max(1, report["transforms"]["A"]["rig_side_counts"]["far_x>0.82"])
report["auto_pick"] = "A" if ratioA <= 1.0 else "B"; report["auto_pick_confidence_ratio"] = float(max(ratioA, 1 / max(ratioA, 1e-9)))   # correct candidate has MORE far-strip mass, see note above
json.dump(report, open(f"{OUT}/fit_report.json", "w"), indent=1)
for tag in ("A", "B"):
    with open(f"{OUT}/splat_xform_{tag}.txt", "w") as fh: fh.write(report["transforms"][tag]["SPLAT_XFORM_16"] + "\n")
    with open(f"{OUT}/M_math_{tag}.txt", "w") as fh:
        fh.write(" ".join("%.6f" % x for x in np.array(report["transforms"][tag]["M_rowmajor_math"]).reshape(-1)) + "\n")
print(json.dumps({k: report[k] for k in ("gravity_tilt_deg", "picked", "auto_pick", "auto_pick_confidence_ratio")}, indent=1))
print("rig counts A", report["transforms"]["A"]["rig_side_counts"], "B", report["transforms"]["B"]["rig_side_counts"])
print("transform A SPLAT_XFORM=%s" % report["transforms"]["A"]["SPLAT_XFORM_16"])
print("transform B SPLAT_XFORM=%s" % report["transforms"]["B"]["SPLAT_XFORM_16"])
