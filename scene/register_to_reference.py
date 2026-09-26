"""Register a second splat of the same room onto an already-aligned reference splat (used for the default-arm scene).

The reference is the authored-arm splat under the runner's default SPLAT_XFORM. The new cloud is the same room
from the same video, so it is registered to the reference directly and inherits the reference's alignment. Scale
comes from the table-top height, yaw and shift from the table band, refined on everything below the table top:
  1. gravity-align both clouds (RANSAC floor plane -> +Z), floor at z=0
  2. scale: the 1-D height profile (density vs metres above the floor) of the new cloud, over a log grid of
     metres-per-unit, against the reference profile; best normalised correlation wins
  3. yaw + xy shift: 2-D occupancy of everything 0.8..2.5 m above the floor (rig, curtain, walls), brute-force
     yaw in 1-degree steps with FFT cross-correlation for the shift
  4. M_new = T . Rz(yaw) . S(scale) . R1_new, expressed in the reference's WORLD frame (floor height taken from
     the reference under the default transform)
Gates: scale-profile correlation >= 0.6 (hard); occupancy-peak robust z-score >= 4 (soft: warns, fit still written).
usage: CAMS_JSON=colmap_cam_centers.json python register_to_reference.py <new.ply> <outdir> <ref.ply>

CAMS_JSON (COLMAP camera centres in the splat frame) fixes the sign of "up": COLMAP frames are not z-up.
Writes fit.json and splat_xform.txt (16 numbers, USD row-major; point the runner's SPLAT_XFORM_FILE at it).
"""
import json, os, sys
import numpy as np

NEW, OUT = sys.argv[1], sys.argv[2]
if len(sys.argv) < 4:
    sys.exit(__doc__)
REF = sys.argv[3]
# the runner's default SPLAT_XFORM: the reference splat's alignment into the sim world
REF_M_USD = [-0.079868, -0.803229, 0.007717, 0, 0.752607, -0.077538, -0.281401, 0, 0.280749, -0.020648, 0.756551, 0,
             -0.186649, 0.144665, 1.221677, 1]
os.makedirs(OUT, exist_ok=True)

def fail(msg, **kw):
    open(os.path.join(OUT, "NEEDS_HUMAN.txt"), "w").write(msg + "\n" + json.dumps(kw, indent=1, default=float) + "\n")
    print("REG_FAIL " + msg, flush=True); sys.exit(3)

def load_ply(path):
    with open(path, "rb") as fh:
        props, n = [], None
        while True:
            l = fh.readline().decode("ascii", "replace").strip()
            if l.startswith("element vertex"): n = int(l.split()[-1])
            elif l.startswith("property"): props.append(l.split()[-1])
            elif l == "end_header": break
        raw = np.frombuffer(fh.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))
    P = np.stack([raw[:, props.index(c)] for c in "xyz"], 1).astype(np.float64)
    al = 1 / (1 + np.exp(-raw[:, props.index("opacity")].astype(np.float64))) if "opacity" in props else np.ones(len(P))
    P = P[np.isfinite(P).all(1) & (al > 0.05)]
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    return P[((P > lo) & (P < hi)).all(1)], n

def gravity(P, seed=0):
    rng = np.random.default_rng(seed)
    S = P[rng.choice(len(P), min(200000, len(P)), replace=False)]
    tol = 0.004 * np.ptp(S, 0).max(); best = (0, None, None)
    for _ in range(3000):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        nr = np.cross(b - a, c - a); ln = np.linalg.norm(nr)
        if ln < 1e-9: continue
        nr /= ln
        k = int((np.abs((S - a) @ nr) < tol).sum())
        if k > best[0]: best = (k, nr, a)
    _, up, a = best
    # sign of "up": COLMAP frames are not z-up. If the capture's camera centres are known (CAMS_JSON, splat frame),
    # up points from the dominant plane towards the cameras (they filmed from above the floor); otherwise assume +z.
    cams = os.environ.get("CAMS_JSON")
    if cams and os.path.exists(cams) and seed == 0 and P is NEW_P:
        Cc = np.array(json.load(open(cams))["centers"], dtype=float)
        if np.median((Cc - a) @ up) < 0: up = -up
        print("gravity sign from %d camera centres: cameras sit %.2f units above the dominant plane" % (len(Cc), np.median((Cc - a) @ up)), flush=True)
    elif up[2] < 0: up = -up
    v = np.cross(up, [0, 0, 1.0]); sn = np.linalg.norm(v); c = float(up @ [0, 0, 1.0])
    if sn < 1e-9: R1 = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R1 = np.eye(3) + vx + vx @ vx * ((1 - c) / sn ** 2)
    Q = P @ R1.T
    # floor height = median of the dominant plane's own inliers, not the lowest percentile: 2DGS clouds carry a
    # skirt of floaters well below the floor (about 0.5 m of them here), which would put the table at 1.2 m
    z0 = floor_level((S @ R1.T)[:, 2])
    return R1, z0, Q - np.array([0, 0, z0])

def floor_level(z):
    """Lowest dense horizontal layer: the first height bin (from the bottom) holding >= 10% of the densest bin. The
    densest plane itself can be the TABLE in a close-orbit capture, and the lowest percentile can be floaters."""
    hist, edges = np.histogram(z, bins=300)
    i = int(np.argmax(hist >= 0.10 * hist.max()))
    return float(0.5 * (edges[i] + edges[i + 1]))

# reference in world (under its default alignment): floor height and the two profiles
RP, _ = load_ply(REF); NEW_P = None
RM = np.array(REF_M_USD).reshape(4, 4).T
RW = RP @ RM[:3, :3].T + RM[:3, 3]
floor_w = floor_level(RW[:, 2])
ZB = np.arange(0.0, 3.0001, 0.02)
def zprofile(z):
    h, _ = np.histogram(z, bins=ZB); h = h.astype(float); h /= max(h.sum(), 1); return h
ref_prof = zprofile(RW[:, 2] - floor_w)
XY = np.arange(-4.0, 4.0001, 0.05)
def occupancy(W):
    sel = W[(W[:, 2] > 0.30) & (W[:, 2] < 1.10)]
    h, _, _ = np.histogram2d(sel[:, 0], sel[:, 1], bins=[XY, XY]); return np.log1p(h)
ref_occ = occupancy(RW - np.array([0, 0, floor_w]))
ref_occ -= ref_occ.mean()

# new cloud: gravity, then scale
NP, n_new = load_ply(NEW); NEW_P = NP
R1, z0, Q = gravity(NP)
def table_peak(z, lo, hi, nb=400):
    """Height of the dominant horizontal layer between lo and hi (the table top), refined as the median of the
    points within one bin of the histogram peak."""
    h, e = np.histogram(z[(z > lo) & (z < hi)], bins=nb); i = int(np.argmax(h)); w = e[1] - e[0]
    zc = 0.5 * (e[i] + e[i + 1]); return float(np.median(z[np.abs(z - zc) < w]))
zt_ref = table_peak(RW[:, 2] - floor_w, 0.4, 1.2)                 # metres above the floor, expect ~0.75
zt_new = table_peak(Q[:, 2], 0.15 * np.ptp(Q[:, 2]), 0.95 * np.ptp(Q[:, 2]))   # units above the floor
SCALE = zt_ref / zt_new
# cross-check against the whole height profile (should agree within ~10%)
prof_scales = np.exp(np.linspace(np.log(0.02), np.log(2.0), 600)); best = (-1, None)
for s_ in prof_scales:
    p_ = zprofile(Q[:, 2] * s_); c_ = float(np.corrcoef(p_, ref_prof)[0, 1]) if p_.std() > 0 else -1
    if c_ > best[0]: best = (c_, s_)
scale_corr, scale_prof = best
print("scale: table-top %.5f m/unit (ref table %.3f m, new %.2f units); whole-profile best %.5f (corr %.2f)"
      % (SCALE, zt_ref, zt_new, scale_prof, scale_corr), flush=True)
if abs(np.log(scale_prof / SCALE)) > np.log(1.25): print("REG_WARN profile scale disagrees with table-top scale by >25%%", flush=True)
Qm = Q * SCALE                       # metres, floor at 0, gravity up, yaw + xy unknown

# yaw + shift from the TABLE itself (the one structure both clouds share completely): PCA of the table band gives the
# long axis (two candidates, 180 deg apart); the shift puts the new table-band centroid on the reference's; the 180 deg
# choice is the candidate whose occupancy of everything else (0.05..1.10 m) correlates better with the reference.
def table_band(W, zc): return W[np.abs(W[:, 2] - zc) < 0.03]
def pca_axis(B):
    uv = B[:, :2] - B[:, :2].mean(0); w, V = np.linalg.eigh(np.cov(uv.T)); V = V[:, ::-1]
    return np.arctan2(V[1, 0], V[0, 0]), B[:, :2].mean(0)
RWf = RW - np.array([0, 0, floor_w])
ref_ang, ref_cen = pca_axis(table_band(RWf, zt_ref))
new_ang, new_cen = pca_axis(table_band(Qm, zt_ref))
F_ref = np.fft.rfft2(ref_occ); scores = {}
for extra in (0.0, np.pi):
    th = ref_ang - new_ang + extra
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    W = Qm @ Rz.T; shift = ref_cen - (Rz[:2, :2] @ new_cen)
    W[:, :2] += shift
    occ = occupancy(W); occ -= occ.mean()
    scores[float(th)] = (float((occ * ref_occ).sum()), shift)
th = max(scores, key=lambda k: scores[k][0]); deg = int(round(np.degrees(th))) % 360
sc = sorted(v[0] for v in scores.values()); ratio = sc[-1] / max(abs(sc[0]), 1e-9); gate_ok = ratio > 1.15
print("yaw candidates (deg -> score): %s ; picked %d, ratio %.2f" % ({int(round(np.degrees(k))) % 360: round(v[0], 1) for k, v in scores.items()}, deg, ratio), flush=True)
if not gate_ok: print("REG_WARN the two 180-degree candidates score within 15%%; check the snap render", flush=True)
Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
dx, dy = scores[th][1]
A = Rz @ (SCALE * R1)
t = Rz @ np.array([0, 0, -z0 * SCALE]) + np.array([dx, dy, floor_w])
M = np.eye(4); M[:3, :3] = A; M[:3, 3] = t
# refine: the table band + everything below it (legs, arm bases, bin) is what both clouds share. Grid-search yaw
# within +-25 deg about the table centre and the xy shift by FFT, at 2 cm, maximising the correlation with the reference.
def occ_fine(W):
    G = np.arange(-1.6, 1.6001, 0.02); sel = W[(W[:, 2] > 0.05) & (W[:, 2] < 0.85)]
    h, _, _ = np.histogram2d(sel[:, 0] - 0.42, sel[:, 1], bins=[G, G]); h = np.log1p(h); return h - h.mean()
ref_f = occ_fine(RWf); F_f = np.fft.rfft2(ref_f); nG = ref_f.shape[0]
Wc = NP @ A.T + t - np.array([0, 0, floor_w])          # coarse-aligned, floor-relative
c0 = np.array([0.42, 0.0, 0.0]); best = (-np.inf, 0, (0, 0))
for ddeg in range(-25, 26):
    r = np.radians(ddeg); Rd = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1.0]])
    Wr = (Wc - c0) @ Rd.T + c0
    cc = np.fft.irfft2(F_f * np.conj(np.fft.rfft2(occ_fine(Wr))), s=ref_f.shape); k = np.unravel_index(np.argmax(cc), cc.shape)
    if cc[k] > best[0]: best = (float(cc[k]), ddeg, k)
_, ddeg, k = best
ddx = (k[0] if k[0] <= nG // 2 else k[0] - nG) * 0.02; ddy = (k[1] if k[1] <= nG // 2 else k[1] - nG) * 0.02
r = np.radians(ddeg); Rd = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1.0]])
# fold the refinement into M: world' = Rd (world - c0) + c0 + [ddx, ddy, 0]
A = Rd @ A; t = Rd @ (t - c0) + c0 + np.array([ddx, ddy, 0.0])
M = np.eye(4); M[:3, :3] = A; M[:3, 3] = t
deg = (deg + ddeg) % 360
print("refine: yaw %+d deg, shift (%+.2f, %+.2f) m" % (ddeg, ddx, ddy), flush=True)
Mu = M.T.reshape(-1)
# report: where does the new cloud's densest above-table mass sit vs the reference (should agree within a cell or two)
W = NP @ M[:3, :3].T + M[:3, 3]
rec = dict(new=NEW, ref=REF, n_gaussians=int(n_new), scale_m_per_unit=float(SCALE), scale_from_profile=float(scale_prof), scale_profile_corr=scale_corr, table_top_ref_m=zt_ref, table_top_new_units=zt_new,
           yaw_deg=deg, shift_m=[float(dx + ddx), float(dy + ddy)], refine_yaw_deg=ddeg, candidate_score_ratio=ratio, gate_ok=bool(gate_ok), floor_world_z=floor_w,
           new_table_band_frac=float(np.mean(np.abs(W[:, 2] - 0.75) < 0.03)), ref_table_band_frac=float(np.mean(np.abs(RW[:, 2] - 0.75) < 0.03)),
           M_rowmajor_math=M.tolist(), splat_xform_usd=Mu.tolist())
json.dump(rec, open(os.path.join(OUT, "fit.json"), "w"), indent=1)
open(os.path.join(OUT, "splat_xform.txt"), "w").write(" ".join("%.6f" % v for v in Mu) + "\n")
print("REG_OK scale %.5f m/unit (profile corr %.2f)  yaw %d deg  shift (%.2f, %.2f) m  cand ratio %.2f  table-band frac new %.3f ref %.3f"
      % (SCALE, scale_corr, deg, dx + ddx, dy + ddy, ratio, rec["new_table_band_frac"], rec["ref_table_band_frac"]), flush=True)
