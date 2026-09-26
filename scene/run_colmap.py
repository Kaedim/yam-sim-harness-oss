"""COLMAP through pycolmap: CPU SIFT (the pycolmap wheel has no CUDA), exhaustive matching, incremental mapping,
then undistortion. Output is laid out the way the 2DGS / 3DGS COLMAP loaders expect:

  <work>/undistorted/{images, sparse/0/{cameras,images,points3D}.bin}   with PINHOLE cameras

  python run_colmap.py IMAGE_DIR WORK_DIR [--threads 32] [--min-reg-frac 0.6] [--sift-max 2000]

Single shared camera, OPENCV model. Feature extraction is skipped if WORK_DIR/database.db already exists, so a
crashed run resumes at mapping.
"""
import argparse, os, shutil, time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("images", help="directory of *.jpg frames (from extract_frames.py)")
ap.add_argument("work", help="COLMAP work directory (created)")
ap.add_argument("--threads", type=int, default=int(os.environ.get("COLMAP_THREADS", "32")),
                help="worker threads (default 32; see the OpenBLAS note in README.md before raising it)")
ap.add_argument("--min-reg-frac", type=float, default=float(os.environ.get("MIN_REG_FRAC", "0.6")),
                help="warn if fewer than this fraction of images register (default 0.6)")
ap.add_argument("--sift-max", type=int, default=int(os.environ.get("SIFT_MAX", "2000")),
                help="SIFT max_image_size (default 2000)")
a = ap.parse_args()

import pycolmap

IMG, WORK = Path(a.images), Path(a.work); WORK.mkdir(parents=True, exist_ok=True)
NT = a.threads; MIN_FRAC = a.min_reg_frac
db, sparse, und = WORK / "database.db", WORK / "sparse", WORK / "undistorted"
n_img = len(list(IMG.glob("*.jpg"))); print("images", n_img, flush=True)

def setopt(o, name, val):
    for tgt in (o, getattr(o, "sift", None)):
        if tgt is not None and hasattr(tgt, name):
            setattr(tgt, name, val); return True
    return False

t0 = time.time()
if not db.exists():
    eo = pycolmap.FeatureExtractionOptions(); setopt(eo, "num_threads", NT)
    setopt(eo, "max_image_size", a.sift_max); setopt(eo, "max_num_features", 8192)
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "OPENCV"
    pycolmap.extract_features(db, IMG, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                              extraction_options=eo, device=pycolmap.Device.cpu)
    print("features done %.0fs" % (time.time() - t0), flush=True)
    mo = pycolmap.FeatureMatchingOptions(); setopt(mo, "num_threads", NT)
    # Exhaustive, not sequential + loop detection: pycolmap 4.2's vocab-tree auto-download aborts the process (not a
    # Python exception, so it cannot be caught). Exhaustive on ~400 frames closes loops and takes minutes to an hour on CPU.
    pycolmap.match_exhaustive(db, matching_options=mo, device=pycolmap.Device.cpu)
    print("exhaustive matching done %.0fs" % (time.time() - t0), flush=True)

def do_map(tag):
    out = WORK / ("sparse_" + tag); shutil.rmtree(out, ignore_errors=True); out.mkdir()
    opts = pycolmap.IncrementalPipelineOptions(); setopt(opts, "num_threads", NT)
    recs = pycolmap.incremental_mapping(db, IMG, out, options=opts)
    if not recs: return None, 0
    k = max(recs, key=lambda i: recs[i].num_reg_images()); r = recs[k]
    print("%s: %d models, best registers %d/%d images, %d points, mean track %.1f, mean reproj err %.3f px"
          % (tag, len(recs), r.num_reg_images(), n_img, r.num_points3D(), r.compute_mean_track_length(),
             r.compute_mean_reprojection_error()), flush=True)
    return out / str(k), r.num_reg_images()

best, nreg = do_map("exh")
if nreg < MIN_FRAC * n_img: print("WARNING only %d/%d registered" % (nreg, n_img), flush=True)
if best is None: raise SystemExit("COLMAP produced no model")
shutil.rmtree(sparse, ignore_errors=True); (sparse / "0").mkdir(parents=True)
for f in Path(best).iterdir(): shutil.copy(f, sparse / "0" / f.name)
shutil.rmtree(und, ignore_errors=True)
pycolmap.undistort_images(und, sparse / "0", IMG, output_type="COLMAP", num_threads=NT)
s0 = und / "sparse" / "0"; s0.mkdir(exist_ok=True)
for f in list((und / "sparse").glob("*.bin")) + list((und / "sparse").glob("*.txt")): shutil.move(str(f), s0 / f.name)
rec = pycolmap.Reconstruction(s0)
cam = list(rec.cameras.values())[0]
print("undistorted: %d images, camera %s %dx%d params %s" % (rec.num_reg_images(), cam.model.name, cam.width, cam.height,
      [round(p, 2) for p in cam.params]), flush=True)
print("COLMAP_DONE registered %d/%d total %.0fs" % (nreg, n_img, time.time() - t0), flush=True)
