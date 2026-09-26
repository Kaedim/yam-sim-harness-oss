"""Video -> sharp JPEG frames for COLMAP.

Dumps --fps frames per second, then keeps the sharpest frame (highest Laplacian variance) of every
--keep-group consecutive frames. A 4:24 video at 3 fps with group 2 gives ~395 frames. ffmpeg comes from
imageio_ffmpeg, so no system ffmpeg is needed.

  python extract_frames.py VIDEO OUT_DIR [--fps 3] [--keep-group 2] [--max-width 1920]

All frames are first written to OUT_DIR_all/; the kept ones are hard-linked into OUT_DIR/ as 00000.jpg, ...
"""
import argparse, os, glob, subprocess

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("video", help="input video (any format ffmpeg reads, e.g. a phone .MOV)")
ap.add_argument("out", help="output directory for the kept frames")
ap.add_argument("--fps", default=os.environ.get("FRAME_FPS", "3"), help="frames dumped per second (default 3)")
ap.add_argument("--keep-group", type=int, default=int(os.environ.get("KEEP_GROUP", "2")),
                help="keep the sharpest of every N dumped frames (default 2)")
ap.add_argument("--max-width", default=os.environ.get("MAX_W", "1920"), help="downscale wider frames to this width")
a = ap.parse_args()

import cv2, imageio_ffmpeg

VID, OUT, FPS, GROUP, MAXW = a.video, a.out, a.fps, a.keep_group, a.max_width
tmp = OUT.rstrip("/") + "_all"; os.makedirs(tmp, exist_ok=True); os.makedirs(OUT, exist_ok=True)
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.check_call([ff, "-y", "-loglevel", "error", "-i", VID,
                       "-vf", f"fps={FPS},scale='min({MAXW},iw)':-2,format=yuv420p", "-q:v", "2", f"{tmp}/f_%05d.jpg"])
files = sorted(glob.glob(f"{tmp}/f_*.jpg")); kept = 0
for i in range(0, len(files), GROUP):
    grp = files[i:i + GROUP]
    best = max(grp, key=lambda f: cv2.Laplacian(cv2.imread(f, 0), cv2.CV_64F).var())
    os.link(best, f"{OUT}/{kept:05d}.jpg"); kept += 1
h, w = cv2.imread(files[0]).shape[:2]
print(f"frames dumped {len(files)} kept {kept} size {w}x{h}", flush=True)
