"""PolaRiS-style geometry: ONE photo (no ChArUco board) -> original TRELLIS at its shipped defaults -> GLB.

Deliberately nothing extra: TRELLIS own rembg (u2net) background removal, seed 1, default sampler params,
to_glb(simplify=0.95, texture_size=1024) exactly as microsoft/TRELLIS example.py. Writes what TRELLIS saw
(the preprocessed 518x518 RGBA) next to the output so the input can be audited.

  python gen_trellis.py --image Photo_x.HEIC --name <asset_id> --out OUT_DIR [--trellis-dir /path/to/TRELLIS]

Writes OUT_DIR/<asset_id>/{<asset_id>.glb, trellis_input_518.png, preview_gs.png, preview_mesh.png, provenance.json}.
Model weights download through Hugging Face, so set HF_HOME if you want the cache somewhere specific.
"""
import os, sys, json, argparse, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--image", required=True, help="the single photo (HEIC, JPEG or PNG)"); ap.add_argument("--name", required=True, help="asset id, used as the output folder and file name")
ap.add_argument("--out", required=True, help="output root directory"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--trellis-dir", default=os.environ.get("TRELLIS_DIR"),
                help="microsoft/TRELLIS checkout, added to sys.path (default: $TRELLIS_DIR, else trellis must be importable)")
a = ap.parse_args()

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
if a.trellis_dir: sys.path.insert(0, a.trellis_dir)
import numpy as np, imageio
from PIL import Image
import pillow_heif; pillow_heif.register_heif_opener()
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

od = os.path.join(a.out, a.name); os.makedirs(od, exist_ok=True)

img = Image.open(a.image)
try:
    from PIL import ImageOps; img = ImageOps.exif_transpose(img)   # phone photos carry rotation in EXIF
except Exception: pass
img = img.convert("RGB")
print("[in ]", a.image, img.size, flush=True)

t0 = time.time()
pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large"); pipe.cuda()
pre = pipe.preprocess_image(img); pre.save(os.path.join(od, "trellis_input_518.png"))
out = pipe.run(pre, seed=a.seed, formats=["mesh", "gaussian"], preprocess_image=False)
glb = postprocessing_utils.to_glb(out["gaussian"][0], out["mesh"][0], simplify=0.95, texture_size=1024, verbose=False)
glb.export(os.path.join(od, a.name + ".glb"))
# audit renders: 4 views of the textured gaussian + the mesh normals
vid = render_utils.render_video(out["gaussian"][0], num_frames=8)["color"]
imageio.imwrite(os.path.join(od, "preview_gs.png"), np.concatenate(vid[::2], axis=1))
vidm = render_utils.render_video(out["mesh"][0], num_frames=8)["normal"]
imageio.imwrite(os.path.join(od, "preview_mesh.png"), np.concatenate(vidm[::2], axis=1))
json.dump({"name": a.name, "input_image": os.path.basename(a.image), "pipeline": "microsoft/TRELLIS-image-large",
           "seed": a.seed, "sampler_params": "TRELLIS defaults", "background_removal": "TRELLIS built-in rembg u2net",
           "to_glb": {"simplify": 0.95, "texture_size": 1024}, "faces": int(len(glb.faces)),
           "seconds": round(time.time() - t0, 1)}, open(os.path.join(od, "provenance.json"), "w"), indent=1)
print("[out]", od, "faces", len(glb.faces), "in %.0fs" % (time.time() - t0), flush=True)
