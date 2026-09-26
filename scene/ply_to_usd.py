"""3DGS-layout PLY -> USD (ParticleField3DGaussianSplat), with Isaac Sim's own gaussian-splat converter, Z up.

The converter ships as the omni.kit.converter.gsplat extension, which is not on python.sh's import path, but the
converter itself is a plain package under the extension's pip_prebundle. This script puts that on sys.path and
calls its main. Run it with Isaac Sim's python, inside the Isaac Sim 6.0.1 container:

  /isaac-sim/python.sh ply_to_usd.py IN.ply OUT.usd [--name Environment] [--isaac-root /isaac-sim]
"""
import argparse, glob, os, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="3DGS-layout PLY (for 2DGS, pad it first with pad_scale2.py)")
ap.add_argument("dst", help="output .usd")
ap.add_argument("--name", default="Environment", help="name of the splat prim in the output USD")
ap.add_argument("--isaac-root", default=os.environ.get("ISAAC_SIM_ROOT"),
                help="Isaac Sim install root to search for the converter (default: $ISAAC_SIM_ROOT, then /isaac-sim, "
                     "the container path, then /opt/IsaacSim)")
a = ap.parse_args()

roots = [r for r in (a.isaac_root, "/isaac-sim", "/opt/IsaacSim") if r]
D = [p for r in roots for p in glob.glob(os.path.join(r, "extscache", "omni.kit.converter.gsplat-*", "pip_prebundle"))]
if not D:
    raise SystemExit("gsplat converter pip_prebundle not found under %s" % roots)
sys.path.insert(0, D[0])
print("using", D[0], flush=True)

# The converter imports pxr. In the container, bare python.sh has no pxr on its path until a Kit app has started
# (the trial runner gets it from SimulationApp), so boot a headless app if needed.
_app = None
try:
    import pxr  # noqa: F401
except ModuleNotFoundError:
    from isaacsim import SimulationApp
    _app = SimulationApp({"headless": True})
    print("headless Kit started for pxr", flush=True)

from usd_convert_gsplat.cli import main
sys.argv = ["usd_convert_gsplat", "-i", a.src, "-o", a.dst, "-n", a.name, "--up-axis", "Z"]
main()
print("CONVDONE", flush=True)
if _app is not None:
    _app.close()
