"""pi0.5 as a socket server for the Isaac trial.

Protocol, both directions: 4-byte big-endian length, then a pickled dict.

Runs in /opt/openpi/.venv because openpi is JAX and Isaac ships its own pinned
torch, so neither can share the other's interpreter. Isaac talks to both over
loopback TCP and does not care which is behind the socket.

Camera naming: the Isaac trial sends base_view / left_wrist_view / right_wrist_view;
pi0.5's YamInputs expects top / left / right, matching the dataset's own
observation.images.* keys. Mapped here so the Isaac client is unchanged.

The config is the reconstructed yam_pi05 TrainConfig in yam_pi05.py beside this file:
the original training config is not public, but the checkpoint ships its own
assets/yam-bimanual-merged/norm_stats.json so normalisation is not guesswork.
"""
import os, sys, socket, struct, pickle, pathlib
# do not let JAX preallocate the whole card; another policy server may be resident
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
sys.path.insert(0, "/opt/openpi")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # yam_pi05.py ships beside this file
import numpy as np

CKPT = os.environ.get("CKPT", "/opt/openpi/pi05_yam")
PORT = int(os.environ.get("PORT", "5566"))
# Isaac camera name -> the key pi0.5's YamInputs expects
CAM_MAP = {"base_view": "top", "left_wrist_view": "left", "right_wrist_view": "right"}

from yam_pi05 import yam_pi05_config
from openpi.policies import policy_config
cfg = yam_pi05_config(assets_dir=str(pathlib.Path(CKPT) / "assets"))
policy = policy_config.create_trained_policy(cfg, CKPT)
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
print("[server] pi0.5 loaded | norm_stats %s | quantile %s | asset_id %s"
      % (dc.norm_stats is not None, dc.use_quantile_norm, dc.asset_id), flush=True)

def enc(a):
    a = np.ascontiguousarray(a)
    return (a.tobytes(), list(a.shape), a.dtype.str)
def dec(t):
    b, shape, ds = t
    return np.frombuffer(b, dtype=np.dtype(ds)).reshape(shape)

def recv_exact(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d: return None
        b += d
    return b
def send_obj(c, o):
    p = pickle.dumps(o, protocol=4)
    c.sendall(struct.pack(">I", len(p)) + p)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", PORT)); srv.listen(1)
print(f"[server] listening on 127.0.0.1:{PORT}", flush=True)

while True:
    conn, _ = srv.accept()
    print("[server] client connected", flush=True)
    n = 0
    try:
        while True:
            hdr = recv_exact(conn, 4)
            if hdr is None: break
            req = pickle.loads(recv_exact(conn, struct.unpack(">I", hdr)[0]))
            try:
                imgs = {}
                for k, v in req["images"].items():
                    dst = CAM_MAP.get(k, k)
                    imgs[dst] = np.ascontiguousarray(dec(v)[:, :, :3]).astype(np.uint8)
                obs = {"images": imgs,
                       "state": dec(req["state"]).astype(np.float32),
                       "prompt": req.get("task", "put the bottles in the bin")}
                a = np.asarray(policy.infer(obs)["actions"], np.float32)[:, :14]
                n += 1
                if n == 1: print("[server] first chunk", a.shape, flush=True)
                send_obj(conn, {"action": enc(a)})
            except Exception as e:
                import traceback; traceback.print_exc()
                send_obj(conn, {"error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()
        print(f"[server] client gone after {n} chunks", flush=True)
