"""MolmoAct2-BimanualYAM behind the trial's own wire protocol.

The Isaac trial talks one protocol to every policy: 4-byte big-endian length, then a
pickled dict, both directions, as pi05_server.py speaks it.
MolmoAct2 instead ships an HTTP server (examples/yam/host_server_yam.py, json_numpy on
/act). Rather than teach the trial a second protocol, this bridges: it speaks the trial's
protocol on PORT and forwards to MolmoAct2's HTTP server on MOLMO_URL, so the Isaac side
is unchanged and the policies stay directly comparable.

Camera naming, same substitution pi05_server.py makes for the same three physical cameras:
    base_view -> top_cam,  left_wrist_view -> left_cam,  right_wrist_view -> right_cam

Start MolmoAct2's server first (it needs its own venv and GPU; see
ops/run_job.sh for the invocation), then this, then the trial with PORT pointing here.
"""
import os, socket, struct, pickle, sys
import numpy as np
import requests
import json_numpy
json_numpy.patch()

PORT = int(os.environ.get("PORT", "5577"))
MOLMO_URL = os.environ.get("MOLMO_URL", "http://127.0.0.1:8202/act")
NUM_STEPS = int(os.environ.get("MOLMO_NUM_STEPS", "10"))
CAM_MAP = {"base_view": "top_cam", "left_wrist_view": "left_cam",
           "right_wrist_view": "right_cam"}

# The trial does not put ndarrays on the wire: enc() sends every array as a 3-tuple
# (raw bytes, shape list, dtype str) and dec() rebuilds it, so the bridge has to speak
# that on both sides -- decode what arrives, and encode the actions we send back, or the
# trial's dec(r["action"]) fails.
def enc(a):
    a = np.ascontiguousarray(a)
    return (a.tobytes(), list(a.shape), a.dtype.str)

def dec(t):
    b, sh, ds = t
    return np.frombuffer(b, dtype=np.dtype(ds)).reshape(sh)

def recvall(conn, n):
    b = b""
    while len(b) < n:
        c = conn.recv(n - len(b))
        if not c:
            raise ConnectionError("peer closed")
        b += c
    return b

# fail fast and loudly if MolmoAct2 is not up, rather than at the first trial step
try:
    h = requests.get(MOLMO_URL, timeout=10)
    print("[bridge] MolmoAct2 health: %s %s" % (h.status_code, h.text[:120]), flush=True)
except Exception as e:
    print("[bridge] MolmoAct2 NOT reachable at %s: %r" % (MOLMO_URL, e), flush=True)
    sys.exit(2)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", PORT)); srv.listen(1)
print("[bridge] listening on 127.0.0.1:%d -> %s" % (PORT, MOLMO_URL), flush=True)

while True:
    conn, _ = srv.accept()
    print("[bridge] trial connected", flush=True)
    try:
        while True:
            n = struct.unpack(">I", recvall(conn, 4))[0]
            req = pickle.loads(recvall(conn, n))
            body = {"instruction": req.get("task", ""),
                    "state": dec(req["state"]).astype(np.float32),
                    "num_steps": NUM_STEPS}
            for iname, mname in CAM_MAP.items():
                if iname in req["images"]:
                    body[mname] = dec(req["images"][iname]).astype(np.uint8)
            try:
                r = requests.post(MOLMO_URL, json=body, timeout=120)
                r.raise_for_status()
                out = r.json()
                act = np.asarray(out["actions"], np.float32)
                if act.ndim == 1:
                    act = act[None, :]
                rep = {"action": enc(act)}
            except Exception as e:
                print("[bridge] inference failed: %r" % (e,), flush=True)
                rep = {"error": repr(e)}
            p = pickle.dumps(rep)
            conn.sendall(struct.pack(">I", len(p)) + p)
    except (ConnectionError, struct.error, EOFError) as e:
        print("[bridge] trial disconnected (%s)" % type(e).__name__, flush=True)
    finally:
        conn.close()
