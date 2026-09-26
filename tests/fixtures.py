"""Synthetic run directories in exactly the format the runner writes, for testing the gate.

The log lines are copied from a real reported run (K1_bot_pi05_plain); only the values the gate
reads are parameterised. Tests change one of them and check the gate names the defect.
"""
import json, os

GOOD = dict(started="2026-09-18T19:15:37Z", arm="kaedim", root="/opt/isaacwork/assets",
            splat="/opt/isaacwork/splat_env2_k080.usd", dz="-0.005", task="bottles", chunk=16,
            iters=400, real=6400, layout="v3", fingers=True, clip=True, rig="rig1", top_h=0.72)


def run_log(**kw):
    c = dict(GOOD, **kw)
    lines = ["%s [Info] [carb] Logging to file: kit.log" % c["started"],
             "[assets] arm=%s root=%s" % (c["arm"], c["root"]),
             "[budget] %s: %d chunks x %d = %d steps (real %d)"
             % (c["task"], c["iters"], c["chunk"], c["iters"] * c["chunk"], c["real"])]
    if c["dz"] is not None:
        lines.append("[splat] SPLAT_DZ %s m -> translation z 1.2167 (explicit)" % c["dz"])
    lines.append("[splat] %s referenced as /World/splat, type 'ParticleField3DGaussianSplat'" % c["splat"])
    if c["fingers"]:
        lines.append("[phys] FILTER_FINGERS on: finger<->finger collision filtered on 2 gripper(s)")
    lines.append("[layout] %s (%s, real start-frame refit): {}" % (c["layout"], c["task"]))
    lines.append("[ctrl] 30.0 Hz, absolute joint targets clipped to each joint's own limits"
                 if c["clip"] else "[ctrl] 30.0 Hz, CLIP_JOINTS=0 clip to [-pi, pi]")
    lines.append("[cam ] top camera: MEASURED %s intrinsics fx 606.64 fy 606.13" % c["rig"])
    lines.append("[cam ] top camera %.4f m above the table (TOP_H)" % c["top_h"])
    return "\n".join(lines) + "\n"


def make_run(runs_dir, run_id, scores, job_env=None, voided=0, **kw):
    d = os.path.join(runs_dir, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run.log"), "w") as fh:
        fh.write(run_log(**kw))
    with open(os.path.join(d, "results.jsonl"), "w") as fh:
        for i, s in enumerate(scores):
            fh.write(json.dumps(dict(trial=i + 1, score=s, voided=False)) + "\n")
        for i in range(voided):
            fh.write(json.dumps(dict(trial=len(scores) + i + 1, score=0, voided=True)) + "\n")
    if job_env is not None:
        with open(os.path.join(d, "job.json"), "w") as fh:
            json.dump(dict(id=run_id, env=job_env), fh)
    return d


MOLMO = dict(chunk=30, iters=213, top_h=0.88, rig="rig6")   # bottles, MolmoAct2
DEFAULT_ARM = dict(arm="polaris", root="/opt/isaacwork/assets_polaris",
                   splat="/opt/isaacwork/polaris_splat/scene/polaris_env_final.usd", dz="-0.011")


def write_real(path, cells):
    """cells: {(task label, policy label): [scores]} in the evaluator's CSV format"""
    with open(path, "w") as fh:
        fh.write("task,policy,run,score,max\n")
        for (task, policy), scores in cells.items():
            for i, s in enumerate(scores):
                fh.write("%s,%s,%d,%s,20\n" % (task, policy, i + 1, s))
