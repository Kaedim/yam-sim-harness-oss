#!/usr/bin/env python3
"""Bootstrap summary of a job's results.jsonl against Robocurve's real per-trial CSV for the same task.
Writes <jobdir>/summary.md and prints a one-line summary for Slack."""
import json, sys, os, csv
import numpy as np
job, task = sys.argv[1], sys.argv[2]          # task: bottles | clear-table | latte
CSV = os.environ.get("REAL_CSV", "/opt/bench/robocurve_kaedim_trial_data.csv")
TASKMAP = {"bottles": ("bottles-in-bin", 20), "clear_table": ("clear-table", 70), "clear-table": ("clear-table", 70),
           "latte": ("move-latte-cup", 10),
           # Blocks max is 30 and bowls max is 50, from Robocurve's published rubric.
           "blocks": ("stack-blocks", 30), "bowls": ("stack-bowls", 50)}
rows = [json.loads(l) for l in open(os.path.join(job, "results.jsonl")) if l.strip()]
# One trial id must count once. Two runner containers writing the same results.jsonl, or a voided-then-rerun
# trial, leave two lines for one id. Keep the LAST line per trial id, which is the re-run for a void and the
# legitimate writer for a duplicate.
_last = {}
for _r in rows: _last[_r["trial"]] = _r
rows = [_last[k] for k in sorted(_last)]
# A VOIDED trial (invalid start scene, object hidden from the camera, physically impossible lift) is not a
# score. Excluding it here is what makes the mean honest; the count is printed so a job with many voids cannot pass as a
# result (a 100% voided job must not read as "0.00").
_all = rows
rows = [r for r in _all if not r.get("voided")]
n_void = len(_all) - len(rows)
sim = np.array([r["score"] for r in rows], float)
if len(sim) == 0:
    open(os.path.join(job, "summary.md"), "w").write("# %s, %s: ALL %d TRIALS VOIDED, no result\n" % (os.path.basename(job), task, n_void))
    print("%s %s: ALL %d TRIALS VOIDED (invalid scenes / hidden objects), NO RESULT" % (os.path.basename(job), task, n_void)); sys.exit(0)
real_name, mx = TASKMAP.get(task, (task, 20))
# the rows carry the runner's own MAX_SCORE; trust that over the table above if present, so the two can never disagree
_rowmax = {r.get("max_score") for r in rows if r.get("max_score")}
if len(_rowmax) == 1:
    mx = int(_rowmax.pop())
real = np.array([float(r["report_score"]) for r in csv.DictReader(open(CSV)) if r["task"] == real_name]) if os.path.exists(CSV) else np.array([])
rng = np.random.default_rng(0); B = 20000
def ci(fn, x): 
    if len(x) < 2: return (float("nan"), float("nan"))
    v = np.array([fn(rng.choice(x, len(x))) for _ in range(B)]); return tuple(np.percentile(v, [2.5, 97.5]).round(2))
out = ["# %s, %s, n=%d sim trials%s" % (os.path.basename(job), task, len(sim), (" (%d VOIDED trials excluded)" % n_void) if n_void else ""), "",
       "| | sim | real |", "|---|---|---|",
       "| scores | %s | %s |" % (" ".join(str(int(x)) for x in sim), " ".join(str(int(x)) for x in real)),
       "| mean [95%% CI] | %.2f %s | %s |" % (sim.mean(), ci(np.mean, sim), ("%.2f %s" % (real.mean(), ci(np.mean, real))) if len(real) else "n/a"),
       "| median | %.1f | %s |" % (np.median(sim), ("%.1f" % np.median(real)) if len(real) else "n/a"),
       "| any progress (>= 5) | %d/%d | %s |" % ((sim >= 5).sum(), len(sim), ("%d/%d" % ((real >= 5).sum(), len(real))) if len(real) else "n/a"),
       "| full score | %d/%d | %s |" % ((sim >= mx).sum(), len(sim), ("%d/%d" % ((real >= mx).sum(), len(real))) if len(real) else "n/a")]
if len(real) >= 2 and len(sim) >= 2:
    d = np.array([np.mean(rng.choice(sim, len(sim))) - np.mean(rng.choice(real, len(real))) for _ in range(B)])
    out.append("| mean diff sim-real, 95%% CI | %s | |" % (tuple(np.percentile(d, [2.5, 97.5]).round(2)),))
pen = {}
for r in rows:
    for k, v in (r.get("pen_mm") or {}).items(): pen[k] = min(pen.get(k, 0.0), v)
out += ["", "worst penetration per object (mm): %s" % json.dumps({k: round(v, 2) for k, v in pen.items()})]
open(os.path.join(job, "summary.md"), "w").write("\n".join(out) + "\n")
print("%s %s: n=%d%s mean %.2f/%d CI %s median %.1f | real mean %s | any-progress %d/%d" % (
    os.path.basename(job), task, len(sim), (" (+%d VOIDED)" % n_void) if n_void else "", sim.mean(), mx, ci(np.mean, sim), np.median(sim),
    ("%.2f" % real.mean()) if len(real) else "n/a", (sim >= 5).sum(), len(sim)))
