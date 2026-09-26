#!/usr/bin/env python3
"""Build the 10-cell sim-vs-real table from run artefacts, with every gate applied in code.

The point of this script is that the paper's numbers and the open-sourced harness produce the
same table from the same files, and neither depends on anyone having remembered a run id or a
config correctly. Nothing here is transcribed: every property is re-read from the run's own
`run.log`, which is what the runner printed while the trial was executing, not from the job file
that was submitted. A job file records what was asked for; run.log records what happened.

Usage (this is how the reported table is built):
    python build_table.py --runs RUNS_DIR --real real_reference_pertrial.csv \
        --runs-list benchmark_runs.txt --json-out table.json

RUNS_DIR contains one directory per run, each holding:
    results.jsonl   one JSON object per trial, with `score` and `voided`
    run.log         the runner's stdout
    job.json        the queue job file (only needed for MolmoAct2 runs, for the solver step
                    count, which the runner does not print -- see SOLVER_STEPS_NOTE below)

Reporting protocol: `--runs-list` names one batch per cell-column, the most recent batch of
twenty trials that passed the gate, so no selection rule is applied when the reported table is
built. Without a run list every directory under RUNS_DIR is a candidate and `--select` decides:
`n20` (default) takes the first 20 trials of the largest single-configuration family, `pool`
uses every trial of that family, `latest` takes the most recent qualifying batch at n=20. All
three order runs by the absolute UTC timestamp Isaac Kit writes on the first line of each
run.log, which is intrinsic to the run; file mtime is NOT used, because copying artefacts
through S3 rewrites it to the upload time.

The consistency test is the 95% CI on (sim mean - real mean), both sides resampled, containing
zero. It is the only test the paper uses. Whether the sim point estimate falls inside real's CI
is not computed: that check treats a 20-trial real estimate as exact.

Exit status is non-zero if any cell has no admissible run, so this can gate a release.
"""

import argparse, csv, json, os, random, re, sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# The frozen configuration. A run is admissible for a column only if its own log
# shows all of these. Anything else is reported in the rejection table with a reason.
# ---------------------------------------------------------------------------

# The asset arms live in configs/arms.json: per arm, the ASSET_ARM name, the asset root folder
# name, the splat file and SPLAT_DZ as the runner prints it ("%+.3f", so -0.0105 m reads -0.011).
# Everything below is a property of the robot cell and is the same for every arm.
ARMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "arms.json")


def load_arms(path=ARMS_FILE):
    with open(path) as fh:
        arms = json.load(fh)
    return {k: v for k, v in arms.items() if not k.startswith("_")}

# Robocurve's per-task step budgets, counted from their own actions.jsonl (not their results
# page, which says 3600 for everything and disagrees with the trajectory files).
REAL_STEPS = {"bottles": 6400, "latte": 3600, "clear_table": 12800, "bowls": 3600, "blocks": 3600}
STEP_TOL = 20          # chunking means we land within a chunk of the real budget, never exactly on it

# MolmoAct2 replans every 30 steps, pi0.5 every 16. The chunk is unambiguous in the log, so the
# policy is inferred from it rather than trusted from the job name.
CHUNK_TO_POLICY = {30: "molmo", 16: "pi05"}

# SOLVER_STEPS_NOTE: MOLMO_NUM_STEPS is the flow-matching denoising count, and 10 is the real rig
# default (molmoact2 config.py). It is NOT the action chunk -- those were conflated until 14 Sep
# 2026. The runner never prints it because the bridge owns it, so it can only be read from the job
# file. Runs that used 30 are a different policy configuration and are excluded.
REQUIRED_MOLMO_NUM_STEPS = "10"

# Top-camera height above the table, per policy. Robocurve evaluate MolmoAct2 from 0.88 m and
# pi0.5 from 0.72 m; run_job.sh:39 sets it from the policy. isaac_trials.py:772 refuses a molmo
# run at 0.72 but has no guard the other way, which is how six PolaRiS pi0.5 runs rendered at
# 0.88. Read from the runner's own "[cam ] top camera ... (TOP_H)" line, never from the job file.
REQUIRED_TOP_H = {"molmo": 0.88, "pi05": 0.72}
TOP_H_TOL = 1e-3

N_REQUIRED = 20        # trials per cell, after voids are dropped
BOOTSTRAP = 20000
BOUNDARY = 0.10        # rubric points; a gap interval ending this close to zero is flagged

TASKS = ["bottles", "bowls", "blocks", "latte", "clear_table"]
POLICIES = ["molmo", "pi05"]

# Robocurve's task names in the real per-trial CSV.
REAL_TASK = {"Bottles in bins": "bottles", "Stack bowls": "bowls", "Stack blocks": "blocks",
             "Move latte cup": "latte", "Clear table": "clear_table"}
REAL_POLICY = {"MolmoAct2": "molmo", "pi0.5": "pi05"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_run_log(path):
    """Everything we gate on, taken from the runner's own output."""
    out = dict(arm=None, root=None, splat=None, dz=None, finger_filter=False,
               task=None, chunk=None, steps=None, real_steps=None,
               layout=None, layout_key=None, rig=None, top_h=None, started=None,
               clip_joints=None)
    with open(path, errors="ignore") as fh:
        for ln in fh:
            # Isaac Kit stamps an absolute UTC time on its first log line. That is intrinsic to
            # the run, so ordering on it reproduces anywhere. File mtime does not: copying the
            # artefacts through S3 rewrites it to the upload time.
            if out["started"] is None:
                m = re.match(r"(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ)", ln)
                if m:
                    out["started"] = m.group(1)
            if ln.startswith("[assets]"):
                m = re.search(r"arm=(\S+)\s+root=(\S+)", ln)
                if m:
                    out["arm"], out["root"] = m.group(1), m.group(2)
            elif ln.startswith("[budget]"):
                m = re.search(r"(\w+): (\d+) chunks x (\d+) = (\d+) steps \(real (\d+)\)", ln)
                if m:
                    out["task"] = m.group(1)
                    out["chunk"] = int(m.group(3))
                    out["steps"] = int(m.group(4))
                    out["real_steps"] = int(m.group(5))
            elif ln.startswith("[splat] SPLAT_DZ"):
                m = re.search(r"SPLAT_DZ\s+([-+][\d.]+)", ln)
                if m:
                    out["dz"] = m.group(1)
            elif "[splat]" in ln and "referenced as" in ln:
                out["splat"] = os.path.basename(ln.split()[1])
            elif "FILTER_FINGERS on" in ln:
                out["finger_filter"] = True
            elif ln.startswith("[layout]") and re.search(r"\[layout\] (v\d)", ln):
                m = re.search(r"\[layout\] (v\d) \(([^,)]+)", ln)
                if m:
                    out["layout"], out["layout_key"] = m.group(1), m.group(2)
            elif ln.startswith("[ctrl]"):
                # CLIP_JOINTS=0 clips only to [-pi, pi] instead of each joint's own range. The
                # runner's own comment calls it an A/B; Robocurve clip to the joint range, so a
                # run with it off is an ablation and not the benchmark configuration.
                out["clip_joints"] = "CLIP_JOINTS=0" not in ln
            elif "MEASURED rig" in ln:
                m = re.search(r"MEASURED (rig\d+)", ln)
                if m:
                    out["rig"] = m.group(1)
            elif "above the table (TOP_H)" in ln:
                m = re.search(r"([\d.]+) m above the table", ln)
                if m:
                    out["top_h"] = float(m.group(1))
    return out


def parse_results(path):
    scores, voided = [], 0
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("voided"):
                voided += 1
                continue
            scores.append(float(d["score"]))
    return scores, voided


def classify(cfg, arms):
    """Which arm this run belongs to, or None if it is a mixed configuration.

    A run only counts for an arm if the arm name, the asset root AND the splat all agree. Runs
    that put one arm's assets in the other's scene are the single easiest way to get a wrong
    number here, so they are rejected rather than silently attributed.
    """
    root = os.path.basename((cfg["root"] or "").rstrip("/"))
    for name, spec in arms.items():
        if cfg["arm"] == name and root == spec["root_name"] and cfg["splat"] == spec["splat"]:
            return name
    return None


def admit(run_dir, cfg, scores, voided, arms):
    """Return a list of reasons this run is NOT admissible. Empty list means it is."""
    bad = []
    column = classify(cfg, arms)
    if column is None:
        bad.append("mixed config arm=%s root=%s splat=%s"
                   % (cfg["arm"], os.path.basename(cfg["root"] or ""), cfg["splat"]))
        return bad, None, None

    spec = arms[column]

    # The 11 Sep gripper fix. Without it the jaws stall at -0.022 instead of closing to 0, which
    # silently lowers every grasping score.
    if not cfg["finger_filter"]:
        bad.append("finger filter off")

    # The splat sits 10.5 mm high if SPLAT_DZ was never applied. Runs predating the ASSET_ARM
    # default print no SPLAT_DZ line at all.
    if cfg["dz"] != spec["dz"]:
        bad.append("splat dz %s, expected %s" % (cfg["dz"] or "never applied", spec["dz"]))

    # Short-budget runs give the policy less time than the real rig had. This is invisible in the
    # score and was the defect behind N20_P_bottles_molmo (3600 of 6400 steps).
    if cfg["steps"] is None:
        bad.append("no budget line")
    else:
        want = REAL_STEPS.get(cfg["task"])
        if want is None:
            bad.append("unknown task %s" % cfg["task"])
        elif abs(cfg["steps"] - want) > STEP_TOL:
            bad.append("budget %d of %d steps" % (cfg["steps"], want))

    policy = CHUNK_TO_POLICY.get(cfg["chunk"])
    if policy is None:
        bad.append("unrecognised chunk %s" % cfg["chunk"])

    if cfg["layout"] != "v3":
        bad.append("layout %s" % (cfg["layout"] or "unknown"))

    if cfg["clip_joints"] is False:
        bad.append("CLIP_JOINTS=0, joint targets clipped to [-pi,pi] not the joint range")

    if policy is not None:
        want_h = REQUIRED_TOP_H[policy]
        if cfg["top_h"] is None:
            bad.append("no TOP_H line")
        elif abs(cfg["top_h"] - want_h) > TOP_H_TOL:
            bad.append("TOP_H=%.4f, %s is evaluated at %.2f" % (cfg["top_h"], policy, want_h))

    if policy == "molmo":
        job = os.path.join(run_dir, "job.json")
        if not os.path.exists(job):
            bad.append("molmo run with no job.json, cannot verify MOLMO_NUM_STEPS")
        else:
            with open(job) as fh:
                env = json.load(fh).get("env", {})
            ns = str(env.get("MOLMO_NUM_STEPS", REQUIRED_MOLMO_NUM_STEPS))
            if ns != REQUIRED_MOLMO_NUM_STEPS:
                bad.append("MOLMO_NUM_STEPS=%s, expected %s" % (ns, REQUIRED_MOLMO_NUM_STEPS))

    # n is deliberately NOT checked here. A run that passes every config gate but has fewer than
    # N_REQUIRED trials is a shard of a sharded run, not a defect: bowls MolmoAct2 was run as
    # RB_BOWLS_a/_b/_c, three identical n=10 jobs. Shards of one config are pooled below in run-id
    # order. Rejecting them here would drop a real cell.
    return bad, column, policy


def signature(cfg):
    """The config properties that must match before two shards may be pooled."""
    return (cfg["finger_filter"], cfg["dz"], cfg["steps"], cfg["layout"],
            cfg["layout_key"], cfg["chunk"], cfg["rig"], cfg["top_h"], cfg["clip_joints"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values, rng, resamples=BOOTSTRAP, alpha=0.05):
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    lo = means[int((alpha / 2) * resamples)]
    hi = means[int((1 - alpha / 2) * resamples) - 1]
    return lo, hi


def gap_ci(sim, real, rng, resamples=BOOTSTRAP, alpha=0.05):
    """Bootstrap CI on (sim mean - real mean), resampling both sides.

    This is the test the cell actually needs. "Is the sim point estimate inside real's CI" treats
    real as exact and the sim as a point, when both are 20-trial estimates of a noisy discrete
    score. If this interval contains zero, the cell is statistically consistent with the real rig,
    which is the claim worth making.
    """
    ns, nr = len(sim), len(real)
    d = sorted(sum(rng.choice(sim) for _ in range(ns)) / ns
               - sum(rng.choice(real) for _ in range(nr)) / nr for _ in range(resamples))
    return d[int((alpha / 2) * resamples)], d[int((1 - alpha / 2) * resamples) - 1]


def p_closer(a, b, target, rng, resamples=BOOTSTRAP):
    """Probability that column A lands closer to the real mean than column B on a re-draw.

    A point estimate of "we are closer" is not a result when the per-trial scores are discrete and
    n is 20. This is the number that says whether a closer-than call would survive another run.
    """
    na, nb = len(a), len(b)
    wins = 0
    for _ in range(resamples):
        ma = sum(rng.choice(a) for _ in range(na)) / na
        mb = sum(rng.choice(b) for _ in range(nb)) / nb
        if abs(ma - target) < abs(mb - target):
            wins += 1
    return wins / resamples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_real(path):
    real = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            task = REAL_TASK.get(row["task"])
            policy = REAL_POLICY.get(row["policy"])
            if task and policy:
                real[(task, policy)].append(float(row["score"]))
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of run directories")
    ap.add_argument("--real", required=True, help="real_reference_pertrial.csv")
    ap.add_argument("--select", choices=["n20", "pool", "latest"], default="n20",
                    help="how a cell picks among admissible runs when --runs-list is not given. "
                         "'n20' (default) takes the first N_REQUIRED trials of the largest "
                         "single-config family, matching real's n=20. 'pool' uses every trial of "
                         "that family, at the cost of unequal n. 'latest' takes the most recent "
                         "qualifying batch at n=20. All three order runs by the start time in the "
                         "run's own log, so they reproduce off-box. The reported table is built "
                         "with --runs-list, which names one batch per cell-column, so no rule "
                         "is applied.")
    ap.add_argument("--runs-list", help="file naming the runs the benchmark uses, one per line, "
                    "'+' joining the shards of one batch, '#' for comments. Discovery is skipped "
                    "and only these runs are considered, so the reported table does not depend on "
                    "what happens to be sitting in --runs. Without it, every directory under "
                    "--runs is a candidate, which is fine for exploring and wrong for reporting.")
    ap.add_argument("--arms", default=ARMS_FILE, help="arm definitions (default configs/arms.json)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="the two arms to compare, A first (default: the first two in --arms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", help="also write the table as JSON")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    arms = load_arms(args.arms)
    A, B = args.compare or list(arms)[:2]
    for a in (A, B):
        if a not in arms:
            sys.exit("arm %r is not defined in %s" % (a, args.arms))
    LA, LB = arms[A].get("label", A), arms[B].get("label", B)
    real = load_real(args.real)

    admitted = defaultdict(list)   # (task, policy, column) -> [(run_id, scores, started, sig, voided)]
    rejected = []

    manifest = None
    if args.runs_list:
        manifest = []
        with open(args.runs_list) as fh:
            lines = fh.readlines()
        for ln in lines:
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            manifest.extend(p.strip() for p in ln.split("+"))
        missing = [r for r in manifest if not os.path.isdir(os.path.join(args.runs, r))]
        if missing:
            sys.exit("runs-list names %d run(s) not present under --runs: %s"
                     % (len(missing), ", ".join(missing)))

    for run_id in sorted(manifest if manifest is not None else os.listdir(args.runs)):
        d = os.path.join(args.runs, run_id)
        if not os.path.isdir(d):
            continue
        log, res = os.path.join(d, "run.log"), os.path.join(d, "results.jsonl")
        # A run with scores but no log is the dangerous case: it looks like data and cannot be
        # verified. Say so rather than skipping it quietly.
        if not os.path.exists(res):
            continue
        if not os.path.exists(log):
            rejected.append((run_id, "?", "results present but run.log missing, cannot verify"))
            continue
        cfg = parse_run_log(log)
        scores, voided = parse_results(res)
        reasons, column, policy = admit(d, cfg, scores, voided, arms)
        if reasons:
            rejected.append((run_id, cfg.get("task"), "; ".join(reasons)))
            continue
        # order key: the run's own recorded start time, falling back to run id
        admitted[(cfg["task"], policy, column)].append(
            (run_id, scores, cfg["started"] or "", signature(cfg), voided))

    # Resolve each cell-column to entries of exactly N_REQUIRED trials. A single run that already
    # has enough is used as-is; otherwise shards sharing one config signature are pooled in run-id
    # order, which is deterministic and does not depend on which shard scored better.
    resolved = defaultdict(list)
    for key, runs in admitted.items():
        if args.select in ("pool", "n20"):
            # Every trial of the largest single-config family. No run is chosen and none is
            # discarded, so there is nothing to select and nothing to defend.
            by_sig = defaultdict(list)
            for run_id, scores, mt, sig, voided in runs:
                by_sig[sig].append((run_id, scores, mt))
            fam = max(by_sig.values(), key=lambda s: sum(len(x[1]) for x in s))
            fam.sort(key=lambda r: (r[2], r[0]))   # chronological, by the log's own timestamp
            pooled = [v for _, s, _ in fam for v in s]
            if args.select == "n20":
                pooled = pooled[:N_REQUIRED]
                used = []
                taken = 0
                for rid, sc, _ in fam:
                    if taken >= N_REQUIRED:
                        break
                    used.append(rid)
                    taken += len(sc)
                fam_label = " + ".join(used) if len(used) > 1 else used[0] if used else "?"
            else:
                fam_label = ("%d runs, n=%d" % (len(fam), len(pooled))) if len(fam) > 1 else fam[0][0]
            if len(pooled) >= N_REQUIRED:
                resolved[key].append((fam_label, pooled, max(r[2] for r in fam)))
            else:
                rejected.append((" + ".join(r[0] for r in fam), key[0],
                                 "pooled family has only n=%d" % len(pooled)))
            continue
        # Candidates are full runs AND shard families, so a sharded run competes on its own date
        # rather than losing to an older full run just because it was split across jobs.
        for run_id, scores, mt, sig, voided in runs:
            if len(scores) >= N_REQUIRED:
                resolved[key].append((run_id, scores[:N_REQUIRED], mt))
        by_sig = defaultdict(list)
        for run_id, scores, mt, sig, voided in runs:
            if len(scores) < N_REQUIRED:
                by_sig[sig].append((run_id, scores, mt))
        for sig, shards in by_sig.items():
            shards.sort(key=lambda r: (r[2], r[0]))
            pooled, names, newest = [], [], ""
            for run_id, scores, mt in shards:
                if len(pooled) >= N_REQUIRED:
                    break
                pooled += scores
                names.append(run_id)
                newest = max(newest, mt)
            if len(pooled) >= N_REQUIRED:
                resolved[key].append((" + ".join(names), pooled[:N_REQUIRED], newest))
            elif not resolved[key]:
                rejected.append((" + ".join(r[0] for r in shards), key[0],
                                 "shards pool to only n=%d, need %d"
                                 % (sum(len(s) for _, s, _ in shards), N_REQUIRED)))
    admitted = resolved

    print("=" * 100)
    print("ADMITTED RUNS")
    print("=" * 100)
    for key in sorted(admitted):
        task, policy, column = key
        for run_id, scores, _ in sorted(admitted[key], key=lambda r: (r[2], r[0])):
            print("  %-12s %-6s %-8s %-30s n=%d mean=%6.2f"
                  % (task, policy, column, run_id, len(scores), sum(scores) / len(scores)))

    print()
    print("=" * 100)
    print("REJECTED RUNS  (reason read from the run's own log)")
    print("=" * 100)
    for run_id, task, reason in sorted(rejected):
        print("  %-30s %-12s %s" % (run_id, task or "?", reason))

    print()
    print("=" * 100)
    print("TABLE   selection rule: %s" % args.select)
    print("=" * 100)
    print("%-20s %7s %5s %8s %-18s %4s %8s %-18s %4s %9s %6s"
          % ("cell", "real", "n", LA[:8], "gap vs real", "cons", LB[:8], "gap vs real",
             "cons", "closer", "P(A)"))

    rows, missing, n_closer, n_decided = [], [], 0, 0
    n_cons = {A: 0, B: 0}

    for task in TASKS:
        for policy in POLICIES:
            rv = real.get((task, policy))
            if not rv:
                continue
            rmean = sum(rv) / len(rv)
            rlo, rhi = bootstrap_ci(rv, rng)

            picked = {}
            for column in (A, B):
                cands = admitted.get((task, policy, column), [])
                if not cands:
                    picked[column] = None
                    missing.append("%s %s %s" % (task, policy, column))
                    continue
                if args.select == "latest":
                    picked[column] = max(cands, key=lambda r: (r[2], r[0]))
                else:
                    # n20 and pool resolve to exactly one entry per cell-column above
                    picked[column] = cands[0]

            # the draw order (A's interval, then B's, then P(A closer)) fixes what seed 0 gives
            mean, gap, cons = {}, {}, {}
            for column in (A, B):
                run = picked[column]
                mean[column] = sum(run[1]) / len(run[1]) if run else None
                gap[column] = gap_ci(run[1], rv, rng) if run else None
                cons[column] = bool(gap[column] and gap[column][0] <= 0 <= gap[column][1])
                n_cons[column] += cons[column]

            a, b = picked[A], picked[B]
            if a and b:
                n_decided += 1
                closer = A if abs(mean[A] - rmean) < abs(mean[B] - rmean) else B
                n_closer += closer == A
                pa = p_closer(a[1], b[1], rmean, rng)
            else:
                closer, pa = "pending", float("nan")

            fmt = lambda c: (("%.2f" % mean[c]) if picked[c] else "--",
                             ("[%+.1f,%+.1f]" % gap[c]) if gap[c] else "--",
                             "Y" if cons[c] else "N")
            print("%-20s %7.2f %5d %8s %-18s %4s %8s %-18s %4s %9s %6s"
                  % ((task + " " + policy, rmean, len(rv)) + fmt(A) + fmt(B)
                     + (closer, ("%.2f" % pa) if pa == pa else "--")))

            row = dict(task=task, policy=policy, real_mean=rmean, real_ci=[rlo, rhi])
            for column in (A, B):
                run = picked[column]
                row[column] = dict(run=run[0], mean=mean[column], scores=run[1]) if run else None
                row[column + "_gap_ci"] = list(gap[column]) if gap[column] else None
                row[column + "_consistent"] = cons[column]
            row.update(closer=closer, p_first_closer=pa if pa == pa else None)
            rows.append(row)

    print()
    for column, label in ((A, LA), (B, LB)):
        print("%-8s consistent with real (gap CI contains 0) : %d of %d"
              % (label, n_cons[column], len(rows)))
    print("%-8s closer to real                           : %d of %d decided cells"
          % (LA, n_closer, n_decided))
    # An interval that ends at zero gives a verdict the bootstrap seed can flip. Say which cells
    # those are rather than let a count read as firmer than it is.
    edge = ["%s %s %s [%+.2f, %+.2f]" % (r["task"], r["policy"], col, g[0], g[1])
            for r in rows for col in (A, B)
            for g in [r[col + "_gap_ci"]] if g and min(abs(g[0]), abs(g[1])) <= BOUNDARY]
    if edge:
        print()
        print("On the boundary (an interval end within %.2f of zero, verdict is seed-sensitive):"
              % BOUNDARY)
        for e in edge:
            print("   %s" % e)
    print()
    print("'gap vs real' is the 95% CI on (sim mean - real mean), both sides resampled.")
    print("'cons' = that interval contains zero, i.e. the cell is statistically")
    print("         indistinguishable from the real rig. This is the only consistency test.")
    print("P(A) is the bootstrap probability that %s comes out closer on a re-draw." % LA)
    print("Values near 0.50 are coin flips and should not be reported as a win.")

    if missing:
        print()
        print("NO ADMISSIBLE RUN for %d cell-columns:" % len(missing))
        for m in missing:
            print("   %s" % m)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(dict(columns=[A, B], labels={A: LA, B: LB}, rows=rows, rejected=rejected,
                           consistent=n_cons, first_closer=n_closer,
                           decided=n_decided, missing=missing,
                           select=args.select, seed=args.seed,
                           bootstrap=BOOTSTRAP), fh, indent=1)
        print("\nwrote %s" % args.json_out)

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
