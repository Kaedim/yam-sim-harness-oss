#!/usr/bin/env python3
"""The headline agreement numbers, computed from the table build_table.py produces.

    python3 build_table.py --runs RUNS --real real_reference_pertrial.csv \
        --runs-list benchmark_runs.txt --json-out table.json
    python3 agreement.py table.json --real real_reference_pertrial.csv

Four measures, each reported per arm with a bootstrap interval, and each compared between the
two arms PAIRED on the cell. Pairing matters: the ten cells differ enormously in difficulty, and
comparing two independent intervals throws that shared variation away. On this data the paired
test separates the arms on error and on progress, and does not separate them on correlation.

  correlation        do the two move together across cells (Pearson, percent of task maximum)
  error              how far apart are they, in points of task completion
  progress           does the robot get equally far: mean gap between the two rung-reach curves
  failure stage      does it stop at the same rung: total variation between the two score
                     distributions

Rungs are the distinct non-zero scores the REAL rig produced in that cell, so the ladder is
reality's, not ours. A cell whose real scores take only one non-zero value supports neither
progress nor failure-stage and is flagged.
"""
import argparse, csv, json, math, random, statistics
from collections import defaultdict

BOOTSTRAP = 20000
MAX_SCORE = {"bottles": 20, "latte": 10, "clear_table": 70, "blocks": 30, "bowls": 50}
TASK = {'Bottles in bins': 'bottles', 'Stack bowls': 'bowls', 'Stack blocks': 'blocks',
        'Move latte cup': 'latte', 'Clear table': 'clear_table'}
POLICY = {'MolmoAct2': 'molmo', 'pi0.5': 'pi05'}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den if den else float("nan")


def reach(scores, rungs):
    """Fraction of trials that got at least as far as each rung."""
    return [sum(1 for s in scores if s >= r) / len(scores) for r in rungs]


def progress_gap(sim, real, rungs):
    """Mean absolute gap between the two rung-reach curves, in percentage points."""
    a, b = reach(sim, rungs), reach(real, rungs)
    return 100.0 * sum(abs(p - q) for p, q in zip(a, b)) / len(rungs)


def failure_gap(sim, real, rungs):
    """Total variation between the two distributions over the rung a trial stopped at.

    Progress asks how far the robot got. This asks where it stopped, which is a different
    question: two systems can reach equally far on average and still pile up at different rungs.
    """
    levels = [0.0] + list(rungs)
    def pmf(v):
        out = []
        for i, lo in enumerate(levels):
            hi = levels[i + 1] if i + 1 < len(levels) else float("inf")
            out.append(sum(1 for s in v if lo <= s < hi) / len(v))
        return out
    a, b = pmf(sim), pmf(real)
    return 100.0 * 0.5 * sum(abs(p - q) for p, q in zip(a, b))


def paired_ci(diffs, rng, resamples=BOOTSTRAP, alpha=0.05):
    n = len(diffs)
    m = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(resamples))
    return m[int((alpha / 2) * resamples)], m[int((1 - alpha / 2) * resamples) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_json", help="build_table.py --json-out")
    ap.add_argument("--real", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-single-rung", action="store_true",
                    help="exclude cells whose real scores take one non-zero value from the "
                         "progress and failure means. They rest on a single point.")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    real = defaultdict(list)
    for r in csv.DictReader(open(args.real)):
        real[(TASK[r["task"]], POLICY[r["policy"]])].append(float(r["score"]))

    T = json.load(open(args.table_json))
    rows = T["rows"]
    A, B = T["columns"]
    LA, LB = T["labels"][A], T["labels"][B]
    R, K, P = [], [], []
    prog, fail, thin = {A: [], B: []}, {A: [], B: []}, []
    cells = []
    for row in rows:
        t, p = row["task"], row["policy"]
        rv = real[(t, p)]
        f = 100.0 / MAX_SCORE[t]
        R.append(row["real_mean"] * f)
        K.append(row[A]["mean"] * f)
        P.append(row[B]["mean"] * f)
        rungs = sorted({s for s in rv if s > 0})
        single = len(rungs) < 2
        if single:
            thin.append(t + " " + p)
        for col in (A, B):
            prog[col].append(progress_gap(row[col]["scores"], rv, rungs))
            fail[col].append(failure_gap(row[col]["scores"], rv, rungs))
        cells.append((t + " " + p, len(rungs), single))

    keep = [i for i, (_, _, s) in enumerate(cells) if not (args.drop_single_rung and s)]
    sub = lambda v: [v[i] for i in keep]

    print("progress gap per cell")
    print("%-22s %8s %8s   %s" % ("cell", LA[:8], LB[:8], "rungs"))
    for i, (name, nr, single) in enumerate(cells):
        print("%-22s %7.1f%% %7.1f%%   %-3d%s" % (name, prog[A][i], prog[B][i],
                                                  nr, "  single rung" if single else ""))
    print()

    ek = [abs(a - b) for a, b in zip(R, K)]
    ep = [abs(a - b) for a, b in zip(R, P)]

    print("=" * 78)
    print("%-34s %9s %9s   %s" % ("measure", LA[:9], LB[:9], "paired difference, 95% CI"))
    print("-" * 78)

    rk, rp = pearson(R, K), pearson(R, P)
    d = []
    for _ in range(BOOTSTRAP):
        idx = [rng.randrange(len(R)) for _ in range(len(R))]
        a = pearson([R[i] for i in idx], [K[i] for i in idx])
        b = pearson([R[i] for i in idx], [P[i] for i in idx])
        if a == a and b == b:
            d.append(a - b)
    d.sort()
    lo, hi = d[int(.025 * len(d))], d[int(.975 * len(d))]
    print("%-34s %9.2f %9.2f   %+.2f  [%+.2f, %+.2f]%s"
          % ("correlation (higher better)", rk, rp, rk - rp, lo, hi,
             "" if lo > 0 else "   contains zero"))

    for label, vk, vp, unit in (
            ("error, points (lower better)", ek, ep, ""),
            ("progress gap, pts (lower better)", sub(prog[A]), sub(prog[B]), ""),
            ("failure-stage gap (lower better)", sub(fail[A]), sub(fail[B]), "")):
        diffs = [b - a for a, b in zip(vk, vp)]      # positive = A closer
        lo, hi = paired_ci(diffs, rng)
        print("%-34s %9.2f %9.2f   %+.2f  [%+.2f, %+.2f]%s"
              % (label, statistics.fmean(vk), statistics.fmean(vp),
                 statistics.fmean(diffs), lo, hi,
                 "" if lo > 0 else "   contains zero"))
    print("=" * 78)
    print("Positive difference means %s is closer to the real robot. An interval that" % LA)
    print("excludes zero is a result; one that contains zero is not, however large the point")
    print("estimate. Correlation is deliberately not the head-to-head measure here.")
    if thin:
        print()
        print("Single-rung cells (real scores take one non-zero value): %s" % ", ".join(thin))
        print("Progress and failure-stage rest on one point there. Re-run with "
              "--drop-single-rung to see the means without them.")


if __name__ == "__main__":
    main()
