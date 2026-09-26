#!/usr/bin/env python3
"""The reported table restated on the median, as a second read of the same runs.

This never touches build_table.py. It consumes that script's --json-out, so the batches are the
ones the committed run list names, the same as the mean table, and the only thing that changes
here is the statistic. "Consistent" is the same paired test build_table.py uses, on the median:
the 95% CI on (sim median - real median), both sides resampled, contains zero.

    python3 build_table.py --runs RUNS --real REAL --runs-list benchmark_runs.txt --json-out table.json
    python3 median_table.py table.json --real REAL

Why both are printed. Per-trial scores are rungs of a discrete ladder, not a continuous measure:
Stack blocks takes two distinct values across all twenty real trials, Move latte cup four. At
n=20 a median can only land on one of those rungs, so it moves in jumps, ties between the two
columns constantly, and its bootstrap CI comes out about twice as wide as the mean's. The median
is the right default on outlier-heavy continuous data. Here it throws away the resolution the
comparison depends on. The mean is the fair read and the median is the robustness check.
"""
import argparse, csv, json, random, statistics
from collections import defaultdict

BOOTSTRAP = 20000
TASK = {'Bottles in bins': 'bottles', 'Stack bowls': 'bowls', 'Stack blocks': 'blocks',
        'Move latte cup': 'latte', 'Clear table': 'clear_table'}
POLICY = {'MolmoAct2': 'molmo', 'pi0.5': 'pi05'}


def med_ci(v, rng, resamples=BOOTSTRAP, alpha=0.05):
    m = sorted(statistics.median(rng.choice(v) for _ in range(len(v))) for _ in range(resamples))
    return m[int((alpha / 2) * resamples)], m[int((1 - alpha / 2) * resamples) - 1]


def gap_ci(sim, real, rng, resamples=BOOTSTRAP, alpha=0.05):
    d = sorted(statistics.median(rng.choice(sim) for _ in range(len(sim)))
               - statistics.median(rng.choice(real) for _ in range(len(real)))
               for _ in range(resamples))
    return d[int((alpha / 2) * resamples)], d[int((1 - alpha / 2) * resamples) - 1]


def p_closer(a, b, target, rng, resamples=BOOTSTRAP):
    """Exact ties are split. On the median the two columns land on the same rung often enough
    that counting a tie as a loss would read as a win for the other arm that never happened."""
    w = 0.0
    for _ in range(resamples):
        da = abs(statistics.median(rng.choice(a) for _ in range(len(a))) - target)
        db = abs(statistics.median(rng.choice(b) for _ in range(len(b))) - target)
        w += 1.0 if da < db else (0.5 if da == db else 0.0)
    return w / resamples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_json", help="build_table.py --json-out")
    ap.add_argument("--real", required=True, help="real_reference_pertrial.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    real = defaultdict(list)
    for r in csv.DictReader(open(args.real)):
        real[(TASK[r['task']], POLICY[r['policy']])].append(float(r['score']))

    T = json.load(open(args.table_json))
    A, B = T["columns"]
    LA, LB = T["labels"][A], T["labels"][B]
    print("TABLE   same runs as %s, statistic: median" % args.table_json)
    print("=" * 100)
    print("%-20s %7s %15s %8s %4s %8s %4s %9s %6s" %
          ("cell", "real", "real 95% CI", LA[:8], "cons", LB[:8], "cons", "closer", "P(A)"))

    kcl = ties = kcons = pcons = 0
    for row in T["rows"]:
        t, p = row["task"], row["policy"]
        rv = real[(t, p)]
        rm = statistics.median(rv)
        lo, hi = med_ci(rv, rng)
        kv, pv = row[A]["scores"], row[B]["scores"]
        km, pm = statistics.median(kv), statistics.median(pv)
        kg, pg = gap_ci(kv, rv, rng), gap_ci(pv, rv, rng)
        kc, pc = kg[0] <= 0 <= kg[1], pg[0] <= 0 <= pg[1]
        kcons += kc
        pcons += pc
        dk, dp = abs(km - rm), abs(pm - rm)
        closer = LA if dk < dp else (LB if dp < dk else "tie")
        kcl += closer == LA
        ties += closer == "tie"
        print("%-20s %7.2f [%5.1f,%6.1f] %8.2f %4s %8.2f %4s %9s %6.2f" %
              (t + " " + p, rm, lo, hi, km, "yes" if kc else "no",
               pm, "yes" if pc else "no", closer, p_closer(kv, pv, rm, rng)))

    print()
    n = len(T["rows"])
    print("%-8s consistent with real (median gap CI contains 0) : %d of %d" % (LA, kcons, n))
    print("%-8s consistent with real (median gap CI contains 0) : %d of %d" % (LB, pcons, n))
    print("%-8s closer to real : %d of %d   (exact ties: %d, %s: %d)"
          % (LA, kcl, n, ties, LB, n - kcl - ties))
    print()
    print("Distinct per-trial scores the real rig produced in 20 trials, which is the ceiling on")
    print("how finely a median can resolve anything:")
    for row in T["rows"]:
        rv = real[(row["task"], row["policy"])]
        print("  %-20s %d values  %s" % (row["task"] + " " + row["policy"], len(set(rv)),
                                         sorted(set(rv))))


if __name__ == "__main__":
    main()
