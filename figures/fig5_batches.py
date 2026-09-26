#!/usr/bin/env python3
"""Figure 5: every gate-passing batch, per cell, against the real rig's 95% interval.

    python3 fig5_batches.py --runs RUNS --real ../scoring/real_reference_pertrial.csv \
                            --runs-list ../scoring/benchmark_runs.txt

The reported table picks one batch per cell-column (see build_table.py --select). This figure
shows the batches that were NOT picked as well, so a reader can see how much the selection rule
could have moved each number. Admission is build_table.admit, unchanged: a batch is a complete
run of at least N_REQUIRED trials, or shards of one configuration pooled in run-id order.
Scores are percent of each task's maximum so the ten cells share one axis.
"""
import argparse, importlib.util, json, os, random, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bt", os.path.join(HERE, "..", "scoring", "build_table.py"))
bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)

MAX_SCORE = {"bottles": 20, "latte": 10, "clear_table": 70, "blocks": 30, "bowls": 50}
LABEL = {"bottles": "bottles", "bowls": "bowls", "blocks": "blocks", "latte": "latte", "clear_table": "clear table"}
POL = {"molmo": "MolmoAct2", "pi05": "π0.5"}
C_K, C_P = "#1f6f4f", "#b5651d"


def batches(runs_dir, reported_groups, arms):
    """(task, policy, column) -> [(label, scores)] for every admissible batch.

    Batches named in the runs list are formed first, exactly as written ('a + b' pools those
    shards). Every other admissible run is then a batch on its own if complete, or pooled with
    the remaining shards of its configuration in run-id order, as build_table does.
    """
    admitted = defaultdict(list)
    for run_id in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, run_id)
        log, res = os.path.join(d, "run.log"), os.path.join(d, "results.jsonl")
        if not (os.path.isdir(d) and os.path.exists(res) and os.path.exists(log)):
            continue
        cfg = bt.parse_run_log(log)
        scores, voided = bt.parse_results(res)
        reasons, column, policy = bt.admit(d, cfg, scores, voided, arms)
        if reasons:
            continue
        admitted[(cfg["task"], policy, column)].append((run_id, scores, cfg["started"] or "", bt.signature(cfg)))
    out = defaultdict(list)
    used = set()
    by_id = {r[0]: (key, r) for key, runs in admitted.items() for r in runs}
    for group in reported_groups:
        ids = [g.strip() for g in group.split("+")]
        if not all(i in by_id for i in ids):
            print("runs-list batch not admissible or not present: %s" % group, file=sys.stderr)
            continue
        key = by_id[ids[0]][0]
        pooled = [v for i in ids for v in by_id[i][1][1]]
        out[key].append((group, pooled[:bt.N_REQUIRED]))
        used.update(ids)
    for key, runs in admitted.items():
        by_sig = defaultdict(list)
        for run_id, scores, mt, sig in runs:
            if run_id in used:
                continue
            if len(scores) >= bt.N_REQUIRED:
                out[key].append((run_id, scores[:bt.N_REQUIRED]))
            else:
                by_sig[sig].append((run_id, scores, mt))
        for sig, shards in by_sig.items():
            shards.sort(key=lambda r: (r[2], r[0]))
            pooled, names = [], []
            for run_id, scores, mt in shards:
                if len(pooled) >= bt.N_REQUIRED:
                    break
                pooled += scores; names.append(run_id)
            if len(pooled) >= bt.N_REQUIRED:
                out[key].append((" + ".join(names), pooled[:bt.N_REQUIRED]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--runs-list", required=True, help="benchmark_runs.txt; these batches are drawn filled")
    ap.add_argument("--arms", default=bt.ARMS_FILE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "fig5_batches.png"))
    args = ap.parse_args()

    reported = []
    for ln in open(args.runs_list):
        ln = " + ".join(p.strip() for p in ln.split("#", 1)[0].split("+")).strip()
        if ln:
            reported.append(ln)

    rng = random.Random(args.seed)
    real = bt.load_real(args.real)
    arms = bt.load_arms(args.arms)
    A_, B_ = list(arms)[:2]
    B = batches(args.runs, reported, arms)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.7, "axes.edgecolor": "#3a3a3a",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "text.color": "#1a1a1a",
        "axes.labelcolor": "#1a1a1a", "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    cells = [(t, p) for t in bt.TASKS for p in bt.POLICIES]
    print("%-24s %-10s %-45s %7s %s" % ("cell", "column", "batch", "mean", "reported"))
    for i, (task, policy) in enumerate(cells):
        f = 100.0 / MAX_SCORE[task]
        rv = real[(task, policy)]
        rmean = sum(rv) / len(rv) * f
        lo, hi = bt.bootstrap_ci(rv, rng)
        ax.add_patch(plt.Rectangle((i - 0.42, lo * f), 0.84, (hi - lo) * f, color="#ececec", zorder=1, lw=0))
        ax.plot([i - 0.42, i + 0.42], [rmean, rmean], color="#3a3a3a", lw=1.1, zorder=2)
        for column, colour, dx in ((A_, C_K, -0.16), (B_, C_P, 0.16)):
            bs = B.get((task, policy, column), [])
            means = {label: sum(sc) / len(sc) for label, sc in bs}
            rep = [l for l in means if l in reported]
            others = [m for l, m in means.items() if l not in reported]
            x = i + dx
            # thin bar: the spread of every other gate-passing batch for this cell and condition
            if others:
                lo_o, hi_o = min(others + [means[rep[0]]] if rep else others) * f, max(others + [means[rep[0]]] if rep else others) * f
                ax.plot([x, x], [lo_o, hi_o], color=colour, lw=2.2, alpha=0.30, solid_capstyle="round", zorder=3)
            if rep:
                ax.scatter([x], [means[rep[0]] * f], s=46, color=colour, edgecolor="white", linewidth=0.8, zorder=5)
            for label, m in means.items():
                print("%-24s %-10s %-45s %7.2f %s" % ("%s %s" % (LABEL[task], POL[policy]) if column == A_ and label == list(means)[0] else "",
                      column, label, m, "REPORTED" if label in reported else ""))

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(["%s\n%s" % (LABEL[t], POL[p]) for t, p in cells], fontsize=7.8)
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylim(0, 80)
    ax.set_ylabel("mean score, % of task maximum")
    ax.grid(axis="y", color="#f2f2f2", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [
        Line2D([], [], color="#3a3a3a", lw=1.1, label="real mean"),
        plt.Rectangle((0, 0), 1, 1, color="#ececec", label="real 95% bootstrap interval"),
        Line2D([], [], marker="o", ls="", color=C_K, markeredgecolor="white", label="authored reconstruction, reported batch"),
        Line2D([], [], marker="o", ls="", color=C_P, markeredgecolor="white", label="default reconstruction, reported batch"),
        Line2D([], [], color="#9a9a9a", lw=2.2, alpha=0.5, label="range of all gate-passing batches")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.6, ncol=3, borderaxespad=0.4)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
