#!/usr/bin/env python3
"""Figure 4: stage-reach curves, real against both reconstructions, every cell.

    python3 fig4_stage_reach.py table.json --real ../scoring/real_reference_pertrial.csv

For each cell the rungs are the distinct non-zero scores the REAL rig produced (agreement.py uses
the same ladder), and each curve is the fraction of trials that reached at least that rung. The
mean absolute gap between a simulated curve and the real curve is the progress disagreement in
Table 6; this figure prints those numbers so the two can be checked against each other.
"""
import argparse, csv, importlib.util, json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ag", os.path.join(HERE, "..", "scoring", "agreement.py"))
ag = importlib.util.module_from_spec(spec); spec.loader.exec_module(ag)

TASKS = ["bottles", "bowls", "blocks", "latte", "clear_table"]
POLICIES = ["molmo", "pi05"]
LABEL = {"bottles": "bottles in bin (20)", "bowls": "stack bowls (50)", "blocks": "stack blocks (30)",
         "latte": "move latte cup (10)", "clear_table": "clear table (70)"}
POL = {"molmo": "MolmoAct2", "pi05": "π0.5"}
C_R, C_K, C_P = "#2b2b2b", "#1f6f4f", "#b5651d"


def load_real(path):
    real = defaultdict(list)
    for row in csv.DictReader(open(path)):
        t, p = ag.TASK.get(row["task"]), ag.POLICY.get(row["policy"])
        if t and p:
            real[(t, p)].append(float(row["score"]))
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_json")
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "fig4_stage_reach.png"))
    args = ap.parse_args()

    real = load_real(args.real)
    T = json.load(open(args.table_json))
    A, B = T["columns"]
    rows = {(r["task"], r["policy"]): r for r in T["rows"]}

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.linewidth": 0.7, "axes.edgecolor": "#3a3a3a",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "text.color": "#1a1a1a",
        "axes.labelcolor": "#1a1a1a", "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 4.6), sharey=True)
    print("%-22s %8s %8s   (progress disagreement, percentage points)" % ("cell", T["labels"][A][:8], T["labels"][B][:8]))
    for i, policy in enumerate(POLICIES):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            rv = real[(task, policy)]
            rungs = sorted(set(s for s in rv if s > 0))
            r = rows[(task, policy)]
            xs = list(range(len(rungs) + 1))
            for scores, colour, lw, z in ((rv, C_R, 1.8, 3), (r[A]["scores"], C_K, 1.3, 4), (r[B]["scores"], C_P, 1.3, 4)):
                ys = [100.0] + [100.0 * v for v in ag.reach(scores, rungs)]
                ax.plot(xs, ys, color=colour, lw=lw, marker="o", ms=3, zorder=z)
            ax.set_xticks(xs)
            ax.set_xticklabels(["0"] + ["%g" % v for v in rungs], fontsize=6.8)
            ax.set_ylim(-3, 103)
            ax.grid(color="#f2f2f2", lw=0.6); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(LABEL[task], fontsize=9, loc="left", pad=6)
            if j == 0:
                ax.set_ylabel("%s\n%% of trials reaching\nat least this score" % POL[policy], fontsize=8.5)
            if i == 1:
                ax.set_xlabel("score reached, rubric points", fontsize=8)
            print("%-22s %8.1f %8.1f" % ("%s %s" % (task, POL[policy]),
                  ag.progress_gap(r[A]["scores"], rv, rungs), ag.progress_gap(r[B]["scores"], rv, rungs)))
    fig.legend(handles=[plt.Line2D([], [], color=C_R, lw=1.8, marker="o", ms=3, label="real robot"),
                        plt.Line2D([], [], color=C_K, lw=1.3, marker="o", ms=3, label="authored reconstruction"),
                        plt.Line2D([], [], color=C_P, lw=1.3, marker="o", ms=3, label="default reconstruction")],
               loc="upper right", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.99, 1.02))
    fig.subplots_adjust(wspace=0.12, hspace=0.35, top=0.88)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
