#!/usr/bin/env python3
"""Figure 3: simulated score against real score, both asset arms, ten cells.

    python3 fig3_agreement.py table.json --real ../scoring/real_reference_pertrial.csv

The standard figure for this claim, and the one every paper in this area carries. Scores are
percent of each task's maximum, which is the scale the correlation is computed on. A point on
the diagonal is a simulated evaluation that returned the real robot's number.
"""
import argparse, json, math, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_SCORE = {"bottles": 20, "latte": 10, "clear_table": 70, "blocks": 30, "bowls": 50}
LABEL = {"bottles": "bottles", "bowls": "bowls", "blocks": "blocks",
         "latte": "latte", "clear_table": "clear table"}
POL = {"molmo": "MolmoAct2", "pi05": "π0.5"}
C_K, C_P = "#1f6f4f", "#b5651d"


def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    d = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_json")
    ap.add_argument("--real", help="accepted for a uniform command line; the table already holds the real means")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_agreement.png"))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.7, "axes.edgecolor": "#3a3a3a",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
        "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white"})

    T = json.load(open(args.table_json))
    rows, (A, B) = T["rows"], T["columns"]
    R, K, P, tag = [], [], [], []
    for r in rows:
        f = 100.0 / MAX_SCORE[r["task"]]
        R.append(r["real_mean"] * f); K.append(r[A]["mean"] * f); P.append(r[B]["mean"] * f)
        tag.append("%s %s" % (LABEL[r["task"]], POL[r["policy"]]))

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lim = (0, 78)
    ax.plot(lim, lim, color="#c4c4c4", lw=0.9, ls=(0, (5, 4)), zorder=1)
    ax.text(4, 6.5, "perfect agreement", fontsize=7.4, color="#9a9a9a",
            rotation=45, rotation_mode="anchor", ha="left")


    ax.scatter(R, P, s=46, color=C_P, alpha=0.9, edgecolor="white", linewidth=0.8,
               zorder=3, label="default reconstruction")
    ax.scatter(R, K, s=46, color=C_K, alpha=0.95, edgecolor="white", linewidth=0.8,
               zorder=4, label="authored reconstruction")
    # label only the two cells the caption discusses, at the default point where the departure is
    for r, k, p, t in zip(R, K, P, tag):
        if t == "blocks MolmoAct2":
            ax.annotate(t, (r, p), textcoords="offset points", xytext=(8, -3), fontsize=7, color="#6a6a6a")
        if t == "latte \u03c00.5":
            ax.annotate(t, (r, p), textcoords="offset points", xytext=(8, -3), fontsize=7, color="#6a6a6a")
    ek = statistics.fmean(abs(a - b) for a, b in zip(R, K))
    ep = statistics.fmean(abs(a - b) for a, b in zip(R, P))
    box = ("authored        r = %.2f,  mean error %.1f\ndefault         r = %.2f,  mean error %.1f"
           % (pearson(R, K), ek, pearson(R, P), ep))
    ax.text(2.5, 74.5, box, fontsize=8, va="top", family="monospace", color="#2b2b2b",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa", edgecolor="#e0e0e0", lw=0.7))

    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("real robot score, % of task maximum", fontsize=9)
    ax.set_ylabel("simulated score, % of task maximum", fontsize=9)
    ax.grid(color="#f2f2f2", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.885), frameon=True, fontsize=8.6, facecolor="#fafafa", edgecolor="#e0e0e0", framealpha=1.0)
    fig.savefig(args.out)
    print("wrote %s" % args.out)
    print("  authored      r %.3f   mean error %.2f" % (pearson(R, K), ek))
    print("  default       r %.3f   mean error %.2f" % (pearson(R, P), ep))


if __name__ == "__main__":
    main()
