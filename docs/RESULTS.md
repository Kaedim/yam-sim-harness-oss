# Results

Every number on this page is printed by the scripts in `scoring/` from the committed run list
(`scoring/benchmark_runs.txt`) at the default bootstrap seed, 0. `make reproduce` rebuilds all
of them, and `tests/test_reproduce.py` checks them against `scoring/table.json`.

Ten cells: five tasks, two policies, 20 trials per cell on each side. Scores are the
evaluator's staged rubric points; where cells are compared across tasks, each is first
normalised to percent of the task's maximum.

## Headline

`python3 scoring/agreement.py scoring/table.json --real scoring/real_reference_pertrial.csv`

| measure | authored | default path | paired difference, 95% CI |
|---|---|---|---|
| score error, percentage points | 6.97 | 17.54 | +10.56 [+5.03, +17.11] |
| progress disagreement, percentage points | 15.90 | 24.23 | +8.32 [+1.78, +16.74] |
| Pearson r | 0.90 | 0.51 | +0.39 [-0.08, +0.87] |
| failure-stage disagreement, percentage points | 42.00 | 48.50 | +6.50 [-3.50, +17.00] |

A positive difference means the authored arm is closer to the real robot. Differences are
paired on the cell, because the ten cells differ far more in difficulty than the two arms do.
Error and progress separate the arms; correlation and failure stage do not.

## Per cell

`python3 scoring/build_table.py ... --runs-list scoring/benchmark_runs.txt`

Mean score in rubric points (task maximum in brackets). "Consistent" means the 95% bootstrap
interval on (sim mean - real mean), resampling both sides, contains zero. It is the only
consistency test used.

| task | policy | real | authored | consistent | default | consistent | authored run | default run |
|---|---|---|---|---|---|---|---|---|
| Bottles in bins (20) | MolmoAct2 | 9.50 | 5.95 | yes | 6.25 | yes | K2_bot_molmo_plain | S1_PS_bot_molmo |
| Bottles in bins (20) | pi0.5 | 10.05 | 9.40 | yes | 8.90 | yes | K1_bot_pi05_plain | B0_PS_bot_pi05_r2 |
| Stack bowls (50) | MolmoAct2 | 29.75 | 25.75 | yes | 16.75 | no | RB_BOWLS_a + RB_BOWLS_b | S2_PS_bowls_molmo |
| Stack bowls (50) | pi0.5 | 29.25 | 30.50 | yes | 23.00 | yes | C2_bowls_pi05 | S7_PS_bowls_pi05 |
| Stack blocks (30) | MolmoAct2 | 3.50 | 7.75 | no | 12.75 | no | C1_blocks_molmo | S3_PS_blocks_molmo |
| Stack blocks (30) | pi0.5 | 6.00 | 5.75 | yes | 5.50 | yes | T1_blocks_pi05 | ZZ1_PS_blocks_pi05 |
| Move latte cup (10) | MolmoAct2 | 5.90 | 5.00 | yes | 3.75 | no | V2_K_latte_molmo | S4_PS_latte_molmo |
| Move latte cup (10) | pi0.5 | 3.65 | 3.75 | yes | 0.25 | no | K3_latte_pi05_plain | S9_PS_latte_pi05 |
| Clear table (70) | MolmoAct2 | 45.35 | 42.50 | yes | 38.65 | yes | V2_K_cleartable_molmo | S5_PS_ct_molmo |
| Clear table (70) | pi0.5 | 47.80 | 41.40 | yes | 35.70 | no | ZZ2_K_ct_pi05 | SA_PS_ct_pi05 |

- Consistent with real: authored 9 of 10, default path 5 of 10.
- The authored mean is nearer the real mean in 9 of 10 cells. Three margins are 0.50 points or
  less (blocks pi0.5, bottles pi0.5, bottles MolmoAct2) and should be read as ties. In bottles
  MolmoAct2 the default path is the nearer, by 0.30.
- Bottles MolmoAct2, authored: the interval is [-6.95, +0.00] at seed 0, so the verdict sits on
  the boundary and other seeds can flip it. `build_table.py` prints every cell in this state.

## How the reported batch was chosen

Several cells have more than one batch of twenty trials that passed the configuration gate
(`docs/CONFIG_GATE.md`). For each cell and arm the reported batch is the most recent by the
run's own logged start time. The choice is committed as `scoring/benchmark_runs.txt`, so the
reported table is built with no selection rule at all.

For comparison, without a run list:

| rule | authored r | default r | authored consistent | default consistent |
|---|---|---|---|---|
| reported run list | 0.90 | 0.51 | 9 of 10 | 5 of 10 |
| `--select pool` (every gate-passing trial) | 0.83 | 0.56 | 7 of 10 | 4 of 10 |

The authored arm is nearer real in 9 of 10 cells under both. `--select latest` does not
reproduce the run list exactly for bowls MolmoAct2: its shard pooling groups `RB_BOWLS_a/b`
with earlier diagnostic shards of the same configuration, so it picks `V2_K_bowls_molmo`
instead. The run list is authoritative.

## Batch variance

At n=20 the spread between gate-passing batches of the same configuration is comparable to
several of the reported gaps: across the 13 cell-arm pairs with more than one batch it ranges
from 0.00 to 8.35 points, median 1.45. Figure 5 (`figures/fig5_batches.py`) plots every batch;
the script prints the full list.

## Median, as a robustness check

`python3 scoring/median_table.py scoring/table.json --real scoring/real_reference_pertrial.csv`

On the median, with the same paired test, the authored arm is consistent in 10 of 10 cells and
the default path in 9 of 10. The mean is the fair read here: per-trial scores are rungs of a
discrete ladder (Stack blocks produced two distinct real values, Move latte cup four), so a
median at n=20 can only land on a rung, its interval comes out about twice as wide as the
mean's, and four of ten cells tie exactly. The script prints the distinct-value counts.

## Policy ranking is not computable here

With two policies, ranking preservation and MMRV need the real rig to order them. Mostly it
does not. Bootstrap probability that MolmoAct2 outscores pi0.5 on the real robot:

| task | P(MolmoAct2 > pi0.5) |
|---|---|
| Bottles in bins | 0.36 |
| Stack bowls | 0.53 |
| Stack blocks | 0.04 |
| Move latte cup | 0.98 |
| Clear table | 0.36 |

Only two of five tasks are decided, so a ranking measure would rest on three coin flips.

## The cell neither arm reaches

**Stack blocks, MolmoAct2.** The real robot scores zero on 65% of trials. Neither simulated arm
ever scores zero, so both score higher than reality (authored 7.75, default 12.75, real 3.50).
It is reported as a failure of the reconstruction, not excluded.
