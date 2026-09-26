# The configuration gate

The two arms should differ only in their objects and their scene splat. Over many runs,
settings drift: a job forgets a variable, a default changes, a run resumes under a newer
runner. `scoring/build_table.py` therefore admits a run only if the run's own `run.log` shows
the frozen configuration. Job files and launch environments record what was requested, so
the gate does not use them, with one exception below. Rejected runs are printed with the
reason before the table.

## What must match

| property | why it matters |
|---|---|
| step budget | a shorter budget gives the policy less time than the real rig had, and it does not show in the score |
| action chunk | identifies the policy: 16 for pi0.5, 30 for MolmoAct2 |
| top camera height | the policy's main view: 0.72 m for pi0.5, 0.88 m for MolmoAct2, as the evaluator runs them |
| layout version | object start poses, refit from the real rig's start frames |
| layout key | two tasks have a per-policy layout because the real rig changed between policies |
| rig | which real rig's camera intrinsics are modelled |
| finger filter | gripper collision handling; without it the jaws stall before closing |
| joint clipping | targets clipped to each joint's range, not to [-pi, pi] |
| solver steps | MolmoAct2's denoising step count. The runner cannot print it, so it is read from `job.json`, and a MolmoAct2 run without one is rejected |
| arm, asset root, splat, splat offset | must all belong to the same arm in `configs/arms.json`; one arm's objects in the other's scene is rejected |

## What the gate cannot see

It can only check what the runner prints. Some settings are read and never logged, object
placement jitter among them, so an ablation that changed only one of those looks like a
benchmark run and would pool into a cell. This is why the reported runs are named in
`scoring/benchmark_runs.txt` and passed with `--runs-list`: the gate rejects wrong runs, and
the list decides which runs are considered at all.

## Pooling

A cell run as several smaller jobs is pooled only across jobs with an identical configuration
signature. Batches that differ in any gated property stay separate, and the run list names
which one is reported.

## Two problems it caught

- Six runs of one arm rendered the top camera at 0.88 m under a policy evaluated at 0.72 m.
  The height was checked before pooling but not on admission; the gate now checks it on
  admission. The runner only refuses the opposite case (MolmoAct2 at 0.72 m).
- MolmoAct2's solver step count is only in the job file. Runs without that file are rejected
  rather than assumed correct.
