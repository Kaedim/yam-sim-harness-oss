# ops: the job queue the published runs used

A per-GPU job queue under systemd, kept because it is how the reported runs were launched. It is
not needed to reproduce anything: `./run_cell.sh` runs a cell directly. It assumes our box layout
(`/opt/isaacwork`, `/opt/bench`, openpi in `/opt/openpi`, MolmoAct2 in `/opt/molmoact2`), so expect
to edit paths before using it elsewhere.

- `install.sh` copies the scripts, `run_trials.sh` and `isaac_trials.py` into `/opt/bench` and
  enables one `yam-worker@N` service per GPU.
- `enqueue.sh ID TASKSET POLICY TRIALS ITERS [ENV=VAL...]` drops a job into `/opt/bench/queue/queued/`.
- A worker claims the job, starts the policy server (pi0.5) or the MolmoAct2 server and bridge on
  port 5566+N or 5577+N, runs the trials with up to three retries (resuming from `results.jsonl`),
  writes a quick `summary.md` (`stats.py`; it compares against the evaluator's raw per-trial CSV at
  `REAL_CSV` if present, otherwise it summarises the sim side only), and syncs to S3 if `S3_BUCKET`
  is set in `/opt/bench/env`. The reported numbers come from `scoring/`, not from these summaries.
- Per-trial clips and composed videos (`incremental_clips.py`, `compose_all.py`) are not shipped;
  those steps are skipped and do not affect results.
- Output: `/opt/bench/jobs/<ID>/` (`results.jsonl`, `run.log`, `summary.md`). `notify.py` appends to
  `notify.log` and posts to a webhook if `SLACK_WEBHOOK` is set.
