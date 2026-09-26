# YAM sim harness

Measure whether your asset pipeline changes how well a simulated policy evaluation predicts
the real robot.

You point the harness at a robot cell that has real per-trial scores and swap the asset arm
(the objects, and optionally the scene splat) while the robot, cameras, layouts, policies and
rubrics stay the same. It reports how far the simulated
evaluation lands from the real one, for each task and policy, with bootstrap confidence
intervals. Every run is checked against a frozen configuration, read from the run's own log,
before it can count.

This is the code and the scoring for the paper *Measuring Asset Reconstruction Effects in Real-to-Sim
Robot Evaluation* [link added at release].

## Results in the paper

A bimanual [I2RT YAM](https://github.com/i2rt-robotics) cell, five tasks, two policies
(pi0.5 and MolmoAct2), 20 trials per cell. The real trials were run and graded by an
independent evaluator, Robocurve, on their own rig under their own staged rubrics. We did not
run the real robot or score the real side.

The cell was reconstructed twice. The two arms differ only in the objects and the scene splat:

| | authored | default path ([PolaRiS](https://arxiv.org/abs/2512.16881) recipe) |
|---|---|---|
| object geometry | reconstructed from one photograph by our pipeline ([projectsim](https://projectsim.ai)) | the same photograph through [TRELLIS](https://arxiv.org/abs/2412.01506) |
| metric scale | estimated from a printed ChArUco board of known size | typed by hand, no measurement |
| collider | per object: signed distance field, convex hulls or convex decomposition | convex decomposition |
| mass, friction, restitution | authored per object | PhysX defaults |
| scene | 3D Gaussian splat of the room | 2D Gaussian splat of the same video |

Centre of mass and inertia are engine-derived in both arms. The robot model, step budgets,
action chunks, cameras, layouts, instructions and rubrics are identical.

| | authored | default path | paired difference, 95% CI |
|---|---|---|---|
| score error, percentage points | 6.97 | 17.54 | +10.56 [+5.03, +17.11] |
| progress disagreement, percentage points | 15.90 | 24.23 | +8.32 [+1.78, +16.74] |
| Pearson r against real | 0.90 | 0.51 | +0.39 [-0.08, +0.87] |
| cells consistent with real | 9 of 10 | 5 of 10 | |

The per-cell table, how the reported batches were chosen, batch variance and the median check
are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Layout

```
harness/     trial runner (Isaac Sim), policy servers, container launcher
harness/ops/ optional per-GPU job queue, one worker per GPU
configs/     one file per reported cell, plus arms.json (the asset arms the gate knows)
scene/       build a Gaussian splat of your cell from a phone video and align it
asset-arms/  reference implementation of the default asset path (TRELLIS + defaults)
scoring/     the configuration gate, bootstrap statistics, the committed run list and table
figures/     the scripts behind every figure in the paper
spec/        camera calibration dumps from the real rig
docs/        results and the configuration gate
tests/       gate, table and statistics tests, plus a check of the published numbers
```

## Install

Scoring and tests need only Python 3.9+ and the standard library. Figures need a few packages:

```bash
pip install -r requirements.txt          # matplotlib, numpy, pillow, usd-core, coacd
make test                                # no data or GPU needed; the 4 reproduction tests skip
```

Running trials needs NVIDIA Isaac Sim 6.0.1 from its container, one GPU per worker (developed
on L40S 46 GB), and a policy served over a socket: `harness/pi05_server.py` and
`harness/molmoact2_bridge.py` are the two provided
(pi0.5 loads with the reconstructed openpi config in `harness/yam_pi05.py`).

## Reproduce the paper's numbers (no GPU)

The run artefacts (`results.jsonl`, `run.log`, `job.json` per run) are released with the
dataset *[link added at release]*. Download them, then:

```bash
make reproduce RUNS=/path/to/runs        # table, Table 4 measures, median check
make test-reproduce RUNS=/path/to/runs   # asserts the output equals scoring/table.json
make figures RUNS=/path/to/runs          # Figures 3 to 5
```

`make reproduce` runs these three commands:

```bash
python3 scoring/build_table.py --runs RUNS --real scoring/real_reference_pertrial.csv \
    --runs-list scoring/benchmark_runs.txt --json-out table.json
python3 scoring/agreement.py table.json --real scoring/real_reference_pertrial.csv
python3 scoring/median_table.py table.json --real scoring/real_reference_pertrial.csv
```

`build_table.py` reads each run's own `run.log`, rejects any run whose configuration does not
match, and prints every rejection with its reason before the table. `--runs-list` names the one
batch per cell and arm that the paper reports, so the table does not depend on what else is in
`RUNS`. All intervals use bootstrap seed 0 by default and reproduce exactly.

Figure 1 is drawn from the asset USDs rather than the runs:

```bash
python3 figures/fig1_objects.py sm_bowl_plasticribbed_07f "ribbed bowl" \
    --a-root ASSETS/authored --b-root ASSETS/default --photo Photo_CreamBowl.png --out fig1_bowl
```

## Re-run the simulated trials

`HARNESS_ROOT` (default `/opt/isaacwork`) holds `assets/` (authored arm, including the robot
in `yam_robot/`), `assets_polaris/` (default arm, same folder names) and the splat USDs. Each
file in `configs/` holds every value that differs between cells:

```bash
python harness/pi05_server.py --port 5566            # serve the policy
./run_cell.sh configs/bottles_pi05.env kaedim        # 20 trials, authored arm
./run_cell.sh configs/bottles_pi05.env polaris       # the same 20 trials, default arm
TRIALS=1 ./run_cell.sh configs/latte_pi05.env kaedim # any value can be overridden
```

The arms are called `kaedim` (authored) and `polaris` (default path) in the code because the
published runs recorded those names and the gate reads them back out of the logs.

The runner reads about 100 environment variables. The reported runs set only the ones in
`configs/`; everything else stayed at its default. The rest is development surface (physics
sweeps, camera matching, ablations). Setting any of it leaves the benchmark configuration, and
`build_table.py` will reject the run and say why. `docs/CONFIG_GATE.md` lists what is checked.

The runner is the code that produced the published runs. It differs from that code only in
comments and in reading its root directory from `HARNESS_ROOT`, which defaults to the path
the runs used.

## Bring your own assets

An asset arm is a directory of USDs with the same folder and prim names as `assets/`:

```
$HARNESS_ROOT/my_assets/
  sm_bottle_glassheineken_8d6/sm_bottle_glassheineken_8d6.usd
  sm_bowl_plasticribbed_07f/sm_bowl_plasticribbed_07f.usd
  ...
```

1. Run the cells with your arm:
   `ASSET_ROOT=$HARNESS_ROOT/my_assets ./run_cell.sh configs/bowls_pi05.env mine`
   (the last argument becomes `ASSET_ARM`, which the run log records).
   To use your own scene capture too, set `SPLAT_USD`, `SPLAT_XFORM_FILE` and `SPLAT_DZ`; see
   [`scene/README.md`](scene/README.md).
2. Declare the arm in `configs/arms.json`: its name, the asset folder name, the splat file and
   `SPLAT_DZ` as the runner prints it (`%+.3f`).
3. Score it against either published arm, or against your own second arm:
   `python3 scoring/build_table.py --runs RUNS --real scoring/real_reference_pertrial.csv --compare mine polaris`.

`figures/usd_physics.py ROOT` prints what each of your USDs authors (collider, mass, material,
combine mode), and `figures/fig1_objects.py` draws any object side by side with another arm.

## Bring your own cell

You need real per-trial scores; a mean without the trials is not enough, because the harness
compares distributions. `scoring/real_reference_pertrial.csv` is the format: one row per trial
with `task`, `policy`, `run`, `score`, `max`. Staged rubrics work best: partial credit for
touched, lifted, placed, placed upright measures how far the robot got, which is where asset
fidelity shows up before it shows up in success rate.

A new cell also needs code, not only configuration: the task layouts and rubric scoring live
in `harness/isaac_trials.py`, and the cell's step budgets, camera heights and task list are the
constants at the top of `scoring/build_table.py`.

## Limitations

- Five tasks by two policies is small. The real rig has no stable policy ordering on three of
  the five tasks, so ranking preservation and MMRV are not computable here.
- n=20 with discrete staged scores leaves a large noise floor. Three of the ten cells are
  decided by half a point or less and should be read as ties.
- Stack blocks with MolmoAct2 is not reached by either arm; it is reported, not excluded.
- The default asset arm was built by us following the published recipe, not supplied by the
  PolaRiS authors.

## Licence and credit

Code: Apache 2.0, see `LICENSE`.

The real per-trial scores in `scoring/real_reference_pertrial.csv` were produced by Robocurve
and are released with their permission. Please credit them if you use the real reference.
