"""The configuration gate: every property the paper says is held fixed must be enforced here.

    python3 -m unittest discover tests
"""
import io, json, os, random, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scoring"))
sys.path.insert(0, HERE)
import build_table as bt                                           # noqa: E402
from fixtures import make_run, write_real, MOLMO, DEFAULT_ARM     # noqa: E402

ARMS = bt.load_arms()
GOOD_SCORES = [10] * 20


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def verdict(self, job_env=None, **kw):
        d = make_run(self.dir, "r", GOOD_SCORES, job_env=job_env, **kw)
        cfg = bt.parse_run_log(os.path.join(d, "run.log"))
        scores, voided = bt.parse_results(os.path.join(d, "results.jsonl"))
        return bt.admit(d, cfg, scores, voided, ARMS)

    def assertRejected(self, fragment, **kw):
        bad, _, _ = self.verdict(**kw)
        self.assertTrue(any(fragment in b for b in bad), "expected %r in %s" % (fragment, bad))

    def test_reported_configuration_is_admitted(self):
        self.assertEqual(self.verdict(), ([], "kaedim", "pi05"))
        self.assertEqual(self.verdict(**DEFAULT_ARM), ([], "polaris", "pi05"))
        self.assertEqual(self.verdict(job_env={"MOLMO_NUM_STEPS": "10"}, **MOLMO), ([], "kaedim", "molmo"))

    def test_short_budget(self):
        self.assertRejected("budget 3600 of 6400", iters=225)

    def test_camera_height(self):
        self.assertRejected("TOP_H=0.8800", top_h=0.88)

    def test_finger_filter(self):
        self.assertRejected("finger filter off", fingers=False)

    def test_splat_offset_never_applied(self):
        self.assertRejected("never applied", dz=None)

    def test_joint_clipping(self):
        self.assertRejected("CLIP_JOINTS=0", clip=False)

    def test_layout(self):
        self.assertRejected("layout v2", layout="v2")

    def test_unknown_chunk(self):
        self.assertRejected("unrecognised chunk", chunk=20, iters=320)

    def test_assets_in_the_other_arms_scene(self):
        self.assertRejected("mixed config", splat=DEFAULT_ARM["splat"])
        self.assertRejected("mixed config", arm="polaris")

    def test_molmo_solver_steps(self):
        self.assertRejected("no job.json", **MOLMO)
        self.assertRejected("MOLMO_NUM_STEPS=30", job_env={"MOLMO_NUM_STEPS": "30"}, **MOLMO)

    def test_voided_trials_are_dropped(self):
        d = make_run(self.dir, "v", [5] * 20, voided=3)
        self.assertEqual(bt.parse_results(os.path.join(d, "results.jsonl")), ([5.0] * 20, 3))

    def test_custom_arm_from_arms_file(self):
        arms = dict(ARMS, mine=dict(label="mine", root_name="my_assets", splat="my_scene.usd", dz="+0.000"))
        d = make_run(self.dir, "m", GOOD_SCORES, arm="mine", root="/data/my_assets",
                     splat="/data/my_scene.usd", dz="+0.000")
        cfg = bt.parse_run_log(os.path.join(d, "run.log"))
        self.assertEqual(bt.admit(d, cfg, GOOD_SCORES, 0, arms), ([], "mine", "pi05"))


class Table(unittest.TestCase):
    """build_table.py end to end on one synthetic cell."""

    def build(self, runs_list_lines, extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            runs = os.path.join(tmp, "runs")
            make_run(runs, "A_full", [8] * 20, started="2026-09-10T00:00:00Z")
            shard = [9, 10, 11, 12] * 2 + [9, 12]         # same spread as real, mean 10.5
            make_run(runs, "A_shard1", shard, started="2026-09-11T00:00:00Z")
            make_run(runs, "A_shard2", shard, started="2026-09-12T00:00:00Z")
            make_run(runs, "B_full", [2] * 20, started="2026-09-10T00:00:00Z", **DEFAULT_ARM)
            make_run(runs, "B_short", [2] * 20, started="2026-09-10T00:00:00Z", iters=225, **DEFAULT_ARM)
            real = os.path.join(tmp, "real.csv")
            write_real(real, {("Bottles in bins", "pi0.5"): [9, 10, 11, 12] * 5})
            lst = os.path.join(tmp, "runs.txt")
            with open(lst, "w") as fh:
                fh.write("\n".join(runs_list_lines) + "\n")
            out = os.path.join(tmp, "t.json")
            p = subprocess.run([sys.executable, os.path.join(REPO, "scoring", "build_table.py"),
                                "--runs", runs, "--real", real, "--runs-list", lst,
                                "--json-out", out] + list(extra), capture_output=True, text=True)
            if not os.path.exists(out):
                return p, None
            with open(out) as fh:
                return p, json.load(fh)

    def test_runs_list_pools_shards_and_rejects_off_config(self):
        p, t = self.build(["A_shard1 + A_shard2", "B_full", "B_short"])
        self.assertEqual(p.returncode, 0, p.stderr)
        row = t["rows"][0]
        self.assertEqual(row["kaedim"]["run"], "A_shard1 + A_shard2")
        self.assertEqual(row["kaedim"]["mean"], 10.5)
        self.assertEqual(row["polaris"]["run"], "B_full")
        self.assertIn(["B_short", "bottles", "budget 3600 of 6400 steps"], t["rejected"])
        self.assertTrue(row["kaedim_consistent"])       # same distribution as real
        self.assertFalse(row["polaris_consistent"])     # 2 vs 10.5: far outside
        self.assertEqual(t["consistent"], {"kaedim": 1, "polaris": 0})

    def test_latest_picks_most_recent_by_logged_start(self):
        p, t = self.build(["A_full", "A_shard1", "A_shard2", "B_full"], ["--select", "latest"])
        self.assertEqual(t["rows"][0]["kaedim"]["run"], "A_shard1 + A_shard2")

    def test_runs_list_naming_a_missing_run_fails(self):
        p, _ = self.build(["A_full", "B_full", "NOT_THERE"])
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("NOT_THERE", p.stderr)

    def test_missing_column_fails_the_build(self):
        p, t = self.build(["A_full"])
        self.assertEqual(p.returncode, 1)
        self.assertEqual(t["missing"], ["bottles pi05 polaris"])


class Statistics(unittest.TestCase):
    def test_gap_ci_is_seeded(self):
        a, b = [0, 5, 10, 20] * 5, [5, 10] * 10
        self.assertEqual(bt.gap_ci(a, b, random.Random(0), resamples=2000),
                         bt.gap_ci(a, b, random.Random(0), resamples=2000))

    def test_identical_samples_are_consistent(self):
        lo, hi = bt.gap_ci([3, 7] * 10, [3, 7] * 10, random.Random(0), resamples=2000)
        self.assertTrue(lo <= 0 <= hi)


if __name__ == "__main__":
    unittest.main()
