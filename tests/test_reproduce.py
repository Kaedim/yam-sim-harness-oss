"""The published numbers, rebuilt from the published runs.

Skipped unless YAM_RUNS points at the downloaded run artefacts (see README, "Reproduce the
paper's numbers"):

    YAM_RUNS=/path/to/runs python3 -m unittest tests.test_reproduce

Checks that build_table.py rebuilds the committed scoring/table.json exactly, and that
agreement.py prints the Table 4 values the paper reports, at the default seed.
"""
import json, os, subprocess, sys, tempfile, unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORING = os.path.join(REPO, "scoring")
RUNS = os.environ.get("YAM_RUNS")
REAL = os.path.join(SCORING, "real_reference_pertrial.csv")

# Table 4 as printed in the paper: authored, default, paired difference and its 95% interval.
TABLE4 = {
    "correlation (higher": "0.90      0.51   +0.39  [-0.08, +0.87]",
    "error, points": "6.97     17.54   +10.56  [+5.03, +17.11]",
    "progress gap, pts": "15.90     24.23   +8.32  [+1.78, +16.74]",
    "failure-stage gap (": "42.00     48.50   +6.50  [-3.50, +17.00]",
}


def run(*args):
    p = subprocess.run([sys.executable] + list(args), capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError("%s failed:\n%s%s" % (args[0], p.stdout[-2000:], p.stderr[-2000:]))
    return p.stdout


@unittest.skipUnless(RUNS, "set YAM_RUNS to the downloaded run artefacts")
class Reproduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.table = os.path.join(cls.tmp.name, "table.json")
        cls.build_out = run(os.path.join(SCORING, "build_table.py"), "--runs", RUNS, "--real", REAL,
                            "--runs-list", os.path.join(SCORING, "benchmark_runs.txt"),
                            "--json-out", cls.table)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_table_matches_committed(self):
        with open(self.table) as a, open(os.path.join(SCORING, "table.json")) as b:
            self.assertEqual(json.load(a), json.load(b))

    def test_no_reported_run_rejected(self):
        with open(self.table) as fh:
            self.assertEqual(json.load(fh)["rejected"], [])

    def test_consistency_counts(self):
        self.assertIn("authored consistent with real (gap CI contains 0) : 9 of 10", self.build_out)
        self.assertIn("default  consistent with real (gap CI contains 0) : 5 of 10", self.build_out)

    def test_table4(self):
        out = run(os.path.join(SCORING, "agreement.py"), self.table, "--real", REAL)
        for measure, values in TABLE4.items():
            line = next((l for l in out.splitlines() if l.startswith(measure)), "")
            self.assertIn(values, line, "Table 4 row %r changed:\n%s" % (measure, line))


if __name__ == "__main__":
    unittest.main()
