# make test                         gate, table and statistics tests (no data, no GPU)
# make reproduce RUNS=/path/runs    rebuild the paper's tables from the run artefacts
# make test-reproduce RUNS=...      assert they equal the committed scoring/table.json
# make figures RUNS=...             Figures 3 to 5 into build/

PY    ?= python3
REAL  := scoring/real_reference_pertrial.csv
LIST  := scoring/benchmark_runs.txt
OUT   := build

.PHONY: test reproduce test-reproduce figures check-runs clean

test:
	$(PY) -m unittest discover tests

check-runs:
	@test -n "$(RUNS)" || { echo "set RUNS=/path/to/downloaded/runs"; exit 2; }

$(OUT):
	mkdir -p $(OUT)

reproduce: check-runs | $(OUT)
	$(PY) scoring/build_table.py --runs $(RUNS) --real $(REAL) --runs-list $(LIST) --json-out $(OUT)/table.json
	$(PY) scoring/agreement.py $(OUT)/table.json --real $(REAL)
	$(PY) scoring/median_table.py $(OUT)/table.json --real $(REAL)

test-reproduce: check-runs
	YAM_RUNS=$(RUNS) $(PY) -m unittest tests.test_reproduce -v

figures: check-runs | $(OUT)
	$(PY) figures/fig3_agreement.py scoring/table.json --out $(OUT)/fig3_agreement.png
	$(PY) figures/fig4_stage_reach.py scoring/table.json --real $(REAL) --out $(OUT)/fig4_stage_reach.png
	$(PY) figures/fig5_batches.py --runs $(RUNS) --real $(REAL) --runs-list $(LIST) --out $(OUT)/fig5_batches.png

clean:
	rm -rf $(OUT)
