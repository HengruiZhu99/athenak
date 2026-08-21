#!/usr/bin/env python3
from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
ALLOC = (ROOT / "perlmutter_allocate_segment.sh").read_text()
SUBMIT = (ROOT / "perlmutter_submit_segment.sh").read_text()
RUN = (ROOT / "perlmutter_run_segment.sh").read_text()

assert "--qos=shared_interactive" in ALLOC
assert "--constraint='gpu&hbm80g'" in ALLOC
assert "--ntasks=1" in ALLOC and "--gpus-per-node=1" in ALLOC
assert "--cpus-per-task=32" in ALLOC and "--time=02:00:00" in ALLOC
assert "--qos=interactive" not in ALLOC
assert "exec sbatch --parsable" in SUBMIT
assert "--qos=shared_interactive" in SUBMIT
assert "--constraint='gpu&hbm80g'" in SUBMIT
assert "--ntasks=1" in SUBMIT and "--gpus-per-node=1" in SUBMIT
assert "--cpus-per-task=32" in SUBMIT and "--time=02:00:00" in SUBMIT
assert "--export=ALL" in SUBMIT
assert "allocation-%j.stdout" in SUBMIT and "allocation-%j.stderr" in SUBMIT
assert "n128:replay:64:128:16:16:16384" in RUN
assert "n256:record:128:256:32:32:16384" in RUN
assert "n512:replay:256:512:64:64:16384" in RUN
assert "time/tlim=\"${RUN_TLIM}\"" in RUN
assert "--time=01:45:00" in RUN
assert "--require-cuda" in RUN
assert "PERLMUTTER_CUDA_O4_REPLAY_QUALIFICATION_PASS" in RUN
assert "ATHENA_TEST_Z4C_STATE_EXTRACTION=selected_negative_chi" in RUN
assert "100% tests passed, 0 tests failed out of 12" in RUN
assert "all(abs(x[\"ulp_difference\"]) <= 1 for x in rows)" in RUN
assert "mesh_refinement/amr_history_mode=\"${HISTORY_MODE}\"" in RUN
assert "mesh_refinement/amr_history_file=\"${HISTORY_FILE}\"" in RUN
print("PERLMUTTER_COMMON_TREE_CAMPAIGN_CONTRACT_PASS")
