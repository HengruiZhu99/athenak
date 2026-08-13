# Candidate-A/C validation and provenance record — 2026-08-13

This record distinguishes source/oracle validation, bounded runtime preflights, the
immutable science campaign, and final post-campaign validation. Absolute local paths are
retained for provenance; the committed evidence bundle supplies repository-relative
copies or hashes where practical.

## Source and build identity

Campaign repository: `/home/hzhu/athenak-candidate-a-c-20260813`

Campaign source: `39e6372fcc7c2ba166a8498a007c858cfde73b6c`

Upstream parent: `5f1993109bcb2e5d588ba41b4efc897408e9959a`

Kokkos: `6739bc623081648af9e752b616d9671527922cbf`

Build directory: `/home/hzhu/build-athenak-candidate-a-c-openmp`

Baseline build: `/home/hzhu/build-athenak-base-5f199310-openmp`

```bash
cmake --build /home/hzhu/build-athenak-candidate-a-c-openmp \
  --target athena z4c_shift_gauge_oracle_test -j 8
/home/hzhu/build-athenak-candidate-a-c-openmp/tst/unit/z4c_shift_gauge_oracle_test
/home/hzhu/build-athenak-candidate-a-c-openmp/src/athena -c
```

Results before the campaign: application and oracle built successfully; oracle exited 0;
configuration reported GCC 13.3, double precision, MPI ON, OpenMP ON. The immutable
manifest contains the complete configuration output.

## Default-path equivalence

The baseline source worktree was `/home/hzhu/athenak-base-5f199310-20260813`.
The base and Candidate-A/C binaries were run for one identical default-gauge cycle.

```text
co_0.txt SHA-256:          3183ed04876a1a3cc5a8dccf0df4a9bbb01bb2aa15a6bdc72a57e3c37646a45e
z4c.user.hst SHA-256:      057558def5a41e19aef47be86ab8c3ca6a2a4de4844e5f0e5751cf45b3c1058e
```

Both hashes are identical between the base and modified executables. Selected raw files
are retained under `validation/default_equivalence/`.

## Parser and bounded runtime preflights

- Unknown `z4c/shift_gauge=invalid` failed during initialization with the exact permitted
  value list.
- Candidate profiles combined with legacy `shift_Gamma`, `shift_alpha2Gamma`, or
  `shift_H` keys failed during initialization.
- A Candidate-C two-cycle, one-rank/eight-thread run exited 0 with finite accepted
  histories and raw gauge-source diagnostics.
- A Candidate-C exact T=0.1 terminal run exited 0 and produced FastFlow horizon shape,
  verbose, and summary records.
- A split T=0.1 to T=0.25 restart demonstrated that the opt-in
  `reset_dt_from_cfl_on_restart` policy replaces only the persisted exact-landing
  remainder proposal. The resumed first step was the ordinary spatial-CFL step
  `0.015625`; the application reached exact accepted T=0.25.
- The first 4-point-MeshBlock qualification preflight exited the application normally at
  T=0.1 but the runner rejected nonfinite native constraint history. It is preserved at
  `/home/hzhu/athenak-candidate-ac-evidence-20260813` as orchestration/preflight evidence,
  not science evidence.
- A second root, `/home/hzhu/athenak-candidate-ac-evidence-r2-20260813`, exposed the
  persisted exact-landing remainder-step behavior. It was not relabeled or continued as
  the primary campaign.

## Immutable primary campaign

```bash
python3 scripts/run_z4c_candidate_ac_qualification.py \
  --repo /home/hzhu/athenak-candidate-a-c-20260813 \
  --binary /home/hzhu/build-athenak-candidate-a-c-openmp/src/athena \
  --run-root /home/hzhu/athenak-candidate-ac-evidence-r3-20260813 --all
```

The API-owned foreground session ended during Candidate-A R0 T=1 without an application
terminal. Its partial attempt was preserved. The identical frozen command then ran under
the transient user service `athenak-candidate-ac-r3-20260813.service`. A second partial
attempt caused by the old process group's cleanup was also preserved before the
persistent service completed that schedule item. Neither interruption is classified as
a PDE failure or a completed outcome.

Every science command is recorded verbatim under
`cases/<case>/segments/segment_<sequence>_t<target>/command.txt`. Each segment also has
stdout, stderr, `/usr/bin/time -v`, exit status, exact accepted time/cycle, accepted-state
admissibility, constraint/source/horizon validation, and the next restart SHA-256.

## Final post-campaign validation

```bash
cmake --build /home/hzhu/build-athenak-candidate-a-c-openmp \
  --target athena z4c_shift_gauge_oracle_test -j 8
/home/hzhu/build-athenak-candidate-a-c-openmp/z4c_shift_gauge_oracle_test
python3 -m py_compile scripts/run_z4c_candidate_ac_qualification.py \
  scripts/analyze_z4c_candidate_ac_qualification.py
python3 scripts/analyze_z4c_candidate_ac_qualification.py \
  --run-root /home/hzhu/athenak-candidate-ac-evidence-r3-20260813 \
  --output /tmp/athenak_candidate_ac_compact_final6 \
  --preflight-root /home/hzhu/athenak-candidate-ac-evidence-20260813 \
  --preflight-root /home/hzhu/athenak-candidate-ac-evidence-r2-20260813 \
  --validation-root default_equivalence=/tmp/athena_standard_equiv.QuxhFT \
  --validation-root parser_smoke=/tmp/athena_candidate_ac_smoke.wlsBn2 \
  --validation-root candidate_c_cycles=/tmp/athena_candidate_c_cycles.4jZ4DV \
  --validation-root terminal_horizon=/tmp/athena_candidate_c_final_ah.IQrxZZ \
  --validation-root restart_dt_reset=/tmp/athena_restart_dt_reset.bhfBMs
diff -qr /tmp/athenak_candidate_ac_compact_final5 \
  /tmp/athenak_candidate_ac_compact_final6
git diff --check
```

Results: both build targets passed; the focused gauge oracle passed; both Python files
compiled; two independent analyzer outputs were byte-identical including plots and hash
inventories; `git diff --check` passed. The first attempted oracle path under
`build/.../tst/unit/` was wrong and returned shell status 127; the actual built executable
is at the build root and passed immediately afterward. No evolution was active during
post-campaign validation.

Analyzer terminal result: 62 outcomes, 61 complete, one failed, 244 verified cumulative-
artifact prefix hashes, 6,424 common-grid profile difference records, and one disclosed
failed-outcome stderr hash drift. All completed-outcome segment artifacts and all promoted
checkpoint hashes validated.
