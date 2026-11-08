# Original-control endpoint diagnostics using unchanged v8

`legacy_n12_endpoint_diagnostics SEED_DIRECTORY ACCEPTED_DIRECTORY NEW_OUTPUT`
runs on24ranks and does not evolve or time a step. It uses the unchanged v8
operator, sampler and FastFlow helper. The original-control trajectory and
performance samples remain produced by the immutable494ec8 numerical source.
This postprocessor is measurement infrastructure, not a replacement control.

The accepted directory contains `endpoint-restart.bin` written by the original
control before its diagnostic RHS, plus48 accepted-final bulk/scri trace files.
The seed directory retains the exact original2486 `run.in` and configuration,
including the saved native2486 FastFlow monopole. Output2556 time and fixed dt
must match exactly. The new process loads the endpoint SP1 through unchanged
v8 LoadCheckpoint, reads every keyed trace row, requires post-load all31 fields
bitwise equal to the accepted control, and restores all8 accepted chi-record
values bitwise. A closure-induced difference is a failure; fields are not
silently replaced or accepted with a new tolerance.

Before the diagnostic RHS it writes its own accepted-final snapshots. The
existing qualified `Diagnose` helper evaluates full, full-harmonic exterior
and fixed-r>.94 constraints, compatibility, compact clock and CFL, then writes
`horizon.json` with all169 coefficients. An additional bitwise comparison
requires every chi record to remain unchanged after diagnostics; step/time also
must remain2556. No statement is made that the original diagnostic RHS leaves
all dependent field/workspace values untouched. Its accepted input is captured
before that phase.

`results.json` schema `legacy-n12-endpoint-diagnostics-v1` reports
`post_load_fields_bitwise_control_endpoint`, `post_load_records_bitwise_control_endpoint`,
`diagnostics_preserved_chi_records_bitwise`, `evolution_steps=0`,
`timing_measurement=false`, full/exterior raw10 sums, fixed outer5 sums and
compatibility/clock/CFL. Its horizon and diagnostic values must be bound to the
exact control endpoint and compared with the control's independent full/fixed
norms and the native final diagnostics under the already adopted gates.

The new TU includes the hash-pinned `legacy_n12_benchmark.cpp` with its main
renamed. Only its helper functions are called; the benchmark entry and its
Step loops are never invoked. Neither that shared file nor any numerical
source is edited. The new builder reuses the frozen v8 archived compiler/FP/link
recipe, replaces only its entry object/output and records all source/dependency
hashes. Donor owns actual make-j64 and allocations. No GPU run or numerical
qualification is implied by source preparation.

The builder's stdlib `inputs --seed ... --accepted ... --output inputs.json`
mode binds the qualified seed manifest and each seed output, the control's
results and endpoint SP1, and all48 accepted traces with complete point counts.
`verify-inputs --manifest inputs.json` must pass before and after the diagnostic
process; retain both verification logs. The schema is
`legacy-n12-endpoint-inputs-v1`, with absolute `files` paths mapped to SHA256,
seed/control directory paths, exact cycle/time/dt, ranks and point counts.
The consumer also binds the actual executable/build/command/exit/terminal job.
