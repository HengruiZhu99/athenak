# Independent snapshot acquisition verification

All three diagnostic acquisitions completed and passed their 96 acquisition
gates on Aurora job `8983924`. Independent [verification](verification.json)
checks all 12 worker-bound artifacts, 2,613 source/Kokkos blobs, frozen build
and admission, twelve distinct GPU tiles, terminal F/exit0 and allocation
release. Every checkpoint remains **scientifically unadmitted**. The original
V4 stability failure is unchanged.

| Coordinate time | Steps | Maximum volume-state deviation | H L2 | M L2 | Native wall seconds |
|---|---:|---:|---:|---:|---:|
| 0.25 | 500 | 1.4450e-7 | 7.2967e-12 | 6.2560e-10 | 65.34 |
| 0.30 | 600 | 5.3594e-6 | 2.5207e-10 | 2.3157e-8 | 77.16 |
| 0.35 | 700 | 2.0670e-4 | 9.0060e-9 | 8.6748e-7 | 89.41 |

All three exceed the unchanged state limit of 1e-8. The final capture also
exceeds the unchanged momentum limit of 1e-7; Hamiltonian norms remain below
1e-7. Recorded history first detects the state violation at approximately
0.25 and the momentum violation at 0.35; these are recorded sample times,
not interpolated crossing times. Null residual maxima remain below 1.23e-15.
Norms use normalized compactified Cartesian coordinate volume.

The worker took 257.07 seconds. Native wall seconds include setup and
diagnostics; they are not GPU-kernel timings. Campaign-home usage at completion
was 1,955,649,304 bytes. All three headers are V4, 88 bytes, epsilon 2, degree 7,
with exactly 500/600/700 accepted steps and current diagnostics at the requested
endpoint. All four arrays of each checkpoint are finite and hash-recorded.
The native admitted flag is true because each run completed without exception;
this does not qualify scientific continuation.

Small metadata, reports, histories and logs are retained under `raw/`.
`terminal-whitelist.json` preserves terminal facts and the full collection
hash without retaining scheduler environment/account metadata. The three
39,362,648-byte checkpoints remain in ignored local raw output; no payloads
are committed here. Seven offline verification-logic tests passed, including
nonfinite metadata serialization without erasing failure flags. Full field
and mode analysis is separate from this acquisition audit.

The three acquisitions use the already failed V4 scheme. Their completion
would provide diagnostic fields, not a new stability result. The verifier
always emits `scientific_not_admitted=true`,
`scientific_restart_admitted=false` and `stability_qualification=false`,
regardless of the native checkpoint flag. Its acquisition outcome is named
`acquisition_passed`; it does not create an admission record.

`verify.py` checks the frozen source and native executable, exact reviewed
plan, epsilon-two preflight admission, actual twelve-rank launch and twelve
GPU tiles, all artifacts, all 96 acquisition gates and strict 88-byte V4
checkpoint headers. It reads and checks finite values in all four checkpoint
arrays, including RK compensation. Native clock/step/header values must match
the bound reports. The original equilibrium thresholds remain independently
reported: state <=1e-8 and H/M <=1e-7, both at each endpoint and at every
recorded history sample. Missing or failed acquisitions remain explicit.

Run from the repository root after the coordinator has downloaded and released
the complete allocation. The existing frozen source/Kokkos Git objects and
portable preflight compile evidence are required; no native test binary is run:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/linear-window-dissipative-v2/acquisition/verify.py \
  --output output/hyperbolic-core-shell/analysis/linear-window-acquisition-verification.json
```

The output path must be new. `--raw-dir` selects another complete local copy.
Only the requested JSON is written. `test_verify.py` checks that passing
reference thresholds still cannot admit a checkpoint, unchanged threshold
violations remain visible, invalid metrics fail, and endpoint/history checks
do not erase an earlier recorded violation.

The native amplitude screen covers volume fields only. Separate field analysis
must assess surface amplitudes, coordinate/weighting choices and approximation
errors before interpreting a snapshot as a linear mode. No checkpoint from
this acquisition is scientifically admitted for restart.
