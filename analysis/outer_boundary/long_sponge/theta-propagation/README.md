# Direct-Theta propagation and incoming-state diagnostics

The final collection contains **783 complete snapshots through50000M for each of three GPU controls**: job8842248 primary and lapse-damping0.1, and job8842283 at tenfold smaller pulse amplitude. All eight-rank binary cohorts pass finite payload, matching headers/time/cycle, unique logical block and full64³ coverage checks. There are no exclusions. The original373-snapshot collection through23808M is preserved in `archive-23808/`.

Read **`GPU_FINAL.md`** for the final spatial/temporal results, incoming-state decomposition, amplitude comparison and physical-constraint/gauge limitations. The previous rapid Theta runaway is absent in these tests, but small Hamiltonian and gauge drifts remain. The initial Gaussian boundary trace and an acquired nonlinear trace both contribute to late plateaus; neither finite completion nor a plateau proves universal stability.

## Files

- `theta-radial-spacetime.png/pdf`: signed radial mean and RMS for all three cases.
- `theta-regional-rms.png/pdf`: protected-core, ramp and outer-plateau coordinate-volume RMS.
- `incoming-c1-gpu.png/pdf`: checkpoint samples of C1 and acquired increments with amplitude scaling; these are state functions, not physical constraint norms.
- `incoming-gpu-final/`: all scalar terms at fixed rank7 face checkpoints and their hashes.
- `INCOMING_TRACE.md`: initial code/local-gate audit; subsequent compact evidence is in `../compact-theta/trace-comparison.md`.
- Each case's `manifest.json` and `profiles.npz`: compact validated data, with hashes for all source rank files.
- `source-provenance.json`, `package-manifest.json`, `provenance/`: source/executable/input/check hashes. Raw simulation files are not copied here.

## Definitions and limits

Binary Theta is float32, promoted to float64 for accumulation. Radial and regional means/RMS use coordinate volume dx³ because the Theta-only binary dumps contain no metric. Proper-volume H/M/Q/Theta diagnostics come from the separate history/checkpoint package in `../gpu/results/`. The core is r≤512M, ramp512<r<1792M and plateau r≥1792M; the separate face-distance≤256M band overlaps them. Radial bins are64M wide. Beyond2048M each bin contains only the cube's part of that shell.

No reflection coefficient, first-injection location or long-term stability theorem follows from these profiles. The fixed-face checker implements the source's active-only D2 stencil and full conformal normal for this uniform evolved-Minkowski fixture. It supports global32³/64³ with eight16³/32³ blocks, derives dimensions/spacing from the header, and reads seed parameters from input metadata. Its zero-initial-trace ratios are null; absolute changes remain available. Post-RK projection and moving coefficients prevent treating a weighted zero-rate closure as an exact nonlinear state invariant.

## Portable checks and plotting

Run with Python3 plus NumPy; plotting additionally needs Matplotlib:

```sh
python3 test_reader.py
python3 test_trace_initial.py
python3 plot.py .
python3 plot_gpu_traces.py .
```

The binary-reader test verifies constant-field coverage and rejects truncation, time/cycle mismatch, duplicate blocks and nonfinite payloads. The trace test creates temporary synthetic checkpoints for both supported resolutions and checks an independent analytic Gaussian value, C1/C2/gauge coefficients, archived initial values and rejection of a curved background. It requires no raw archived checkpoint.

## Refresh

`refresh.py` runs read-only simulation-output aggregation remotely, writing only a separate analysis directory, then fetches compact profiles/manifests. It caches valid unchanged cohorts using size and nanosecond mtime and preserves source-file hashes. It verifies an ALCF hostname and expected Flare project directory before writes. Supply a currently authorized Aurora control socket; no jobs are submitted or modified.

```sh
python3 refresh.py --host hzhu@aurora.alcf.anl.gov --socket CURRENT_SOCKET \
  --run /lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review/runs/long_theta_loweta_8842248 \
  --remote-analysis /lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review/long-sponge-study/theta-propagation-8842248 \
  --output LOCAL_OUTPUT
```

For D use parent run `long_theta_amp_8842283`, a separate analysis directory,
and `--case theta_amplitude=theta_loweta_amp1e7_coremask`. For fixed-face measurements, see the command/prefixes in `GPU_FINAL.md`. Absolute paths stored in historical manifests are provenance, not local script dependencies.
