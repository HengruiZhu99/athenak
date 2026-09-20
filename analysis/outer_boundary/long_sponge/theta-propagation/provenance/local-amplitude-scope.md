# Tenfold smaller direct-Theta seed

The ready long-run input is `../theta_loweta_amp1e7_coremask.athinput`, SHA256
`9a72f639936cabba73c5181d03f7cbc836673882ee091fb4cca7d3682518b99e`.
Compared with the frozen primary input, only the output basename and
`outer_sponge_test_theta_pulse_amplitude` (1e-6 → 1e-7) change. It retains
the same 384M Gaussian width, dt=3.2M, 50000M target, 512–1792M radial sponge,
kappa1=0, shift_eta=0.02, lapse damping=0.01, history core mask, and per-rank
restart/Theta/constraint output intervals 1000/64/128M.

The authorized local eight-rank preflight ran **three cycles to 9.6M** and
exited 0. All saved payload fields were finite and all ghost metrics passed
the positive-definiteness check. The executable SHA256 was
`67ff395e3c7af43425ef13fe4f0ebecef00256db9450f6ac912f5ab39620f1f8`.
The original primary zero gate is unchanged and remains applicable.

Initial checkpoint comparison includes all 25 residual Z4c fields and ghost
cells on all eight ranks. Theta equals 0.1 times the primary seed to
1.34e-16 relative to the expected peak; all other fields are bitwise equal.
For the flat initial geometry and Khat=0, physical K=2Theta,
H=(8/3)Theta² and M_i=−(4/3)∂iTheta. Thus physical H amplitude scales by
0.01 and momentum amplitude by 0.1. Measured initial core/exterior squared
integrals scale by 0.0001 for H and 0.01 for M and Theta, with relative
errors below 1.3e-14.

These are initial-data checks. No amplitude scaling is imposed on evolved
nonlinear states. A later plateau scaling as amplitude, amplitude squared,
or neither is evidence to interpret alongside complete field/constraint
diagnostics and nonzero initial incoming boundary data; it does not by itself
establish an arithmetic floor or stability.

`input-manifest.json`, `scaling-validation.json`, the initial/final checkpoint
validation JSONs, and `run/provenance.json` retain the proof. `run_gate.py`
refuses to overwrite an existing run; `check_scaling.py` reproduces the
comparison. No GPU job was submitted by these scripts.
