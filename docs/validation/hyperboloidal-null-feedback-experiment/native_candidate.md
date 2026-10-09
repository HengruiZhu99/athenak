# Ignored native null-feedback experiment

This directory holds a private native AthenaK experiment of the audited research
candidate. Tracked `src` and CMake source are unchanged from
`27c19d20696ea6dd4704032c51dfd026218f64f2`. The private target's forced include
loads the production gauge before defining call-site aliases, then redirects
physical-P layer calls through `research::Gauge`. The wrapper explicitly
assembles every candidate `pole.beta[i]/Omega`. Only the `athena` target receives
the injection; Kokkos and production sources remain unchanged.

The candidate adds the rederived spatial source in an independent smooth collar
`.85 < r < .95`, where the gauge's harmonic weight is exactly one. Sigma is 5.
`Nraw-Nraw_ref` is evaluated by the factored geometric/null difference from the
previous scratch audit. The live speed bound in `z4c_newdt.cpp` is unchanged.
Inputs use a=.5, geometry layer .2–.8, gauge .45–.85, symmetric quadratic
continuation, damping 5, physical-P lapse xi=1.5, PoleCFL=.03, spatial order 4,
RK3, N24, width .5, angular lapse/shift pulse .1/.02. No Omega floor, Theta
falloff assignment or variable weighting is introduced.

`native-build-receipt.json` records source hashes before/after compilation,
compiler identity, all 182 translation-unit commands, target flags, overlay
hashes, the executable hash and the unchanged timestep-source hash.
`overlay-audit-receipt.json` records exact compiled overlay full20 equality with
the prior scratch candidate at a=.5,.75,1,2 and all 288 principal cases.
`native-experiment-report.json` checks active native arrays at each output:
finite fields, positive lapse/chi and SPD physical metric, and retains native
history and constraint-budget diagnostics. Individual phase directories retain
immutable binaries, native output arrays, restart files, inputs and logs.

Reproduce using the recorded CMake configure/build commands, then an interpreter
with numpy and sympy:

```sh
python run_overlay_audit.py
python native_phase.py reference
python native_phase.py short
python native_phase.py half
python inspect_native.py reference short half
# Only after the previous regular-field gates pass:
python native_phase.py long
python inspect_native.py reference short half long
```

Run phase directories must be fresh because the standard harness preserves an
immutable executable. The first three runs reached t=.05/.02/.5 respectively.
The stationary reference's maximum dumped active-field difference is 2.10e-14;
its native final H/M/Z are 8.06e-14/3.22e-14/1.34e-15. The t=.5 pulse has
H/M/Z=.154861/.219269/.081520 and positive fields across all 21 snapshots.
Against the same-step production source-off control
.155697/.238606/.093403, the reductions are .54%/8.10%/12.72%. These constraints
remain too large for acceptance. The t=2 extension is a duration increase with
the same initial pulse, candidate and parameters.

The off-constraint full-collar source identity remains
`Box(Omega)=Omega*Wref+2*Zbar^i*Omega_i+sigma*(Nraw-Nraw_ref)/Omega`.
The source preserves O(Omega) only with the appropriate null and temporal-Z
compatibility; it does not impose them on arbitrary live fields. In particular,
reference geometry/gauge with `P-Pref=Omega*.01` still gives
`Omega*Qdot -> .02` and `ThetaDot -> -.02`. Frozen pole stability, a complete
principal symbol, exact reference cancellation and these finite native runs
supply no full nonlinear regularity-closure or stability proof.
