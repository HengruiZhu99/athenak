# Local hard-freeze implementation gate

This compact record covers the pre-Perlmutter implementation audit for the
matched continued-reference versus hard-frozen-reference discriminator.  It
is not evidence for trumpet stability or convergence.

## Results

- A RelWithDebInfo Serial build with AddressSanitizer, UndefinedBehaviorSanitizer,
  and Kokkos bounds checks completed the full Ref-GH source-unit executable.
- The new controlled-reference oracle evaluated 35 off-axis samples spanning
  five intermediate activation coordinates and seven radii.  Every temporal
  scalar jet, temporal frame/metric/connection derivative, reference-gauge
  temporal derivative, and temporal frame-motion term was exactly zero.
- A two-cycle moving-reference restart was changed to `hard_freeze` and then
  evolved for two more cycles.  Re-expression in the frozen frame changed
  `Psi/Phi` only by `1.11022e-15` while making the required finite Pi change
  `3.25627e-05`.  The post-freeze maximum norms of both
  `reference_dt_frame` and `reference_dt_connection` were exactly zero.
- The test remained finite through `t=1.77403292946998123e-02`; it is only a
  code-path gate at a deliberately tiny uniform grid.

## Exact compact output

```text
reference-GH flat covariant source unit passed: samples = 1000, max error = 6.93889e-17
reference-GH nonflat covariant source unit passed: samples = 128, max error = 3.88578e-16
reference-GH controlled hard-freeze time-derivative oracle passed exactly: samples=35 max_jet=0 max_time_invariance=0
moving_status=0
restart=rst/moving_smoke.00002.rst time=0.0088701652012898886 xi=0.0011087706501612361
hard_status=0
reference-GH hard-freeze restart reprojected at time=0.00887017 xi=0.00110877 old_xi_dot=0.125 old_xi_ddot=0 new_xi_dot=0 new_xi_ddot=0 state_change_Linf=1.11022e-15 Pi_change_Linf=3.25627e-05
reference-GH controlled Schwarzschild final: time=0.0177403 state Linf=1.00001 bad-state=0
reference_dt_frame maximum=0
reference_dt_connection maximum=0
```

## Local-only full logs and hashes

The disposable full logs were not committed.  They were retained at the
following local paths during this checkpoint:

```text
da38d9837c9597a4a411f5c1003debdb766bf6d1f6d05b795f17c5227aa21b33  /tmp/refgh-hard-freeze-sanitize.hL1vSG/source_unit.log
3075925410846154dc38cb67a4ec143047079407be0d13719ad725fc11ca5912  /tmp/refgh-hard-freeze-restart.4rqJkK/moving.log
900f143c3a23802b93621c49b258f0f17ef639dcbceee041f4b415fafb9cc138  /tmp/refgh-hard-freeze-restart.4rqJkK/hard.log
a268b33cae7bd96e0db56a4e8ac1f8c905fb6cef89c2a9f5100f8dd1e8a6f4c9  build-relative-damped-sanitize/src/athena
```

The sanitizer invocation used
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
