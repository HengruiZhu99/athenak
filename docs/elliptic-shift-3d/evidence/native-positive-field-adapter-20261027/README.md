# Native positive-field adapter qualification

`NativeCompositeFields` appends `positive_storage=0` after the existing
`chi_psi_power` constructor argument. Existing calls/defaults and both legacy
wormhole W-factor expressions are preserved. Positive modes require
`puncture=true`; unsupported modes reject.

Mode1 stores chi/alpha directly, mode2 stores log chi and log(alpha/Aref), and
mode3 stores their squares. Aref=(1+|x|²)/2 is stationary. PositiveFieldChart
provides value/rate conversion without floors. All positive modes retain direct
signed Atilde, Theta, Lambda and telegraph flux, with only Khat's existing +3
stored offset. Metric diagonal offsets and algebraic beta/rate conventions are
unchanged. The reconstructed variables occupy native conformal evolution slots;
this adapter adds no extra Omega factors or physical-ADM conversion.

Build only this target with two jobs, then run its existing CTest:

    cmake --build build-shift-3d --target elliptic_shift_3d_native_fields_tests -j2
    ctest --test-dir build-shift-3d -j2 --output-on-failure -R '^elliptic_shift_3d_native_field_layout$'

The recorded test passes seven chart cases over twenty mapped/reordered blocks
and the actual 25-field native layout. It independently checks positive field
values, signed fields, analytic rate conversion, centered finite-difference
physical rates, and exact preservation of ghosts/unmapped blocks for both values
and rates. Tiny chi=1e-80 and alpha=2e-70 remain representable without floors.
Invalid direct-zero, log-endpoint and negative-square storage, bad modes, bad
layouts and wrong chi convention reject explicitly.

Observed errors: roundtrip2.23e-16, rateFD1.15e-11, independent positive state/rate
zero, tiny relative1.16e-14, untouched ghosts/blocks zero. Exact source/header
hashes match before and after CTest. The first compilation failed on four
unqualified test enum references (eight diagnostics); namespace qualification
fixed this test-only issue. The manifest retains both compile outcomes.

This qualifies field conversion only. The actual native owner remains configured
for Minkowski/default0 until separately changed: Scatter has regular-chart ghost
assumptions, diagnostics have default Unpack/clock assumptions, and constructor
and checkpoint metadata enforce the existing Minkowski setup. No native
wormhole evolution, AMR, MPI or coupled scientific acceptance is established here.
