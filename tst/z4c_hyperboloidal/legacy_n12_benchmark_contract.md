# Matched two-node N12 predecessor benchmark

This is benchmark infrastructure, not an optimization or performance result.
Use 24 GPU-tile ranks on two Aurora nodes only. No further long trajectory or
eight-node run is required. Frozen production, native baseline and predecessor
sources remain unchanged.

## Common initial state

The source is the independently admitted native cycle2486 checkpoint
`dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec`.
`n12_benchmark_seed.py` requires its primary/independent admissions and the
verified actual-Aurora native/legacy mapping. It copies the historical SP1
cycle2486 header/topology/physics and compensation arrays verbatim, replacing
only the 31 bulk and surface fields with the native checkpoint values. This
does not use the slightly different historical fields as the benchmark seed.
The native eight-member chi histories are copied into two separate files; SP1
does not store these histories. The legacy wrapper restores them before any
step and compares every actual initial field/history bit to the converted seed.
Native restart initialization reapplies algebraic and scri projections. The
conversion proves equality to the source checkpoint, not automatically to that
postinitialization native state. A bounded native readout must establish its
actual pre-warmup equivalence; any derived-value roundoff uses the unchanged
short gates and is reported separately from source-file bitwise equality.

Both spherical-policy RK branches return before the compensated formula; final
acceptance writes the next compensation as zero. The preserved SP1 compensation
bytes therefore do not change this formulation's continuation. This is a
source-level property to retain in provenance, not permission to alter another
formulation's restart state.

The legacy finder receives every native persisted FastFlow parameter, including
`last_a0_0`, and performs only the final search. Native restart initialization
also suppresses an initial search. The immutable original campaign entry point
performs initial and final diagnostic RHS evaluations and fixed-radius constraint
diagnostics; it does not run FastFlow. Its use is an additional field-only
wrapper-validation control, with those extra RHS/closure phases preserved. It
does not supply a matched horizon-finder schedule or seeded per-cell chi history.

## Reference selection and numerical qualification

The archived v3, v6 and v8 implementations are all contenders. v3 has the fastest
single two-node N24 screening result; v8 has repeated final/poison evidence.
Those N24 results cannot select the N12 reference. Build the same new benchmark
entry point against each unchanged numerical source with its archived compiler,
floating-point flags, Kokkos/MPI libraries and macro choices. Also build it
against the immutable original mathematical control. Record all source, binary,
command, dependency and library hashes before and after.

The selected implementations retain their previously qualified environment:
small workgroups for v3/v6/v8; the separate bulk stream for v3/v6/v8; host surface
staging only where previously qualified (v8). Actual archived launch evidence
is authoritative if it identifies a different setting. Presence-based flags
must be unset when disabled, including sparse-workspace poisoning.

The legacy initial 7,137,936 field values and 1,842,048 history values must match
the common seed bitwise; native postinitialization scope is stated above.
Compare complete final bulk/surface fields and all eight histories
across native controls and predecessor variants. Prefer exact equality when
arithmetic order is preserved. Otherwise use the existing short gates:
`1e-12 + 5e-10*abs(reference)` for fields, and
`1e-14 + 5e-10*abs(reference)` for chi. Seeded histories require exact event and
nonfinite counts and exact first-floor times. Their two minima and correction
amplitudes/sums use that existing chi gate: they have chi units, and the old
zero-floor-only history comparator is inapplicable to the nonzero cycle2486
seed. This applicability extension was fixed before any benchmark GPU result;
bitwise status and all maximum differences remain reported. No tolerances are
chosen from these results. Endpoint boundary compatibility must be finite and
at most `1e-7`; clock/CFL are compared, not forced to vanish. Complete horizon
coverage, all 169 harmonic coefficients, full and horizon-exterior constraint
diagnostics, and areal radius/mass use the existing diagnostic gates:
RMS `5e-10 + 1e-6*abs(reference)` and areal radius/irreducible mass in M units
`max(1e-10,1e-7*abs(reference))`. Harmonic completeness is required; compact
radius/individual coefficient changes are reported without introducing a new
solver-tolerance gate. Preserve
all initial and accepted-vs-diagnostic-phase distinctions.

The wrapper writes accepted-final 31+8 data before its untimed diagnostic RHS.
It then uses the predecessor's unchanged physical constraint sampler, full
harmonic horizon mask and FastFlow solver. It verifies diagnostics leave all
chi histories unchanged. A fixed `r>.94` diagnostic is supplementary; it does
not replace the complete horizon exterior. Original kernel validity checks
remain enabled throughout. A separate sparse-poison correctness run is excluded
from performance results.

The rebuilt original-source control predates the harmonic-mask diagnostic API.
Its measurement entry therefore retains the original full/fixed-cut diagnostics
and writes an accepted endpoint SP1 after the snapshot, outside timing. A
separate, zero-step v8 diagnostic process loads that exact endpoint, restores
the saved eight-member histories, and verifies all31+8 bits before its diagnostic
RHS. It uses the same cycle2486 horizon seed for a final-only search. Original
and postprocessor full/fixed RMS values are cross-checked with the existing
same-checkpoint gate `5e-11*(1+abs(reference))`; the postprocessor then supplies
the full-harmonic exterior and horizon comparisons above. This dependency is
described in `legacy_n12_endpoint_diagnostics_contract.md`. Neither the original
timed Step nor the three successfully built optimized packages is changed.

## Timing

All runs start at cycle2486, execute 10 warmup steps, then three windows of 20
complete RK4 steps, ending at cycle2556. The original fixed timestep is
`0.00046944203735617445`, double precision, N12/106 blocks, 229392 evolved bulk
points and 864 scri points, KO=.1, chi floor=1e-10, constant shift speed and fixed
origin. Native block weights remain one. Native next scheduled output is beyond
the benchmark endpoint; diagnostic/output work is outside all timed windows.

Each window starts after a global Kokkos fence and MPI barrier. It stops after
the end fence and local timestamp; MPI_MAX and per-rank attribution occur only
after that timestamp. There are no newly inserted per-step fences, reductions,
diagnostics, timers or observers in the legacy wrapper. Its detailed phase
timers remain disabled for the entire run. Loaded Kokkos profiling libraries
and profiling environment variables reject primary timing. Existing numerical
Step synchronization and validity checks remain untouched.

Report all three raw maximum-rank window times, divided by 20 for complete-step
time, and their median/range. The initial 70-cycle suite qualifies the machinery
and screens references; acceptance of parity/improvement needs sufficient
independent repeats and order reversal to separate differences from run noise.
Cell updates count 229392 per complete RK4 step, never four times that number.
Startup/seed verification, warmup, accepted-state output and final diagnostics
are reported separately and are not cross-implementation setup speed claims.

Native normal, timed-normal and timed-observer controls independently establish
timing-wrapper equivalence and observer cost. The parity target is the fastest
qualified predecessor's unprofiled median versus native unprofiled complete-step
time, never the stage observer or a historical N24/48-rank measurement. Failed
correctness, missing coverage or incomplete jobs withhold performance admission.
