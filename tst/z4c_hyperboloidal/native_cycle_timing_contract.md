# Complete-cycle timing of the frozen native executable

This test-only harness is preparation for later performance tests. No GPU timing
is admitted until the required 3551-step reproduction passes and the parent
authorizes a specific bounded run. CPU validation timings establish plumbing
only. The scientific observer executable is not a timing executable.

Build an exactly reversible insertion overlay of `src/driver/driver.cpp`, using
that translation unit's actual CMake compiler, FP/device flags and include paths.
Replace only its object in the original link command. Keep the original main,
all numerical objects, source tree and executable unchanged. Hash the compiler
commands, overlay/inverse proof, original object, all linked inputs, dependencies,
header, contract and resulting executable. Never guess `/usr/bin/c++` or amend
the original FP flags. Build with `make -j64` in a new isolated directory.
Restore the original translation unit's implicit quoted-include directory with
an explicit `-iquote` after relocation, so `driver.hpp` still resolves identically.

Configure positive warmup steps, steps per window and repeated window count in
the environment; require total steps to equal the normal driver's cycle limit
minus its loaded initial cycle. A window includes complete native RK4 cycles:
all four stages, accepted-state work, clock increments, the native output check,
and the native post-cycle `NewTimeStep`/STS bookkeeping. The ordinary main and
driver retain their argument parsing, restart, stopping and finalize semantics.
Do not change dt, physics, grid, native equal weights, task dependencies or state.
Reject adaptive meshes, non-classical RK4, wall-clock stopping and watchdog mode
for this fixed-workload harness.

At the first cycle of each window, fence the device, then execute one MPI barrier
and record a local monotonic start time. At the last cycle, after native
post-cycle timestep bookkeeping, fence the device and record local elapsed time
immediately. Only then reduce elapsed times with MPI_MAX. These boundary costs
are explicit: the ending device fence is included; the starting fence/barrier
and ending MPI reduction are excluded. There are no added per-stage or per-step
fences, collectives, device reductions, timing samples or file writes inside a
window. Intermediate callbacks perform host-only consistency checks. Windows
remain separate; do not sum rank times or average rank elapsed times.

Keep ordinary output outside timed windows. For a saved step3551 restart, retain
the existing next common diagnostic cadence (355 further steps) and cap the
entire warmup-plus-window run at 150 steps. A preflight reproduces the native
float-time/dcycle output predicate for every prospective cycle and conservatively
requires the whole short run, including warmup, to have no scheduled history,
horizon, snapshot or restart write. A callback at the
actual native write site records any unexpected timed output as a failure,
without suppressing the write. Set only the console cadence `time/ndiag` high
enough that no timed cycle prints; this nonphysical output override is recorded.
The harness never changes output cadence or stopping controls itself. The
original final outputs still run after all windows and are timed separately.

Report initial/final cycles and times, exact dt, backend, ranks, native block
count, actual global evolved bulk/scri point counts and complete RK4 steps.
Bulk cell updates are evolved bulk points multiplied by complete steps, never
four times this count. For each window retain its cycle/time bounds, maximum
rank elapsed seconds, seconds per complete step and bulk updates per second.
Report warmup separately. Report startup from the original main's wall timer
(which starts after Kokkos/MPI and argument setup) through completed driver
initialization, and report final output/finalization separately. Do not relabel
that startup interval as total process startup; an external launcher may record
whole-process elapsed time separately.

Reject primary timing when a Kokkos profiling library is loaded or a profiling
environment is requested. Retain the checked environment and profiling-library
status. Profiling runs, if later authorized separately, cannot supply primary
throughput. At 24 versus 96 ranks use the same timing executable, checkpoint,
grid, dt, warmup and window sizes. Compute strong-scaling efficiency from the
maximum-rank window times as T2/(4*T8), alongside absolute throughput; no result
is established by the harness alone.

Before GPU use, run an eight-step CPU equivalence case with warmup2 and two
windows of3. Compare the full native 31-field/eight-record initial/final payloads
bitwise to the existing normal executable, and compare ordinary history, horizon
summary and full shape files byte-for-byte. Validate exactly two three-step
windows and complete eight-step clocks, and reject overlapping output/profiling
configuration in explicit negative controls. A successful timing record is not
a substitute for these state comparisons or the scientific reproduction gates.
