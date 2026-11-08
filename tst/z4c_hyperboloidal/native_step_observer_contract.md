# Read-only observer of the original native entry point

The normal native executable does not serialize its stage clock/CFL maxima.
This correctness-only executable adds an observation task to the existing
`after_stagen` list, after every pre-existing end task. It reads the stored bulk
`rhs_cfl` and the surface compatibility/clock observations; it calls no RHS,
projection, algebraic closure, horizon search or timestep update. Its device view
is const. Record all four stage rows and global maxima at every accepted step.
The reported CFL is the bulk maximum, without a `cfl_number` multiplier or a
surface-penalty contribution, matching the pinned original N12 solver.

Build from the actual native `main.cpp` compile command and original link inputs.
The generated main differs only by the observer header include, `Install` after
`Driver::Initialize`, and `Finish` after `Driver::Execute`. Save a reversible
overlay proof and hashes of the original main, header, runtime sources, linked
objects, compiler/link commands and final executable. The original numerical
objects and executable remain unchanged. All argument parsing, problem setup,
restart loading, outputs and `Driver::Finalize` remain the original code.

Before using the observer for the long correctness run, compare the observed
eight-step checkpoint against the original executable at the same decomposition:
all 31 fields and eight chi-record members must agree bitwise. Check unchanged
history/horizon cadence and data too. Exercise restart through the ordinary
`-r` path, matching all step/stage clocks and complete state to continuous
execution. A test prefix's extra final horizon search is a separately reported
output-phase effect, not permission to introduce off-cadence searches into the
common-time long run.

Compare maximum-of-four stage clock and CFL values to the unchanged original
solver's per-step CSV, using the previously adopted short/long numerical gates.
Require complete ordered stage coverage, finite nonnegative values and normal
compatibility at most `1e-7`. Completion metadata alone is not a reference
comparison. For the long run preserve all 3551 complete steps and original
common diagnostic times. Bind every segment to its observed executable and
unchanged underlying numerical object manifest.

The observer adds reductions, synchronization and file output. Its timing is
not an accepted performance measurement. Performance windows use the unchanged
normal native executable with detailed observation disabled. The clean/poison
and decomposition tests of that executable's numerical objects remain separate
evidence; observer synchronization cannot replace them.
