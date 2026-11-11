# Circumbinary diagnostics: implementation and validation

Date: 2026-11-11. Primary base `origin/circumbinary-cyclic-zoom` was fetched and verified
as f48f3126c1096d4df0d3f72909f9075969c4efa6. The manuscript worktree is based on
475dadceb44d13ca7e8ad289934a9c336721d891. Original checkouts and unrelated work were
preserved. Both isolated worktrees use `codex/circumbinary-diagnostics`.

## Review units and files

| Unit | Main changes |
|---|---|
| A: surface conventions | `src/pgen/dynbbh.cpp`, `src/utils/surface_grid.*`, `flux_generalized.*`, `circumbinary_geometry.hpp`; Bondi/restart registration, transfer spheres retained, distinct component Kerr surfaces, relative currents, full labels, ownership/validity |
| B: accepted-step state | `src/driver/driver.cpp`, `src/pgen/pgen.hpp`, `src/outputs/restart.cpp`, `circumbinary_accumulator.hpp`; RK recurrence, qualification/coverage epochs, read-only output, versioned header state, reset provenance and post-zoom timestep restoration |
| C: regional budgets and final hardening | `circumbinary_budget.*`, `valencia_sources.hpp`, `valencia_diagnostics.hpp`, `valencia_surface_flux.hpp`, evolution/recovery/source/remap call sites; conserved J first moments, complete boundary orientation, actual mutation reductions and continuum residuals |
| Analysis compatibility | `src/outputs/history.cpp`, `outputs.hpp`, `vis/python/athena_read.py`, `scripts/{disk_diagnostics,post_process_cbd,cbd_history}.py`; complete labels, restart segments, explicit frames and correction selection |
| Validation | `tst/diagnostics/`, this report, companion definitions document and CPU measurement JSON |

Unit C also carries final cross-unit validation, formatting and lifecycle hardening
(reset provenance and the measured post-zoom restart timestep issue). Its tests
exercise the final combined implementation.

The manuscript repository adds only `analysis/native_diagnostics/{cbd_history.py,
test_cbd_history.py,README.md}` (commit d2ed0ac). The adapter is shared verbatim with
the primary repository. No manuscript claims, numerical results or normalization
reference data were rewritten. The actual historical `cbdlib.py`/`budget.py` consumer
chain and continuation extraction scripts were inspected before defining the adapter.

## Scientific contract

The companion `circumbinary-diagnostics.md` gives the full equations, labels, flags,
configuration restrictions and parameter choices. Key distinctions are:

* Legacy mdot/edot remain inward-positive; legacy pdot/ldot remain outward-positive,
  with global-origin angular momentum and fixed-surface terms. Negate the latter
  two when comparing with manuscript inward F_P/F_J.
* Appended `*_in_moving_*` channels already contain `-(f-q w)` contracted once with
  the outward measure. M0 has net, nonnegative inward and nonnegative outward rates.
  Eeu is Eulerian energy; Et is coordinate-energy flux, without an energy closure.
* H1/H2 remain configured coordinate transfer spheres. BH1_Kerr/BH2_Kerr are modeled
  component Kerr level sets, not located apparent/event horizons. Their qualified
  cumulative rest mass is neither prescribed gravitational-mass growth nor excision
  removal. They never replace moving-5M transfer-rate normalization denominators.
* Phi_B remains raw half-unsigned Heaviside-Lorentz flux. Analysis retains sqrt(4 pi)
  and the separate per-hole versus fixed-total normalization choices.
* Region J uses conserved densitized momentum and Cartesian first moments, without
  another sqrt(gamma). Budgets are inferred prescribed-field continuum balances,
  not physical BH spin-growth measurements or exact numerical cut-cell closure.
* A whole incomplete surface is NaN. Partial mass/area are explicitly named. Reset,
  frozen, underresolved, nonfinite and ownership failures remain explicit. Global
  reductions occur in driver order outside retryable tasks; globally replicated
  histories contribute only on rank zero.

Histories preserve output-time transfer geometry and expose state, geometry and
actual metric-setter times separately. Accepted-step integration uses the documented
actual stored metric epoch under the existing prescribed-ADM stage timing. RK weights do not
remove its first-order limitation for time-dependent geometry. No evolution metric
clock or physical parameters were changed. A narrowly scoped restart fix retains the
already selected post-zoom next timestep while honoring any smaller fresh limit.

State uses history schema 2 and restart state format 2 in the existing ParameterInput
header, with a compact configuration/trajectory identity, qualified mass, durations,
committed cycle/time, source/jump scalars and surface reset-provenance flags. Old
checkpoints begin a new explicit epoch. Changed configuration needs a consumed
`flux_new_epoch` request. Repeated output/checkpoint/finalization does not integrate.
A genuine startup while frozen can add an unavailable recovery-event observation;
it adds no fabricated finite jump or mass.

## Commands and observed results

The tested build was Release, AppleClang 21, Kokkos 4.6.1 Serial, MPI enabled,
double precision, Apple M5 Pro (arm64). Actual runs used one and two local MPI ranks,
including one-to-two-rank restart. No GPU or multi-node run is claimed.

```sh
cmake -S . -B build-diag -DPROBLEM=dynbbh -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-diag -j 8
python3 tst/diagnostics/run_helpers.py build-diag
ATHENA_DIAGNOSTICS_EXE="$PWD/build-diag/src/athena" \
ATHENA_DIAGNOSTICS_MPI=1 ATHENA_DIAGNOSTICS_BASE_EXE=/tmp/athenak-diagnostics-base \
/tmp/athenak-diagnostics-python/bin/python -m unittest discover -s tst/diagnostics -p 'test_*.py'
```

The temporary Python environment supplies NumPy/h5py; SciPy is needed only by the
full snapshot-interpolation path. The base executable was built from the audited
revision before patching, for the old-checkpoint test. Omitting its environment
variable skips that optional baseline comparison. The complete local suite passed
**19 tests**, and the separately compiled production-helper executable passed.
After the final geometry-formatting cleanup, the build, helper executable and both
metric/clock executable tests were checked again. `git diff --check` passed. The
repository C++ linter and custom whitespace checks reported zero findings on changed
lines; unchanged source still has pre-existing lint findings, so this is not a full
repository style-pass claim. Retained stdout is in
[circumbinary-diagnostics-checks.txt](circumbinary-diagnostics-checks.txt).
The manuscript adapter suite passed **3 tests**:

```sh
python3 -m unittest discover -s analysis/native_diagnostics -p 'test_*.py'
```

The executable suite covers healthy and unavailable mass intervals, history cadence,
repeated finalization/checkpointing, old checkpoints, changed configuration/new epochs,
rank changes, an AMR interface, partial spheres, frozen visits, checkpoints immediately
before and after pending zoom events, regional/source-jump restart state, injected
actual conserved changes, native schema parsing and Python/C++ metric comparison.
One additional manufactured checkpoint injects reset provenance without changing an
evolved variable; repeated checkpoints retain it until a new evolved state.

The source injection uses the existing uniform acceleration operator for one tiny
step, **only as an accounting test**. It independently reconstructs the near-region
moment of primitive density from the actual binary dump and checks g_y dt times that
moment. RK2 is checked against its two final half-weights, using the RK1 endpoint as
its stage-1 state. Binary output is float32, so comparison uses 1e-7 relative tolerance.
The RK1 expected change is about 1.7153863e-6 and differs from the measured change by
3.5e-8 relatively. This is not a physical GR acceleration prescription validation.

The production-header tests separately cover:

* zero-spin, tilted/spinning and boosted Kerr level sets and map ordering;
* isolated-Kerr proper areas, and a superposed metric whose area is not forced to
  the isolated value;
* pointwise translating comoving flow (including both gross transport magnitudes),
  static-fluid expansion, outward flow and tangential reparameterization;
* equivalent rotating/inertial currents, coordinate-energy identity and vector rotation;
* raw magnetic flux with a nonunit determinant and the one-hemisphere factor;
* rotating-flat torque cancellation, nghost 2/3/4 and inactive-derivative access guards;
* unchanged coordinate-source arithmetic, RK1/2/3 auxiliary quadrature, stage correction
  propagation, repeated commits and gaps;
* disjoint regions/all-boundary signs and a conservative-remap example that changes
  the angular first moment while preserving total momentum.

## Separate convergence checks

These are three different checks, not a claim of physical evolution convergence.

1. **Surface quadrature, manufactured isolated Kerr.** At ntheta=40,80,160 (nphi twice
   ntheta), the largest tested relative area errors were 2.654e-4, 6.633e-5, 1.658e-5.
   Static-fluid expanding-surface mass errors were 2.571e-4, 6.426e-5, 1.606e-5.
2. **Temporal manufactured data.** RK2's quadratic-rate integral has the expected
   1/(6 N^2) bias over unit time; RK3 integrates the tested quadratic exactly.
   The actual uniform-mesh RK2 setter ordering samples the previous stored epoch
   first and current mesh epoch second; its linear-rate error is 1/N-1/(2N^2).
   This is first order. An actual inertial run verifies the separate clock columns.
3. **Geometric regional budget, manufactured steady current.** With
   Wv=(a x-b y,a y+b x,a z), a=0.02,b=0.03, rho=1,P=0, the angular-current source is
   5ab(x^2+y^2). Using production surface currents and the production region
   classifier, the summed absolute regional residual is 0.0275389, 0.000700399,
   0.000499034 for 16,32,64 Cartesian cells per axis. This demonstrates geometric
   residual reduction; it does not establish a clean asymptotic order or physical
   simulation convergence.

The actual patched executable also samples the isolated single-hole spatial metric
at dx=1,0.5,0.25 with fixed 64x128 angular quadrature. Proper-area errors relative to
16 pi are **4.1123%, 1.0473%, 0.2701%**. This tests metric interpolation/spatial sampling,
not convergence of accretion, fluid dynamics or binary torque. Small lifecycle and
performance fixtures explicitly lower the qualification threshold to exercise paths;
they are not production-resolution demonstrations.

## Measured overhead

The accompanying [CPU measurement JSON](circumbinary-diagnostics-cpu.json) contains all runs, environment metadata and area measurements. It was produced with:

```sh
/tmp/athenak-diagnostics-python/bin/python tst/diagnostics/measure_short_cases.py \
  build-diag/src/athena /tmp/athenak-diagnostic-measurements-provenance
```

The control has 32^3 active cells, 20 accepted steps, 16x32 angular points per surface,
history every five cycles and a final checkpoint. Each mode has three runs. The
reported evolution timer excludes initial setup and includes end-of-run output. This
is a short one-rank CPU control; it is not an estimate for production GPU/MPI jobs.
The single-hole control has R3.5, R4, H1 and BH1_Kerr surfaces. The additional
regional mode uses near1, cavity and annulus, with two fractional subcells per
direction. Medians from the retained raw measurements are:

| Mode | Median evolution time (s) | Relative to disabled |
|---|---:|---:|
| disabled | 7.425037 | reference |
| surfaces | 7.454244 | +0.39% |
| budgets | 7.672532 | +3.33% |

The surface-only difference is within short-run timing noise. These timings measure
the implemented paths without claiming a general speed improvement.

## Remaining limits

* Regional closure currently requires stationary corotating prescribed ADM, fixed
  puncture exclusions and no radiation/Z4c. Fully moving/deforming regional budgets,
  energy closure and fluid/EM exchange closure remain separate work.
* Excision-boundary fluxes can be unavailable because their stencils contain reset
  data or are underresolved. Near-hole storage/torque and actual mutation records
  remain useful; the complete near-hole residual stays NaN rather than omitting X.
* The branch's pre-existing MHD RK4 path lacks its required delta recurrence; new
  accepted-step integration explicitly supports RK1/2/3 only.
* The existing mixed metric/state stage timing limits time-dependent inertial
  interpretation. Startup can also refresh an inertial background at the current
  mesh time; trajectory/restart equivalence for that evolution is not established.
  Matched restart tests use stationary backgrounds. No higher order or exact discrete
  closure is claimed.
* Lower-dimensional derivative/torque helpers were tested; no new low-dimensional
  Bondi configuration was enabled or claimed as an executable validation.
* GPU, multi-node MPI, long production runs, magnetic/radiation physical convergence,
  nonideal-EOS runtime regressions and bulk historical HDF5 re-extraction were not run.
  The old disk-structure statistics do not replace native whole-surface validity.
* The existing bounded restart text header is retained; compact state avoids enlarging
  raw zoom records. Oversized custom parameter headers remain subject to that bound.
