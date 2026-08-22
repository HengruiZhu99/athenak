# Simple dynamic Ref-GH regularization for isotropic Schwarzschild

## Controlling conclusion

**SIMPLE DYNAMIC REGULARIZATION NOT ESTABLISHED.**

The equation-preserving infrastructure, dynamic-frame oracles, native-cell
estimator, static wormhole test, PVC full-output cycle, and checkpoint/restart
gate pass. The required T5 open-loop localized transition does not: all three
resolutions fail closed near `t=1.62--1.65M`, before the controller is legally
allowed to activate. T6 and T7 were therefore not run. There is no claim of a
completed `t=4M` transition, a closed-loop result, a stationary trumpet, or
long-time stability.

This study is vacuum Ref-GH only. It does not cover fluids, Kerr-Schild data,
horizon finding, binaries, spin, moving punctures, CPBC, or hybrid methods.

## Source and invariants

The isolated branch is `codex/ref-gh-simple-controlled-trumpet-20260822`, based
exactly on `f962c946534bf0cd3d80abaae351973b37560a43`. The implementation retains
the exact 50 evolved fields:

- 10 symmetric `Psi_AB`;
- 10 `Pi_AB`;
- 30 `Phi_IAB`.

It does not change the GH continuum equations, FD stencils/order, low-storage
RK coefficients, compatible-Phi algorithm, KO definition, native constraints,
`gamma1=-1` structure, or source equations. No evolved field is clipped,
reset, projected, stuffed, or forced toward the reference.

The intended additions are:

- analytic genuinely time-dependent spatial-frame and Schwarzschild reference
  jets through second spacetime derivatives;
- generic dynamic reference curvature and covariant-source support;
- a controller-generation-keyed reference cache;
- four controller scalars advanced with the PDE's same low-storage RK
  recurrence and stage ordering;
- a fixed native-Cartesian shell estimator and global conditioning monitors;
- restart persistence for the four scalars without changing the 50-field
  restart array;
- Schwarzschild wormhole, controlled-transition, and estimator-calibration
  problem generators and gates.

## T0--T4 qualification

The compact local record is
`fo_gh_artifacts/ref_gh_simple_controlled_transition_20260822/local_validation.txt`.

| Gate | Result |
|---|---|
| Existing exact Minkowski | exact zero |
| Dynamic-lapse exact evolution | `1.665335e-15` maximum |
| Dynamic-spatial exact evolution | `1.665335e-15` maximum |
| Dynamic-spatial cache oracle | at most `4.163336e-17` |
| Stationary-trumpet initial RHS | `7.482764509e-17` |
| Frame vs coordinate curvature | `8.67502e-18` |
| Dynamic coordinate flatness | `2.55872e-17` |
| Covariant source oracle | `4.996e-16` |
| Static wormhole through `0.1M` | finite, exact initial match, no intervention |

The local planted-mismatch matrix used 15 combinations: `32/48/64` cells and
`delta=-0.25,-0.10,0,0.10,0.25`. Its maximum absolute errors were
`5.6704e-16` for `e_G` and `9.08802e-17` for `e_alpha`.

The final source-qualified Aurora gate is job `8775444` on one PVC tile. It
finished with PBS exit zero. The 15-case GPU matrix passed with maximum errors
`3.26297e-16` and `1.32954e-16`. A full-output evolved RK4 cycle stayed finite,
advanced controller generation to 4, wrote a checkpoint, and a real restart
advanced generation to 8. The retained checkpoint SHA-256 and histories are in
`aurora_gate_8775444/`.

## Estimator and controller policy

The final estimator was not tuned after calibration:

- fixed shell `0.15M <= r <= 0.40M`;
- native Cartesian cells only, without interpolation;
- weight `(M/r)^3`;
- global MPI reductions;
- shell required on the uniform finest level and away from AMR interfaces;
- freeze, rather than fabricate a slope, when the shell is invalid.

The safety distance is configurable in cell units. Its conservative default is
four finest cells. The production ladder documents and uses one finest cell,
because four cells exceed `r_fit_min` at `dx=M/16` and `M/24`. This changes
only the feedback activation guard, not the controller ODE or PDE equations.

## T5 grid and causal audit

The common physical grid was:

- domain `[-6M,6M]^3`;
- root grid `3 x 3 x 3` MeshBlocks;
- one fixed refinement of the central root block;
- 26 active level-0 plus 8 active level-1 blocks, 34 total;
- finest coverage `[-2M,2M]^3`;
- `nghost=4`, RK4, CFL `0.05`;
- root/MB cells `96/32`, `144/48`, and `192/64`;
- `dx_min=M/16`, `M/24`, and `M/32`;
- puncture at a Cartesian cell vertex on every active level.

Each startup reproduced the same tree. Eight distinct PVC tiles were mapped to
eight MPI ranks; the 34 blocks were distributed as six ranks with four blocks
and two ranks with five blocks. The maximum recorded characteristic speed was
`0.829900`. A conservative signal from a face at `6M` reaches `r=2M` no earlier
than `(6-2)/0.829900 = 4.82M`, and reaches the outer fit-shell radius `0.4M` no
earlier than `6.75M`. All failures occurred before `1.65M`; outer-boundary
contamination is therefore not a plausible cause.

Aurora mesh-only mode wrote and flushed the correct tree, then its one-rank
SYCL process faulted during teardown with status 139. The production harness
accepts only status 0 or that exact post-output status after checking the root,
block count, and physical-level count. Evolution faults are never accepted.

## T5 open-loop result

All three cases used `delta_q=delta_p=0` and feedback disabled.

| Resolution | Last history | Last cycle diagnostic | First fatal evidence |
|---|---:|---:|---|
| `M/16` | `1.600432M` | `1.611025M` | invalid effective timestep before the next cycle-10 diagnostic |
| `M/24` | `1.603305M` | `1.642029M` | invalid global conditioning at stage `1.646229M` |
| `M/32` | `1.602108M` | `1.631120M` | invalid effective timestep before the next cycle-10 diagnostic |

At the matched `t=1.6M` history rows, all three were still finite and reported
`bad-state=0`, but rapid growth was already unambiguous:

| Metric at `1.6M` | `M/16` | `M/24` | `M/32` |
|---|---:|---:|---:|
| GH L2 | `7.344e-1` | `4.200e-1` | `2.016e-1` |
| reduction L2 | `3.918e-2` | `1.672e-2` | `7.727e-3` |
| common all-domain H L2 | `1.239` | `1.049` | `8.077e-1` |
| common all-domain M L2 | `3.257e-1` | `1.246e-1` | `5.246e-2` |
| `e_G` | `-2.806e-1` | `-2.846e-1` | `-2.775e-1` |
| `e_alpha` | `-1.120` | `-1.162` | `-1.131` |
| relative metric condition | `1.275` | `1.281` | `1.281` |
| relative lapse range | `[0.980,3.89]` | `[0.978,4.11]` | `[0.997,4.21]` |
| physical lapse minimum | `9.475e-3` | `4.539e-3` | `2.636e-3` |

Thus the last written rows do not contain NaNs or a nonpositive lapse; the hard
failure occurs inside the following RK stages when either the physical metric
can no longer supply a valid timestep or the global relative-conditioning pass
detects an invalid cell. The exact first-fatal messages are retained in the
case logs, not reconstructed from the histories.

## Pre-failure convergence

The instability is not a simple resolution-reversed truncation trend. Before
the common failure time, important errors decrease with refinement. Selected
pairwise orders are:

| Time | Metric | coarse/medium | medium/fine |
|---:|---|---:|---:|
| `0.5M` | GH L2 | 3.345 | 3.393 |
| `0.5M` | common `2<=r<4` H L2 | 3.815 | 3.449 |
| `1.0M` | GH L2 | 3.218 | 3.748 |
| `1.0M` | reduction L2 | 2.794 | 3.962 |
| `1.5M` | GH L2 | 1.977 | 2.234 |
| `1.6M` | GH L2 | 1.378 | 2.552 |
| `1.6M` | common all-domain H L2 | 0.410 | 0.910 |
| `1.6M` | common `2<=r<4` H L2 | 1.632 | 1.620 |

The all-domain Hamiltonian norm converges slowly because it includes the
puncture without an evolving lapse/chi mask; the fixed outer and interface
regions show cleaner convergence. The complete values and orders are in
`open_loop_prefailure_convergence.{json,tsv}` and `matched_orders.tsv`.

Convergence before a nearly resolution-independent terminal failure is useful
evidence, but it is not stability and cannot satisfy T5.

## Why T6/T7 were not launched

The reference-transition support has
`r_full(t)=0.6 exp(-t/1.5) M`. Even with zero safety buffer, it does not clear
the fit shell until `t=2.079M`. With the documented one-cell buffer, the first
possible feedback times are `2.888M`, `2.568M`, and `2.430M` from coarse to
fine. Every open-loop member fails by about `1.65M`, while `r_full` is still
about `0.20M` and the global transition amplitude is only about 0.338.

Enabling the controller would therefore reproduce the open-loop evolution
exactly up to the same failure. Moving the shell, violating the transition-
outside-shell guard, or changing the prescribed transition would be a new
algorithmic experiment. None was done. Closed-loop stability, controller
trajectory consistency, post-transition behavior, and approach to the known
stationary trumpet remain untested.

## Performance and provenance

The bounds-off production executable SHA-256 was
`8a08b488aa6b223606db7019ff0e863238ccafda73965be4fb7cd01e849df798`.
The fine job used source `ebec8d28`; the medium/coarse harness used `d51e318a`,
whose only intervening change made failure preservation continue to the next
resolution. Both used the identical executable above.

Lower-bound active-zone throughput, using the last logged complete cycle, was
approximately `4.73e5`, `5.91e5`, and `6.68e5` zone-cycles/s from coarse to
fine. The fine, medium, and coarse evolutions consumed about 8137 s, 2929 s,
and 707 s respectively before failing. These are diagnostic research runs, not
production-performance qualification.

Large checkpoints and field outputs are intentionally not committed. Their
exact Aurora directories are recorded in `aurora_locations.txt`. Compact
histories, logs, mesh evidence, mapping, provenance, convergence tables, and
hashes are committed under
`docs/fo_gh_artifacts/ref_gh_simple_controlled_transition_20260822/`.

## Remaining blocker

The localized analytic reference transition itself loses a valid physical or
relative metric during the first third of its activation, before any permitted
feedback can operate. The pre-failure convergence argues against dismissing it
as a single-resolution numerical accident, but does not identify whether the
root cause is the prescribed reference path, the associated GH gauge response,
or a continuum instability. Resolving that requires a new formulation/design
goal; weakening safety checks or engaging the controller inside the fitting
shell would not qualify this goal.
