# Explicit causal domain transfers

`--transfer-af-profile` starts a separately bound finite domain from a saved
AF wormhole profile. It preserves every retained stored field and grid node
bitwise, keeps the original time lattice, verifies all continuum gauge and
physical inputs, and records source profile/manifest/executable hashes. It
never bypasses strict checkpoint restart validation. Profiles without an
integrator low part cannot transfer compensated parent runs. Scalar-domain
transfers are not implemented.

The same positive AF-reference variables remain invertible at r_min>0. The
first point now receives the actual physical tensor RHS, not an AF coefficient
ODE or center parity. Every RK stage solves the metric-derived Q=0 shift
branch with the outer null-normal shift. Q=0 is the inherited metric-freezing
inner gauge condition, not a Hamiltonian/momentum/connection constraint.
The initial shift change from the new quadrature closure is explicitly
recorded. Every subsequent stage rejects an incoming physical or lapse-gauge
characteristic in both Python and C++.

All46 test groups pass (API30; `evidence/ctest-v39.log`). The transfer fixture
checks exact retained fields, bitwise strict restart, independent native rates
at the finite left point, compiled/Python agreement with/without puncture KO,
source hash validation and incoming-characteristic rejection. Its native rate
difference is4.76e-14. This deliberately coarse fixture has large constraints
and is not admitted as physical evidence.

Generic analysis now rejects domain transitions unless retained-only scope is
explicit. Retained-domain analyses cannot implicitly omit the original prefix
and then claim whole-wormhole success. `prefix_stage_extrema` uses only actual
cumulative stage records through an exact saved endpoint, includes the final
accepted state, rejects missing/sparse coverage, and never imports later
manifest extrema into an earlier window. A frozen .5 pulse-window audit checks
the new analysis route against completed source-bound trajectories.

The predeclared exploratory pilot transfers nu=.25 at3.2 and r_min=1/64,
retaining505/1009/2017 points. All source fields remain free. Coarse original
prefix scri gates fail and cannot be erased by a clean exterior or by survival.
Fine full-AF N4097/8193 prefixes are separate active tests. The first pilot
runs keep all characteristics outward but develop constraint errors at the
left closure. Their dissipation/boundary consistency still needs repair and
independent cut-location/time/solve/convergence studies. No transferred
trajectory is a successful wormhole-to-trumpet result.

## Inner dissipation consistency (API31)

At a finite cutoff, the truncated -B^T W B matrix has an h^2 first-active-row
source even for a smooth cubic field. Its outer taper remains nonzero there.
The new selectable fixed physical inner width L multiplies W by

    d^6/(L^6+d^6), d=third-difference midpoint-r_min.

It retains nonnegative weights and negative cell-energy contribution. It
changes the first active-row consistency from h^2 to h^8, independently
verified with orders1.97/1.99 versus7.97/7.99. It introduces no parity, new
physical boundary data or field reset. The default L=0 preserves old runs.
All47 tests pass (`evidence/ctest-v40.log`); the updated tapered-filter copy/
native-rate/restart test and final-report analysis guards pass separately.
Accepted final reports augment existing stage extrema without discarding
failed-stage manifest peaks or promoting sparse sampled diagnostics.

The L=.03 pilot and isolated no-KO/untapered controls remain unaccepted.
The taper reduces constraint errors but does not by itself establish the
required convergence. A separate exact CMC-data diagnostic isolates severe
conditioning in the AF curvature scalings after a trumpet forms. At1009
retained points, the same analytic physical data give cutoff H/M errors
.00173/.00304 in the AF-scaled chart, versus1.28e-8/3.25e-12 with physical
curvature variables and a positive trumpet metric chart. No evolution or
constraint projection enters that comparison; it motivates a variable-change
control, not a claimed cure.
