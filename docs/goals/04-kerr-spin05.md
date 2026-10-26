# Goal 4: Kerr evolution at a/M = 0.5

Status: provisional; refine from Goal 3 before activation. Former milestone 7.
Prerequisite: [Goal 3](03-schwarzschild.md) passes. Follow
[shared rules](README.md).

## Objective

Extend the admitted hyperboloidal Z4c treatment to a single Kerr black hole with
dimensionless spin a/M = 0.5, preserving the immutable Minkowski reference.
Establish spinning-geometry consistency, controlled gauge evolution and convergent
3D metric/constraint behavior before claiming radiative accuracy.

## Execution

1. Admit exact spinning initial geometry, constraints, compactification and
   asymptotics independently. Account for non-conformal-flatness and angular
   dependence. The scalar's stationary analytic trumpet does not by itself
   establish suitability for dynamic Z4c or for stationary 1+log slicing.
2. Establish whether the initial metric and chosen gauge form a stationary
   solution. Separate stationary preservation from gauge relaxation. Do not
   interpret gauge transients as gravitational ringdown.
3. Run low-cost local pilots and then the locked three-resolution matrix,
   measuring mass, angular momentum, horizon geometry, inner profiles,
   H/M/Theta/Z, endpoint compatibility and actual characteristics.
4. Add small angular gauge disturbances and compare against the appropriate
   stationary/relaxing control. Test radial and angular resolution separately
   before assuming a balanced grid resolves the spinning geometry.
5. Add physical radiative perturbations only after admitting their initial
   constraints and gravitational extraction at scri. Then compare frequencies
   and damping with independent spin-two Kerr predictions, using stable fit
   windows and a justified time normalization. Spin-zero scalar frequencies
   remain a separate fixed-background control.
6. After uniform CPU scientific acceptance, validate Cartesian AMR and short GPU
   agreement without repeating completed controls or changing acceptance budgets.

## Gates to finalize from Goal 3

Lock field/derivative/constraint convergence, mass/spin drift, horizon diagnostics,
source-aware timestep controls, inner/interface/outer sensitivity, exact-scri
regularity and characteristic signs. Declare separate gates for stationary
preservation and gauge relaxation. Radiative claims require additional independent
data/extraction/QNM gates and must not be inferred from survival alone.

## Required final task: define and refine the successor

Write a new successor Markdown goal, with its scope determined by this goal's
results. Recommended candidates are localized constraint-satisfying radiative
initial data and gravitational extraction at scri, or a missing AMR/GPU integration
gate. Specify prerequisites, independent solutions, quantitative criteria and
resource limits. If this goal fails, propose the smallest repair goal instead
of automatically escalating spin or adding a binary. Update this file and the
index with evidence and readiness; do not activate the successor automatically.

## Results

No full-Z4c Kerr result is asserted by this plan. Existing a/M = 0.5 scalar
evolutions are frozen-background evidence only.
