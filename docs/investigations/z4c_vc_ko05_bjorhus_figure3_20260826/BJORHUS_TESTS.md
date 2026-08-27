# Full-field Z4c Bjorhus test record

The implementation-specific record is
[`../z4c_full_bjorhus_cpbc_20260826/BJORHUS_TESTS.md`](../z4c_full_bjorhus_cpbc_20260826/BJORHUS_TESTS.md).

Before the production discriminator, the integrated branch passed these local
CPU tests:

```text
athena.z4c_sommerfeld_derivatives
athena.z4c_full_constraint_bjorhus
athena.z4c_shared_geometry_policy
```

The manufactured Bjorhus executable covers an outgoing pulse, an incoming
pulse, the Cartoon axis/physical-boundary ownership intersection, and
deterministic point ownership.  Its fixed incoming case reports zero corrected
incoming residuals to roundoff and

```text
induced_outgoing_rate = 0.686111
```

The full serial Athena build and native-VC Cartoon Minkowski smoke also passed
in the separate implementation worktree.  MPI compilation/linking passed;
local multi-rank execution was not qualified because that machine's OpenMPI
runtime was unusable.  The Perlmutter campaign reruns the three focused tests
on an A100 before any Brill discriminator case and records their output as
runtime evidence.

These tests establish algebraic and dispatch behavior.  They do not establish
nonlinear stability, boundary-error reduction, or equivalence of the central
physical trajectory; those are the purpose of the A/B/C/D Brill comparison.

## Perlmutter validation and nonlinear finding

On the final source commit, all three focused executables passed on a Perlmutter
A100. Their SHA-256 values are recorded in
`evidence/cpbc/logs/cpbc_cuda_tests.log`.

The first nonlinear replay exposed a bug that Cartesian manufactured data could
not see: `MakeFullConstraintBjorhusFrame` interleaved writing `normal_d` with
computing `normal_u`, so off-diagonal inverse-metric terms consumed
not-yet-written components. The strict residual gate caught errors near
`5.2e-13`; it was not weakened. Commit
`d39822c6522688749fe5ead8025907bc055f02f8` separates the loops and adds a
non-diagonal positive-definite metric test. The corrected CUDA replay crossed
the previously fatal events and reached `t=0.05` with no residual-gate hit.

The longer Rout16 CPBC case nevertheless developed a boundary-corner runaway
and failed the strict characteristic-speed classification at `t=3.244461`.
This is a nonlinear red result for the experimental closure, not a test-suite
failure. The planned four-way discriminator was intentionally terminated before
completion.
