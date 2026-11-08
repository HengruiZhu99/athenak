# AMR interpolation accuracy review

The original surface test required an observed L-infinity order of at least
5.5 at both N12→24 and N24→48 for `sin(5.1u) cos(4.7v)`. That test **failed**:
the first ratio gives 5.471545, the second 5.546411. The numerical method and
the original threshold have not been changed to make that result pass.

An independent 60-digit Decimal calculation on exact angular grids reproduces
those errors. This is the finite-resolution truncation error of the prescribed
six-point stencil. It is not evidence of lost sixth-order consistency. At a
finite spacing the leading derivative coefficient and the location of the
maximum error can change with resolution; an asymptotic order does not impose
a universal lower bound on every successive finite-resolution error ratio.

For six-point Lagrange interpolation of a one-dimensional function, the error
is bounded by the maximum sixth derivative times the absolute node product,
divided by 6!. For this separable test, define

    Eu = 5.1^6 / 720 * |product(u-u_i)|
    Ev = 4.7^6 / 720 * |product(v-v_j)|.

An independent bound for the tensor interpolation is `Eu + Ev + Eu*Ev`.
The alternate bound `Eu + Lu*Ev`, where Lu is the absolute row sum of the
u interpolation, follows directly by applying the operator in succession.
Both are finite-spacing bounds; neither uses measured errors to set a
tolerance. The numerical test adds its predeclared coefficient-norm roundoff
allowance. These bounds scale as the sixth power of spacing for the same
relative stencil positions and bounded derivatives.

The revised admission argument retains the original coarse-grid failure and
requires all of the following, without modifying the interpolation:

- All 36 tensor monomials with angular degree ≤5 pass the original roundoff
  bounds for both refinement and coarsening.
- The same smooth waveform passes a pointwise analytic remainder bound plus
  the fixed roundoff allowance at N12, 24, 48, 96 and 192.
- Actual floating-point operator results, not just the high-precision
  prediction, satisfy the unchanged order threshold ≥5.5 on N48→96→192.
- Sparse surface transport, unchanged-state copies, algebraic reconstruction,
  history conservation and owner migration pass at two and eight ranks.
- The independent bulk row, native regrid, coupled evolution, restart and
  unchanged N12/Cartesian regression gates remain separate requirements.

This review changes the interpretation of a premature asymptotic-rate test;
it does not relax the legacy-vs-native field, constraint or horizon comparison
tolerances. The finite-N surface error remains a real interpolation error and
must be reported. Bounded coupled AMR tests do not establish long-time AMR
convergence, particularly for the measured post-coarsening clock transient.
