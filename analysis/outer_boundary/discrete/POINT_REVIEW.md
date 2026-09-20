# Damped boundary helper review

Reviewed the root's prototype in `athenak-outer-boundary-fix` and compiled its
actual `z4c_constraint_radiation.hpp`, using the existing OpenMP build.

The added terms match the derived frozen kappa2=0 constraint equations:
`f_theta += sigma*(Theta-sqrt(chi)*n^i Z_i)` and `f_Z += sigma Z`, with
`sigma=alpha*kappa1`. The conformal/physical normal factor is correct. Constructor
guards exclude kappa2 != 0 and time-rolled kappa, for which the simple source
product would be inconsistent. Optional geometric falloff defaults off.

The characteristic update maps the scalar correction with weights
`-lambda_in*FTheta/alpha`, `+lambda_in*FQ_normal/c_light`, and transverse
constraint rows with `+lambda_in*FQ_transverse/c_light`. These signs agree with
the frozen normal identities `FTheta=alpha Dn C1+lower` and
`FQ=-c_light Dn C2+lower`. The unchanged scalar/vector/tensor solve supplies
uniquely owned p-field writes; metric RHS stencil inputs remain immutable.
There is a separate original-path input-validity bug: the characteristic-rate
`NormalDerivative(rhs)` may read an uncomputed tangential RHS ghost at an
internal block edge when the full inverse metric tilts the contravariant
normal. Immutability does not imply that input is initialized. The helper test
below uses its own active-only derivatives and does not cover that original
characteristic-rate path.

Local compiled point tests passed 67 calls, including all 26 signed orientations:

* exact-zero error 0;
* manufactured damped result error 3.82e-16;
* independent metric-Z time derivative error 1.44e-10 with finite difference
  epsilon=1e-4;
* matched-metric Gamma physical-Z error 0;
* nonzero response .0643206;
* all state and RHS ghosts remained NaN poisoned, with no status failures.

Artifacts: `point-test/result.json`, compiler command and build log. Test sources
are in `athenak-outer-boundary-fix/tst/unit/z4c_damped_boundary/`.

Remaining limitations are material: D4 active-cell metric-defined Gamma and D2
transport differ from the volume discretization; active-only derivatives also
change at internal block edges. The curved helper is a chosen reference
transport model, not the exact nonlinear radiation condition. The original
gauge/TT rows remain a separate source of continuum surface modes. Pointwise
correctness does not override the failed full-evolution tests.
