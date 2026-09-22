# Stationary Kerr trumpet geometry gate

The provider implements the stationary Kerr trumpet of Dennison, Baumgarte and
Montero, [arXiv:1409.1887v2](https://arxiv.org/pdf/1409.1887v2), with `R0=M` and
`r=R-M`. Its Cartesian algebra avoids polar coordinate singularities. It is
stationary ADM geometry; it does not claim to satisfy the unmodified standard
1+log gauge. A residual gauge requires its explicit background subtraction.

`kerr_trumpet::Evaluate(M,a,xyz,out)` accepts finite `M>0`, `|a|<M`, and `r>0`.
Spin is along the z axis and `a` has length units. No lapse clipping, repair,
excision, or regularization is applied. `alpha`, `beta`, `gamma`, `chi`, and
`conformal_metric` include first and second Cartesian derivatives. `K`,
`trace_K`, `conformal_A`, and `conformal_Gamma` include first derivatives only.
There is no placeholder or approximate K Hessian.

The lapse obeys `0<alpha<1` away from the excluded puncture. For
`Sigma=R^2+a^2 cos(theta)^2` and `X=(R^2+a^2)^2-a^2 r^2 sin(theta)^2`,
`alpha^2=r^2 Sigma/X`, and
`X-r^2 Sigma=(R^2+a^2)(2Mr+M^2+a^2)>0`.
The spatial metric is positive definite there. The lapse vanishes as `r` tends
to zero. The coordinate horizon is `r_H=sqrt(M^2-a^2)`, about `0.43589M` for
`a/M=0.9`; matching the Schwarzschild coordinate spacing therefore reduces
horizon resolution. At zero spin the formulas reduce to the existing
`R0=M` Schwarzschild trumpet. Bitwise identity requires the existing explicit
zero-spin path because floating-point expression ordering differs.

Run, with a Python environment containing NumPy:

```sh
python3 tst/unit/kerr_trumpet/validate.py --output /absolute/evidence/directory
```

The wrapper builds a standalone double-precision C++ library. Validation includes
200 positions with positive/negative spin, exact axes, horizon/interior/exterior
points, independent spherical tensor conversion, physical ADM constraints and
stationary metric/curvature equations, conformal identities, mass scaling,
spin-reflection symmetry, asymptotic mass/angular momentum, and invalid input
rejection. Sixth-order value-only finite differences independently test
analytic derivatives and convergence. The finest Hamiltonian stencil can reach
cancellation error; the gate requires sixth order before that floor, further
decrease, and a separate absolute fine-grid error bound.

These are geometry tests, not GPU, MPI, AMR or perturbation-stability tests.
The paper notes angular dependence at the puncture; reference subtraction
alone does not establish stability of perturbations crossing its neighborhood.
