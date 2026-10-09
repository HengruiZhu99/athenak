# Common-rho dense-mass scalar radial model

This math-only gate passes all 50 predeclared cases: N = 4, 8, 12, 16, 24; L = 0, 1, 2, 3, 4; and outer radius rb = 0.7, 1.0. It checks polynomial radial scalar-wave integration by parts and a dissipative scalar boundary term on the whole ball. It does not construct an actual Z4c radial matrix, choose a Z4c boundary map, run evolution, or establish Z4c stability.

The accepted source is `model_gate.py`, with the predeclared parameters and tolerances in `plan.json`, all per-case results in `results.json`, and the dense mathematical matrices in `model-matrices.npz`. The accepted calculation took 0.0930 seconds internally, with Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1 and one OpenBLAS thread. A source/runtime dependency record and byte-verified immutable index accompany this report.

## Polynomial space and exact identities

Write a scalar harmonic as f(r,angles) = r^L W(rho) Y_Lm(angles), rho = r^2, with normalized angular integral and W a polynomial of degree at most N-1. Every L uses the same N Gauss-Jacobi nodes for weight rho^(1/2) on [0, rb^2]. The nodes exclude both endpoints; endpoint evaluation is the exact polynomial trace. Origin regularity comes from r^L times a polynomial in r^2 over the whole ball, with no inner boundary or annulus.

For nodal Lagrange polynomials li, define

```
Mij = (1/2) integral_0^(rb^2) rho^(L+1/2) li lj d rho,
D_L = 4 rho D_rho^2 + (4L+6) D_rho,
Sij = (1/2) integral_0^(rb^2) rho^(L-1/2)
      [(L li+2 rho li') (L lj+2 rho lj') + L(L+1) li lj] d rho.
```

The stiffness is the full physical gradient integral: both the radial derivative and angular L(L+1) term are retained. Differentiating r^L W(r^2) gives radial derivative r^(L-1)(L W+2 rho W'). Thus integration by parts gives

```
M D_L = -S + rb^(2L+1) t^T [L t + 2 rb^2 t D_rho],
```

where t is the polynomial outer trace row. The lower endpoint flux is zero. For L>0 it contains r^(2L+1) times bounded polynomial factors; for L=0 the radial derivative supplies a further r^2, so the leading flux is O(r^3). The code also records the exact zero and small-radius values for W = 1+rho+rho^2.

The dense mass and stiffness use Gauss-Jacobi quadrature with their respective weights rho^(L+1/2)/2 and rho^(L-1/2)/2. After removing each weight, the mass and stiffness integrands have degree at most 2N-2. Orders 2N+8 and independently 3N+17 therefore integrate these polynomials exactly in exact arithmetic; the recorded differences measure floating evaluation and quadrature errors. No unweighted quadrature or endpoint singular evaluation is used.

The independent strong radial action is built from analytic first and second derivatives of Jacobi polynomials, integrated against the mass weight. It is compared with both the barycentric common-node action and the weak action M^-1(-S+boundary). The weak action is not used to define the strong action.

## Scalar wave boundary identity

For the scalar model only, use

```
W_t = P,
P_t = D_L W - M^-1 rb^(2L+2) t^T
      [P_b + (L/rb) W_b + 2 rb W_rho,b].
E = (P^T M P + W^T S W)/2.
```

The bracket is the scalar outgoing condition in factored variables, P_b + (partial_r f)_b/rb^L. Its contribution cancels the integration-by-parts boundary work and leaves exactly

```
dE/dt = -rb^(2L+2) P_b^2.
```

The calculation checks the complete finite-dimensional symmetric matrix identity H J + J^T H = -2 rb^(2L+2) diag(0, t^T t), H = diag(S,M), and eight fixed-seed polynomial states per case. This is an algebraic semidiscrete scalar-model identity, not a time-integration experiment. The generator assembled here is solely this scalar polynomial model; no physical Z4c matrix or spectrum is computed.

For L=0, S has the static constant W direction in its nullspace. W=1, P=0 has exactly zero energy and exactly zero generator action in every recorded case. Consequently the natural wave energy is a seminorm on all configurations at L=0. For L>0, the physical gradient Gram matrix is positive; the code tests its Cholesky factor. At L=0 it tests only the nonconstant modal block, leaving the constant nullspace intact. No regularization, floor, or extra inner condition is added.

## Conditioning and numerical results

The common-node raw mass condition number reaches 2.8690650e9 at N=24, L=4. Scaling by its own diagonal reduces the maximum to 2.7684501e4. For energy and residual checks, the same polynomial space is transformed by the common-node evaluation matrix of orthonormal Jacobi polynomials for weight rho^(L+1/2)/2. This is a congruence, not a change of collocation nodes. The modal mass condition number is at most 1.000000000001092; the nodal-to-modal evaluation matrix condition number reaches 5.3563654e4. Both conditioning measures and mass/stiffness congruence residuals are retained for every case.

All principal scaled tolerances were declared as 2e-9 before the mathematical runs. Matrix residuals are Frobenius residuals divided by max(1, the sum of the independent term norms). The report retains absolute Frobenius and maximum-entry residuals as well; the worst absolute and worst scaled residual need not occur in the same case.

| Check | Worst scaled residual | Largest absolute Frobenius residual |
|---|---:|---:|
| Nodal integration by parts | 2.88615e-13 | 3.75870e-9 |
| Modal integration by parts | 2.45307e-13 | 1.13816e-6 |
| Scalar wave energy matrix | 3.82113e-13 | 1.60989e-6 |
| Strong versus weak action | 5.41404e-13 | 1.13815e-6 |
| Common-node versus analytic modal action | 5.82720e-13 | 8.90440e-7 |
| Modal mass versus identity | 9.97036e-14 | 9.76891e-13 |

The larger absolute operator residuals at N=24 accompany large derivative/stiffness entries. They are not represented as near-zero absolute error. The sampled energy-rate error reaches 1.66157e-10 absolutely and 5.42331e-16 under the predeclared rate scale, which includes the product of the energy-gradient and generator-action norms. The matrix identity is the primary cancellation check; these sampled scalar rates provide an additional readback.

The negative control substitutes diagonal common-node quadrature weights multiplied by rho^L for the exact dense mass. This happens to be exact for L=0 and L=1 because the polynomial degree remains within the common N-point rule's 2N-1 exactness. It fails for L>=2: relative mass errors range from 0.02864 to 0.17162 over the cases. In particular, common nodes do not justify a universal diagonal mass for all harmonic sectors.

## Preserved failures and reproducibility

The first attempt completed the mathematical calculation but exited when JSON serialization encountered a NumPy boolean. Its exact source, plan, stderr, stdout, matrices and failure receipt remain in `history/001-numpy-bool-serialization`. The next attempt passed with identical matrix bytes. A subsequent serializer hardening changed only unsupported-value handling to raise rather than emit null; its previous source, result, logs and receipt remain in `history/002-before-strict-serialization`. The accepted rerun uses the same mathematical formulas, nodes, N values and tolerances. Its stderr is empty. A readback-only verifier initially expected eleven arrays per case, while the mathematical source saves ten; that source/assertion/receipt is preserved in `history/003-readback-array-count` and the verifier count is corrected. No failed numerical gate is discarded and no tolerance is relaxed.

Reproduce from the repository root with the installed scratch dependency directory:

```
OPENBLAS_NUM_THREADS=1 PYTHONPATH=build-layer-research/boundary/python-deps python3 \
  build-layer-research/boundary/rho-dense-mass-model-20261009/model_gate.py
```

This overwrites only the working model output, so use a fresh copy to preserve the immutable receipt. `check_and_freeze.py` verifies already-saved outputs, hashes source/runtime dependencies and creates the immutable copy; it does not rerun the mathematical calculation.

An independent sibling-agent read-only source/math review confirms the mass/stiffness factors, Jacobi normalization and derivatives, matrix congruence/transposes, scalar boundary cancellation and L=0 seminorm. It did not rerun the numerical model. Its observation that congruence residuals were recorded without checks-dictionary entries is addressed by explicit saved-result assertions in the readback verifier under the already-declared 2e-9 tolerance, with no model rerun.

The passing scalar gate does not admit a full variable-coefficient Z4c Galerkin/SBP estimate, normalized harmonic momentum boundary map, constraint-preserving boundary condition, actual radial operator or evolution. Complete origin/core action, finite-radius principal/constraint coupling and bulk variable-coefficient control remain separate gates. It also does not alter or identify the cause of the existing Cartesian discrete mode.
