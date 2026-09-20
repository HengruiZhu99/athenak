# Independent C++ physical-constraint radiation review

Reviewed read-only against the independently constructed radial full-state operator and exact constraint identities. Scope: the opt-in centered stationary M=1 trumpet pilot. No claim is made for a generic Kerr/Liu background, nonlinear stability, or AMR/MPI execution from source inspection alone.

## Formula and integration findings

- The scalar signs are correct in the existing outward conformal-normal convention: `delta C1_rate=-lambda_light_in*FTheta/alpha` and `delta C2_rate=+lambda_light_in*FQn/c_light`. For the analytic radial trumpet, `c_light=alpha*sqrt(chi)=chi`, reproducing the tested radial pilot.
- The transverse coefficient is also correct at principal order: `C_A=-2*A_nA/sqrt(chi)-Gamma_A+D_n g_nA`, hence `F_QA=-c_light*D_n C_A+lower`; its rate correction has the same positive `lambda_light_in/c_light` sign.
- The helper computes the physical covector `Z_i=gtilde_ij*(Gamma_evolved^j-Gamma_metric^j)/2`, then transports `Z_full-Z_background` using the reference physical connection. It does not radiate raw evolved Gamma.
- In the spherical reference, `Z_r=q/2`, reference `Gamma^r_rr=-1/(r*(r+1))`, and `omega=v/(r+1)`. Thus `2*F_Zr=q_t+v*(q'+q/r)` in the linear radial limit. The covector connection is essential to obtaining q/r instead of q/(r+1).
- The metric-defined connection is `g^al*g^bc*(D_b g_cl-D_l g_bc/2)`. Its implemented time derivative differentiates both inverse metrics and the metric derivative, using `g_inverse_t=-g_inverse*g_t*g_inverse` and the same discrete derivative on metric RHS. This is the correct discrete Frechet derivative; omitting either inverse term would be wrong.
- Gauge target branches and the two TT radiation targets remain the existing `zero_rate` targets. The physical constraint targets retain their immutable full-volume characteristic rates. No direct Theta/Gamma overwrite or small-state reset is introduced.

## Ownership and exact-zero argument

The helper differentiates metric RHS only; it reads Theta/Gamma RHS at its own cell. The boundary operator writes Khat, Theta, A and Gamma only. Its X1/X2/X3 kernels retain disjoint active-cell ownership, including the existing composite-normal edge/corner treatment. Consequently no newly differentiated RHS field is mutated by these kernels, and neighboring threads do not read each other's changed momentum RHS. All nested reads stay in the local active-cell range; no new ghost-RHS exchange is needed. The full, background and RHS arrays are separately allocated. Existing task dependencies place the boundary task after volume/source completion; no additional synchronization is justified by this change.

For bitwise-identical finite full/background states, every `RadiationZ` evaluation follows the same floating-point operations and subtracts to exact zero. With zero residual RHS, metric inverse time derivatives, metric-defined Gamma_t, Z_t and Theta_t are zero. All transport and falloff additions then remain zero. This is an algebraic argument; the independent compiled zero/stage and thread-count tests remain necessary runtime evidence.

## Found accuracy defect and verified accuracy correction

The first revision used independently selected three-point second-order derivatives both inside metric-defined Gamma and outside in D(Z). That composition was only first order at a boundary. For example, at x=0 with f=x^3, forward D at0 and centered D at h,2h give `D(Df)(0)=4.5*h`, while the exact second derivative is0. This was reported before promotion.

The helper author replaced the inner metric/metric-RHS derivative with an active-only five-point fourth-order derivative and added the at-least-five-active-cells guard; outer transport remains second order. I inspected the revised coefficients, clamped window/indexing, shared Gamma/Gamma_t derivative functional and parser/helper guard. The independent compiled cubic-metric check measures orders 2.172, 2.095 and 2.050 in `cartesian-design/cubic-convergence-v2.json`, versus 0.974, 0.991 and 0.996 for v1. This repairs the composed derivative accuracy. It does not establish stability: v2 real three-dimensional pulse controls still fail near 21–22M, including the single 16^3-block case.

The independent frozen-face Fourier matrix now reproduces a fast shifted oblique branch. The source formulas can have the correct radial limit, ownership and exact-zero behavior while their complete discrete boundary closure is unstable. See `PUBLISHED_BOUNDARY_PLAN.md`; this implementation is not approved as a cure.

## Limits that must remain explicit

- `R=|x|+characteristic_radiation_areal_shift` assumes a fixed center/reference amplitude radius. No generic Kerr areal-radius interpretation follows from its name.
- The physical Z time derivative assumes a stationary reference. A time-dependent background would require its time derivative as well; no such case is validated here.
- A second-order boundary helper does not inherit sixth-order accuracy from the interior. Block-edge one-sided stencils and face/corner composite normals need convergence and multi-block tests.
- P-only differential characteristic corrections can change outgoing rates at finite resolution. The radial pilot demonstrated convergence of the physical conditions, not exact finite-grid physical-constraint preservation or a three-dimensional stability proof.
