# Native-VC modified-Cartoon identity audit

Date: 2026-08-23

## Contract and disposition

AthenaK stores reduced-plane components in the order

```text
(rho, z, y_suppressed) = (x1, x2, x3).
```

The active native-VC axis is `i == layout.is`, with physical coordinates
`rho=0,h,2h,...`. Scalars are even. Vector parity is `(odd, even, odd)`.
For a symmetric tensor, `rr`, `zz`, `yy`, and `ry` are even, while `rz` and
`zy` are odd. The active-axis identities are

```text
V_rho = V_y = 0
T_rr = T_yy
T_rz = T_ry = T_zy = 0.
```

All production `rho=0` branches now use analytic limits; none divides by
`rho` or `rho^2`. The first four positive-rho VC layers reconstruct regular
coefficients as polynomials in `s=rho^2`, avoiding the formal one-order loss
from dividing an ordinary centered derivative error by `rho=O(h)`.

Disposition: `DEFECT_FOUND_AND_REPAIRED` for the source-level VC axis
closure. Physical qualification is limited separately by the outer boundary
and by one seam-axis gauge component described in `FIRST_LOSS_LOCALIZATION.md`.

## Complete provider inventory

The notation below is

```text
T_rr = P + s Q,  T_yy = P,  T_ry = s R,
T_rz = rho U,    T_yz = rho V,  T_zz = W.
```

| Production branch | Regular continuum expression | Away-axis implementation | Axis/near-axis implementation |
|---|---|---|---|
| scalar `d_y` and active-suppressed mixed | `0` | `0` | `0` |
| scalar `d_yy F` | `2 F_s` | `D_rho F/rho` | `D_rhorho F` at axis; `2 F_s` fit on VC layers 1--4 |
| vector `d_y V^rho` | `-B` | `-V^y/rho` | `-d_rho V^y` at axis; fitted `-B` near axis |
| vector `d_y V^y` | `A` | `V^rho/rho` | `d_rho V^rho` at axis; fitted `A` near axis |
| vector `d_y V^z` | `0` | `0` | `0` |
| vector `d_yy V^z` | `2 C_s` | `D_rho V^z/rho` | `D_rhorho V^z`; fitted `2 C_s` |
| vector `d_yy V^{rho,y}` | `2 rho (A_s,B_s)` | `D_rho V/rho-V/rho^2` | zero at axis; fitted coefficient derivative near axis |
| vector `d_rho d_y V^{rho,y}` | `(-2 rho B_s,2 rho A_s)` | rotated quotient | zero at axis; fitted coefficient derivative near axis |
| vector `d_z d_y V^{rho,y}` | `(-B_z,A_z)` | `(-d_z V^y/rho,d_z V^rho/rho)` | radial derivative of the corresponding odd component; fitted coefficient derivative |
| tensor `d_y T_{rr,yy}` | `(-2 rho R,2 rho R)` | `(-2 T_ry/rho,2 T_ry/rho)` | zero at axis; fitted `R` near axis |
| tensor `d_y T_ry` | `rho Q` | `(T_rr-T_yy)/rho` | zero at axis; fitted `Q` near axis |
| tensor `d_y T_{rz,yz}` | `(-V,U)` | `(-T_yz/rho,T_rz/rho)` | radial derivative limits; fitted odd coefficients |
| tensor `d_yy T_rr` | `2(P_s+sQ_s)` | `D_rho T_rr/rho-2(T_rr-T_yy)/rho^2` | derived radial second-derivative limit; fitted `P,Q` |
| tensor `d_yy T_yy` | `2(P_s+Q)` | `D_rho T_yy/rho+2(T_rr-T_yy)/rho^2` | derived radial second-derivative limit; fitted `P,Q` |
| tensor `d_yy T_ry` | `-2R+2sR_s` | `D_rho T_ry/rho-4T_ry/rho^2` | `-2R` limit; fitted `R` |
| tensor `d_yy T_{rz,yz}` | `2 rho(U_s,V_s)` | `D_rho T/rho-T/rho^2` | zero at axis; fitted coefficient derivative |
| tensor `d_yy T_zz` | `2W_s` | `D_rho T_zz/rho` | `D_rhorho T_zz`; fitted `2W_s` |
| tensor `d_rho d_y T_{rr,yy}` | `(-2R-4sR_s,2R+4sR_s)` | Cook quotient identities | `(-2R,2R)` limits; fitted `R,R_s` |
| tensor `d_rho d_y T_ry` | `Q+2sQ_s` | Cook quotient identity | `Q` limit; fitted `Q,Q_s` |
| tensor `d_rho d_y T_{rz,yz}` | `(-2rho V_s,2rho U_s)` | Cook quotient identities | zero at axis; fitted coefficient derivatives |
| tensor `d_z d_y` nonzero branches | `(-2rho R_z,2rho R_z,rho Q_z,-V_z,U_z)` | analytic quotient of centered `d_z` | analytic limits and fitted regular coefficients |
| vector divergence | `2A+d_z V^z` at axis | `d_rho V^rho+d_z V^z+V^rho/rho` | analytic axis limit; fitted `A` near axis |
| active rho/z first, second, mixed, advection, KO | ordinary Cartesian derivatives of parity extension | centered O2/O4/O6 | same centered stencil through exact parity ghosts |

These branches are implemented by `ScalarFirst/Second`,
`VectorFirst/Second/Mixed`, and `TensorFirst/Second/Mixed` in
`src/z4c/cartoon_derivatives.hpp`. Tensor variance is a template argument;
mixed-index tensors do not reuse an all-lower rule implicitly.

## Production consumers

The same derivative provider is used by:

- the full Z4c RHS in `src/z4c/z4c_calcrhs.cpp`;
- ADM reconstruction and H/M constraints in `src/z4c/z4c_adm.cpp`;
- Gamma initialization in the Z4c problem generators;
- curvature and Weyl diagnostics in `src/z4c/curvature_diagnostics.hpp` and
  `src/z4c/z4c_calculate_weyl_scalars.cpp`;
- the VC Cartoon Sommerfeld evaluator in `src/z4c/z4c_Sbc.cpp`.

## Derivation authority and tests

The away-axis identities are the SO(2) reduced-plane formulas of Cook et al.,
arXiv:1603.00362, Appendix C, after mapping its radial coordinate to `rho`.
The reduced-hyperplane construction follows Pretorius,
arXiv:gr-qc/0407110. Axis expressions follow by smooth parity and
l'Hopital limits. `docs/z4c_cartoon_half_plane_operator_table.md` is a
generated source-to-formula inventory.

Manufactured production-kernel tests cover O2/O4/O6 scalar, vector, tensor,
ADM, constraint, and complete-RHS branches:

```text
athena.z4c_vc_cartoon_derivatives_o2/o4/o6
athena.z4c_vc_cartoon_axis_scalar
athena.z4c_vc_cartoon_axis_vector
athena.z4c_vc_cartoon_axis_tensor
athena.z4c_vc_cartoon_axis_adm
athena.z4c_vc_cartoon_axis_constraint
athena.z4c_vc_cartoon_axis_rhs_regularity
athena.z4c_cartoon_production_kernels
```

The final current-source host rerun passes all of these. The Perlmutter CUDA
gate also passed the selected native-VC production kernels before each
fixed-grid control. No current-source SYCL runtime was available, so SYCL
remains unqualified rather than inferred from host/CUDA.
