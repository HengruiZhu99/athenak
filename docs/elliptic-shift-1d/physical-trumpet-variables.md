# Positive metric profiles with physical curvature variables

At a causal finite cutoff, the AF-scaled curvature variables can become poorly
conditioned as a trumpet forms. The new target chart keeps Ar, Khat+3, Theta
and Lambda as ordinary independent physical fields, and evolves

    U=log(chi/p(r)), V=log(alpha/A), A=(1+r^2)/2.

There is no field solve or reset in this variable change. The compact profile
uses p=r^2/D, D=r^2+Omega^2 s, with s=1 below.1, s=0 beyond.3 and the same
C-infinity reciprocal-distance logistic transition used for the initial-data
construction. The outer profile and its first two jets are set to exactly
1,0,0. It is coordinate-only, contains no black-hole mass, and does not enter
the lapse reference/source. The older global p=r^2/A^2 remains a control.

For X=chi/r^2 and ell=(log X)_r, the analytic reference jets give

    chi_r=2rX+chi ell,
    chi_rr=2X+4rX ell+chi(ell_r+ell^2),
    ell=U_r-D_r/D, ell_r=U_rr-D_rr/D+(D_r/D)^2.

The implementation uses direct deviation jets in the outer collar to preserve
small differences from the reference. The physical rates are divided by chi
or alpha to obtain U_t or V_t. Upwind transport acts on U,V with analytic
reference derivatives retained; physical curvature/constraint variables keep
their own transport. The canceled inner Hamiltonian/momentum expressions are

    Q=2X+2rX ell+2chi ell_r-chi ell^2/2,
    H=2K^2/3-6chi+2Omega chi(2+r ell)
      +Omega^2(Q-3Ar^2/2) - matter,
    M=Omega(Ar_r-3Ar ell/2)-2K_r/3+2rAr + matter.

Here K=Khat+2Theta. These are algebraic identities, not uses of H=M=0. The
same geometric mass numerator supplies the scalar potential, avoiding a second
poorly canceled evaluation. No Einstein equation, damping product, live-lapse
choice, Minkowski gauge source, Omega, or independent constraint field changes.

The domain event records the new chart. Physical curvature/trace/Theta/Lambda
are copied bitwise, while the positive totals' invertible log conversion error
is measured explicitly. Outer lapse deviations use log1p; only collapsed inner
values use the independently stored positive total. Strict restart binding and
every-stage incoming-characteristic guards remain active. The required full AF
prefix is assessed separately and cannot be replaced by post-cut results.

All50 CTest groups pass (API32, `evidence/ctest-v41.log`). Profile jets are
independently differenced and the flat outer frame is checked exactly. The128
new native/compiled cases cover vacuum/scalar, both transports/endpoints, both
restoring forms/damping products and with/without tapered KO. Maximum native,
compiled and short-step differences are3.91e-14,2.85e-14 and8.89e-16. Independent
constraint violations survive and scalar RK flux agrees. The new-chart domain
fixture verifies physical-field preservation, native finite-left rates, causal
rejection, source-hash checks and bitwise strict restart.

The frozen three-grid pilot and no-KO/global-profile/old-AF middle-grid controls
are governed by physical-trumpet-domain-protocol-v1.json. These are numerical
conditioning tests, not an admitted wormhole-to-trumpet solution. The profile
does not prescribe a limiting areal radius: X remains an evolved function.
