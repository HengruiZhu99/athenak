# Independent Q/null first-jet review

PASS, restricted to the frozen linear analytic-reference map. The scientific
index is `5490057dfc04e060bca65ec7ee1a3bb363a34cab0fe5c890f23d477a7bda7ec1`.
The independent script reads saved matrices and reconstructs their full 20x80
singular map and 24x80 leading-condition map from unrestricted Cartesian value
and first-derivative coordinates at the north point. It does not compile or run
the tensor kernel, a native binary, propagation or an evolution.

The physical ADM reconstruction uses `K=P+2Theta`,
`K^j_i=Omega A^j_i+K delta^j_i/3` and
`gamma_phys=Omega^-2 gtilde/chi`. At the outer reference,

```
H0 = -6(chi-h_nn)/a^2-4(P+2Theta)/a,
M0_i = 2 A_in/a-2 partial_i(P+2Theta)/3,
Z0_i = (Lambda_i-partial_j h_ij)/2.
```

Differentiating the actual reference `alpha=(1+r^2)/(2a)`,
`beta^i=-x^i/a`, `Omega=(1-r^2)/(2a)` gives N1, angular null/Q
identities and H1. Angular differentiation retains derivatives of the radial
unit vector. Independent maps equal every saved rationally reconstructed matrix;
their maximum raw floating difference is 3.552713678800501e-15.

For all four a values the reconstructed R0 rank is 11, nullity 69. The basic
leading conditions have rank 10, rising to 18 when R0 is included. Full angular
identities and H1 have rank 17, rising to 20 with R0. Adding Theta1 and the two
tangential tracefree shear residues to those angular/H1 conditions has rank 20
and includes all R0 rows. The value pole has rank(M)=rank(M^2)=11. These are exact
row-space statements for the rationally reconstructed reference first-jet map,
not general nonlinear identities or a preserved boundary manifold.

The two omitted-residue controls have appropriately limited meanings:

* `Theta=Omega, P=-2Omega` retains reference physical K and therefore identically
  vanishing linear physical ADM constraints and spatial Z, but Theta is nonzero
  in the interior. It is outside the exact Einstein/Z4 sector even though its
  leading Theta value vanishes. Its nonzero normal derivative sources Lambda.
* The tangential A control satisfies the listed local leading conditions,
  including H1 and Theta1, yet has a nonzero tangential shear residue. The gate
  does not construct exact Einstein data throughout a neighborhood or sphere.
* `P=Omega^2` has vanishing first-jet conditions but nonzero higher physical ADM
  coefficients. Its next singular residue cannot disprove invariance of the
  exact Einstein-compatible ideal. The varying reference lapse in its exact
  first RHS is required and is correctly retained in the frozen source.

The strong pure-gauge Einstein witness has zero initial H/M/Z/Theta and R0, but
one such cancellation does not establish general ideal invariance. An exact
Einstein-compatible higher-jet followup is required. No source, stability,
closed scri, ghost, puncture or black-hole admission follows from this review.

All 37 frozen files, 372 unchanged inputs, five successful commands and two
outside executables are verified. Release and ASan/UBSan Debug saved JSON are
byte-identical. Original mixed-auto compile and missing angular-Theta assertion
failures remain preserved by the scientific gate. An independent checker first
failed on SymPy multiplication by a Python boolean; its exact draft and observed
error metadata are retained under history. Explicit int conversion repaired only
that independent algebra checker before the accepted final run.
