# Complete rational gauge WIP: independent source review

Source/math inspection PASS for the captured uncompiled header. No blocking
formula or binding-layout correction was found. This is not an executable,
unit-test, physical-state, integration or native acceptance. The original 63
far-dual comparisons remain failed. No compiler, candidate import, numerical
arithmetic, target generation, scientific query or payload decoding was used.

The exact header identity is recorded in `source-before.json`; its included
backend and the frozen independent whole-row pencil are captured separately.
The backend's own source/unit qualification remains a separate prerequisite.

The `Fields` layout has 33 atoms: three scalar fields, four vectors and two
nine-entry matrices. Live plus reference fields, 36 supplied scaled-connection
atoms, three Omega-gradient atoms and Omega total 106 value/direction pairs.
All are decoded by bits before positive-scalar or singularity guards. An
invalid atom rejects the call rather than allowing a zero-factor shortcut.
The algorithm retains every matrix entry and direction; it does not assume
symmetry, trace-free directions, determinant one or zero determinant tangent.

`beta_d[j][i]` is partial_j beta^i. The reference `physical_P` must bind the
actual `p.k_physical`; live `physical_P` is the stored off-constraint P, with
no P+2Theta substitution. Reference beta is the actual `p.beta`, its gradient
is `p.state.beta.d`, lapse and lapse gradient are `p.alpha` and `p.dalpha`,
and the remaining hatted fields bind the consumed `p.state` fields. The
connection and Omega atoms are supplied inputs, not reconstructed geometry.
The present header does not implement that future native adapter.

The exact algebraic domain is positive live/reference lapse and chi,
positive Omega and two exactly nonsingular submitted metrics. Either
determinant sign is allowed. This deliberately does not rely on an
entry-rounded inverse or rounded Geometry validity. A physical caller must
separately enforce its symmetry/SPD, reference and full-geometry contracts.
The joint twelve-row entry point requires positive Omega even when a caller
only wants to inspect parts; this is the declared assembled-output domain.

The cofactor loop deletes row j and column i for adjugate_ij and uses the
sign (-1)^(i+j). Its determinant is sum_j g_0j adjugate_j0. Both identities
hold for general nonsymmetric matrices. The `Dual` operations differentiate
these complete polynomial expressions. In particular a zero primal factor
does not erase the other factor's nonzero first direction.

Let Q=D Dr and d=h Q, with C and Cr the two adjugates. The implementation's
`live_Vn`, `ref_Vn`, K, Kh, N and M are exactly the frozen pencil's

    live_Vn=a² chi C Dr,  ref_Vn=h² ch Cr D,
    N=live_Vn-ref_Vn,  K=live_Vn-beta betaᵀ Q,
    Kh=ref_Vn-betah betahᵀ Q,  M=K-Kh.

The regular lapse numerator is Q times
h beta·grad(alpha) - alpha betah·grad(h), giving the required regular row
after division by d. The pole lapse numerator h[Q Ta-alpha M:Gamma0]
has Ta=-alpha²P+alpha h Ph-alpha(beta-betah)·grad(Omega). Thus both the
physical-P sign and all connection terms agree with the source pencil.

The regular beta numerator h[Q ordinary+gradient] contains the complete
live/reference Lambda difference, beta_j beta_d[j][i] advection difference,
and C_ij Dr[(alpha²/2)chi_j-alpha chi alpha_j] minus its reference analogue.
The one-half is an exact dyadic atom; there is no early floating product or
inverse rounding. The pole beta numerator h times

    2 N_ij Omega_j - M_jk Gamma[i+1][j][k]
      - (K_jk beta_i-Kh_jk betah_i) Gamma[0][j][k]

has the correct derivative/component indices and the combined dL/beta
identity. All sums retain the full nine connection entries. These formulas
are mathematical equivalents of the old real rows, not a claim that the
old rounded graph passed the failed comparisons.

Rows 0–3 are regular parts, 4–7 pole parts, and 8–11 separately assembled
RHS rows. Each part is rounded from its own exact numerator/d. Each RHS is
rounded independently from (Omega numerator_regular+numerator_pole)/(Omega d),
without reusing rounded parts. First directions use the complete quotient
cross numerator n_dot d-n d_dot and d²; all reference, coefficient and Omega
directions are included. Field-only zero reference directions are a later
caller specialization, not a hidden assumption here.

The frozen pencil's degree/coefficient bounds apply to these exact polynomial
identities: degree at most eleven for part numerators, twelve for assembled
numerators and twenty for the largest complete quotient-direction cross
numerator. The literal partial sums have the same unsigned coefficient-sum
majorants. A looser source-level bound is also sufficient: coefficient sum
below 2^128, degree at most twenty, atom grid 2^-1074 and atom magnitude below
2^1024 place every raw polynomial between grid exponent -21481 (including
the exact half) and magnitude exponent 20608. Its normalized active integer
magnitude therefore needs fewer than 42090 bits. Denominator squares have
lower degree. The audit's rounded-atom times denominator and difference stay
inside this bound. Maxfinite comparison adds only its fixed 53-bit factor;
final quotient grid alignment needs at most the larger active magnitude plus
54 bits because the post-alignment quotient is at most 53 bits. Compare-scaled
uses virtual shifts. These bounds fit well inside the backend's 131072-bit
and exponent resource limits; conservative extra-limb checks do not approach
that capacity. This source argument does not replace runtime backend tests.

The rounding contract is strong: any exact absolute result above maxfinite
gets explicit overflow, even where IEEE rounding alone could return maxfinite.
Exact zero becomes +0; negative nonzero tiny values may become -0. Rounding
audits compare rounded*denominator to numerator exactly, and multiply the
delta sign by denominator sign, so their error direction is valid even for
negative determinant products. `grid_exponent` is the grid of the exact
rational's binade, or -1074 for subnormal rounding; it is not necessarily the
grid of a rounded value that carried to the next binade. Overflow audit
defaults must be interpreted only with the explicit nonfinite status.

Part and assembled finite flags are independent and include both primal and
direction rows. A part can overflow while an exactly canceled assembled row
remains finite. A capacity or invalid-atom exception clears both aggregate
finite flags; partially written rows are not an accepted result. `evaluated`
alone does not imply either finite flag. Internal logic invariants and system
allocation failures are not silently converted to successful outputs.

No repair of upstream rounded metric construction, supplied reference or
connection samples, full geometric RHS, principal matrix, nonlinear state,
scri preservation or native evolution follows from this review. Independent
Fraction/bit oracles, invalid/range/resource controls, complete-direction
tests and actual adapter/source gates remain future work.
