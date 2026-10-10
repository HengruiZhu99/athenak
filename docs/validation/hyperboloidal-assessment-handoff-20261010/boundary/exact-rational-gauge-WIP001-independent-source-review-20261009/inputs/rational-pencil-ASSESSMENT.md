# Exact whole-row RWM rational contract: independent source pencil

Status: mathematical/source preparation only. No candidate import, arithmetic
evaluation, CAS, compilation, target generation, array/payload decoding, query,
operator, eigensolve or evolution was performed for this note. The existing
63 far-dual failures and all earlier readback failures remain failures.

The literature agent's separate complete-row pencil had only captured context
when ownership was checked. This independent note derives the full rows and
bounds from the pinned `GaugeFar`, retained `LegacyGauge`, `Assemble`, and
production `Geometry` source. It does not adopt that agent's provisional bound
as a proof. The already frozen rational-gradient and exact-inverse pencils are
source context, not executed prerequisites or implementations.

## 1. Submitted atoms and domain

Interpret each finite binary64 input as its exact dyadic real value. The live
consumed fields are

    a, x, P, beta[3], alpha_d[3], chi_d[3], beta_d[3][3], Lambda[3], g[3][3].

There are 33 scalar atoms. `beta_d[j][i]` means partial_j beta^i. The reference
has the corresponding 33 atoms, denoted

    h, y, Ph, bh[3], h_d[3], y_d[3], bh_d[3][3], Lambdah[3], gh[3][3].

Here `Ph` binds `p.k_physical`, `h_d` binds `p.dalpha`, `bh` binds `p.beta`,
and the other hatted fields bind the actual `p.state` members consumed by the
source; these are not inferred from another reference construction. The fixed
coefficient atoms are `Gamma[A][i][j]=c.scaled[A][i][j]` (36), `O[i]=p.domega[i]`
(3), and `Omega` (1): 106 total submitted atoms. The supplied scaled connection
is not recomputed or replaced. In particular, this contract does not make its
already rounded entries exact samples of an unrounded analytic connection.

Every consumed atom also has a finite, independently supplied first direction.
The ordinary physical field-only binding fixes all reference, coefficient and
Omega directions to zero. For the algebraic contract and bounds, they may
instead be arbitrary; this includes the `T omega` direction allowed by the
existing templated `Assemble`. No differentiation of the procedure that
originally constructed a supplied reference/connection atom is implied.

The algebraic row domain requires a,x,h,y>0, Omega>0 for assembled outputs,
and exactly nonsingular submitted g and gh. Determinants may have either sign
in this algebraic extension. A physical caller must separately establish its
metric symmetry/SPD, complete geometry, reference validity and other source
conditions. The new exact path must not depend on a rounded determinant or
precomputed `Geometry.inverse` remaining finite: that would defeat its stated
submitted-matrix contract. Original source-validity behavior is a separately
gated integration question. No determinant floor, identity substitute,
condition-number cutoff, on-constraint relation or seed-specific branch is
introduced here.

All nine matrix entries and all nine first directions are retained, including
nonsymmetric or traceful directions. The formulas also extend to a nonsymmetric
primal matrix. No determinant-one, symmetric-copy, STF or dot(det)=0 identity
is used. A zero primal gradient/matrix/scalar factor must not delete a nonzero
direction. Theta, A, metric spatial jets and other unconsumed fields do not
enter these rational rows; the existing complete physical/source binding and
unused-jet controls remain separate. Exact reference equality can yield zero
primal rows and nonzero field-only tangents.

## 2. Real whole-row identities

Let G=g^-1 and Gh=gh^-1 be real inverses of the submitted matrices. Define

    V = a^2 x G,           Vh = h^2 y Gh,
    L = V - beta beta^T,   Lh = Vh - bh bh^T,
    dV = V-Vh,             dL = L-Lh,             db=beta-bh.

The exact regular and pole rows of the pinned far graph are

    Ra = sum_j beta_j alpha_d[j] - (a/h) sum_j bh_j h_d[j],

    Sa = -a^2 P + a h Ph - a sum_j db_j O_j
         - a sum_jk dL_jk Gamma[0][j][k],

    Rb_i = a^2 x Lambda_i - h^2 y Lambdah_i
           + sum_j (beta_j beta_d[j][i] - bh_j bh_d[j][i])
           + sum_j [G_ij ((a^2/2) chi_d[j] - a x alpha_d[j])
                    - Gh_ij ((h^2/2) y_d[j] - h y h_d[j])],

    Sb_i = 2 sum_j dV_ij O_j
           - sum_jk dL_jk Gamma[i+1][j][k]
           - sum_jk (L_jk beta_i - Lh_jk bh_i) Gamma[0][j][k].

The two reductions in the shift follow without a symmetry assumption:

    beta_j (beta_d[j][i]-bh_d[j][i]) + db_j bh_d[j][i]
      = beta_j beta_d[j][i] - bh_j bh_d[j][i],

    dL_jk beta_i + Lh_jk db_i = L_jk beta_i - Lh_jk bh_i.

Likewise the source's `db_i beta_j + bh_i db_j` equals
`beta_i beta_j-bh_i bh_j`. Combining the complete live chi/lapse gradient
flux before contraction avoids rounding the separately canceling terms. This
is a real-algebra identity, not a claim that the old floating graph passed.
P stays the off-constraint stored physical-P variable; it is not replaced by
P+2Theta or an Einstein constraint.

Define separately assembled real outputs

    Fa = Ra + Sa/Omega,        Fb_i = Rb_i + Sb_i/Omega.

Each of the eight parts and four assembled outputs is to be rounded once from
its own exact rational value. The first direction is the derivative of that
real rational function, then rounded once; it is not the derivative of the
discontinuous rounding map or of entry-rounded inverse values. Calling old
`Assemble` on rounded exact-engine parts does not satisfy the final-RHS
contract. A separate exact assembled path is necessary even when every part
is individually correctly rounded.

## 3. Polynomial common denominator before dyadic scaling

Let C=adj(g), Cr=adj(gh), D=det(g), Dr=det(gh). Here C_ij is cofactor_ji:
the transpose is already included, so G_ij=C_ij/D. Define Q=D Dr and

    N_ij = a^2 x C_ij Dr - h^2 y Cr_ij D,
    K_ij = a^2 x C_ij Dr - beta_i beta_j Q,
    Kh_ij = h^2 y Cr_ij D - bh_i bh_j Q,
    M_ij = K_ij-Kh_ij = N_ij-(beta_i beta_j-bh_i bh_j)Q.

Then dV=N/Q, L=K/Q, Lh=Kh/Q and dL=M/Q exactly. All shift rows have common
denominator Q after clearing their fixed one-half coefficient. The lapse
regular row requires h; all parts can use 2hQ. Parts' numerator polynomial
degrees are at most 11 with denominator degree 7. An assembled numerator has
degree at most 12 with denominator degree 8 after adding Omega. Their complete
dual cross numerators have degrees at most 18 and 20, respectively. Directions
are counted as degree-one atoms; a full product rule preserves total degree.
These degrees refer to unrounded submitted scalar atoms, not matrix-entry
rounding or a rational inverse cache.

## 4. Exact common-scale integer construction

Set s=2^-1074 and U=2^2098. Every finite binary64 value or first direction is
an integer times s with integer magnitude strictly less than U. In this section
capital letters denote those integer atoms, while the submitted integer metric
matrices are denoted m,mh. Write D=det(m), Dr=det(mh), C=adj(m), Cr=adj(mh).
All following expressions use integer additions, subtractions, products and
fixed binary shifts only. Q,N,K,Kh,M now mean the same displayed polynomials
with integer atoms. Because of their degrees,

    dV = s^2 N/Q,   L = s^2 K/Q,   Lh = s^2 Kh/Q,   dL = s^2 M/Q.

Define the following integer polynomials (repeated factors remain repeated):

    Ua = H sum_j B_j A_j - A sum_j Bh_j H_j,
    Ta = -A^2 P + A H Ph - A sum_j (B_j-Bh_j) O_j,

    Tlambda_i = A^2 X Lambda_i - H^2 Y Lambdah_i,
    Tadv_i = sum_j (B_j Bji - Bh_j Bhji),
    Z_i = sum_j [A^2 C_ij Dr X_j - H^2 Cr_ij D Y_j
                 -2 A X C_ij Dr A_j + 2 H Y Cr_ij D H_j],

    Tpole_i = 2 sum_j N_ij O_j - sum_jk M_jk Gamma[i+1][j][k],
    Upole_i = -sum_jk (K_jk B_i-Kh_jk Bh_i) Gamma[0][j][k].

Here Bji,Bhji are the integer beta spatial derivative atoms; no tensor index
is summed implicitly outside the displayed sums. Let

    t = 2^-4297 = s^4/2,       q = H Q.

The integer numerators for the exact parts are

    nRa = 2^2149 Q Ua,
    nSa = H [2^1075 Q Ta - 2 A sum_jk M_jk Gamma[0][j][k]],
    nRb_i = H [2 Q Tlambda_i + 2^2149 Q Tadv_i + 2^2148 Z_i],
    nSb_i = H [2^1075 Tpole_i + 2 Upole_i].

Every part equals t*n/q. For example the gradient factor is
t*2^2148=s^2/2, the pole's degree-three factor is t*2^1075=s^3, and its
degree-four factor is 2t=s^4. This checks the one-half coefficient and all
heterogeneous degrees explicitly. No numerical powers, reciprocal, early
division or floating product are required.

For assembled output i with regular/pole numerators nRi,nSi and integer
Omega atom OMEGA, use

    nFi = OMEGA*nRi + 2^1074*nSi,    qF = OMEGA*q,
    Fi = t*nFi/qF.

The factor 2^1074 clears the s in Omega=s*OMEGA. This is the exact assembly of
unrounded parts. For any part or assembled pair (n,q), differentiate the
integer expressions completely and return the two exact rationals

    value = t*n/q,
    direction = t*(n_dot*q - n*q_dot)/q^2.

This includes Omega's direction when supplied and all complete cofactor and
determinant directions. It requires neither assuming dot(D)=0 nor separately
rounding dot(G). The physical caller's reference-zero directions are only a
specialization. All finite directions must be validated before a zero-factor
shortcut; a zero-primal atom can contribute through its differentiated term.

## 5. Coefficient, degree and bit bounds

The coefficient sums below are unsigned majorants of the fully expanded
integer polynomials. They remain valid with independent nonsymmetric entries
and directions, and without using cancellations or equalities of inputs.

| Polynomial | Degree | Absolute coefficient-sum bound |
| --- | ---: | ---: |
| D,Dr | 3 | 6 |
| any C,Cr entry | 2 | 2 |
| Q | 6 | 36 |
| N entry | 8 | 24 |
| K or Kh entry | 8 | 48 |
| M entry | 8 | 96 |
| Ua | 3 | 6 |
| Ta | 3 | 8 |
| Tlambda_i | 4 | 2 |
| Tadv_i | 2 | 6 |
| Z_i | 8 | 216 |
| Tpole_i | 9 | 1008 |
| Upole_i | 10 | 864 |

For example a C*Dr entry has coefficient sum at most 12; N has two such
terms, K adds a beta product times Q, and M adds two such beta products.
Z has three index values times (12+12+24+24). Tpole has
2*3*24+9*96=1008. Upole has 9*(48+48)=864. Thus these numbers bound every
literal partial sum as well as the final polynomial.

Before combining unlike degrees, the part-numerator majorants are

    |nRa| < 216 * 2^2149 * U^9,
    |nSa| < 288 * 2^1075 * U^10 + 1728 * U^11,
    |nRb| < 144 * U^11 + 216 * 2^2149 * U^9
                              + 216 * 2^2148 * U^9,
    |nSb| < 1008 * 2^1075 * U^10 + 1728 * U^11,
    |q| < 36 * U^7.

Consequently the following deliberately conservative strict bounds suffice.
Powers of two in this table are bounds, not computed sample maxima.

| Integer quantity | Strict magnitude bound |
| --- | ---: |
| nRa | <2^21039 |
| nRb_i | <2^23087 |
| any part n | <2^23090 |
| part q | <2^14692 |
| any part n_dot | <2^23094 |
| part q_dot | <2^14694 |
| part direction numerator n_dot*q-n*q_dot | <2^37787 |
| part direction denominator q^2 | <2^29384 |
| assembled nF | <2^25186 |
| assembled qF | <2^16790 |
| assembled nF_dot | <2^25190 |
| assembled qF_dot | <2^16793 |
| assembled direction cross numerator | <2^41981 |
| assembled direction denominator qF^2 | <2^33580 |

Proof of the dual bounds: differentiating a degree-d monomial generates d
terms, each replacing exactly one primal atom with its direction, which has
the same format bound U. Thus a coefficient majorant grows by at most d.
For parts d<=11 and q has degree 7; for assembled n d<=12 and qF degree 8.
The cross numerators retain both terms, including denominator variation. For
example the assembled cross numerator is bounded by
2^(25190+16790)+2^(25186+16793)<2^41981. Reference-zero specialization can
reduce these majorants but is not needed for them.

The maximum integer degree is 20, whereas the largest cleared coefficient
shift is 2149. These are separate facts. The 41981-bit bound is a bound on
the complete shifted integer numerator, not 41981 additional bits beyond its
degree or exponent. Adaptive removal of common powers of two can reduce work,
but no reduction is needed to prove finite workspace and none is assumed in
these bounds.

## 6. Correct rounding and all remaining integer workspace

Normalize each nonzero primal denominator to positive sign. Its direction
changes sign with it; q^2 is already positive. Reject an exact zero D,Dr,h
(and Omega for assembly) before a zero-numerator shortcut. Validate every
consumed primal/direction atom as finite; NaN/Inf is not made harmless by a
zero factor.

Adopt the explicit strong output-domain contract |exact output|<=maxfinite,
where maxfinite=(2^53-1)*2^971. A larger exact magnitude is an overflow status
even if it would round back to maxfinite. For either value or direction with
positive denominator v and signed numerator u, its exact real is
t*u/v. The domain check is the integer comparison

    |u| <= v*(2^53-1)*2^5268.

For the largest direction denominator the right side is <2^38901; the
largest numerator is <2^41981. No floating maxfinite multiplication occurs.
If u=0, return canonical +0 after all input/domain validation. If u<0 is
nonzero but rounds to zero, return -0. Signed input zero does not alter the
declared exact-zero convention.

For an admitted nonzero magnitude, select the binary64 significand-grid
exponent k: k=-1074 below minimum normal, otherwise
k=floor(log2(|t*u/v|))-52. Then -1074<=k<=971. Determine the exponent with
integer bit lengths and an exact shifted comparison, never floating log or
reciprocal. The desired number of grid units is exactly

    |u| / (v*2^(4297+k)).

The shift lies from 3223 through 5268. Integer quotient/remainder division
gives Qr,R with 0<=R<veffective. Increment Qr iff 2R>veffective, or on an
exact tie iff Qr is odd. Handle the significand carry into a new normal
exponent and a subnormal carry into minimum normal. The strong-domain check
prevents an admitted magnitude from rounding above maxfinite. Return output
bits directly; nonzero tiny outputs and rounding to zero are allowed.

For the largest direction denominator, veffective is <2^38848 and 2R is
<2^38849. The exact min-normal comparison uses v*2^3275, <2^36855. The
overflow comparison is <2^38901. The absolute rounding-error bound is one
half grid spacing, 2^(k-1), represented as an integer exponent/status rather
than a potentially underflowing binary64 estimate.

Every polynomial multiplication/addition/shift is bounded by the coefficient
majorants above if the implementation constructs these identities directly.
Unsigned multiplication partial accumulations are bounded by the full product
of operand magnitudes; signed-sum partial magnitudes are bounded by the sum
of absolute terms. The dual cross multiplication peaks below 41981 bits.
Division can maintain a remainder below veffective and use one extra carry
bit for doubling. Exponent selection need only compare bit lengths and a
shift chosen to align them; it need not materialize a shift exceeding the
largest operand width. Zero cancellation, signed normalization and subtraction
do not require an unbounded intermediate.

Accordingly the proposed 131072-bit resource cap has ample margin for all
mathematical values and the stated carry/comparison operations. A later source
proof must still verify its actual limb multiplication, carry bounds, signed
addition, shift, long division, aliasing, allocation and capacity checks.
This note does not certify an unseen backend or select an implementation.
A 32-bit-limb implementation must use a wide enough carry accumulator (or an
explicit split carry) and reject capacity exhaustion explicitly. The value
bound alone is not a C++ undefined-behavior or allocation proof.

## 7. Minimal future engine interface and independent gates

A small interface can take the 106 named binary64 value/direction bit pairs,
with an explicit part/assembled output mask, and return twelve independent
value/direction results. Each result carries status, bits, exact/inexact flag,
rounding direction and rounding-grid exponent. Shared internal exact
polynomials and denominators are fine; an entry-rounded inverse is not.
The input matrix orientation, spatial derivative order and Ph binding must
be part of the interface contract. A parts-valid flag and an assembled-valid
flag should be separate: representable assembled cancellation need not imply
representable individual parts. If an integration insists on all twelve
representable outputs, declare that stronger conjunction explicitly.

Statuses should distinguish invalid atom, nonpositive scalar domain, exact
singularity/zero denominator, exact output overflow, integer capacity failure
and accepted exact/inexact rounding. Keep active-limb/max-bit and per-row
rounding audit data. There is no global warning suppression, binary64 floor,
post-hoc tolerance or ignored derivative input.

Future independent targets can use exact Fraction dual Gauss--Jordan inversion
with exact nonzero primal pivots, followed by the literal live-minus-reference
raw rows (the pinned independent oracle's structure). This is structurally
separate from the cofactor/common-denominator candidate. Assemble the exact
Fraction rows before rounding. Retain normal/subnormal/tie/maxfinite and
negative-zero hand controls, arbitrary nonsymmetric/traceful first directions,
zero-primal/nonzero-direction entries and gradients, zero exact rows, huge
canceling terms, exact singularity and genuine value/direction overflow.
Input construction rounding is outside the submitted-atom target.

Any later RWM integration must retain the original 2373-record/4900-call
complete-dual registry/targets/tolerances and its independent source/FD/closed
controls, the 15740 original local registry and legacy-near comparisons, with
a fresh source identity. Neither an exact-engine unit pass nor a local helper
pass establishes native/continuum stability, scri-class preservation or black
hole adoption. The reference connection remains Minkowski throughout the
eventual wormhole-to-trumpet requirement. All existing FAIL outcomes remain
linked as history rather than silently upgraded.
