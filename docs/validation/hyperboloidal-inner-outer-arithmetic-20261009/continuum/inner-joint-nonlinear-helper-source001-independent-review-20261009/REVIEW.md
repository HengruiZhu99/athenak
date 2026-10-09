# Independent source review of inner nonlinear helper proposal001

PASS for the declared fixed local Release/Debug query gate. No mathematical or
admission blocker was found in the reviewed source. This is a source review,
not a numerical gate result or an execution release. Root must supply the exact
authorization and capture the outer invocation. The candidate remains held.

The reviewed source index is `6b10f2625f831843745b662a18726c4587701f681f155dc858bfe6b230f225b4`.
All 25 indexed source files and 2,470 unique protected inputs were rehashed,
copied where appropriate, and checked again after review. No candidate import,
syntax/compiler check, CAS, numerical calculation, kernel query, array load,
operator, spectrum, or evolution was performed. The copied held plan and frozen
RWM/lift dependencies remain byte exact.

## Equations and reference binding

The source implements the plan with stored `P=Kphysical-2Theta`; its lapse
correction is a deviation of P from the Minkowski physical trace Phat, rather
than a replacement of the evolved P equation by a conformal Q equation. The
Theta input remains independent and off constraint. There is no BH reference,
BH RHS subtraction, lapse restoration, eta term, floor, or clamp.

Writing A=alpha^2 chi, X=(1-W)G0, the coefficient is
`k=(X+WA)/(X+(1+W)A)`. The implemented normalized mantissa/exponent construction
and relative-seed derivative
`dk=-XA(2 dalpha/alpha+dchi/chi)/(X+(1+W)A)^2` agree with this expression at fixed
external reference position and fixed W/G0. Its W=0 numerator is divided before
final exponent scaling. Product-rule terms are separately scaled, so a zero
primal does not automatically discard a nonzero tangent.

The grouped inverse-metric identity is
`dgi=-gi (g-ghat) ghi`, and `dV=dA gi+Ahat dgi` and
`dL=dV-db beta_live^T-betahat db^T` have the correct signs. Expanding Dchi and
Dalpha in the regular beta row recovers exactly the literal RWM terms plus the
planned B/Lambda and chi-gradient corrections. In particular, the residual
reference-gradient coefficients are dA/chihat and dA/alphahat. The alpha pole is
`-alpha[(alpha+2(1-W))deltaP+deltaalpha Phat+db.dOmega+dL:C0]`.
The beta pole contains both spatial and temporal reference connection terms.
Every complete nonflat reference term is retained; no constant-core connection
is used in the transition. All deviations vanish at the analytic reference.

W is obtained directly from SmoothCutoff, avoiding the old LayerCoefficients
division by live alpha. W=1 returns the frozen full RWM helper before evaluating
any inner coefficient or grouped inner product. The reference connection is
evaluated first from the fixed reference and xyz, as required by that helper.
Position/reference jets must have zero dual seeds under the declared contract;
the implementation explicitly rejects a nonzero radius seed. This is a
live-field directional derivative implementation, not a spatial derivative of
the cutoff or of unprovided higher reference jets.

## Coverage and oracle independence

The fixed grids contain 336 reference records, 4,704 source records, 1,452
coefficient values, 160 relative coefficient duals, 144 closed core witnesses,
6,384 principal coefficient records, 2,520 full-source dual records, 16 invalid
coefficient cases, six invalid source cases, and 18 nonrepresentable cases.
The source grid includes nonflat transition/collar references, varied a and
orientations, an SPD tensor family, independent Theta, and four prescribed
high-contrast families. It does not claim complete nonlinear state coverage.

The multiprecision oracle evaluates the literal unfactored RWM split and the
proposal correction, not the new grouped implementation. Both independently
constructed reference connections are compared in all 4x4x4 slots, with the
helper's stationary temporal-lower-index slots represented by zero. It reports
the literal stationary reference residual separately from the exact deviation
fixed point. The large-A core cancellation witness uses the independently
simplified G0 Lambda target; it does not pretend that 240/280-digit subtraction
of enormous terms resolves that small target. The other closed core target and
ordinary/contrast precision comparisons remain separate.

Actual22 calls the unchanged C0 ConformalRHS with physical P, live kappa1=10/alpha
and kappa2=0, and adds the candidate or frozen RWM gauge once. The existing
analytic Minkowski geometry subtraction is explicit. Its reference kappa1=10
has no damping contribution because reference Theta and Z are zero; this is
not an independently changed physical damping law. Candidate and baseline
geometry rows are required to match bitwise. The four gauge rows and derivatives
are compared against MP, and their three-level centered-FD sequence is gated at
the final level with the declared convergence/floor rule. No best-level
selection or tolerance relaxation is introduced. Input determinant/A-trace
normals are recorded without claiming an undeclared normal threshold. Geometry
row equality is source attribution, not an independent physics validation.

The principal records validate coefficients of P, Lambda, alpha/chi gradients
and beta gradients against the planned source. The previously accepted actual20
principal suite remains an explicit dependency. This new local helper gate
does not itself repeat a complete eigensymbol or prove hyperbolicity.

## Admission and failure preservation

The runner uses fixed local recipe/index paths, requires exact root recipe and
index hashes and an allowed build, isolated unoptimized -I -B Python, bytecode
off and one-thread environments. Source/prerequisite/runtime pins are checked
before MP imports or compilation. Both checkers retain explicit optimization
rejection; no scientific assertion can be disabled by PYTHONOPTIMIZE through
the prescribed runner. The oracle binds the actually consumed recipe hash to
the authorization; its --recipe is not merely checked against an unused local
file. The runner invokes it with the pinned local recipe.

Compiler dependencies must be in the saved source/input inventory before any
probe query. Every fixed mode preserves complete stdout/stderr, return code and
scientific JSONL; both binaries and dependencies are retained and rehashed.
Fresh Release001/Debug001 paths prevent retry overwrite. Post-query input,
source, dependency and executable checks guard the accepted result. A root
outer launcher remains necessary for pre-attempt guard failures and the exact
authorization/runtime invocation; this requirement is explicit in the recipe
and implementation. No native/global/evolution admission follows.

## Arithmetic limits of this review and fixed gate

Finite q.valid is not an accuracy certificate for arbitrary positive live
fields. The near-reference forms
`Dchi=(chi_j-chihat_j)-(chi-chihat)chihat_j/chihat` and its alpha analogue can
lose a tiny relative-gradient term when the live field is far below its
reference. For example, with chi_j=0 and chi much smaller than chihat, the true
Dchi is `-chi chihat_j/chihat`; subtracting two rounded reference-sized terms
can instead give zero. An alpha^2 factor can make that lost contribution
representable and significant. This regime is outside the prescribed full
contrast families and needs a separate additive witness/regime grouping gate
before any universal arithmetic claim.

Likewise, the coefficient's relative derivative formula may overflow for
arbitrary absolute dual seeds at tiny alpha, and a scaled-away primal u does
not justify discarding a sufficiently large absolute-seed derivative. The
declared extreme coefficient tests use relative seeds; full-source dual tests
use the ordinary mixed state. They do not certify all absolute seeds or
arbitrary AD types. The source already explicitly limits the registered scalar
and fixed-reference-position contract. Nonzero-k and scaled products have no
universal correct-rounding claim. These limits do not block the declared fixed
local gate, but must remain visible in its result interpretation.
