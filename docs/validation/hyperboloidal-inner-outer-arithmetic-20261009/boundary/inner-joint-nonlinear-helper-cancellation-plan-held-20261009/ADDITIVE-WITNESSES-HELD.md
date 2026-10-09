# Proposed additive normal-target arithmetic witnesses (not executed)

These are not substitutions for any of the fixed15740 main records. They
are source-only proposals for a separately named arithmetic supplement,
requiring review/admission before implementation or execution. Artificial
reference scalar gradients below are coefficient identities, not a claim
that these numbers form a complete Minkowski LayerPoint. The main actual
nonflat reference/source/dual gate remains unchanged.

Use binary-exact positive powers and unit reference scalar fields h=y=1.
Require nonzero normal-target relative error<=2e-10 (the existing core
witness threshold); preserve zero targets separately as absolute<=2e-10.
An independent closed analytic target must be used, not a low-precision
subtraction of huge MP summands. No runtime floors or clamps.

1. Anti-correlated dA: a=2^-300,x=2^601 gives A0=2,Ah=1,dA=1.
   A0/Ah=2 is within a putative aggregate near interval, while both
   individual field ratios are far. This witnesses the individual-field
   branch criterion and can expose cancellation of O(2^601) summands.
2. Amplified dc: a=2^300,x=2^-600,x_i=0,y_i=1 gives
   dc=-2^-600 and a^2*dc=-1. The latter is a nonzero normal target even
   though the live chi is tiny. Keep the complete product factors so its
   value is not erased before the amplification. A0=1 is finite.
3. Amplified dal: a=2^-300,x=2^600,a_i=0,h_i=1 gives
   dal=-2^-300 and -a*x*dal=1, again with A0=1. The unfactored far form
   must retain the live tiny contribution.
4. Separate future dV hardening witness: a=2^-200,x=2^-100,
   gi=diag(2^501,2^-501,1),ghi=I gives dV_11=1. This tests the tensor
   extension, not the primary three-scalar patch; no current metric-contrast
   source coverage is inferred from it.

For live-field duals, an independently differentiated closed target should
check both components of each witness. Keep the same relative field seeds
as the admitted high-contrast contract; arbitrary absolute seeds that make
relative derivatives overflow remain outside that contract. A zero primal
factor with a nonzero dual seed must still retain its product-rule term.

Optional branch-local arithmetic controls may use unit references and
primal x/h or x/y at1/2 and2 and at their neighboring binary-exact offsets
plus/minus2^-40, with fixed finite field-dual seeds. Both real formulas
have the same derivative; the floating algorithm's branch is selected by
the primal. These controls can check a finite stencil, not prove universal
floating C1 continuity. The original actual20 dual/three-level finite
difference checks remain mandatory and unchanged.

The stated direct-far evaluation corrects an algebraic representation, not
the PDE, damping, P storage or reference. Its success must still be measured;
these exact targets are proposed before any numerical query, not post-hoc
normalization of a failed source row.
