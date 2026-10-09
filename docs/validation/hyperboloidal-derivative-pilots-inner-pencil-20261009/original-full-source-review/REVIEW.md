# Independent source-only differentiated-ray review

Disposition: the analytic formulas are consistent with the independent
fixed-k pencil c09efb27. Standalone source admission is **blocked** by one
recipe-path binding defect. No numerical result or derivative acceptance is
claimed. All six original sources were copied and pinned before their content
was read; no scientific import, syntax execution, CAS, array read, ray query,
inverse-map calculation or evolution occurred.

The reviewed original main source is
49bb7fa0d80f471464f7b413ef3f1f524e301330d539a0d3ec41a7f0e6a68ee5
under original index9649d93e6f34a1c9e2b79c542509258b6a51af73eadfbfbc3cc9b7d1fc808146.
These bytes are preserved here and must not be rewritten by a correction.

## Admission defect

`main()` reads `recipepath=Path(args.recipe).resolve()` and runs that parsed
recipe. Its authorization instead pins only `HERE/derivative-recipe.json`.
There is no equality or consumed-file hash check linking those two paths.
Thus a different command-line recipe could alter points, levels or gates while
the local authorized recipe remains unchanged. The before/after inventory
also protects only the local recipe. Require the consumed resolved path to
equal the authorized local path, or require its exact hash and explicitly
protect it. This is a source prerequisite defect, not a failed numerical or
mathematical derivative test. The author reports a separate launcher that
passes the local recipe; that does not repair standalone `main()`, and that
launcher was not independently reviewed here. Both root and author were
notified before any derivative execution.

## Analytic checks by inspection

The cutoff logistic derivatives, Omega''' and
L''=-Omega''-r Omega''' are correct. The inverse-radius relations
r_q=Omega^2/L and r_qq=r_q*d_r(r_q) use ordinary derivatives. The radial
H_i,H_ij,H_ijk formulas include the v_qq and all transverse terms. The
Cartesian core avoids radial division at the origin. Native normal data
G=s*Omega retain the complete reference shift/height terms and factor the
outer pulse as Omega^4. They agree algebraically with the prescribed normal
datum, including its negative inverse-map sign.

For fixed future-null k, lambda_a=h(delta_a0-H_i delta_ai)/D,
lambda_ab=-h H_ij y_a^i y_b^j/D and y_ab=-k lambda_ab have the correct
signs. `pullback` retains the second source derivative and the y_ab term.
Product/quotient jets therefore include both denominator first derivatives,
its second derivative, and all numerator cross terms. The independently
built K_i=-k^j H_ji and K_ij=-k^l H_lij test the required third graph jets.
The alternative F=G/h,K=D/h quotient is algebraically equivalent; it shares
the elementary Jet2 arithmetic and is a factoring check, not an independent
derivative engine.

The stable D=hK expression is correct for the fixed future-null ray and
h^2-b^2=Omega^2. Lorentz/null identities, positive D, first/second implicit
root identities and direct-versus-factored D jets are retained as separate
gates. The fixed boost is constructed outside differentiation and its output
components remain inertial Cartesian. No derivative of the event-dependent
choice of boost is taken or needed for these fixed-frame evaluations.

The initial graph/wave oracle is correct: with c=Omega/h and A=s/c,
u_T=A,u_i=-H_i A,
u_TT=(-2 H_i A_i-A Delta H)/c^2,
u_Ti=A_i-H_i u_TT and
u_ij=-H_i A_j-H_j A_i-A H_ij+H_i H_j u_TT.
It is independent of the differentiated-ray quotient. The flat constant and
affine controls and the CMC harmonic l=0,1,2 closed forms have the stated
normal datum. Packed Hessian and wave-trace indices are consistent.

## Coverage, cost and limits

The recipe has100 native rows,100 closed-form rows and28 initial rows.
Five angular levels sum to9472 rays. Including both precisions and the
declared boost counts gives493568 differentiated-ray evaluations, plus
163840 coarea value nodes. The center/core/exact outer branches reduce cost
for some events. The worst transition path is considerably more expensive
than this ray count: each ray root permits512 iterations, each of which may
invoke a512-iteration radius inversion and a partial height quadrature of
up to128 nodes. This is finite by the fixed caps but does not establish an
acceptable wall-clock cost. A separately sourced/released small timing gate
would be reasonable before the full batch; no thresholds or adaptive retries
are recommended here.

The fixed80/110-digit comparison applies to native final-level results;
controls and initial limits are separately checked against analytic targets
at both precisions. The final full64 comparisons retain polar/azimuthal
refinements separately. These are finite-event gates and do not certify a
uniform quadrature error. The current events deliberately exclude long-time
failure-position derivatives. Only u gradients/Hessians and u/Omega values
are claimed; no differentiated conformal field, inverse/native target time,
Jacobian, injectivity, caustic or spacelike-slice verdict follows. No
continuum/native or later wormhole-to-trumpet acceptance is implied.

The latest user requirement remains a later wormhole-to-trumpet inner
transition with the Minkowski hyperboloidal reference retained throughout.
