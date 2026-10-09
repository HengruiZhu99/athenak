# Independent source-only review of the held coupled-inner suite v2

The mathematical/probe/analyzer source review passes. Execution admission of
the current one-shot runner does not pass without an explicit Python
optimization guard. Nothing in this review executes the proposed gate.

The reviewed source index is d8df2620a1c8d4c8542caf4aeb52699d10b016cc700b8c2d88657a372e262479,
and the recipe is e14ba9a0bf913babddba1c6fdba252cdedd6897d6bf84184f13f4b0dd548c676.
The original count-error proposal and the corrected ASSESSMENT-v2 remain
unchanged. The review rehashed 1,438 unique declared files before and after
source inspection: 13 indexed local files, 15 external context entries, and
1,418 recipe dependency entries, with duplicates collapsed for this count.
Compiler and Python executable files were hashed only, never invoked.

## Admission issue

check_exact.py and analyze.py implement their scientific acceptance tests with
assert statements. run_gate.py invokes both with its pinned Python executable
but neither prevents inherited PYTHONOPTIMIZE nor explicitly rejects a
nonzero sys.flags.optimize. With Python optimization enabled those assertions
are removed, while both children still construct passed=true summaries. The
runner's checks of those summaries therefore do not restore the omitted
tests. The recipe does not pin an execution environment that excludes this
case. This finding follows directly from the source; no optimized execution
or counterexample batch was run.

A separate exact outer launch setting PYTHONOPTIMIZE=0 and protecting that
environment can address the immediate admission condition. A fresh minimal
guard revision can instead explicitly reject optimization and isolate the
child Python processes. The v2 source must remain preserved either way.
This is a guard finding, not a failure of the proposed principal equations.

## Mathematical and extraction checks

The actual tensor extraction retains the production constrained-20 lift and
the same separated derivative-order normalization. Its base metric is SPD
with determinant one in the two declared Cartesian/oblique frames. The trace
and Theta jet factors use Omega, and the normalized trace variable is stored
P/Omega. The lapse pole couples to P, not to physical K=P+2Theta. Both lapse
and shift poles are assembled exactly once by the preserved rwm::Assemble.
The constant reference connection is deliberately zero: this is a principal
probe, not a nonflat source or reference fixed-point test.

The candidate gives A0=alpha^2 chi, B=(1-W)G0+W A0, mu=B/A0,
f=1+2(1-W)/alpha, and ec=2mu^2/(1+mu)^2. The additional chi-gradient term
changes the wave-map ec=1/2 to this ec. The general-law controls use the same
special coupling while varying f independently. The literal scalar block
has A_t=-2ell/3+c/3-h/2+2Lambda/3, retaining the corrected Lambda term.

For H=h+2c, V=Lambda+2c and X=c-CV, the displayed transformation and inverse
give four scalar wave pairs with squared speeds f,q,1,1, where
q=(4mu-2ec)/3 and C=2(1+mu)^2/(4mu^2+5mu+3). The cancellation is
C(q-1)=2(mu-1)/3. This transformation has no division by q-1, f-q or mu-1.
The explicit transverse and tensor left bases and their inverses also agree
with the literal blocks. They remain algebraically defined at the tested
mu=1, q=1, f=1 and q=f collisions for positive f and mu. This source review
does not execute the Fraction identities or compute numerical residuals.

The 118 fixed cases are 72 candidate grid cases, eight extra nonharmonic
mu=1 cases, two candidate q=f cases, and 36 general-law controls. The
analyzer independently reconstructs this registry and all 20 expected rows.
Its explicit two-sided inverse and left-eigenfield tests establish the
intended completeness check without a numerical rank or eigensolver. The
18 Fraction cases are fixed rational conjugacy/inverse tests, accompanied
by the pencil identity; they are not a nonlinear or all-parameter batch.

## Other runner checks

The runner binds the consumed recipe to its local resolved path, requires
the exact source-index/recipe authorization and matching scope, and uses a
single fresh attempts/gate001 directory. It records command exits, complete
stdout/stderr, executable hashes and actual compiler dependencies, rejects
undeclared dependency paths or any command stderr, checks all 118/18 summary
counts, requires Release/ASanUB raw outputs to be byte-identical, and finally
rejects protected-source drift. The indexed recipe pins the compiler, Python,
production files and prior header dependency context. These safeguards are
consistent with the stated one-shot scope apart from the optimization issue.

## Limits

This is a source-only review of a frozen constant-reference principal test at
strictly positive finite alpha and chi. It is not a measured matrix, exact
symbol, nonflat fixed-point, finite-frequency, lower-order, puncture,
black-hole, native or evolution gate. No uniform hyperbolicity statement at
alpha=0 or chi=0 follows. The candidate has not been selected for production;
the later goal remains a wormhole-to-trumpet transition while retaining the
Minkowski hyperboloidal reference.
