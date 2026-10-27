# Independent review of original-equation shift acceptance

Accepting a finite candidate from independently recomputed original residuals
is mathematically appropriate when it satisfies every disclosed public
criterion. Requiring the auxiliary mixed-norm flag as well is a sufficient,
more conservative test, but is not necessary. Replacing that sufficient gate
with explicit physical/conformal original-residual checks does not increase
the requested physical residual tolerance. The change must remain disclosed,
retain the auxiliary status/targets, and preserve the previously failed runs.

This review adds only `test_shift_acceptance_audit.cpp` and evidence/docs. It
does not edit CompositeShiftStage or FGMRES. The independently compiled test
runs against the parent's new acceptance-hook API. It performs algebraic
verification on1280 mapped composite3D cells, with general3D forcing and
inhomogeneous vector boundary data. No physical gauge source, evolution or
scientific milestone is claimed by this test.

## Necessary versus sufficient norm conditions

Let normalized physical and conformal cell weights be p_i and q_i. For the
original vector residual r, define

    P² = sum_i p_i |r_i|²,
    C² = sum_i q_i |r_i|²,
    S² = (1-m)P² + m C².

The inner FGMRES uses S. With

    c = min_i [(1-m)+m q_i/p_i],

S² >= c P². Therefore S <= sqrt(c) T **guarantees** P <= T. Its converse is
false: a residual can have P,C <= T while S exceeds sqrt(c) T. The published
physical target remains

    T = max(atol, rtol * physical_RMS(original_source)).

The relative target must not silently be changed to use the affine-lifted RHS
or a preconditioned residual. For the parent's proposed dual-norm contract,
the actual acceptance set is

    finite original residuals;
    P <= T;
    C <= T when m>0;
    max_component |r| <= supplied point guard when enabled.

Directly verifying these conditions is a valid convergence criterion for the
specified discrete boundary-value equation. The auxiliary convergence flag
still describes whether its stronger surrogate target was attained. Neither
criterion establishes discretization accuracy, solution conditioning,
interface order, correct physical boundary derivation or evolution stability.
Those remain separate scientific gates.

Explicitly checking C is essential. A balanced norm alone can hide one norm
when its mixing coefficient/measure is small. The old Stage code explicitly
checked P and the point guard after the auxiliary flag; its norm bound was
derived for P. The new explicit C check should be described as an additional
original-residual protection, not assumed to follow from the old P bound.

## Forced one-iteration budget checks

The test builds the original source with nonzero affine boundary data and
reapplies the actual full operator with those data for every verification.
A separate call to classic FGMRES, **without** the external acceptance hook,
uses the same source, preconditioner, balanced norm and sufficient bound.
The two candidates are compared with independently accumulated norms.

| One-iteration case | Physical RMS | Conformal RMS | Inner target | New Stage result |
|---|---:|---:|---:|---|
|Public target1e-10 met|9.50000e-11|9.49739e-11|7.18219e-11|Accept; inner flag false, external flag true|
|Public target1e-10 missed|1.10000e-10|1.09970e-10|7.18219e-11|Reject|

The passing case's maximum component residual is5.90e-10, below the unchanged
1e-7 point guard. Its balanced RMS is9.49870e-11: it misses the auxiliary
7.18219e-11 target while satisfying both requested original RMS tolerances.
The failure at1.10 times the public target confirms that original tolerance
was not relaxed. Both cases use the same one-iteration budget.

## Mixed-measure and point-guard counterexamples

A single positive-radius cell receives chi=1e-80, with all other chi=1. The
original log-volume measure retains every cell; no point is excluded or
floored. Localized original residuals are produced by changing the original
source at one node while retaining the actual nonzero candidate and boundary
callback. Independent fresh `Apply(beta,boundary)` checks give:

* Near the volume-dominant node, P=1.10e-10 > T=1e-10, while S=1.38e-12 < T
  for m=.9999. A naive balanced-threshold check would accept the bad candidate.
* Outside that node, C=1.10e-10 > T, while P=1.45e-68 and S=7.78e-11 < T for
  m=.5. Physical RMS alone or a naive balanced threshold would miss this.
* With an explicitly chosen public RMS target1e-5 and the unchanged point
  guard1e-7, both RMS norms pass but one component is2.00e-7. The point guard
  rejects that candidate.

The new Stage rejects all three **initial candidates**: none reports success
at iteration0. One genuine FGMRES correction then brings each final candidate
inside all its requested bounds, where acceptance is appropriate. The final
original norms are independently recomputed and compared with Stage's flag.
These corrected solves are not mislabeled as tests whose final candidate must
remain a failure.

## Affine roundoff and zero inner residual

The affine decomposition is exact in real arithmetic, but evaluating
A_0(beta)+A_b(0) and A_b(beta) can round differently. The test uses a constant
finite Cartesian beta and its matching inhomogeneous boundary data. Choosing
the source from the former expression gives **exactly zero** inner residual,
yet fresh evaluation of the original equation gives

    P = C = 2.02873451281723e-14,
    requested T = 1.01436725640862e-14,
    maximum component = 1.13886235167133e-13.

The new Stage correctly reports failure with zero iterations. It neither
accepts the zero auxiliary residual nor divides an Arnoldi vector by zero.
This case demonstrates why the original equation must be reapplied, rather
than treating the affine residual as an infallible oracle near roundoff.
A separate tiny identity-system test explicitly makes external acceptance
false at zero inner residual; it also returns failure safely. NaN original
source/residual data are rejected.

## Failed scientific control remains failed

The original strict1e-12 coupled control is preserved verbatim and hashed
from `enriched-line-pilot-20261027/tight-tolerance.stderr`. It stopped after240
iterations with physical RMS9.93097e-13, conformal RMS8.54696e-13 and maximum
7.74802e-11, while the last auxiliary RMS was9.26484e-13 against its stricter
approximately7.07e-13 target. Its last reported candidate fits the proposed
original bounds, but that observation does **not** turn the aborted trajectory
into a completed successful evolution. A new bounded run must recompute and
satisfy the original equation and boundaries at every applicable stage.

## Reproducibility and outcome

`evidence/shift-acceptance-audit-20261027/manifest.json` records the exact
compiler command, dependency hashes, compiled-source archive, binary hash,
complete results and0 exit status. The new test compiled in2.41 seconds and
ran in.782 seconds; its compiled dependencies remained unchanged. The old
failed control's exact stderr and hash are included separately. The test
source is ready for a parent-owned CMake/CTest target; this review did not
modify build registration or shared solver headers.

The independent evidence supports the new acceptance policy, with no reason
to retain the auxiliary flag as a mandatory extra gate once the complete,
finite original criteria are explicitly checked. It also supports retaining
`inner_norm_converged`, `external_acceptance`, both targets and stop context,
so a budget stop, roundoff obstruction and verified original solve remain
distinguishable.
