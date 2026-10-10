# Gaussian v2 timing failure: source mechanism and precision proposal

This is a source and compact saved-result assessment. The original v2 timing gate remains **FAIL**: 20 records, 11,816 identity checks, 1,692 compact precision checks, six failed identities, unchanged sources. No candidate is imported or evaluated here. Neither scientific JSONL is decoded; both remain hash/size metadata. This note does not qualify the full 5,010-record gate.

## Saved association

The six names are `connection_conformal_011`, `012`, `013`, `022`, `023`, and `033`. The saved cumulative stdout adds all six at record 9: 110 digits, epsilon=3/4, nominal r=1-10^-18, native t=1/10, direction (2,-3,6)/7. The second block, at 150 digits, adds none. The largest saved scaled error is 2.1852675344935690572361968577952126847368959786740140369629208700537002560745741791596216688728670743704820930e-42, versus the unchanged 1e-55 threshold. The saved result contains no precision-comparison failures.

The independent saved audit a8d0797e/ef04dd1f reaches the same association. Its audit and this note preserve the failed child, outer and root receipts; actual process completion does not turn the failed timing stage into acceptance.

## The checked identity is correct, but its raw embedding side is ill-conditioned

Write z^a=(t,x^i), X^A=(T,Q^I), Q=x/Omega, E^A_a=partial_a X^A. `geometry.embedding_connection` forms

    Gamma_phys^a_bc = (E^-1)^a_A partial_b partial_c X^A

using the generic determinant/cofactor inverse in `taylor3.py`, then `diagnostics.all_checks` tests the five-term conformal transformation

    Omega Gamma_phys^a_bc - Omega Gamma_bar^a_bc
      + delta^a_b Omega_c + delta^a_c Omega_b
      - bar(g)_bc bar(g)^ad Omega_d = 0.

This has the correct signs for g_phys=Omega^-2 bar(g). The spatial pair a=0 has neither Kronecker term. Its label `bounded_outer_conformal` identifies the primary ADM branch; it does not assert that the independently constructed raw embedding connection has bounded intermediate arithmetic.

On the exact outer branch, with n=x/r and L=Omega-r Omega_r,

    Q_i^I = Omega^-1 (I-nn)^I_i + L Omega^-2 n^I n_i,
    det(partial Q/partial x) = L Omega^-4.

Individual Cartesian triple products in the cofactor/determinant expansion can be O(Omega^-6). Their radial rank-one cancellation recovers O(Omega^-4), so a nominal d-digit evaluation can lose two powers of Omega in the inverse's relative accuracy. In the time row, the subsequent contraction combines O(Omega^-3) second embedding derivatives to obtain Gamma_phys^0_ij=O(Omega^-1), losing another two powers before the final Omega scaling. A useful conservative *conditioning budget*, rather than a proved forward-error bound, is therefore

    absolute error of Omega Gamma_phys^0_ij = O(10^-d Omega^-4).

At this registry endpoint Omega=(1-r)(1+r) is about 2e-18: four inverse powers can consume about 72 decimal digits. The observed 110-digit residuals near 1e-42 and the absence of new failures at 150 digits are consistent with this mechanism. The source and compact result alone do not prove which intermediate supplies the observed error constant; no intermediate error attribution or exact precision scaling has been measured here.

The 188-component precision comparison covers compact fields, their time rates, Omega, J and D. It does not compare every raw connection component. Passing those comparisons therefore does not certify the ill-conditioned raw-connection identity. Conversely, these six failed comparator rows alone do not establish a geometric or gauge equation defect.

## Proposed precision-only sibling, still held

Keep every mathematical module and comparator formula byte-identical. Retain the identity/component thresholds 1e-55, all rational events, nominal admission selectors, 318 units, 20 timing records, 5,010 full records, 3,330,320 full identity checks and 470,752 full compact comparisons. Retain all geometry, height quadrature orders and their separate 1e-30 scalar context gate, payload/resource caps, raw physical-reference RWM binding and negative slicing control.

Use fixed levels 180 and 220 digits. Preserve the existing root guard-digit offsets: scalar residual tolerances 1e-155 and 1e-195, numerical widths 1e-160 and 1e-200, 16 Newton opportunities. Increase only each level's bisection budget from 512 to 768. At the endpoint, the four-power conditioning budget then leaves approximately 108 and 148 digits before constants; this is a design margin, not a uniform rounding theorem or a promised pass. The more stringent scalar root widths also avoid leaving the old scalar-root error budget as a possible dominant term.

The fixed initial scalar bracket spans are below 2 on the declared registry. For inner events the bound is 16|epsilon|sigma/(3pi); for outer events r>=.95 gives |z|<1/8, |p|<=1/2 and sigma=.35<1/2, so the implemented polynomial bound also gives span<2. A 768-step pure bisection fallback reduces such a span below 1e-229 (2^10>10^3). This exceeds the required 1e-200 width budget without relying on Newton convergence. Scalar finite-precision endpoint signs/residuals and width remain numerical gates, not outward interval enclosures.

The 80-digit unit recipe remains unchanged. Only the two geometry precision levels, their scalar root settings, and truthful precision metadata change. Replace the hard-coded `110_vs_150_compact_component` label with `180_vs_220_compact_component`; update PLAN/SCHEMA and lineage/admission pins accordingly. Do not alter the 1e-55 thresholds or mark any currently admitted connection row diagnostic-only. Do not manufacture a successful v2 timing prerequisite: a fresh sibling needs its own units, timing and measured-cost review before any full release.

No sibling implementation, numerical execution, native eligibility release, raw RHS query, inverse coverage, nonlinear/BH admission, or continuum claim is made by this proposal.
