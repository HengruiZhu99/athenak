# Source-only fallback: analytic projected-constraint point maps

This is a proposal, not an executed scientific gate. Keep both ordinary-FD
attempts failed and byte-identical. No new C++ source/API/compilation is needed.

For fixed Cartesian x, J,m,channel,phase, the existing
`--manufactured-rate-batch` returns actual tangent RHS22 and physical initial
constraints8 after the complete analytic `SeedPhysical`/`LiftPhysical` map.
The constraints use the supplied reference metric/chi jets through second
order and A through first order. They do not require higher reference jets.
The dual constraint variation is linear in the three envelope data
`(W,W_rho,W_rhorho)` at that point. This is pointwise linearity, not a claim
that three radial functions span all fields on the domain.

Query envelopes 0,1,2, whose WJets are `(1,0,0)`, `(rho,1,0)`,
`(rho²,2rho,2)`. From their eight-vectors Q0,Q1,Q2 form

    map_v   = Q0
    map_d   = Q1 - rho Q0
    map_dd  = (Q2 - rho² Q0 - 2rho map_d)/2.

Contract those three maps with the exact Jacobi polynomial envelope jets of
the *same* modal coefficient vectors X, Jbulk X, Jsat X, and their sum. This
directly evaluates C_ref[Phi(X)] and C_ref[Phi(Y)] without differencing the
22 reconstructed primitive fields. It does not evaluate C_ref[L_actual Phi(X)]
for a general nongauge seed; that continuum comparator remains separate.

Proposed held controls: at the same 21 centers/all8 J0 channels, direct
envelopes3,4,5,6 must agree with the three-map contraction at the existing
5e-11 scaled threshold. Independently reconstruct the RHS22 WJet map from
the same three manufactured calls and compare it with the existing source
point maps at every center (same threshold, with per-row and global errors).
This verifies that the new readback binds the same actual source, reference,
seed/lift and envelope conventions. Rehash the unchanged executable, API,
operator, old attempt source/call receipts and old NPZ/schema before/after.
Reject nonfinite data and preserve all stderr/failures.

After those controls, produce analytic initial/bulk/SAT/total physical-eight
vectors for all 14 fixed witnesses at all 21 centers. Check total=bulk+SAT and
the four pure-gauge initial zeros with the existing thresholds. Compare every
available ordinary-FD value/Richardson vector from both stopped attempts with
these analytic projected vectors, retaining all differences and all failed
FD gates; do not require the stopped continuum sequences to pass by relabeling.

For gauge-only seeds the continuum constraint-rate zero is supported by the
stationary Einstein-sector tangency derivation and the separately frozen
actual gauge-rate controls. If using that exact identity to label a gauge
projection defect, cite it explicitly and retain the current failed ordinary-FD
cross-checks. Do not substitute it for a general nongauge continuum comparator.
The latter remains an open part of the full original fourteen-witness defect
gate unless a separately reviewed source-derivative method is supplied.

No eigenspectrum, propagation, CPBC, continuum stability, physical energy or
nonlinear constraint closure follows from these pointwise measurements.
Execution requires a separate root-reviewed source and hash-bound release.
