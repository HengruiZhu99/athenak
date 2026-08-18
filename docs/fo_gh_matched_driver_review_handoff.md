# Independent review request: matched-driver FO-GH puncture formulation

Please perform a detailed mathematical, code, and evidence audit of branch
`codex/fo-gh-matched-driver-puncture-20260818`. Compare it against parent
`e0d8c653d30d41a676467c23e02f4969f7629156`; do not merge or modify the branch
during review. The controlling conclusion is **FORMULATION NOT ESTABLISHED**,
so look especially for any false positive or false negative in the analytic
stop.

Review these files first:

1. `docs/fo_gh_matched_driver_formulation.md`
2. `docs/fo_gh_matched_driver_validation.md`
3. `tst/test_suite/fo_gh/matched_driver_pullback_audit.py`
4. `tst/test_suite/fo_gh/matched_einstein_map_audit.py`
5. `tst/test_suite/fo_gh/matched_hyperbolicity_audit.py`
6. their three `test_matched_*.py` test modules
7. `docs/fo_gh_artifacts/matched_driver_20260818/`

Please independently check:

- the sign conversion from the published Lindblom--Szilagyi driver to
  `Z=theta+eta_H H`;
- the old-map inverse and its `r^-2p` mixing obstruction;
- the exact pullback under `W=A*chi*T`, including all `D0 W` and `dt W`
  connection terms;
- the `A=alpha^2`, `Y=dA` kinematics and recovery of 1+log/Gamma-driver
  targets;
- whether every named driver production intermediate has a correctly derived
  nonnegative trumpet power, rather than a cancellation-only finite sum;
- independence and completeness of the constrained 58D parent/regular maps;
- the parent principal symbol and Eq. (B16) symmetrizer assembly;
- the use of the physical covector and the transformation
  `A_V=J^-1 A_Z J`, `H_V=J^T H_Z J`;
- whether QR-orthonormalizing each repeated-speed eigenspace makes the reported
  condition number a valid basis-independent subspace-angle diagnostic;
- whether another bounded, puncture-regular variable scaling can avoid the
  observed condition growth, or whether avoiding it necessarily introduces
  `1/(A*chi)` singularity;
- finite-difference sensitivity, chart choices, and any numerical artifact
  that could fake the fitted `r^-7.23` trend;
- the Perlmutter first-bad-state localization and all distinctions between
  observed values, inferred geometry, unavailable diagnostics, and causal
  hypotheses.

Reproduce the three commands in the formulation document. The first two must
exit zero. The hyperbolicity audit must pass its finite-radius checks and then
exit 2 at the explicit conditioning stop. Inspect, rather than merely trust,
the saved text outputs.

Return findings ordered by severity with file/line references. Explicitly
answer:

1. Is the exact matched-driver derivation correct?
2. Is the 58D finite-radius equivalence actually established?
3. Does the characteristic evidence justify the requested hard stop?
4. What additional analytic calculation could overturn or strengthen it?
5. Is any production or numerical-stability claim made without evidence?

Do not interpret the old-control replay as evidence that the unimplemented
matched formulation fixes or causes the instability.
