# Read-only review prompt: Brill N256 stage-3 chi failure

Repository: https://github.com/HengruiZhu99/athenak

Evidence branch: `codex/amr-history-record-replay-brill-20260817`

Diagnostic code branch: `codex/brill-chi-stage3-mechanism-20260818`

Diagnostic code commit: `0b398079cb2e33bbb8dbb485078b13d83ecb71b8`

Artifact directory: `docs/investigations/brill_chi_stage3_mechanism_20260818/`

Please perform a skeptical, read-only review of `REPORT.md`, `conditional_control_summary.json`, `evidence_manifest.json`, the v9 phase-1 CSV/JSON files, the v14 control log/replay ledger, the spectra/timeline products, and the committed source.

Established observations:

1. Native N256 cycle 5546, RK stage 3 has two active chi candidates crossing from affine bases about 0.918 to about -1.33/-1.34.
2. Removing the production advection contribution alone leaves both candidates positive. Curvature and KO are not individually necessary.
3. Owner ghosts equal authoritative active values; stitched and block-local O6 derivatives agree exactly.
4. Radial advection changes from positive O2 to very negative O4/O6, indicating a strong short-wave/order-sensitive radial structure.
5. In a stitched all-variable patch, chi ranks ninth by the chosen high-frequency fraction; Axy, Gamma_y, Gamma_x, Khat, and Theta rank higher.
6. The two target blocks requested refinement in 271 retained rows beginning at cycle 4542, but a 735-cycle evidence gap prevents a claim of uninterrupted requests.
7. The sole control applied exact earlier refinement at 0 ULP timing and exact tree checksum. It then failed earlier at `t≈10.43145 M` with 1,066 invalid chi parent stencils at a different location. It did not reach the native active-cell failure.

Primary questions:

- Is `ADVECTION_DOMINATED_FAILURE` the correct proximate classification of the native stage-3 crossing?
- Does the O2/O4/O6 sign reversal indicate a near-Nyquist mode, or can it arise from a smooth but steep profile under these stencils?
- Given the control’s distinct earlier boundary-parent failure, what can and cannot be inferred about parent under-resolution and replay-tree lag?
- What is the smallest trigger-only diagnostic that distinguishes RK-produced invalid active chi from active restriction overshoot, communication/BC corruption, and same-level coarse-refresh overshoot in the V14 control?
- Are there arithmetic, indexing, authority-prefix, or diagnostic-observer flaws in the committed code or evidence interpretation?
- Which existing cheap indicator, if any, is worth a bounded future test, and what evidence threshold should gate that test?

Please clearly separate observations, mathematical deductions, hypotheses, and unsupported possibilities. Recommend at most one smallest decisive next diagnostic or source-level correction.

Do not recommend chi floors, clipping, absolute values, weakened positivity gates, broad parameter sweeps, or unsupported convergence/Figure-3/continuum claims. Do not treat the diagnostic history branch as production AMR policy.
