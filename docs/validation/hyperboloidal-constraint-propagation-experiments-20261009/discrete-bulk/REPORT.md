# Frozen bulk discrete stencil gate

Passed full20 constant-coefficient interior audit; no native runtime source was edited.
Launch HEAD b37a20f2d7a8ccc42148f1a814a80b2957e17b53; runtime implementation 27c19d20696ea6dd4704032c51dfd026218f64f2.

- 1080 full20 actual dual kernel/constraint cases, 378 composed nonzero modified-covector complete basis cases and 60 flat gauge-constraint cases.
- Maximum canonical principal matrix error 6.349750e-13, left-eigenfield residual 8.518553e-13, normalized basis condition 11.529153.
- Corrected native gauge defect formulas agree to 9.387019e-13; composed gauge constraint leakage 1.366576e-12.
- Exact modified-zero cases have N²=0 to 5.149339e-14. Nonzero Nyquist modes have a KO-damped Jordan propagator and a uniform-h D+ norm bound 133.69501 for h<=.125 and epsilon=.1. No uniform diagonalizer or contraction is claimed.
- Actual nested Dx manufactured second-Hessian refinement ratios: [16.02792698546989, 16.006988199417705, 16.000691290216864].
- All 372 recorded production/test/config/audit input hashes unchanged through compile/run/check. All four commands returned zero.

The first incorrect symbolic Z shift defect omitted delta_i beta_i/6. The failed assertion and exact source, rerun stdout/stderr and receipt are preserved under failed-first-formula/. The full actual kernel exposed the term before any accepted claim.

Scope: flat constant-coefficient bulk, principal-only gauge, algebraically constrained det/trace variables, both actual RHS and the same constraint stencil. Actual native diagnostics using the original Dxx are a separate acceptance measure. Wide-layer coefficient gradients/product defects, nonlinear constraint closure, native ghost/active-line KO, near-Nyquist uniform energy and global/native stability remain outside this gate.

Run run_audit.py with the recorded Python environment to reproduce; exact commands, compiler, source and executable hashes are in receipt.json. Large full20 matrices remain ignored and hash-indexed.
