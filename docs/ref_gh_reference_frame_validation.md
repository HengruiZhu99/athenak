# Reference-frame FO-GH paused validation report

## Scope and conclusion

This branch adds an independent 50-field first-order generalized-harmonic module
using a fixed reference frame.  It is based on parent
`68000b4a753056d5f18333a63175d9e003a32300` and does not modify fluid coupling.

The flat-reference foundation is implemented and locally verified: algebraic
frame/source audits pass, exact Minkowski is preserved to roundoff, the linear
wave converges at approximately fourth order, and the robust-stability noise
decreases through one light-crossing time at all three tested resolutions.  A
tabulated stationary Schwarzschild n=2 1+log trumpet reference and common
ADM/history diagnostics are also implemented.  The exact regular stationary
state remains bounded through the local t=1 test at three resolutions.

Work is intentionally paused here for formulation review.  This is **not
puncture qualification**: the wormhole-to-trumpet transition, analytic physical
outer boundary treatment, SMR, restart, GPU, and long-time puncture stages are
not implemented or claimed.  In particular, no result through t=20M exists.

## Implemented surface

- Exact 50-field state: 10 symmetric `Psi_ab`, 10 `Pi_ab`, and 30 `Phi_iab`.
- Fixed-size Kokkos-compatible reference geometry and Minkowski provider.
- Characteristic field helpers and reference-frame algebra checks.
- Standard coordinate-GH source construction and product-rule frame transform.
- AthenaK runtime/task-list/mesh/ADM integration under the separate `ref_gh` module.
- Exact Minkowski, periodic linear-wave, and robust-Minkowski problem generators.
- Generated stationary n=2 1+log trumpet table and a device-safe interpolating
  reference provider with first and second derivatives.
- A stationary-trumpet problem generator with an initial-RHS fail-closed check.
- Reconstruction of common ADM variables and fixed-region H/M histories, plus
  native GH, reduction, curl, conditioning, characteristic-speed, and bad-state
  histories.
- Python audit entry points under `tst/test_suite/ref_gh/`.

## Formulation erratum

The controlling draft gave the lower-order scalar-wave terms in the `Pi`
equation with signs inconsistent with its own definitions
`Pi=-n^a partial_a Psi` and `Box Psi=S`.  Direct comparison with the coordinate
GH first-order equation requires

```text
- Phi_i D^i(alpha) + alpha S
```

rather than the opposite signs.  The implementation uses the consistent signs.
The independent source audit agrees with the coordinate expression to
`2.78e-17`; the draft signs disagree at order `2.8e-1` in the audit sample.

## Verification evidence

All executable tests used a clean detached worktree and a Kokkos Serial Debug
build.  Compact numeric output and build provenance are committed in
`docs/fo_gh_artifacts/reference_frame_20260818/`.

Algebra/source audit maxima:

- frame duality: `1.11e-16`
- frame orthonormality: `1.11e-16`
- metric round trip: `8.33e-17`
- derivative round trip: `9.71e-17`
- transformed wave product: `1.11e-15`
- characteristic eigenvalue error: `2.00e-15`
- symmetrizer residual: `1.11e-16`; minimum eigenvalue `1.251e-1`
- scalar source versus coordinate GH: `2.78e-17`

Linear-wave `L1` errors at 8, 16, and 32 cells are respectively
`3.5510e-10`, `2.3566e-11`, and `1.5295e-12`.  The observed orders are
`3.9134` and `3.9456`.

The one-crossing robust-Minkowski runs at 8, 12, and 16 cells ended with growth
factors `0.8375`, `0.8182`, and `0.8016`.  Their measured growth rates are all
negative, with no resolution-growing mode in this bounded test.

The trumpet table has `R0=1.312408289173401`,
`Rc=1.540569415042095`, and `C2=1.5543095902183304`.  Its maximum sampled
implicit-equation residual is `4.89e-15`.  The independent interpolation audit
found maximum value, first-derivative, and second-derivative errors of
`4.70e-13`, `4.32e-10`, and `8.45e-7`, respectively, over the resolved range
`r>=1/64`.

### Stationary-trumpet observations

The bounded uniform-grid t=1 runs on `[-2,2]^3` used 8, 12, and 16 cells per
direction.  Their regular-field Linf errors were `6.85e-10`, `6.84e-10`, and
`6.25e-10`; native constraint Linf errors were `6.46e-10`, `5.68e-10`, and
`4.01e-10`.  The native integrated GH constraint at t=1 was between
`2.45e-17` and `7.16e-17`, the frame Gram condition estimate remained one, and
the bad-state flag remained zero.  This supports only a short stationary-state
gate, not wormhole puncture stability.

The common ADM finite-difference constraints show a sharply different pattern.
At t=0, whole-domain H L2 increases from `0.4008` at n=16 to `0.4487` at n=32,
and H Linf increases from `1.1699` to `1.3157`.  Those maxima are dominated by
the increasingly close samples to the coordinate puncture.  In the fixed
`2<=r<4` shell, H L2 instead decreases
`5.46e-5 -> 1.21e-5 -> 3.78e-6`, and M L2 decreases
`1.19e-4 -> 2.47e-5 -> 7.74e-6` on n=16,24,32.

This split is the main review issue.  The evolution differentiates the regular
reference-frame variables and reconstructs coordinate derivatives
algebraically; it does not finite-difference the singular coordinate metric.
The requested common ADM diagnostic, however, uses the existing common ADM
finite-difference operator on the reconstructed coordinate fields.  A reviewer
must determine whether the near-puncture nonconvergence is purely the expected
coordinate-diagnostic loss of regularity, exposes an error in the ADM adapter,
or signals a deeper mismatch in the reference-frame formulation.  It must not
be hidden with an alpha or chi mask in the primary fixed-region histories.

## Reproduction

```bash
python3 tst/test_suite/ref_gh/reference_frame_audit.py
python3 tst/test_suite/ref_gh/standard_gh_source_audit.py
python3 tst/test_suite/ref_gh/trumpet_reference_audit.py
cmake -S . -B build_cpu -DPROBLEM=built_in_pgens \
  -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_SERIAL=ON
cmake --build build_cpu -j4
```

Run `athena` with the committed `ref_gh_minkowski.athinput`,
`ref_gh_linear_wave.athinput`, `ref_gh_stability.athinput`, and
`ref_gh_stationary_trumpet.athinput`; override
`mesh/nx1`, `meshblock/nx1`, and output basenames for resolution ladders.

The committed compact tables are the review artifacts; large raw outputs are
deliberately excluded.  The stationary runs used a Kokkos Serial Debug build.
No CUDA or multi-rank qualification is claimed.

## Preservation note

The main checkout contains pre-existing modifications to `kokkos`, `bvals`, and
`src/fo_gh/fo_gh.cpp`, plus unrelated untracked source trees.  None are included
in this branch.  In particular, the dirty `bvals_cc.cpp` currently does not
compile because it references an undeclared `i`; that is unrelated to this
reference-GH implementation and was not altered.
