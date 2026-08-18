# Reference-frame FO-GH validation status

## Scope and conclusion

This branch adds an independent 50-field first-order generalized-harmonic module
using a fixed reference frame.  It is based on parent
`68000b4a753056d5f18333a63175d9e003a32300` and does not modify fluid coupling.

The flat-reference foundation is implemented and locally verified: algebraic
frame/source audits pass, exact Minkowski is preserved to roundoff, the linear
wave converges at approximately fourth order, and the robust-stability noise
decreases through one light-crossing time at all three tested resolutions.

This is **not puncture qualification**.  The stationary 1+log trumpet reference,
wormhole-to-trumpet transition, SMR, restart, GPU, and long-time puncture stages
are not implemented or claimed on this branch.

## Implemented surface

- Exact 50-field state: 10 symmetric `Psi_ab`, 10 `Pi_ab`, and 30 `Phi_iab`.
- Fixed-size Kokkos-compatible reference geometry and Minkowski provider.
- Characteristic field helpers and reference-frame algebra checks.
- Standard coordinate-GH source construction and product-rule frame transform.
- AthenaK runtime/task-list/mesh/ADM integration under the separate `ref_gh` module.
- Exact Minkowski, periodic linear-wave, and robust-Minkowski problem generators.
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

## Reproduction

```bash
python3 tst/test_suite/ref_gh/reference_frame_audit.py
python3 tst/test_suite/ref_gh/standard_gh_source_audit.py
cmake -S . -B build_cpu -DPROBLEM=built_in_pgens \
  -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_SERIAL=ON
cmake --build build_cpu -j4
```

Run `athena` with the committed `ref_gh_minkowski.athinput`,
`ref_gh_linear_wave.athinput`, and `ref_gh_stability.athinput`; override
`mesh/nx1`, `meshblock/nx1`, and output basenames for resolution ladders.

## Preservation note

The main checkout contains pre-existing modifications to `kokkos`, `bvals`, and
`src/fo_gh/fo_gh.cpp`, plus unrelated untracked source trees.  None are included
in this branch.  In particular, the dirty `bvals_cc.cpp` currently does not
compile because it references an undeclared `i`; that is unrelated to this
reference-GH implementation and was not altered.
