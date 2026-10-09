# Completed constraint-propagation experiments, 2026-10-09

These are raw byte copies of completed private experiments. The evolved
production implementation is `27c19d20696ea6dd4704032c51dfd026218f64f2`;
launches and collection use branch HEAD `b37a20f2d7a8ccc42148f1a814a80b2957e17b53`.
The subsequent evidence commit does not identify a newly compiled production
executable. The branch is `z4c_hyperboloidal_layer`.

No stable finite pulse, nonlinear regular scri closure, black-hole transition,
or boundary energy estimate is established. Every-stage projection, halved
actual timestep, small amplitude, and composed evolution derivatives are
separate controlled experiments. The small pulse is not the required finite
pulse. The Minkowski reference remains the target for the later resolved
wormhole-to-trumpet transition.

`catalog.json` records every copied file's original repository-relative path,
size and SHA256. Each frozen subdirectory also preserves its original index.
The catalog excludes only itself. `verify_constraint_archive.py` independently
checks the catalog and all JSON numbers without reading scratch data:

```
python3 verify_constraint_archive.py /absolute/path/to/this/archive
```

Executables, objects, libraries, CSR matrices, full BIN/RST snapshots and large
arrays stay local. Receipts preserve their exact paths, sizes/hashes and, where
applicable, shapes. Native BIN is binary32; active evolved RST arrays are
binary64. Historical `physical_metric_eigen_*` receipt keys mean eigenvalues
of gtilde/chi, the Penrose spatial metric. Physical eigenvalues additionally
scale by Omega^-2; their positivity is equivalent on these strict-interior
grids. No full-precision drift is inferred from BIN differences.

## Evidence and reproduction

The checked source snapshots, compiler/link commands, flags, dependency
hashes, Python arguments, exit statuses and timings are retained. This is an
audit bundle, not a standalone package. Do not execute its captured scripts
inside the archive: their original relative layouts differ and some create
new receipts. For reproduction, use a separate checkout and scratch trees,
restore sources to the original layouts described below, rebuild dependencies,
and run the recorded commands there. Never overwrite the frozen originals.
Absolute paths in original command arrays document the actual invocation;
adapt only the checkout/runtime prefixes when reproducing elsewhere.

Requirements are the recorded AppleClang C++17/CPU Serial/double Kokkos build,
Kokkos `6739bc623081648af9e752b616d9671527922cbf`, the existing
`build-layer-release` generated headers/static libraries/native objects, and
Python with NumPy/SymPy/mpmath. The global matrix audit additionally uses SciPy.
Original compiler versions and all source/link input hashes are in receipts.
Fresh output hashes can depend on toolchain versions. The source and exact
observed evidence hashes remain the comparison reference.

- `subsidiary/`: restore its top-level sources into
  `build-layer-research/continuum/constraint-propagation/`, then run its
  `run_audit.py`. It compiles the actual-kernel dual probe and checks fulltensor
  C0 subsidiary identities, coefficient-aware tangency and principal energy.
  The exact commands are in `receipt.json` and `check-receipt.json`.
- `discrete-bulk/`: restore top-level sources into
  `build-layer-research/continuum/discrete-bianchi/`, then run `run_audit.py`.
  The script compiles the native/composed full20 symbols and manufactured
  derivatives, and checks the exact gauge source and Nyquist KO bound.
- `tensor-negative/`: restore top-level sources into
  `build-layer-research/continuum/tensor-discretization/`. Its `run_audit.py`
  uses the preceding bulk check source. It reproduces the exact defective
  coupled-gauge resonance; there is no native evolution for this candidate.
- `C1-identity/`: restore sources into `build-layer-research/continuum/covariant-z4/`.
  Its `run_audit.py` checks finite-Omega nonlinear physical-ADM identities and
  the unchanged full20 principal system, including the separately derived
  covector connection repair. Explicit double poles remain. It supplies no
  native timestep, regular scri closure or C1 evolution acceptance.
- `inner-trumpet/`: restore to `build-layer-research/inner-trumpet-gate/` and
  use `run_audit.py`. Its earlier detached-wormhole gate dependencies are in
  the preceding spatial-norm archive. Release/sanitizer tensor jets and the
  local stationary asymptotic proofs are not black-hole evolutions. Its
  [-S,S] resolution formula differs from these native runs' span2.1 box;
  the formulation report supplies the corresponding corrected cell radii.
- `rst-reader/`: restore to
  `build-layer-research/time-projection-controls/rst-reader-gate/`. The ABI
  probe and validation commands are in `probe-receipt.json`,
  `checked-receipt.json` and `supplemental-receipt.json`. Complete native
  snapshot files are required for the original 87-pair comparison.
- `composed-gate/`: restore its captured private source paths from `captured/`
  relative to the checkout, and test sources to the original gate directory.
  `run_gate.py` checks the actual ng4/radius4 patch in Release and sanitizers.
  Original diagnostic loading and fourth-order/upwind/KO choices are retained.
- `full-tensor-stage/`: `sources/` preserves projected-v1 and raw22-v2 source
  variants; source/build receipts retain their original layouts and exact
  commands. The six retrospective audit gates test implementation consistency,
  not physical acceptance. This gate contains only validated stage/short-time
  results; the separately frozen global gate records completed t2 results.
- `full-tensor-global/`: the completed continuous projected20 Cartesian
  propagator, independent all-state Taylor/Arnoldi comparison, native nonlinear
  directional diagnostics, sources, plots and original dependency inventories.
  Its continuous trajectories differ from the native final-only finite RK3
  map. The component/derivative norm is not a proved tensor energy. Its Ritz
  candidates have insufficient residuals for an eigenvalue claim.
- `small-and-control-review/` and `composed-final-native/`: completed independent
  binary64 amplitude/control/composed evolution reviews, frozen source copies,
  endpoint and explicitly interpolated history comparisons. The original
  frozen indices retain their full source paths; catalog entries map the copies.
- `native-controls/` and `native-runs/`: exact override files, generated final
  inputs, commands, source-at-launch manifests, logs, histories, results and
  full-precision audits for each completed run. The stage and composed builders
  recompile only the stated objects and verify every reused link input. The
  base spatial-norm executable/build receipt and source injection are in the
  preceding archive and its retained local scratch tree.

For example, from a reconstructed original layout the common audit command is
`python audit_native_control.py half` (or `stage`, `stage-reference`). The
composed auditor takes a run directory and same-grid baseline directory; all
actual arguments are in the receipts. `audit_small_native.py` checks amplitude
scaling against the zero-reference and finite-pulse runs. These audits require
the retained full binary64 restarts and refuse changed source/executable hashes.

Failed initial assertions and the failed rvalue-wrapper compilation are
preserved with their original source and error output. Whitespace in captured
sources/logs is retained deliberately. Original archives remain immutable;
later results belong in a new catalog.
