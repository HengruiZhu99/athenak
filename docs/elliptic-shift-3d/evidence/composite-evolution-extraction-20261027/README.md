# Mechanical extraction of the tested composite tensor stage engine

The tested pilot engine now lives in
`src/z4c/hyperboloidal/composite_evolution.hpp` as
`z4c::hyperboloidal::CompositeTensorEvolution`. The header owns its includes,
22-field layout constants, tensor-component index arrays and the unchanged
stationary compactification helper `CompositeOmegaAt`.

This prepares reusable code for a native owner. It does **not** wire the engine
into AthenaK Z4c Tasks, native fields, Driver/RK stages, shell checkpointing or
live regridding. No Cauchy guard was relaxed and no native execution claim is
made by moving the class under `src`.

## Exact scope of the edit

The class was translated mechanically from the test source. Only its name,
constructor names, same-namespace qualification, class-owned aliases/constants
and the Omega helper name changed. The physics, source construction, boundary
conditions/rates, positive storage, enrichment, global shift solve, numerical
controls, telegraph sources, CFL guards, logging and classical RK4 statements
remain unchanged.

`tst/elliptic_shift_3d/coupled.cpp` retains its existing global field/tensor
constants, mesh helper, CLI arguments, profile extraction, checkpoint/restart
loops and scientific gates. Its global `Pilot` alias names the production
class, and global `Omega` delegates to `CompositeOmegaAt`. The entire
`main`/profile suffix is byte-identical. `translation-proof.json` records the
original and translated source/class hashes and exact transformation list.
The qualified extraction header hash is
`9fc6302f55223c91e265aff4763e7966a61c9c6f0c23df275ce3c9064783c0f6`.
The pre-extraction test source and qualified header/test snapshots are retained.

## Public interfaces preserved for the future native owner

The unchanged public engine exposes:

* `state`, `beta_delta`, `jets`, `scri`, `boundary_rhs`, `beta_padded` and
  the existing solver/closure diagnostic fields;
* mesh-based construction with all existing geometry, chart, refinement,
  smoother and source options;
* `Initialize`, `InitializeWormhole`, `Padded`, `GeometryJets`, `Boundary`,
  `SolveShift`, `BoundaryRates`, `RHS` and classical-RK4 `Step`;
* the original positive-field encoding, decoding and rate interfaces.

The native owner can drive `RHS` using its validated native stage/register
adapter; merely calling the extracted `Step` is not native Driver integration.
The object's grid address/lifetime must remain stable while its existing
surface, enrichment and multigrid caches refer to it. An externally retained
`Boundary(padded)` callable also retains the original reference to that padded
vector; the caller must keep it alive. These are preserved ownership contracts,
not new runtime behavior added by this extraction.

## Bounded before/after evidence

The current pilot was rebuilt before the edit and captured as the baseline.
Only `elliptic_shift_3d_coupled_pilot` was rebuilt, with two CPU workers. Each
compute/test command had a180-second bound and a newly created process group;
no timeout or signal was needed. No other job was touched.

The same three cases were run before and after extraction:

| Case | Cells | Duration | Before/after result |
|---|---:|---:|---|
|Minkowski,1 shell,2 RK4 steps|1512|t=.001|Exact profiles, angular samples, checkpoint bytes and diagnostics|
|Minkowski,2 shells,2 RK4 steps|2808|t=.001|Exact profiles, angular samples, checkpoint bytes and diagnostics|
|Matched wormhole,N8,7 shells,zero steps|25600|t=0|Exact profiles, angular samples and diagnostics|

Every numerical stdout value, flag and iteration count is identical; only
wall-clock seconds were excluded. The Minkowski cases checked a midpoint
restart and bitwise continuation. Their maximum state deviations are
1.27e-16 and7.85e-16, and tensor-pole residuals3.10e-15 and1.16e-14.
These checks demonstrate preservation, not a new formulation result.

The zero-time wormhole uses the existing selected `.24` matched initializer,
M=1/3, Minkowski reference and exponent-independent positive log storage.
Its recorded initial H RMS.509064345238, M RMS.028348478014, zero Theta/Z,
field minima, physical chart flags and enriched-cell count exactly match the
archived `enriched-buffered-restart48-20261027/initial.stdout` values.
`zero-time-archive-comparison.json` records all17 comparisons. The older
archive's point-multigrid solve used190 iterations, whereas the current
radial-line solve uses24; that algorithm change predates this extraction and
is not presented as a bitwise old-algorithm comparison. Current baseline and
post-extraction solver diagnostics are bitwise equal.

The zero-time wormhole still has large initial Hamiltonian/momentum defects;
its process exit0 is an initial algebraic/closure compatibility gate, not
passing coupled black-hole scientific validation. Earlier full evolution
failures remain unchanged.

## Compilation and regression results

The post-extraction pilot rebuild passed in4.28 seconds. Both existing
renamed-main harnesses, `diagnose_wormhole.cpp` and
`instantaneous_radial_snapshot.cpp`, compiled unchanged with `-fsyntax-only`.
A separate header-only translation unit includes just the new public header
and standard type traits; it verifies field layout and the public RHS/Boundary
return interfaces. It compiles successfully, demonstrating self-contained
header dependencies rather than relying on test-only include order.

All28 existing `elliptic_shift_3d_*` CTests passed in13.13 seconds. The pilot
was the only requested rebuild target; the other pre-existing test executables
were run as instructed. Neither baseline nor post-extraction qualification
observed changes to the existing header tree. The qualified class snapshot
is preserved separately so later parent-owned halo/native integration edits
cannot be confused with this mechanical qualification.

## Reproduction and retained files

`validate_composite_extraction.py` records every command, timeout, source and
binary hash, stdout/stderr, exact output comparisons and restart results.
`baseline.json` and `after.json` contain the complete manifests. The test
source copy before extraction and qualified header/test snapshots allow the
translation to be inspected independently of later changes.

The validation sequence was:

```sh
python3 tst/elliptic_shift_3d/validate_composite_extraction.py baseline \
  --output docs/elliptic-shift-3d/evidence/composite-evolution-extraction-20261027
# Apply only the recorded mechanical extraction.
python3 tst/elliptic_shift_3d/validate_composite_extraction.py after \
  --output docs/elliptic-shift-3d/evidence/composite-evolution-extraction-20261027
```

Use a fresh output directory for a new qualification. The existing evidence
profiles/checkpoints/initializer were losslessly gzip-compressed only after
all exact comparisons. `compressed-files.json` retains original byte counts
and SHA256s for restoration; original upstream input files were never changed.

The next implementation step remains parent-owned native Task/field/RK wiring,
followed by native shell/restart/regridding execution tests and the still
failing coupled scientific gates. This extraction adds no new algorithm or
scientific success claim.
