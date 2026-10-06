# Trumpet initial-data consumer

This branch starts from `codex/hispid-pgen` at9495fdb9. The existing TDE and
residual-Z4c branches are unchanged. The native implementation lives on
`HengruiZhu99/TwoPuncturesC:codex/hispid-trumpet`.

The actual checkpoint reader now accepts legacy version1 as QI and version2
with an explicit `seed_family qi` or `seed_family trumpet_r0_m` field following
the parameterization. The linked native header/library must support the new
ABI-safe family entry points. Unknown/missing families and version/payload
mismatches are errors. Native image checks cover the new constructor and
family-query symbols as well as existing sampling symbols.

The pgen creates the matching sampler and verifies its resolved family.
For individual surface guesses, the R0=M trumpet uses coordinate horizon
radius sqrt(m^2-a^2), followed by the same Lorentz contraction of the angular
guess. QI uses half that radius. This guesses a single-seed surface; it does
not supply binary horizon acceptance or prove attenuation enclosure.

`tst/test_suite/z4c/check_trumpet_checkpoint.py` compiles the real C++ reader
with a small standalone sampler probe. The final retained run under
`trumpet-checkpoint-20261005-final/` exactly matches native/Python physical
metric, extrinsic curvature and metric gradients for both formats, and
rejects three malformed metadata cases. The initial loader failure and
intermediate result are retained separately. Executables are local ignored
artifacts; source/image hashes and checkpoint bytes are retained.

At the standalone-reader milestone, full AthenaK compilation, mesh import
and horizon searches were still pending. The later consumer results below
supersede that status. No evolution or binary acceptance is claimed by the
standalone reader check.

## Full production consumer build, 2026-10-05

The complete `z4c/hispid` executable from consumer commit ea925d1e compiled
successfully on Perlmutter (allocation59402262, one shared A100 allocation).
It uses the CPU sampler library built from native commit ff5e8ae, row power3,
and the repository's pinned Kokkos6739bc623081648af9e752b616d9671527922cbf,
with Serial execution and MPI disabled. The allocation has released.

`trumpet-production-build-20261005/` preserves configure/build logs, executable
and library hashes, archive hashes and completion receipt. The isolated remote
executable is
`/pscratch/sd/h/hzhu/codex-hispid-trumpet-20261005/build-athenak/src/athena`.
This build established compilation/linkage of the full pgen. Subsequent
isolated spin-0.99 import and horizon searches passed at angular orders
8/12/16, with finest mass 1.00000000000409, spin 0.989999999999997, and
expansion RMS 4.538e-11. Physical binary acceptance remains pending.

## Parallel direct horizon sampling

`<problem>/hispid_parallel_geometry = true` opts the native HiSpID callback
into host-parallel surface sampling. Define it in the input file before using
a command-line override. Build AthenaK with OpenMP and choose the thread count
with `OMP_NUM_THREADS`. The default remains serial. The callback must support
concurrent read-only calls after one serial warmup; its owner clears the flag
when releasing the provider. Worker exceptions are collected and propagated
before any incomplete surface can advance.

A matched isolated spin-0.99 trumpet at lmax=8, ntheta=16 on Perlmutter,
using 16 threads, agrees with the previously validated serial mass/spin/area
to 2.64e-15 scaled error. Expansion RMS is 4.42e-11; runtime was 12.29 s versus
102.03 s for the earlier serial run (different allocations, not a scaling
study). A deliberately too-narrow mesh exercises callback failure and exits
with no horizon data rows. FastFlow creates a header-only summary at startup;
its existence is not evidence of an accepted horizon. The original test's
incorrect file-absence assertion and its corrected assessment are retained in
the native solver's `validation/trumpet/parallel-consumer/` records. The
three-order angular study is not repeated for this backend control.

`check_hispid_parallel_geometry.py` reproduces these two checks using a bound,
passing serial baseline. `perlmutter_trumpet_parallel.sh` records the isolated
build recipe; its output directory must be fresh when rerunning the control.
Binary physical convergence and Gamma=10 horizon acceptance remain pending.

The checkpoint reader now uses the native header's per-axis extent limits,
including polar extents through512, with radial/azimuthal limits still256.
Sampler memory budgets and image binding are unchanged. The focused reader
control accepts `--polar-extent 384` to check the actual reader/sampler path
and reject polar513 without repeating the earlier family-metadata matrix.
Full horizon validation of any newly refined solved binary remains pending.

The polar384 control passed on Perlmutter: the actual C++ reader and native
CPU sampler match Python physical fields and metric gradients exactly, and
polar513 is rejected with `Invalid checkpoint grid extent`. Records are in
the native branch's `validation/trumpet/polar-refinement/reader384/` directory.

## Solved moderate binary: consumer and horizons

The full OpenMP consumer built with the larger polar limit has now loaded the
256x512x16 moderate trumpet checkpoint and passed the l=8/12/16 horizon schedule
plus an independent l=16 quadrature increase from32 to48 polar points, using
16 geometry threads. The checkpoint remains diagnostic because its separate
constraint-resolution sequence fails decreasing bulk error.

The checkpoint-bound producer/CPU-sampler comparison is exact over2454 points.
The pgen's ADM/Z4c roundtrip error is4.13e-16. Final component irreducible masses
are0.5887780113536 and0.3935773862669; coordinate-spin magnitudes divided by
horizon mass squared are0.3895967639598 and0.3726775137246. Final expansion RMS
values are2.57e-10 and3.15e-9. Both retained surfaces enclose their modified balls,
including the empirical angular-refinement buffer. Mass and spin-vector changes
are below1.7e-11. The coordinate spin integral is not an AKV estimator.

Native commit76ac385 preserves the complete inputs, logs, surfaces and assessment:
[moderate binary horizon evidence](https://github.com/HengruiZhu99/TwoPuncturesC/tree/76ac385/validation/trumpet/convergence/horizon256/surfaces).
The production executable SHA256 is
`4d5c4f53bf297ebe4b390ec72a74dd379d6f0182be723d9a0ae83c155dec24e1`.
Run `check_hispid_binary.py` with `--allow-diagnostic`, the checkpoint-bound
`--migration-proof`, `--geometry-threads 16`, `--strict-expansion`, and
`--harmonic-storage factorized` to reproduce this consumer path. Acceptance of
these horizon checks does not promote the input's constraint-validation status.

### Reusing converged horizon shapes

The optional `problem/hispid_horizon_shape_guess_0` and `_1` inputs read one
complete finite spherical-harmonic coefficient record from the existing
`hispid.horizon_shape_N.txt` format. A lower order is padded with zero higher
modes; malformed, excessive, or incomplete coefficient records are rejected.
The coefficients only initialize FastFlow. Geometry, expansion, properties,
and acceptance are recomputed. Omission preserves the existing initial guess.

`check_hispid_binary.py --initial-shapes FILE0 FILE1 --reuse-shapes` binds
input files by SHA256, verifies native consumption, and reuses only a passed
row as the next guess. All existing import, enclosure, expansion, angular,
and quadrature criteria remain in effect.

Perlmutter validation59410814 on the focused spin99 diagnostic checkpoint:
l12 and l16 pass in35.49 and50.42 seconds, versus670.69 and1226.91 seconds
from seed-radius initialization. Mass and coordinate spin agree within1e-12.
These checks validate initial-guess reuse; the input binary still fails
independent physical constraint requirements and is not accepted physical data.

The full focused spin99 horizon schedule59410814 subsequently passed orders
12/16/20 and the20x60 quadrature control. Final component mass is.50001785189996,
Mirr=.37776358547565, coordinate chi=.98992952098846, and expansion RMS5.976e-9.
Maximum mass-relative/spin-vector changes are1.106e-11/3.788e-11. Both retained
surfaces enclose their modified balls with a.03226304058 margin after the
empirical refinement buffer. Full evidence is in the native branch under
`validation/trumpet/spin-focused-map/horizon160-warm/surfaces`. Independent
physical constraints of the input remain outside acceptance.
