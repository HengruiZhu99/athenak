# Ref-GH PVC performance, convergence, and stationary-trumpet qualification

Finalized 2026-08-22 on branch
`codex/ref-gh-pvc-performance-convergence-20260821`.  This report supersedes
the interim `ref_gh_pvc_performance_wrapup_20260821.md` for this branch.

## 1. Result and qualification boundary

The equation-neutral Ref-GH refactor passes the checked/full-output Aurora PVC
gate, improves the matched one-tile bounds-check-off benchmark from
`1.150275e5` to `1.272989e6` active zone-cycles/s (`11.0668x`), retains
approximately fourth-order smooth perturbed-trumpet convergence at three exact
common times, and completes the exact stationary 64/96/128 ladder through
`t=20` with no bad states.

The stationary ladder establishes long-time stability through `t=20`, not
truncation convergence of an exact stationary solution.  Its error is
roundoff/secular-accumulation dominated and grows with resolution because finer
grids take more steps.  The independent perturbed evolution supplies the
truncation-convergence evidence.  No result beyond `t=20` is claimed.

The optimized one-tile Ref-GH kernel remains `5.6054x` slower than the matched
mature-Z4c control.  The 1129-Real-per-point full reference allocation also
remains large.  Thus the numerical and portability gates are complete, but
further performance and memory work would still be useful before calling this
an optimal production implementation.

No fluid coupling, Kerr-Schild data, or horizon finding was added or tested.
The GH equations, 50 evolved fields, finite-difference stencils/order,
compatible-Phi algorithm, RK4 coefficients/stage times, CFL, dissipation, and
physical boundary conditions were not changed.

## 2. Starting state and prior portability failure

The starting handoff was `8d5ca88d`, based on numerical source `6eed3d9d`.
That cache refactor had eliminated the original Aurora Level Zero `NotPresent`
fault enough to pass a full-output cycle and a fine stationary run through
`t=20`, but its measured 64/96/128 rates were only about `4.52e5`, `5.38e5`,
and `6.20e5` aggregate active zone-cycles/s with debug bounds checking enabled.
Only the fine long run had completed, so no three-resolution stability claim
was then possible.

The current campaign used a clean worktree rooted at
`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821`.
The original dirty checkout was not modified.

## 3. Build and runtime configurations

Both gate builds used Intel oneAPI DPC++/C++ 2025.3.2, Kokkos 4.7.2 at
submodule commit `6739bc623081648af9e752b616d9671527922cbf`, MPI enabled,
Kokkos SYCL and Serial enabled, OpenMP disabled, `INTEL_PVC`, and Level Zero.
The checked build enabled `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`; the production
build disabled it.  `MPICH_GPU_SUPPORT_ENABLED=1` was used and each eight-rank
run mapped ranks 0--7 to distinct PVC tiles 0.0 through 3.1.

Final hardened gate executable SHA-256 values:

- checked: `7cf89edd636ce753f0f9dd2d9485288049c52b5be80c1273072dcded3a81854f`;
- bounds-check-off performance: `3d1fa89001374748a93ddcfa44513e4c0531af9373f353c2d5161d361e0cd70c`.

The later stationary binary64 diagnostic build was source commit `d7b9646f`
and executable SHA-256
`65bdc913f1d3494002a57ea619a132197344b93d1cb3e586bc2ead478b6f4e82`.
The source changes after the hardened gate affect cbin output precision and
supplementary common-ADM output only.  Their local binary32/binary64 round-trip
tests passed, and their full PVC pathway was exercised at t=0 and t=20 in all
three stationary runs.

## 4. Equation-preserving implementation changes

The accepted changes are:

1. Reordered component/cell reference-cache traversals so the cell coordinate
   is contiguous for `LayoutRight` (`5a3831ba`).
2. Added explicit static/time-dependent provider validity.  A stationary
   reference is built once instead of at every RK stage; a time-dependent
   provider still builds at all five observed RK stage times (`5d864487`).
3. Honored `include_diagnostics=false` during production cache updates and
   skipped diagnostic Ricci construction in RK-stage work (`338b18f0`).
4. Added a lean covariant scalar-source evaluator that computes the same ten
   symmetric source components without materializing five reporting-only
   sector matrices; the full evaluator remains the independent oracle
   (`fc2d61ad`).
5. Reused point-coordinate geometry and fused scalar-source/Pi work while
   preserving the contractions and result ordering within the accepted
   tolerance (`17ab455f`).
6. Hardened diagnostic reconstruction for collapsed extents and fail-closed
   invalid ADM states (`326ba670`, `afc5b6b6`).
7. Added optional binary64 cbin fields and independent common-ADM cbin output;
   existing binary32 behavior and official history definitions remain
   unchanged (`cf42409f`, `552cdb89`).

Two exact-symmetry compression variants were rejected before benchmarking.
They changed the unchanged checked oracle to `5.95e-13` and `9.97e-13`, above
the `5.68434e-14` threshold.  No tolerance was weakened, and commit `47d0ad99`
restored the accepted arithmetic.

## 5. Performance progression

All one-tile rows use the same 64-cubed active-cell stationary benchmark,
four-stage RK4, and CFL 0.05.

| Change | Aurora job | zone-cycles/s | Incremental result |
|---|---:|---:|---:|
| Checked baseline | 8773102 | `7.835008e4` | bounds checking on |
| Production baseline | 8773102 | `1.150275e5` | bounds removal `1.4681x` |
| Traversal reorder | 8773363 | `1.126323e5` | `0.9792x`; retained for portable layout |
| Static reference caching | 8773414 | `3.033790e5` | `2.6935x` from traversal |
| Diagnostic split | 8773745 | `3.550192e5` | `1.1702x`, not causally assigned |
| Lean production source | 8774182 | `9.592259e5` | `2.7019x` from preceding job |
| Source/Pi fusion | 8774435 | `1.325066e6` | `1.3814x` from lean source |
| Final hardened source | 8774480 | `1.272989e6` | `0.9607x` from prior job; treated as run variability |
| Mature Z4c control | 8773273 | `7.135622e6` | `5.6054x` final Ref-GH |

The static-cache plus later source refactors give `4.1960x` from the accepted
static-cache result to the final hardened source and `11.0668x` over the
original bounds-check-off Ref-GH baseline.  The best adjacent job measured
`11.5196x`; the lower hardened number is used for conservative final claims.

The completed eight-tile stationary production runs measured aggregate rates
of `6.393318e6`, `8.737631e6`, and `8.757475e6` zone-cycles/s for 64/96/128.
These larger-domain aggregate rates are not substituted for the matched
one-tile Ref-GH/Z4c comparison.

## 6. Profiles and compiler pressure

Before static caching, repeated reference construction accounted for about 66
percent of synchronized kernel time.  After caching, scalar-source and Pi RHS
were 72.02 and 12.79 percent.  The final fused profile is:

| Kernel | Calls | Total seconds | Percent |
|---|---:|---:|---:|
| scalar source and Pi RHS | 81 | 3.731902 | 59.8571 |
| stationary reference Ricci | 1 | 0.661297 | 10.6067 |
| stationary frame audits | 1 | 0.652466 | 10.4651 |
| flat constraints | 2 | 0.323625 | 5.1907 |
| reference metric jets | 1 | 0.285106 | 4.5729 |
| reference connection | 1 | 0.123971 | 1.9884 |
| Psi RHS | 81 | 0.099631 | 1.5980 |
| dissipation | 81 | 0.070971 | 1.1383 |

The checked fused kernel compiled SIMD32 with 256 registers and recorded spill
counts of 129 and 26 in two variants.  Smaller Ref-GH kernels recorded spills
between 10 and 43; several templated RHS kernels required compiler retries.
The exact records are in `compiler_pressure.tsv`.  These are measured compiler
events, not proof that spills alone cause the remaining Z4c gap.

## 7. Reference-cache allocation and update lifetime

| Category | Reals/cell | Bytes/cell | 64-cubed block plus four ghosts |
|---|---:|---:|---:|
| provider/update jets | 64 | 512 | 0.17798 GiB |
| update workspace | 416 | 3328 | 1.15686 GiB |
| persistent evolution | 313 | 2504 | 0.87043 GiB |
| derivative/diagnostic | 336 | 2688 | 0.93439 GiB |
| total | 1129 | 9032 | 3.13965 GiB |

These arrays currently use the full 72-cubed four-ghost extent.  Static cache
validity removes their reconstruction from the stationary hot path, but does
not shrink allocation.  Exact symmetry compression was not accepted because
it failed the unchanged oracle.  The remaining 1129-Real footprint is therefore
reported as an unresolved memory-efficiency issue, not described as minimal.

## 8. Correctness and full-output PVC gates

Final hardened-source job 8774480 passed:

- independent flat/nonflat source oracle maxima `5.55112e-17` and
  `3.33067e-16`;
- stationary initial RHS Linf `1.10048e-16`;
- positive-time field Linf `9.992007e-16`;
- native-constraint Linf `2.461044e-14`;
- full native/common histories and an actual restart;
- checked/performance history difference at most `1.0913293364176594e-16`;
- a five-stage time-dependent provider test with five provider builds and
  evolved error `9.992007e-16`;
- lower-dimensional diagnostic extents and fail-closed ADM reconstruction;
- no `NotPresent`, page fault, nonfinite state, or bad-state regression.

The previous PVC segfault remains eliminated in the checked gate, performance
gate, binary64 convergence job, and 17,625 total stationary ladder steps.

## 9. Binary64 perturbed-trumpet convergence methodology

Athena cbin output now optionally writes field payloads as IEEE binary64 while
retaining binary64 coordinates.  The primary convergence job used binary64
field and native-constraint dumps.  Each 64/96/128 member was launched
independently to the exact same endpoint at t=0.2, 0.4, and 0.6; an earlier
nominal-output set with mismatched endpoint times was explicitly rejected.

The analysis compares a fixed cell-centered sample grid inside r<1 using
sixth-order tensor-product interpolation.  Unequal-grid Richardson orders use
the exact 64/96/128 spacing relation rather than assuming equal refinement
ratios.  Per-variable results are preserved in the JSON files.

## 10. Perturbed-trumpet convergence result

| t | quantity | L2 order | Linf order |
|---:|---|---:|---:|
| 0.2 | all fields | 4.4019 | 4.9647 |
| 0.2 | dynamic fields | 4.4013 | 4.9647 |
| 0.2 | native constraints | 5.4195 | 4.1201 |
| 0.4 | all fields | 4.1973 | 4.7000 |
| 0.4 | dynamic fields | 4.1998 | 4.7000 |
| 0.4 | native constraints | 4.4279 | 4.1427 |
| 0.6 | all fields | 4.0610 | 4.5094 |
| 0.6 | dynamic fields | 4.0659 | 4.5094 |
| 0.6 | native constraints | 5.4752 | 4.1533 |

This establishes approximately fourth-order convergence for the evolving
field aggregate and native constraints at multiple times.  Psi L2 orders are
4.7628, 3.6955, and 3.5714, while Psi Linf orders are 2.5770, 1.9442, and
1.3438.  Therefore the result is not presented as fourth order for every
individual pointwise variable near the puncture.

Maximum characteristic speeds are 0.61058, 0.61194, and 0.61261.  With outer
faces at coordinate distance 2 and an r<1 analysis region, the earliest simple
face-to-region arrival estimates are t=1.638, 1.634, and 1.632.  The accepted
comparison ends at t=0.6 and is causally before those estimates; no
post-boundary convergence claim is made.

## 11. Stationary 64/96/128 campaign

Aurora capacity job 8774589 used source `d7b9646f`, eight ranks, and eight
distinct PVC tiles on node `x4115c5s6b0n0`.  All cases started from t=0, used
the same [-2,2]-cubed physical domain, four-ghost cells, RK4, CFL 0.05, fourth-
order Ref-GH differencing, exact trumpet boundary data, history every 0.1, and
restart checkpoints every 2.0.

| N | cycles | field Linf | native constraint Linf | initial RHS Linf | bad states |
|---:|---:|---:|---:|---:|---:|
| 64 | 3908 | `6.465162e-12` | `9.321782e-11` | `1.100478e-16` | 0 |
| 96 | 5875 | `9.628409e-12` | `2.032577e-10` | `1.344374e-16` | 0 |
| 128 | 7842 | `1.326683e-11` | `3.583889e-10` | `1.060398e-16` | 0 |

At t=20, the metric-condition maxima are within `1.0e-13` of one, maximum
characteristic speeds remain 0.6106--0.6126, and no lapse, regular-field, CFL,
or determinant diagnostic indicates loss of state admissibility.

The PBS allocation ended with exit status 1 only after all evolution cases had
passed: Aurora's default Python 3.6 could not parse the analysis scripts.  The
transferred outputs passed under Python 3.12.  The launcher now explicitly
selects installed Python 3.10, and a Python-3.12-only multiline f-string was
rewritten equivalently.  All analysis scripts parse with Aurora's actual
Python 3.10, and the regenerated local tables are byte-identical.  This
postprocessing environment error does not make a partial evolution result;
each final time, final diagnostic row, cbin, and checkpoint chain is present
and verified.

## 12. Secular and roundoff scaling

Stationary Psi-error RMS and reduction-constraint RMS grow nearly linearly in
time from t=1 to t=20.  The fitted time exponents are:

| N | Psi-error RMS exponent | reduction RMS exponent |
|---:|---:|---:|
| 64 | 0.9980 | 1.0152 |
| 96 | 1.0011 | 1.0154 |
| 128 | 1.0024 | 1.0125 |

Across resolutions at t=20, field Linf scales as N^1.033 and as
cycles^1.028.  Its error per cycle is `1.654e-15`, `1.639e-15`, and
`1.692e-15`, a 3.2 percent span.  Native-constraint Linf scales as N^1.942
and cycles^1.932.  The latter is consistent with a derivative constraint
adding approximately one inverse-grid-spacing factor to cycle-proportional
state roundoff.  This is an inference from measured scaling, not a proof of a
specific floating-point accumulation mechanism.

The histories remain smooth and bounded at 1e-11 state error and 1e-10 native
constraint error, and the independent masked common-ADM truncation data remain
unchanged.  The fine-grid-larger values therefore do not provide evidence of a
resolution-dependent numerical instability through t=20.

## 13. Supplementary fixed-mask common ADM constraints

The cbin diagnostic reconstructs common ADM gamma_ij and K_ij and invokes the
same fourth-order Hamiltonian/momentum operator used by the official history.
It then applies fixed physical masks offline; no evolving lapse or chi mask is
used.  These results are supplementary and do not redefine history columns.

At r>=0.5 and t=20:

| quantity | N=64 | N=96 | N=128 | fitted p in error proportional to N^-p |
|---|---:|---:|---:|---:|
| H L1 | `8.640e-6` | `1.752e-6` | `5.508e-7` | 3.969 |
| H L2 | `5.842e-5` | `1.214e-5` | `3.811e-6` | 3.934 |
| H Linf | `1.789e-3` | `3.955e-4` | `1.346e-4` | 3.731 |
| M L1 | `1.601e-5` | `3.209e-6` | `1.010e-6` | 3.985 |
| M L2 | `9.390e-5` | `1.914e-5` | `6.013e-6` | 3.962 |
| M Linf | `2.396e-3` | `5.260e-4` | `1.783e-4` | 3.748 |

The r>=0.25 L1/L2 orders range from 3.83 to 4.06.  At r>=0.125, L1/L2 remain
near fourth order while Linf is 2.67--3.00 because the fixed mask approaches
the puncture.  Whole-domain Linf grows mildly with N because it samples the
puncture singularity and is not an admissible convergence measure there.

Across every recorded mask/norm, the maximum relative t=0 to t=20 change is
`5.89e-5`; most are much smaller.  This separates stable stationary evolution
from the puncture-dominated finite-difference truncation profile.

## 14. Scientific conclusion

Approximately fourth-order perturbed-trumpet convergence is established at
t=0.2, 0.4, and 0.6 before the estimated outer-boundary arrival.  Exact
stationary-trumpet evolution is stable at 64/96/128 through t=20: all cases are
finite and admissible, no bad state occurs, state error follows timestep-count
scaling, and fixed-mask common ADM constraints converge at roughly fourth
order outside the puncture.

This is not a claim of a dynamical trumpet forming from one-puncture Z4c data,
nor a horizon/trumpet-profile qualification.  It is specifically the Ref-GH
analytic stationary reference plus smooth perturbation tests required by the
controlling performance/convergence goal.

## 15. Remaining bottlenecks and limitations

- The final one-tile Ref-GH rate is only 17.84 percent of mature Z4c.
- The fused scalar-source/Pi kernel remains 59.86 percent of synchronized
  kernel time and has measured register pressure/spills.
- Setup-only Ricci/audit/metric-jet work sums to about 25.6 percent in the short
  synchronized profile; it is amortized in long stationary runs.
- The full reference allocation is 1129 Reals per point over four ghosts.
  Shrinking lifetimes/extents remains unimplemented because it was not the
  demonstrated runtime hotspot and exact symmetry compression failed the
  oracle.
- Individual Psi/Phi pointwise convergence near the puncture is weaker than
  fourth order even though aggregate evolved fields and constraints pass.
- Perturbed convergence is qualified only through t=0.6, before the boundary
  estimate, and stationary stability only through t=20.
- No 12-tile node-level matrix was run; the goal correctly deferred generic
  scaling experiments while dominant per-tile kernels remained expensive.

## 16. Provenance and artifacts

Compact evidence is under
`docs/fo_gh_artifacts/ref_gh_pvc_performance_20260821/`.  The principal sets
are `baseline_8773102`, `static_cache_8773414`,
`diagnostic_split_8773745_8774035`, `lean_source_8774182`,
`fused_source_pi_8774435`, `robust_final_8774480`,
`perturbed_exact_binary64_8774567`, and `stationary_t20_8774589`.

Large builds, binaries, cbin field dumps, and restart files are excluded from
Git.  Full Aurora data remain beneath:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821`

The stationary cbin hashes and exact remote relative paths are recorded in
`stationary_t20_8774589/binary64_cbin_sha256.tsv`.  Each compact artifact tree
has or will have a SHA-256 manifest.  The controlling branch is pushed without
force and preserves unrelated work.
