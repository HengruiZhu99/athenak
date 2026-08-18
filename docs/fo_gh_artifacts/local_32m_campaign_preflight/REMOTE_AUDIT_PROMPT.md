# Remote audit prompt: FO-GH/Z4c puncture comparison

Please perform a detailed, read-only scientific and implementation audit of
the FO-GH/Z4c puncture-comparison work on branch
`codex/fo-gh-puncture-driver-20260817`.  Do not alter, commit, push, launch
runs, or weaken any acceptance threshold.  Focus only on the vacuum FO-GH
solver and its comparison with mature Z4c; fluid coupling, Kerr-Schild data,
and horizon finding are out of scope.

Start by recording the full commit and dirty state.  Read:

- `docs/fo_gh_artifacts/local_32m_campaign_preflight/README.md`
- `docs/fo_gh_puncture_validation.md`
- `inputs/fo_gh/fo_gh_puncture_compare_smr.athinput`
- `inputs/z4c/onepuncture/z4c_onepuncture_compare_smr.athinput`
- `tst/test_suite/fo_gh/analyze_puncture_comparison.py`

Then audit the actual implementation, especially:

- `src/coordinates/adm_constraints.cpp` and its wiring in `adm.hpp`,
  `adm.cpp`, and `src/CMakeLists.txt`;
- common-history region selection, normalization inputs, SUM/MAX MPI
  reductions, and output naming in `src/outputs/history.cpp` and
  `src/outputs/outputs.hpp`;
- Z4c restart behavior in `src/pgen/z4c/z4c_one_puncture.cpp`;
- the CPU regression additions in
  `tst/test_suite/fo_gh/test_fo_gh_puncture_evolution_cpu.py`.

Check the following claims independently against source and compact evidence:

1. The common Hamiltonian and momentum constraints use the same reconstructed
   ADM `gamma_ij` and `K_ij`, the same finite-difference stencil, and the same
   proper-volume and fixed-coordinate regions for both formulations.
2. The primary common histories apply no evolving lapse or chi mask.  Native
   histories remain secondary: FO-GH uses `alpha >= excise_lapse`; Z4c
   literally uses `chi >= excise_chi`.  With the identical initial data,
   `chi=alpha^2`, so the defaults `excise_chi=0.0625` and
   `excise_lapse=0.25` are equivalent only at initialization.
3. L1, L2-square, Linf, and volume columns reduce correctly across MPI ranks;
   empty regions do not produce invalid maxima.
4. The six fixed regions and six refinement-interface shells match the
   campaign specification and are not resolution-dependent masks.
5. The FO-GH and Z4c input decks represent identical regularized one-puncture
   data, the same physical domain and fixed SMR geometry, and the intended
   N=8/12/16 resolution ladder when overridden as documented.
6. The initial-data comparison and convergence-analysis script reads the
   correct columns and computes normalized L1/L2 values and pairwise orders
   without overstating convergence.
7. A Z4c checkpoint restart cannot silently reinitialize analytic puncture
   data.
8. The reported one-A100 memory lower bound follows from the authoritative
   3,200-leaf-block tree and actual field counts.  Verify whether N=12 and
   N=16 are impossible on one 40-GB A100 before runtime buffers, and identify
   any overlooked memory-saving implementation that would preserve the exact
   requested grid and diagnostics.

Also review the local verification claims: fresh FO-GH, Z4c, and two-rank MPI
builds; serial/MPI agreement; machine-precision initial-data equality; and
checkpoint/restart equality.  Treat the README as a report, not as proof:
flag any claim that lacks a committed reproducer or sufficient retained
artifact.

Return a severity-ranked report with exact file/line references.  Separate:

- correctness defects;
- scientific-comparison or convergence threats;
- missing evidence/reproducibility gaps;
- performance/resource risks;
- cosmetic or maintainability issues.

For every issue, state the observed evidence, consequence, and smallest safe
remedy.  Explicitly say whether the branch is ready for (a) one-A100 preflight,
(b) the coarse N=8 case, and (c) the full N=8/12/16 t=32M campaign.  Do not
interpret compilation, a short smoke test, or a reached checkpoint as
long-time stability or production qualification.
