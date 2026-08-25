# Completion audit

This ledger maps the controlling Goal Mode requirements to current evidence.
The optional late Rout=128 N256 control is explicitly identified as optional
and omitted; it is not used to weaken or block the independent KO study.

## Repository and preservation

- Exact base: `953f2724c00a2efd2f9fad91ae9a784639954a3b`, tree
  `a03bd3f61b9d766adc7083c87ee701bd6d62becb`.
- Campaign branch:
  `codex/z4c-vc-boundary-convergence-ko-scan-20260824`.
- The base-to-campaign diff contains only this investigation directory. No
  production numerical source or qualified VC derefinement repair changed.
- The Perlmutter executable source tree is source-identical to the exact base;
  its distinct commit contains documentation history only. Exact executable,
  source, input, and coefficient hashes are in `EVIDENCE_MANIFEST.json`.

## Boundary experiment

| requirement | disposition | authoritative evidence |
|---|---|---|
| Reuse authenticated Rout=16 N128/N256/N512 | achieved | `SMALL_DOMAIN_RADIAL_CONVERGENCE.md`; original run hashes in the manifest |
| Production-matched 2D Cartoon measure | achieved | `STATIC_AUDIT.md`; exact ring measure and canonical VC ownership |
| R=4/8/12/14/full and shell localization | achieved | `analysis/small_domain_radial/`; shell and maxima CSVs |
| Direct pairwise and self-order curves/checkpoints | achieved | `radial_constraint_convergence.csv`, `convergence_checkpoints.csv`, and the small-domain report |
| Conservative characteristic reach audit | achieved | `BOUNDARY_CAUSALITY.md` and three causality CSVs |
| Rout=128 dyadic minimum-resolution layout | achieved | `SMR_LAYOUT_PROOF.md` and authenticated input |
| Inner initialization equivalence | achieved | 33,153 canonical vertices; `analysis/n256_inner_initial_equivalence.json` |
| N256 record authority through t=6.5 | achieved | exit 0, authority/ledger/logs, `LARGE_DOMAIN_AUTHORITY.md` |
| N256 record/replay identity | achieved | bitwise-identical 264x71 histories, SHA-256 `701f3093893017ea064087e96ddc0dbcd4136cae36b50f0330c657d98c5fb9c5` |
| N128/N512 common-tree replay through t=6.5 | achieved | exact requested trees, exit 0, run evidence |
| Matched Rout16/Rout128 radial convergence | achieved | `LARGE_DOMAIN_CONVERGENCE.md`, `BOUNDARY_COMPARISON.md`, terminal tables/plots |
| Central trajectory unchanged | achieved for qualified interval | maximum small/large deviation `6.68e-9` dex |
| Optional late Rout128 N256 control | not run, optional | recorded limitation; not used in either primary verdict |

Boundary verdict: `BOUNDARY_CONTAMINATION_CONFIRMED`.

## Independent KO experiment

| requirement | disposition | authoritative evidence |
|---|---|---|
| Original Rout=16 N256; only diss changes | achieved | per-run commands and common input; diss overrides 0.02/0.05/0.10/0.20/0.50 |
| Fresh native record authority per case | achieved | five authority/ledger families; no replayed baseline tree |
| Stage A through t=6.5 | achieved | all five exit 0 |
| Stage B through at least t=9.5 | achieved | all five exit 0 |
| Stage C toward t=11.3 | achieved | four reach 11.3; baseline bounded-terminated at numerical timestep collapse t=11.191917 |
| Global and R=4/8/12 C/H/M/Z | achieved | `analysis/ko_stageC/ko_radial_constraints.csv` and plots |
| axisKret/reference overlay and quantitative deviation | achieved | overlay plot and `ko_summary.csv` |
| chi, pivots, dt, requests, first refinement, leaves, level | achieved | state-health CSV, histories, authority ledgers, and summary CSV |
| Constraint and curvature locations | achieved at retained snapshots | two extrema summary CSVs; 233 curvature snapshots; no between-output maximum claim |
| Full required final table | achieved | `KO_SCAN_N256.md`, `REPORT.md`, and `REPORT.pdf` |

KO verdict: `KO_STRONG_EFFECT`.

## Resource and artifact closure

- New GPU work used two one-GPU `gpu_shared_interactive` allocations, one MPI
  rank per A100-PCIE-40GB; N512 peaked at 36,581 MiB without escalation.
- Both allocations completed, and the final scheduler check found no campaign
  or user allocations remaining.
- Every required Markdown/LaTeX/PDF deliverable, analysis script, CSV, and
  figure is present. `SHA256SUMS` is the complete local inventory and validates
  from the investigation directory.
- `REPORT.pdf` was compiled after the final content update, all seven pages
  were rendered, and every page was visually checked for clipping and table or
  figure defects.

## Claim boundary

This campaign qualifies the stated boundary diagnosis through t=6.5 and a
one-resolution KO stability discriminator through the stated terminal times.
It does not establish late Figure-3 convergence, a continuum-stability result,
a horizon or critical exponent, or a production KO default.
