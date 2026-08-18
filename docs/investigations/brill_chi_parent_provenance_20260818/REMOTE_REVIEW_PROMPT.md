# Read-only review prompt: N256 active-fine chi failure

Please perform a skeptical, read-only review of:

- Repository: https://github.com/HengruiZhu99/athenak
- Branch: `codex/amr-history-record-replay-brill-20260817`
- Report: `docs/investigations/brill_chi_parent_provenance_20260818/REPORT.md`
- Evidence manifest: `docs/investigations/brill_chi_parent_provenance_20260818/evidence_manifest.json`
- Preserved diagnostic patch/source bundle:
  `docs/investigations/brill_chi_parent_provenance_20260818/evidence/final-diagnostic-source/`

Qualified numerical source is commit
`ac75c8d348da91b38cbc6855b5fba51cd3089663`, tree
`6284882bd06e8db379495675aba7a4f153fb4afa`. The diagnostic source is a
default-off, intentionally uncommitted patch whose exact bytes are preserved
and checksummed.

## Decisive evidence

The N256 replay stopped at cycle 5546, RK stage 3, time
`0x1.5124ccccccd9bp+3` (10.535742187500366 M). Immediately after
`ExpRKUpdate`, before restriction, two active fine cells became negative:

- GID 35, rho=5.138671875, z=-0.001953125, chi=-1.3316138503433481
- GID 60, rho=5.138671875, z=+0.001953125, chi=-1.3409347716159825

At stage 2 of the same cycle, active and consumed-coarse minima were finite and
positive. The cells form an equatorially paired feature on adjacent same-level
block active edges. Replay events 10 and 11 matched their authority times and
trees exactly. Shadow AMR recorded 62,312 derefinement and 690 refinement
requests; the two failing blocks eventually requested refinement, but chi was
already invalid at the post-update checkpoint.

The required final classification is `ACTIVE_FINE_CHI_FAILURE`. No production
numerical correction or further control run was performed because this was an
explicit stop condition.

## Questions

1. Verify that S0 is placed strictly after the RK update and before every
   restriction/cache/communication/BC/prolongation writer, and identify any
   hidden asynchronous operation that could invalidate that ordering.
2. Inspect the chi RHS and RK update path. Which individual terms can plausibly
   produce a roughly -1.7 change from a previously positive state in one stage?
3. Could the RHS stencil at these adjacent-block active-edge cells consume
   stale, parity-inconsistent, or coarse-fine ghost data even though the stored
   active value first becomes invalid at S0?
4. Does the equatorial pairing and identical rho favor a legitimate symmetric
   numerical mode, an interface stencil issue, or another mechanism?
5. Propose the smallest trigger-only term-resolved diagnostic needed to identify
   the first bad RHS contribution. Specify exact checkpoints and fields.

Please keep observations, inferences, and hypotheses separate. Do not propose
chi floors, clipping, weakened positivity gates, broad parameter sweeps, or a
restriction fallback unsupported by this provenance. Do not claim convergence,
a continuum instability, or Figure-3 reproduction from these data.
