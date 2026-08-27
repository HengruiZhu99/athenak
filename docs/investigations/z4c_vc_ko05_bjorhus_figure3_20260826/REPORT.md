# KO=0.5 native-VC Brill campaign and experimental CPBC assessment

## Final verdict

KO=0.5 does not provide a stable, convergent Figure-3 collapse evolution. The
matched-tree N128/N256/N512 campaign is near fourth order early, then loses
convergence first outside `r<=4M` around the rho approximately 5M mode. N256
fails at `t=14.405106M` with an indefinite conformal metric after a catastrophic
noncentral runaway. N512 reaches the forced common stop but is also severely
unstable; N128 is much milder. The nonmonotone resolution ordering rules out a
convergence claim.

The numerical curves end at common central proper time about 8.827, before the
published first peak near 10.31 and deep minimum near 12.62. Figure 3 is not
reproduced, and neither its minimum nor rebound is reached.

The Cartoon constraint history uses the proper axisymmetric ring measure, so
the observed jumps are not caused by a fictitious collapsed-y cell width.

The experimental Bjorhus option exposed and motivated repair of a real
non-diagonal frame-initialization bug. Final CUDA manufactured tests pass, but
the bounded boundary campaign remains incomplete. The Rout16 CPBC case develops
a corner-local boundary runaway and fails at t=3.244461; the Rout128 CPBC case
was intentionally cancelled at t=1.508791. No claim that CPBC improves the
boundary is supported.

## Answers to the requested questions

1. **Stable to t=30?** No. The common campaign stops at t=14.405106.
2. **Failure mode?** N256 fails a strict positive-definite metric pivot after a
   rho approximately 4.88 runaway. N512 has large constraints/curvature and a
   collapsed timestep at the common stop. N128 remains much less contaminated.
3. **Does the rho approximately 5 mode converge away?** No demonstrated
   convergence; its terminal amplitude is strongly nonmonotone.
4. **Constraint orders?** Approximately fourth order early. Outside r<=4 the
   orders degrade by t about 8 and become negative/nonmonotone later; see
   `CONVERGENCE.md` and the CSVs.
5. **Figure 3 reproduced?** No; the evolution ends before the first peak.
6. **Central observable convergent?** Not in the requested Figure-3 region.
7. **Does CPBC reduce boundary constraints?** Not established. The comparison
   is incomplete and the Rout16 CPBC option is unstable.
8. **Does CPBC leave the central trajectory unchanged?** Only over an early
   common interval; this is not a qualified late-time result.
9. **Dominant remaining error?** Current evidence strongly favors a bulk/AMR
   high-frequency/interface mechanism near rho=5 over the outer boundary or
   Cartoon axis, but it does not mathematically exclude boundary effects or
   identify a unique source bug.
10. **Next production configuration?** None is scientifically qualified for a
    critical-collapse production run. The smallest next step should localize
    the first loss of resolution/transfer consistency on the matched hierarchy,
    not extend the collapse or sweep parameters.

## Evidence versus inference

Observed: exact hierarchy replay; early fourth-order regional inventories;
late nonmonotone runaway near rho=5; N256 metric-pivot failure; incomplete CPBC
discriminator; Rout16 CPBC corner failure.

Inferred: a bulk/AMR mode is more likely than direct outer-boundary contamination
for the main campaign. This remains a ranked hypothesis, not a proof.

Not claimed: t=30 stability, Figure-3 reproduction, convergence, horizon
formation, physical critical behavior, or a fully isolated production source
bug.

## Artifact map

- `KO05_AUTHORITY.md`: trajectory and hashes.
- `CONVERGENCE.md`: global/regional orders and limitations.
- `FIGURE3_REPRODUCTION.md`: published-curve comparison.
- `BJORHUS_DERIVATION.md`, `BJORHUS_TESTS.md`: CPBC mathematics/tests.
- `CPBC_BOUNDARY_COMPARISON.md`: bounded incomplete discriminator.
- `analysis/`: plotting scripts, compact CSV/JSON, and figures.
- `evidence/`: compact run evidence and strict source manifests.
- `EVIDENCE_MANIFEST.json`: artifact hashes and dispositions.
