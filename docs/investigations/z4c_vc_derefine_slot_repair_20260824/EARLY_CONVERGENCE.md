# Early common-tree convergence qualification

## Result

The documented early N128/N256/N512 gate passes through `t=2.5 M`, beyond the
unpatched N512 failure time `2.4953913377 M`.

These three evolutions used production-repair commit
`d2596707e808aea7ec6167df937d71dc4dbe429e`. The later final review source
`6dd20656a305f2543bbbd7001550c6ac67019180` adds only default-off diagnostics,
test ownership fixtures, and an input declaration; it makes no production
numerical change. The bounded N128/N256/N512 event-3 histories were rerun at
the final source and are bitwise identical to the corresponding repaired
event-3 histories.

All runs replayed the same 24 accepted physical-time hierarchy events exactly,
ending with 176 MeshBlocks and maximum physical refinement level 4.  Replay
reported zero-ULP time error and exact leaf sets/checksums.  The production
configuration remained O4 + q6, RK4, CFL 0.15, KO 0.02, scale-controlled
telegraph lapse, Gamma-driver shift, and zero Z4 damping.

## History norms

At `t=2.5 M`:

| Resolution | C | H | M | Z |
|---|---:|---:|---:|---:|
| N128 | 4.7009e-2 | 2.0589e-2 | 6.3785e-3 | 4.5044e-3 |
| N256 | 2.9713e-4 | 9.7700e-5 | 3.4845e-5 | 3.8616e-5 |
| N512 | 9.2886e-5 | 1.8963e-5 | 6.8082e-6 | 1.6706e-5 |

Each terminal norm improves monotonically with resolution.  Across the common
axis-proper-time interval, median effective orders are 7.81 (C), 7.89 (H),
7.86 (M), and 7.74 (Z).  These high integral-norm orders are observations over
this bounded smooth interval, not a claim of the formal global order.

## Field-level gate

Fields were sampled at common coordinate times 0, 0.25, 0.5, 1, 1.5, and 2 M
on the common N128 lattice.  Minimum finite trusted-core effective orders
include:

| Quantity | Minimum order |
|---|---:|
| chi | 3.697 |
| conformal metric components | 3.566--3.818 |
| Khat | 3.670 |
| Theta | 3.276 |
| alpha | 2.763 |
| evolved-minus-metric Gamma components | 2.992--3.711 |
| H constraint | 2.868 |
| Kretschmann scalar | 2.167 |

Symmetry-forced zero components have undefined ratios and are excluded rather
than counted as failed convergence.

## Admissibility, axis, and causality

- Minimum chi over sampled trusted/full regions: 0.32113 (N128), 0.32136
  (N256), and 0.32137 (N512).
- Minimum conformal-metric SPD pivot: 0.19738 at every resolution.
- Same-level and coarse/fine shared-node spreads: zero in sampled outputs.
- Maximum axis regularity correction: `8.76e-15`, `3.63e-14`, and `1.45e-13`
  for N128, N256, and N512; all nonfinite counts are zero.
- At N256 and `t=2.5 M`, integrated causal distance is 3.7815, leaving trusted
  radius 12.2185 from the outer boundary.

## Interpretation and limit

Observation: the repaired source passes the stated early gate and no longer
exhibits the old resolution-growing event-3 failure.

Inference: the native-VC derefinement slot corruption was an independent and
material source defect.

No complete Figure-3 evolution was run.  This document does not claim late-time
convergence, critical behavior, horizon formation, or general VC-AMR
qualification.

Machine-readable results are in `evidence/analysis/early_history` and
`evidence/analysis/early_fields`; compact plots are in their `figures`
subdirectories. Exact hashes for all retained restart files are listed in
`EVIDENCE_MANIFEST.json`; restart payloads remain on Perlmutter and are not
committed to Git.
