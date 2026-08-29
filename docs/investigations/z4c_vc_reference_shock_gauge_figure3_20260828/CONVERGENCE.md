# Same-tree N128/N256/N512 convergence analysis

## Verdict

`EARLY_O4_COMPATIBLE; COLLAPSE_WINDOW_ORDER_DEGRADES; FULL_CONVERGENCE_NOT_ESTABLISHED`

The three runs use the same physical MeshBlock bounds, exact accepted N256
LogicalLocation tree, and event times. Only the cells per physical MeshBlock
change in a 2:1 sequence: 16, 32, and 64. This is a controlled same-interface
resolution comparison.

The result is a strong resolution discriminator, but not a uniform convergence
demonstration. Central curvature and lapse are approximately fourth order at
early times, while observed order degrades during collapse. Constraint
amplitude orders differ materially between the N128/N256 and N256/N512 pairs.
N256 also ends at proper time 11.286, so no three-level order exists across the
deep-minimum/rebound interval.

## Run dispositions

| case | cells/MB | final coordinate time | final proper time | disposition |
|---|---:|---:|---:|---|
| N128 | 16 x 16 | 45.0 | 19.33240 | reached tlim; constraint invalid |
| N256 | 32 x 32 | 30.0 | 11.28631 | reached tlim; constraint invalid |
| N512 | 64 x 64 | 38.65233 | 14.98253 | healthy walltime stop after gate; constraint invalid at peak |

All cases replayed the same 212-leaf final hierarchy. There were no independent
AMR decisions.

## Figure-3 resolution sequence

| case | peak proper time | peak `log10(abs(Kretschmann))` | peak C integral |
|---|---:|---:|---:|
| N128 | 10.31396 | 4.29765 | 107.608 |
| N256 | 10.30333 | 5.01349 | 48.2330 |
| N512 | 10.30811 | 5.38112 | 4.09930 |
| published | 10.30683--10.31384 | 5.47778--5.48688 | not available |

The peak time is resolution insensitive, while peak amplitude moves
monotonically toward the published result and constraint contamination drops
substantially. N512 alone resolves the published minimum/rebound morphology;
N128's nominal late extrema are constraint dominated and N256 does not reach
that interval.

## Observed orders

All curves are interpolated at the same central proper time. For central fields
the reported Richardson order is

```text
p = log2(abs(u128-u256) / abs(u256-u512)).
```

Median orders are:

| window in proper time | central Kretschmann | central lapse |
|---|---:|---:|
| 0--8 | 4.86 | 3.93 |
| 8--10 | 3.34 | 3.36 |
| 10--11.286 | 2.10 | 1.40 |

The early values are compatible with the configured O4 bulk method. The
collapse-window degradation is direct evidence that the sequence is no longer
uniformly asymptotic near the first peak.

For constraints, history stores `integral C^2 dV`; the analyzed amplitude is
its square root. Representative median pairwise amplitude orders are:

| family/window | N128/N256 | N256/N512 |
|---|---:|---:|
| C, 0--8 | 3.46 | 1.06 |
| H, 0--8 | 3.63 | 2.41 |
| M, 0--8 | 3.60 | 2.58 |
| C, 8--10 | 2.76 | 0.61 |
| H, 8--10 | 3.11 | 2.89 |
| M, 8--10 | 3.10 | 3.03 |
| C, 10--11.286 | 1.03 | 2.38 |
| H, 10--11.286 | 0.98 | 2.40 |
| M, 10--11.286 | 1.01 | 2.42 |

The positive trend confirms resolution sensitivity, but the pair mismatch—most
clearly for aggregate C—precludes a single observed-order claim.

## Interpretation

The same-tree result strongly supports bulk under-resolution as a major cause
of the N256 constraint catastrophe: doubling cells per physical MeshBlock
reduces the peak constraints by roughly an order of magnitude, moves the
curvature amplitude toward the published result, suppresses rho≈5
fourth-difference content, and extends survival through the full Figure-3
interval.

It does not prove that AMR interfaces are harmless. All runs use the same
interfaces, and N512 still develops order-unity squared constraint integrals at
the peak. A persistent interface/axis contribution can coexist with bulk
under-resolution.

## Claim boundary

Supported:

- early central-field behavior compatible with O4 convergence;
- clear monotonic resolution improvement in the peak and constraints;
- failure of a uniform asymptotic order through collapse;
- no three-level late-time order beyond N256's final proper time.

Not supported:

- a fully convergent Figure-3 reproduction;
- fourth-order convergence through the peak;
- identification of a unique source-level bug;
- changing the production gauge, KO, transfer, or positivity checks.
