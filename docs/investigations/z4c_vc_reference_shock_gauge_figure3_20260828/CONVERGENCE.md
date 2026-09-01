# Same-tree N128/N256/N512/N1024 convergence analysis

## Verdict

`FINE_TRIPLE_O4_COMPATIBLE_THROUGH_FIRST_PEAK; CONSTRAINT_CONVERGENCE_NONUNIFORM; LATE_CONVERGENCE_NOT_ESTABLISHED`

The four runs use the same physical MeshBlock bounds, exact accepted N256
LogicalLocation tree, and event times. Only the cells per physical MeshBlock
change in a 2:1 sequence: 16, 32, 64, and 128. This is a controlled
same-interface resolution comparison. N128--N512 ran on Aurora PVCs and N1024
ran on four Perlmutter A100s; a matched N512 interval agrees across those
backends to roundoff scale, strongly deprioritizing the backend change as a
confounder without making it mathematically identical.

The fine N256/N512/N1024 triple is approximately fourth order in central
curvature and lapse through the first-peak interval. The peak constraint
amplitudes fall by about another order of magnitude at N1024. This materially
strengthens the convergence evidence through the first peak. It is not a
uniform late-time convergence demonstration: N256 ends at proper time 11.286,
so no three-level order exists across the deep-minimum/rebound interval, and
the N1024 peak still crosses the campaign's strict C-integral validity gate.

The initial constraints themselves converge cleanly. Initial C squared
integrals for N128, N256, N512, and N1024 are `1.30505e-1`, `6.85048e-4`,
`2.88883e-6`, and `1.28813e-8`. The N512/N1024 amplitude order is `3.90`.
The later loss of a uniform order therefore does not originate in unresolved
initial-data interpolation.

## Run dispositions

| case | cells/MB | final coordinate time | final proper time | disposition |
|---|---:|---:|---:|---|
| N128 | 16 x 16 | 45.0 | 19.33240 | reached tlim; constraint invalid |
| N256 | 32 x 32 | 30.0 | 11.28631 | reached tlim; constraint invalid |
| N512 | 64 x 64 | 38.65233 | 14.98253 | reached analysis endpoint; constraint invalid at peak |
| N1024 | 128 x 128 | 38.65233 | 14.98066 | reached tlim; constraint invalid at peak |
| N1024 CFL 0.40, KO 0.10 | 128 x 128 | 16.72465 | 8.68734 | failed state-admissibility gate |

All cases replayed the same 212-leaf final hierarchy. There were no independent
AMR decisions. N1024 accepted both replay events at the exact hexadecimal
authority times with zero ULP time error and matching tree checksums.

## Figure-3 resolution sequence

| case | peak proper time | peak `log10(abs(Kretschmann))` | peak C integral |
|---|---:|---:|---:|
| N128 | 10.31396 | 4.29765 | 107.608 |
| N256 | 10.30333 | 5.01349 | 48.2330 |
| N512 | 10.30811 | 5.38112 | 4.09930 |
| N1024 | 10.30964 | 5.47867 | 0.0388727 |
| published | 10.30683--10.31384 | 5.47778--5.48688 | not available |

The N1024 peak time and amplitude both lie inside the published bands, with no
time or amplitude fit. N512 and N1024 both resolve the published rebound:
N1024 gives `(tau, log10(abs(Kretschmann))) = (13.21271, -2.81258)`. Its deep
minimum occurs at `tau=12.61950` but reaches `-7.88841`, deeper than the
published `-6.54553`--`-5.20673` range. N128's nominal late extrema are
constraint dominated and N256 does not reach that interval.

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

For the fine N256/N512/N1024 triple, the corresponding medians are:

| window in proper time | central Kretschmann | central lapse |
|---|---:|---:|
| 0--8 | 4.10 | 3.99 |
| 8--10 | 3.89 | 3.79 |
| 10--11.286 | 5.91 | 3.34 |

The fine triple is compatible with O4 convergence through the first peak. The
Kretschmann value above four in the final window is treated as local
superconvergence of the sampled curve differences, not as an accuracy-order
claim above the configured method.

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

For the N512/N1024 pair, median constraint-amplitude orders are near zero away
from collapse for several families because both curves sit on a shared
initial-data/diagnostic floor. In the `10--11.286` collapse window they rise to
`2.69` (C), `4.00` (H), `3.74` (M), and `0.27` (Z). Peak amplitudes decrease as:

| amplitude | N128 | N256 | N512 | N1024 |
|---|---:|---:|---:|---:|
| C | 10.373 | 6.945 | 2.025 | 0.197 |
| H | 9.405 | 6.413 | 1.905 | 0.178 |
| M | 5.036 | 3.087 | 0.826 | 0.0673 |
| Z | 0.802 | 0.235 | 0.0680 | 0.0633 |

The N1024 squared C integral first crosses `0.01` at `tau=10.17835` and peaks
at `0.0388727` near `tau=10.30543`; it never crosses `0.1`. This is a large
improvement, but the pre-existing strict peak-validity gate is still failed.

The N512/N1024 constraint curves nevertheless do not share a uniform order.
Their median amplitude orders over `tau=0--8` are `0.007` (C), `0.081` (H),
`0.111` (M), and `0.001` (Z), consistent with a shared evolution/diagnostic
floor after the high-order-convergent initial slice. Over `tau=10--11.286`
they become `2.69`, `4.00`, `3.74`, and `0.27`, respectively. C/H/M improve
substantially during collapse; Z does not show useful fine-pair convergence.

## Same-resolution CFL/KO ablation

The exact-tree N1024 ablation changed two controls together: CFL `0.15 ->
0.40` and KO `0.50 -> 0.10`. Its initial state and all initial constraint
integrals are identical to baseline N1024. It remains close through roughly
`tau=6`, but its C squared integral crosses `0.01`, `1`, `100`, and `10000` at
`tau=7.54619`, `7.85416`, `8.24578`, and `8.65239`. The final history row has
C squared integral `4.9679e4`.

The strict state diagnostic then fails at coordinate time `16.72465 M`, RK
stage 1, near `(rho,z)=(0.0625,19.96875)`. Chi remains positive (`0.91660`),
but the conformal metric has a negative second SPD pivot and
`det(gtilde)=-0.29184`. The partial curve ends at `tau=8.68734`, before the
published first curvature peak. This rules out the combined CFL/KO setting as
a remedy; because two parameters changed, it does not identify the individual
cause.

## Interpretation

The same-tree result strongly supports bulk under-resolution as a major cause
of the N256 constraint catastrophe. N1024 reduces the peak C/H amplitudes by
another factor of about ten relative to N512 and brings the central first peak
into the published band while retaining the exact same interfaces.

It does not prove that AMR interfaces are harmless. All runs use the same
interfaces, and N512 still develops order-unity squared constraint integrals at
the peak. A persistent interface/axis contribution can coexist with bulk
under-resolution.

## Claim boundary

Supported:

- fine-triple central-field behavior compatible with O4 convergence through
  the first peak;
- clear monotonic resolution improvement in the peak and C/H/M constraints;
- direct, unshifted N1024 first-peak agreement with the published band;
- no three-level late-time order beyond N256's final proper time.

Not supported:

- a constraint-qualified Figure-3 reproduction;
- three-level convergence across the late minimum and rebound;
- identification of a unique source-level bug;
- changing the production gauge, KO, transfer, or positivity checks.

The updated plots and machine-readable summary are under
`analysis/aurora_n128_n256_n512_perlmutter_n1024_cfl040_ko010/final/`.
