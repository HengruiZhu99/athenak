# Common-hierarchy symmetric-O4 Brill convergence report

## Verdict

The N128/N256/N512 experiment does **not** establish O4 convergence. It gives a
stronger negative result: on the exact executed prefixes of one physical AMR
schedule, the constraint histories generally grow with resolution, most sampled
Z4c fields have zero or negative self-convergence order, and N512 reaches the
same metric-positive-definiteness guard earlier than N128.

The formal dispositions are:

| question | disposition |
|---|---|
| replay | `EXACT_CROSS_RESOLUTION_REPLAY` (executed prefixes) |
| bulk fields | `DIVERGES_WITH_RESOLUTION` |
| constraints | `DIVERGES` |
| failure trend | `FAILURE_EARLIER_WITH_RESOLUTION` |
| N256 authority for N128 | `AUTHORITY_UNDERRESOLVED` |
| N256 authority for N512 | `AUTHORITY_OVERRESOLVED` |
| optional temporal control | `NOT_RUN` |
| overall | `O4_NONCONVERGENT` |

This is neither a convergence claim nor a Figure-3 reproduction. It also does
not isolate a unique source bug. In particular, the N128 and N512 failures have
the same inadmissible metric-pivot class but occur at different physical
locations, so the data do not support a single fixed-interface attribution.

## Frozen comparison

The three cases used Brill amplitude `A=-0.047`, the same IrisK coefficients,
the same domain `rho=[0,16]`, `z=[-16,16]`, the same physical 4-by-8 root
MeshBlock lattice, O4/RK4, `CFL=0.15`, KO `0.02`, high-order AMR transfer,
`dchi_max=0.01`, derefinement below `0.25*dchi_max`, max-K-scaled telegraph
lapse with `tau=kappa=1`, Gamma-driver shift with `eta=2`, no Z4c damping, and
no chi floor. Cells per physical MeshBlock were 16-by-16, 32-by-32, and
64-by-64 at N128, N256, and N512 respectively.

Production moved from queued Aurora capacity work to one Perlmutter
`shared_interactive` A100-80GB at the user's direction. Aurora job `8770096`
had already qualified the implementation; job `8770135` was cancelled while
queued and produced no science. Perlmutter used executable SHA
`bf0f69b0cb68fdf2fb0035b4a29d55a30f1cef224acce9da77b9089f2207bd3b`,
built from `d0d0b648bab09afb33453132075f1b813306526a` with CMake-cache SHA
`232dd20f55063c684b25b0b6910ad3f2dca4a17b88ed3ab31f404dd01f4dc2f6`.
All 12 selected CUDA tests and the short record/replay/restart qualification
passed before production.

## Authority and replay

N256 produced an immutable 164-event authority schedule (event 0 through 163),
SHA `874551cc68e7dab4d40b854b31ab6b42aff9d2eae0ca9faf5985c41ef14a589f`.
Its last event is at `t=16.898790783062829 M`, cycle 10021, with 1166 leaves,
maximum logical level 17, and tree checksum `3ba7925c8d151eff`.

N128 reproduced events 1--27 exactly before its state failure; N512 reproduced
events 1--17 exactly before its state failure. Every executed event had the
authority time bit-for-bit (`0 ULP`) and the exact authority tree checksum.
Neither replay reached the full authority end, so `EXACT_CROSS_RESOLUTION_REPLAY`
is deliberately limited to the executed prefixes.

The timestep analysis separates deliberately clipped replay steps from the
underlying `0.15*min(dt_spatial,dt_source)` candidate. N512's very small early
history `dt` values are authority-event clips and must not be interpreted as
an early physical timestep collapse. The underlying N512 candidate never fell
below `1e-4 M` before its state failure; its terminal candidate was
`1.61744e-4 M`. N128's underlying candidate crossed `1e-3 M` at
`t=16.73724 M` and `1e-4 M` at `t=16.79613 M`. N256 crossed `1e-3`, `1e-4`,
and `1e-5 M` at `t=14.08066`, `16.84305`, and `16.89671 M` respectively.

## Native-AMR shadow

The replay tree was authoritative, but each replay case evaluated its own
native `dchi` request without allowing it to change topology.

| resolution | records | agrees | would refine earlier | would derefine |
|---|---:|---:|---:|---:|
| N128 | 143,504 | 56 | 133,268 | 10,146 |
| N512 | 686,490 | 12,886 | 609 | 672,961 |

Thus raw `dchi` says the N256 logical tree is mostly too shallow for the coarse
N128 representation and overwhelmingly more refined than the fine N512
representation wants. This rules against the narrow claim that N512 failed
because the replayed N256 tree was simply too coarse according to the same
`dchi` sensor. It does **not** globally rule out parent under-resolution of
another field or a sensor blind spot.

See [native-AMR decisions](figures/native_amr_shadow.png) and
[native dchi histories](figures/native_amr_sensor_vs_tau.png).

## Field self-convergence

Fields were sampled without changing the evolution state onto one physical
meridional lattice (`Delta rho=Delta z=0.25`) at common central proper times
`tau_c=5, 7.5, 9`. Spatial sampling used a five-point tensor interpolant within
active leaf blocks; four retained snapshots supplied cubic temporal
interpolation. Errors use a coordinate-ring RMS (weight proportional to
`rho`; the constant `2*pi` cancels). Results are also split into the axis,
block interiors, and coarse-fine neighborhoods.

Entire-domain effective orders were:

| field | tau=5 | tau=7.5 | tau=9 |
|---|---:|---:|---:|
| chi | -0.558 | -0.308 | -0.176 |
| alpha | 1.482 | 1.037 | 0.950 |
| K | -0.347 | -0.079 | 0.071 |
| Theta | -0.040 | -0.075 | 0.048 |
| Axx | -0.342 | -0.029 | 0.140 |
| Axy | 1.481 | 0.098 | 0.298 |
| Ayy | -0.029 | 0.003 | 0.062 |
| Gamma-x | -0.150 | 0.474 | 0.535 |
| Gamma-y | -0.254 | 0.127 | 0.304 |

Ideal O4 behavior would give `p` near 4. Only the lapse and isolated components
show weak positive, sub-fourth-order behavior; chi and most geometric fields
are flat or divergent. Block interiors do not rescue the result: median orders
over the three times include chi `-0.357`, K `-0.177`, Theta `-0.180`, and Axx
`-0.134`. Coarse-fine neighborhoods are at least as poor (chi median `-0.601`).

The retained binary inventory did not include a spatial Kretschmann field, so
Kretschmann is compared only through authenticated history maxima/axis values.
See [field convergence](figures/field_convergence_order.png) and the exact
[field table](data/field_convergence.csv).

## Constraint histories and AMR-event jumps

On the trusted common interval `tau_c <= 9`, median effective orders are:

| history quantity | median p |
|---|---:|
| C-norm2 | -1.220 |
| H-norm2 | -0.915 |
| M-norm2 | -1.303 |
| Z-norm2 | -1.036 |

For example, near `tau_c=7.5`, C is approximately `20.4`, `46.3`, and `104.2`
at N128, N256, and N512. This is divergence with resolution, not merely failure
of the terminal tail.

The production history measure is already the correct proper axisymmetric ring
measure:

`2*pi*rho*dx1*dx2*sqrt(abs(det(gamma)))`.

It contains no fictitious collapsed-y width. The hierarchy-correlated jumps
therefore are not caused by a missing division by `dx3`. Corresponding-event
median absolute log10 C jumps are `0.0132`, `0.0146`, and `0.00104` at N128,
N256, and N512; rare jumps remain large at every resolution. The common-tree
experiment does not show clean h-convergence of the jumps.

See [constraints versus proper time](figures/constraints_vs_tau.png),
[constraint order](figures/constraint_convergence_order.png), and
[event-jump comparison](figures/authority_event_jump_convergence.png).

## Terminal scaling

| case | terminal boundary | t/M | tau_c/M | location | evidence |
|---|---|---:|---:|---|---|
| N128 | state guard, `nonpositive_metric_pivot_1`, post-RK stage 1 | 16.79703 | 10.40293 | rho=0.00586, z=-0.00586, near axis/block edge, relative level 6 | `b8c1b543...` |
| N256 | scheduler timeout amid runaway; no state JSON | 16.89879 | 10.38367 | terminal C=3.36e6, max|K|=1.88e4, max Kretschmann=1.22e18 | manifest `e5073e81...` |
| N512 | state guard, `nonpositive_metric_pivot_1`, post-RK stage 3 | 15.35962 | 9.37982 | rho=4.83203, z=-0.00391, exactly a block edge, relative level 3 | `2222d531...` |

N512 fails roughly `1.02 M` earlier in central proper time than N128 and at a
different physical location. N256 was not restarted because it was already in
a catastrophic runaway with a collapsing physical timestep; a walltime
continuation could not turn it into qualifying convergence evidence. This is a
qualification limitation: N256 has no captured state-failure record.

The result favors a resolution-sensitive discrete instability over ordinary
under-resolution. It does not determine whether the origin is bulk evolution,
an AMR/interface operation, or an under-resolved variable not tracked by
`dchi`.

## Figure 3 context

[The overlay](figures/fig3_o4_common_tree_n128_n256_n512.png) uses the existing
authenticated rendered-paper curves without changing their digitization or
axis definitions. It includes the three common-tree trajectories and the prior
N256 O4 `dchi=0.02` trajectory only as explicitly unmatched secondary context.
The plot demonstrates disagreement/runaway; it is not a Figure-3 reproduction.

## Evidence, inference, and next diagnostic

Observed:

- exact replay times and trees for every executed N128/N512 authority event;
- negative/near-zero field orders at `tau_c=5,7.5,9`;
- negative median constraint orders through `tau_c=9`;
- native N512 `dchi` overwhelmingly asks to derefine the authority tree;
- an earlier N512 metric-SPD failure at an off-axis block edge.

Inferred:

- simple N512 parent under-resolution according to `dchi` is not the dominant
  explanation;
- increasing cells per physical MeshBlock does not stabilize this hierarchy;
- the instability is resolution sensitive and is not explained by the
  axisymmetric history measure.

Still hypotheses:

- a field other than chi is under-resolved before transfer;
- high-order transfer/interface coupling amplifies a high-frequency mode;
- a bulk/gauge mode grows and only becomes visible at a block/interface guard.

The smallest decisive follow-up is a short stage/writer-resolved provenance
window around the earlier N512 failure, recording the first inadmissible metric
or high-frequency field after RK update, restriction, receive/BC, coarse-cache
refresh, and prolongation. It should not change floors, guards, gauge, KO, CFL,
or transfer operators.

## Limitations

- N128 and N512 failed before the complete 164-event authority schedule.
- N256 ended by scheduler timeout during runaway rather than by a detailed
  state-admissibility record.
- The half-CFL control was not run: divergence and the earlier N512 failure are
  already unambiguous, so it would not repair the primary qualification.
- Spatial Kretschmann convergence is unavailable from the retained binary set.
- Common-tree replay controls topology but cannot alone identify the writer
  responsible for the instability.
- No convergence, physical critical behavior, horizon, or Figure-3
  reproduction claim is made.

Machine-readable results are in [comparison_summary.json](comparison_summary.json),
and every local product is covered by the detached checksum layers.
