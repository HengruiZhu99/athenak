# Cut-sphere versus cubed-sphere: gauge pulse and puncture

The new spherical atlas avoids the old puncture crashes through `t=0.2`, but
develops large constraint errors. The original near-scri gauge-pulse failure
remains unresolved: its matched-collar spherical runs time out too early.
**These results do not justify blaming or clearing the continuum formalism.**
Here “cut sphere” means the old Cartesian mesh's outer active-cell mask and
ray-based ghost treatment; neither puncture comparison introduces an inner
excision boundary.

The [gauge comparison](cubed-sphere-spherical-comparison-20261013.md) and its
frozen evidence remain unchanged. Fresh puncture inputs, histories, logs,
source snapshots, checks and plots are in the
[puncture capsule](validation/cubed-sphere-puncture-comparison-20261013/README.md).
These are working-tree experiments based on commit
`24616eb8d52060414f234192548abbfc7758efb0`, not results from that commit alone.

| Test | Old Cartesian cut sphere | New six-patch spherical outer boundary | Interpretation |
| --- | --- | --- | --- |
| Original disturbed Minkowski, wave-map gauge | N24 fails at `0.79277344`, negative chi; N32 at `1.58373036`, negative lapse | Wide collar completes `t=2`; matched collar, quadratic and quartic ghosts, reaches only `0.0422561` before the 450-second cap | Wide-collar completion is encouraging; the matched failure has neither been reproduced nor excluded |
| Wormhole, native layer gauge, cubic ghosts | Invalid physical ADM state at reported mesh time `0.13174219`, negative metric determinant | Completes `t=0.2`, 702 steps | The old crash does not recur over this interval on this different operator/grid |
| Same wormhole, quartic ghosts | Negative lapse at reported mesh time `0.08925391` | Completes `t=0.2`, 702 steps | Higher ghost degree alone does not repair the cut-sphere evolution |
| Same wormhole, cubic ghosts, half timestep | Negative metric determinant at `0.13159961`, same cell as the full-step failure | Completes `t=0.2`, 1403 steps; constraints nearly unchanged | A simple timestep reduction does not cure either the native crash or the spherical constraint growth |

The native failure timestamps above are the reported `mesh_time`, not a
reconstructed Runge–Kutta substage time. The cubic failures occur at
`(-0.21875,-0.65625,-0.65625)`, `Omega=0.04541015625`; their determinants are
`-1.79592939` and `-0.70062814`. The quartic failure occurs at
`(-0.04375,-0.65625,-0.74375)`, `Omega=0.00712890625`, with lapse
`-0.0776539012`. The raw logs are decisive; two early receipt extractors missed
the macOS `uncaught exception` phrase. The independent review recounts those
failures without altering the original receipts.

For the pulse, the completed atlas control has minimum Omega `0.0486184`,
22.4 times that of the N24 failed cell. The capped matched-collar control has
minimum Omega `0.00217048`, within 0.016% of the old value. Those two outcomes
cannot be exchanged when judging the original failure.

The pulse and puncture are distinct tests. The pulse retains `a=0.5`, layer
`[0.05,0.95]`, wave-map gauge and `kappa1=10`. The puncture reproduces the
archived wormhole input: `M=0.5`, `R=1`, `a=1`, layer `[0.35,0.75]`, native
physical-trace layer lapse with preferred source disabled, `kappa1=5`,
dissipation `0.1`, and cubic ghosts unless explicitly varied. The original
two-step puncture smoke is reproduced before extending it. Its initial
Hamiltonian maximum is `4.95e-14`; the physical ADM dump recovers mass `0.5`
within `4.21e-6` over 2144 sampled cells, within the existing binary-output
tolerance. The fields change during those two steps.
The initial constraint checks use analytic initial jets; they check the
initializer/reconstruction, not convergence of the subsequent finite-difference
evolution operator.

The puncture atlas uses a uniform `16^3` Cartesian core of half-width
`0.25245594360687235` and six `41 x 12 x 12` shell patches. Its minimum active
Omega is `0.0071289062499997604`, matching the cut-sphere puncture's minimum
to roundoff. The fixed maximum timestep `0.00028515624999999044` matches the
native value and satisfies the mapped CFL calculation. The half-step case
halves the timestep cap, pole factor and CFL. This is a stronger collar/time
control than the wide-collar pulse run, but still changes core resolution,
sampling, derivatives, interfaces and logical dissipation along with topology.
The native mesh has 6152 active cells with spacing `0.0875`; the atlas has
4096 core and 35424 shell cells, with core spacing about `0.031557`.

![Puncture computational domain](validation/cubed-sphere-puncture-comparison-20261013/computational-domain.png)

The outer face is the sphere `r=1`, where Omega is zero. Evolved cell centers
have positive Omega; the probe extrapolates deviations into outer ghosts.
There is no evolved exact-scri degree of freedom or derived characteristic
constraint-preserving scri boundary condition. The six-patch topology makes
the geometric surface precise; it does not supply that missing condition.

The geometric and gauge reference remains **Minkowski throughout**. The
puncture uses the same analytic `LayerWormhole` initial data as the native
code. To preserve the native derivative policy, the adapter separately freezes
that initial profile and its analytic jets, then differentiates evolving
deviations from it. It does not subtract a black-hole RHS or replace the
Minkowski reference. Only the analytic Minkowski geometric residual is
subtracted as before. Completed mapped runs serialize both actual consumed const
reference caches and the separate initial reconstruction profile before and
after; each is byte-identical within its build. Raw struct padding means these
payload hashes are not a portable CPU/GPU equivalence test.
Using a Minkowski reference does not require the black-hole fields, including
their finite mass-dependent shear, to equal Minkowski at scri. A future scri
closure must respect those physical deviations rather than zeroing them.

Aurora admission job **8924150** matches eight two-step CPU metrics in both
regions with maximum relative difference `4.59e-10`, below `1e-7`; both its
two-step and `t=0.02` cases complete. All five CPU regressions pass in 155.07
seconds, and the mass-zero pulse regression agrees with the pre-puncture
adapter within `8.28e-13` relative error in the compared metrics. Evolution
job **8924158** completes all three `t=0.2` cases. Both jobs use
`capacity`, one node, a 30-minute allocation, and `CompactBinaryMerger`, in
sequence. Individual evolution cases take about 84–168 seconds. The executable
is Release SYCL with precise floating point, not a bounds-checked Debug build.

| Puncture comparison | H RMS | Conformal momentum RMS |
| --- | ---: | ---: |
| Cut sphere, two steps | 0.000235102 | 0.000332594 |
| Atlas, two steps, combined core/shell | 0.000133581 | 0.000117419 |
| Cut sphere, `t=0.02` | 0.155226 | 0.187453 |
| Atlas, `t=0.02`, combined core/shell | 0.0749575 | 0.0570536 |
| Atlas, `t=0.2`, cubic, shell only | 1.400397 | 1.393406 |
| Atlas, `t=0.2`, quartic, shell only | 1.400395 | 1.393405 |
| Atlas, `t=0.2`, half-step cubic, shell only | 1.398865 | 1.392518 |

These are coordinate-volume RMS values, not invariant physical-volume norms.
Combined atlas norms weight squared core/shell norms by their coordinate
volumes. The sampled domains/volumes differ slightly. Smaller early atlas
norms therefore do not establish convergence or isolate a boundary benefit.
Its smaller minimum lapse also reflects sampling closer to the puncture and
should not be interpreted as evidence of a faster physical collapse.

![Puncture constraints](validation/cubed-sphere-puncture-comparison-20261013/puncture-constraint-comparison.png)

Changing outer ghost degree changes final shell H RMS by only `9.39e-7`
relative; halving the timestep changes it by `0.001094` relative. The large
constraint growth survives both controls. The shell maximum `8.46330` is at
`(0.679432,-0.0566193,-0.169858)`, radius `0.702628`, inside the transition
layer. In the fixed `r>0.9` outer collar, H RMS is `0.00637551` and its maximum
is `0.0124895`. The bulk maximum is at a core corner, radius `0.409937`, also
in the layer; bulk interface-band H RMS is `0.100561`, versus `0.00513663` in
the fixed interior. This localizes the dominant error away from scri at this
time. It does not exclude boundary contamination at other times or establish
which layer/interface terms cause the growth.

A separate bulk/angular factorial control, Aurora job **8924165**, holds
the map, 41 radial cells and gauge fixed, uses the common timestep cap
`0.00025`, and reaches `t=0.1` in every case (400 steps). It exits zero in
2 minutes 21 seconds under the same capacity/one-node/30-minute policy.

| Bulk / shell angular count | Bulk H RMS | Shell H RMS | Bulk M RMS | Shell M RMS |
| --- | ---: | ---: | ---: | ---: |
| B16 / A12 | 0.0426786 | 0.791474 | 0.0194265 | 0.801481 |
| B32 / A12 | 0.0776364 | 0.791498 | 0.0473117 | 0.801525 |
| B16 / A24 | 0.0479037 | 0.791440 | 0.0158947 | 0.821393 |
| B32 / A24 | 0.0204880 | 0.791355 | 0.0168076 | 0.821364 |

Coupled bulk/angular refinement reduces bulk H RMS by 52.0%, while bulk-only
refinement increases it by 81.9%. Shell H changes by less than 0.015%, and its
maximum increases with angular resolution. Thus the core/interface operator
responds to coupled refinement, but shell accuracy is not demonstrated.
The angular grid samples minimum Omega `0.00698510` instead of `0.00712891`
(about 2% smaller); all four actual timestep maxima are `0.00025`. This is
not a perfectly fixed active collar or a three-grid convergence test.

![Independent puncture refinement](validation/cubed-sphere-puncture-comparison-20261013/puncture-refinement.png)

Radial control job **8924173** then holds B16/A12 fixed and varies radial
counts 41/82/123. All three reach `t=0.05` with maximum timestep `0.00008`,
unchanged reference/profile payloads and finite diagnostics. The job exits
zero in 2 minutes 49 seconds under the same capacity policy. There are 625
full steps and a final roundoff-sized step in each case.

| Radial count | Minimum active Omega | Shell H RMS | Shell M RMS | H RMS in fixed `0.375<=r<0.75` | M RMS in that region |
| --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 0.00712891 | 0.372526 | 0.334239 | 0.602959 | 0.537844 |
| 82 | 0.00357085 | 0.173929 | 0.204776 | 0.281058 | 0.329576 |
| 123 | 0.00238199 | 0.0965301 | 0.126280 | 0.155898 | 0.203465 |

The fixed layer region excludes the outer collar. Its H/M RMS fall by
**74.1% / 62.2%** from coarsest to finest, so a significant part of the error
is radial-resolution dependent. This supports improving the coupled spatial
operator before rejecting the formalism. It does not establish its correctness:
the sampled minimum Omega also decreases, angles/bulk remain fixed, and the
finest layer H maximum is still `1.06764`. The core/interface feature near
`r=0.4` barely improves with radial refinement. No fourth-order or fully
coupled convergence result is claimed.

![Radial puncture refinement](validation/cubed-sphere-puncture-comparison-20261013/puncture-radial.png)

No wormhole-to-trumpet transition,
horizon convergence or long-term stability is accepted. The ordinary Z4c
extension's earlier Cartesian bulk AMR/GPU tests remain valid within their
documented scope, but this hyperboloidal research adapter still requires
uniform bulk storage and does not validate hyperboloidal AMR, restart or
multi-rank evolution.

The next steps, in order, are:

1. **Converge the puncture spatial operator before extending its runtime.**
   Prioritize radial resolution through the layer together with balanced
   bulk/angular interface resolution; the separate controls above identify both
   needs. Validate a graded map to avoid spending most cells and timestep cost
   on the outer collar. Keep that active collar fixed in the next coupled
   comparison. Recount H/M/Z in fixed physical regions and at interfaces on
   at least three adequate grids. Require errors
   to decrease systematically, with no larger peak transferred to an interface;
   half-step changes must be below 10% of the finest spatial difference.
   Completion with shell H of order unity fails this gate.
2. **Finish an affordable matched-collar pulse reproduction.** Validate a
   graded radial map and reduce probe overhead, with derivative, interpolation
   and CPU/GPU parity checks before evolution. Require fresh runs through both
   old failure times and `t=2` at comparable minimum Omega within short jobs.
   Then compare closure degrees and independently refined grids. The present
   timeout provides no evidence that the spherical surface cures that failure.
3. **Derive and validate the scri closure alongside the spatial operator.**
   Check characteristic directions, null/shear/Z4 regularity and discrete
   constraint fluxes. A decisive closure must reduce outer-collar errors under
   refinement without importing incoming constraint/gauge modes. Polynomial
   ghost degree by itself has not demonstrated this.
4. **Use native evidence to adjudicate arithmetic and oracle proposals.**
   Recompute the exact binary64 failed-stage operands in higher precision and
   check the raw native RHS of any manufactured slicing throughout its proposed
   interval. Finite readbacks, paired subtraction or an oracle that loses
   admissibility do not explain or repair these native failures. Independently
   check that the raw, unprojected geometric RHS preserves initially vanishing
   vacuum constraints on analytic wormhole jets in fixed positive-Omega regions;
   the constraint time derivatives must tend to zero with differentiation error.
   This gives a boundary-independent consistency gate if coupled errors plateau.
5. **Return to the scientific acceptance tests.** Quantify the nonspherical
   part of the Minkowski disturbance separately from its mainly monopole
   amplitude, and require convergent constraints and curvature over several
   crossings. Then resolve the puncture/trumpet interior, measure convergent
   horizon area and mass, and demonstrate approach to a trumpet while retaining
   the immutable Minkowski hyperboloidal reference. Add AMR to the hyperboloidal
   adapter only with interface/refinement and restart validation.
