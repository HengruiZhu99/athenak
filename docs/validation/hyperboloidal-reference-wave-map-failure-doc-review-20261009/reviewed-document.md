# Physical reference wave-map: failed native angular pulses

The private physical-Minkowski reference wave-map gauge fails the prescribed
large angular pulse at N16 and N24. The matched C0 N16 control also fails.
These are original native process failures, with unchanged protected source
identities. The passing [short preflights](hyperboloidal-reference-wave-map-native-audit.md)
and [local principal/manufactured gates](hyperboloidal-wave-map-consistency-audit.md)
remain local evidence; they do not establish evolution stability. No candidate
has been adopted into production.

This checkpoint records three terminated cases from the original nine-case t2
matrix. The other six controls are still running at the checkpoint. Longer t6
and t12 continuations are withheld. No partial result is promoted to a completed
t2 comparison or used to infer a convergence order.

The eventual target remains a substantial angular disturbance on Minkowski,
followed by a single black hole surviving the inner wormhole-to-trumpet
transition with the Minkowski hyperboloidal reference throughout. The globally
harmonic diagnostic core supplies no puncture gauge for that later stage.

## Native failures

All three processes return -6 after the native physical-ADM guard throws. The
reported stage values below are copied from the original stderr, not inferred
from earlier saved arrays. S=1, a=.5 and M=0; time is coordinate time.

| Original case | Abort time | Cycle | Omega at first failing cell | Guard evidence |
|---|---:|---:|---:|---|
| Wave-map N16 large | 1.3671453700579306 | 16910 | .0405078125 | alpha=-1268505.8884 |
| Matched C0 N16 large | 1.9994906249988413 | 24728 | .0026953125 | det(g-tilde)=-.03791297 |
| Wave-map N24 large | .79277343749968521 | 12177 | .002170138889 | chi=-.0071212457 |

The failed N24 wave-map case terminates earlier than N16. Their smallest sampled
Omega values differ nonmonotonically with grid placement, so neither the abort
times nor these grids establish an asymptotic resolution trend. The genuine
half-timestep and N32 controls remain separate pending experiments. None of the
three failures is an imposed timeout or a rewritten target time.

Byte-exact process capsules preserve the input, launch/source/executable
bindings, original output inventory, stderr and history:

- [Wave-map N16](validation/hyperboloidal-reference-wave-map-native-t2-failure-20261009/README.md).
- [Matched C0 N16](validation/hyperboloidal-reference-wave-map-native-t2-c0-N16-failure-20261009/README.md).
- [Wave-map N24](validation/hyperboloidal-reference-wave-map-native-t2-wave-N24-failure-20261009/README.md).

RST/BIN arrays and large console logs remain hash/size/origin metadata in those
capsules. Original local payloads are retained unchanged.

## Saved states before the aborts

The diagnostic observer reads every available binary64 restart, preserves every
guard breach if present, and calls the unchanged native geometry probe only on
admissible fields. All 55+80+32=167 saved arrays are finite, have positive lapse
and chi, SPD conformal spatial metric, and determinant/trace normals within the
original 1e-10 gates. These are projected saved states. They do not describe the
later unrecorded RK stage that aborts.

| Original case | Saved arrays | Last saved time | H RMS | M RMS | Z RMS | Theta RMS |
|---|---:|---:|---:|---:|---:|---:|
| Wave-map N16 large | 55 | 1.350028125 | 2.206429456 | 5.073640740 | 1.753900395 | .251394032 |
| Matched C0 N16 large | 80 | 1.975071094 | 2.443448686 | 2.263636753 | .955292804 | .056518425 |
| Wave-map N24 large | 32 | .775000000 | .173395634 | .394913140 | .099392226 | .007197742 |

These endpoint times differ; the table is not a matched-time gauge comparison.
M and Z norms contract with the Penrose inverse spatial metric
`barGammaInv=chi*gtildeInv`. The reported minimum metric eigenvalue in the
observer is that of g-tilde; the Penrose metric minimum is separately named.

At the last saved snapshots, squared Z-constraint fractions at r>=.9 are
.9097344, .9440420 and .9835844 respectively. For wave-map N24, the squared H,
M and Theta fractions in that same shell are .633883, .765426 and .846903.
These show outer concentration of the measured error, without separating a
continuum pole instability from boundary error, constraint growth, coordinate
distortion or time-integration effects.

![Saved N16 diagnostics](validation/hyperboloidal-reference-wave-map-partial-observations-20261009/partial-plots-N16/attempt001/partial-N16-diagnostics.png)

The plotted curves stop at each last saved state. Dotted lines mark the later
original abort times. Initial geometric constraints vanish at reference
floating-point levels; the pulse initially changes only lapse and shift.

An independent manual-struct parser, cofactor determinant/trace calculation and
field observer reproduces all 167 snapshot identities and all 25 finite field
extrema exactly. The original native history and native probe H/M/Z/Theta RMS
columns differ by at most 4.44e-16. Those constraint columns are two native
diagnostic paths, not an independent differentiated constraint calculation.
The C0 observer's value-only wave-map gauge-pole columns evaluate wave-map
expressions on C0 fields; they are not C0's evolved gauge poles.

All observer receipts retain `partial_diagnostic_only=true` and
`accepted_native_run=false`. Observation protocol success cannot change the
original native failures. Sources, binaries, inputs, compiler dependencies and
output inventories are verified before and after observation.

The [partial-observation archive](validation/hyperboloidal-reference-wave-map-partial-observations-20261009/README.md)
contains exact source, scalar observation JSON, commands, logs, receipts,
independent readbacks and scientific plots. Its collector rehashes all originals
and 2408 protected external identities before and after copying; no live native
run tree is copied. Arrays, compiled payloads and files over 1MiB are excluded.

## A distinct stationary black-hole limitation

Two independent pencil derivations examine changing to the conformal-reference
condition

```
barg^{bc}(Gamma[barg]-Gamma[barghat])^a_bc+2 Zbar^a=0.
```

This changes an algebraic gauge source and removes its explicit off-reference
simple pole when the live conformal inverse metric is bounded. It does not by
itself supply the required Schwarzschild height mass logarithm. The full
four-dimensional relation is

```
Hphysical^a/Omega^2 = Hconformal^a+U^a/Omega,
U^a=(4 barg^{ad}-s barghat^{ad}) Omega_d,
s=barg^{bc} barghat_bc.
```

For stationary, spherical, Killing-aligned exact-Einstein data with a smooth
nondegenerate future end, write reference inertial coordinates
`Y0=T+psi(R)`, `YI=f(R)nI`, and `F=1-2M/R`. The conformal-reference temporal
equation integrates to `R^2 F psi' Omega(f)^4=D`. The assumed smooth end forces
D=0. The spatial equation, even allowing `f=R+c log(R/ell)+d+...`, requires
`c=0,d=-3M`. But `hBH=hhat(f)-psi` must contain `2M log(R/ell)` for a smooth
future hyperboloidal Schwarzschild end. These necessary conditions conflict.

This is a conditional stationary obstruction, not a finite-time instability
theorem or an explanation of the Minkowski pulse failures. Time-dependent
coordinates and separately justified outer sources remain open. It provides
no reason to change the user's Minkowski reference.

Joint preferred-source matching is also pencil-only. Prescribing
`Box_barg Omega=Box_barghat Omega` through the Omega-squared coefficient selects
`f=R-3M+...`. On that same branch, allowing the required height logarithm needs
a compact temporal GH-source correction tending to `+6M/S^2` relative to the
physical-reference base, or `-6M/S^2` relative to the conformal-reference base.
The earlier physical-reference `+2M/S^2` coefficient belongs to a different,
unprojected spatial-harmonic branch and must not be combined with this one.

A formal area-factor source `+/-2(s_area-1)/(r Omega)` displays those limits on
the stated stationary branch and vanishes at the Minkowski reference. It is
not a mass estimator or an admitted gauge: generic angular/off-constraint
states need not have `s_area-1=O(Omega)`. A metric-only preferred spatial source
projection also retains `+2 Zbar^a Omega_a` in the off-constraint Box identity.
Higher stationary coefficients, full nonlinear source closure, boundary jets,
null/shear/trace/Z4/Theta tangencies and puncture behavior remain unresolved.

The [full assessment](validation/hyperboloidal-reference-wave-map-partial-observations-20261009/conformal-gauge-assessment/ASSESSMENT.md),
[joint-source derivation](validation/hyperboloidal-reference-wave-map-partial-observations-20261009/conformal-gauge-assessment/JOINT-PREFERRED-SOURCE.md),
[independent stationary check](validation/hyperboloidal-reference-wave-map-partial-observations-20261009/independent-stationary-assessment/DERIVATION.md)
and [independent joint check](validation/hyperboloidal-reference-wave-map-partial-observations-20261009/independent-joint-source-assessment/DERIVATION.md)
record assumptions and exact equations. No CAS, numerical asymptotic experiment,
kernel query or gauge implementation was used for these assessments.

## Continuing work

The remaining original native controls continue without altered input, source,
boundary or timestep recipes. Separately, the new radial source has passed its
qualified local and angular closure checks; a fresh source-matched volume
operator is being prepared, without reusing the old C0 bulk energy/operator.
An exact-flat coordinate-wave IVP for the actual finite native pulse is also
being derived as an independent continuum comparator. Neither control has yet
established native stability. Production equations are unchanged, so previously
passing production regressions were not repeated for this checkpoint.
