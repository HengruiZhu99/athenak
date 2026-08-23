# Ref-GH transition-path and Phi-ordering diagnosis

## Current controlling status

This bounded campaign is complete. The result is:

```text
A VIABLE OPEN-LOOP SCHWARZSCHILD REFERENCE PATH IS ESTABLISHED
```

The qualified candidate is the fixed-core metric homotopy with
`r_core=0.30M`, `tau_transition=8M`, compatible Phi ordering, and controller
off. It reached `t=4M` at `dx_min=M/16, M/24, M/32` without an invalid state,
timestep collapse, nonpositive physical lapse, or loss of relative-metric
admissibility. The primary native errors improve with resolution. This is the
goal's bounded open-loop result, not a claim that the transition has completed:
the smoothstep amplitude is only `0.5` at `t=4M`, and no stability beyond
`t=4M` is established.

The isolated branch is
`codex/ref-gh-transition-path-ordering-diagnosis-20260822`, created exactly
from parent commit `d36c8315e47a949cc188cb01b982db3af8b50737` after verifying that
the parent local, upstream-tracking, and remote refs all matched that SHA.
Controller feedback remains disabled and `delta_q=delta_p=0` throughout the
primary diagnosis.

## Phase 0 diagnostic correction

The diagnostic-only dynamic Ricci path now contracts
`ReferenceRiemann` as `R^C_ACB` from the same cached Cartan tensor used by the
production source. It no longer calls `ProviderRiemann`, which returns zero for
generic nonstationary providers. No production-source arithmetic changed.

Opt-in `ref_gh/max_location_diagnostics=true` records a compact TSV row for
the maximum value and location of reference Ricci/Riemann, spin and spin
derivative, Psi/Q/Delta/Pi/Phi, GH/reduction/curl constraints, all five source
sectors, and `gamma_ij beta^i beta^j/alpha^2`. Each row records time, cycle,
radius, `r/r_core`, refinement level, rank, MeshBlock gid, Cartesian location,
and active-cell indices. Duplicate calls at the same time/cycle are suppressed.

A current-source local one-cycle smoke produced 18 unique records at `t=0`
and 18 at `t=0.0460447286M`. In particular, dynamic reference Ricci is now
nonzero where expected (`0.4923076` on the deliberately coarse smoke grid),
directly discriminating the repaired contraction from the old zero result.

## Phase 1 reference-only scan

The CPU scan sampled 32,769 radii for every combination of three paths and the
nine required times. It used the analytic reference two-jets, completed Cartan
geometry, and the production covariant lower-order source evaluated at
`Psi=eta, Pi=Phi=0`.

The spatial blend derivatives fit the prescribed shrinking-core scalings:

| quantity | fitted log slope versus `r_core` | expected |
|---|---:|---:|
| `max |dB/dr|` | `-0.9999999446` | `-1` |
| `max |d2B/dr2|` | `-1.9999999764` | `-2` |

The rapidly growing reference geometry is localized to the moving shell. At
`t=1.6M`, the maxima and their shrinking/fixed-core ratios are:

| quantity | shrinking width | fixed core | ratio |
|---|---:|---:|---:|
| reference Ricci norm | `3.6329e3` | `7.4354` | `488.6` |
| reference Riemann norm | `3.7845e3` | `10.9648` | `345.1` |
| spin-derivative norm | `2.7591e3` | `9.7217` | `283.8` |
| matched-state source norm | `7.2659e3` | `14.8708` | `488.6` |

The corresponding maximum radii are approximately
`r/r_core=1.19--1.20`, while the spin maximum is at `r/r_core=1.45`.
This is strong reference-only evidence for candidate C (shrinking transition
width), but it does not establish evolution stability by itself.

## Local validation and grid

The current-source Kokkos-Serial Debug build completed. Existing flat,
nonflat, and dynamic-spatial source oracles passed with maximum errors
`6.94e-17`, `3.33e-16`, and `4.996e-16`; exact Minkowski remained zero.
The medium input reproduces the required 34-block tree: a `3^3` root, 26 level-0
blocks plus eight level-1 blocks, finest coverage `[-2M,2M]^3`, and
`dx_min=M/24` for root/MB cells `144/48`.

## Medium Aurora path discriminator

The source-qualified executable was built on Aurora job `8776204` from commit
`dc7a4082468d5fbbd5d7995ee3ffaa3e43807441` with Kokkos SYCL/PVC and GPU-aware
MPI. Eight distinct PVC tiles were recorded. The 34-block tree matched the local
audit. Controller feedback, `delta_q`, and `delta_p` remained zero.

The frozen-wormhole case passed `t=2.2M`, restarted from a real checkpoint, and
reached `t=4M` with `bad_state=0`. Its final normalized native norms were:

| path/time | GH L2 | reduction L2 | curl L2 | physical lapse min |
|---|---:|---:|---:|---:|
| frozen, `t=4M` | `2.1665e-4` | `5.1274e-4` | `1.9415e-4` | `4.5307e-3` |

This rules out a failure of the frozen-wormhole physical GH evolution near the
old `t=1.62--1.65M` window.

The original-rate fixed-core case reached `t=2.2M`, restarted, then failed at
RK stage time `3.18655M`; its last history sample was `t=3.151955M`. The failure
was the existing fail-closed relative-metric conditioning guard, not a nonfinite
value silently accepted by the wrapper. At the last location sample:

| diagnostic | maximum | `r/r_core` |
|---|---:|---:|
| GH constraint | `2.943e1` | `2.016` |
| reduction constraint | `2.785e-1` | `1.514` |
| curl constraint | `2.372e1` | `1.600` |
| `Pi` | `5.944e1` | `1.564` |
| `Phi` | `5.530e1` | `1.600` |
| source `QQ` | `1.283e3` | `1.739` |
| source `DeltaDelta` | `2.724e3` | `2.016` |

The fixed-width moving-core case was run independently on job `8776416`, with
all 12 PVC tiles mapped distinctly. It failed earlier from an invalid effective
timestep. Its last history sample was `t=1.901478M`; max curl was `1.323e2`, max
`Pi` was `4.409e3`, and max `Phi` was `1.787e2`. Reference Ricci peaked at
`8.680e2` near `r/r_core=1.282`, while the evolved-field maxima were farther out
near `r/r_core=2.7--3.0`. Thus holding only the width fixed does not remove the
path failure.

The first capacity wrapper used `set -e` inside a function called from `if !`.
Bash suppresses `errexit` in that context, so it incorrectly advanced after the
fixed-core continuation returned nonzero. The job was cancelled as soon as this
was detected, and every validator now uses explicit `|| return 1`. The separate
12-tile fixed-width result above is the authoritative Phase-4 result.

## Rate ablation

Because both original-rate moving-width and fixed-core paths failed, the
bounded rate scan tested only `tau_transition=8M` and `16M` on the fixed-core
path at medium resolution. Both runs passed a real `t=2.2M` checkpoint/restart
and reached `t=4M` on 12 distinct PVC tiles:

| case at `t=4M` | amplitude | GH L2 | reduction L2 | curl L2 | condition max |
|---|---:|---:|---:|---:|---:|
| fixed core, tau 8M | `0.5` | `3.0641e-3` | `8.4551e-4` | `1.4093e-2` | `3.7446` |
| fixed core, tau 16M | `0.103515625` | `2.2666e-4` | `4.5318e-4` | `6.4200e-4` | `1.9557` |

This establishes a rate threshold for the bounded `t=4M` test. It does not
show that arbitrary slowing is robust, and the tau-16 run has traversed only
about ten percent of the prescribed homotopy. Tau 8M was promoted because it
made the larger transition while remaining admissible.

## Standard Phi-ordering derivation and tests

The runtime option is:

```text
ref_gh/phi_ordering = compatible | standard
```

The default remains `compatible`, in a separate device kernel whose arithmetic
is unchanged. With `Phi_I = E_I(Psi)` and
`[E_I,E_J] = c^K_IJ E_K`, define the non-coordinate-frame curl constraint

```text
C_IJ = E_I(Phi_J) - E_J(Phi_I) - c^K_IJ Phi_K.
```

The compatible update contains `beta^J E_I(Phi_J)`. Algebraically,

```text
(d_t Phi_I)_standard = (d_t Phi_I)_compatible - beta^J C_IJ,
```

so its principal part is exactly

```text
d_t Phi_I - beta^J D_J Phi_I + alpha D_I Pi ~= 0.
```

The structure-coefficient term is part of `C_IJ`; it was not guessed or omitted.
The implementation and device test share the same point-local helper.

Local current-source evidence:

| test | result |
|---|---:|
| compatible-to-standard algebra maximum error | `6.9389e-16` |
| flat/nonflat/dynamic source oracles | unchanged, pass |
| compatible exact Minkowski | zero error |
| standard exact Minkowski | zero error |
| compatible vs standard smooth-wave error, `nx=16` | identical |
| standard wave order, `8->16` | `3.9134` |
| standard wave order, `16->32` | `3.9456` |

Aurora job `8776594` rebuilt this implementation with Kokkos SYCL/PVC, mapped
12 ranks to 12 distinct PVC tiles, and passed the device source/algebra unit and
exact standard-ordering Minkowski tests. The original shrinking-width path with
standard ordering nevertheless failed at RK stage time `1.64623M`; the last
history row was `t=1.603305M`. Its normalized GH, reduction, and curl norms were
`0.41993`, `0.019631`, and `1.25048`. The smoothstep amplitude was `0.31887`,
the relative shift-speed ratio was `0.0674`, and controller state remained
zero. This agrees with the compatible-ordering failure window and rules out
Phi ordering D as the primary cause. Compatible ordering is sufficient for the
promoted fixed-core tau-8 path.

## Medium-resolution decision matrix

| case | ordering | last time | result |
|---|---|---:|---|
| frozen wormhole | compatible | `4.0M` | pass |
| fixed core, tau 4M | compatible | `3.15195M` | fail at RK stage `3.18655M` |
| fixed-width moving core, tau 4M | compatible | `1.90148M` | invalid effective timestep |
| shrinking width, tau 4M | standard | `1.60330M` | fail at RK stage `1.64623M` |
| fixed core, tau 8M | compatible | `4.0M` | pass and promoted |
| fixed core, tau 16M | compatible | `4.0M` | pass, only 10.35 percent transitioned |

The evidence implicates a combination of the prescribed interpolation and its
rate. Width collapse is harmful but not the sole cause: fixing only the width
still fails, while fixing the core at the original rate also eventually fails.
Standard Phi ordering does not rescue the original path.

## Three-resolution open-loop gate

Aurora jobs `8776640`, `8776641`, and `8776642` ran the single promoted
candidate on the identical 34-MeshBlock SMR tree using all 12 PVC tiles. The
coarse job used the debug queue; medium and fine used capacity. All three used
source `6884b094b07646b9206fdd23c56389b08a7122bd`, executable SHA-256
`07f336a5dca87d40088e2256715afca332abd226e09dfe1783d8160a4927a4e5`, and
Kokkos submodule `6739bc623081648af9e752b616d9671527922cbf`. Each run passed an
actual restart from `t=2.2M` and reached exactly `t=4M`.

Final normalized errors and observed pairwise orders are:

| metric at `t=4M` | M/16 | M/24 | M/32 | order 16->24 | order 24->32 |
|---|---:|---:|---:|---:|---:|
| GH L2 | `1.40885e-2` | `3.06406e-3` | `1.00004e-3` | `3.763` | `3.892` |
| reduction L2 | `3.17406e-3` | `8.45509e-4` | `3.87666e-4` | `3.262` | `2.711` |
| curl L2 | `4.85626e-2` | `1.40931e-2` | `4.73797e-3` | `3.051` | `3.789` |
| Psi-reference L2 | `2.33243e-1` | `1.74655e-1` | `1.31418e-1` | `0.713` | `0.989` |
| common ADM momentum L2 | `3.20385e-2` | `9.45671e-3` | `3.59953e-3` | `3.009` | `3.358` |

All three had `bad_state=0`, `feedback=0`, `delta_q=delta_p=0`, positive
physical lapse minima (`9.5403e-3`, `4.5307e-3`, `2.6365e-3`), and bounded
relative-metric condition maxima (`3.7756`, `3.7446`, `3.7329`). No
resolution-dependent earlier failure occurred.

There is an important scientific limitation. The common ADM Hamiltonian L2 in
the `2M<=r<4M` shell is resolution-reversed after about `t=2M`; at `t=4M` it is
`7.361e-3`, `1.350e-2`, `1.900e-2`. The `4M<=r<8M` shell shows the same trend.
This outgoing feature is outside the fixed transition shell (`0.30M<r<0.60M`),
while the global common Hamiltonian L2 still decreases
`1.216 -> 1.047 -> 0.826`. It is retained as a limitation, not hidden or used
to claim physical/dynamic-regularization qualification. The bounded goal's
primary transition and native-constraint gate passes; a longer campaign must
resolve the outward Hamiltonian feature before any production claim.

## Feedback smoke disposition

No feedback run was launched. For the surviving fixed-core path,
`r_full=(1+kappa_core)r_core=0.60M`, while the immutable fitting shell starts at
`r_fit_min=0.15M`. The implemented legal-activation condition requires

```text
r_full + controller_fit_buffer_cells*dx_min < r_fit_min,
```

which is impossible at every tested resolution. Moving the fitting shell to
force activation is explicitly forbidden by this goal. Phase 9 is therefore
structurally unavailable for this survivor; this does not weaken the completed
open-loop result, and no closed-loop result is claimed.

## Reproduction, provenance, and handoff

The exact launchers are `scripts/ref_gh/aurora_transition_path_medium.pbs`,
`aurora_fixed_width_debug12.pbs`, `aurora_fixed_core_rate12.pbs`,
`aurora_standard_ordering_debug12.pbs`, and
`aurora_tau8_resolution_gate12.pbs`. Compact history, max-location, mapping,
mesh, build, command, status, and hash evidence is under
`docs/fo_gh_artifacts/ref_gh_transition_path_ordering_diagnosis_20260822/`.
The tables are reproducible with `analyze_transition_path_medium.py` and
`analyze_transition_resolution_gate.py`.

Large restart/output data remain intentionally uncommitted at:

```text
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_transition_path_20260823T0225Z_dc7a408/
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_standard_ordering_20260823T0525Z_6884b09/
```

At completion, `qstat -u hzhu` reported no jobs. The branch is
`codex/ref-gh-transition-path-ordering-diagnosis-20260822`; final local/upstream
SHA equality is recorded in the final push handoff because a commit cannot
self-record its own SHA.
