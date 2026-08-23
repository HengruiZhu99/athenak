# Ref-GH transition-path and Phi-ordering diagnosis

## Current controlling status

This campaign is in progress. Phases 0 and 1 are complete. The medium frozen,
fixed-core, and fixed-width discriminators are complete; the rate scan is
running. No three-resolution open-loop result or feedback result is claimed.

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

## Standard Phi-ordering derivation and local tests

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

This is local implementation evidence only. PVC compilation/execution and the
medium original shrinking-path standard-ordering evolution remain pending.

## Pending decision sequence

Job `8776454` is testing fixed-core `tau_transition=8M` and `16M` independently
through restart-chained `t=4M` gates. After that job completes, the pushed PVC
standard-ordering discriminator will run the original shrinking path to at least
`t=2.2M`. Only a medium candidate that reaches `t=4M` without invalid state and
with controlled constraints may enter the three-resolution gate. Feedback stays
disabled until that gate passes.
