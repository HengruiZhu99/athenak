# Ref-GH transition-path and Phi-ordering diagnosis

## Current controlling status

This campaign is in progress. Phase 0 and the reference-only Phase 1 scan are
complete; no medium-resolution Aurora evolution result is claimed yet.

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
width), but it does not establish evolution stability. Frozen-wormhole and
fixed-core medium-resolution evolutions remain the required discriminator.

## Local validation and grid

The current-source Kokkos-Serial Debug build completed. Existing flat,
nonflat, and dynamic-spatial source oracles passed with maximum errors
`6.94e-17`, `3.33e-16`, and `4.996e-16`; exact Minkowski remained zero.
The medium input reproduces the required 34-block tree: a `3^3` root, 26 level-0
blocks plus eight level-1 blocks, finest coverage `[-2M,2M]^3`, and
`dx_min=M/24` for root/MB cells `144/48`.

## Pending decision sequence

The next source-qualified Aurora job runs only the cheapest medium-resolution
sequence: frozen wormhole to `2.2M` then `4M`; fixed core only if frozen passes;
and fixed-width moving core only if fixed core passes. Standard Phi ordering,
rate ablations, a three-resolution gate, and feedback remain conditional on
those results and are not yet implemented or claimed.
