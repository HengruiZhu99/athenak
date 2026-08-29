# Ref-GH single-puncture robustness: Phase 0 local checkpoint

Date: 2026-08-29 (America/New_York)

This is a compact, source-frozen Phase 0 checkpoint.  It does not establish
long-time robustness, q-controller convergence, or Aurora PVC qualification.

## Provenance

- Branch: `codex/ref-gh-single-puncture-robustness-20260829`
- Starting commit: `53f70903c82f2f8670df7aad6aa14e7ef646ad82`
- Frozen qualified implementation: `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`
- `git diff --exit-code a09caf70 -- src`: exit 0
- Kokkos submodule: `6739bc623081648af9e752b616d9671527922cbf`
- Host: `Kip`, Linux x86_64
- GCC/G++: 13.3.0
- CMake: 3.28.3
- Configuration: Release, Kokkos Serial, MPI off, OpenMP off,
  `REF_GH_SOURCE_UNIT=ON`
- Executable SHA-256:
  `d1e335849b01cfb2b2bf0ddad2e8e5ee3d962023b3d2068018d95ed87458f49b`

The production `src/` tree was not changed.  The campaign input and offline
accepted-state health analyzer live outside `src/`.

## Independent regeneration

Two fresh deterministic SymPy regenerations were executed concurrently.  They
were byte-identical to one another and to the committed generated headers.

| Generated file | SHA-256 |
|---|---|
| `analytic_radial_q_geometry_generated.hpp` | `6b4b3976f8cfc62924aabaf6cb960cd79a09677e8a7dd59441b5d3a4f90184e7` |
| `analytic_radial_q_gauge_generated.hpp` | `f6ec5bb54d10c490f65f78d9d9a9e7df07c7f080eef4d6edd3fff55d45053908` |
| `analytic_radial_q_source_generated.hpp` | `b5dff1d44bd1e0ed53070fdd5fa4b30d0c4c176a13c2d0924dd6b0ad51012f21` |
| generator script | `2c19efa4affd02c0d97ebbe26ea51a3f6bbb23467f3cb3c3e07c134320e26cc3` |

Regeneration A took 321.53 s and regeneration B took 318.20 s.  Their scratch
directories were `/tmp/refgh-longtime-regen-a.CaoQFv` and
`/tmp/refgh-longtime-regen-b.8JOgUu` on the originating host.

## Source-unit/oracle gate

`source_unit.log` is the complete fresh transcript (SHA-256
`1a495c76f9acede64c8b4870f4e7fae1d1eaff1f664dfdf3b77813785dcc54c5`).
All frozen gates passed without tolerance changes.  In particular:

- coefficient oracle: 216 samples, maximum error `8.88178e-15`;
- expanded radial oracle: 2160 samples, conditioned error `1.48837e-13`;
- generated geometry oracle: 2376 samples, error `2.33147e-15`;
- moving mixed-jet gauge oracle: 2160 samples, motion error `1.24829e-14`;
- compact boundary oracle: 2160 samples, metric error `4.56474e-14`;
- all-61 RHS oracle: 4320 samples, error `2.84217e-14`, compatible and
  standard Phi orderings;
- production cache oracle: conditioned scaled Linf `1.63758e-14`;
- exact Minkowski cycle-zero check: maximum error 0.

## Accepted-state telemetry smoke

`health_smoke.json` records a deliberately tiny 16^3, cycle-zero analysis
smoke.  It validates the file plumbing and health calculations; it is not a
physical-resolution result.  Relative spatial `G` was SPD, the relative lapse
was real and finite, all state telemetry was finite, and native GH/reduction/
curl constraints excluding the full puncture-overlap stencil were at roundoff.

The offline q estimate in this deliberately coarse smoke is not a controller
qualification result.  The input had the controller disabled and the sampled
shell (`2h <= r < 8h`) spans a very coarse range.

## Remaining Phase 0 work

The one-tile/eight-tile Aurora PVC smoke remains required.  No Aurora
qualification claim is made by this local checkpoint.
