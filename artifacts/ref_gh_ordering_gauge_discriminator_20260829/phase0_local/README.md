# Phase 0 local frozen gates

Date: 2026-08-29 (America/New_York)

This directory records the fresh local part of the ordering/gamma2/gauge
discriminator's frozen regression gate.  The production `src/` tree was
byte-identical to `a09caf707f88d9fb6ca71f9abf62c9302fde3bac` when these
checks ran.  No tolerances were changed.

Two independent SymPy 1.14 regenerations produced byte-identical geometry,
moving-gauge, and contracted-source headers.  Each was also byte-identical to
the committed header.  Their scratch locations and timings are recorded in
`provenance.txt`.

The fresh Release/Serial source-unit executable passed the coefficient,
expanded-radial, generated-geometry, mixed gauge/dtTheta, compact-boundary,
and compatible/standard all-61 RHS oracles.  The complete transcript is
`source_unit.log`; configure and build transcripts are retained alongside it.

This local result is not an Aurora PVC qualification and contains no
long-time stability evidence.
