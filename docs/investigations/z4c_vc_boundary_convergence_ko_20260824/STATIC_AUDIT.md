# Static audit

## Authority and scope

This campaign starts from commit
`953f2724c00a2efd2f9fad91ae9a784639954a3b` on
`codex/z4c-vc-figure3-native-authority-20260824`.  The starting tree is
`a03bd3f61b9d766adc7083c87ee701bd6d62becb`; the base worktree was clean.
The new branch is
`codex/z4c-vc-boundary-convergence-ko-scan-20260824`.

No VC restriction, prolongation, derefinement, boundary, gauge, Z4c, or
admissibility source has been changed.  The boundary experiment changes only
the domain and its minimum-resolution layout.  The KO experiment uses the old
domain and changes only `z4c/diss`.

## Production constraint measure

The Cartoon history implementation already uses the physical axisymmetric ring
measure.  For native VC storage it is

```text
2*pi*rho*dx1*dx2*w1*w2*sqrt(det(gamma))
```

with local nodal trapezoidal endpoint weights.  The suppressed direction does
not contribute a fictitious `dx3`.  Shared block-endpoint copies are retained
with half weights so adjacent leaf dual volumes tile.  Thus the reported jump
or convergence flattening is not a collapsed-y normalization artifact.

The history quantities named `C-norm2`, `H-norm2`, `M-norm2`, and `Z-norm2`
are extensive integrals of squared constraint magnitudes.  They are neither
square-rooted norms nor volume-normalized RMS quantities.  The new radial
postprocessor preserves exactly that convention.

Relevant source is in `src/z4c/z4c_history_quadrature.hpp` and
`src/outputs/history.cpp`.  A zero-step reconstruction smoke test reproduces
the in-memory full-domain history sums to relative differences no larger than
`2.21e-8` for the nonzero C/H/Z inventories; M is identically zero.

## Persistent SMR qualification

AthenaK's `<refined_regionN>` blocks seed an adaptive tree but do not by
themselves establish a persistent minimum level.  The ordinary dchi criterion
would be allowed to derefine those blocks.  The large-domain input therefore
uses both:

1. the requested nested box seeds at levels 1, 2, and 3; and
2. the existing `z4c_amr/radius_N_rad` plus `radius_N_reflevel` machinery as a
   conservative persistent minimum-resolution floor.

The radii circumscribe the seeded boxes: `sqrt(2)*64`, `sqrt(2)*32`, and
`sqrt(2)*16`.  This retains the requested resolution everywhere in the old
`[0,16] x [-16,16]` box, but it refines a somewhat larger exterior footprint
than the conceptual square-only layout.  This is an explicit capacity cost,
not a change to dchi: `dchi_max=0.02` and
`dchi_derefine_factor=0.25` remain unchanged.

## Local preflight

A fresh OpenMP pgen-enabled host executable at the branch source performed
zero-step N256 initializations on both domains, using at most 16 threads.

- Rout=16: 32 root leaves.
- Rout=128: 104 leaves, distributed as 24/24/24/32 over physical levels
  0/1/2/3.
- The large tree reports physical ceiling 23 and logical ceiling 26.
- Both initializations completed with valid positive lapse/metric data.
- The 33,153 canonical inner vertices and all 257 axis vertices have identical
  coordinates.
- Twenty-three of 25 evolved variables agree bitwise.  `Gamma_x` and
  `Gamma_y` differ only at `1.94e-15` and `1.58e-15` maximum, respectively;
  all shared-node spreads are zero.

This proves expected deterministic-precision inner initialization equivalence.
It is not an evolution or GPU qualification.

Evidence is under `evidence/local_preflight/` and
`analysis/n256_inner_initial_equivalence.json`.
