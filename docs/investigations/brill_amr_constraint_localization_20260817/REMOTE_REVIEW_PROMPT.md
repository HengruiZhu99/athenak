# Read-only review request: Brill cycle-1722 AMR seam constraint jump

Repository: https://github.com/HengruiZhu99/athenak

Branch: `codex/brill-amr-coarse-cache-coherence-20260817`

Diagnostic source commit: `55f9147bc80d574636c47bcd1dac86178d921988`

Investigation directory:
`docs/investigations/brill_amr_constraint_localization_20260817/`

Please perform a skeptical, source-grounded, **read-only** mathematical and code
review. Start with:

1. `REPORT.md`
2. `verdict.json`
3. `evidence_manifest.json` and `SHA256SUMS`
4. `existing_byte_analysis/existing_byte_summary.json`
5. `existing_byte_analysis/constraint_jump_by_origin_and_stencil.csv`
6. `existing_byte_analysis/parent_region_native_constraint_comparison.csv`
7. `same_level_seam_analysis/same_level_seam_summary.json`
8. `same_level_seam_analysis/same_level_derivative_comparison.csv`
9. `derivative_order_analysis/derivative_order_summary.json`

## Established evidence

The exact event is N256 Brill `A=-0.047`, cycle 1722, `t=9.50625 M`,
74 to 86 MeshBlocks, old max level 2 to new max level 3, O6/RK4/CFL 0.15,
KO 0.02, `dchi_max=0.01`, high-order AMR transfer, no constraint damping,
no chi floor. The restart SHA-256 is
`83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea`.

- T3_06, after all ghost/BC work but before projection, already has proper
  `C=87.7346940100`, `H^2=12.8143694748`, `M^2=62.1172137184` versus T0
  `14.1509665303`, `0.5622987772`, `0.7757016837`.
- Projection changes the final integrals by only 0.56--0.66%; T4 and T5 are
  byte-identical.
- All 25 active evolved fields are byte-identical from T2 through T3_06.
- Same-level seam stencils account for 62.220% of the proper C jump, 96.586%
  of H^2, and 55.290% of M^2. ACTIVE_ONLY accounts for 34.935%, 3.316%, and
  41.296%; coarse-fine/axis regions are small.
- Old explicit parents 28 and 45, immediately below/above `z=0`, each explain
  about half the global C jump after refinement. Balance parents are negligible.
- For 12 worst seam cells, all 2,730 same-level receiver ghost values are
  bitwise equal to sender-active values. Production local O6 D1/D2/mixed
  derivatives exactly equal derivatives stitched from neighbor active bytes.
- The hotspot is near `rho=5.1, z=0`, not the Cartoon symmetry axis `rho=0`.
- Production C++ O2/O4/O6 shadow results on the exact same state are in the
  derivative-order tables. Treat them as spectral-sensitivity evidence, not as
  proof of an O6 implementation bug.
- The Cartoon diagnostic uses the proper ring measure
  `2*pi*rho*sqrt(gamma)*drho*dz`; this is not a collapsed-y normalization jump.

The required disposition is
`same_level_seam_derivative_dominates`. This localizes the immediate constraint
injection but does not yet isolate the upstream construction mechanism.

## Questions

1. Trace the parent-to-child high-order transfer and block/topology code for
   parents 28 and 45. Is there a concrete reason independently constructed
   child samples on opposite sides of `z=0` can be value-consistent yet have
   large O6 derivative incompatibility at their new same-level seam?
2. Does any interpolation orientation, one-sided block-edge closure, missing
   stitched parent neighborhood, or parent-cache provenance differ across this
   seam despite exact final ghost bytes?
3. Can the evidence distinguish P5 transfer smoothness from a genuinely
   under-resolved parent state? If not, specify the smallest **zero-PDE**
   diagnostic that does.
4. Audit whether the exact O6 provenance mask is complete for all pure, mixed,
   and Cartoon suppressed-direction reads. Identify any classification error
   that could materially change the Case-B disposition.
5. Suggest the smallest source-level correction only if the existing bytes and
   source prove a specific invariant violation. Otherwise recommend one bounded
   diagnostic, with exact acceptance/rejection criteria.
6. Check the report's qualification boundary, especially the statement that the
   existing 3D coarse-cache unit is not a full mixed-level 3D integration test.

## Prohibitions

Do not propose chi floors/clipping, weaker finite-positive gates, broad gauge or
parameter sweeps, a global limited-O2 substitution, production promotion of P8,
or a long evolution before a concrete mechanism is isolated. Do not claim
convergence, Figure-3 reproduction, physical critical behavior, or a source bug
from correlation alone. Keep observations, deductions, hypotheses, and missing
evidence separate.

Please return: (a) whether the primary disposition is supported; (b) the most
likely remaining mechanism with precise source anchors; (c) counter-hypotheses;
and (d) one smallest decisive next diagnostic or narrowly justified correction.
