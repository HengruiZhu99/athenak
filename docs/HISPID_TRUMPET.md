# Trumpet initial-data consumer

This branch starts from `codex/hispid-pgen` at9495fdb9. The existing TDE and
residual-Z4c branches are unchanged. The native implementation lives on
`HengruiZhu99/TwoPuncturesC:codex/hispid-trumpet`.

The actual checkpoint reader now accepts legacy version1 as QI and version2
with an explicit `seed_family qi` or `seed_family trumpet_r0_m` field following
the parameterization. The linked native header/library must support the new
ABI-safe family entry points. Unknown/missing families and version/payload
mismatches are errors. Native image checks cover the new constructor and
family-query symbols as well as existing sampling symbols.

The pgen creates the matching sampler and verifies its resolved family.
For individual surface guesses, the R0=M trumpet uses coordinate horizon
radius sqrt(m^2-a^2), followed by the same Lorentz contraction of the angular
guess. QI uses half that radius. This guesses a single-seed surface; it does
not supply binary horizon acceptance or prove attenuation enclosure.

`tst/test_suite/z4c/check_trumpet_checkpoint.py` compiles the real C++ reader
with a small standalone sampler probe. The final retained run under
`trumpet-checkpoint-20261005-final/` exactly matches native/Python physical
metric, extrinsic curvature and metric gradients for both formats, and
rejects three malformed metadata cases. The initial loader failure and
intermediate result are retained separately. Executables are local ignored
artifacts; source/image hashes and checkpoint bytes are retained.

Full AthenaK compilation, mesh import and horizon searches for this new seed
family remain pending. No evolution, binary solve, GPU qualification or
performance campaign is claimed by the standalone reader check.
