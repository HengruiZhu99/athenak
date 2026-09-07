# Restart layout and continuation checkpoint

New PC-GH restart headers carry `restart_layout=legacy_pcgh55`,
`restart_layout_version=1`, `restart_layout_fields=55`. This identifies the
current collision-lineage storage ABI, including the legacy and advective
reduction extensions that share those slots. `intrinsic_pcgh50`, version 1,
50 fields is reserved for `formulation=intrinsic_clean`. The intrinsic mesh
mode is still rejected before allocation; no 50-field restart is produced yet.
The writer describes the actual supported allocation, not user-supplied labels.

`main.cpp` captures and validates saved metadata immediately after reading the
restart parameter header, before `-i` or CLI overrides. It subsequently compares
the immutable snapshot with the requested formulation and protected metadata.
Incomplete, inconsistent, unknown and incompatible identities fail before mesh
payload reading. Overrides cannot repair/relabel a corrupt saved identity.
Fresh runs default to `formulation=legacy`; existing equations remain selected
by the existing gauge/reduction controls.

An untagged historical PC-GH file is rejected by default because several
incompatible historical formulations have 55 fields. For a file independently
established to use the collision-lineage layout, the explicit compatibility path is:

    athena -r /verified/legacy.rst pc_gh/restart_untagged_layout=legacy_pcgh55

This declaration is an assertion about established provenance, not an automatic
converter or a proof based on payload size. Do not use it for the unrelated
from-scratch 55-field layout. This deliberately tightens restart input policy;
old untagged files need the declaration, while tagged files do not.

## Restart-continuity defect found and repaired

A real two-step legacy comparison exposed a preexisting restart difference.
With FD6/RK3, KO0.3, both historical projections enabled, and identical dt=1e-4,
one step plus restart disagreed with uninterrupted two-step evolution by
1.2980323e-7 normalized. This exceeded the frozen 2e-12 tolerance.

The signed initialization budget isolates the change: restriction, exchange,
physical-boundary handling and prolongation leave active fields unchanged in
this periodic fixture. The extra initialization algebraic/GH reset changes Q
by up to 0.0013389647. Normal final-stage auxiliary reconstruction sets Q to a
finite difference of g, which need not have exactly zero metric trace. A restart
therefore applied an additional trace-Q correction absent from continuation.

Driver initialization now refreshes boundaries and diagnostics while skipping
that extra PC-GH reset only when loading a restart. Fresh initialization and
normal RK-stage projections are unchanged; regridding calls retain their existing
default behavior. Tagged and explicitly declared untagged resumes are now
bitwise identical to uninterrupted evolution in all 55 fields, including ghosts.
Fresh results before/after this repair are also bitwise identical. This does
not claim that the legacy formulation's physical qualification has improved.

## Tests and reproduction

`check_restart_layout.py` uses the production executable built with the existing
legacy_oracle test adapter. On restart that adapter now leaves the loaded state
alone; it no longer rejects the attempt or refills initial data. It still
requires one uniform block for fresh initialization.

```sh
python analysis/pc_gh_clean_reduction/check_restart_layout.py \
  --binary /absolute/path/to/legacy-oracle-athena \
  --output /absolute/path/to/new-test-directory
```

Nineteen process controls cover actual writer/reader continuity, tagged inspection,
untagged default rejection and explicit declaration, wrong declarations,
wrong names/versions/counts, incomplete metadata, CLI and input-file relabeling,
formulation mismatch, and fresh intrinsic-mode rejection. All pass on Serial.
The compared payload is the sole 55x16x16x1 trailing CC block in this audited
single-block fixture, not a general-purpose restart extractor.

The first attempt used an invalid fixture option and was corrected to the
actual legacy `collision_factorized` target. The second exposed the scientific
continuity failure above. The third passed continuity but exposed missing CLI
registration for the new declaration key; registration now occurs after the
saved-layout snapshot and before CLI parsing. All attempts remain recorded.

Compact controls, same-cell budget summaries, pre-fix field errors, build logs,
source/binary identities and hashed external restart inventories are under
qualification-runs-20260907/pcgh-clean-reduction/restart-layout-001/.
The original failing executable and raw budget are retained externally.

Remaining: MPI/CUDA and multilevel restart regression, tracker-payload controls,
actual intrinsic 50-field allocation/task/output/restart integration, compatible
conversion where defined, and coherent intrinsic transfers. No intrinsic
restart, mesh evolution or physical gate is qualified by this checkpoint.
