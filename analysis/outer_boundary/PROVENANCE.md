# Exact production baseline for the outer-boundary investigation

The baseline is commit `8b694211` plus the static central refinement floor in
`src/pgen/z4c_tov_ks.cpp`. Every file present in the read-only campaign source
snapshot was compared by SHA-256; only that 21-line addition differs from the
commit. `production-source-manifest.json` records the snapshot bytes.

The campaign is `review/domain2048-20260919` in the review checkout. Its source
snapshot has no `.git` directory: running Git there climbs into the parent
repository and does **not** identify the production revision. The campaign's
`static-floor.patch` compares a different working-tree state and includes an
unrelated removed guard which is already absent from commit `8b694211`.
The byte comparison, not that patch's header, establishes this baseline.

The independent task checkout is `athenak-outer-boundary-fix`, branch
`project/fix-residual-z4c-outer-boundary`. Changes and new tests belong here or
in sibling `outer-boundary-fix-20260920`; campaign artifacts are read-only.
No production restart, monitor update or existing-job modification is authorized.
The latest user override allows dedicated Aurora diagnostics in either debug or
debug-scaling to avoid queues, at most two nodes and one-hour allocations.
The requested tidal allocation is registered by Aurora as MHDTidal (not TidalMHD).

Production uses sixth-order volume differences, RK3, G=1 background-adapted
gauge, lapse residual damping 0.1, kappa1=0.1/kappa2=0, linear residual ghosts,
characteristic_cpbc/zero_rate, and no inner/outer sponge. Later experimental
physical-constraint-radiation code from project/tde is not part of this baseline.
