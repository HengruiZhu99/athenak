# Exact-pullback FO-GH compact evidence

This directory supports `docs/fo_gh_exact_pullback_formulation.md` and
`docs/fo_gh_exact_pullback_validation.md`. It contains only compact logs and
provenance; large restart, table, and simulation-output files remain excluded.

The controlling result is `FORMULATION NOT ESTABLISHED`: the exact pullback of
the improved gauge driver fails the requested trumpet-regularity gate. The
Perlmutter control campaign was then stopped by request. Coarse reached its
known `dt=0` failure at 3.431611M; medium was cancelled while finite at
2.108827M; fine was not launched.

`local/` records the algebra oracle, direct tests, and diagnostic telemetry
smokes. `perlmutter/` records the corrected-source build, eight-GPU mapping,
mesh trees, and control logs. `invalid_minkowski_8rank.log` is negative
provenance: that launch used a one-MeshBlock input on eight ranks and failed
before evolution, so it is not a passed Minkowski preflight.
