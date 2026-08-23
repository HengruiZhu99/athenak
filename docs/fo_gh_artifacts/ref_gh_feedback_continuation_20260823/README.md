# Compact Ref-GH feedback-continuation evidence

This directory contains only compact local T0--T2, restart, smoke, and mesh
evidence.  Large field and restart outputs are not committed.

`local/t0` contains source/reference regressions, `local/t1` contains the exact
prescribed-law comparison, `local/t2` contains manufactured controller tests,
`local/restart` contains continuous/split logs and comparison JSON,
`local/postschema` validates the final 32-history-slot schema, and `local/mesh`
contains the authoritative enlarged-domain tree.

`status.json` distinguishes completed local evidence from pending Aurora and
scientific gates.  `SHA256SUMS` hashes every committed compact artifact.
