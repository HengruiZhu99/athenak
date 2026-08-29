# Phase-0 frozen baseline

This directory preserves the cheap local baseline required before the fully
subtracted gauge repair.  No production RHS source had changed when these
checks ran.

- `configure.log`, `build.log`: fresh Release/Serial build.
- `source_unit.log`: complete source-unit gate, including coefficient,
  expanded radial, geometry, moving gauge/`dtTheta`, and both-ordering all-61
  oracles.
- `standard_gh_source_audit.log`, `binary64_stationary_source_audit.log`, and
  `reference_frame_audit.log`: directly invoked independent Python audits.
- `regen_*`: two deterministic SymPy regenerations and equality status.
- `provenance.txt`: source/build identity and exact commands.

This is local algebra/oracle evidence, not repaired evolution or Aurora PVC
qualification.
