# Matched-driver compact evidence

This directory contains compact, reproducible evidence for the matched
`A*chi`-weighted gauge-driver investigation. Large simulation outputs are not
stored here.

Current status: **FORMULATION NOT ESTABLISHED**. Analytic driver gate V0 and
the finite-radius 58D map pass, but characteristic conditioning on exact
wormhole data grows by more than six orders of magnitude between
\(r=0.5M\) and \(r=0.0625M\). The explicit conditioning stop prevented the
production implementation and candidate numerical ladder.

Contents:

- `v0_matched_driver_audit.txt` and `v0_direct_tests.txt`: exact driver map,
  target, power, conditioning, and high-precision evidence;
- `v1_einstein_map_audit.txt`: independent finite-radius 58D map evidence;
- `v1_hyperbolicity_stop.txt` and `v1_direct_tests.txt`: symmetrizer and
  characteristic-subspace conditioning evidence;
- `perlmutter/`: compact build, rank/GPU, input, history, and first-bad-state
  replay evidence. No restart or bulk field dump is committed.
