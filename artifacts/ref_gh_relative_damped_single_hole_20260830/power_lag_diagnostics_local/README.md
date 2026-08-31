# Local paired-power diagnostic gate

This is a cycle-zero Kokkos-Serial implementation check, not an evolution or
scientific result.  It used the controlling power-lag input with overrides to
one uniform `32^3` MeshBlock on `[-1M,1M]^3`, compatible Phi ordering,
`continuation_mode=frozen`, and no restart output.  The grid has `h=M/16` and
the complete FD4+KO stencil exclusion has radius three cells.

All four requested shells were populated after exclusion: inner 264 cells,
blend 3168, outside 5032, and legacy 1960.  At fresh matched data,
`xi=xi_dot=xi_ddot=0`; every paired mean `q_phys-q_ref` was at binary64
roundoff (`5.46e-17` or smaller in absolute value), and every shell was valid.
The physical and reference means are not forced to one another: for example,
the inner mean is `1.31519135693417` and the blend mean is
`1.07951231379451`, documenting the spatial variation of the blended
wormhole profile.

The source-unit gate was also run in Serial and ASan+UBSan/Kokkos-bounds
builds without a sanitizer report.  The common physical/reference pure-power
identity passed at `4.44089e-16`, with its existing `2e-13` tolerance
unchanged.  The pre-existing source-unit result that the naive direct-FD
same-shell estimator does not converge remains explicitly `FAIL`; the
fixed-coordinate estimator remains `PASS`.  No threshold was weakened.

Exact source-unit command:

```text
ASAN_OPTIONS=detect_leaks=0 build-relative-damped-sanitize/src/athena \
  -i tst/inputs/ref_gh_generic_singular_estimator.athinput
```

History hashes:

```text
b39cf8c955499018ca7079e013983f0a1c2b0f272004171b517758ceab77a9d5  frozen_history_smoke.ref_gh.hst
a7898a044c07e915d1d2934cb81a01a5885dfe0a4fd981f28195fb2c11032d7d  frozen_history_smoke.ref_gh_power.hst
0f183e38b8b9c10c63cb668e58e50103eb067947f00377e5b8e688e995193d62  frozen_history_smoke.user.hst
```
