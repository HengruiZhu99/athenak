# Corrected non-diagonal FO-GH RHS local audit

Date: 2026-08-17

This bounded local audit was run after correcting the twice-raised
`Atilde^{ij}` construction and the `Atilde^{ik} a_k` / `Atilde^{ik} X_k`
contractions.  It is CPU evidence only and does not qualify long puncture
stability.

Build:

```text
cmake -S . -B /tmp/athenak_fogh_geometry_audit \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=built_in_pgens \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=OFF
cmake --build /tmp/athenak_fogh_geometry_audit -j 8
```

- GCC 13.3.0;
- Kokkos 4.4.0 Serial;
- double precision, MPI off, OpenMP off;
- executable SHA-256
  `b8492dd4d898dade4ebe3d1962b652a8e1aa0071555b8a939da21780b3daef5e`.

Focused results:

```text
non_diagonal_geometry_oracle=PASS
non_diagonal_rhs_oracle=PASS
compatible_gradient_and_robust_advection=PASS
exact_minkowski_uniform_max_error=0
exact_minkowski_smr_max_error=0
robust_minkowski_uniform_final_initial_ratio=0.9538776460544441
robust_minkowski_smr_final_initial_ratio=0.9924378887265064
uniform_wave_errors=1.1837283930837828e-10,7.8606380663851870e-12,5.1512035377972631e-13
uniform_wave_orders=3.912547857635,3.931664973139
smr_wave_errors=4.1373154902046849e-15,1.4482049818613045e-15
smr_wave_order=1.514429155356
puncture_smoke_time=0.02
puncture_history_columns=22
puncture_checkpoint_fields=43
puncture_finite_flag=1
```

The corrected-source lapse-masked `t=0.01M` constraint ladder was:

| N | H L2 | M L2 | GH L2 | reduction L2 |
|---:|---:|---:|---:|---:|
| 16 | 1.0773104515e-2 | 2.1630523571e-4 | 4.9293887842e-5 | 6.6264116981e-3 |
| 24 | 4.9997856521e-3 | 1.6579610873e-4 | 1.8465399989e-5 | 5.3002216281e-3 |
| 32 | 1.9959395230e-3 | 8.3976444148e-5 | 7.7041006544e-6 | 1.3943473947e-3 |

Inputs were the committed files under `tst/inputs/fo_gh_*.athinput`; outputs
were isolated under `/tmp/fogh-corrected-focused.CbBkDk` and were not used as
repository source artifacts.
