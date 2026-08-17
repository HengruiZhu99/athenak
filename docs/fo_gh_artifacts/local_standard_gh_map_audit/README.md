# Local non-diagonal standard-GH map audit

Date: 2026-08-17

This artifact records focused CPU/Serial evidence for the non-diagonal
ADM-to-regular-to-standard-GH conversion fix.  It does not qualify GPU behavior
or long puncture stability.

## Defect and correction

The independent complete-map oracle initially reported nine `Pi_ab` failures.
`RegularToStandardGh` populated `gamma(i,j)` and immediately contracted
`gamma(i,k)` and `gamma(j,k)` into `d0gamma(i,j)`.  For a non-diagonal metric,
some required symmetric components had not yet been initialized.  The
production fix separates reconstruction of all `gamma_ij` and `K_ij` values
from the subsequent `d0gamma` contraction.

The new oracle uses a symmetric positive-definite non-diagonal ADM metric,
nonzero extrinsic curvature, shift, lapse gradient, shift gradient, and metric
gradient.  It independently checks `chi`, `K`, `pi`, `X`, `a`, `B`,
`gtilde`, `Atilde`, `Q`, contracted conformal Christoffels, and every component
of reconstructed `g_ab`, `Phi_iab`, and `Pi_ab`.

## Build provenance

```text
cmake -S . -B /tmp/athenak_fogh_geometry_audit \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=built_in_pgens \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=OFF
cmake --build /tmp/athenak_fogh_geometry_audit -j 8
```

- GCC 13.3.0
- Kokkos 4.4.0 Serial
- MPI and OpenMP disabled
- executable SHA-256:
  `8de429a9a0983dacd034cc7fa7183a5c9ec68b04b7b9aedfb09d31c3cf1368b5`

## Focused results

```text
fo_gh_algebra_unit=PASS
fo_gh_geometry_unit=PASS
fo_gh_rhs_unit=PASS
fo_gh_compatible_unit=PASS
fo_gh_tensor_unit=PASS
puncture_smoke_t0.02=PASS
history_checkpoint_family_norm_agreement=PASS (rtol 2e-14)
puncture_bounded_t0.2=PASS
puncture_constraint_ladder_t0.01=PASS
puncture_restart_two_cycle_bitwise=PASS
```

The corrected `t=0.2M` final masked family norms were:

```text
H=1.0789087558046768e-02
M=1.7204459662178654e-03
GH=9.204779293571142e-04
reduction_plus_curl=6.586416951287929e-03
```

The `t=0.01M` three-grid ladder was:

| N | H L2 | M L2 | GH L2 | reduction+curl L2 |
|---:|---:|---:|---:|---:|
| 16 | 1.0773104514567337e-2 | 8.693651530896672e-5 | 4.929388784165813e-5 | 6.626411698073900e-3 |
| 24 | 4.999785652084058e-3 | 4.986971841631338e-5 | 1.846539998943078e-5 | 5.300221628107450e-3 |
| 32 | 1.995939523047033e-3 | 2.660895102977995e-5 | 7.704100654368556e-6 | 1.394347394714937e-3 |

The `24 -> 32` observed orders were `3.1919964833`, `2.1835434644`,
`3.0385834023`, and `4.6416592335`.  Direct and resumed two-cycle checkpoint
arrays were bit-for-bit equal.

Raw small text outputs from these runs are committed beside this README.  The
approximately 7 MB restart payloads are intentionally omitted; their numerical
effect is represented by the committed direct/resumed checkpoint arrays and
their SHA-256 manifest.
