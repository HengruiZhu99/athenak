# FO-GH later-source Perlmutter evidence index

This index records the small text evidence retained in the isolated Perlmutter
campaign directory
`/pscratch/sd/h/hzhu/fo-gh-current-20260817.I9kMlW`.  The source was commit
`3ec9c3bd326f22c7dedf792572876f4c2a8683a1`; it predates the subsequently found
twice-raised/symmetric `Atilde^{ij}` RHS correction and is therefore diagnostic
evidence, not qualification of current source.

Provenance:

- allocation `57189778`, node `nid001017`, QOS `gpu_shared_interactive`;
- one NVIDIA A100-SXM4-40GB with CUDA kernels observed in telemetry;
- CUDA 12.9.41, Kokkos 4.7.2 with `SERIAL;CUDA` and `AMPERE80`;
- executable SHA-256
  `92be53e30fd5faef89f8ae8f21d237e450426d40f6b4bb93f8923aec8cfba2c6`;
- superproject archive SHA-256
  `c6d8c62653a9c1f2cfbe612d3d7eeb062f29898ad7b0e796c179450fb6a50f0f`;
- Kokkos archive SHA-256
  `82bfc22185bdf2ba29ce70288282e98ad46176f5770032dd19d41bfa2493e`.

Preflight summary:

```text
robust_uniform_ratio=0.9094251438662729
robust_smr_ratio=0.9833602289681211
uniform_wave_orders=3.912553996421923,3.932015783503866
smr_wave_order=1.607008040548791
regrid_gradient_residual=1.150350259869475e-14
puncture_history_columns=22
puncture_checkpoint_fields=43
puncture_restart_max_abs_difference=0
```

The doubled-domain `[-8M,8M]^3` `N=32,48,64` runs were finite through `5M`.
Their same-spacing `r<2M` histories agree with the `[-4M,4M]^3`
`N=16,24,32` controls to about one percent or better, excluding outer-boundary
arrival as the cause of the central transition through `5M`.  The lapse mask's
included volume jumps at different times on each resolution, so whole masked
near-region norm reversals are not comparisons over a common domain.

At `t=5M`, the doubled-domain exterior `r>=2M` L2 values were:

| N | H | M | GH | reduction+curl |
|---:|---:|---:|---:|---:|
| 32 | 1.932295e-3 | 1.080558e-3 | 1.726396e-3 | 2.876394e-4 |
| 48 | 6.810973e-4 | 6.294860e-4 | 1.738814e-3 | 1.304778e-4 |
| 64 | 6.006968e-4 | 5.172465e-4 | 1.751615e-3 | 8.386673e-5 |

The `48 -> 64` orders are `0.43664`, `0.68264`, `-0.02550`, and `1.53633`.
The exterior reduction family improves, while exterior GH stalls.  This is a
failed convergence gate, not long-puncture qualification.
