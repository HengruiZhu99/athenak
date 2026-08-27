# Static q-controlled trumpet, t=0.1M

All primary values use the complete puncture-stencil exclusion mask.
Regions are r=[0.125,0.25), [0.25,0.375), and [0.375,0.5) M.

| q | h/M | metric r1 | constraint r1 | lapse r1 | shift r1 | q_est-q_T |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 1/16 | 1.556801e-03 | 1.567032e-02 | 6.444606e-07 | 9.893657e-07 | +1.242427e-05 |
| 0.9 | 1/24 | 3.621081e-04 | 4.100768e-03 | 1.758973e-07 | 9.510215e-08 | +1.263384e-05 |
| 0.9 | 1/32 | 1.232180e-04 | 1.460406e-03 | 3.850384e-08 | 1.442096e-08 | +1.073770e-05 |
| 1.0 | 1/16 | 3.765876e-13 | 5.519238e-14 | 1.526557e-15 | 6.938894e-17 | +0.000000e+00 |
| 1.0 | 1/24 | 5.755396e-13 | 9.277162e-14 | 2.164935e-15 | 1.040834e-16 | -2.220446e-16 |
| 1.0 | 1/32 | 7.958079e-13 | 1.108995e-13 | 2.581269e-15 | 1.370432e-16 | +0.000000e+00 |
| 1.1 | 1/16 | 2.013644e-03 | 1.560934e-02 | 7.927454e-07 | 1.013912e-06 | -2.010482e-05 |
| 1.1 | 1/24 | 4.811541e-04 | 4.081915e-03 | 2.154767e-07 | 9.941538e-08 | -2.336683e-05 |
| 1.1 | 1/32 | 1.660661e-04 | 1.452856e-03 | 6.971021e-08 | 2.052136e-08 | -2.419247e-05 |

| q | quantity | p(16,24) | p(24,32) | monotone |
|---:|:---|---:|---:|:---:|
| 0.9 | physical_metric_region1 | 3.597 | 3.747 | yes |
| 0.9 | constraint_region1 | 3.306 | 3.589 | yes |
| 0.9 | physical_lapse_region1 | 3.203 | 5.281 | yes |
| 0.9 | physical_shift_region1 | 5.776 | 6.557 | yes |
| 1.0 | physical_metric_region1 | n/a | n/a | n/a |
| 1.0 | constraint_region1 | n/a | n/a | n/a |
| 1.0 | physical_lapse_region1 | n/a | n/a | n/a |
| 1.0 | physical_shift_region1 | n/a | n/a | n/a |
| 1.1 | physical_metric_region1 | 3.531 | 3.698 | yes |
| 1.1 | constraint_region1 | 3.308 | 3.591 | yes |
| 1.1 | physical_lapse_region1 | 3.213 | 3.923 | yes |
| 1.1 | physical_shift_region1 | 5.727 | 5.485 | yes |

The innermost physical-metric Linf is not monotone for either static
mismatch: q=0.9 gives 4.840438e-2, 6.566646e-3, 1.120053e-2 and
q=1.1 gives 6.657521e-2, 1.294006e-2, 1.559001e-2.  This negative
result is retained; no additional post-hoc puncture mask was applied.
