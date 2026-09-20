# Fixed-face incoming-trace comparison

A nonzero initial Gaussian tail explains part of the late incoming trace, but it is **not the complete cause**. The compact seed starts with exactly zero incoming data and later acquires a small, nearly constant trace. Reducing its amplitude tenfold reduces that trace by approximately one hundredfold; halving the timestep changes it by only about 0.06% near 5000M. This supports a quadratic nonlinear contribution, without identifying its precise source operation.

These are fresh local CPU runs on the same uniform 32³ mesh, eight 16³ blocks/eight MPI ranks, dx128M. They use κ1=0, η=0.02, lapse-residual damping0.01, G=1, the 512–1792M radial sponge and the historical zero-rate boundary. All four final eight-rank checkpoint cohorts passed the independent finite-payload and ghost-metric SPD checks in `results.json`. They are not AMR, full-BH, GPU or matter stability tests.

The fixed sample is the first-active +x face cell `(1984,64,64)M`, rank7/block7/level0. Every term uses the actual full conformal metric normal and the same active-only D2 derivative as the boundary code. The JSONs retain checkpoint hashes, all term values, actual floating-point times and cycle numbers.

| Case | End time | Stop | C1 initially | C1 at endpoint |
|---|---:|---|---:|---:|
| Compact A=1e−6, dt6.4 |22540.8M|Walltime; target50000M not reached|0|1.95507905e−13|
| Gaussian A=1e−6, dt6.4 |15648M|Walltime; target50000M not reached|5.40272185e−10|5.41636651e−10|
| Compact A=1e−7, dt6.4 |6000M|Target reached|0|1.98501702e−15|
| Compact A=1e−6, dt3.2 |6000M|Target reached|0|1.95440383e−13|

The compact radius512M and Gaussian σ512M profiles differ in the interior as well as at the boundary. Their comparison is therefore not a controlled alteration of the tail alone.

## Amplitude and timestep comparisons

The two amplitude cases have an exactly matched saved time, 5004.8M. C1 is `1.954578797526e−13` for A=1e−6 and `1.936835095090e−15` for A=1e−7. The ratio is **0.009909219815**, within0.91% of the quadratic prediction0.01 and far from linear scaling0.1. The smaller trace has greater relative sensitivity to absolute arithmetic/projection errors; no scaling law is imposed on the nonlinear evolution.

The smaller-timestep checkpoint is at **5001.6M**, while the reference is at **5004.8M**, an offset of **−3.2M** (exact stored values and their difference are in the JSON). C1 is `1.953409919113e−13` versus `1.954578797526e−13`, ratio **0.999401979386**, a difference of−0.059802%. No time interpolation was used. These two timesteps at slightly different times do not establish a formal convergence order. They do show that the acquired trace is largely unchanged when dt is halved.

The compact reference's C1 is zero initially, `1.95458e−13` by the first saved checkpoint at5004.8M, and stays near `1.955e−13` through22540.8M. The available checkpoint cadence only brackets its acquisition between initialization and5004.8M. The initial CPBC diagnostic also reports exactly zero incoming amplitudes and correction. This does not locate the later injection within that interval or identify a particular RK stage, projection, or spatial stencil.

## Meaning of the nonlinear trace

The measured state function is

```
C1 = sqrt(chi) Theta + chi Gamma_n/2 + n_u^a D_a chi_res .
```

The zero-rate boundary sets the weighted RHS combination

```
R = sqrt(chi) F_Theta + chi n_d,a F_Gamma^a/2
    + n_u^a D_a F_chi
```

to zero. It does not include time derivatives of the coefficients and metric normal. With fixed discrete derivative stencils and an unchanged derivative-component selection, differentiation of the state function gives

```
d_t C1 = R
       + dot(chi) Theta/(2 sqrt(chi))
       + dot(chi) Gamma_n/2
       + chi dot(n_d,a) Gamma^a/2
       + dot(n_u^a) D_a chi_res .
```

The extra terms are quadratic around zero residual on flat background. Algebraic projection/recasting after RK stages can add separate jumps to C1. The amplitude scaling is consistent with these effects; it does not distinguish them. A pointwise characteristic state function is not automatically an exact nonlinear Riemann invariant. Its drift is therefore not, by itself, evidence of a numerical defect or a growing physical constraint norm.

At the compact endpoint, the C1 terms are `7.12827940e−14 + 6.29600728e−15 + 1.17929104e−13 = 1.95507905e−13`. Its actual-basis minus fixed-flat-reference value is only `6.915e−24` at this late time. That small instantaneous difference does not bound the accumulated moving-basis contribution earlier, while the pulse was crossing the boundary.

For Gaussian data, C1 retains its initial value to0.253% at15648M, while its acquired increment is `1.36446622e−12`. Thus an initial tail and a separately acquired nonlinear trace can coexist. Neither a plateau nor a small positive trace alone establishes exponential instability.

## Initial compatibility and default-parity review

`boundary-support-review.json` confirms exact zero for all25 residual fields throughout every physical ghost layer and seven inward active layers at initialization, covering the ng4 volume stencil composed with the D2 boundary derivative. All24 physical block faces are included, including edge/corner ghosts: 2,325,400 values per tested case. This guarantees zero initial trace/derivative data in this fixture, not zero for all subsequent evolution.

`physical-payload-review.json` compares all29,306,880 raw physical payload bytes (MHD, face magnetic fields and Z4c) after three cycles, ignoring changed text/metadata headers. Compact MPI1 versusMPI2, Gaussian explicit versusdefault, and new versusold Gaussian executable payloads are exactly equal. The new assertions are in `tst/regression/z4c_compact_theta.py`; these reviews reused existing checkpoints and launched no duplicate evolutions.

The next discriminating diagnostic is to record the explicit moving-coefficient terms above and C1 immediately before/after projection during the first boundary arrival, alongside physical H/M/Q/Theta. Merely damping or canceling this state trace would not remove the independently established positive linear boundary roots if it only multiplies the same boundary operator by `(lambda+nu)`.
