# Final direct-Theta GPU propagation and incoming traces

All three controls reached the **50000M target**, cycle15626: job8842248 primary and lapse-damping0.1, and job8842283 with tenfold smaller pulse amplitude. Each has **783 complete density-independent Theta snapshots** including the final time. Every snapshot passed eight-rank matching headers/time/cycle, complete finite payloads, unique logical blocks, physical bounds and exact64³ coverage; there were no exclusions. Final checkpoint files are `.00050.rst`, not `.00051`. Independent all-rank checkpoint/metric and physical-constraint checks are recorded in `../gpu/results/`.

These are evolved residual Minkowski (M=0), sixth-order uniform64³ controls, eight32³ blocks/eight MPI ranks, dx64M, dt3.2M, G=1, κ1=0, η=0.02, and the smooth radial sponge from512 to1792M, maximum rate0.001/M. There is no BH/horizon, AMR or matter. The two primary runs have a direct Gaussian Theta seed A=1e−6, σ384M; the amplitude control has A=1e−7. Lapse damping is0.01 except in the0.1 ablation. These profiles and amplitudes differ from the separate local compact-support controls.

## Propagation and late spatial distribution

The signed radial means show outward-propagating alternating bands and decaying core oscillations, followed by a small, spatially structured plateau. The final maxima are all at first-active face-near-axis cells: primary and small amplitude at `(32,−32,2016)M`, lapse0.1 at `(32,32,−2016)M`, radius2016.508M. These tiny late peaks do not locate the first injection of an unstable mode.

| Case | Final core Theta RMS | Final ramp RMS | Final outer-plateau RMS | Final max abs Theta |
|---|---:|---:|---:|---:|
| A=1e−6, lapse0.01 |1.50129e−14|1.13616e−13|1.31830e−13|4.58480e−13|
| A=1e−6, lapse0.1 |2.41736e−14|1.14419e−13|1.33145e−13|4.59390e−13|
| A=1e−7, lapse0.01 |6.96525e−16|4.97097e−15|5.41107e−15|2.80345e−14|

These binary-derived RMS values use **coordinate volume**. The core is r≤512M; the ramp is512<r<1792M; the outer plateau is r≥1792M. Outer radial bins cover only the cube's available part of each shell. Binary Theta is float32, accumulated in float64. The separate proper-volume history norms below have different weighting and regions.

Descriptive outer-region log slopes over the actual35008–50000M samples are `+1.2331e−9`, `+3.4241e−7` and `−3.7249e−9 /M`, respectively. These are near-plateau slopes, not measured eigenvalues or asymptotic stability guarantees. In particular the lapse0.1 core is still decaying over this interval. Do not infer growth from the ratio to a much earlier oscillatory minimum.

## Retained initial and acquired incoming data coexist

At the same fixed first-active +x face cell `(2016,32,32)M`, rank7/block7/level0, the Gaussian seeds have nonzero initial incoming state. We reconstruct the actual scalar row

```
C1 = sqrt(chi) Theta + chi Gamma_n/2 + D_n chi_res
```

including all terms and the full-metric normal. Eleven rank7 checkpoints, at approximately5k intervals, give:

| Case | C1 initially | C1 at50000M | Acquired increment |
|---|---:|---:|---:|
| A=1e−6, lapse0.01 |1.02769262e−12|1.54579403e−12|5.18101413e−13|
| A=1e−6, lapse0.1 |1.02769262e−12|1.54592598e−12|5.18233361e−13|
| A=1e−7, lapse0.01 |1.02769262e−13|1.07613509e−13|4.84424731e−15|

The primary's final C1 is `4.58246673e−13 −9.43318884e−14 +1.18187925e−12`. The small-amplitude sum is `2.75814584e−14 −1.11068512e−14 +9.11389020e−14`. The final actual-basis minus fixed-flat-reference differences are only2.10e−22 and2.12e−25, respectively; this does not bound earlier accumulated moving-basis contributions.

The measured total C1 ratio is **0.06961698**, not0.1. The acquired-increment ratio is **0.009349998**, near the quadratic expectation0.01. Keeping the measured initial tail and scaling the primary increment quadratically predicts small-amplitude C1=`1.07950276e−13`; the measured value is only **0.312% lower**. That mixed contribution explains why a nonzero final plateau need not scale linearly with seed amplitude. Two amplitudes cannot separate higher-order nonlinear terms from accumulated arithmetic/projection effects, particularly in the smaller acquired increment.

The local compact controls independently start with exactly zero incoming data but acquire a quadratic trace; see `../compact-theta/trace-comparison.md`. Thus the initial Gaussian tail is **not the complete explanation**. The boundary freezes a weighted RHS rate; time derivatives of its state-dependent coefficients/normal and post-RK projection can alter the nonlinear state function. The explicit identity and amplitude/timestep evidence are in that report. C1 is not automatically an exact nonlinear Riemann invariant. Neither its plateau nor its nonzero value alone proves a growing physical constraint mode.

The first saved post-initial GPU checkpoint is5001.6M. The present dataset brackets, but does not identify, the earlier source operation or RK stage. No reflection coefficient is inferred from Theta profiles; that would require an appropriate incoming/outgoing flux decomposition and controlled incident packet.

## Physical constraints and gauge still have small drifts

The following values come from the independent final histories in `../gpu/results/final-cd-comparison.json`, using exterior **proper volume r>512M**:

| Case | Theta RMS | Hamiltonian RMS | Momentum RMS | Z RMS |
|---|---:|---:|---:|---:|
| A=1e−6, lapse0.01 |1.25828e−13|4.47913e−16|2.05166e−16|4.55223e−14|
| A=1e−6, lapse0.1 |1.26980e−13|4.46369e−16|2.08968e−16|4.67719e−14|
| A=1e−7, lapse0.01 |5.26301e−15|6.01105e−17|1.36233e−17|3.40836e−15|

Exterior Theta RMS changes by factors0.999928,1.002639 and1.000665 over40000–50000M. The previous rapid Theta runaway is absent in these tests, but **not every diagnostic has saturated**. In the small-amplitude case, Hamiltonian RMS rises20.66% during that interval, with descriptive log slope1.8723e−5/M and R²0.999765. The primary Hamiltonian RMS rises0.732%. Those changes remain tiny in absolute size but cannot be dismissed as proven harmless roundoff or extrapolated indefinitely.

The lapse0.01 residual-lapse maximum also rises35.77% in the primary and49.72% in the small-amplitude case, ending at2.634e−12 and1.417e−12. Lapse0.1 behaves differently: its lapse maximum falls by a factor0.1979 to1.063e−12, and its exterior H/M/Z norms decline in that interval, although its exterior Theta plateau is slightly higher. This supports a later **matched lapse-damping 0.1 strong-field comparison**, after the separate 0.01 gate/run establishes meaningful limits. No such matched strong-field comparison is included here; these flat-space results do not validate a curved BH or a star.

## Reproduction and provenance

`plot.py .` regenerates all three radial/RMS figures from the compact NPZ profiles. `plot_gpu_traces.py .` regenerates the incoming-state comparison and its quantitative JSON. Every source rank-file hash remains in each case's binary manifest; the fixed-face JSONs record checkpoint hashes and checker hashes. Raw dumps and checkpoints remain on Aurora. The original23808M collection is preserved under `archive-23808/`.

To reproduce a fixed-face trace, use `checkpoint_face_trace.py RUN OUTPUT --prefix PREFIX --indices 0 5 10 15 20 25 30 35 40 45 50`. C runs are under `long_theta_loweta_8842248/theta_primary` and `/theta_lapse01`, with prefixes `theta_loweta_coremask` and `theta_loweta_lapse01_coremask`; D is `long_theta_amp_8842283/theta_amplitude`, prefix `theta_loweta_amp1e7_coremask`. These state reads supplement the full-cohort validator; they do not replace it.
