# Independent review of the complete-step mode experiment

The saved evidence supports **two approximate real growing modes of this implemented discrete vacuum map**, with rates 0.046484/M and 0.035734/M. I found no stage/time-count or map-composition error. This is strong numerical evidence, not a rigorous spectral enclosure, a complete spectrum, or a continuum instability result.

I read the isolated hook, map wrapper, calibration, Arnoldi and validation scripts, and the original RK/task/projection implementation. The hook imports the complete residual state including physical ghosts after initialization, refreshes reconstructed ADM state without re-extrapolation or projection, and the normal first-stage CopyU refreshes RK storage. Stationary background, zero matter feedback and the verified fixed timestep make the restricted gravitational map autonomous. The saved pulse and late-direction composition checks are bitwise identical. All 339 call metadata records present at review had correct cycle/time/shape; 40 steps are 3M and 100 steps are 7.5M, not counts of RK substages. The immutable binary hash matches its manifest.

Calibration correctly rejects a peak 1e-6 as cancellation-limited, especially in extrapolated corners. The chosen 1e-3 central difference agrees with 3e-3 and 3e-4 within a few parts per million for both extracted modes. A finite-amplitude central response is not exactly a linear operator, and seed/mode calibration is not exhaustive calibration of every Krylov direction. Independent one-step and uninterrupted nonlinear checks, rather than Arnoldi Ritz values alone, carry the evidence.

The following results were independently recomputed from retained arrays:

| cubic mode | active one-step eigen-residual | active tangent projection defect | active Theta one-step residual | active Rayleigh rate /M |
|---|---:|---:|---:|---:|
| faster | 5.80e-9 | 1.20e-11 | 3.00e-8 | 0.046484001 |
| slower | 4.47e-8 | 6.41e-11 | 2.31e-7 | 0.035734421 |

The tangent calculation enforces tr(delta-g)=0 and tr(delta-A)-A_background:delta-g=0 with the full symmetric tensor contraction, including doubled off-diagonal entries. This checks the correct linearized A constraint, not the incorrect condition tr(delta-A)=0. Full-state tangent defects are 1.0e-7 and 2.2e-7, consistent with finite-difference/cancellation errors in the ghosts. Ghosts account for 99.16% and 98.62% of the full Euclidean norm, so active-only checks are essential; they pass more tightly and show that the extracted growth is not a norm artifact confined to ghosts. Linear-ghost modes also pass these checks, though the slower mode has the larger quoted residual.

The 7.5M uninterrupted tests reproduce predicted shapes/amplitudes to 3.1e-5 and 7.0e-5. Their interval is less than one e-fold, so they are a local tangent-map validation, not a long nonlinear stability test. The half-timestep disabled-damping control preserves both rates and shapes, arguing against this being an RK timestep artifact at the tested resolution.

Two reporting cautions remain important. First-stage volume/KO/boundary/projection projections are Euclidean directional work on an already boundary-dependent global mode; their signs do not prove where the mode originates. Second, the old-mode responses after enabling lapse-scaled damping have 1.5–7% active shape residuals and much larger ghost residuals. Their Rayleigh rates are responses of an old direction under a changed operator, **not new eigenvalues**. A fresh Arnoldi/extracted-mode check is necessary before calling those rates modes of the changed system.

No harness, source or evolution was changed by this review. `review_mode_evidence.py` reproduces the small array checks and writes `mode-peer-review.json` in this directory.
