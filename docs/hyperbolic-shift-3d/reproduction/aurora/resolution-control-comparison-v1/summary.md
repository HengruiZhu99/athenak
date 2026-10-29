# Original-binary Minkowski resolution controls

All times and growth rates use coordinate time. Pending cases have no inferred outcome.

| Case | Run outcome | Science gates | Last diagnostics | Reached / failure time | State rate | Points; fit time range; R² | Median s/step | Active cell-steps/s |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| n16, dt=0.0005 | failed | failed | 0.45 | 0.4895 | 74.492 | 5; 0.15–0.35; 0.99963 | 0.33999 | 2.289e+05 |
| n24, dt=1/3000 | failed | failed | 0.3 | 0.31367 | 116.35 | 6; 0.1–0.225; 0.99999 | 0.79402 | 3.3079e+05 |
| n32, dt=0.00025 | bounded | failed | 0.23225 | 0.23225 | 153.87 | 4; 0.075–0.15; 0.99999 | 1.4874 | 4.1857e+05 |
| n16, dt=0.00025 | failed | failed | 0.475 | 0.486 | 75.217 | 9; 0.15–0.35; 0.99972 | 0.34161 | 2.2781e+05 |

The fit uses the longest contiguous set of positive state deviations in [1e-10, 0.001], with at least three recorded samples. H, M and Z use the same time selection; their individual fits, exact values, standard errors and R² are in analysis.json. These are discrete growth estimates, not convergence orders. Fit windows can differ between grids.

Throughput excludes the first history interval and includes diagnostic/communication overhead. Active cell-steps/s counts all 19 blocks across the node; it is not per GPU. Component timers remain separate in the JSON because closure includes communication and cumulative timers are MPI maxima, not an additive cost partition.

Recorded accepted steps per measured diagnostic interval: n16, dt=0.0005: 100; n24, dt=1/3000: 75; n32, dt=0.00025: 29/100; n16, dt=0.00025: 100. Different diagnostic cadences change overhead per step; these timings do not isolate kernel efficiency. Median coordinate-time advance per wall second is also retained in the JSON.

Plots end at recorded diagnostic times. No final-state value is imputed from stale report diagnostics, and no curve is extended to a later rejection time. Null curves are the recorded running maxima. A bounded checkpoint is not a completed endpoint or a stability admission.

Run outcome and scientific gates are independent: a trajectory can reach its endpoint while failing a state/constraint tolerance, or fail numerically while its null closure remains precise. Recorded worker science checks are reevaluated against the report; missing checks remain unassessed. Completion, configuration and checkpoint checks are retained separately. Passing the recorded science checks alone is not a stability qualification.

H is the recorded evolved-variable Hamiltonian; M and Z are the recorded Penrose-metric covector norms, all normalized by compactified coordinate volume. When tr_g A drifts, this H need not equal the Hamiltonian of fully reconstructed ADM data. Zero/nonfinite values are omitted on logarithmic axes without a floor; gaps remain gaps.

H near the initial residual is sensitive to floating-point cancellation and differentiated roundoff. The JSON reports the minimum fitted value relative to its initial magnitude and flags samples within 100 times that empirical scale, without removing them. This factor is a sensitivity indicator, not a measured error bound. R² does not establish resolved constraint growth; OLS standard errors are descriptive because samples are correlated. Two points always remain insufficient, regardless of their apparent slope.

Failure times alone cannot distinguish truncation error from a changed roundoff seed. These controls cannot identify a unique unstable operator. No assertion that spatial resolution is ruled out is generated automatically.
