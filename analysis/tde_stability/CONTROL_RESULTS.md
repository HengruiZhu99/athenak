# Perturbed trumpet controls, 19 September 2026

**No tested configuration is a validated stability cure.** Exactly zero residuals remain exactly zero in the stage/MPI regressions, but a compact lapse perturbation excites a growing mode. None of these experiments changes the ongoing TDE executable, input, checkpoint, or debug-scaling job. Aurora diagnostic jobs use `debug` only.

The common vacuum control has M=R0=1, horizon coordinate radius1M, lapse pulse amplitude1e-8 at(.75,0,0)M with width.5M, sixth-order spatial derivatives, RK3, background-adapted gauge, cubic residual ghosts and characteristic zero-rate outer conditions. Unless a row says otherwise, the box is +/-2M, dx=.25M, kappa1=.1, kappa2=0, KO coefficient.5, with no interior treatment or outer sponge. Matter feedback is disabled. Comparisons of different box sizes are separate problems, not resolution convergence tests.

## Completed controls

| Change | Outcome | Interpretation |
|---|---|---|
| Original local single16³ block | Invalid active state at675M; first invalid ghost input567.15M | Original growing reference |
| Half timestep, .075→.0375M | Reached300M; gamma200–300=.036026/M versus.035977/M | Growth essentially unchanged; not explained by this timestep |
| KO.8 /1.0 | Reached300M; gamma200–300=.038406/.038561 per M | Early reduction followed by exponential growth |
| kappa1=0 /.02 | Invalid active states564/638.025M | Weaker damping does not cure it |
| kappa1=.5 | Invalid active state706.05M | Stronger damping delays failure only |
| Harmonic lapse | Walltime stop226.5M; gamma100–200=.0311/M | Still growing; not1000M completion |
| Adapted lapse coefficient.25 | Invalid active state671.025M | Slower gauge does not cure it |
| Shift driver2 | Invalid active state701.025M | Avoiding the separate gauge-speed coincidence does not cure the small-box mode |
| Shift damping8 | Walltime stop228.375M; gamma100–200=.0709/M | Worse than reference |
| Analytic background derivatives plus FD residuals | Reached300M; gamma200–300=.047158/M, exteriorThetaL2=1.693e-5 | Initial Hamiltonian seed removed, later growth worsened |
| Wider box+/-4M, dx=.25M, GPU8ranks | Invalid active state507M; invalid ghost logged435–436.05M | Removing initial boundary-stencil overlap does not cure later growth |
| Same wide box, outer sponge width2M/rate1 | Reached1000M; finalThetaMax=.3377, exteriorThetaL2=1.491; negative ghost determinant/NaNs logged887.025–888M | Numerical completion with invalid ghosts and large violations, **not a valid control** |
| Refined small box+/-2M, dx=.125M, GPU8ranks | Invalid active state381M; gamma100–200≈.07055/M; invalid ghosts343.0125–344.025M | Smaller initial error but faster growing mode; this is not the wide-box case |
| Smooth interior residual sponge, radius.7M/rate1 | Reached300M withThetaMax=.230, exteriorL2=.214; invalid ghosts271.05–272.025M | Much worse; gamma100–200=.08014/M |
| Same interior layer/rate5 | Invalid active state244.05M; invalid ghosts214.05–215.025M | Much worse; gamma100–200=.11816/M |
| Weighted conformal residual q=delta-chi/chi-background, with analytic product derivatives | Reached300M with exteriorThetaL2=.0864; invalid ghosts277.05–278.025M; gamma200–300=.07819/M | Worse than both original and analytic-jet controls |

Growth rates here are logarithmic amplitude slopes, not squared-norm slopes. Oscillating controls require an envelope/phase analysis; raw fits and R² are retained in the JSON summaries. Uninstrumented legacy ghost-event times are neighboring merged-log records, not exact globally synchronized event times. Active-cell history bad-metric counts remained zero before these ghost failures. The read-only C2P diagnostic gives exact cycle-start/stage information when enabled.

All three applications in Aurora job8840259 ran in the same1node/8rank allocation. The wide and refined applications exited143 after MPI abort; the sponge application exited0 at the requested simulation target. The PBS wrapper exited0 because it completed the test loop. Application and scheduler exits are distinct.

The interior trials used an isolated, default-off pure-vacuum diagnostic gate: only the smooth additive source `-sigma(r)*residual` on all25 fields, no hard freeze, state projection, fluid repair or exterior source. Zero preservation, actual source values, unchanged exterior RHS, and rate5's0.2M source timestep cap were checked before evolution. Atdx=.25M the layer is only2.8cells in radius and the horizon buffer is smaller than the sixth-order stencil reach. Gauge characteristics also propagate outward inside the horizon, so this is not causally isolated excision. At200M its exteriorTheta peaks move just outside the horizon to r=1.1388M. This implicates the *added layer* in those new modes, without retroactively attributing the original no-sponge instability to interior treatment. No interior prototype has been promoted to the task branch.

The weighted-chi prototype passed20 polynomial product-derivative checks to1.37e-15 and preserved exact zero through three RK stages and the initial pure-matter response. It changes a consistent discretization of the same equations, with no clipping or field reset. Those checks did not predict perturbation stability. All three isolated formulation/interior prototypes are documented in [PROTOTYPE_RESULTS.md](PROTOTYPE_RESULTS.md).

## Mechanism distinguished from outcome

The first volumeTheta injection is delta-alpha times the discrete background Hamiltonian defect, at(.625,.125,-.125)M. The small box's boundary stencil then creates a separate exterior injection on the first stage. In the wider control, a full later oscillation instead has positive volume Hamiltonian work concentrated near the puncture, predominantly through conformal-factor curvature and shear. The initial defect term, directTheta advection, explicit constraint damping, KO and direct boundary correction all removeTheta energy over that later cycle. See [UPSTREAM.md](UPSTREAM.md) for the independent reconstruction, spatial attribution and limitations.

Thus removal of the initial seed is insufficient. The evidence favors further analysis of the coupled puncture discretization and regularity, while preserving nonzero physical response. It does not establish that the production stellar failure has exactly the same eigenmode, or that the boundary is harmless in every field channel. ResettingGamma, clipping small residuals, or globally freezing metric/gauge evolution is not supported by these tests.

## Reproduction and records

The task branch contains opt-in source-order and Hamiltonian-balance operators in ed34472a and the read-only ghost-metric diagnostic in8ca2c9b7. These defaults remain off. The isolated analytic-derivative prototype has not been promoted into this branch.

The five local follow-ups (weaker kappa, half timestep, KO) use executable SHA256 `afa84b1c2dde149a3e17a510d879bce99bc2476952e8f711d2f0243ac0cce170`, twoOpenMP threads, and the exact inputs/command-line overrides recorded in `evidence/local-followup-manifest.json`. The source changes are only the default-off diagnostic/operators; the new options remain disabled during these controls. Larger-box and refined GPU controls use the original immutable executable SHA256 `94fd7c3670c4e69e20bedf3c8d24b9b19aaec7e3cd4dad1a5a6de87a9f608130`.

Full inputs, histories, logs and snapshots remain locally under `review/stability-isolation-20260919`, with separate subdirectories for each case. Compact summaries and figures are tracked under `evidence/`. `summarize.py` now records primitive-recovery errors and metric-input diagnostic events separately from active-history finiteness and the stopping reason; reaching a target is never itself a validity check.
