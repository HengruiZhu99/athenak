Investigate and implement a puncture-compatible dynamical propagation and damping scheme for the PC-GH first-order reduction constraints, following the complete standard FO-GH γ₂ construction of Lindblom et al.

Work in:

/Users/hz0693/research/athenak-pcgh-localization-20260902

Start from branch `codex/pc-gh-gamma2-20260904` at or after commit `5811268b`. Preserve all existing run data and unrelated untracked files. Read the PC-GH derivation, implementation, regularization audit, qualification log, symbolic scripts, production code, AMR transfer, boundary, restart, diagnostics, and waveform paths before modifying equations.

Derive the complete γ₁=-1 FO-GH γ₂ system, including the coupled Π and Φ terms—not merely a guessed `-γ₂ R` source—and pull it back exactly into the regular variables

{w, gtilde, K, Atilde, Z, Cperp, rho, beta, p, Q, L, B}.

Prefer a smooth bounded coordinate-time damping rate λ=αγ₂ so the production equations contain no divisions by w, rho, alpha, or chi. Derive the subsidiary evolution of all reduction and curl constraints, including gradients of a spatially varying λ. Verify signs, dimensions, characteristic fields, damping rates, strong hyperbolicity on r>0, and puncture-limit regularity with independent symbolic regression tests.

It is acceptable if uniform strong hyperbolicity cannot be proved at the single compactified point r=0, provided the regular-variable principal system remains finite, no evolved field develops a resolution-growing puncture divergence, the degeneracy is no worse in practice than stable moving-puncture Z4c, and exterior solutions demonstrably converge. State this distinction precisely rather than claiming a theorem at r=0.

If the derivation passes, implement backward-compatible γ₂/λ options, defaulting to zero and remaining distinct from GH gauge damping `kappa`. Keep reduction projection disabled in the primary tests. Instrument every RK stage and AMR operation with reduction/curl maxima, locations, refinement levels, and where practical constraint fluxes.

Qualification order:

1. Exact Minkowski and manufactured compact constraint pulses, independently exciting p, Q, L, and B. Confirm propagation speed, fitted damping rate, stiffness behavior, and convergence.
2. Shifted/gauge-wave tests showing that damping leaves constraint-satisfying solutions unchanged.
3. Isotropic Schwarzschild wormhole-to-trumpet or stationary single-puncture evolutions at three resolutions, with and without fixed AMR. Track puncture powers, GH/ADM/reduction/curl constraints, metric eigenvalues, symmetry, and exterior convergence.
4. An AMR-interface pulse test separating continuum damping from noncommuting transfer injection.
5. Only if those gates show a clear advantage, run the head-on merger on Della using the established large-domain Sommerfeld plus extrapolation setup, reduction projection disabled, frequent restarts, unique scratch directories, and sufficient duration to pass the previous 73.8M failure and reach 100M if stable. Compare against saved Z4c and projected-PC-GH constraints, puncture tracks, symmetry, and waveforms.

Existing evidence to respect: ordinary SMR prolongation injects Q curl; the early hard-Q projection is net corrective, so do not assume projection caused the late instability. The projected long run lost conformal-metric positivity at 73.7999M, while the no-projection γ₂=0 run failed near 4.2246M.

If a correctly derived dynamical γ₂ system fails specifically near the puncture while the exterior and AMR controls behave well, then investigate a hybrid scheme: ordinary dynamical γ₂ outside plus smooth, puncture-centered inner relaxation or projection. Use a symmetry-preserving mask that moves with each puncture, has fixed physical core/taper widths, treats p/Q/L/B consistently, and analyzes curl sources proportional to ∇P×R. Prefer finite relaxation, possibly IMEX or analytically integrated, over hard projection. Require mask-size independence and convergence of projection corrections and exterior waveforms.

Do not use floors, clipping, excessive KO dissipation, weakened diagnostics, or short survival as evidence of success. Commit focused changes and preserve exact commands, inputs, hashes, build configuration, logs, restarts, tables, and plots. Lead the final report with an honest classification: qualified dynamical scheme, partial improvement, failed dynamical scheme, or hybrid fallback required.