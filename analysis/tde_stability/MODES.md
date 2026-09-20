# Complete-timestep vacuum mode tests

The Schwarzschild trumpet vacuum control is still perturbatively unstable. Growing modes survive the G=2 gauge, linear ghost extrapolation, lapse-adjusted damping and coupled covariant-source experiments. The strongest tested damping reaches1000M with valid metrics but retains a verified weak growing constraint mode. No candidate is being promoted to the TDE calculation. Dedicated Aurora work uses `debug`; the production job and `debug-scaling` are unchanged.

## What is now established

The isolated probe applies the implemented **complete RK3 timestep**, including volume derivatives, dissipation, characteristic boundary updates, ghost extrapolation and algebraic projection. It imports/exports all 25 residual fields including ghosts after initialization. Ordinary restart initialization is not silently treated as the timestep map: re-extrapolating already projected ghosts can change the operator.

The reference control has a single 16^3 block in [-2,2]^3 M, dx=0.25M, sixth-order spatial derivatives, G=2 background-adapted gauge, kappa1=0.1, kappa2=0, KO=0.5 and zero-rate characteristic RHS closure. There is no inner excision, sponge or matter feedback. Central differences in perturbation amplitude approximate the tangent map; they are not automatic differentiation.

| Configuration | Faster mode gamma /M | Slower mode gamma /M | E-fold times /M |
|---|---:|---:|---:|
| G=2, cubic ghosts | 0.0464840 | 0.0357344 | 21.51, 27.98 |
| G=2, linear ghosts | 0.0455908 | 0.0318548 | 21.93, 31.39 |
| G=2, constant damping product sigma=0.3 | 0.0351450 | 0.0243533 | 28.45, 41.06 |
| G=2, coupled covariant sources, sigma=0.3 | 0.0230762 | 0.0091801 | 43.34, 108.93 |

Here gamma is log(multiplier)/time. The linear and cubic modes have almost identical active shapes. Positive rates in both modes disqualify all four configurations from the vacuum stability gate. The finite Krylov search does not establish the entire spectrum.

Validation includes exact-zero preservation, bitwise-identical composed versus uninterrupted maps, perturbation-amplitude convergence, independent one-step checks, and direct perturbation evolution. Eigen-residuals are about 10^-6; active-cell rates agree independently. Signed physical Hamiltonian, momentum and conformal-connection constraints amplify with the same eigenvalues. This is not a growth artifact confined to a ghost-inclusive norm or to Theta alone. Halving the timestep preserves the baseline rates.

The eigenmodes couple inner constraint amplitudes and outer geometry. The faster mode's Hamiltonian peak is around r=0.545M and momentum peak around r=0.217M; its connection constraint is strongest near an outer face. Fourth corner ghosts acquire large extrapolated amplitudes. Those maxima describe a developed global mode, **not its initial injection site**. The previously identified lapse-times-background-Hamiltonian seed and the much later first invalid outer ghost are distinct observations.

## Lower-order damping experiment

The default-off prototype replaces the damping product alpha*kappa1 by kappa1 consistently in Khat, Theta and Gamma, without an explicit division by lapse. Its sources vanish on the physical constraint surface. It preserves the existing Gamma damping coefficient and leaves physical matter forcing unchanged; it is not all-field sponge damping.

Independent source differences match the analytic changes to 8.94e-23. The default path is bitwise identical to the original executable; exactly zero vacuum remains zero through all RK stages, including ghosts. The initial matter RHS is unchanged. Three matched local lapse-pulse controls reached300M with finite active/ghost metrics and no invalid-state or recovery events, but their late constraint norms grew:

| Damping product | Fitted gamma, 200–300M | Final max abs(Theta) |
|---|---:|---:|
| alpha*0.1 | 0.0357335/M | 2.67e-8 |
| 0.1 | 0.0324707/M | 3.48e-8 |
| 0.3 | 0.0246992/M | 5.90e-10 |

The smaller amplitude with sigma=0.3 is useful evidence of reduced growth, not saturation. The independent mode test above still finds a faster growing mode that this particular lapse pulse excites weakly. Source patches and compact validation evidence are archived as experimental artifacts; production equations are unchanged.

## Coupled covariant-source experiment

A second isolated default-off prototype adds the complete audited covariant geometric constraint terms to Khat, Theta, A and Gamma, together with the published CCZ4 Gamma damping convention. This is a coupled formulation experiment, not a selective negative Theta source. It preserves the physical constraint surface, uses identical full/background evaluations, and leaves gauge and matter forcing unchanged.

The C++ source helper agrees with independent physical-Christoffel calculations to1.74e-18 over72 points. Actual matched late-state volume-RHS differences agree to3.76e-16; the default-off path and initial matter forcing are bitwise unchanged. Exactly zero residual remains exact through RK stages and complete-map composition, including ghosts. The source-only and diagnostic-hook binaries are separately immutable and hashed.

| Coupled-source damping | Outcome of fresh300M control |
|---|---|
| sigma=0.1alpha | First recorded invalid ghost199.7625M; abort228M |
| sigma=0.1 | Reached300M/exit0, but invalid ghosts from298.05M; rejected checkpoint |
| sigma=0.3 | Valid300M checkpoint, but late exterior Theta growth0.017526/M and independently verified growing eigenmodes |
| sigma=1 | Reached1000M with valid active/ghost metrics; constraint growth persists |

The sigma=1 case continued from its validated300M checkpoint and reached the1000M simulation target, not a walltime stop. The final max abs(Theta) is2.1471e-10 and exterior Theta L2 is6.9042e-10. All checkpoint payloads are finite; lapse, chi and all metric principal minors remain positive including every ghost. No invalid-state or fluid-recovery event was logged. Nevertheless, the exterior norm rises2.924 times over750–1000M, with fitted gamma0.0043653/M and log-fit R-squared0.99893. Earlier growth varies by window and has not saturated. See [long comparison](evidence/modes/covariant-sources/long-comparison.png) and [window measurements](evidence/modes/covariant-sources/long-results.json); the dotted line marks the300M restart for sigma=1 and the endpoint of the shorter controls.

A separate32-vector complete-timestep search supports a weak oscillatory growing sigma=1 mode: gamma0.0017866/M, e-fold559.7M, period251.7M. The direct active eigen-residual is8.5e-5; an independent one-step rate is0.0017844/M. Two-amplitude active responses agree to3.2e-9, and signed physical H/M/Q constraints amplify consistently with the same complex eigenvalue. A faster real Ritz candidate near0.0047/M remains unconverged and is not an established mode. The identified oscillatory branch alone rules out a perturbation-stability pass; it does not uniquely explain the late pulse-control rate. Initial sigma=0.3 eigenvector responses under sigma=1 rotate and must not be presented as eigenvalues. See [mode convergence](evidence/modes/covariant-sources/modes/sigma1-mode-convergence.png).

These controls disable matter feedback. Passive fluid density grows to1.19e75 in the1000M run despite remaining finite. This is a vacuum gravitational test, not acceptable atmosphere or stellar behavior; enabling normal matter feedback requires a separate fluid-cleanup test. No residual clipping or resetting is used.

## Independent MPI/GPU domain and resolution controls

Aurora job8840650 completed three independent G=2 controls in `debug`, one node/eight ranks, with17-minute application caps inside a one-hour allocation. The wrapper exits1 because two applications failed; application results are inspected individually.

| Box / spacing | Application result | Relevant late growth |
|---|---|---:|
| +/-2M, dx0.25M | Abort672M; first recovery warning bracket590.025–591M at fourth corner ghost | gamma200–300=0.0357483/M |
| +/-4M, dx0.25M | Abort476.025M; first recovery warning bracket421.013–422.025M at fourth corner ghost | gamma200–300=0.0516419/M; oscillatory norm |
| +/-2M, dx0.125M | Clean walltime stop333.3375M, not1000M | gamma200–300=0.0610559/M |

The refined final cohort has all eight matching rank headers, finite payloads and valid active/ghost metrics. It nevertheless has max abs(Theta)=0.001344 and continuing exponential growth. No failed case is restarted, and the already growing baseline is not extended merely to obtain a1000M completion. The serial eight-block half-timestep comparison stopped cleanly at244.275M and reproduces the earlier baseline growth rate; this is separate from target completion.

## Principal part versus full evolution

The sampled projected sixth-order principal symbol with G=2 has no positive real mode at the tested trumpet radii/phases. Independent Minkowski and stencil checks agree to roughly5e-14, and sampled RK3 gains with upwind advection and KO are at most one. These checks do not prove global stability, especially at a puncture or with boundaries.

Strong hyperbolicity concerns the principal part and an appropriate well-posed initial-value problem. It permits finite exponential bounds and does not imply every perturbation decays. Variable coefficients, lower-order constraint terms, boundary conditions, and the discrete closures still require analysis. The complete-map tests supply evidence of growth without establishing continuum ill-posedness. The coupled source experiment likewise demonstrates why one negative diagonal term cannot establish stability of the full system.

Pointwise checks at positive radius also do not prove a uniform estimate through the puncture limit, where this background's lapse and conformal factor vanish. Two resolutions are insufficient to determine the continuum limit of the measured growing mode.

## Independent spherical boundary diagnostic

An independently derived four-field continuum subsystem evolves physical H, radial M, Theta and radial Q on a fixed trumpet annulus. It retains the background gradients and the damping-profile derivative. At an inner radius below the horizon both constraint wave speeds leave the annulus, so no inner data are imposed. At the outer radius the two incoming constraint waves are prescribed.

On r in[0.2,4]M, setting the incoming principal wave to zero gives a growing mode with gamma=0.0262079/M for the original damping. Adding the spherical outgoing approximation,

`W_t + (c-beta_r)*(W_r + W/r) = 0`, `W=(Theta,Q_radial)`,

changes the largest real eigenvalue to -0.1123134/M. The constant damping0.1/0.3 comparisons similarly change from positive to negative. The entire finite degree128 spectrum was inspected; degree64/128 leading rates agree within4.1e-12/M. Independent off-grid checks of the original four equations support the numerical reduction.

This is strong boundary sensitivity **in the specified reduced finite-domain problem**. It is not an intrinsic infinite-domain bulk instability or a production cure. Neither condition equals the current full-state characteristic zero-rate closure. The old componentwise Sommerfeld test also differs: it acts on evolved variables with fixed speeds rather than physical constraints with curved speeds and advection. A useful next change requires a consistent lift to the full incoming constraint rows, preserving gauge/radiation data and including tangential and coefficient terms. Adding -u/r indiscriminately is not that derivation. See the annulus report and independent peer review in the evidence directory.

Using areal radius R=r+1 and the physical orthonormal connection constraint gives separate leading radiation weights: Theta falls as1/(r+1), while Q_radial falls as1/r. The corresponding constraint-only leading eigenvalue is-0.07279069/M for original damping, with degree64/128 agreement and no positive finite eigenvalues. An independent eight-field spherical metric/gauge operator reproduces this constraint subsystem exactly symbolically and agrees with the Cartesian point operator to7.67e-16 relative error. Physical mass and radial-coordinate perturbations preserve H/M/Q/Theta=0 exactly, even when the full-state characteristic amplitudes do not vanish. This checks that zeroing full-state characteristic amplitudes is not equivalent to preserving the physical constraint surface.

The full system additionally has one incoming shift-gauge characteristic at the inner annulus boundary, although physical constraints are outflow there. A first direct endpoint row-replacement discretization reproduces the desired damped constraint branch but also creates fast positive eigenvalues growing with collocation resolution. It is therefore rejected as a stability claim. The consistent metric/gauge boundary closure remains unresolved; a constraint-only spectrum is insufficient. No such boundary prototype has been inserted into AthenaK or the TDE input. The [full-state closure report](evidence/modes/constraint-lower-order/radial-full-boundary/RESULTS.md) records the convergence, endpoint defects and required compatible characteristic treatment.

## Checkpoint validity

`check_checkpoint.py RUN --ranks N` validates a complete saved uniform M=R0=1 trumpet checkpoint, including matching rank headers, all payload finiteness, positive lapse/chi, and Sylvester positive-definiteness checks in every active and ghost cell. It is deliberately restricted to a serial run or a uniform one-block-per-rank partition; it must not be reused for production AMR checkpoints.

The covariant-source sigma=0.1 trial provides an important negative test: it reaches300M with exit0 and all checkpoint numbers finite, yet all eight fourth corner ghosts have a negative second principal minor. Algebraic projection gives determinant approximately1 without making an indefinite metric positive definite. This checkpoint is rejected; finite histories, determinant positivity or exit0 alone would miss the failure.

## Vacuum gate and Liu spinning initial data

Before spin, require both exact equilibrium preservation and controlled perturbations with bounded/decaying post-transient constraints through1000M, finite active/ghost metrics, no recovery errors, resolution/amplitude checks, MPI/GPU and representative refinement-transfer coverage. A walltime stop or finite completion with positive exponential growth does not pass. No residual clipping, threshold reset or disabling physical evolution is allowed to manufacture a pass.

[Liu, Etienne and Shapiro](https://arxiv.org/abs/1001.4077) give Kerr wormhole puncture initial data with a finite coordinate horizon near extremality. That is not the present positive-lapse stationary trumpet. Independent high-precision geometry checks verify their constraints and show that stationary data on both sheets require a signed lapse; replacing it with a positive lapse produces a genuine nonzero continuum geometric RHS. Fixed-background subtraction must not cancel that physical evolution and call it truncation-error removal.

The planned first spin test therefore uses the Liu geometry as genuine initial data with positive precollapsed lapse and full moving-puncture evolution, after the vacuum gate. A stationary positive-lapse spinning residual reference would require a separate construction or relaxation. Spin0.5 has initial coordinate horizon radius0.466506M; dx0.0625M resolves its diameter with14.93 cells. The geometry preparation is not a compiled-provider or evolution pass.

## Reproducibility and scope

All experiments start from `project/tde` commit1b93538a in separate worktrees. Compact evidence, diagnostic patches, numerical scripts and source/provenance records accompany this report under `evidence/modes/`. Large raw mode arrays, logs and checkpoints remain in the local `review/stability-modes-20260919` directory. They are not inserted into production inputs. A test of zero vacuum, a pulse, a frozen symbol, or a one-block eigenmode is not by itself validation for an evolved star or AMR.
