# Liu Kerr preparation: geometry passes; evolution is gated

The Liu radial map and initial data are implementable, but they cannot be installed as a stationary positive-lapse residual background by changing only the radial coordinate. Independent ADM checks below expose a continuum stationarity obstruction. The chosen first-spin route is therefore **constraint-satisfying Liu initial data with genuine subsequent full geometric and gauge evolution**, after the vacuum gate. No spinning evolution, Aurora action, or shared-source edit was performed. An empty detached preparation worktree was briefly created and then removed; implementation is deferred until the vacuum gate passes.

## Primary-source scope

[Liu, Etienne & Shapiro, arXiv:1001.4077](https://arxiv.org/abs/1001.4077) introduces a Kerr puncture radial coordinate whose initial horizon retains finite coordinate size near extremality. Its initial slice is an Einstein–Rosen bridge. Their BSSN moving-puncture evolutions relax toward a trumpet; the initial wormhole is not that final trumpet. They use sixth-order spatial differences, RK4, Sommerfeld boundaries, and compare initial analytic versus precollapsed gauge choices. The reported stationary-hole high-spin example covers 180M. This is useful initial-data evidence, not proof that AthenaK's fixed residual Z4c equations, matter coupling, or current boundaries will be stable. The complete six-page primary PDF was read, with the formula page also rendered to verify notation.

## Formulas to implement and independently test

Here `r` is the new Cartesian radius, `R` is Boyer–Lindquist radius, and spin is along +z. Use `M=1` initially, `a/M=0.5`, and restrict `|a|<M`; the extremal throat requires a separate limiting analysis. Define

\[
r_\pm=M\pm\sqrt{M^2-a^2},\quad c=r_+/4,\quad
R=(r+c)^2/r,\quad J=dR/dr=1-c^2/r^2,
\]
\[
\Sigma=R^2+a^2\cos^2\theta,\quad
\Delta=(R-r_+)(R-r_-),\quad
A=(R^2+a^2)^2-\Delta a^2\sin^2\theta.
\]

The spherical spatial metric is diagonal:

\[
\gamma_{rr}=\frac{\Sigma(r+c)^2}{r^3(R-r_-)},\quad
\gamma_{\theta\theta}=\Sigma,\quad
\gamma_{\phi\phi}=A\sin^2\theta/\Sigma.
\]

The only nonzero extrinsic-curvature entries are the symmetric pairs

\[
K_{r\phi}=\frac{Ma\sin^2\theta}{\Sigma\sqrt{A\Sigma}}
\left[3R^4+2a^2R^2-a^4-a^2(R^2-a^2)\sin^2\theta\right]
\frac{1+c/r}{\sqrt{r(R-r_-)}},
\]
\[
K_{\theta\phi}=-\frac{2Ma^3R\cos\theta\sin^3\theta}{\Sigma\sqrt{A\Sigma}}
(r-c)\sqrt{(R-r_-)/r}.
\]

These are Eqs. 11 and 13–15 of the [primary paper](https://arxiv.org/pdf/1001.4077), expressed with `c`. The remaining formulas and stationarity analysis here are our independent algebra/code checks. The trace is `K=0`. For stationary coordinates, `beta^phi=Omega=-2MaR/A`, `beta^r=beta^theta=0`.

Avoid spherical axis divisions in the provider. With `n=x/r`, `v=(-y,x,0)`, `C=z/r`, `k=(0,0,1)`, set `B=Sigma/r^2`, `F=gamma_rr`, and

\[
\gamma_{ij}=B\delta_{ij}+(F-B)n_i n_j+
\frac{a^2(\Sigma+2MR)}{\Sigma r^4}v_i v_j,\qquad
\beta^i=\Omega v^i.
\]

If `K_rphi=sin(theta)^2 U` and `K_thetaphi=C sin(theta)^3 V`, the axis-regular algebraic transformation is

\[
K_{ij}=\frac{U}{r^2}(n_i v_j+n_j v_i)
+\frac{CV}{r^3}[(Cn_i-k_i)v_j+(Cn_j-k_j)v_i].
\]

No cell may coincide with `r=0`. At `r=c` the nonextremal spatial metric and these K expressions are finite; do not evaluate `J^2/Delta` as a numerical 0/0. The two intervals `r<c` and `r>c` are two exterior sheets joined at the throat, not the horizon-penetrating interior used by the present trumpet.

For the code convention, define `D=det(gamma_cart)`, `psi4=D^(1/3)`, `chi=D^(-1/3)`, `gtilde=chi*gamma`, `Atilde=chi*K`, `Khat=Theta=B_i=0`, and `Gamma^i=-partial_j(gtilde^{ij})`. Do not initialize Gamma to zero: spinning data are not conformally flat. Generate/test Cartesian metric first derivatives or a small derivative helper for direct Gamma, then use the same provider for ADM and Z4c caches. The existing code's precollapsed lapse is unambiguously `alpha=psi4^(-1/2)=sqrt(chi)`; use this definition instead of relying on the paper's ambiguous printed conformal-factor shorthand.

At `a/M=0.5`, `r_h=c=0.46650635094610965M` and initial diameter `0.9330127018922193M`. `dx=0.0625M` gives 14.93 cells across; `dx=0.125M` gives only 7.46. The Schwarzschild limit is the isotropic wormhole, with horizon 0.5M, not the current `R=r+1` trumpet with horizon 1M.

## Lapse sign and residual-equation obstruction

From the smooth K above, the lapse satisfying the stationary ADM equations on both sheets is

\[
\alpha_{\rm signed}=(r-c)\sqrt{(R-r_-)\Sigma/(rA)}.
\]

Its square is `Delta*Sigma/A`; it is positive outside, negative on the second end, and zero at the throat. `L_beta gamma=2 alpha_signed K` verifies the sign directly. Replacing it by its positive absolute value changes the interior metric RHS to `-4 |alpha| K`; changing the sign of K there instead creates an incompatible nonsmooth join. The absolute lapse also has a cusp at the throat. A smooth positive precollapsed lapse and zero shift are legitimate initial gauge data, but then the geometry must evolve.

This matters before any discretization experiment. Current residual evolution is `d(delta)/dt=F_h(bg+delta)-F_h(bg)` with a fixed background. If positive-lapse Liu data have `F_cont(bg)!=0`, exact zero preservation removes a **continuum geometric evolution**, not just background truncation error. A vacuum zero test alone would falsely appear successful. The adapted residual gauge does not fix that geometric inconsistency. Moreover the present adapted 1+log driver contains `2*alpha_bg`, which is negative on the inner sheet for the signed lapse; its lapse characteristic squared speed becomes negative. Existing positive-lapse validity/CPBC checks reject this state. Do not bypass those checks or silently floor/absolute-value the lapse.

`check_geometry.py` constructs symbolic derivatives and contracts the ADM equations at 60 digits for `a/M=0,0.5`, seven radii `r/c={0.1,0.5,0.99,1,1.01,2,10}` at `theta=pi/3`. It tests the full geometric metric/K RHS, not just constraints. [geometry-results.json](geometry-results.json) records:

- Maximum `|H|=2.14e-61`, momentum norm `7.93e-62`, and exactly zero K trace.
- Signed-lapse stationary metric/K RHS norms below `8.7e-62`, including the throat as a spatial initial-data limit.
- At `a=.5`, `r=c/2`, positive-absolute lapse gives metric RHS norm `0.220376/M` and K RHS norm `0.0394233/M^2`.
- Precollapsed lapse/zero shift is also nonstationary: at the throat the corresponding norms are `0.118698/M` and `0.115577/M^2`.

These are continuum algebra checks in spherical coordinates, not tests of a compiled AthenaK provider, Cartesian axis limits, MPI, AMR, or evolution. `wall_seconds` in that JSON measures the point contractions after symbolic compilation.

## Existing implementation and required changes

Audit at commit `1b93538a`:

| Existing path | What it supplies / what must change |
|---|---|
| `src/pgen/z4c_tov_ks.cpp:2755` | Only Kerr–Schild, Schwarzschild puncture, and Schwarzschild trumpet; both puncture branches explicitly reject spin. Liu is absent. |
| `src/pgen/z4c_tov_ks.cpp:1320` and `:1400` | ADM and direct residual-background callbacks can host a new provider, but reuse neither KS tensors nor their derivatives. |
| `src/pgen/z4c_tov_ks.cpp:1488` | Generic ADM-to-direct-Z4c algebra and Gamma construction are reusable once given the correct Liu metric derivatives. |
| `src/z4c/z4c_adm.cpp` | General ADM/Z4c conversion is available; discrete Gamma construction is a useful independent check, not an excuse to set it zero. |
| `src/z4c/z4c_calcrhs.cpp:558` | Adapted lapse/shift drivers act on residuals; full/background subtraction also acts on geometric equations. Gauge stationarity cannot establish geometric stationarity. |
| `src/z4c/z4c_Sbc.cpp:391`, `src/z4c/z4c.cpp:770`, `src/dyn_grmhd/dyn_grmhd.cpp:538` | Positive-lapse assumptions invalidate the signed stationary bridge in the current evolution path. |
| `src/outputs/history.cpp:33`, pgen horizon/AMR/excision helpers | Many masks use oblate Kerr–Schild radius. Liu needs spherical coordinate-radius masks with `r_h=c`; setting physical spin in an old KS mask gives the wrong region. |
| `src/pgen/z4c_tov_ks.cpp:2777,2803,2890` | Coordinate spin consistency, orbit helper rejection, and puncture restrictions need explicit mode-aware validation. No automatic all-ingoing KS placement for Liu. |

Spinning Kerr–Schild already exists, but is a different slice and does not implement Liu. The direct zero-background machinery, audits, per-rank outputs, and AMR transfer tests can be reused. Existing star velocities/orbit positions must later be transformed into the chosen new coordinates; the old KS orbit helper must remain forbidden for Liu.

## Implementation decision and strict test gate

1. **Finish the current nonspinning vacuum gate first.** Require exact zero through stage operations and long evolution, then bounded perturbation behavior through at least 1000M, including exterior/core constraint norms, ghosts, gauge fields, changed timestep/resolution, MPI and a representative refinement interface. A finite stop with positive fitted exponential growth does not pass. No Liu/spinning evolution starts before the root investigator explicitly records that gate as passed.
2. **Use the chosen full-evolution route.** Initialize the smooth Liu wormhole with positive precollapsed lapse and allow genuine full moving-puncture geometric/gauge evolution. This requires a separate full-evolution initial-data path; current fixed subtraction cannot be reused unchanged. If a residual background is later required, a time-dependent reference or a separately constructed gauge-relaxed stationary Kerr background is a distinct extension. A stationary positive-lapse rotating trumpet is a different slice, not Liu's radial map.
3. **Implement geometry in the isolated worktree only.** A dedicated `kerr_liu` helper should return ADM fields, first derivatives, conformal Gamma, and explicit gauge mode. Add mode-specific parameter validation, coordinate diagnostics, and radius masks. Retain all lapse/metric failure checks. Default to spin 0.5; prohibit matter and production use while validation is incomplete. No sign-changing stationary lapse in the current positive-lapse gauge path.
4. **Geometry gate before timestepping.** Compile the helper and compare against this independent reference off-axis, at both sides of the throat, at the throat limit, and on/near the axis. Test `a=0` isotropic reduction, spin parity, SPD metric, unit conformal determinant, traceless A, Gamma from independent differentiation, convergent H/momentum, and ADM mass/angular momentum. Explicitly measure unsubtracted geometric RHS for the chosen initial lapse. It is expected nonzero for a full moving-puncture relaxation and must not be subtracted away accidentally.
5. **First permitted spin run after both gates.** Vacuum `M=1,a=.5`, centered, unboosted, no matter/sponge/excision/projection reset; use the already validated vacuum numerical configuration and an initially resolved horizon (`dx<=0.0625M`). Stage/time targets 1M, 10M, 100M, then 1000M only after clean diagnostics. Measure constraints, horizon area/spin, gauge relaxation, exact stopping reason, and compare two spacings. MPI/refinement follow before any atmosphere or star. AMR must follow the changing horizon size during relaxation. New gauge behavior is expected; it is not residual instability by definition, and must be judged against constraints and convergence.

No input here is a submission-ready spinning run. The chosen continuum/gauge interpretation is explicit; implementation and spinning timestepping remain gated on successful vacuum tests.
