# Frozen finite-difference boundary diagnostic

This analysis isolates a **new rapid failure introduced by the experimental physical-constraint-radiation boundary**, distinct from the original slower residual instability in the star campaign. It does not establish a cure for either problem. No AthenaK source, production input, or Aurora job was changed by this diagnostic.

## Main result

The current experimental differential p-only update has a rapidly growing **oblique outer-face mode with nonzero normal shift**, already in a constant-coefficient Cartesian model. At the actual initial face point, rotated to normal x,

- `(x,y,z)=(1.875,-.125,-.125) M`, `r=1.883314... M`;
- `alpha=r/(1+r)=0.6531769730886202`, `chi=alpha^2=0.4266401581732121`;
- `beta=(.2255366474924469,-.01503577649949646,-.01503577649949646)`;
- `G=2`, adapted lapse coefficient `2 alpha`, `kappa1=.1`, `kappa2=0`, `eta=2`, KO coefficient `.5`;
- 16 normal cells over 4 M (`h=.25 M`), cubic polynomial ghosts, helper D4 metric derivative and D2 outer derivative;
- tangential Fourier angle `k_y h=pi/2`, `k_z=0`.

The maximum eigenvalue is **`1.2127989530 - .0904818081 i` per M**. Its eigenvector has 99.94% of its unweighted active-state squared norm in the rightmost three normal cells, and nonzero physical linear conformal-Gamma constraint `Q_i=Gamma_i-div(h)_i`. The relative eigenpair residual is `6.1e-14`. RK3 with `dt=.0375 M` gives growth `1.2127945182/M`: this is a semidiscrete boundary mode, not an RK timestep instability. The actual nonlinear run's face-centered growth is independently measured by the parent task; similarity of rates is evidence for this mechanism, not an identity proof.

The old `zero_rate` boundary is spectrally neutral to floating-point precision in this particular frozen oblique test. That result **does not validate the old full trumpet/star evolution**, which has an independently established slow unstable mode.

With only the normal shift retained, the new boundary gives `gamma=1.219615/M`. At fixed grid angle `k_y h=pi/2`, refinement gives approximately inverse-h growth:

| h/M | gamma M |
|---:|---:|
| .25 | 1.219615 |
| .125 | 2.361881 |
| .0833333 | 3.498427 |

At fixed physical wavenumber `k_y=2pi/M`, growth instead remains approximately `1.22–1.26/M`. This supports a high-frequency boundary issue. It is not a proof of the continuum variable-coefficient initial-boundary-value problem, nor a complete discrete spectrum over all tangential wavevectors.

## Why the earlier flat and radial tests missed it

Purely normal scalar and transverse-vector principal tests with zero shift did not show this rapid exponential mode. The full Cartesian STF tensor system is needed to retain tangential coupling. With actual alpha/chi but zero shift the selected branch falls to `.00341/M`; with alpha=chi=1 but the actual nonzero shift an oblique branch still grows at `.9050/M`. Thus curvature gradients, corners, AMR, and internal block edges are **not required by this reduced reproducer**. They may affect the real run.

## Candidate results: none passed

All values below refer to the selected actual-coefficient oblique test unless stated otherwise.

| Candidate | Maximum real part, per M | Assessment |
|---|---:|---|
| Current radiation, tau=1 | 1.21280 | Rejected |
| Current radiation + old gauge, replacing only TT by Eq25 principal option | 1.24253 | Rapid face mode persists |
| tau=c/(c+beta_n)=.65418 | .70643 | Still grows; not a cure |
| tau=.5 / .25 | .67150 / .57354 | Still grows; no monotone stability guarantee |
| Inner metric derivative changed to volume D6 | 1.21177 | Does not remove mode |
| Literal direct Theta/A allocation, existing gauge/TT zero-rate | .74077 | Rejected |
| Direct allocation plus existing tangential-principal gauge/TT targets | .74022 | Rejected |
| Direct+TP with six-/eight-point ghost extrapolation (degree5/7) | .85246 / .95959 | Rejected; other angles also grow |
| Direct allocation plus paper-derived gauge/TT principal completion | 1.37079 | Principal-only D0 substitution; rejected |
| Same principal completion and matched D6 inner/outer derivatives | 1.36320 | Rejected in this prototype |
| Same completion using actual selected Khat/Theta D0 rows | 1.37497 | Bounded peer-review correction; still rejected |
| Actual-D0 correction and matched D6 inner/outer | 1.36315 | Still rejected |

The paper-derived rows are a frozen adaptation to the **current G2 gauge**, including its Theta coupling. They are not a completed nonlinear implementation of a published boundary system. Explicit eta corrections are included when damping is enabled; unchanged volume constraint damping, KO, and upwind-minus-centered contributions are retained in the replaced gauge and TT rows. Peer review identified that the first prototype substituted principal-only D0Khat/Theta in the longitudinal gauge equation; a single bounded comparison corrected those to the actual selected RHS minus centered advection and still found growth. This does not complete a source-aware discrete outgoing-W closure for the shift's own KO/advection terms. The constraint Theta/A allocation follows the separately reviewed formula in `cartesian-design/DIRECT_ALLOCATION_AUDIT.md`. These limitations matter when interpreting its failure.

A later bounded test replaced **only** the TT principal update, keeping v2 constraint radiation and the original zero-rate gauge. It also failed the selected oblique gate: `lambda=1.2425276968-.0906094351i/M`, 99.96% of state squared norm in the rightmost three cells, and relative eigenpair residual `5.93e-14`. This is saved in `tt-only-replacement.json`; no other parameter sweep accompanied that test.

The direct allocation is

```
delta Theta = -FTheta
delta A_nn = -sqrt(chi) FQ_n + alpha*chi div(Q)/3 - sqrt(chi)*sigma Q_n
delta A_nA = -sqrt(chi) (FQ_A + sigma Q_A)/2
```

with `sigma=alpha*kappa1`. Scalar and transverse A coefficients are deliberately different. The existing gauge-map completion solves Khat/Gamma rates with the prescribed new Theta; the full-paper principal variant instead replaces the relevant gauge RHS, as documented in the script.

## Independent linear-ghost inconsistency

Linear ghosts (`extrap_order=2`, polynomial degree1) are inconsistent with retaining the centered sixth-order second derivative at the first active cell. For the smooth function `q=x^2`, exact `q''=2`, but the discrete boundary value is **`-7/30` independent of h**. Cubic and quadratic extrapolation reproduce 2 for this test.

In the damped flat normal test, linear ghosts have nonzero-Q growing branches approaching finite rates under refinement: scalar `.07198,.07465,.07595/M`; transverse `.11197,.11467,.11595/M` for h=.25,.125,.0625. Quadratic ghosts reduce these to roughly first-order-vanishing rates, but remain unstable at finite resolution. This is a separate concrete defect and does not explain why the new cubic radiation run also fails rapidly.

## Operator and verification scope

`fd_boundary.py` implements the separate normal scalar and vector systems. `fourier_boundary.py` uses 20 independent variables with conformal metric and A trace eliminated algebraically: chi; five h; Khat,Theta; five A; three Gamma; lapse; three shift. Normal active cells use code-matched sixth-order centered D1/D2, sign-correct sixth-order upwind advection, eighth-derivative KO `-epsilon/(256h)`, polynomial ghost extension, and p-only physical-face corrections. Tangential directions use the corresponding Fourier symbols. Mixed derivatives are tensor products of the centered D1 symbols. Boundary characteristic q-RHS derivatives use the code's one-sided second-order active stencil. The helper's metric D4 and outer D2 are active-only in the normal direction and centered Fourier symbols tangentially.

The model does not contain puncture coefficients varying across the domain, background K/A or their derivatives, physical falloff/connection terms, nonlinear algebraic projection, MPI, AMR, multiple-face corners, or the helper's one-sided tangential stencils at block edges. It is a frozen face test, not a replacement simulation.

The full matrix reduces to the separately implemented scalar/vector matrices at zero tangential wavenumber within `4.7e-14`, including actual face coefficients, shift and damping. The tangential-principal target reduces to zero for a purely normal wave within `4.7e-14`. The existing repository characteristic algebra test is imported and passes. See `sector-crosscheck.json`, `actual-sector-crosscheck.json`, and `tp-normal-limit.json`. A compact executable regression in `verify_operator.py` checks these limits, exact polynomial moments, and the shifted oblique mode against the neutral old-boundary control. Both peer agents independently confirmed the main v2 translation and reviewed the provisional paper completion; the principal-vs-actual D0 caveat above was their sole concrete formula-translation finding.

Eigenvalues of size about 1e-6 around defective neutral modes are numerical noise in this double-precision matrix analysis; they are not reported as physical exponential growth. No positive-growth conclusion here relies on those roots. Nonnormal transient growth and all untested wavevectors remain outside the reported spectral checks.

## Reproduction and next step

Use Python3 with NumPy and Matplotlib; keep this directory anywhere inside the AthenaK repository so the characteristic reference module is discoverable.

```
OPENBLAS_NUM_THREADS=1 python3 reproduce_selected.py coefficient-isolation.json
OPENBLAS_NUM_THREADS=1 python3 reproduce_selected.py oblique-refinement.json
OPENBLAS_NUM_THREADS=1 python3 reproduce_selected.py completion-tests.json
OPENBLAS_NUM_THREADS=1 python3 reproduce_selected.py full-gauge.json
OPENBLAS_NUM_THREADS=1 python3 verify_operator.py
python3 plot_evidence.py
```

The next justified step is a **discrete boundary energy / normal-shift-compatible closure analysis** using this oblique reproducer and a coherent constraint/gauge/radiation boundary operator. Further production tuning or simply increasing extrapolation order is not justified by these tests. Require the new closure to pass the frozen shifted oblique matrix and finite-time tests before another nonlinear trial. All tested candidates above remain rejected, and the old campaign's slower instability remains unresolved.
