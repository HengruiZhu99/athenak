# Spherical continuum constraint-mode diagnostic

The independently derived spherical constraint subsystem has a converged growing mode on a finite annulus when the incoming **principal** wave characteristic is set to zero. Adding the usual spherical `1/r` term to that radiation condition removes the positive eigenvalue in the tested collocation spectra. This is evidence that lower-order boundary terms matter in this particular continuum initial-boundary value problem. It does **not** identify the origin of the measured 3D AthenaK modes, establish an infinite-domain formulation instability, or validate a production boundary change.

No AthenaK source, physical evolution, or Aurora job was changed. All computations here are local linear eigenproblems. The earlier damping audit is [AUDIT.md](AUDIT.md); an independent source/formulation review found no coefficient error and is saved at [annulus peer review](../mode-analysis/annulus-peer-review/REVIEW.md). Its independently evaluated off-grid and endpoint checks also pass, with the scope limitations retained below.

## Subsystem and assumptions

Use `R=r+1`, `alpha=r/R`, `chi=alpha²`, radial shift `b=r/R²`, and `K=1/R²`; `M=R0=1`. The conformal metric is the identity and physical mixed extrinsic curvature is `K^i_j=K(delta^i_j−2 n^i n_j)`. Algebraic determinant/trace constraints are assumed enforced. The radial perturbation variables are the physical ADM Hamiltonian `H`, covariant physical momentum `M_i=m n_i`, `Theta`, and `Q^i=q n^i=Gamma_evolved^i−Gamma_metric^i=2 gtilde^ij Z_j`. This is the vacuum, spherically symmetric, linearized constraint subsystem of the continuum equations audited in AUDIT.md. It omits nonspherical sectors and does not evolve independent metric/gauge perturbations.

With `D0=partial_t−b partial_r`, primes denoting radial derivatives, `kappa2=0`, and the repository's Gamma damping coefficient retained, the equations used are

```
D0 Theta = alpha*H/2 + alpha*chi*(q' + 2q/r)/2 − 2sigma*Theta
D0 q     = 2alpha*(m + Theta') − 2sigma*q
D0 H     = −2alpha*chi*m'
           + (−4alpha*chi/r + alpha*chi' − 4chi*alpha')*m
           + 2alpha*K*H + 4alpha*chi*K*q' − 4sigma*K*Theta
D0 m     = (b' + alpha*K)*m − alpha*H'/2 − alpha'*H
           − alpha*chi'*q' − (2chi*alpha' + alpha*chi')*q/r
           + 2sigma*Theta' + 2sigma'*Theta
```

Current damping means `sigma=0.1alpha`; the two lapse-adjusted controls use constant `sigma=0.1` and `0.3`. In particular, the spatial derivative of sigma is retained. The Ricci addition is `C_ij=partial_(i Q_j)`, and the difference from vacuum ADM evolution is `T_ij=alpha C_ij−sigma Theta gamma_ij`. Independent symbolic checks recover the covariant-divergence terms in the radial H and m equations exactly; no background-coefficient gradient was dropped.

## Boundary conditions and wave reduction

For `W=(Theta,q)`, eliminate `(H,m)` to obtain a variable-coefficient two-field wave system. Its principal part is

```
W_tt = 2b W_tr + (c²−b²) W_rr + lower-order terms,
c = alpha*sqrt(chi) = alpha².
```

The physical coordinate characteristic speeds are `−b±c`. Defining `Pi=W_t−b W_r`, the outer incoming wave is `Pi+c W_r`. At `r_inner<1`, both wave speeds are negative and point out of the computational annulus into the hole: **no inner data are prescribed**. At outer radius `L>1`, two incoming components require data.

The first boundary choice (`p=0`) sets the local incoming principal field to zero. The second (`p=1`) adds a spherical outgoing approximation:

```
W_t + (c−b)*(W_r + p W/r) = 0, at r=L.
```

For an eigenmode this is `lambda W+(c−b)*(W'+pW/r)=0`. These are homogeneous boundary conditions on the constraint subsystem. `p=0` is not exactly nonreflecting at finite radius in a spherical background; `p=1` is also an approximation, not an exact absorbing condition for the coupled variable-coefficient equations. Neither has been mapped to a well-posed, constraint-preserving boundary treatment for the full Z4c metric/gauge system, and neither equals the production residual-ghost extrapolation rule. The earlier per-variable Sommerfeld evolution controls therefore do not duplicate this experiment.

## Results

For `r_inner=0.2M`, `L=4M`, polynomial degree 128, all 514 finite generalized eigenvalues were inspected. The two omitted infinite eigenvalues arise from the algebraic boundary rows. The largest real parts are:

| Constraint damping | `p=0` growth rate `gamma M` | `p=1` growth rate `gamma M` |
|---|---:|---:|
| `sigma=0.1alpha` | +0.0262079071663 | −0.1123134495271 |
| `sigma=0.1` | +0.0259246669330 | −0.0842609432227 |
| `sigma=0.3` | +0.0488040955867 | −0.0128583882910 |

For `p=0`, each tested degree 128 matrix has one positive eigenvalue. For `p=1`, none has positive real part (threshold `1e−8/M`). The largest-eigenvalue branches agree between degrees 64 and128 to at worst `4.1e−12/M`. Negative finite spectra do not by themselves prove a continuum energy estimate, absence of transient nonnormal growth, or nonlinear stability.

The `p=0` growing mode extends broadly into the exterior. For current damping, normalized Theta at `r=4M` is about 0.64 of its maximum at `r_inner=0.2M`; its maximum is not evidence of an injection at that location. Moving the inner cutoff from 0.1 to 0.9M changes the degree 96 eigenvalue by only about `1e−12/M`. At 0.05M it agrees within `2.3e−9/M`, with less spatial resolution near the cutoff. This supports independence from an imposed inner boundary in the tested annuli, **not** regularity or finite energy at the puncture limit `r→0`.

Outer radius matters: for current damping and `p=0`, `L=2,4,8M` give `gamma M=0.0238583944,0.0262079072,0.0184092683`. The constant-sigma controls also remain positive under this boundary choice. More coordinate damping is not monotonically stabilizing in this coupled subsystem.

## Numerical checks and reproducibility

- [radial_collocation.py](radial_collocation.py) directly discretizes the original four equations with Chebyshev collocation. Outer H/m evolution rows are replaced by the two wave boundary conditions; Theta/q equations remain there. It does not introduce independent gradient variables or a reduction constraint. Full finite-spectrum counts are recorded, rather than only eigenvalues near a chosen shift.
- [radial_modes.py](radial_modes.py) independently eliminates H/m, symbolically verifies the wave principal factorization, and uses a first-order upwind six-field wave reduction. For current damping and `p=0`, its growing eigenvalue changes from0.02638474 at 96 cells to0.02621063 at 6144, approaching the direct collocation result 0.02620791. Its finite-resolution reduction-constraint defects are appreciable; they are preserved in output and are why this method is only a convergence cross-check.
- [radial_verify.py](radial_verify.py) verifies radial covariant identities symbolically and interpolates the leading eigenfunctions to 2003 off-collocation points. It evaluates the original four equations there independently of the eigenmatrix. For the three degree 128 `p=0` cases, relative residuals normalized by the sum of individual term L2 norms are between `4e−13` and `7.4e−12`; the last eight Chebyshev coefficient norms are about `1e−14` relative to the full series. Boundary residuals are also saved. The check covers points arbitrarily close to both endpoints, not just interior matrix rows.
- [radial-boundary-comparison.json](radial-boundary-comparison.json) records the matched boundary/damping comparisons and all-finite-spectrum counts. [radial-sweep.json](radial-sweep.json) records the cutoff/radius tests. [radial-offgrid-validation.json](radial-offgrid-validation.json) records independent checks. Individual JSON files contain eigenprofiles and diagnostics.

Example (bundled Python includes NumPy/SciPy; scripts locate the existing local SymPy dependency):

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python radial_collocation.py \
  --n 128 --inner .2 --outer 4 --rate .1 --wave-power 1 \
  --output annulus-example.json
```

The spectra take seconds locally; the 6144-cell sparse upwind cross-check took about 19 seconds. No long PDE evolution was performed here.

## Interpretation and next step

The continuum constraint sector can support a finite-boundary growing mode even when its frozen principal symbol has only wave speeds. The lower-order radiation term can change that conclusion completely in this annulus. Therefore one should not infer a puncture-origin instability merely from a peak inside/near the horizon, nor infer an intrinsic bulk formulation instability from this positive annulus eigenvalue.

A useful next step is to derive the full Z4c constraint-preserving incoming boundary operators, including spherical falloff and compatibility with metric/gauge and physical wave data, then validate zero background, controlled constraints, and matter response in an isolated implementation. This diagnostic alone does not justify replacing the production boundary, increasing damping, or claiming the TDE run cured. The relation between these spherical continuum modes and the measured 3D discrete modes remains unestablished.
