# Continuum half-space audit of the experimental v2 closure

The proposed v2 closure admits a rapidly growing **continuum boundary mode** in the frozen half-space problem. The measured finite-grid fast branch is therefore not attributable solely to its D2/D4/D6 stencil mismatch. This identifies a failure of the proposed new closure; it does **not** identify or repair the earlier slow instability of the production configuration.

## Problem and scope

Use the exact principal Z4c equations with algebraic metric/A traces eliminated, adapted lapse driver2α, shift driverG=2, and optional original lower-order κ1=.1,κ2=0,η=2 terms. Freeze coefficients at (1.875,−.125,−.125)M: α=.6531769730886202,χ=.4266401581732121,β_n=.2255366474924469. Background gradients, reference-connection terms, areal decayω, mesh interfaces, projection roundoff and discretization are absent. This is not the full variable-coefficient trumpet equation.

The ansatz is exp(λt+s x+i k_y y), on x<0 with outward normal+x. Modes with Re(s)>0 decay into the domain. Constant tangential shift can be removed exactly by a Galilean frequency shift; the actual β_y=−.01503577649949646 gives Im(λ)=β_y k_y.

Eliminating the ten momentum variables through the invertible q_t–p coupling gives a10×10 quadratic normal pencil with20 normal roots. An **ordered complex Schur invariant subspace** selects all ten decaying solutions, including generalized normal modes. The initial raw-eigenvector scan had basis conditioning up to1.8e10 and is retained only as exploratory provenance. Validated results use Schur subspaces; the boundary trace basis condition is41.6 at the central root.

For a smooth bulk solution to also satisfy the modified boundary RHS, all momentum corrections must vanish. The nonsingular p-map then requires four incoming gauge rates zero, FΘ=FQ_i=0, and two incoming TT rates zero. For Re(λ)>0 the zero rates can be divided by λ. The physical radiation factor is λ+(c−β_n)s−iβ_Ak_A with c=α√χ. This equivalence and the symbol/Schur implementation were independently peer-reviewed by gauge_sponge.

## Validated growing root

At k_y=2π/M with κ/η retained,

    λ = (1.2307116366820878 − 0.09447256998367227 i)/M.

The real part is found by a sign-changing determinant root in a q-boundary-value chart, independent of Schur basis phases. The row-scaled smallest singular value is3.87e−16; determinant magnitude1.10e−17. The reconstructed smooth normal profile has maximum relative defect7.84e−14 in all20 original bulk equations and normalized boundary defect1.66e−16. It decays by a norm factor1.53e−5 between x=0 and−2M. These are numerical residual checks, not a symbolic proof.

| k_y M | Growth without κ/η | Growth with κ1=.1,η=2 |
|---:|---:|---:|
| π | .5987161270 | .6273294212 |
| 2π | 1.1974322540 | 1.2307116367 |
| 4π | 2.3948645081 | 2.4310861485 |

The undamped principal roots scale linearly with k_y (γ/k_y≈.1905772622). Homogeneity of the principal equations extends such a root to arbitrarily high frequency by scaling λ,s,k and the momentum amplitudes together. This is evidence of a failure of the frozen principal boundary well-posedness condition, rather than a timestep-only instability. The original zero_rate closure has smallest singular value.14025 at the central damped root; that comparison excludes this root only, and is not a general stability proof for zero_rate.

## Physical content

Normalize the largest state component at the boundary to1. Along the reconstructed profile, physical constraints remain at numerical residual level: max|H|3.43e−14,‖M‖6.95e−14,|Θ|8.01e−15,‖Q‖2.55e−14. In contrast, the linear electric and magnetic Weyl norms reach.144996 and.0102603. The mode is therefore **constraint-satisfying with nonzero physical curvature**, not a pure coordinate mode. Lapse/Khat are nearly zero, while metric, tracefree extrinsic curvature and shift participate.

This distinction matters: the finite-grid v2 direction has substantial H/M/Θ/Q. The continuum boundary mode and its discrete counterpart need not have identical constraint content. It would be incorrect to call the continuum eigenfunction an incoming constraint-wave instability or equate it with the earlier slow TDE mode.

## Two bounded boundary alternatives

Only the requested alternatives at k_y=2π were examined:

1. **Dirichlet residual lapse and shift at the boundary**, retaining physical FΘ/FQ and old TT rows: the old root is removed (σmin=.03593 there), but an exact positive root occurs at λ=β_n k_y=1.417088549755/M. Ordered Schur includes the coincident/generalized normal modes at this point. This root is curvature-free to numerical accuracy (E2.78e−13,B3.40e−12), with boundary lapse/shift zero. It is not a stability pass.
2. **Replace only the two TT rows by the frozen published radiation equations**, retaining old gauge rows and physical FΘ/FQ: the old growth rate remains a boundary root (σmin6.73e−16). Its nullvector now has E9.57e−14,B6.04e−14 and is curvature-free. Thus the computed nullvector at this root is curvature-free, while coordinate growth remains; other branches were not excluded. It is not a stability pass.

No C++ v3 option was added, no further boundary variants were tested, and no Aurora/production actions were taken.

## Reproduction

Run `OPENBLAS_NUM_THREADS=1 python3 run_all.py` with NumPy and SciPy. All constants and characteristic rows are included locally; no historical worktree or binary is needed. `validated-root.json`, `physical-gauge-validation.json`, `dirichlet-physical-validation.json`, and `weyl-physical-validation.json` contain the numerical evidence. NPZ files retain the Schur normal generator, boundary matrix and mode coefficients. The scripts were reproduced successfully on local CPU; no GPU portability or production validation is claimed.
