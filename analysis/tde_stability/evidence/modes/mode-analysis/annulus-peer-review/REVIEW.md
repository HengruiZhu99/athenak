# Independent review of the radial constraint annulus experiment

I found no algebraic error in the continuum source reduction or radial tensor coefficients. The packaged numerical evidence supports a growing mode of the **specified finite-annulus constraint IBVP** under its p=0 boundary condition, and removal of that positive branch in the tested p=1 collocation spectra. It does not establish an infinite-domain instability, ill-posedness, or a cure for the full AthenaK boundary problem.

Reviewed read-only: `constraint-lower-order/AUDIT.md`, `RADIAL_ANNULUS.md`, `radial_modes.py`, `radial_collocation.py`, `radial_verify.py`, the six matched degree128 profiles, and actual `src/z4c/z4c_calcrhs.cpp`/`z4c_Sbc.cpp`. No reviewed source or result file was edited. My independent numerical profile check is `check_profiles.py`; results are `profile-checks.json`.

## Actual formulation and physical constraints

The source's Ricci calculation uses evolved Gamma only in its derivative term, while undifferentiated Christoffel contractions use metric-derived Gamma. Thus its continuum Ricci difference is exactly C_ij=gtilde_k(i partial_j)Q^k. On the conformally flat trumpet this is partial_(i Q_j). This identity would be different if the undifferentiated evolved Gamma were used; the audit correctly follows this repository.

Reconstructing physical K=Khat+2Theta and K_ij from the A/chi equations gives the ADM modification T_ij=alpha C_ij−sigma Theta gamma_ij at kappa2=0. The two Theta contributions in full K must be included: Khat damping is +sigma Theta, twice Theta damping is −4sigma Theta, so full K damping is −3sigma Theta. Its tensor trace contribution is precisely T, not an extra ad hoc Hamiltonian addition.

Subtracting metric-Gamma evolution from evolved-Gamma evolution gives D0 Q^i=2alpha(M_i+partial_iTheta)−2sigma Q^i on this background. The nonadvective shift-gradient terms cancel because the source uses metric Gamma in those terms. No extra contravariant Lie-derivative term should be inserted into this coordinate-component equation. The proposed audit retains the actual Gamma factor2.

The variables H and M_i are physical ADM constraints, not the Ricci-modified Ht or momentum norms squared. Algebraic determinant and trace constraints are assumed enforced. Gauge perturbations do not add independent linear driving terms to the closed constraint subsystem because the continuum background constraints vanish. This does not identify the full metric/gauge boundary problem with the reduced constraint IBVP.

## Radial tensor checks

For M_i=m n_i, physical divergence is

D_i M^i = chi m' + (2chi/r−chi'/2)m.

Writing the mixed tensor T eigenvalues as A=alpha chi q'−sigma Theta radially and B=alpha chi q/r−sigma Theta tangentially gives

D_j T^j_r−D_r tr(T)
= −alpha chi' q'−(2chi alpha'+alpha chi')q/r
  +2sigma Theta'+2sigma' Theta.

The Hamiltonian addition is 4K A, because K gamma^ij−K^ij=2K chi n^i n^j on this trumpet. The momentum shift term is beta' m, not an additional beta m/r. These independent expansions reproduce every coefficient in the four-field collocation matrix. The wave elimination retains variable coefficients and the radial vector/angular terms.

## Numerical evidence

I reconstructed the saved eigenprofiles using Chebyshev coefficients and independently evaluated the original four PDEs at1001 off-collocation points **and both endpoints**, using explicit analytic background derivatives and radial tensor divergences. Across all six matched degree128 cases:

- original-equation relative L2 residual <=3.9e-10;
- endpoint residual relative to the global equation peak <=3.2e-9;
- outer-condition residual relative to field norm <=2.4e-13.

The missing H/M evolution rows at the outer endpoint therefore do not hide a large violation of the original equations. The latest code records the entire finite generalized spectrum before selecting detailed low-frequency profiles:514 finite eigenvalues, two infinite algebraic boundary eigenvalues. The p=0 matrices each have one positive eigenvalue; p=1 matrices have none above1e-8/M. The reported degree64/128 convergence and independent upwind convergence are consistent with the profile checks. This remains numerical evidence, not a spectral enclosure or energy estimate.

## Boundary interpretation

For W=(Theta,q), Pi=W_t−b W_r and c=alpha sqrt(chi), the tested conditions are

- p=0: Pi+c W_r=0;
- p=1: Pi+c W_r+(c−b)W/r=0.

The second is W_t+(c−b)(W_r+W/r)=0. The falloff coefficient is **c−b**, not c. At r_inner<1 both physical wave speeds −b±c leave the annulus, so imposing no inner data is consistent. At the outer boundary one incoming field for each wave is prescribed.

The p=1 term is a leading spherical outgoing approximation. The radial q is a vector/l=1 amplitude, and the full equations have curved-background potentials and couplings; a common1/r falloff is not an exact absorbing condition at r=4M. The growth sign changing when this lower-order boundary term changes is strong evidence of boundary sensitivity in this IBVP. It prevents attributing the p=0 eigenvalue to intrinsic bulk or puncture instability. Inner-cutoff independence on tested annuli does not prove regularity/finite energy at r→0. Positive finite growth is compatible with well-posedness and strong hyperbolicity.

## Why this is not the current full-state CPBC

`ApplyResidualCharacteristicBC` uses incoming characteristic combinations of geometric state and normal derivatives. Its scalar constraint amplitudes include, in the local conformal frame,

U2 = sqrt(chi) Theta + chi Gamma_n/2 + d_n chi,
U3 = 4 Khat/(3sqrt(chi)) + 2 Theta/(3sqrt(chi))
     −2 A_nn/sqrt(chi) − Gamma_n + d_n h_nn.

Here A/h are trace-free projected components. `zero_rate` sets the corresponding frozen-coefficient RHS combinations to zero. It does not directly prescribe Pi_Theta+c Theta_r or Pi_q+c q_r. A relation exists at the principal level through differential constraints, but variable coefficients, angular terms, lower-order sources, and gauge/metric data must be retained to derive the actual induced constraint boundary conditions. At nonlinear order the frame/coefficient time derivatives also matter.

Likewise, the previously failed `Z4cSommerfeld` control is not this experiment. It overwrites RHS of Theta, Khat, evolved Gamma and A componentwise using fixed speeds1 orsqrt2 and u/r, with second-order derivatives. It does not use Q=Gamma−Gamma_metric, curved wave speeds/advection, or preserve the same incoming/outgoing geometric characteristic decomposition. Its failure does not rule out a consistently derived radiative constraint BC.

## Actionable next derivation, not an implementation prescription

On the spherical continuum subsystem, the p condition is equivalently the following incoming physical-constraint data, with v=c−b:

H = −(2c/alpha) Theta' −chi q'−2chi q/r
    +(4sigma/alpha−2p v/(alpha r)) Theta,

m = −Theta'−c q'/(2alpha)
    +(sigma/alpha−p v/(2alpha r))q.

These follow directly by substituting the evolution equations into the wave boundary condition. They supply a concrete target for deriving a full-state lift. They are **not** a prescription to reset H, m, Theta or evolved metric variables.

A consistent full-state candidate must translate these incoming constraint data (and their transverse counterparts) into the four incoming constraint rows, preserve independent gauge/radiation boundary data, account for the actual face normal versus radial direction and tangential derivatives, retain the lower-order coefficient/source terms, and preserve the physical constraint surface. One cannot obtain it by merely appending −amplitude/r to all present characteristic targets or −u/r to selected evolved fields.

Before any production use, compare the lifted spherical/full-state IBVP with this reduced model; check exact background, outgoing/reflected constraint pulses, full physical H/M/Q, actual discrete eigenmodes, and mesh/MPI interfaces. No such full-state lift is established by this annulus experiment alone.
