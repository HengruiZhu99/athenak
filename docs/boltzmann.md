# Axisymmetric gray photon transport

## Running the constraint-solved torus

The `photon_torus` family imports a solved maximal Einstein–Vlasov slice and its
calibrated launch prescription. Set `problem/torus_file` to the IrisK
`IRISK_PHOTON_TORUS_V1` handoff. Positive physical axial spin selects
`boltzmann/sign=prograde`, and negative spin selects `retrograde`.
The optional `problem/torus_audit_prefix` saves native vertex constraints,
curvature and independent raw-ordinate moment checks by rank. A coupled run
and refinement driver are included. [Torus input](../inputs/boltzmann/photon_torus.athinput) · [Native qualification](../tst/unit/boltzmann/torus_native.py)

## Running the ring

From a checkout with the Kokkos submodule initialized, build the MPI executable and run the sample in its own directory.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAthena_ENABLE_MPI=ON
cmake --build build --target athena -j 4
mkdir -p run-ring
mpiexec -n 2 build/src/athena -i inputs/boltzmann/ring.athinput -d run-ring
```

The sample evolves a prograde ring from a flat spatial metric with zero extrinsic curvature. Its amplitude, radius, spatial width, signed direction cosine and angular width are input parameters. This is trial initial data. Its Hamiltonian and momentum constraint violations have continuum matter contributions from the first slice. The sample is suitable for exercising matter backreaction and the horizon finder.

The block `<boltzmann>` selects `nmu_half` uniform bands per signed hemisphere, an even `npsi` for the meridional polygon, and `sign = prograde`, `retrograde` or `both`. An even polygon supports the exact radial-reflection permutation at the axis. Set `backreaction = false` for test radiation. Setting `fixed_metric = true` additionally freezes the supplied geometry for prescribed-background transport. The coupled configuration uses SSP RK, vertex-centered Z4c, the half-plane Cartoon map, outflow at the radial exterior and outflow or periodicity in the axial direction. The AMR criterion `variable = boltzmann_E` uses Eulerian photon energy density. The angular mesh is identical on all spatial levels.

## Continuum and discrete equations

Use spatial coordinates $x^i=(\rho,z,\phi)$ and the ADM decomposition. Write $\gamma=C C^T$ with $C$ upper triangular and positive diagonal. Its last orthonormal vector points along the axial Killing field. The unit sky direction is $n=(s\cos\psi,s\sin\psi,\mu)$ with $s=\sqrt{1-\mu^2}$. Define $l_i=C_{ia}n^a$, $l^i=\gamma^{ij}l_j$, and $p_i=\epsilon l_i$. The null Hamiltonian $H=\alpha\epsilon-\beta^i p_i$ gives

$$
v^i=\alpha l^i-\beta^i,\quad
F_i=\dot p_i/\epsilon=-\partial_i\alpha+l_j\partial_i\beta^j
+\tfrac12\alpha l^j l^k\partial_i\gamma_{jk},\quad
q=\dot\epsilon/\epsilon=\alpha K_{ij}l^il^j-l^i\partial_i\alpha .
$$

Differentiating $p_i=\epsilon C_{ia}n^a$ yields
$\dot n=C^{-1}F-qn-C^{-1}(\partial_t+v^i\partial_i)C\,n$.
Use $\dot\mu=\dot n_3$ and $\dot\psi=(\cos\psi\,\dot n_2-\sin\psi\,\dot n_1)/s$ with one-based frame indices. The coframe derivative includes the cylindrical factor in the azimuthal leg and uses $\partial_t\gamma=-2\alpha K+\mathcal L_\beta\gamma$. Even a straight Minkowski ray changes sky direction in this frame. The cyclic coordinate gives $\dot p_\phi=0$ and $p_\phi=\epsilon\sqrt{\gamma_{\phi\phi}}\mu$, so each signed hemisphere has zero flux at its endpoints.

Integrating the collisionless Liouville equation over photon energy yields one gray energy intensity $I$ per solid angle. Its conservative variable $U=\sqrt{\det\gamma_{\mathrm{cyl}}} I=\rho\sqrt{\det\gamma_{\mathrm{Cartoon}}}I$ obeys

$$
\partial_t U+\partial_\rho(v^\rho U)+\partial_z(v^z U)+
\partial_\mu(\dot\mu U)+\partial_\psi(\dot\psi U)=qU .
$$

The spatial flux is exactly zero on the axis. Shared angular fluxes telescope in $d\Omega=d\mu\,d\psi$. MC-limited slopes in space and angle give nonnegative face states. The explicit step limit includes outgoing face rates and negative redshift. Coarse cells receive the area-averaged fine flux. Restriction averages the stored $U$ directly. Tensor-product quadratic subdivision reconstructs children with the parent mean and scales all four deviations together when positivity is at risk. Applying another radial weight during restriction would double count the cylindrical volume element.

For $E,S_i,S_{ij}$ integrate $1,n_a,n_a n_b$ analytically over the constant-intensity angular cells, then transform covectors with $C$. Matter supplies ten components in the native Cartoon Cartesian basis. At each stage, ghost cells at the target level supply the centered cell-to-vertex transfer. The native vertex topology plan synchronizes all copies of each shared source. At the axis the transverse stresses agree and components prohibited by rotational symmetry vanish.

## Diagnostics and regression tests

The user history records total photon energy, matter $J_z$, cumulative outward numerical fluxes of both, integrated geometric energy source, exterior matter $J_z$, and the minimum stored ordinate, in that order. A full revolution contributes $2\pi$. Compute the energy residual as $E(t)-E(0)+F_E(t)-W(t)$ and the angular-momentum residual as $J_z(t)-J_z(0)+F_J(t)$. The exterior current excludes cell centers inside the accepted harmonic horizon surface. The horizon file records finder acceptance, area and axial spin. A candidate qualifies for an extremality-gap measurement only after finder acceptance. Use binary `variable = tmunu` to inspect the ten cell-centered moments. Restart preserves the angular layout and cumulative budgets across rank counts.

```sh
cmake -S . -B build-tests -DAthena_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-tests --target athena_boltzmann_operators -j 4
ctest --test-dir build-tests -R '^boltzmann_operators$' --output-on-failure
python3 tst/unit/boltzmann/evolution.py build/src/athena --ranks 2 --turn --kerr --joint
```

The executable regression uses MPI, NumPy and h5py. It exercises refine/derefine, redistribution, restart with a different rank count, reflected rotation signs, and homogeneous radiation FLRW. The longer options compare radial flux and arrival time against straight-ray transport and compare a stationary Kerr beam against an independent Cartesian Hamiltonian backtrace with frequency shifts. The host operator test covers isolated spatial, latitude and polygon reconstruction, limiting at subdivision, cell-to-vertex stress transfer, FLRW redshift, and the Kerr radial potential and Killing charges.

## Measured transport accuracy

Double-precision CPU/MPI runs on Apple silicon used the serial Kokkos execution space on each rank. With the other directions isolated, doubling resolution from 48 to 96 gave $L^1$ orders of 2.02 in space, 2.02 in polygon angle, and 1.98 in signed direction cosine. The manufactured vertex transfer gave second-order convergence at the axis and in uniform interiors. The coarse–fine stencil passed the same order threshold. A two-rank regrid created and removed three blocks, moved data between ranks, and balanced energy to $1.2\times10^{-15}$ relative to the initial energy plus the accumulated boundary-flux magnitude. The single-rank restart reproduced all active ordinates within the regression tolerance. The coupled radiation FLRW interior had maximum relative energy-density error $1.39\times10^{-5}$ at $t=0.5$.

For the flat turning ring, the adaptive spatial mesh used 32 cells per root direction and one additional refinement level. Profiles compare signed radial momentum against straight rays through a cylinder near $\rho=3$ at 16 times through $t=8$. Arrival time is the centroid of inward radial flux over that interval. The angular-momentum flux profile uses the independently integrated radial Killing current. The retrograde pulse reproduced the reflected prograde history through the turn and matched its midpoint restart.

| Bands per hemisphere × sectors | Relative $J_z$ budget residual | Relative $L^1$ radial flux error | Absolute arrival time error | Relative $L^1$ radial $J_z$ flux error | Scaled energy residual |
|---|---:|---:|---:|---:|---:|
| 2 × 8 | 0.3664 | 0.5083 | 0.3617 | 0.5852 | $1.24\times10^{-15}$ |
| 4 × 12 | 0.1889 | 0.2862 | 0.1731 | 0.3822 | $1.56\times10^{-15}$ |
| 8 × 20 | 0.0776 | 0.1661 | 0.0807 | 0.2208 | $1.83\times10^{-15}$ |

An independent uniform-grid sequence refined both space and angle at fixed $t=4$. The $16^2$, $32^2$, and $64^2$ spatial meshes used $2\times8$, $4\times12$, and $8\times20$ ordinates respectively. Their relative $J_z$ residuals were 9.26%, 3.13%, and 1.02%. The largest scaled energy residual was $5.1\times10^{-15}$.

In stationary Kerr with $M=1$ and $a=0.6$, the detector at $(\rho,z)=(3.25,0.25)$ saw incoming momentum at $t\simeq 3$ and outgoing momentum at $t=9$ at the two higher angular resolutions. At $t=9$ the independent Hamiltonian backtrace gives $S_\rho=3.276\times10^{-4}$. The $4\times12$, $8\times20$, and $12\times32$ ordinates gave $-0.1425$, $0.7102$, and $1.4406$ in units of $10^{-4}$. Even the last run underestimates the escaping momentum by 56%. The sign and convergence establish a turning-beam check. Quantitative strong-field spin budgets require additional angular resolution.

## Coupled collapse trials

The trials matched the initial discrete photon energy $4.8378593$ and matter $J_z=9.9123436$ by adjusting the ring amplitude and angular tilt. All used two spatial AMR levels. The radius-6 runs ended at $t=16$, and the radius-10 run ended at $t=20$. The baseline root spacing was 0.5, with $3\times12$ prograde bins and horizon-finder $\ell_{\max}=4$. Each of the next three rows changes one resolution. The final row changes the launch radius and expands the outer domain.

| Trial | Initial H-norm2 / M-norm2 | Final H-norm2 / M-norm2 | Final exterior $J_z$ | Escaped $J_z$ | Relative $J_z$ residual | Accepted candidates |
|---|---|---|---:|---:|---:|---:|
| Radius 6 | 117.38 / 24.54 | 53.09 / 6.13 | 8.55538 | 0.00613 | 13.63% | 0 / 66 |
| Root spacing 0.25 | 122.90 / 25.70 | 53.87 / 6.14 | 8.56365 | 0.00706 | 13.53% | 0 / 135 |
| $6\times20$ ordinates | 117.38 / 25.13 | 51.00 / 6.05 | 9.39392 | 0.00314 | 5.20% | 0 / 171 |
| Finder $\ell_{\max}=8$ | 117.38 / 24.54 | 53.09 / 6.13 | 8.55538 | 0.00613 | 13.63% | 0 / 66 |
| Radius 10 | 70.43 / 15.24 | 21.06 / 2.19 | 9.45761 | 0.00413 | 4.55% | 0 / 99 |

The displayed constraint measures are the Z4c history columns. On the initially flat slice with zero extrinsic curvature, the Hamiltonian constraint is $-16\pi E$ and the momentum constraints are $-8\pi S_i$. The geometric ADM mass and angular momentum of that initial slice vanish. Increasing the launch radius reduced the initial violation at fixed photon charges. Halving the spatial spacing at radius 6 increased both initial squared norms. These data carry a continuum constraint mismatch. The maximum energy-budget residual across the five runs, including the integrated geometric source and normalized by initial photon energy, was $4.6\times10^{-15}$.

FastFlow rejected every candidate, including in the refined runs. The smallest recorded dimensionless direct residual was 0.90 across the five trials, with an acceptance tolerance of 0.03. The lapse reached about $2\times10^{-5}$ at radius 6. There is no accepted area or horizon spin from which to calculate $1-8\pi|J_H|/A$ or its uncertainty. Resolved subextremal formation remains an open science milestone.

To reproduce the five trials, create the parent output directory first. Each invocation writes its generated input, evolution and constraint histories, horizon candidate log, and restart files in a fresh directory.

```sh
mkdir -p results
python3 tst/unit/boltzmann/collapse.py build/src/athena results/base --ranks 2
python3 tst/unit/boltzmann/collapse.py build/src/athena results/spatial --zones-per-unit 4 --ranks 4
python3 tst/unit/boltzmann/collapse.py build/src/athena results/angular --nmu 6 --npsi 20 --ranks 4
python3 tst/unit/boltzmann/collapse.py build/src/athena results/finder --lmax 8 --ranks 2
python3 tst/unit/boltzmann/collapse.py build/src/athena results/far --radius 10 --ranks 3
```

## Fourth-order phase-space transport

On a uniform-level mesh set `boltzmann/transport_order=4` and
`time/integrator=ssprk104`. Both Boltzmann input decks expose the order.
The implementation needs four spatial ghosts, at least five cosine bins per
hemisphere and at least six even polygon sectors. A mesh may contain many
blocks distributed across MPI ranks. Coarse–fine reflux and the existing
positive subdivision rule remain available with order 2. A coarse donor would
need a budget shared by its fine subtending faces before the order-4 positivity
proof could be applied across a refinement boundary.

The cell unknown is the phase-volume average of densitized gray energy. For
each face the code reconstructs the average over that face with an upwind
quartic polynomial constrained by five normal cell averages. The positive
speed stencil at the face between cells −1 and 0 has weights
$(2,-13,47,27,-3)/60$ on cells −3 through +1. Negative speed uses its reflection.
Cosine stencils slide inside their own hemisphere. The polygon wraps periodically.
The axis and the edges of each signed hemisphere carry exactly zero normal flux.

For a smooth speed $a$, the face-average product through fourth order is

$$
\langle aU\rangle_f =
\left[a_f+\frac{1}{24}\sum_{t\perp f}\Delta_t^2 a\right]\langle U\rangle_f
+\frac{1}{48}\sum_{t\perp f}(a_{+t}-a_{-t})(U_{+t}-U_{-t})
+O(h^4).
$$

Here $\Delta_t^2 a=a_{+t}-2a_f+a_{-t}$ and the transverse values of $U$
in the covariance use the mean of the adjacent normal cells. For the redshift
source the sum includes all four dimensions. Each flux enters its adjacent
phase cells with opposite signs. This face-product quadrature follows the
multidimensional finite-volume treatment of the Vlasov equation.
[Banks and Hittinger](https://www.osti.gov/servlets/purl/991524)

The polygon-face characteristic contains an integrable pole in direction
cosine. Its cosine-bin integral uses four Gauss points in colatitude with
measure $\sin\theta\,d\theta$. Four adjacent bin averages determine a cubic
in colatitude. The code integrates the velocity times each basis function and
uses the resulting weights on the four normal reconstructions. Transverse
corrections in the two spatial directions apply to this integrated speed. In a
prescribed metric the velocity and quadrature weights are computed once.
A fourth-order vertex midpoint supplies the spatial face metric. A consistent
fourth-order metric halo and centered derivatives supply the angular
characteristics on both sides of an MPI interface.

For positivity, a donor divides its available energy among eight faces and a
possible redshift sink. Let $S$ be the sum of absolute normal speeds divided
by their cell widths across all eight faces plus the absolute source
coefficient. Each face receives a fraction $(|a_f|/(h_f S)+0.02)/1.18$ of the
positive Euler budget. The source uses the corresponding unit-width fraction.
These fractions sum to one. The limiter reserves one percent for rounding and
retains subnormal donor mass until incoming flux restores normal precision. A
single signed face flux is capped against the actual donor. The code applies
the resulting signed source and open-boundary fluxes to the history budgets
with the same RK coefficients as the ordinates. The limit is inactive in the
smooth convergence test. Limiting is expected in underresolved tails.
[Conservative flux limiting for positivity](https://arxiv.org/abs/1203.1540)

The ten-stage fourth-order SSP integrator uses two state registers and Euler
substeps of one sixth of the full timestep. The coupled Z4c and Boltzmann
updates advance on the same stages. The next matter CFL bound uses the accepted
metric, including after restart. The conserved energy balance is
$E+E_{\rm escaped}-W_{\rm metric}$. Axial angular momentum is diagnosed from
the finite ordinates and has a separately measured discretization error. Stress
output and vertex stress transfer retain the native finite-ordinate moment
convention. Spatial and angular transport orders are measured on smooth data.
[Ketcheson’s SSP methods](https://epubs.siam.org/doi/10.1137/07070485X)

The `boltzmann_fourth_accuracy` CTest evaluates the actual face kernels against
independent cell-integrated variable-coefficient fluxes. Its direction indices
0 through 3 correspond to radius, height, direction cosine and polygon angle.
Index 4 measures time order on a nonlinear ODE. The companion scripts
`torus_fv4_checks.py` and `qualify_fv4.py` check the physical energy budget,
signed-spin reflection, restart ordinates, CPU/GPU parity and multi-GPU timing.
