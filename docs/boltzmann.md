# Axisymmetric gray photon transport

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
