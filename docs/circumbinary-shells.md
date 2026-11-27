# Momentum budgets outside the component horizons

The diagnostics branch includes the fork's circular corotating frame and cyclic zoom revision `f48f3126`. Prescribed inertial binaries can use the same surfaces and shell diagnostics with dynamic ADM. Two-center binary zoom uses a stationary prescribed ADM background in the corotating frame. The input for the matched runs initializes Bondi inflow immediately. [Binary input](../tst/diagnostics/binary_shell.athinput) · [Frame and zoom setup](circumbinary_cyclic_zoom.md)

## Surfaces and volumes

These settings belong in the existing `<problem>` block. An `hst` output with `user_hist=true` in `<problem>` writes the columns. Use `data_format = %24.16e` for balances.

```ini
flux_modeled_horizons = true
flux_ks_radius1 = 1.4
flux_ks_radius2 = 1.2
flux_shell_budget = true
flux_rsurf_inner = 3.75
flux_rsurf_outer = 4.75
flux_dr_surf = 0.5
shell_subcells = 8
```

`BH1_Kerr` and `BH2_Kerr` follow the analytic, boosted component Kerr–Schild surfaces at $r_+=m(1+\sqrt{1-|\chi|^2})$. A positive `flux_ks_radius1/2` adds a surface at that fixed Kerr–Schild radius in simulation length units. Zero disables the extra surface. Read the literal `BH*_KS*` label from the history header. In the hole's rest coordinates the level set is an oblate spheroid with polar semiaxis $r$ and equatorial semiaxis $\sqrt{r^2+m^2|\chi|^2}$. The equal-time boost gives the Cartesian quadric, including for oblique boosts. Both quadrature points and the normal velocity use this same quadric. These are prescribed component horizons. A numerical apparent horizon of the combined binary spacetime requires a separate horizon finder. [Boosted Kerr geometries](https://arxiv.org/abs/gr-qc/0002076)

Each surface has mass, fluid and electromagnetic energy, Cartesian linear and angular momentum fluxes, proper area, magnetic flux and coverage. The modeled horizons also have accepted-step rest-mass integrals. The optional component radii are independent sampling surfaces. In the example the three volumes are the interior of $R=3.75$ outside both component horizons, $3.75<r<4.25$, and $4.25<r<4.75$. The first sphere must enclose both disjoint holes. Common outer interfaces cancel when adding neighboring budgets. A single-hole run subtracts one horizon. The existing Euclidean H1/H2 transfer volumes remain available with `flux_budget`.

## What the history measures

For coordinate momentum $U_i=\sqrt\gamma S_i$, the densitized flux and source are

$$
F^j{}_i=\sqrt\gamma (\alpha W^j{}_i-\beta^j S_i),\qquad
G_i=\sqrt\gamma\left[-E\partial_i\alpha+S_j\partial_i\beta^j
+\tfrac12\alpha W^{jk}\partial_i\gamma_{jk}\right].
$$

The inward rate through a boundary moving at $w^j$ is $-\int(F^j{}_i-w^jU_i)\,d\Sigma_j$, with the **coordinate** oriented surface element in this densitized expression. Shell inflow equals outer inward rate minus each inner inward rate. The implementation uses the equivalent undensitized currents with proper surface covectors. For $J_z=xU_y-yU_x$ the volume source is $xG_y-yG_x+F^x{}_y-F^y{}_x$. The last two terms matter in curvilinear spacetime fields even on a Cartesian mesh. The reported quantities are gas plus electromagnetic field with respect to the stated coordinate generators. [Valencia GRMHD equations](https://doi.org/10.1088/1361-6382%2Facf2d9)

A shell integral sums stored conserved densities over coordinate volume. Cut cells use analytic vertical intersections with the bounding sphere and horizon quadrics. `shell_subcells` chooses 1–8 transverse midpoint samples per dimension, default 8. Bounding spheres identify wholly inside/outside cells before quadrature. Angular momentum uses first moments of the included fraction. Its source integrates the affine rotational shift at that same centroid. The metric and evolved state follow the solver's native sampling and derivative stencil.

For each of `Px`, `Py`, `Pz` and `Jz`, `shell_R*_..._stored`, `_source` and `_boundary_in` are instantaneous. The `_interval_boundary`, `_interval_source` and `_interval_other` columns integrate one accepted RK1/2/3 step. The last measures actual non-coordinate changes to the evolved conserved state. The residual gives storage change minus these three signed terms before remapping. Use cycle-cadence history to combine multiple intervals. The separate `_grid_jump_partial_cumulative` sums measured zero-duration zoom and AMR changes. `grid_jump_unavailable_events` counts transitions whose two endpoints could not be qualified.

Invalid metric or primitive data, nonfinite contributions, contaminated interpolation support, and overlap with a frozen zoom region affect qualification. The histories give sampled/rejected coordinate volume and a validity bit. A whole-volume quantity or interval requiring missing data is NaN. Explicit `_partial_stored` and `_partial_source` sum the admissible cells with their original weights. After restarting from a checkpoint with rejected volume, only whole-shell rejection is known until recovery refreshes the flags. A temporal outlier screen would suppress resolved physical transients and is unnecessary for this policy.

The [extended refinement and MPI audit](circumbinary-convergence.md) tests ten spatial resolutions and the production metric-update path.

## Accretion and angular momentum from history alone

`flux_horizon_transport` accumulates eight quantities per modeled component horizon. It defaults on when the requested columns fit in the history schema. Set it explicitly to require the complete ledger. Each Runge–Kutta stage reuses the surface flux already evaluated for net rest mass. The ledger commits a step only when all stages and its endpoint qualify. The sign is positive into the moving surface.

| Quantity prefix | Meaning of its cumulative change |
|---|---|
| `M0_in`, `M0_out` | Gross inward and outward baryon mass, each positive. Their difference is the net baryon delivery over the same qualified steps. |
| `E_inertial_f`, `E_inertial_em` | Fluid energy (including rest mass) and electromagnetic energy with respect to inertial coordinate time. In the rotating frame the inward current is $\dot E_{\rm rot}+\Omega_{\rm frame}\dot J_z$. |
| `Jz_orbit_f`, `Jz_orbit_em` | Angular momentum carried by the inward linear momentum, with lever arm from the barycenter to the hole center $\mathbf C$, $(\mathbf C\times\dot{\mathbf P})_z$. |
| `Jz_center_f`, `Jz_center_em` | Remaining inward axial angular momentum about the hole center. Adding the centered and orbital terms recovers the inward barycentric angular momentum for each stress component. |

For example, `E_inertial_em_qualified_cumulative_BH1_Kerr` gives the signed electromagnetic contribution integrated since `transport_epoch_start_BH1_Kerr`. To measure a mean rate, difference two cumulative values in the same epoch and divide by the change in `transport_covered_duration_BH1_Kerr`. Report the covered fraction of the elapsed interval alongside the mean. A frozen or underresolved horizon increases `transport_unavailable_duration_BH1_Kerr` and leaves the totals fixed. Using sparse history output preserves the stage-integrated measurement. Restarting an older checkpoint preserves its accumulated net baryon mass and starts the new channels at the restart time.

For a fixed circular binary, the same columns give the energy delivered with respect to the instantaneous component boost on the inertial slice. For each stress component and each hole $h$,

$$
\Delta E_{\rm boost,h}=\gamma_h\left(\Delta E_{\rm inertial,h}-
  \Omega_{\rm bin}\,\Delta J_{z,\rm orbit,h}\right),\qquad
\gamma_h=[1-(\Omega_{\rm bin} R_h)^2]^{-1/2}.
$$

Here $R_h$ is the barycentric orbital radius, $\Omega_{\rm bin}=a^{-3/2}$ in the prescribed $M=1$ orbit, and $R_1=a q/(1+q)$, $R_2=a/(1+q)$ with $q=m_2/m_1$. This follows from $\mathbf v_h\cdot\dot{\mathbf P}=\Omega_{\rm bin}(\mathbf C\times\dot{\mathbf P})_z$. It is a matter-plus-field accretion proxy in the simulation's density units. The simulation prescribes the component masses and spins. A geometric horizon mass or spin measurement requires separate horizon geometry and gravitational flux information. [Dynamical horizon balance laws](https://doi.org/10.1103/PhysRevLett.89.261101)

### Material between a 5M transfer sphere and its horizon

To measure how much gas and angular momentum accumulate between each 5M sphere and its component horizon, add these settings to `<problem>` and enable user history. For an older checkpoint, an input overlay passed with `-i` can introduce the new parameters.

```ini
flux_modeled_horizons = true
flux_horizon_transport = true
flux_horizon1 = true
flux_horizon2 = true
flux_radius1 = 5
flux_radius2 = 5
flux_transfer_inventory = true
```

At history cadence, `inventory_H1_M0_stored` integrates conserved baryon density between H1 and its boosted Kerr surface. `inventory_H1_Jz_stored` integrates barycentric $xS_y-yS_x$ with densitized momenta and cut-cell first moments. H2 has equivalent columns. Each transfer sphere must enclose its horizon and fit inside the mesh, and the two spheres must be disjoint. The reduction visits only blocks overlapping each sphere's bounding box. The inventory defaults off and requires no volumetric output.

`M0_boundary_in`, `Jz_f_boundary_in`, and `Jz_em_boundary_in` with the same inventory prefix are instantaneous outer-minus-horizon inward rates. A positive `M0_boundary_in` means that more baryons are entering through the transfer sphere than are crossing the horizon at that instant. Gravity also exerts a volume torque, and floors or mesh changes can affect storage. Use the shell source and remap histories above when evaluating a full balance. `sampled_volume`, `rejected_volume`, and `storage_valid` qualify the stored quantities. A rejected cell or an immediate restart sample with unknown reset provenance makes the whole storage measurement unavailable. Each boundary rate retains the qualification of both surface integrals.

The unequal-mass control compares the inventory to an independent $24^3$-voxel integration of each intersecting cell in a conserved-variable dump. With eight transverse samples per cell, maximum differences over both holes were 0.0066% for stored baryon mass, 0.0204% for stored angular momentum and 0.0113% against the analytic sphere-minus-ellipsoid volume. On refining the evolution mesh from $96^3$ to $128^3$, stored mass changed by at most 0.063% and stored angular momentum by 0.158%. Net mass supply through the two boundaries changed by 2.94% on the primary and 0.102% on the secondary. The difference of two surface fluxes can converge nonmonotonically. These are short controls with magnetized inflow. [Regression tests and mesh settings](../tst/diagnostics/test_lifecycle.py)

## Verification on StellarAI

The manufactured pressure test applies the divergence theorem independently to a sphere containing two unequal, off-center, obliquely boosted, tilted Kerr ellipsoids. Its nonzero $P_x,P_y,P_z,J_z$ fluxes agree with analytic volume and first-moment results to relative tolerances of $10^{-4}$ on the surface and $8\times10^{-4}$ in the volume. A second check inverts the rest-frame Cartesian Kerr radius at sampled points on each quadric. It covers the horizon, two other radii and three spin/boost configurations with a $5\times10^{-14}$ absolute tolerance. The field and area quadratures have their own discretization errors. [Production-helper tests](../tst/diagnostics/helpers.cpp)

The evolved control starts identical unequal-mass binaries at $q=0.75$, separation 4 and spins 0.3 and 0.2. It evolves magnetized inflow with RK2 to $t=0.02$ on $48^3$, $64^3$ and $96^3$ grids, comparing the same physical endpoint. Rotate inertial momentum into the corotating basis and subtract $\boldsymbol{\Omega}\times\mathbf{P}$ from the transformed momentum source. The barycentric axial angular momentum uses the same generator in both frames. The initial translation and torque source identities pass with $5\times10^{-10}$ absolute tolerance. The moving inertial background follows AthenaK's existing mesh-time update within a step, so this short test establishes finite-resolution agreement. [Matched-frame check](../tst/diagnostics/check_binary_shells.py) · [Resolution and binary zoom test](../tst/diagnostics/shell_resolution_zoom.py)

The closure fraction below is $\sum |\mathrm{residual}|/\sum(|\mathrm{boundary}|+|\mathrm{source}|+|\mathrm{other}|)$ over accepted steps. For reflection-symmetric $P_z$, the absolute residual rate is the informative measure. The manufactured case exercises a nonzero vertical flux.

| Frame | Outer radius | Grid | $P_x$ | $P_y$ | $J_z$ |
|---|---:|---:|---:|---:|---:|
| Corotating | 3.75 | 48³ | 1.578% | 0.760% | 1.376% |
| Corotating | 3.75 | 96³ | 0.112% | 0.308% | 0.484% |
| Inertial | 3.75 | 48³ | 1.086% | 2.267% | 3.065% |
| Inertial | 3.75 | 96³ | 0.414% | 0.753% | 1.058% |
| Corotating | 4.25 | 96³ | 0.026% | 0.392% | 0.283% |
| Inertial | 4.25 | 96³ | 0.056% | 0.336% | 0.257% |
| Corotating | 4.75 | 96³ | 0.018% | 0.284% | 0.278% |
| Inertial | 4.75 | 96³ | 0.028% | 0.214% | 0.240% |

At $96^3$ the largest absolute $P_z$ residual rate across these shells was below $2\times10^{-12}$. The largest frame difference in the two horizons' fluid and EM momentum-vector fluxes was 0.174%, and for axial angular momentum flux it was 0.354%. The central shell's stored momentum vector differed by 0.0051% between frames. The weak outer axial source is sensitive to cancellation. For example, its absolute frame difference at $R=4.75$ is about $4\times10^{-5}$ against inward transport of about 0.093. The small puncture masks are intentionally well inside the resolved component surfaces. These runs test exterior diagnostics over a short interval, with finite grid, quadrature and metric-time errors.

The two-center zoom control uses a stationary corotating equal-mass binary. It requires both horizons to acquire the frozen-support flag, then return to qualified coverage. The central volume reports its rejected fraction, suppresses the whole-shell integral and keeps a finite partial integral while the region is frozen. The CPU suite covers old checkpoints, one-to-two-rank restart, inherited shell rejection, repeated checkpoints, cadence and remap accounting. Binary torque output was also compared cell by cell for a plane, an intersecting line, a selected mesh block and ghost zones. The single-block regression caught an output record-count error that could append uninitialized records after the selected block.

The final B200 campaign was Slurm job 61229. Across all builds, failed trials and verification runs, Slurm recorded 1,847 allocated GPU-node seconds (0.513 node-hours). The CPU suite passed all 21 tests, including the original cyclic-zoom branch's pre-diagnostics checkpoint. In each frame, 249 matched CPU/GPU history columns agreed to a maximum absolute difference of $7.2\times10^{-11}$. [Measurements, timings and allocation accounting](circumbinary-shells-b200.json)

| B200 measurement | Reference | Updated |
|---|---:|---:|
| Eight cycles with common history diagnostics and a torque slice each cycle, median of three | 0.4640 s | 0.3750 s |
| Synchronized torque kernel for the plane, median after first output | 47.26 µs | 26.91 µs |
| Synchronized torque kernel for the full volume | 49.98 µs | 49.77 µs |

The selected plane took 43% less kernel time, and the eight-cycle workload took 19% less time. Full-volume kernel timings were effectively tied. The three saved torque components matched the reference exactly at binary output precision. Adding the two fixed KS surfaces brought the median to 0.3928 s. Enabling the three shell budgets as well brought it to 0.4320 s, an additional 10.0% over those surfaces at this cadence. These timings include evolution and output on a $48^3$ mesh. The dedicated kernel timings include the synchronization performed by the probe.

## Reproducing the runs

The CPU checks used the repository's pinned Kokkos, double precision and MPI. The B200 runs used double precision on one GPU per allocation. StellarAI's CUDA 13.2 toolchain requires the newer Kokkos snapshot used by the gravitational-collapse build. The archive was `stellarai-kokkos-6739bc623.tar.gz` with SHA-256 `82bfc22185bdf2ba29ce70288282e98ad46176f5770032dd19e19d41bfa2493e`. The following commands assume a clean scratch export whose empty `kokkos/` directory has been populated from that archive. In the wrapper shipped with the archive, set the default architecture to `sm_100` if it still says `sm_70`.

```sh
module load cudatoolkit/13.2 hpcx-latest/hpcx anaconda3/2024.10
cmake -S . -B build -DPROBLEM=dynbbh -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" -DAthena_ENABLE_MPI=OFF \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_BLACKWELL100=ON -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF
cmake --build build --parallel 24
python3 tst/diagnostics/check_binary_shells.py build/src/athena paired
python3 tst/diagnostics/shell_resolution_zoom.py build/src/athena physical
mkdir benchmark
python3 tst/diagnostics/bench_diagnostics_gpu.py "$BASELINE_EXE" build/src/athena benchmark
```

The performance reference was `bac1f12d` with the two NVCC compilation fixes and the full-overwrite DualView synchronization fix required to run repeated surface histories. Both timing executables used the same CUDA compiler, Kokkos, input and output cadence. Cost comparisons against the reference use user history only because the original built-in MHD history could not launch its oversized GPU reduction. The final paired and zoom controls also exercised standard MHD history. The clean CUDA build and matched runs started from a fresh source export. Slurm requests used the `gpu` partition, `gpu-test` QoS, one node, and one B200.
