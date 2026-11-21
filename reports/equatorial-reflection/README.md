# Equatorial photon-ring validation

The opt-in boundary is `mesh/ix2_bc=reflect` with `mesh/x2min=0` and the existing Cartoon axis at `x1min=0`. The refined family uses 24 bins in the prograde azimuthal direction cosine, 48 meridional sectors, fourth-order transport and source projection, and SSPRK(10,4). The baseline full-domain solver revision is `e430bbe7`. The full-domain reference deck is `inputs/boltzmann/refined_ring.athinput`. Use the identical elliptic initial-data file for each half/full pair.

## Reflection and diagnostics

In the evolved component order \( (\rho,z,\phi) \), the reflection matrix is `diag(1,-1,1)`. A Cartesian tensor component acquires one minus sign per axial index. The odd Z4c fields are gXY, gYZ, AXY, AYZ, GammaY, betaY and BY. Lapse, chi, trace variables and the other components are even. The odd radiation moments are Sxy, Syz and Sy. Syy, the vertical normal stress, is even.

The azimuthal tetrad leg is aligned with the axial Killing vector. Its tangential components and the meridional radial component keep their signs under reflection, while the vertical tetrad component reverses. The angular grid uses `mu = n_phi` and meridional angle psi, with `n_z = sqrt(1-mu²) sin(psi)`. Reflection keeps the cosine bin and reverses the sector index. Cell averages mirror about the intervening face. Geometry ghosts mirror about the equatorial vertex, where odd fields vanish. The axis half-turn sends psi to pi minus psi. The two permutations commute, including at the axis-equator corner. Reflection is applied to the restricted parent cache before interpolation and to fine ghosts afterward. The normal flux of each opposite-direction pair sums to zero, including when the positivity limiter acts.

The full-volume normalization is twice the represented hemisphere for radiation energy, axial J, escaped charges, metric work and integrated constraint squares. Nodal quadrature retains its endpoint weights. Unique equatorial nodes count once in the axis diagnostics. Centroids are ratios, and their numerator and denominator have the same reflection multiplicity. For the snapshot comparison the weight is Eulerian E times proper volume and the radius is `rho sqrt(g_phi_phi_cartesian)`. Vertical stress is the integrated covariant axial stress divided locally by the axial metric coefficient. The initial ADM charges come directly from the elliptic solution on the full spacetime. The SHA-256 hashes of the two files are recorded below.

| Initial spin | Initial ADM mass | Initial ADM J | SHA-256 of ring.adm |
|---|---:|---:|---|
| 0.9508 | 5.0000000 | 23.7700000 | `0c34be12b8873c4c5c6b22cb226f86de05f3856eab63ec388dc0858c6e900730` |
| 1.0600 | 5.0229322 | 26.7436388 | `5c53d9f2babff9474e2888d45a2ee852a76cf51d825a6badeebb0eb07353a0ff` |

## Convergence, MPI and restart

| Measurement | Result |
|---|---:|
| Composite divergence, L1 order from 96 to 192 cells | 4.103 |
| Reflux-interface face at the equator, maximum flux error at 192 cells | 1.62e-10 |
| Order of that junction measurement | 4.999 |
| Counterstreaming manufactured solution, maximum divergence error at 192 cells | 1.18e-9 |
| Order of that streaming measurement | 4.999 |
| Fixed-metric pulse relative energy residual, full / half | 1.07e-15 / 6.42e-16 |
| Maximum half versus upper full-domain ordinate difference | 1.39e-17 |
| Maximum half versus reflected lower full-domain ordinate difference | 7.45e-16 |
| Maximum change in active ordinates after the two-rank to one-rank restart | 0 |

The two-rank half-domain test created three leaves, migrated four block records and derefined three leaves. It restarted from the five-leaf checkpoint immediately after refinement. With backreaction active, at a common time of 0.25 code units the energy differences between half and full coverage were 3.13e-10 and 2.30e-11 at the two spatial resolutions. The difference between the *full* solutions at those resolutions was 8.06e-6. The corresponding half/full Hamiltonian constraint squared-integral differences were 6.21e-8 and 4.17e-9, versus 8.49e-2 from full-domain refinement. These are smooth-pulse tests, with angular resolution held fixed within each comparison.

The same AMR and two-rank to one-rank restart passed on two StellarAI B200 GPUs. The fixed-metric energy residual was 2.14e-16 for each domain. The largest active-ordinate difference was 1.39e-17 against the positive full-domain half and 4.10e-16 against the reflected negative half. The restarted ordinates agreed exactly. In the coupled CPU tests the maximum differences across all geometry components were 9.89e-8 and 2.01e-8 at the two resolutions, versus 4.23e-4 between the two full-domain grids at coincident vertices.

## Horizon finder

The existing finder already evolves an axisymmetric Legendre generating curve and has a direct meridional least-squares correction. The half-domain adapter folds the sampling locations and pulls back both tensors and metric derivatives. In the half-domain origin branch, even Legendre modes enforce zero equatorial slope. The direct correction uses a northern quadrature with mirror weights, then independently verifies both hemispheres, both regular poles and the equator. Displaced searches keep their signed centers and all modes. Two disjoint analytic Schwarzschild neighborhoods recover both components with nonzero local dipoles in the displaced parameterization. Horizon area and the axial Killing integral already cover a whole surface. Their normalization is independent of the domain used to store the geometry.

On analytic puncture exteriors with mass 1 and spins 0, 0.95 and 0.995, all 18 direct fits converged from spherical radii 0.8, 1.0 and 1.2 times the analytic radius using signed or folded samples. The largest dimensionless RMS residual was below 1e-8. All passed absolute checks of 2e-6 on area and 1e-6 on mass and J. For spin 0.995 the centered direct solve took roughly 0.15–0.22 ms folded and 0.20–0.38 ms full in one local CPU timing. The flat-space and deliberately defective polar-cap controls were rejected. Independent Kerr-Schild checks through spin 0.999 recovered negative/positive outgoing expansions on spheroids just inside/outside the analytic horizon, respectively.

### Archived native slices

Two distinct conditions are reported. The marginal-surface fit targets zero outgoing expansion. A trapped witness requires both future null expansions to be negative. The witness check samples 4,099 angles, including the axis limits and the equator, with cubic splines, local cubic and local quintic interpolation of the same native geometry.

| Initial spin | Last preceding checked slice without a verified witness, t/M | Earliest verified in the scan, t/M | Worst max(h Θ+) over three interpolants | Witness minimum radius in radial cells |
|---|---:|---:|---:|---:|
| 0.9508 | 14.33926 | 14.40700 | −1.56e-4 | 39.92 |
| 1.0054 | 33.0521 | 33.13747 | −4.83e-4 | 67.17 |

At the earlier 0.9508 witness on cycle 33480, an independent enclosing marginal fit gave area 655.03660, mass 4.9351511 and spin 0.99753989 on the full archived slice. Using only its northern data gave 655.03378, 4.9351503 and 0.99754019. Sampling the same saved geometry at twice the grid spacing changed area by 0.01462, mass by 4.27e-6 and spin by 1.55e-6. Over the next 0.06799 M of actual evolution the enclosing area, mass and spin changed by 8.16, 0.00289 and −0.000921. Use the measured geometric-sampling scale for same-frame charge comparisons and the one-frame changes when comparing independently detected onsets. The enclosing curve lies at least 0.137 code units outside the verified interior in every tested direction. The largest absolute marginal residual across interpolation methods was 1.31e-4 in units of h Θ+. These are same-slice reconstruction differences.

The 0.9508 result used a shallower target of −0.0005 and degree 48. The archived monitor had accepted an origin marginal surface by cycle 33600, while its fixed −0.006 target missed a trapped interior even at cycle 33720. A successful witness on cycle **33480** moves the detection two saved frames earlier than the previous verified frame 33720. The interval is a cadence-dependent detection bracket. The 1.0054 monitor also missed the strictly trapped interior on cycle 60360 while accepting an origin marginal fit. The reference files alongside this report contain coefficients, charge measurements, interpolation checks, and the unsuccessful neighboring trials.

On the previously reported detection frames, an independent degree-24 direct refit with an enclosing-contour seed gave areas 669.69 and 980.80. A spherical seed converged to smaller roots with areas 613.59 and 919.17. The full-meridian residual test accepted both branches. Comparing the *same archived enclosing contours* using the full metric and the reflected northern metric preserved spins 0.995799 and 0.801977 on those two frames. Report the contour and seed when comparing spins. A strictly interior witness has its own area and spin. For 1.0054, refitting the marginal curve at degree 48 gave spin 0.80220 at first verification. Repeating the same fit on coincident nodes at twice the spacing gave 0.80193. On the next slice the trial surface extended outside the contiguous finest patch and the coarser native-node fit gave 0.81715. The archive alongside this report retains these sampling checks. The archived in-evolution finder used 0.279 s and 0.344 s on those frames including its multi-branch search. The independent centered refits took 0.08–0.26 s per seed on a local CPU.

## Mesh experiment

Each layout uses 24-by-24-cell blocks. The production-boundary pair has radial outer face 36 and axial distance 18. The four factorial layouts have both outer faces at distance 36. Only the latter four isolate coarsening and reflection at comparable face distances. Coarsening changes root spacing from 0.25 to 0.5 and moves the inner and ring boxes one refinement level deeper. The mesh builder supplies the required 2:1 transitions. The embedded projected ring box is 5.10–6.90 in rho and ±0.90 axially. A leaf-by-leaf intersection check measured 0.0625 spacing in **both** directions over the whole box in every layout.

| Outer distances (rho, axial) | Full leaves at initialization | Half leaves at initialization |
|---|---:|---:|
| (36,18), original root | 60 | 30 |
| (36,36), original root | 96 | 48 |
| (36,36), coarse root | 54 | 27 |

At 30 M in the archived 1.0600 dispersal run, the centroid was 3.73220 M and the 90th-percentile circumferential radius was 4.69606 M. Only 16.58% of the represented Eulerian energy was on the finest level. The outer decile had energy-weighted mean spacing 0.25. Reproducing the outgoing radius and its resolution is a separate late-time check on a coarsened exterior.

The matched 0.9508 and 1.0600 endpoint jobs use the original physical outer distances and compare full and half coverage with the same new executable. They retain the 120-cycle finder and snapshot cadence. Compare physical timestamps as well as cycles. The coarsened layout remains experimental until the late-time centroid, vertical stress, fields, constraints, budgets and trapping checks pass. The ongoing bisection uses its original executable and deck.

### Two-B200 timing and field comparison

The first two-GPU pilot used two B200s on StellarAI node `sai-n02g3`, with SHA-256 `5a5afe424b37baec95a472e6702d36a4eb75c49cf48b8935b85e2bea11c1ede6`. The four layouts use the same pilot binary, input geometry and angular bins. This binary predates the final polar-limit and even-mode finder changes and the reduction of redundant fine-ghost fills. The separate eight-GPU experiment uses the updated executable.

| Layout, outer distances (36,36) | Mean active blocks | Steps to 0.2 M | Wall seconds | GPU-hours | Peak MiB per GPU |
|---|---:|---:|---:|---:|---:|
| Full, original exterior spacing | 114.95 | 569 | 801.73 | 0.44541 | 16,446 |
| Half, original exterior spacing | 57.47 | 569 | 395.04 | 0.21947 | 16,440 |

The measured symmetry saving was 50.73% through 0.2 M. Wall time includes initialization, evolution, diagnostics and the final checkpoint. Memory is the largest one-second device sample. Both layouts preallocate storage for 80 blocks per rank.

Across all six saved times, the maximum difference in the circumferential centroid was 4.45e-16 M and the integrated vertical stress differed by at most 6.94e-18. On the common spatial lattice, the largest absolute field differences in the ADM, radiation and constraint output groups were 5.96e-8, 4.29e-18 and 2.95e-7 respectively. At the endpoint the full-domain centroid was 2.39940 M. Changing the diagnostic integration to two-cell bins shifted this value by 0.00187 M, and the last saved interval changed it by 0.03476 M. Diagnostic quadrature and temporal sampling errors are distinct from the evolution errors measured in the smooth refinement test. Over the same interval the full/half energy-budget residuals were 2.24e-15 / 2.35e-15. Both axial angular-momentum budgets drifted by 9.09e-5 relative to the initial matter J. The largest full/half difference in the integrated Hamiltonian constraint squared was 2.50e-12.

The eight-GPU selection job compares elapsed GPU-hours at fixed outer distances, rank count, node and evolution interval. A candidate that wins that comparison is queued against the full original-spacing square mesh at both physical endpoints. The two original-boundary endpoint pairs independently test the effect of symmetry. Promotion requires the long-interval field, charge, outgoing-shell and horizon comparisons.

## Reproduction

Run these commands from the solver root with Kokkos and MPI available. The archive `reports/equatorial-reflection/initial-data.tar.gz` contains both elliptic data files and their metadata. To run the StellarAI scripts from another checkout, extract it into the directory assigned to `EQUATOR_DATA`.

```sh
cmake -S . -B build -DAthena_ENABLE_MPI=ON -DAthena_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target athena athena_boltzmann_fourth_amr athena_z4c_cartoon_m0_fastflow_unit_test
ctest --test-dir build -R '^(boltzmann_fourth_amr|boltzmann_equatorial_amr|athena.z4c_cartoon_m0_fastflow)$' --output-on-failure
python tst/unit/boltzmann/equatorial_amr.py build/src/athena /tmp/new-equator-amr-test
python tst/unit/boltzmann/equator/layout.py /tmp/input --square --half --coarse --check build/src/athena
```

The StellarAI launch scripts are in `tst/unit/boltzmann/equator`. Set `EQUATOR_DATA` to the directory containing the two calibrated run folders. `stellar.sh` launches the four layouts in one allocation. `performance.py` reads each completed directory, verifies the block census against the driver's block-cycle total, and reports physical-time-weighted block count, steps, elapsed seconds, allocated GPU-hours and sampled peak device memory. `compare.py` compares every ADM, stress and constraint field on a common meridional lattice, including parity in the reflected lower half, and records the physical time offset of each pair. `endpoint_review.py` combines the profiles, budget checks and trapped-surface review near the first online marginal detection. `witness.py` scans earlier archived slices against known shapes. The offline analysis requires NumPy, SciPy and h5py.
