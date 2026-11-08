# N12 reference for the native AthenaK migration

This is a recovery record, not a claim that the native implementation reproduces the prior evolution. No production files were modified and no remote jobs were submitted. The remote campaign inputs are hash-verified against the local copies, the target snapshot hashes/header and diagnostics have been recovered, and the evolution build manifest has been audited. Remaining provenance gaps are listed at the end.

## Reference campaign and evidence

The preserved N12 evolution is `scri-ko01-20261106/campaign-constant20m` under `/lus/flare/projects/CompactBinaryMerger/hzhu` on Aurora. The later N24/N36 root, `scri-resolution-20261107`, contains the retrospective N12 FastFlow diagnostics at `n12-fastflow/output`. This association is explicit in the production `tst/hyperbolic_shift_3d/endpoint_policy/resolution_20m/analyze_n12.pbs`: it reads the N12 campaign's `central_smr20.in` and `horizon-analysis/snapshots.txt`.

The local comparison artifact contains 42 N12 diagnostic rows from t/M=0 through 20, including an exact recorded row at t/M=5.0009660239551499, step 3551. Its source path and SHA-256 are recorded in `reference-manifest.json`; `evidence/n12-fastflow.csv` preserves the complete CSV. The root agent's read-only remote inventory confirms that the local `central_smr_constant20.in` and `central_smr_constant.cfg` exactly match the remote campaign's `central_smr20.in` and configuration, both in text and SHA-256. Input SHA-256 is `d660b72deb0f3679cf63a0a28e1298e817b927e7906533eecef020d11f1a202f`; configuration SHA-256 is `9c786cc8c3312ddf93389a4b3f1f6e32921811a61f4d131ee53990fb80a1d363`.

The remote campaign state confirms a fresh first segment, job 9047352, from t/M=0 to 13.752304484349663 at step 9765; the target step 3551 lies in this segment. The second segment, job 9047631, reaches t/M=20 at step 14202. Both exit codes are zero and the state is complete. The second segment's fields/boundary/restart initial hashes exactly match the first segment's final hashes, as recorded by the campaign controller. This establishes campaign completion and restart continuity. The step-3551 binary hashes were separately verified and are recorded below.

The production source worktree inspected read-only is `/Users/hz0693/.codex/worktrees/scri-ko01/hyperboloidal`. Comparing every source hash in the recovered N12 `build-static-checkpoint-manifest.json` gives 763 matches and nine differences, with no missing local files. The matching files include the central configuration, initial/reference data, spherical policy, gauge, grid/SMR definitions, algebraic chart, and KO implementation. Later changes affect `coupled_mpi.cpp`, the endpoint run/control files, `hyperbolic_composite_device.hpp`, FastFlow, and tests. The detailed lists are in `evidence/n12-source-audit.json`; current local driver code must not be mistaken for the exact N12 binary source.

## Numerical problem

| Item | Local input/source evidence |
| --- | --- |
| Spacetime/initial data | Unboosted spherical wormhole, coordinate mass M=1/3; arbitrary 3D Cartesian perturbations subsequently evolve, without angular averaging. |
| Initial/reference map | `NativeWormhole(1/3, 2e-13, 20, 40, polynomial_c6, .97)`, through `CentralZ4cConfiguration` and `SphericalReference3D`. The compact slicing endpoint is .97. |
| Units | Reported t/M is `3*time`; lengths and horizon masses in M units also multiply coordinate values by 3. |
| Formulation | Configured hyperboloidal conformal Z4c with the selected spherical numerical policy, independent Cartesian algebraic chart, stationary conformal factor, separate scri trace. The recovered compile definitions confirm spherical policy, configured formulation, native wormhole, and the gauge options. |
| Gauge policy | `shift_speed_policy=constant`; stationary-reference outer gauge; outer lapse damping .25; inner telegrapher gauge. |
| Inner reference region | `inner_z4c_radius_M=20`, central layout `cartesian-shell`. This is a reference areal radius, not the compact Cartesian half-width. |
| KO strength | .1. The hash-matched physical KO8 operator uses radius-four mixed-level halos and a radial taper; the build does not define the alternate consistent-KO10 option. |
| Chi floor | Stored chi floor 1e-10, applied to finite RK candidates in both intermediate stages and accepted states; nonfinite values are not repaired by the floor. |
| Time integrator | Four-stage RK4 in the resident evolution engine. |
| Nominal timestep | Coordinate dt=0.00046944203735617445; dt/M=3*dt. |
| Run control | N12 defaults from omitted `<mesh>` section; tlim=6.666666666666667, nlim=-1, soft wall-time limit=3200 s. |
| Puncture tracker | Local N12 input has no tracker section; default is disabled. The retrospective N12 FastFlow call uses the origin. N24 input explicitly enables tracking, so its tracker setting must not be silently imported into this reference. |

The inner telegrapher constants in the hash-matched source are tau=.1*M, kappa=.2*M, shift_eta=2/M, kappa1=.01/M; the normalized shift coefficient is the square of the native isotropic scale. The configured outer gauge has bulk_mu=4, ratio=2, damping=1, restoring=4, kappa=.005, tau=.02, epsilon=1e-4, kappa1=4, kappa2=-.99. The decoded checkpoint identity gives native compact radius 0.64029066119359423, isotropic scale 0.10116863541976966, normalized shift coefficient 0.010235092792698271, omega transition end 0.73021799589519565, gauge/coefficient window [0.76618892977583619,0.87410173141775793], restoration end 0.94604359917903913, and conformal plateau .42.

The older descriptive configuration identity contains a stale `chi-floor=1e-8:.03` phrase. `ConfigureSphericalNumerics()` appends the authoritative selected-policy suffix `SP1-numerics-v1:ko=...:chi-floor=...`, and restart validation compares the entire identity. Do not infer the active floor from the earlier phrase alone.

## Grid to reproduce

The base Cartesian cube is [-.1,.1]^3, divided into 3^3 parents. The selector half-widths (.03,.03,.03) select the central parent; refinement replaces that whole parent with eight children. The refined cube is therefore [-1/30,1/30]^3, not [-.03,.03]^3. Each leaf has N=12 nominal cells per axis. Coarse spacing is 1/180 and fine spacing is 1/360.

There are 34 Cartesian leaves (26 coarse plus eight fine) and 72 mapped shell blocks (six faces times 12 radial layers), for 106 leaves total. Shell radii are `.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1`; the first shell starts at .08, and angular coordinates are equiangular. There is no shell angular subdivision. Four ghost cells are allocated beyond three evolved overlap cells. The core's internal block interfaces do not add evolved overlap; the outside of the Cartesian cube does. The first shell has angular overlap and outward radial overlap. The outermost shell uses endpoint-inclusive radial spacing .06/12=.005, with the scri endpoint at r=1 stored separately.

The source-derived active count is 229392 bulk points and 864 scri points. The calculation is

```
Cartesian: (3*N+6)^3 + 7*N^3
First shell: 6*(N+3)*(N+6)^2
Other shells: 66*N^3
Scri: 6*N^2
```

The recovered N12 `plan.json` and `mesh.json` confirm these counts and the 26 coarse/eight refined Cartesian leaves plus 72 shells. At N=24 the same formula gives 1629504 bulk points and 3456 scri points, agreeing with the prior performance workload. Native migration must represent the same overlap and sampling semantics explicitly; adopting equal native block weights is a scheduling choice, not permission to alter this numerical grid.

## Through-5M comparison target

The saved N12 sample first crossing 5M is step **3551**, t/M=**5.0009660239551499**. An evolution with tlim=5/3 clips its last step and does not produce this same state. For an equal-work regression through 5M, retain the nominal dt and compare after 3551 full steps, using the recorded snapshot time. If a strict 5M endpoint is required, the common-time comparison must account for the different final timestep.

The checkpoint header and `time.json` independently agree with this sample: coordinate time 1.6669886746517166 and exact nominal dt 0.00046944203735617445. The diagnostic time record reports CFL 0.23842793790802749, clock-stage maximum 0.37126366000113326, determinant residual 8.8817841970012523e-16 and tracefree residual 1.2434497875801753e-14. The preserved FastFlow CSV reports:

| Diagnostic | Value |
| --- | ---: |
| Horizon found | 1 |
| Compact mean horizon radius | 0.030549496850187822 |
| Areal horizon radius/M | 1.9999860425879357 |
| Irreducible horizon mass/M | 0.99999302129396783 |
| Expansion RMS | 1.4887607237254066e-5 |
| Full H RMS | 0.0073257648025702423 |
| Full M RMS | 0.0039666959397409406 |
| Full Theta RMS | 0.00063679744011741466 |
| Full Z RMS | 0.00074930798279592369 |
| Exterior H RMS | 0.0066650464364187327 |
| Exterior M RMS | 0.003785788884853934 |
| Exterior Theta RMS | 0.00062887813299441033 |
| Exterior Z RMS | 0.0007488764374473728 |

These are **RMS**, despite the short CSV names: `sqrt(sum(v*C^2)/sum(v))`. Momentum and Z use physical-metric contractions. L2 is recovered by multiplying the RMS by sqrt(region volume). The exterior region uses the complete harmonic FastFlow surface, not a coordinate-radius cut. The replay also compares full and fixed-outer-region norms to the original saved diagnostic values; the recovered result confirms zero maximum absolute difference across all 42 snapshots and successful horizons for all 42. Its CSV SHA-256 exactly matches the previously preserved local comparison CSV. This verifies the replay's diagnostic reproduction, not a native migration's state correctness. Compare the entire available history through the target, not only this final row.

The target chi-policy record confirms 56664 floor events in eight distinct cells, 56608 nonpositive raw candidates, zero nonfinite candidates, zero surface floor events, first floor at t/M=0.33095663633610284, and maximum correction 4.5490970865975129e-8. Minimum current chi at the target is 4.3253203421804223e-5; the minimum historical raw candidate is -4.5390970865975128e-8. Thus the floor is an active part of this reference, even though the target state itself is above the floor.

The inspected FastFlow wrapper defaults to lmax=12, ntheta=24, 300 flow iterations, alpha-beta constant .1, initial radius .02, mass tolerance 1e-10, expansion RMS tolerance 1e-3. It samples actual 3D chart data, including mixed-level halos, with six-point tensor interpolation. The recovered horizon confirms lmax=12, center [0,0,0], and a complete surface. The file named `n12-fastflow/build-manifest.json` actually describes `evolve-history`, not `fastflow_diagnostics`; it must not be used to claim exact replay binary provenance.

## Saved state and restart semantics

The target checkpoint is `/lus/flare/projects/CompactBinaryMerger/hzhu/scri-ko01-20261106/campaign-constant20m/segment-01/step-3551/restart.bin`. Its decoded magic is `HYP3D_MPI_SP1`. The 72-byte binary header stores magic[16], four int32 values (N=12, nf=31, kind=2, admitted=1), three uint64 values (bulk points=229392, surface scalars=26784, step=3551), then two float64 values (time, dt). In this exact binary, the length-prefixed 1235-byte formulation identity immediately follows; there are no intervening optional configuration records. Next comes the length-prefixed 6091-byte `endpoint-static-mesh-v1` identity. It binds N=12, ghost width=4, overlap=3, AMR flag=1, endpoint-grid flag=1, block count=106, and each block's face/layer/level/bounds.

Four point-major float64 arrays start at byte offset **7414**, in order: bulk values, bulk RK compensation, surface values, surface RK compensation. Each point has 31 stored slots; only 26 are independent RK variables (slots 6, 12, 28, 29, 30 are not independent). Total checkpoint length 114214390 bytes equals the decoded metadata plus these four arrays exactly. The reader checks exact dt, formulation/topology identity, lengths, finite contents, and absence of trailing data. This is a standalone research checkpoint, not a native AthenaK restart format. An adapter or coordinate/field comparison is required; directly feeding this binary to the native restart reader is not valid.

| Step-3551 artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| fields.bin | 56889216 | `7d37c940eea84a04dc471bb90178c3e2facf153f14382cd2a47cf1fba6a320be` |
| boundary.bin | 214272 | `343d4b04c6dc6456eb8bdf84bf1ddaa97fc1aee9531bdb1299c4d8573394e6ef` |
| restart.bin | 114214390 | `e43367df1a1a71a43d0e145157ca5a6810b6fb521b8cee7dbb2c1d594dc18cab` |

Snapshots also contain `fields.bin`, `boundary.bin`, `time.json`, and chi-policy diagnostics. Later tracker-enabled runs add a text sidecar `tracker.txt` with `HYP3D_TRACKER_V1 step time x y z`. The original N12 campaign's actual file inventory must establish which sidecars were written. The current production `run.cpp` adds online FastFlow/history diagnostics; those additions must not be retroactively attributed to the preserved N12 binary.

## Runtime provenance and remaining recovery

The preserved N12 launcher specifies 4 Aurora nodes, 48 MPI ranks, 12 GPU-tile ranks/node, one OpenMP thread/rank, `oneapi/release/2025.3.1`, composite Level Zero hierarchy, PCI device order, GPU-aware MPI, and `gpu_tile_compact.sh`. The actual plan confirms SYCL and 48 ranks. The CPU binding list is `1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100`. The recovered build manifest uses `icpx`, `-fp-model=precise`, `-O2`, `-ffp-contract=off`, SYCL target `spir64_gen`, device `12.60.7`, and MPI_DOUBLE. It reuses objects/libraries under `/home/hzhu/scri-resident-20M-20261105/build-100m` and records `evolve-static` SHA-256 `494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753`. The full compile/link argv are preserved in the evidence manifest.

The root agent checked Aurora qstat during this recovery; the captured `evidence/n12-remote-inventory.json` shows three held old jobs and no active jobs. This child could not directly query Aurora because its inherited sandbox blocks the SSH control socket, but independently verified the captured input text against its recorded hashes and the local files. Recheck the live queue before any future submission.

The campaign binary-hash list and snapshot list are now attached under `evidence/`. The evolve-static hash agrees with the recovered build manifest; the recorded shared-library hash is also preserved.

Remaining recovery gaps:

- Exact `fastflow_diagnostics` build provenance is not established by the misleadingly located `n12-fastflow/build-manifest.json`; that file's compile/link commands build `evolve-history`.
- Large state arrays were hashed remotely but not downloaded or decoded in this recovery. A full-state native comparison must read the actual arrays and verify these hashes first.

The native implementation should first match initial physical fields and mesh sampling, then the full N12 history and scri/constraint diagnostics through step 3551. Numerical tolerances must be established explicitly for the actual native arithmetic and interpolation changes; this recovery record does not authorize dropping a field, changing KO/floor/dt, or accepting only horizon-mass agreement.
