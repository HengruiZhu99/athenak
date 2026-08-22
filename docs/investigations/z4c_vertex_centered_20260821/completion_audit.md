# Goal Mode completion audit

This audit maps the explicit definition-of-done items in the governing Goal
Mode prompt to current authoritative evidence.  `PROVED` means the requirement
is directly covered at the requested scope.  `PARTIAL` means a narrower scope
passes but the full requirement is not established.  `CONTRADICTED` means
current evidence demonstrates failure.  `PENDING` means the required action
has not yet completed.

## Implementation and regression requirements

| requirement | status | authoritative evidence |
|---|---|---|
| `grid_centering=cell` remains default and preserves CC | `PROVED` | `baseline/README.md`; exact history, timestep, and four numerical-payload hashes |
| VC uses true `N+1` active nodal ranges | `PROVED` | `z4c_vc_layout`, `z4c_vc_coordinates`; `docs/design/z4c_vertex_centered_design.md` |
| collapsed directions remain one plane | `PROVED` | `z4c_vc_collapsed_dimension`; Cartesian/Cartoon derivative tests |
| Z4c-native state and derived arrays use selected geometry | `PROVED` | compile-time grid-policy dispatch and native ADM/constraint/curvature tests; changed-file inventory |
| no hot-kernel runtime centering branch | `PROVED` | host-selected templated dispatch; source/static policy tests |
| deterministic same-level shared-node synchronization | `PROVED` on host/CUDA | topology, boundary-index, static multilevel, MPI history, and zero duplicate mismatch evidence |
| coincident coarse/fine nodes inject exactly | `PROVED` on host/CUDA | `z4c_vc_amr_transfer`, dynamic AMR, and boundary-index tests |
| hanging nodes use symmetric interpolation and are not independent | `PROVED` on host/CUDA | AMR transfer/topology/task-order tests |
| MPI face/edge/corner cases pass | `PROVED` on host/CUDA | topology/boundary tests, rank-change restart, MPI AMR history |
| dynamic refine/derefine/load balance pass | `CONTRADICTED` at full backend scope | host/CUDA pass; exact-final SYCL faults in the first dynamic refine test after its first accepted transaction |
| VC restart across rank counts | `PROVED` on host/CUDA | `z4c_vc_restart`, `z4c_vc_restart_rank_change` |
| output/history identify and integrate nodal data | `PROVED` on host/CUDA | output/history quadrature tests and nodal VTK/PDF contracts |
| active Cartoon `rho=0` node uses analytic limits | `PROVED` | axis scalar/vector/tensor/RHS tests and bounded axis telemetry |
| axis parity/state/RHS regularity | `PROVED` | O2/O4/O6 manufactured axis and production derivative tests |
| native VC ADM/constraints/curvature | `PROVED` | native diagnostic code paths and fixed-grid constraint output |
| explicit VC-to-CC ADM adapter | `PROVED` | `z4c_vc_to_cc_adm` and consumer fail-closed policy |
| nonvacuum VC qualified or rejected | `PROVED` as rejection | `EXPLICITLY_UNSUPPORTED`; construction fails before unsupported allocation/use |

## Backend, integration, and handoff requirements

| requirement | status | authoritative evidence |
|---|---|---|
| exact-final host tests | `PROVED` | 28/28 fast selected tests; explicit O4 wave run gives orders 3.994--4.001 |
| complete Phase-16 stable-name test matrix | `NOT_ESTABLISHED` | the 30 selected tests numerically combine many requested checks under broader names; requested aliases such as axis-AMR, per-rank shared-node, per-order prolongation, Fourier transfer, native ADM, CC regression, and especially a production `z4c_vc_gauge_wave` test are absent |
| CUDA device tests | `PARTIAL` | pre-portability source passes 30/30; exact-final CUDA build passes but its one-GPU test is pending renewed NERSC credentials |
| SYCL/PVC device tests | `CONTRADICTED` | jobs 8774394 and 8774420 pass tests 1--16 then reproduce the same dynamic-refinement GPU page fault |
| Gate E bounded VC fixed-grid Brill | `PROVED` | N128/N256/N512 reach `t=0.5 M`, field orders 3.314--4.081, duplicate mismatch exactly zero |
| Gate F CC-versus-VC common-tree discriminator | `PROVED` as a negative discriminator | six exact replay events at zero ULP; resolution-worsening post-RK SPD failure and negative constraint orders; no qualification claim |
| backend-comparable memory/timing record | `NOT_ESTABLISHED` | layout/buffer geometry is tested; comparable SYCL synchronization/RHS/AMR/restart/output timing is unavailable because the device gate fails |
| final report/prompt/manifests committed and pushed | `PENDING` | owned files are assembled locally; final CUDA disposition and checksum freeze precede the documentation commit |

## Completion conclusion

The implementation is substantial and the CC, host, MPI, fixed-grid VC, axis,
restart, output, and bounded-discriminator evidence is reviewable.  The Goal
Mode definition of done is nevertheless **not satisfied**: exact-final SYCL
dynamic AMR is reproducibly broken, the complete requested stable-name matrix
is absent, exact-final CUDA execution is pending, and the final evidence
commit/push has not yet occurred.  The formal overall verdict must remain
`VC_Z4C_NOT_QUALIFIED`.
