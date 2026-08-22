# Native VC implementation inventory

Base: `6daa774d7451dbc5f7cac640c6e32a6fd11de7f9`

Production implementation tip: `5d37b5e5c278ac4a1afd52f9553dee6ffed48d0e`

## Atomic commit sequence

1. `1b1f5327` docs: define Z4c vertex-centered architecture
2. `73595956` z4c: add centering enum, immutable layout and coordinates
3. `b762a3d7` z4c: allocate and dispatch native CC/VC storage
4. `6b9e4711` z4c: implement VC Cartesian and Cartoon derivatives
5. `21a31199` z4c: dispatch native kernels by centering
6. `a17f2276` z4c: enforce evolved Cartoon vertex-axis regularity
7. `97be8d58` bvals: define canonical vertex topology keys and roles
8. `b3bb490c` z4c: build vertex role map from mesh topology
9. `859ac399` z4c: add deterministic shared vertex communication core
10. `266d95ac` z4c: add native vertex AMR and uniform boundary dispatch
11. `5a5a738a` z4c: add native vertex ADM conversion and smooth carrier
12. `70f8e06e` z4c: add native vertex coarse-fine communication
13. `798e3ecc` z4c: qualify dynamic vertex AMR lifecycle
14. `47523e0a` io: add native vertex restart contract
15. `391329ff` io: add native vertex output and history contracts
16. `0f65f63f` mesh: bind AMR history to Z4c centering
17. `314243eb` z4c: qualify native vertex diagnostic consumers
18. `0e62f70b` z4c: import direct Brill data on native vertex grid
19. `8c698023` z4c: qualify vertex smooth-wave interfaces
20. `4396cf95` z4c: make vertex dispatch CUDA capture safe
21. `449dac7b` z4c: preserve vertex axis regularity under KO dissipation
22. `684402ae` z4c: classify only the physical Cartoon vertex axis
23. `5d37b5e5` mesh: capture native vertex AMR device views portably

## Changed production/build files

### Build and registration

- `CMakeLists.txt`
- `src/CMakeLists.txt`

### Boundary communication and topology

- `src/bvals/buffs_vc.cpp`
- `src/bvals/bvals.hpp`
- `src/bvals/bvals_vc.cpp`
- `src/bvals/physics/z4c_bcs.cpp`
- `src/bvals/vertex_boundary_indices.hpp`
- `src/bvals/vertex_topology.hpp`

### Coordinates, driver, mesh, and AMR history

- `src/coordinates/adm.cpp`
- `src/coordinates/cell_locations.hpp`
- `src/driver/driver.cpp`
- `src/mesh/amr_history.cpp`
- `src/mesh/amr_history.hpp`
- `src/mesh/amr_history_format.cpp`
- `src/mesh/amr_history_format.hpp`
- `src/mesh/load_balance.cpp`
- `src/mesh/mesh_refinement.cpp`
- `src/mesh/mesh_refinement.hpp`
- `src/mesh/mesh_refinement_vc.cpp`
- `src/mesh/meshblock_pack.cpp`
- `src/mesh/vertex_amr.hpp`

### Output and restart plumbing

- `src/outputs/basetype_output.cpp`
- `src/outputs/derived_variables.cpp`
- `src/outputs/formatted_table.cpp`
- `src/outputs/history.cpp`
- `src/outputs/outputs.hpp`
- `src/outputs/restart.cpp`
- `src/outputs/vtk_mesh.cpp`

### Problem generators and task registration

- `src/pgen/pgen.cpp`
- `src/pgen/pgen.hpp`
- `src/pgen/tests/z4c_linear_wave.cpp`
- `src/pgen/tests/z4c_vc_minkowski.cpp`
- `src/pgen/z4c/kerr_puncture.cpp`
- `src/pgen/z4c_irisk_xcts.cpp`
- `src/tasklist/numerical_relativity.hpp`

### Z4c implementation

- `src/z4c/cartoon_axis_boundary.hpp`
- `src/z4c/cartoon_derivatives.hpp`
- `src/z4c/cartoon_m0_fastflow.cpp`
- `src/z4c/cartoon_meridional_sampler.hpp`
- `src/z4c/cartoon_vertex_axis.hpp`
- `src/z4c/curvature_diagnostics.cpp`
- `src/z4c/vertex_to_cell.hpp`
- `src/z4c/weyl_tetrad.hpp`
- `src/z4c/z4c.cpp`
- `src/z4c/z4c.hpp`
- `src/z4c/z4c_Sbc.cpp`
- `src/z4c/z4c_adm.cpp`
- `src/z4c/z4c_amr.cpp`
- `src/z4c/z4c_amr.hpp`
- `src/z4c/z4c_calcrhs.cpp`
- `src/z4c/z4c_calculate_weyl_scalars.cpp`
- `src/z4c/z4c_grid.hpp`
- `src/z4c/z4c_history_quadrature.hpp`
- `src/z4c/z4c_newdt.cpp`
- `src/z4c/z4c_restart.cpp`
- `src/z4c/z4c_restart.hpp`
- `src/z4c/z4c_symmetry.cpp`
- `src/z4c/z4c_symmetry.hpp`
- `src/z4c/z4c_tasks.cpp`
- `src/z4c/z4c_update.cpp`
- `src/z4c/z4c_vertex_topology.cpp`
- `src/z4c/z4c_vertex_topology.hpp`

## Added/changed test inputs and tests

### Inputs

- `tst/inputs/z4c_vc_brill_direct_fixed.athinput`
- `tst/inputs/z4c_vc_linear_wave_o4.athinput`
- `tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput`
- `tst/inputs/z4c_vc_output.athinput`
- `tst/inputs/z4c_vc_static_multilevel_wave.athinput`

### Mesh/history tests

- `tst/unit/mesh/amr_history_format_test.cpp`
- `tst/unit/mesh/amr_history_integration_test.py`
- `tst/unit/mesh/amr_history_shadow_static_test.py`

### Z4c tests

- `tst/unit/z4c/cartoon_production_kernel_test.cpp`
- `tst/unit/z4c/cartoon_vertex_axis_test.cpp`
- `tst/unit/z4c/shared_geometry_policy_test.cpp`
- `tst/unit/z4c/z4c_grid_layout_test.cpp`
- `tst/unit/z4c/z4c_irisk_import_static_test.py`
- `tst/unit/z4c/z4c_policy_migration_test.py`
- `tst/unit/z4c/z4c_symmetry_dispatch_test.cpp`
- `tst/unit/z4c/z4c_symmetry_validation_test.cpp`
- `tst/unit/z4c/z4c_vc_brill_direct_init_test.py`
- `tst/unit/z4c/z4c_vertex_amr_history_test.py`
- `tst/unit/z4c/z4c_vertex_amr_test.cpp`
- `tst/unit/z4c/z4c_vertex_boundary_indices_test.cpp`
- `tst/unit/z4c/z4c_vertex_derivatives_test.cpp`
- `tst/unit/z4c/z4c_vertex_dynamic_amr_test.py`
- `tst/unit/z4c/z4c_vertex_history_quadrature_test.cpp`
- `tst/unit/z4c/z4c_vertex_linear_wave_convergence_test.py`
- `tst/unit/z4c/z4c_vertex_output_test.py`
- `tst/unit/z4c/z4c_vertex_restart_test.py`
- `tst/unit/z4c/z4c_vertex_static_multilevel_test.py`
- `tst/unit/z4c/z4c_vertex_task_order_static_test.py`
- `tst/unit/z4c/z4c_vertex_to_cell_test.cpp`
- `tst/unit/z4c/z4c_vertex_topology_test.cpp`

## Design and baseline documents

- `docs/design/z4c_vertex_cartoon_axis_regularization.md`
- `docs/design/z4c_vertex_centered_design.md`
- `docs/design/z4c_vertex_centered_sources.md`
- `docs/investigations/z4c_vertex_centered_20260821/baseline/README.md`

## Owned investigation artifacts added in the final evidence commit

- Perlmutter/Aurora launch and run scripts
- fixed-grid/common-tree input decks
- deterministic analysis script
- CSV tables and PNG figures
- curated backend evidence
- `REPORT.md`, `REMOTE_REVIEW_PROMPT.md`, inventories, and strict manifests

Build directories, executables, restarts, large native-shadow streams, and raw
scheduler staging roots are intentionally excluded from Git.  Their locations
and hashes are carried by the evidence manifest.
