# Static Cartesian core resolution control (V8)

V8 uses the V7 equations, logical inner-transition exchange, dissipation and
scri closure unchanged. It partitions the same Cartesian cube into 2×2×2
blocks, each containing n³ cells, halving the core spacing while retaining
all 18 shell blocks and their coordinates. There are 26 blocks (106496 cells
at n16). This is a static spatial-resolution control, not dynamic AMR.

Native Serial tests at n6 passed full CPU/device halo, bulk and closure parity
for Minkowski and a nonspherical perturbation. Null error was below1.12e-15.
Actual one-/two-rank MPI checkpoints and direct/restarted checkpoints match
bitwise; the rebuilt V7 control matches its preserved checkpoint bitwise.
Five malformed mesh headers, five malformed normal configurations and both
cross-schema restarts were rejected. The Python controller/decoder tests
passed 90 cases. These are short component tests, not duration or convergence
admission, and the Serial backend is not evidence of GPU performance.

Independent native/Python coordinate generators agree within2.78e-17, preserve
the Cartesian volume, and give identical shell coordinates and volumes for
both core layouts. Real V8 checkpoint export/reanalysis completed using the
explicit core2 diagnostic operator. V8 adds a checked16-byte topology record
(header120 bytes); V7 remains104 bytes. New n16 initial data use the preserved
isotropic donor initializer and library; hashes are retained in initial-n16.json.

Reproduce using targets hyperbolic_shift_3d_mpi_core2,
hyperbolic_shift_3d_core2_device, hyperbolic_shift_3d_core_coordinates and
hyperbolic_shift_3d_reanalyze. The MPI test is test_sampled_normal_mpi.py
--version 8 with the rebuilt V7 binary as --control. Run the Aurora Python
unittest modules test_preflight and test_control. No long 3D core2 run has
been admitted yet. The proposed n16 timestep is.00025, matching the concurrent
unrefined-core timestep control.
