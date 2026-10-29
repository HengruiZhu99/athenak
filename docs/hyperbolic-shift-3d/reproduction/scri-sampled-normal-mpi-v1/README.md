# Experimental V5 local MPI integration checks

The sampled-normal target passes these **short Serial-backend integration and restart checks**, using one or two local MPI ranks. No actual twelve-rank MPI or SYCL execution is recorded here. The independent [device component test](../sampled-normal-device-local-v1/README.md) simulates twelve surface ownership partitions in one process; actual twelve-rank GPU validation requires Aurora preflight.

This remains an experimental normal-factorization change. The [manufactured study](../scri-sampled-accuracy-v1/README.md) finds roughly third-order complete sampled boundary-rate accuracy. Four steps do not establish stability, wormhole-to-trumpet fidelity, long-time convergence, or an improved physical scheme.

## Integration and restart identity

`hyperbolic_shift_3d_mpi_sampled_normal` enables the existing algebraic projection,
cross-face angular jets and angular dissipation, and fixes the normal scheme
to `sampled_native_numerator6`. It retains the V4 positional CLI, including
optional angular epsilon at the last position (default 2). Other targets retain
the analytic-normal default.

Surface-row ownership differs from bulk-block ownership. After the projected
surface is gathered, a dedicated exchange fills the block halos needed by each
rank's five radial numerator donors. The device helper uses the existing native
bulk jet arithmetic and actual interior Omega. Its samples enter the normal
pole derivative before unchanged regularity/Q terms and dependent-D
reconstruction. Final D is gathered, then the ordinary bulk halo fill runs.
No geometric regularity or independent Lambda reset is added.

The checkpoint format is distinct:

| Byte offset | Record |
|---:|---|
| 0 | Existing 72-byte header, magic `HYP3D_MPI_V5` |
| 72 | 16-byte `<dii>`: epsilon, degree=7, reserved=0 |
| 88 | 16-byte `<4i>`: scheme=1, radial_nodes=6, bulk_derivative_order=4, reserved=0 |
| 104 | Existing four arrays: volume, volume compensation, surface, surface compensation |

All configuration fields and admission are checked before reading state arrays.
V1–V4 layouts are unchanged. V5 reports additionally declare
`scri_normal_derivative`, `scri_normal_radial_nodes`, and
`scri_normal_bulk_derivative_order`.

## Recorded results

The fixed 19-block grid has n=6. Pulse runs use dt=0.00025 and compare four direct
steps to two steps plus a two-step restart, reaching coordinate t=0.001.

- One-rank and two-rank direct checkpoint files are bitwise identical.
- The two-rank restart checkpoint is bitwise identical to the one-rank direct checkpoint, including both compensation arrays and final D.
- Rebuilding the V4 target against the new optional code leaves its checkpoint bitwise identical to the frozen pre-change V4 executable.
- Epsilon-zero V5 and a two-step Minkowski check pass. Minkowski state deviation at t=0.0005 is 6.1573e-16.
- The pulse's maximum surface/ghost null residual is 1.1103e-15; final determinant and trace-free residuals are 6.6614e-16 and 6.5053e-19.
- Each of the four wrong normal-configuration integers and a truncated normal record rejects. V4↔V5 cross-restarts, mismatched epsilon and an explicitly unadmitted test copy also reject.

The entire serial launch sequence completed in 30.54 seconds. Per-segment
running diagnostic maxima need not be identical across restart; the comparison
uses the full checkpoint bytes and current final diagnostics. The synthetic
unadmitted test copy only changes a valid fixture's flag to zero to verify
rejection; no failed scientific checkpoint is promoted or evolved.

## Reproduction and evidence

From the repository root, build the two explicit targets, then run the bounded
script (each launch uses at most two ranks):

```sh
cmake --build build-hyperbolic-mpi --target hyperbolic_shift_3d_mpi_sampled_normal hyperbolic_shift_3d_mpi_dissipative -j2
python3 tst/hyperbolic_shift_3d/test_sampled_normal_mpi.py --binary build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_sampled_normal --control build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_dissipative --mpiexec /opt/homebrew/bin/mpiexec --output-dir /tmp/new-sampled-normal-checks
```

Use the local MPI launcher path appropriate to the host. The output directory
must be empty. To reproduce the historical executable comparison, also pass
`--frozen-control PATH` pointing to the independently preserved V4 binary;
its tested SHA256 is recorded in `frozen-v4-control.json`. Without this optional
argument the script explicitly reports that comparison as missing (`null`).

`execution.json` preserves the exact recorded command; `verification.json`
binds all original artifacts, including the checkpoint files retained locally.
Only small reports, histories and logs are copied into `results/` here. Every
original artifact hash was rechecked while packaging. `source-binding.json`
binds the compiled sources and tested binaries, and `inventory.json` binds the
retained evidence. No remote allocation or duration run was launched by these
tests.
