# Independently evolved overlap experiment (V9)

This experiment adds three independently evolved layers around the Cartesian
core and each of the six cube-to-sphere transition blocks. Three interpolation
ghosts lie outside those evolved layers. Nominal cell positions, spacing,
Cartesian field components, continuum equations, gauge coefficients and the V7
outer-shell/scri operators are unchanged. Bulk KO uses the same negative-square
operator extended over each patch's full evolved region. There are no lapse
floors, Lambda resets or enforced spherical symmetry.

For physical radii below .45, interpolation requires a complete centered
six-point-per-direction stencil of evolved donor values. The nominal owner is
preferred where valid; otherwise an eligible overlapping patch is selected by
interior margin. A missing centered stencil is an error, not extrapolation.
At larger radii the existing radial-chain interpolation remains unchanged.
This is an adaptation of the extra evolved overlap idea, not a copy of Llama's
full discretization or a stability proof.

All nominal cells retain their original ordering, followed by overlap cells.
Physical-volume constraint norms count nominal coverage once; positivity,
finite-state and maximum-error checks include every evolved cell. At n16 this
uses123688 evolved cells rather than77824 (59% extra). Checkpoint V9 has120 header
bytes and binds mesh descriptor (2,1,3,3); it rejects V7/V8 restarts. The Python
checkpoint exporter still deliberately rejects V9; native reports and the V9
preflight decoder are the supported diagnostics for this experiment.

## Completed local evidence

- Complete centered active donor stencils at n12,n16,n24 with three evolved layers.
  Maximum absolute-weight sum falls from1029.68 in the V7 inner-halo plan to
  2.67–2.69. Constant interpolation error is at most2.7e-15.
- Analytic sin(x+2y-3z) check: at n16 the transition-shell maximum Hessian
  component error falls from.04031 to.01399. Whole-shell RMS improvements are
  smaller; gradient errors are nearly unchanged. These are spatial operator
  tests, not evolution-convergence claims. See geometry-results.json.
- Complete CPU/Serial-device bulk and surface parity is exact for Minkowski and
  nonspherical data at n12. Twelve-way surface ownership, independent D traces,
  poisoned exterior radial ghosts and invalid-input rejection pass. Null
  closure is below1.34e-15.
- Two-step pulse runs: one/two MPI ranks and restart produce identical complete
  checkpoints. The original V7 control remains bitwise equal to its frozen
  checkpoint. Four altered mesh descriptors and both cross-schema restarts fail.
- 93 controller/preflight Python tests pass.
- n16 wormhole data on all original cells are bitwise identical to the existing
  input. Extra points receive the same analytic isotropic initial data.

Local reproducibility: build targets hyperbolic_shift_3d_evolved_overlap,
hyperbolic_shift_3d_overlap_device, hyperbolic_shift_3d_mpi_overlap. Geometry CLI
accepts n, overlap-width and optional coordinate CSV path. test_overlap_mpi.py
records exact command lines and checksums in its verification.json. Raw evidence
is under output/hyperbolic-core-shell/evolved-overlap-local-v1.

Long-time Minkowski/puncture stability and dynamic AMR compatibility are not yet
established. GPU preflight and the matched n16 stability experiment are next.
