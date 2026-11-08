# Refined surface operator GPU qualification

This is a bounded operator test, not a new evolution or convergence campaign.
Build and later run the existing `surface_smr.cpp` and `surface_smr.athinput`
byte-for-byte. Their SHA256 values are respectively
`2f4641583d28e2d875c4234b1e2c3386710c3f20fe84e664b8e66ad43345fad3` and
`0a1ac2023d0abb6ad17931090a61c97a3505b97e7fd4f472f6d5a9342ebc2719`.
The static input retains its original dt; there is no Driver evolution or dt
override. The independently chosen AMR evolution dt .0002 is not this fixture.

`surface_smr_gpu.py` takes the actual diagnostics.cpp compile recipe from the
qualified coherent AMR candidate and preserves its SYCL/device/FP options.
It replaces only the ordinary main object and executable output in the actual
link recipe. Candidate headers and all retained objects belong to the same
coherent build. The frozen N12 baseline and existing candidate remain read-only;
full candidate/baseline verification occurs before and after compilation.
The new helper and all build outputs stay in a separate validation directory.
No GPU execution or submission is performed by this builder.

After build review and separate authorization, the proposed numerical check is
one MPI invocation at 24 ranks and one at 96 ranks, serialized, using the same
helper and input bytes, pinned oneapi/release/2025.3.1 environment, existing
12-ranks-per-node CPU binding and `gpu_tile_compact.sh`. Each invocation receives
only the unchanged input path. Use a fresh working directory, capture complete
stdout/stderr, actual command, environment, binding, executable/input hashes,
exit status and terminal scheduler record. Verify helper/candidate/baseline
before and after each invocation. A bounded per-invocation timeout may reject
an incomplete run; timeout never counts as successful qualification.

Acceptance is the existing C++ gate, with all expected records present once:

- `nodal_volume comparisons=22464 internal_radial_starts=576
  analytic_jacobian=passed`.
- `outer_smr blocks=113 surface_patches=9 comparisons=4228848 failures=0`,
  finite nonnegative `max_error_over_bound <= 1` and finite nonnegative
  `max_absolute`, followed by `collective_rebuild_rejection=passed`.
- `PASS native refined scri operators`, normal process exit0 and terminal job
  success. The C++ checks also require four refined outer faces and four inner
  radial siblings, and collective rejection after one-owner metadata corruption.

Both decompositions must pass separately. The coefficient-norm error allowance
remains `256*epsilon*max(1,row_L1)*4`; analytic volume allowance remains
`128*epsilon*expected`. No tolerance is chosen from GPU results. Report both
maximum errors and counts without requiring bitwise equality of aggregate
roundoff diagnostics across decompositions. No timing/performance claim follows.

This checks actual device surface sampling, radial jets, normal lifts and
refined nodal volumes against independent analytic values. Angular bulk ghosts
are deliberately manufactured and corrections are supplied constants. It does
not independently validate the normal-compatibility solve, generic bulk donor
interpolation, full refined RHS or long-time adaptive solution. Those scopes
remain separate from this focused check.
