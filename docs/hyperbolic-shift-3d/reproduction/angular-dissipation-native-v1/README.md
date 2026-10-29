# Optional angular dissipation: local native verification

The degree-seven scri candidate passes its local implementation checks at
source commit `d98e2a99bbf1d5546956be3d724c82715ca704ec`. These tests use
Kokkos Serial; **SYCL preflight and sustained stability are not established by
this record**. The original refinement and failed V1/V2/V3 duration evidence
remain unchanged.

The native operator uses the fixed reference-sphere areas and nearest actual
surface-node reconstruction studied in the
[prototype](../scri-dissipation-prototype-v1/README.md). Twice-orthogonalized QR
matches retained NumPy SVD coefficients within 1.41e-15 and the n6 test source
within 4.67e-14. Weighted energy and adjoint identities agree within 6.85e-15;
constant data give exactly zero. Native/device-interface and twelve ownership
partitions agree exactly on this Serial backend. Tests cover n6/n8 degree
seven and n6 degree five. The native conservative paired-incidence spectral
bound at n6/degree seven is 32.8718, larger than the prototype bound that first
combines diagonal contributions; both bound the same added operator.

The closure tests include an analytic nonspherical additive-rate perturbation
and the actual polynomial operator. Dependent-D response errors are below
2.84e-17, compatibility below 2.78e-17 and null error below 1.12e-15. Absent
rates, zero rates and disabled host strength preserve the previous behavior.
Invalid shape, nonfinite input, forbidden dependent-field rates, output alias,
cross-face configuration and isolated-Q CFL cases are rejected.

Fifteen native MPI cases verify:

- V4 epsilon=0 and V3 produce bitwise identical four-array checkpoint payloads.
- A two-rank two-plus-two-step pulse restart matches a one-rank four-step run,
  including the complete V4 header and all 267,840 payload doubles.
- Wrong epsilon, cross-schema, unadmitted checkpoint and invalid strengths are
  rejected. An overflowing finite-strength CFL failure still writes valid JSON
  with null nonfinite diagnostics and an unadmitted checkpoint.

The short pulse uses n6, dt=.00025 through coordinate t=.001. It is a restart
and integration fixture, not the requested long-duration physical test.
The Python controller/export/preflight suites pass 69 tests. Both readers also
validate the native checkpoint/report configurations; the GPU adapters reject
the Serial backend as GPU evidence.

`assessment.json` binds source and component binaries. `component-tests.log`,
`ctest.log`, `python-tests/` and `mpi/` retain checks and small raw outputs.
`mpi/verification.json` records hashes of every raw artifact, including the
large local checkpoints retained outside Git. Historical fourteen-case
evidence preceded the overflow-report fix; this package contains the final
fifteen-case rerun.

Rebuild the targets named in `tst/hyperbolic_shift_3d/CMakeLists.txt`, then run
the `hyperbolic_shift_3d_angular_dissipation`,
`hyperbolic_shift_3d_scri_dissipation_closure`,
`hyperbolic_shift_3d_scri_device` and
`hyperbolic_shift_3d_dissipative_mpi` CTest cases. The latter uses at most two
MPI ranks and has a 420-second external limit. Launcher tests are
`python3 tst/hyperbolic_shift_3d/aurora/test_control.py` and
`python3 tst/hyperbolic_shift_3d/aurora/test_preflight.py`.
