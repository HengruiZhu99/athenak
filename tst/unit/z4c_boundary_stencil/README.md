# Active-only characteristic boundary derivative

Run against a host-accessible CPU build (Serial or OpenMP):

```sh
python3 tst/unit/z4c_boundary_stencil/run.py \
  --build-dir /path/to/build --output-dir /path/to/results
```

The runner reuses the build's compiler flags and linked libraries. GPU-only builds are outside this unit test's scope.

The test leaves all ghosts NaN and evaluates the actual `BoundaryCoordinateDerivative2` header on active data. It checks1,536 derivatives of a manufactured quadratic, including mixed terms and anisotropic spacing. It then constructs the raised normal from a positive-definite, determinant-one metric with nonzero off-diagonal components and tests98 configurations spanning six physical faces,12edges and8corners, including internal tangential block boundaries. Every expected normal derivative is nonzero, so ignoring physical responses cannot pass.

A retained copy of the original production derivative is a failure witness: it reads poisoned tangential ghosts and returns NaN in72/98 configurations. The new helper returns finite, accurate derivatives in all98. An exactly diagonal normal hides the original bug. The test also verifies exact zero preservation, state/RHS derivative linearity, and zero derivatives in all three inactive directions despite poisoned center/ghost values.

Observed double-precision errors: coordinate3.42e-14, normal1.55e-14, linearity1.69e-14; exact-zero and inactive-direction errors zero. The test checks the helper's ownership and consistency contract. It does not by itself establish full coupled-boundary evolution stability or CPU/GPU equivalence.
