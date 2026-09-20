# Damped constraint boundary helper point test

Run against a configured CPU build with host-accessible Kokkos views:

```sh
python3 tst/unit/z4c_damped_boundary/run.py \
  --build-dir /path/to/cpu-build --output-dir /path/to/test-results
```

The runner currently supports the local macOS OpenMP toolchain used for this
review; it records its full compiler invocation. It is not a portable GPU test.

The compiled test calls the actual C++ helper. It verifies:

* exact zero for equal full/background states, including nonzero states;
* a quadratic manufactured constraint field on all 26 signed face/edge/corner
  normals, with nonunit conformal metric, lapse and chi;
* the derived damping coupling
  `sigma*(Theta-sqrt(chi)*Q_normal/2)` and `sigma*Q`, independently evaluating
  the normal covector contraction rather than reusing the helper's Z conversion;
* nonzero physical response rather than accidental resetting;
* NaN-poisoned state and RHS ghosts are not read;
* the metric-defined Gamma time derivative by independently finite-differencing
  the complete Z functional on a nonconstant metric;
* zero physical Z when evolved Gamma equals metric-defined Gamma.

These are operator-value tests. They do not assert that the complete boundary
condition is constraint-preserving in the nonlinear system, or that an evolution
is stable. Evolution and continuum-symbol results must be reported separately.
