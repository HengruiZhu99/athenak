# Experimental CPU sampled-normal closure component checks

The optional CPU closure passes these ordering and default-compatibility tests. It is **default-off and experimental**. The independent [manufactured accuracy study](../scri-sampled-accuracy-v1/README.md) measures roughly third-order complete boundary-rate accuracy for this sampled variant, compared with approximately fourth order for the existing AD variant. Neither these checks nor the directional-action experiment establishes stable or faithful evolution.

## Implemented scope

`HyperbolicComposite` accepts a fifth constructor argument,
`NormalDerivative::sampled_numerator`; its default remains
`NormalDerivative::analytic_jets`. The read-only
`SampledPoleNormalDerivative(row, padded, optional_endpoint_pole)` helper returns
**+d_n N**. It evaluates the existing native geometric pole numerator, including
`HyperbolicDamping`, at the five existing half-cell radial donors and actual
endpoint. Interior samples retain their actual Omega and native fourth-order
bulk Cartesian jets. The derivative uses the same six-node radial layout as
the original boundary jets, with anchored numerator differences.

Only the geometric normal-pole derivative changes. The regular RHS, weak
`4 N / h` and `-8 C / h` terms, angular dissipation, clock diagnostic and B rates
retain their definitions. Independent Lambda, Theta and outgoing fields are
not reset. Current-stage geometric ghosts are filled before numerator
sampling; dependent D is reconstructed from the final geometry rates, and
all ghosts are refreshed before the bulk RHS. The default path performs no
new ghost fill or additional floating arithmetic.

This package adds no runtime selector to the MPI/GPU runner and exercises no
checkpoint or restart path. Subsequent experimental integrations, if any,
require their own identity and tests.

## Recorded checks

All runs use a Serial CPU backend on the fixed 19-block n=6 core/shell grid.
They evaluate RHS functions only; no timestep is taken.

| Check | Recorded result |
|---|---:|
| Default RHS and surface versus preserved V4 header | Bitwise identical, 6 fixtures |
| Compared serialized output per implementation | 6,749,712 bytes |
| Sampled Minkowski bulk / surface maximum RHS | 1.221e-12 / 1.074e-12 |
| Only-normal-term replacement error | 4.614e-16 |
| Differentiated-null dependent-D response error | 2.086e-17 |
| Final-D ghost auxiliary-rate error | 0 |
| Auxiliary-rate difference with deliberately stale D ghosts | 2.844 |
| Surface null residual | 1.110e-15 |
| Dependent-D compatibility residual | 2.429e-17 |

The default fixtures span face-local jets, cross-face jets and cross-face jets
with angular epsilon=2, each on Minkowski and nonspherical perturbed fields.
The test also checks exact D-only invariance of geometric numerators, use of
the passed RK-stage data instead of persistent `model.state`, nonzero
independent Lambda/Theta responses, preservation of independent surface
values, and rejection of malformed helper inputs/selectors.

The same test source was compiled separately against the preserved V4
`hyperbolic_composite.hpp` in the [normal discretization audit](../scri-normal-discretization-audit-v1/README.md), with that isolated include directory placed before the current source directory and `HYPERBOLIC_SAMPLED_DEFAULT_ONLY=1`. Both full31 volume RHS arrays, surface RHS arrays and final surface arrays have the common serialized SHA256
`484852e6c4a48ebd35921671bd90ef7451b07cdd9b85e6b0e693f08f6e504a98`.
The archive binds the header, test, transitive local headers, linked libraries,
compiler version and executable hashes. Large binaries and raw fixture arrays
are retained locally, not copied here.

## Reproduction

From the repository root, with the existing local build configured:

```sh
cmake --build build-hyperbolic-mpi --target hyperbolic_shift_3d_scri_sampled_normal -j2
ctest --test-dir build-hyperbolic-mpi -R '^hyperbolic_shift_3d_scri_sampled_normal$' --output-on-failure
```

For the frozen-header comparison, the two `*-final-execution.json` records
contain the exact compile commands used. Compile the identical fixture with
and without the isolated V4 include override described above, then run each
binary with `--dump-default PATH` and compare those files byte for byte.
The isolated include contains only the preserved model header; its source
hash and all other resolved local source hashes are recorded in
`source-binding.json`. Paths in the execution records identify the actual
local run and should be adjusted for a different checkout.

`current-final-test.log`, `frozen-final-test.log`, `final-comparison.json` and
`results.json` preserve the small results. The original build records contain
an initial compile-duration field as well as the explicitly named final
build-and-test duration; the final comparison took 20.10 seconds in total.
`inventory.json` binds every retained package file. No trajectory, GPU,
restart, continuum-convergence or long-time stability claim is made here.
