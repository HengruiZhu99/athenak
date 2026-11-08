The native physics headers expose the existing N12 equations without an evolution
driver, flat global field buffers, mesh ownership, or MPI transport. The original
BSD attribution and source path are retained in every ported mathematical header.

`Fields` fixes the 31-slot layout. Only 26 slots are independent: slots 6 and 12
are algebraically reconstructed, and slots 28–30 are dormant. Native tasks must
reconstruct dependent fields after every RK stage and interpolation/refinement.
The runtime chi floor belongs to that stage policy; neither Omega nor chi is
silently floored inside the point equations.

`mapped::Point` and `mapped::Jet` alias the native chart atlas types. Tensor
components stay Cartesian. Only derivative directions use the map Jacobian and
Hessian. `BlockResidualReader` reads native `(block,field,k,j,i)` arrays, subtracts
the stored reference and divides the chi residual by radius squared.
`BuildPointState` restores analytic Cartesian reference jets, reconstructs the
algebraic chart, applies the compatible lapse Hessian, and optionally adds a
caller-supplied Cartesian shift-normal lift. The ordinary stencil needs four
ghost cells. Endpoint-adjacent radial reconstruction is supplied separately.

`EvaluateBulkPoint` takes complete jets and explicit transport/KO contributions.
It returns all 31 rate slots, an error status, the shift/lapse gap and three
logical characteristic speeds. Native tasks apply their own timestep factors.
Lambda KO is additionally applied to D, while Lambda transport is not. Native
tasks supply zero KO for dormant fields. `EvaluateScriPoint` retains the original
full endpoint RHS, clock, compatibility defect and validity semantics, including
the original tensor-pole derivative denominator convention. In particular, it
does not add a new endpoint gauge-validity rejection.

`endpoint.hpp` provides the seven-radial-node polynomial weights and normal lift;
the six interior centers and evolved scri endpoint remain distinct. Donor
selection, cross-face angular interpolation, surface ownership and normal-system
solution are native task responsibilities. `PhysicalKO8::Line` accepts positive
line weights and physical field samples, without owning a mesh or donor plan.

The Serial oracle uses the exact legacy bulk `operator()` and sampler methods,
extracted from the read-only source at test-build time. It compares nonspherical
bulk and scri jets, reference restoration, algebraic expansion, atlas map jets,
characteristic speeds, invalid-state behavior, packed field layout, fourth-order
stencils, compatible lapse Hessian and biased transport. Independent checks cover
endpoint polynomial differentiation, normal-lift zeros/normal derivative, and
the KO quadratic dissipation identity.

Run from the native worktree, after configuring/building Serial Kokkos:

```sh
python3 tst/z4c_hyperboloidal/physics_oracle.py \
  --legacy /path/to/read-only/scri-ko01/hyperboloidal \
  --build build-native-baseline \
  --output build-native-baseline/physics-oracle
```

The script invokes `make -j64`. Current validation: 432 bulk states, 256 scri
states, 37 invalid cases and 492247 scalar comparisons; maximum scaled error
`7.0359255910187767e-14`. This establishes local equation/stencil equivalence. It
does not establish full native evolution, cross-chart transport, GPU performance,
AMR/restart equivalence, or the 5M trajectory.
