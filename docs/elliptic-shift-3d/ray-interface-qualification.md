# Ray-interface qualification at the user pause

All experiments below are standalone geometry/derivative prototypes under
`tst/elliptic_shift_3d`. No production halo, physics kernel, gauge or active
field value is changed. Existing failed methods remain recorded. The user
paused further work after the final bounded AMR command completed; there are
no running child experiments and no N8/N10 accuracy controls were started.

The ray extension adds actual Cartesian active-cell normal planes to each
native shell angular ray. On such a plane only the two tangential directions
are interpolated. Plane aliases are deduplicated using the geometry's
half-cell lattice; a plane not aligned with its actual AMR owner's normal
nodes is skipped rather than invented. The sorted physical radial chain then
uses six existing Cartesian/shell nodes and requires radial bracketing.
Global Cartesian components and the full existing mapped J/H derivative
contract remain intact. No singular field powers, floors or parity enter.

The first uniform N6/bulk2 five-query proof passes70/70 interface halos and all
five composed stencils. Its complete uniform interface sweep finds288 failures
among3888 halos: one-face angular interpolation extrapolates at a patch edge.
The worst value L1 is80.8755, despite maximum individual radial-ray L1 only
1.5041. Those failures remain in
`evidence/ray-extended-all-interface-uniform-20261027`.

The separate cross-face prototype fits all21 degree-five angular monomials
to49, then64 nearest actual angular rays from the six native faces. It uses a
query-centered tangent quotient chart, exact angular-node identity, pivoted
twice-reorthogonalized QR, rank tolerance1e-11, condition<=1e8,
reproduction<=1e-10 and angular value L1<=8. Each donor retains its native
face and angular indices for its own Cartesian/radial chain. There is no
recursive interpolation or ghost dependency cycle. Final volume value L1<=16
and composed gradient/Hessian L1<=32/64 remain unchanged.

| Completed test | Coverage | Result |
|---|---|---|
|Uniform N6,10800 cells|3888 halos;1648 affected active cells|All pass|
|AMR N6,47088 cells|15408 halos;6256 affected active cells|24 halo failures;240 cells unsupported|

Uniform maxima are volume value L1=11.01085, angular L1=1.59015,
angular condition189.14 and reproduction6.8e-16. Composed gradient/Hessian
maxima are7.21985/17.0912. Plan construction costs2.850s and retains2366832
entries,37.87MB as a compact `(active ordinal,weight)` payload; this excludes
`std::map` allocation overhead. Three repeated one-scalar Fill evaluations
average0.02695s. The entire bounded command completes in8.049s.

AMR retains15384 accepted halos. The remaining24 fail the unchanged final
value-L1 guard, maximum50.9642. Every angular QR still qualifies. All6016
fully covered active stencils satisfy composed caps, maxima10.3892/23.2118;
240 active stencils have an unsupported halo and are not accepted. Plan
construction costs43.035s and accepted rows retain30841086 entries,493.46MB
as compact payload, again excluding map overhead. Fill timing0.39781s per
scalar covers only the accepted rows and is explicitly partial. The command
completes normally in87.706s, within its180s one-CPU limit. The full list of24
failed plans is preserved; no further failure investigation occurred after
the pause.

Constants and seeded-noise composition controls pass in the completed five-query
tests. Radial logarithmic manufactured profiles improve substantially, but the
angular method is not exactly Cartesian P5 on a sphere. Cartesian-polynomial
and strong-angular errors remain visible. Three related resolutions and the
intended fourth-order accuracy gate are therefore **not qualified**. No
production replacement, coupled evolution, original-vector solve, performance
scalability or generic puncture-regularity claim follows from these tests.

The primary reusable API is
`interface_probe::CrossFaceRayValue::Build(const Grid&,const Real x[3],Real cap=16)`
in `tst/elliptic_shift_3d/crossface_ray_interface_value.hpp`, with angular helper
`interface_angular_weights.hpp`. Returned `weights` map **active ordinals** to
coefficients. A caller preparing a public `Grid::Row` must convert ordinals via
`g.active[id]`, retain its target, and require zero physical-boundary weight.
Build reads geometry only. Prepare all rows before any mutation; subsequent
halo rebuilding overwrites an injection. `extrapolated_halos` is separate.
The existing evolution engine accepts a mesh rather than an injected grid, so
these helpers are not wired into its production construction/lifecycle.

Reproducible runners are `run_ray_extended_halos.py`,
`run_ray_interface_qualification.py` and
`run_crossface_ray_qualification.py`. Exact arguments, frozen source revisions,
resource limits, outputs and failures are in the matching evidence manifests.
The final pause inventory records source and archive hashes; generated binaries
and unpacked source trees are excluded from version control.
