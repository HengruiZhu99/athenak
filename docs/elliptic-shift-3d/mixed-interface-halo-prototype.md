# Mixed physical P5 interface-halo falsifier

These separate standalone experiments retain the original fourth-order mapped
derivative operator, including all Jacobian and second-chain-rule terms. They
replace only Cartesian/shell crossing ghost values read by the same five
query stencils used in the amplification localization. All active field values
and the physical equations remain unchanged. They are not production options.

The first catalog fits all56 physical Cartesian P5 monomials to nearest global
active donors, supports125,175,256. Full rank, condition<=1e8, reproduction<=1e-10,
value-functional L1<=16 and final composed gradient/Hessian L1<=32/64 are required.
Constants are reconstructed with an anchor; mixed Cartesian/shell donors must
bracket the query in its dominant physical normal direction. There is no
lower-degree or old-halo fallback for an unsupported query.

Of70 unique crossing halos,46 qualify and24 do not. All three shell-side query
stencils qualify, reducing composed gradient/Hessian amplification to:

| Query | Original | Mixed physical P5 |
|---|---|---|
|0|18.4219 /54.1258|2.71879 /4.62238|
|6|64.7774 /156.531|2.56173 /6.02835|
|126|5.98054 /14.6291|2.80099 /8.30345|

Both worst Cartesian query stencils remain unsupported. At ghost82871, nearest
256 donors contain245 Cartesian and11 shell points. Full rank and reproduction
pass, but value L1=18.6388 exceeds16. Thus simple nearest selection remains too
one-sided despite normal bracketing.

A second catalog preserves these accepted plans, then tries positive physical
cell-volume least-squares weights and quarter/half shell-balanced subsets.
It retains all guards and a predeclared maximum donor distance8h for the new
balanced subsets. Coverage stays46/70. Volume weighting on the original nearest
256 raises the example L1 to19.2747. Quarter-shell175 has acceptable L1=13.2021,
but reaches0.180967, beyond8h=0.166667, and is rejected. No old failures are
reclassified as passes and no support or amplification cap is loosened.

Independent constant/noise checks pass on the three qualified stencils;
functional composition agrees with direct padded evaluation within3.1e-27 for
1e-12 noise. Seven scalar profile cases in both positive fields retain mapped
jet and full physical-source errors against analytic data, plus an all-exact-halo
control. Some physical source errors worsen; reduced weight amplification does
not itself establish accuracy. For example at point126 the initial chi H error
changes from-0.0436233 to-0.0412906 while a source-error maximum rises from
0.0147239 to0.0148915.

The ray audit independently confirms the proposed geometric cause: for a
near-axis Cartesian halo at radius0.2608..0.2658,32 of36 angular donor rays have
cube boundary radii larger than the fixed sampling radius. Those samples lie
inside the Cartesian cube on their rays, whereas the existing radial chain
contains only shell data. A separate topology-aware ray extension is the next
hypothesis; it does not broaden the failed physical P5 support bound.

Commands are `run_mixed_interface_halos.py` and `run_balanced_interface_halos.py`
under `tst/elliptic_shift_3d`, with `--output <new-directory>`. Evidence resides in
`evidence/mixed-interface-halos-20261027` and
`evidence/balanced-interface-halos-20261027`. Each preserves every attempted
support, accepted donor geometry, profile/noise controls and a frozen source
capsule. No coupled evolution or vector solve is claimed.
