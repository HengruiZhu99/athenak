# Native scri operator contract

`SurfaceExchange` belongs to one native `MeshBlockPack`. Each outermost leaf
owns one N-by-N surface patch on the same rank as its bulk cells. The class does
not repartition the surface, replicate its evolved state on other ranks, or
gather normal corrections. Cartesian field components retain their common
component basis across angular chart boundaries.

The stage dependency is:

1. Expand the independent algebraic chart. Project the local surface lapse and
   shift to the prescribed clock/null trace. Set D to Lambda plus the outgoing
   extrapolation of D-Lambda from the final five bulk radial nodes. These donors
   are in the same native block and require no communication.
2. Form bulk reference-subtracted residuals, including the chi/r-squared
   transformation. Surface residuals subtract the boundary reference directly.
3. Post and pack two persistent `MappedExchange` plans: direct evolved bulk
   donors for surface differentiation, and surface value donors. Every rank
   participates in plan construction, including ranks with no surface patches.
   Only requested values move during a stage; receive tasks poll nonblocking.
4. Form the surface Cartesian jets from six interior radial nodes plus the
   separately evolved endpoint and centered five-node angular stencils.
   Out-of-patch angular values use six-node value interpolation in each donor
   direction. Tensor-row coefficients are merged in first-encounter order.
5. Restore reference jets and solve the three-by-three normal-compatibility
   system at each owned surface point. The initial native implementation retains
   the original four boundary evaluations and pivoted elimination.
6. Differentiate the clamped six-by-six interpolation of the solved correction
   within the owning patch, multiply it by the polynomial normal extension,
   then evaluate the complete surface RHS. The extension vanishes at the six
   interior nodes and the endpoint, with unit normal derivative at the endpoint.

The correction interpolation uses only the owning patch, as in the N12
reference. Consequently step 6 and its adjacent bulk work need no correction
exchange. The bulk endpoint reconstruction applies to the final **three** radial
rows; the normal extension applies only to the final **two** rows. The final
three rows omit the extra radial biased-transport correction while retaining
the ordinary angular corrections. `Kernel::ResidualJet` reads completed bulk
angular halos and received surface donors. `Kernel::NormalCorrection` reads
the owned correction patch.

All device arrays scale with local owned blocks/patches and their requested
donors. A thin `(local_block,field,k,j,1)` staging view lets the existing donor
transport address surface values by native GID; unowned/internal entries are
poisoned. Unused exterior radial ghosts are not initialized by this operator.
Any consumer requiring such ghosts must supply an explicit reconstruction.

The independent regression `surface_oracle.py` links actual native executable
objects and compares against the read-only legacy `SphericalTransfer3D` on the
complete N12 mesh. It covers a reference state and an arbitrary nonspherical
perturbation, every surface projection, all surface jets, every normal
correction and surface RHS component, and all three adjacent bulk radial jet
rows. It uses the adopted short-operator gate without adjustment and reuses the
same sparse plans across both scenarios. Legacy global arrays exist only in
this test. MPI rank counts are local processes, not Aurora nodes or GPU tests.

This contract does not establish native AMR transfer/restart correctness or a
coupled evolving trajectory. Those require the separate migration gates.

## Validated operators

The complete N12 oracle passes on two and eight local MPI ranks. It compares
2,897,856 scalar components across the two states. Projection and solved normal
corrections agree bitwise; maximum absolute differences are 2.50e-16 in surface
jets, 1.05e-17 in adjacent radial jets, and 2.23e-15 in surface rates. The maximum
normal defect is below 9.64e-16. The eight-rank test includes three ranks with no
owned surface patch. These results require the pinned reference policy
`-ffp-contract=off` for the native math objects as well as the oracle translation
unit; mixing contraction policies produced amplified derivative discrepancies
and was corrected without changing the acceptance gate.

`surface_smr.py` independently tests an outer-shell refinement: 113 native
blocks, four refined +X surface patches, five coarse patches, and four inner
radial siblings. All descendants of the endpoint-grid root layer retain its
radial node offset zero; only the outer children carry a surface patch. Both
two- and eight-rank runs pass 4,228,848 checks of constant/radial-quadratic jets
and patch-local normal lifts with distinct prescribed corrections. The
manufactured consistency allowance is declared from each coefficient row's
absolute sum: `256*epsilon*max(1,row_L1)*4`. The largest error uses 0.01125 of
this allowance. This condition-based polynomial test is separate from the
strict legacy-state comparison above.

The same refined-topology test corrupts the face-owner metadata on one rank.
All ranks reject the geometry plan before entering transport metadata exchange,
then reject attempts to project or post receives with that failed plan. Restored
metadata rebuilds successfully. This tests the collective construction guard,
including ranks that have no surface geometry to build themselves.
