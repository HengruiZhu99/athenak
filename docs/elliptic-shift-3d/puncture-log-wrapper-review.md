# Composite puncture-log wrapper review

The opt-in `PunctureLogPlans` wrapper was reviewed read-only against the pilot
integration. This review added only a standalone test and this report/evidence;
it did not change the helper, pilot or CMake while the coupled pilot was running.

The integration retains the actual stored value at each active cell and changes
only its reconstructed physical chi/alpha derivatives in selected Cartesian
cells. It does not assign `fitted_value` to an evolved field. The lapse chart
restores the global Minkowski reference A=(1+|x|²)/2 and its first/second
derivatives. This is a positive-field representation; no selected lapse gauge,
reference connection or physical RHS is replaced. The cached `chi` values sent
to the shift stage remain consistent because the override does not change values.
Mapped/scri derivatives remain on their existing paths. Accepted active-cell
constraint diagnostics call a fresh RHS and use the resulting enriched jets.

`test_puncture_log_plans.cpp` exercises the actual wrapper on:

- N=8, one Cartesian block plus one shell layer, centered puncture;
- the same grid with X=(.037,.013,−.021), nonzero velocity and stage metadata;
- N=4, a 2³ Cartesian block layout with one block refined, plus two shell layers.

The translated case is a numerical reconstruction test. It does not translate
the selected global lapse source, prove a moving black-hole solution, or evolve
the tracker. The mixed refinement case checks local donors across active coarse/
fine Cartesian leaves; it is not a coupled AMR evolution or regridding test.

For every selected row the test checks that there are exactly 125 distinct active
Cartesian donors, no shell donors, and no closer omitted active Cartesian point
(up to ties). The mixed refinement case includes donors from both refinement
levels. Constants are preserved exactly. The reconstructed jets for chi and
alpha match analytic data for q=p log rho+H with p=0,4,2,1.09,.5 and nonradial
polynomial H through degree five. The alpha comparison explicitly restores the
reference A and all its derivatives.

All 50 selected rows pass 100 constant controls and 500 physical reconstruction
controls. Maximum h-scaled stored-jet error is 2.14e−14, physical-jet error
8.89e−16, and normalized condition estimate 966.46. Physical values change by
exactly zero. Deliberately non-polynomial donor data also retain the actual
current value at every target. Complete active state vectors remain unchanged
by all sampling. All five separate changes to puncture position, velocity, time,
stage and revision are rejected by cached plans.

The first review identified missing mesh and input-contract checks. The parent
added finite positive radius validation, exact interleaved state shape and field
validation, mesh descriptor/revision checks, and a deleted temporary-grid
constructor. The unit test now requires separate rejection of zero, negative,
infinite and NaN radii; zero/negative component counts, out-of-range fields,
wrong component count and short/long state vectors; and changes to revision,
half-width, shell radius, both Cartesian map bounds, face, layer, refinement,
block count, active count and both runtime/descriptor cell shapes. A compile-time
assertion rejects construction from a temporary non-const grid.

The first extended run caught one remaining guard bug: changing only
`grid.mesh.cells` with unchanged `grid.n` and revision did not reject. It failed
with `missing mesh rejection: descriptor cell shape`; the failure and exact
sources are preserved in `descriptor-cells-guard-failure/`. The parent added the
missing `grid.mesh.cells` comparison, and the unchanged bounded test then passed. The corrected run rejects all 15 invalid radii,
24 storage-contract violations, 36 independent mesh mutations and 15 puncture
metadata changes across the three configurations. Restoring original mesh
metadata restores valid sampling in every mutation control. The earlier passing
algebra/reconstruction results are preserved separately in `initial-review/`;
the main evidence files now contain the corrected negative-contract run.

The wrapper holds a pointer to a live lvalue grid, which must outlive its plans.
Metadata guards reject stale use; they do not rebuild donors or solver state.
Native AMR/regridding still needs an explicit lifecycle reconstruction. Changes
to raw active-index ordering or cached geometry arrays must obey the owner's
mesh revision contract; this is not a hash of every mutable grid array.

The current pilot constructs plans at a fixed origin. Future motion requires
building them from the accepted/stage puncture position and replacing them when
that geometry changes; mutating the public position invalidates evaluation and
does not automatically rebuild donors. Time/stage stamps here are conservative
cache guards, not new evolution sources. The unchanged global-z selected lapse
gauge still needs a separate moving-source compatibility derivation.

The .1 and rho/h≤8 target cutoffs define a hybrid derivative region. These
algebraic tests do not establish its interface/collar convergence. The local
prototype's nonanalytic wormhole remainder and fractional angular limitations
remain. The improved coupled H control reported by the parent does not imply
that momentum, pole, radial collapse or full scientific gates pass.

Reproduce from the repository root after the existing CPU build is present:

    python3 docs/elliptic-shift-3d/evidence/puncture-log-wrapper-20261027/reproduce.py

The script compiles only the new test with the existing CPU flags and imposes a
30-second execution bound. Evidence includes full results, stderr, source hashes,
commands, source HEAD and wall timing. No existing jobs were signaled or modified.
