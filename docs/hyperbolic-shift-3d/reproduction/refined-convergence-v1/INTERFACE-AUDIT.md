# Interface attribution: observations and next discriminating controls

The completed BN1 family does not show decreasing exterior constraint residuals
at10M. At fixed compact r>.3, the N24 contribution from .3–.4 is98.1289% of
the Hamiltonian squared integral and96.6291% of momentum. All recorded complete
horizon candidate enclosures lie inside that cutoff. This cannot be dismissed
as the puncture's interior coordinate singularity. The original branch retains
the full three-resolution evidence and its qualifications; no new BN1 runs.

The initial-geometry audit found a positive normalized spacelike margin and
positive sampled map Jacobians. In the suspect old region .3–.4 the normalized
spacelike margin was exactly1. This disfavors an *initial* near-null slice as
the immediate explanation, but does not establish regular evolved slicing.
Likewise, null projection at scri only tests that algebraic boundary condition.
It does not show that tensor-pole and clock compatibility residuals vanish.

## What the SMR adapter changes, and what it retains

`CartesianAMR::Descriptor` reuses the native octree, sets
`amr_cartesian_core=true`, and retains `evolved_overlap=3`. The `Lower`/`Upper`
rules in `CompositeDifference` suppress independently evolved overlap at
internal Cartesian block faces, including native refinement interfaces.
They retain it at the outer Cartesian cube and on the first spherical layer.
Shell layers above zero have no independently evolved extra cells. Ghost
values and cross-layer interpolation still couple these separately evolved
regions; there is no algebraic projection equating their evolved copies.

`DiagnosticVolume` assigns zero volume to extra evolved cells. The nominal
Cartesian/first-shell overlap uses a smooth partition of unity from the shell
inner radius to the cube half-width. Thus diagnostics do not simply double
count those extra cells. Their separate field/derivative values remain useful
for testing coupling, and must not be averaged away in a profile plot.

Per-rank performance records can legitimately report zero overlap and zero
radial-closure rows: some ranks own only interior Cartesian cells. Summing all
ranks is required. The N12 RF2 admitted fixture contains65,664 evolved overlap
cells and2,592 donor-seven radial-closure cells. SMR has therefore not removed
the overlap behavior implicated by the older field-copy comparison.

BN1 used equal widths in its first two shells, permitting exactly coincident
centers among first-layer extra cells and second-layer nominal cells. RF2's
moderate first two widths are .08 and .09. Its copies generally do not coincide;
do not label a nearest-neighbor difference as a jump at an identical location.
Keep native positions, or separately qualify any interpolation used to compare
them. Different shell positions and the v=0 RF2 control also prevent attributing
a BN1-to-RF2 norm change to one mechanism alone.

## Read-only RF2 diagnostic

`build_endpoint_diagnostic.py` builds a separate local analysis executable from
a renamed production translation unit plus `endpoint_diagnostic.inc`. It uses
the native RF2 identity/checksum reader, native octree reconstruction and
accepted-state installation. It does not step or write a checkpoint. The CPU
constraints use the configured `OmegaAt`, original mapped derivatives and
donor-seven last-row closure with the **current** refreshed boundary values.
The immutable host plan's initially stored boundary values are not substituted.

The reader writes unsmoothed native-center rays, separate copy/block/level IDs,
radial integrals and a common coordinate cutoff. Cartesian transverse offsets
follow each level's native spacing; they are not one continuous exact x-axis
line. Every output must reproduce the production kernel's full and exterior
integrals. Use only separately admitted endpoints for endpoint accuracy claims.

The N12 two-step qualification reproduces saved production-history norms to
1.57e-16 relative and production volume integrals to2.10e-16. An incompatible
configuration rejects before analysis; the input checkpoint's SHA256 and length
are unchanged. Evidence and build/source hashes are in `endpoint-reader-check/`.
This establishes analysis consistency, not continuum correctness or convergence.

Next: use this reader on the current stationary run's admitted endpoint, locate
radial concentrations relative to the shifted Omega/gauge/shell interfaces, and
compare resolved derivative features. Then distinguish spatial/interface error
from evolved gauge behavior through matched stationary resolution controls
before the v=.1 sequence. Keep the old original-scri accuracy limitation and
cumulative-versus-instantaneous residual distinction explicit.
