# LG1: move the gauge transition outward, keep damping fixed

The active goal is to diagnose the DO1 constraint growth and admit N24 only
following reviewed N16 accuracy success. The damping-only DG1 run is still
running. At the matching4.65M diagnostic its full Hamiltonian norm is.331950,
compared with DO1 .335618 and IT1 .065295; both DG1 and DO1 lose horizon acceptance.
This provisional result indicates that stronger damping alone does not remove
the growth. Final10M common-domain analysis remains required.

LG1 changes only the inner endpoint of the gauge blend, from.2 to.4, retaining
outer endpoint.9 and C4 smoothstep9. The physical telegrapher gauge is exact
throughr=.4. The previous restored IT1 constraint-damping interpolation remains
fixed on.2--.56, including its donor enhancement; it does not move with the gauge.
All outer gauge terms, restoring collar and scri treatment are unchanged. Compared
with DG1, the only scientific change is this gauge-weight placement. Characteristic
sampling on preserved IT1/DO1 states is recorded in direct-outer-investigation-v1;
it does not prove global hyperbolicity or future evolution stability.

The new executable hyperbolic_shift_3d_mpi_late_gauge uses unique checkpoint magic
HYP3D_MPI_LG1,168-byte header,scheme3. It must not load DG1 or any old evolved state.
Fresh original N16 wormhole text,dt=.0005,targetcoordinate10/3=10M,direct chi/alpha
RK/KO,physical lapsefloor1e-10,positivechi floor1e-8 inside r<.03 and finite
nonpositivechi rescue everywhere remain unchanged. Keep original FD4,KO,grid,
angular epsilon2,scri scale1,horizon lmax12/ntheta24/tolerance.001 and diagnostic
interval.05. Retain negative/zero candidate counts and horizon quality flags.

The N16-only helper and controller reject N24,wrong schemas,modified dt/settings,
and failed/unadmitted checkpoints. Local point tests cover physical gauge rates,
independent .4--.9 placement,unchanged outer limit and fixed damping; device parity
also passes. Remote build and binding records are preserved in build/ and n16-prepared/.
This is a prepared experiment, not a validated formulation.
LG1 N16 job9018058 is submitted after DG1 completed and failed accuracy review.
Do not duplicate its saved submission or restart the completed DG1 experiment.

Local fresh LG1 reached coordinate.001 in2steps; its LG1 restart reached.0015
in3total steps. Both completed with null residual below1.3e-15. These smoke
checks establish executable/schema/restart operation, not duration accuracy.

All125 controller/campaign tests passed. Frozen scientific source commit:
e7c0b8ce2bdd639cfb9532b196fa5badb2b1c3b0. Remote build uses the existing
SYCL/PVC+MPI configuration,oneapi/release/2025.3.1 and make -j64.

The provisional DG1 history supporting the4.65M decision is preserved in
checks/dg1-progress-through4p65M.csv. It is a live-run snapshot, not terminal
verification. The first remote configure attempt omitted the donor spherical
API directory; its error log is retained remotely. That directory was added
from the same frozen commit before restarting configuration. No science changed
and no LG1 scheduler job was submitted during this build repair.

## Frozen execution and collection

The Aurora SYCL/PVC+MPI build completed. All544 archived scientific/build source
files match the frozen Git archive;1486 Kokkos files match the prior manifest.
Binary SHA256 d64e108a0d0ce213ca30ae97709a54f2f61f9f8d771d308b0aacc0f3ad3c7c0a.
Manifest source-late-gauge-v1-manifest.json SHA256
 eafc0f7b133f961ffaf0dfc28bf02610b17bd3be406e7d794cb2417e4bf287c7.
Frozen launcher launcher-late-gauge-v1/control.py and late_gauge_campaign.py;
verify every file against provenance.json, including late-gauge-science.json.
Prepared N16 plan campaigns/n16-late-gauge-fresh10m-v1-segment01/plan.json SHA256
30fc9facaeaa32feb7f8d4f9d1449e36e2bca56b884b0a4d1fcbfd1818be3e81.
Binding measured3,559,388,349bytes with1,213,326,068bytes headroom after reservations.

Follow late-gauge-10m-state.json and the plan's saved submission before taking
any action. No duplicate queued/running/ambiguous submissions. Submit only after
the DG1 allocation is terminal, evidence collected/verified, matching lock released,
and final constraints reviewed. Check ALL user debug jobs; preserve unrelated jobs.
One CompactBinaryMerger/debug node,12 MPI ranks/GPU tiles;worker3200s,command3100s,
native exit reserve180s. Home task cap5,000,000,000bytes including planned output
and atomic checkpoint reserves; do not weaken storage checks.

After terminal evolution collect using the frozen controller; copy the full campaign
locally retaining campaigns/... layout. SHA256/length-verify every worker artifact
before any remote purge. Healthy completion/walltime stops require frozen helper
verify --root ROOT --plan PLAN: admitted LG1 header168,scheme3,exact configuration,
dt/dimensions,time/steps matching report,finite payload and forward progress.
N16 checkpoint length79,725,224bytes. Release only the matching terminal root lock
once. Healthy walltime stops below10M may continue sequential N16 plans with
--restart verified_checkpoint --sha256 verified_hash and a bound horizon-seed
sidecar where present. Numerical failures are preserved and never resumed; do not
silently change dt,equations or floors. Resolve infrastructure/integrity failures
before another job. Keep required restart data until its successor is verified.

At10M compare LG1 with IT1,DO1 and DG1 using matching checkpoint diagnostics and
a common cutoff outside all four candidate horizons. Inspect full/candidate-exterior
histories,radial error bands,horizon quality and sign/floor counts. Finite evolution
alone does not admit N24. Review accuracy against IT1 and then either investigate
further or explicitly admit a matching N24 test under the active goal.

Current LG1 job9018058 follows the frozen N16 plan above. The existing20-minute
monitor was updated to this trial; N24 remains gated on reviewed N16 accuracy.

## Additional frozen-state check while LG1 runs

The prior goal turn made concrete progress: DG1 completed and was verified, its
accuracy failure was quantified, and LG1 was submitted. Fresh scheduler evidence
now confirms LG1 job9018058 is running; worker status is running and its first
native history reaches.6M. No continuation or duplicate was submitted.

The read-only audit_frozen_gaps.py evaluates direct-outer coefficients on the
preserved DO1 and DG1 10M checkpoints, for both .2--.9 and .4--.9 placements.
It checks three squared-speed differences divided by the positive inverse-metric
normal contraction Q: transverse-shift minus lapse, longitudinal-shift minus
lapse, and lapse minus physical. All are positive at every inspected cell in
.1<r<.95, including the separate .13<r<.35 overlap-neighborhood selection.
For DG1 with its actual .2--.9 gauge, minimum exterior gaps are respectively
.131066,.243416,.0131169; in the overlap neighborhood they are.873289,1.199322,
and.0443238. The hypothetical .4--.9 placement on the same frozen state retains
positive gaps. Exact checkpoint/coordinate hashes and node counts are saved in
frozen-gap-audit.json. State labels describe checkpoints; all evaluated gauge
coefficients in this audit are the direct-outer formula, not the old IT1 formula.

This does not establish global strong hyperbolicity, exclude other degeneracies,
or determine earlier-time behavior. It provides no evidence for a lapse/shift
speed collision in the error-dominated region at the final snapshots. We still
need the LG1 evolution result to test gauge placement causally.

compare.py is prepared for verified LG1 completion: use --root LOCAL_RAW_ROOT
--plans SEGMENT_PLAN_PATHS_IN_ORDER. It verifies artifact hashes, requires a
verified admitted10M LG1 checkpoint, and compares IT1,DO1,DG1,LG1 histories and
native constraints on one common cutoff outside all candidate horizons. It
records sign/floor counters and horizon acceptance gaps; empty radial bins remain
missing. Its N24 flag stays false pending scientific review. The matching local
LG1 diagnostic executable is built. Syntax checks alone do not validate a new
10M result; no LG1 endpoint comparison has been performed yet.
