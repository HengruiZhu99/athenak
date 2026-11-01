# Fresh SG1 N32 to 10M

The user explicitly requested N32 to10M after the verified matched SG1 N16/N24
runs. Start from original N32 wormhole generator text at t=0; do not interpolate
an evolved coarse solution or restart the superseded failed N32 campaign.

Scientific source remains47ae0c35dcef877550ac4371e74d0948842b68b7. Manifest
`source-short-gauge-v1-manifest.json` SHA256
`a5fd4a1cf9924927dfa19ddf9d27e91c6d13a91a0d12b8e997e6868853801cff`.
Use the existing SYCL/MPI binary
`build-sycl-mpi-short-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_short_gauge`,
SHA256 `1b75a50ec11f8b3908de9172bd5baec57eb8bc17d382d1fba25173ab4352996f`.
No scientific rebuild or equation change is needed.

N32 dt=0.00025, dt/dx=0.0444444, target coordinate10/3=10M. Preserve the
0.2--0.56 inner blend, direct outer driver, restored IT1 damping, original collar,
angular epsilon2 and scri penalty scale1. Direct signed chi/alpha RK candidates
and physical-variable KO from step0; lapse floor1e-10, positive chi floor1e-8
inside r<0.03, finite nonpositive chi rescue everywhere including interpolants.
Record negative/zero candidates before clipping; signs alone do not terminate.
Nonfinite arithmetic or invalid geometry remains a numerical failure. Keep
horizon lmax12/ntheta24/tolerance0.001, diagnostic interval0.05 and candidate
exterior diagnostics with all acceptance flags and gaps.

Input `inputs/puncture-continuous-n32-short-gauge-v1.txt`,184,696,992bytes,
SHA256 `acc4f740feb68a3378dc6129a2f897422fb77e07ca97aa8a469cc5162b9ccaa1`.
The new launcher is `launcher-short-gauge-n32-v1`; verify every file against
its provenance before use. The helper/controller supports the explicit policy
`n32_user_authorized_short_gauge_fresh_to10M`. N24's existing N16-review gate
remains unchanged; the N24-only review is not repurposed as N32 authorization.
All129 controller/campaign/checkpoint tests passed, with new N32 cases for fresh
initialization, dt/schema/floor binding, forbidden changes and continuations.

Checkpoint HYP3D_MPI_SG1, header168,scheme3; N32 length518,617,768bytes. After a
terminal healthy walltime stop, collect with the new frozen controller and copy
the full campaign locally under output/hyperbolic-core-shell/short-gauge-n32-v1/aurora,
retaining campaigns/... paths and launcher. Hash/length-verify every artifact and
run the frozen helper verify. Require admitted exact configuration, finite
payload, matching report/header time/steps and forward progress. Release only
the matching terminal lock once. Continue using a new sequential N32 plan with
--restart verified_checkpoint --sha256 verified_hash and any horizon-seed sidecar.
Do not supply the N24 review option. Never resume numerical failures or silently
change dt, equations or floors. Resolve infrastructure/storage/integrity failures
before another submission. Do not duplicate queued/running/ambiguous jobs.

Use CompactBinaryMerger/debug, one node/job at a time,12 MPI ranks/GPU tiles;
worker3200s,command3100s,native exit reserve180s. Check ALL user debug jobs before
submission and preserve unrelated held jobs. Home task limit5,000,000,000bytes
includes planned outputs and atomic checkpoint reserve. The obsolete old N32
AC1 3M input and completed SG1 N16/N24 initial text were copied locally and
SHA256/length-verified before retirement. Preserve receipts and active SG1
science. For continuations, retire verified obsolete outputs/initial text as
needed, retaining the current required restart until a successor is verified.

At terminal completion/failure collect evidence and compare with N16/N24 using
`short-gauge-v1/compare_resolutions.py --n32-plans ...` and a fresh output directory.
Use matching native SG1 diagnostics and one cutoff outside all final candidates
for admitted matched10M endpoints; never treat a failed checkpoint as accepted.
Retain missing horizon masses and separate candidate/floor/boundary counters.
Assess radial residuals near0.16--0.30 and0.775--0.9. Three grids have unequal
spacing ratios; do not infer a single order by assuming factor-two refinement.
Stop/delete the20-minute monitor after collecting, comparing and publishing the
trial disposition. A10M completion alone does not establish long-time accuracy.
