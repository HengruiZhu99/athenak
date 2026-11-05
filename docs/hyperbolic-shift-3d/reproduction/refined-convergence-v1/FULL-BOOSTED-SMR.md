# Combined boosted SMR candidate and fixed-topology memory control

This local candidate retains the full goal: stationary and exact v=.1 boosted
punctures with resolved, regular evolution through10M. The test below reaches
only0.00025M; it is not a long-time stability or convergence result. The stationary frozen source/build are unchanged; current job status is in state.json.

## Memory change and safeguards

After native AthenaK construction or checkpoint topology restore, fixed, uniform
and SMR policies never call the field-transfer machinery again. `SealTopology`
fences device work and releases the four native fine/coarse state/compensation
arrays before constructing the coupled evolution. The tree and registered view
objects remain alive. Subsequent Load/Gather/Regrid/refinement calls reject.
The dynamic fixture retains its storage and existing lifecycle. Checkpoints
store the coupled arrays and topology independently; their identity is unchanged.

This retires1,406,862,336 allocated bytes in the one-rank N12 uniform control
and5,499,552,768 bytes in one-rank N12 SMR. Separate native MPI migration
send/receive buffers remain allocated; this does not claim to free all AMR
storage. Peak resident memory in the matched fresh uniform control decreased
from27,036,172,288 to26,325,630,976 bytes (2.63%). Timing was166.56 versus
167.79 seconds in separate local runs, insufficient to claim a speedup or
strong scaling. Full combined-mesh peak RSS was30,366,629,888 bytes.

MPI1/2 native tree tests preserve constant state and compensation through real
refinement before sealing, preserve geometry/support afterwards, and reject
further transfer calls. The unchanged uniform-grid two-step checkpoint is
byte-identical to the previous executable. Restarting the original400-leaf
SMR control through the new executable reproduces all eight saved arrays
byte-for-byte. Frozen production executable hash remains unchanged.

Separate local candidate binary SHA256:
`567df95bee6fd0901b76ea37cc70d6525e4a170b78f6d9de29536908e640550b`.
Build commands, source/header/library hashes, memory samples and original logs
are in `sealed-topology-controls/`. The first coordinator finished every native
command successfully, then encountered a read-only checker typo
(`horizon_finest_margin` versus actual `candidate_finest_margin`). Its executed
script and disposition are preserved. The corrected read-only completion check
passed without rerunning or modifying native output.

## Full boosted configuration

Input `central20_boosted_smr_tube_radial24_angular2_reference.cfg` combines
400 native Cartesian SMR leaves with432 shell blocks:832 blocks and1,513,728
evolved points. N=12, finest Cartesian spacing.0020833333, dt=1/24000, exact
global boost+.1, the same conformal factor/initial geometry, stationary-reference
outer gauge, floors, KO8 and original scri operators. Only outer radial/angular
sampling changes relative to the earlier508-block SMR tube control.

The one-rank full combined test completes two steps with exact native checkpoint
framing, finite payload, positive metric, configuration, timestep and identity
checks. All three horizon rows are accepted, with final mass.33333337907057
(relative rest-mass error1.3721e-7), final expansion RMS.000185576, and finest
candidate margin.0248051 after the four-cell buffer. No floor/sign interventions
occur. Maximum characteristic CFL is.193715435; surface/ghost null residuals
are at most1.34e-15. This only establishes the short interval.

| Full SMR grid | Initial H, r>.3 | Initial M, r>.3 | Step2 H | Step2 M |
|---|---:|---:|---:|---:|
| radial18, original angles | 0.0142958570575 | 0.216508754962 | 0.0139971475992 | 0.213175594504 |
| radial24, doubled transition angles | 0.0014046873524 | 0.0121250567972 | 0.00111552159911 | 0.0119095241524 |

Initial H/M reductions are10.1773x/17.8563x, and step2 reductions are
12.5476x/17.8996x. Native M is the conformal-metric norm. On the same400-leaf
topology, all31 nominal Cartesian field components at r<.3 are byte-identical
to the earlier SMR control after two steps; maximum scaled difference over
the entire nominal Cartesian core is3.38e-18. This is a localized exterior
resolution effect in the short control, not a changed central gauge or an
evolved-convergence proof. The earlier coarse-core attribution results remain
separate and are not substituted for this full-puncture-grid test.

The fresh two-rank combined run also passed. Its entire957,169,752-byte
checkpoint is byte-identical to the admitted one-rank result (SHA256
`0a397ec21f10c4ad212f7cff912802a1b4def346021e5e2bfd11a4c2ede368b9`).
See `combined-smr-mpi/verification.json`. This is local Serial-MPI evidence,
not GPU validation or a combined-grid long restart test.

The visually inspected block-section figure `sealed-topology-controls/combined-grid.png` shows
nominal block boundaries (not individual cells), retained Cartesian/shell
overlap and the initial horizon enclosure with four-cell buffer. The plotted
Cartesian topology is verified identical to the400-leaf native tube.

## Reproduction and remaining work

- `build_sealed_topology_candidate.py`: separate local build; frozen binary untouched.
- `run_sealed_topology_controls.py`: bounded unit, exact-state and full SMR checks.
- `finalize_sealed_topology_controls.py`: read-only completion after the recorded checker typo.
- `analyze_combined_boosted_smr.py`: matched-core and fixed-cut constraint comparison.
- `check_combined_boosted_smr_mpi.py`: fresh full-grid MPI2 vs MPI1 checkpoint comparison.
- `plot_combined_boosted_mesh.py`: nominal grid diagram.

Aurora access is restored. Job9038758 was admitted at a healthy2.017625M
walltime checkpoint and continued as9040850. The user now prioritizes boosted
evolution and convergence: prepare its separate build while that allocation
finishes, then advance the boosted family. The incomplete stationary resolution
family remains unfinished evidence. Git push authentication is still unresolved.
The short local result does not authorize a convergence claim.
