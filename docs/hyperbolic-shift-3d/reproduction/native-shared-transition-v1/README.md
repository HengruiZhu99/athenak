# Active convergence investigation

The persistent goal has started on `codex/native-initializer-shared-transition`. Read [STATUS.md](STATUS.md) and [goal-specification.txt](goal-specification.txt) for the current requirements, including the additional KO consistency repair. The earlier queued specification below is retained as history; the active goal takes precedence.

# Queued native initializer and shared-transition investigation

Status: queued after the existing SG1 N32-to-10M campaign reaches its terminal disposition and its evidence/comparison are preserved. No new scientific changes or jobs have been made for this follow-up.

The complete authorized specification is preserved verbatim in [user-request.txt](user-request.txt). Its SHA256 is `6bfe4dce1a4fc0354967078ff4a12f91f331089dc906afeb744830c3a97a1bdb`.

Start implementation on a separate branch, `codex/native-initializer-shared-transition`, from the current SG1 branch after the N32 results are committed. Preserve the frozen SG1 builds and prior results. The existing 20-minute monitor carries the handoff; do not create duplicate monitors or jobs.

Sequence:

1. Finish the current N32 campaign, including healthy-walltime continuations, artifact verification, N16/N24/N32 comparison and terminal lock release.
2. Implement the native production C++ wormhole initializer with verified quadrature/inversion and analytic scri limits; remove production Python generation and the fixed shift table.
3. Establish initial-data accuracy and convergence at N16/N24/N32, separating continuum, finite-difference, exchange/reconstruction/projection and norm/cutoff errors. Use independent measurements and regional localization.
4. Audit complete scri operator consistency with nontrivial regular manufactured fields. Diagnose and separately document any actual defect; roundoff null closure is insufficient.
5. Implement the single-W shared restoration on the existing SG1 0.2–0.56 transition, retaining inner and near-scri limits and unchanged auxiliary equations.
6. After the audits, run matched fresh N16 cases A (native initializer, original collar) and B (native initializer, shared restoration) to 10M. Compare both with preserved SG1. Report any consistency obstruction before proceeding with a long evolution presented as a convergence test.

No new long N24/N32 evolutions are authorized by this follow-up. Preserve direct chi/lapse, floors/sign logging and other matched parameters. Use CompactBinaryMerger/debug, one node and 12 ranks/GPU tiles at a time, SYCL/MPI builds with make -j64, and the 5GB home task cap. Verify evidence before purge and restart only admitted healthy checkpoints. Push implementation, tests, plots and conclusions on the separate branch.
