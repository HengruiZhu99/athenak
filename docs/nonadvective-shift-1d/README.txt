Nonadvective adapting-reference Gamma-driver experiment

Outcome: REJECTED boundary prototype. The bulk equations and scalar-speed algebra are implemented, but the independent endpoint derivative-jet closure fails consistency with derivatives reconstructed from the metric. Both N257 puncture tests stop at coordinate t=.15=.45M on the scri-numerator guard. N65 and N129 pulse controls also fail. No100M nonadvective run was admitted, and no long-time settling or convergence claim is made.

Start with report/main.pdf. The report includes the boundary consistency counterexample, all failures, exact-time comparisons, exterior-normalized constraints, mass/horizon diagnostics, lapse/shift/metric profiles, boundary/clock residuals, characteristic separation, floor/positivity and interior diagnostics. The primary connection-constraint comparison uses independently reconstructed metric derivatives rather than the smaller evolved-jet diagnostic.

The experiment is isolated on codex/nonadvective-adaptive-reference. No change was merged into the main gauge. Inherited unfinished edits to the earlier advective manuscript were preserved and excluded from this experiment's commit.

Reproducibility:
- request.txt is the original specification.
- inputs/nonadvective_shift_1d/protocol-v1.json (repository root) was declared before trajectories.
- scalar-symbol-v1.json and implementation-audit-v1.json record algebraic/RHS checks.
- boundary-step*.json/.npz preserve numerical linearization and method controls.
- boundary-consistency-counterexample.json and constraint-sampling-audit.json expose the failed reduction consistency.
- comparison-final/ contains the final assessment and standalone PNG/PDF figures.
- evidence/ contains actual terminal manifests, histories and checkpoints, including failed cases.
- frozen-experiment-source.zip contains the exact scientific snapshot and its bound macOS native library. Current diagnostic-only source options default to their frozen values; default-source-equivalence.json verifies initial/RHS fixtures exactly.
- report/report-source.zip contains a standalone LaTeX project; run pdflatex twice from its extracted directory.
- independent-review-prompt.txt is a critical review and next-step prompt.

The native library was reused only after source/library/checkpoint binding checks. To build a new library from the repository, use the existing spherical adapter workflow:
cmake -S tst/elliptic_shift_1d -B build-nonadvective-shift -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON
cmake --build build-nonadvective-shift --target spherical_z4c -j 2
Python dependencies for the experiment/figures are NumPy and Matplotlib; symbolic checks use SymPy1.14. A different compiler/platform is not guaranteed to recover a bitwise-identical trajectory. The exact bound binary is archived for the recorded local result.

Execution commands and hashes are retained in the manifests. Failed/bounded trajectories must not be resumed or relabeled. The archived campaign is a historical reproduction harness, not an admitted production gauge; do not start it automatically. Independently review and correct the boundary construction before another qualification campaign.
