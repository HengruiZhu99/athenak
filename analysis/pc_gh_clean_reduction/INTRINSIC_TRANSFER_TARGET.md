# Intrinsic reconstruction target and transfer audit

The new `intrinsic_transfer_target.hpp` defines the available-halo reconstruction
for the 30 intrinsic auxiliary slots using the canonical 50-field index enum.
It reconstructs p=Dw, l=D(rho*w), S=Ds, and B=Dbeta. It does not reconstruct Q
as an independent tensor, import the legacy factor-two lapse convention, or
introduce chart Jacobian/Hessian mixing into fundamental S residuals.

Every active cell uses the existing centered Dx arithmetic. Near the outer
stored ghost edge, the order-p polynomial stencil shifts only far enough to
stay within the available extent. The generated `HaloDerivativeWeight` table
is shared with the legacy experiment; FD coefficients are not duplicated.
Alpha uses the shared `LapseProduct` accessor, multiplying rho*w at each point.
An inactive dimension returns zero. Invalid auxiliary indices or insufficient
extent produce a nonfinite target rather than an out-of-bounds stencil.

This is an explicit available-halo closure, not proof of exact centered
reduction preservation at ghost points whose centered stencil would require
more primary support. In particular, the active RHS's derivative of these
ghost auxiliaries can have a different leading interface error than its
interior FD label suggests. That error must be measured in the actual mesh
transfer tests before promotion.

## Executed standalone test

FD2/4/6 tests cover all stored cells in a 16x16 grid and 16x16x16 grid, including
every face/edge/corner ghost layer. Independent polynomial primary potentials
have degree up to p. Linear w and rho give a genuinely composite quadratic
alpha. Analytic continuous derivatives provide the independent polynomial
oracle. A guarded accessor returns nonfinite values for any read outside the
stored domain or any attempted read of an auxiliary as a primary.

The final CPU test has 13,056 points and 391,680 auxiliary comparisons; all pass
the frozen normalized tolerance 2e-12, with maximum error 3.230e-14. All 30
per-slot errors are retained. The final source uses canonical enum constants;
the earlier successful numeric-index prototype remains as intermediate
evidence. Final source, exact binary and raw-input/output hashes are recorded in
`intrinsic-transfer-target-001/source-manifest.json` and `artifact-manifest.json`.

```sh
cmake -S analysis/pc_gh_clean_reduction/compiled -B ORACLE_BUILD
cmake --build ORACLE_BUILD --target intrinsic_transfer -j4
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_transfer_target.py \
  --binary ORACLE_BUILD/intrinsic_transfer --output NEW_TARGET_RUN
```

Use the recorded Kokkos/Athena CMake configuration for ORACLE_BUILD. This test
does not transfer data between blocks, prove curl preservation, exercise
restriction/prolongation, or qualify CUDA/MPI for this new helper.

## Concrete remaining mesh changes

The current `TransferResidualGhosts` still iterates legacy I_P1..npcgh and calls
the legacy target; its private communicator is initialized with 55 fields.
`InitializeIntrinsic` still rejects multilevel meshes and allocates no intrinsic
coarse or residual buffers. These guards remain unchanged in this checkpoint.

The next implementation must allocate/communicate 50-field residual storage,
derive residual restriction and prolongation using the actual transferred
primary potentials, and select the intrinsic target for the appropriate ghost
closure. Same-level copies already pass exact decomposition checks; introducing
a shifted derivative there must not be mislabeled an exact commuting operation.
The complete operator must include coarse/fine faces, edges/corners and repeated
exchange, with signed increments and explicit expected truncation orders.
Physical boundaries and moving/adaptive topology remain separately unimplemented.

## Quantitative KO error-vector control

Using the previously saved matched FD6 KO0.3 and KO0 solution differences, the
new `check_KO_error_vectors.py` predicts the coarser vector without fitting:

    D_coarse_pred = 64 D_fine_KO0
                    + 128 (D_fine_KO0.3 - D_fine_KO0).

The factors are fixed by order-six FD and order-seven smooth KO truncation.
For rho, the prediction's relative residual is 0.04462 and its alignment with
the actual coarse vector is 0.999935, passing the predeclared 0.1/0.99 controls.
The matched KO-response difference has order 6.94416 and alignment 0.999998.
This quantitatively supports competing error terms as the explanation for
the previous poorly aligned rho ladder. No exponents or amplitudes were fit.

These are differences between matched evolved solutions at common points,
not a raw same-stage KO RHS budget. Nonlinear response is included in the
measured KO effect. The original KO0.3 per-field Richardson limitation remains
in the ledger; this model does not turn that ladder into an asymptotic proof.
All signed vectors and per-component residuals are retained under
`intrinsic-KO-vector-001` in the external root and compact evidence directory.

```sh
python3 -W error analysis/pc_gh_clean_reduction/check_KO_error_vectors.py \
  --with-ko KO03_RUN/fd6-signed-differences.npz \
  --without-ko KO0_RUN/fd6-signed-differences.npz --output NEW_VECTOR_CHECK
```

## Production integration checkpoint

The target is now called by `TransferResidualGhosts` for intrinsic states.
Auxiliary loops begin at 20, use the runtime field count 50, and private fine
and coarse residual buffers and their communicator also carry 50 fields.
Static periodic refinement is enabled for explicit `coherent_transfer=none`
and `residual_shifted` comparisons. The default remains `none`; adaptive and
nonperiodic intrinsic meshes are rejected. The residual path preserves every
primary and all active auxiliaries; only stored auxiliary ghosts are rebuilt.
No independent Q field, GH reset or auxiliary projection was added.

CPU integration checks at FD2/4/6 in 2D/3D use 7/15 leaves with one refinement
level. All six matched initial-state comparisons preserve every stored primary
and active state exactly. All 12 serial/two-rank stored arrays agree bitwise.
Auxiliary ghost changes reach 0.00123547 (FD6 2D); these corrections are measured,
not assumed small or beneficial. The available-halo closure can also alter
same-level ghosts, so exact same-level commutation is not claimed.

Two FD6 2D one-step SMR controls reach t=0.001 with finite state. The uniform
one-step default is bitwise equal to the previous diagnostic executable.
The 19 legacy restart controls and six legacy static transfer controls pass.
An initial legacy test used the wrong problem generator and all six runs
aborted before transfer; its logs/results remain beside the corrected fixture.
A separate mistargeted parent-directory launch failed for missing input and is
also preserved. Neither failure is represented as a numerical instability.

Evidence: `qualification-runs-20260907/pcgh-clean-reduction/intrinsic-transfer-integration-001/`.
Exact binaries and large restart arrays remain outside Git under
`/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/`; source hashes,
input files, binary hashes and raw inventories are in the evidence directory.
The `intrinsic-transfer-source-001/athena-serial` and `athena-mpi` copies are the
exact intrinsic executables; `athena-legacy-transfer` has a different problem
generator. The integration runner accepts `--binary`, `--template`, `--output`
and optional `--ranks 2 --mpiexec /opt/homebrew/bin/mpiexec`. The template is
`intrinsic-time-convergence-001/dt0.001/used.athinput` under that external root.
Its output directory must have no existing case directories.

Remaining: independent signed before/after residual and curl budgets, interface
accuracy and convergence, repeated-exchange injection, refined restart
continuity and CUDA checks. The refined restart reader is opt-in decoding only;
its uniform global-array/wrap helpers still reject nonuniform leaves. No
intrinsic interface accuracy, physical convergence or black-hole gate is passed.
