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
