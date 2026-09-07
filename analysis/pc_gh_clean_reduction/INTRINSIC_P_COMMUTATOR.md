# Scalar p transfer commutator measured in the actual stage

The opt-in stage diagnostics now retain source and received private residual
buffers. A diagnostic-only replay also transfers the already synchronized G
state through the same private transfer routine. It is cleared before the real
residual transfer, and never writes the evolution array. Payload labels are
`transfer_residual` and `transported_state_probe`; global evolution-state helpers
reject both. Ordinary stage payloads keep their existing format.

The final probe-enabled executable is bitwise neutral against the preserved f8
reference in both transfer arms, for two FD6/RK3 steps on the seven-leaf 2D n8
point6_2d fixture. Serial and two-rank results agree, including all 18 sampled
source/received/probe buffers. Exclusive-write rejection still passes. The 19
legacy restart controls pass; the transfer-only legacy fixture's call was updated
to explicitly pass null driver/stage zero and passes a syntax check using the
actual build flags. Its full transfer campaign was not rerun in this checkpoint.

## Decomposition

Let G be p before reconstruction, R its local available-halo derivative of w,
E=G-R, and T the actual fixed-topology linear transfer applied to private fields.
The instrumented reconstruction is G+=R+T(E). Thus on ghost cells,

    G+ - G = [T(G)-G] + [R-T(R)],
    T(R) = T(G)-T(E).

T(R) is inferred by linearity from two actual buffer transfers, not sampled from
an analytic continuum field. The transfer weights and topology are fixed in
this fixture. An independent rational seven-node polynomial derivative of the
stored w stencil checks R=G-E at every stored scalar p point. Centered and
shifted closure locations are included, as is the inactive pz direction.

Across all six stages:

- The independent local derivative check has maximum error 2.185e-15.
- Repeating ordinary transfer changes p by **exactly zero** at every ghost.
- The measured p ghost correction is therefore entirely the derivative/transfer
  mismatch R-T(R), with maximum closure discrepancy 2.184e-15.
- The maximum correction across ghosts is approximately 6.016e-5. This is an
  operation increment, not a continuum error or stability measure.

At cycle0/stage1, on the normal ghost stencil adjacent to the previously
identified gid3 fine corner, x-direction p corrections at stored i=12/13/14,
j=11 are -3.16227e-6, -2.48747e-6, +1.05964e-5. The corresponding y-direction
values are -2.43240e-6, -1.91352e-6, +8.15117e-6. The sign change is retained;
there is no scalar-max subtraction in this decomposition. The first ghost uses
a centered derivative of stored primary data, while later ghosts require the
available-support closure. The trace alone does not attribute the mismatch
solely to that closure: the primary interpolation and source derivatives also
enter T(R).

This connects the previous instantaneous p-to-C RHS attribution to a measured
numerical mismatch. It does not establish the cause of the longer convergence
failure or qualify the interface. The next correction should address primary
interpolation/derivative compatibility at fine ghost layers, beginning with
coverage and polynomial tests of a higher-order prolongation control. A damping
retune does not remove the measured commutator. No physical gate advances.

## Reproduction and scope

Evidence: `qualification-runs-20260907/pcgh-clean-reduction/intrinsic-residual-trace-001/`.
Raw payloads and exact final binaries are under the same named external test
root in `/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/`. The final
source/binary manifest applies to `probe-serial` and `probe-mpi`; the earlier
residual-only trace is preserved as an intermediate result.

Run `check_intrinsic_smr_stage_budget.py` with the same fixture/reference arguments
as the preceding stage budget. The residual arm now emits 48 payloads per rank;
the none arm still emits30. `analyze_intrinsic_p_commutator.py --dumps
<probe-serial/residual_shifted-dump> --output <new directory>` reproduces the
scalar trace. The source/received buffer identity and independent derivative
checks use the frozen 2e-12 tolerance. Analysis used the external venv Python
with `-W error`. No remote run or numerical-operator change occurred this turn.
