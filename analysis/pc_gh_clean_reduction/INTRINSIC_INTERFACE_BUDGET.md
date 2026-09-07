# Initial intrinsic interface budget: negative accuracy result

The static-periodic residual reconstruction integrated at `5c21e673` passes
storage invariance checks but **increases curl error** in the FD6 2D analytic
initial fixture. It is not a qualified high-order interface repair. No production
equation, transfer default, projection, damping or KO parameter changed in this
analysis checkpoint.

The fixed physical mesh has seven leaves, four refined, and block edge lengths
8/16/32. The existing n8 data were reused; four new transfer-only n16/32 runs
use the exact saved `intrinsic-transfer-source-001/athena-serial` binary. No
physical evolution or black-hole run was performed. The analytic state is the
independently phased sinusoid in all 50 fields, and is off reduction. The runner
preserves primary and active state exactly between matched ordinary-transfer
and residual-reconstruction arms.

On each leaf, centered active stencils use the actual stored ghosts. The
analysis measures all 30 reductions, 30 intrinsic curls and 18 Q-curls, their
signed corrections, errors against analytic jets, component RMS/maxima and
signed maximum values with recoverable cell indices. Q is independently
materialized as `(dT)T^T+T(dT)^T`; its exact derivative is evaluated by complex
step through that algebraic expression using analytic primary and S jets.
Exact-ghost controls recover orders 5.963–5.992; two complex-step scales agree
within the frozen 2e-17 check. Actual initial active fields match analytic values
within 5e-16. Only first derivatives are used; no stale RHS/ghost derivatives.

| Group RMS error | n8 | n16 | n32 |
|---|---:|---:|---:|
| Reduction, either arm | 7.646e-5 | 2.679e-5 | 9.454e-6 |
| Intrinsic curl, ordinary | 5.153e-5 | 1.806e-5 | 6.374e-6 |
| Intrinsic curl, reconstructed | 3.143e-4 | 1.509e-4 | 7.460e-5 |
| Q-curl, ordinary | 8.570e-5 | 3.003e-5 | 1.060e-5 |
| Q-curl, reconstructed | 3.871e-4 | 1.812e-4 | 8.875e-5 |

RMS uses physical cell area, with total area 1.3, not cell-count weighting.
The exploratory RMS-decrease screen (rates >=0.5, frozen before the runs)
passes, but it is explicitly insufficient for interface qualification. At n32,
curl error is 11.70 times ordinary transfer, and Q-curl is 8.38 times larger.
Active reduction correction is identically zero because both its active G and
stored primary stencil are unchanged. Curl correction vanishes away from block
boundaries. Thus the difference measures the auxiliary ghost reconstruction,
not a difference of independently evolved solutions.

Partitioning active cells by physical level and the number of block faces within
three cells locates the largest added error in coarse block corner layers.
Their maximum intrinsic curl error is 7.373e-4, 7.043e-4, 6.903e-4: essentially
constant, compared with a decreasing ordinary-transfer maximum. These are block
corner masks, not a claim that every selected cell borders a refinement face.
Their shrinking area explains why falling full-area RMS does not establish
pointwise convergence. The saved NPZ arrays retain every signed component and
leaf/cell mapping; no bad cells or corner layers are removed from full data.

## Verified restriction contribution and next falsification

`MeshRefinement::RestrictCC` uses four-cell averaging in 2D even when
`high_order_cc=true`. For a sinusoid, its amplitude factor is exactly
`cos(kx*hc_x/4)*cos(ky*hc_y/4)`, where hc is coarse spacing. This prediction
matches all 50 stored fields at 192/320/576 coarse ghost cells inside fine
patches to 2.22045e-16. The measured bias falls by four per refinement:
9.607e-6, 2.408e-6, 6.023e-7. Its leading smooth error is
`(hc_x^2*dxx(u)+hc_y^2*dyy(u))/32`.

This verifies a second-order error feeding derivative reconstruction. Its
amplification by two spatial derivatives is a plausible explanation for the
constant corner maximum, not a completed causal proof of the full residual
transfer operator. Next: isolate a higher-order 2D point-value restriction for
intrinsic fields, with polynomial/halo checks first, and repeat these exact
signed comparisons. Keep the ordinary and residual arms distinct and preserve
this failed accuracy baseline. Repeated-stage injection and 3D/CUDA evidence
remain necessary after the interface correction is characterized.

## Reproduction and evidence

Evidence directory:
`qualification-runs-20260907/pcgh-clean-reduction/intrinsic-interface-budget-001/`.
Raw arrays and checkpoints remain under
`/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/` with inventories and
hashes; the previous integration source/binary manifest remains authoritative.
The new runner inputs and commands are in n16/n32 `results.json`.

Run `analyze_intrinsic_interface_budget.py --folders <n8 integration directory>
<n16 directory> <n32 directory> --output <new directory>` to reproduce signed
budgets. Run `check_intrinsic_interface_oracle.py --output <new directory>` and
`check_intrinsic_restriction_bias.py --folders <same three directories>
--output <new JSON>` for the independent checks. All used the external venv
Python with `-W error`. `plot_intrinsic_interface_budget.py --results <analysis
results.json> --output <png>` generates the visually checked figure.

## Isolated six-point restriction control

`intrinsic_restriction=point6_2d` now selects tensor degree-five point-value
restriction for intrinsic fine state and private residual buffers. The default
is `legacy`; no shared MeshRefinement operator or legacy equation changed.
The option rejects 3D and blocks smaller than six active cells. It uses six
active source points per direction, shifting at block edges so that restriction
never reads unsynchronized new-stage ghosts. The scalar midpoint weights are
`[3,-25,150,150,-25,3]/256`; at the first coarse center they are
`[63,315,-210,126,-45,7]/256`, reversed at the opposite end. These follow from
Lagrange interpolation, reproduced directly in the helper. The edge absolute
weight sum is 766/256 per axis: polynomial accuracy does not imply contractivity
or coupled interface stability.

The actual compiled helper passes 12,420 coarse-sample monomial comparisons
(degree 0–5 in each direction, nx6/8/16/32) with maximum 5.552e-16, against a
2e-12 bound. A source accessor returns NaN for any non-active read; no such
read survives the check. All six matched CPU mesh cases preserve active state
and primaries exactly. The n8 two-rank arrays match serial bit for bit. Default
uniform one-step and static residual transfer match the previous executable
bitwise; 19 legacy restart controls and the new 3D rejection check pass.

With this restriction, full-area errors become:

| Group RMS error | n8 | n16 | n32 |
|---|---:|---:|---:|
| Reduction, either arm | 1.386e-6 | 5.946e-8 | 2.620e-9 |
| Intrinsic curl, ordinary | 9.072e-7 | 3.971e-8 | 1.759e-9 |
| Intrinsic curl, reconstructed | 1.238e-5 | 5.723e-7 | 2.594e-8 |
| Q-curl, ordinary | 1.542e-6 | 6.671e-8 | 2.940e-9 |
| Q-curl, reconstructed | 1.866e-5 | 8.533e-7 | 3.851e-8 |

Coarse-corner reconstructed intrinsic curl maxima are now 5.835e-6, 6.088e-7,
7.056e-8, removing the previously near-constant defect in this control. Full RMS
rates are about 4.5, while coarse-corner maxima approach order three. This is
consistent with remaining transfer truncation and multiple derivatives; it is
not a claim of FD6 interface convergence. Residual reconstruction still makes
curl RMS 14.75 times larger than ordinary transfer at n32 with the same improved
restriction. Both restrictions were changed together for U and E, so this
control identifies their combined effect, not an isolated attribution to only
one buffer. The previous negative baseline is retained unchanged.

Evidence is `qualification-runs-20260907/pcgh-clean-reduction/intrinsic-point-restrict-001/`;
raw files and exact binaries reside under the same named external test directory.
Its `source/manifest.json` hashes the tested production tree and serial/MPI/unit
executables. Use the same runner with `template.athinput` and `--block-n 8/16/32
--orders 6 --dimensions 2`, then the existing analytic budget analyzer. No CUDA,
repeated-stage injection, physical convergence or puncture gate is claimed.
Next isolate the remaining prolongation/residual reconstruction error using the
same signed diagnostics, then test repeated synchronized transfers and evolution.
