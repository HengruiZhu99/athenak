# Read-only external review prompt

Please perform a skeptical, source-grounded, **read-only** review of the vertex-centered Cartoon/Z4c Brill qualification in:

```text
Repository: https://github.com/HengruiZhu99/athenak
Branch: codex/z4c-vc-brill-transfer-qualification-20260823
Evidence/report commit: ba8646c631941a13c5d40267712081473c213f63
Source/test commit used for the runs: 278b63a740a947de55ad8bdd1c333095c68fedcd
Exact base: 2d59f85c11cb0da4614c84a695d64f032fb9eec7
```

The only intended branch change after the evidence/report commit is this reviewer prompt.

Read these first:

```text
docs/investigations/z4c_vc_brill_transfer_qualification_20260823/REPORT.md
docs/investigations/z4c_vc_brill_transfer_qualification_20260823/TRANSFER_SELECTION.md
docs/investigations/z4c_vc_brill_transfer_qualification_20260823/BACKEND_MATRIX.md
docs/investigations/z4c_vc_brill_transfer_qualification_20260823/VERDICTS.md
docs/investigations/z4c_vc_brill_transfer_qualification_20260823/EVIDENCE_MANIFEST.md
```

The root artifact checksum file is:

```text
docs/investigations/z4c_vc_brill_transfer_qualification_20260823/EVIDENCE_MANIFEST.sha256
SHA256: 3069f2196e9bc78658efa7c64bc48ba3401f5684ab1919fd22830e15384fcb78
```

Key numerical evidence:

1. Matched q4 VC prolongation is suborder for O4 Z4c. Its interface state defect is O4, but the worst second-derivative RHS defect is approximately O2. Dynamic nonconstant-AMR convergence is only 2.16–2.73 order.
2. Elevated q6 prolongation restores approximately O4 RHS/interface behavior and 4.12–4.56 order dynamic-AMR convergence. It is the only surviving transfer candidate, but it was not promoted because the physical prerequisite failed.
3. Current-source CUDA, two-rank MPI, shared-node, migration, and q4/q6 restart tests pass; accepted restart states are bit-exact. Current-source SYCL is pending.
4. Direct VC IrisK initial data at N128/N256/N512 have exact common-node agreement for all 22 directly assigned state fields, approximately O3.95 discrete Gamma compatibility, zero shared-node spread, positive chi/SPD pivots, and roundoff algebraic axis regularity. The raw Hamiltonian norm approaches a roughly `3.3e-4` floor and is origin/seam limited.
5. The no-AMR fixed-grid Brill sequence reaches `tau_c=3.08 M` at all three resolutions but becomes nonconvergent. At `tau_c=3`, proper ring RMS C is `0.1837, 0.5701, 1.4908`; H, M, and Z also increase with resolution.
6. Terminal region-resolved differences show mostly O2–O4 behavior in `r<=8`, better constraint orders away from MeshBlock seams, but negative orders on the symmetry axis and at the outer boundary. State/RHS/constraint duplicate shared values remain bitwise exact.
7. The global Cartoon history measure is already the proper axisymmetric ring measure, `2*pi*rho*sqrt(det(gamma))*drho*dz` with nodal trapezoid weights. There is no fictitious collapsed-y `dx3` factor.

Primary source areas to audit:

```text
src/z4c/z4c_rhs.hpp and derivative providers used by Cartoon VC
src/z4c/z4c_Sbc.cpp
src/z4c/z4c_tasks.cpp
src/z4c/z4c_axis*.hpp/cpp and parity/regularization helpers
src/z4c/z4c_history_quadrature.hpp
src/outputs/history.cpp
src/bvals/bvals_vc.cpp
src/mesh/vertex_amr.hpp
src/mesh/mesh_refinement_vc.cpp
```

Please answer the following, separating **observation**, **mathematical inference**, and **hypothesis**:

1. Is the report's conclusion logically sound that AMR transfer cannot be the sole cause because fixed-grid evolution already loses convergence?
2. Do you find any source-level inconsistency in the VC Cartoon axis parity, `1/rho` regularization, derivative closure, algebraic projection, RHS boundary ordering, or shared-vertex ownership that can explain negative axis orders despite bitwise duplicate agreement?
3. Is the nonconvergent outer-boundary behavior consistent with the implemented Sommerfeld treatment and its tensor/gauge characteristic assumptions, or is there an evident coding/centering defect?
4. Could the global worsening be explained entirely by an outward boundary layer, or do the terminal core/RHS orders prove an independent central/axis problem? Please use the committed CSVs rather than intuition.
5. Does q6 remain the mathematically appropriate candidate after the physical gate is repaired, or is there a reason the synthetic evidence is insufficient or misleading?
6. What is the **smallest decisive next diagnostic or narrowly scoped source correction** that identifies the first offending component, RHS term, writer, and physical location without another long run?

Please scrutinize in particular whether a value can be canonical/shared-node exact yet have a wrong axis derivative because parity or regularized limits are applied inconsistently. Also check whether the `rho=0` diagnostic measure and the separate unweighted axis census are interpreted correctly.

Do not:

- suggest a chi floor, clipping, weakened positivity gate, or relaxed restart equality;
- propose a broad gauge/KO/CFL/damping/AMR parameter sweep;
- attribute the result to AMR transfer alone without reconciling the fixed-grid evidence;
- recommend production q6 selection before a corrected fixed-grid Brill gate and common-tree comparison;
- claim convergence, a Figure-3 reproduction, horizon formation, critical behavior, or physical self-similarity;
- modify code or request new production runs.

Return a concise verdict, the most likely code paths ranked by evidence, any concrete formula/indexing errors you can identify, and one bounded next experiment or patch with explicit pass/fail criteria.
