# Read-only AthenaK Z4c Candidate-A/C qualification review

You are a senior numerical-relativity and scientific-software reviewer with read-only
access to the AthenaK repository `HengruiZhu99/athenak`. Perform a strictly read-only,
independent assessment of the committed Z4c one-equation shift-gauge experiment and its
runtime evidence. Do not edit files, run evolutions, create artifacts in the checkout,
create commits, open pull requests, or push. Lightweight read-only source inspection and
pure parsing/checksum commands are allowed. Treat repository documentation and prior
claims as hypotheses, not proof.

## Identity to verify first

The review target will be the pushed branch
`codex/z4c-candidate-a-c-gauge-qualification-20260813`. Resolve and report the exact
remote commit yourself before analyzing anything. The compact evidence is under
`docs/evidence/2026-08-13/z4c_candidate_ac_qualification/`. The final evidence commit and
campaign-source SHA are recorded in the committed manifest and report; do not assume a
SHA quoted in this prompt if the branch moved. Report:

- remote URL, branch, commit, parent commit, and submodule SHA;
- whether the tree you inspect is clean and whether the evidence manifest names the same
  source commit, binary hash, input hash, compiler, MPI/OpenMP configuration, and Kokkos
  backend;
- any inconsistency between Git identity, manifest identity, report identity, and raw
  artifact hashes.

The upstream baseline parent is `5f1993109bcb2e5d588ba41b4efc897408e9959a`
(`hengrui/main` at campaign start). The experiment was built with GCC 13.3, double
precision, MPI enabled, Kokkos OpenMP, one MPI rank, and exactly eight OpenMP threads.
Independently verify all of that from the committed evidence.

## Scientific question

Assess whether either opt-in modification gives a robust, resolution-improving stationary
single-Schwarzschild-puncture evolution without compensating numerical changes, and
especially whether Candidate C deserves further work in IrisK.

The baseline production equation is

```text
D0 beta^i = Gamma_tilde^i - eta beta^i,
D0 = partial_t - beta^j partial_j,
eta = 2/M.
```

Candidate A is

```text
D0 beta^i = alpha^2 chi Gamma_tilde^i - eta beta^i.
```

Candidate C is

```text
Q = alpha^2 chi,
G_C(Q) = Q + (1-Q)/(1+Q) = (1+Q^2)/(1+Q),
D0 beta^i = G_C Gamma_tilde^i - eta beta^i
             + (alpha^2/2) gtilde^{ij} partial_j chi
             - alpha chi gtilde^{ij} partial_j alpha.
```

All Candidate-A/C coefficients must use raw positive `chi`, not a guarded or clipped
surrogate. The baseline path must remain bit-for-bit unchanged when the new typed option
is not selected.

## Files and tests to inspect

At minimum inspect and cite exact line numbers from:

- `src/z4c/z4c_shift_gauge.hpp`
- `src/z4c/z4c.hpp`
- `src/z4c/z4c.cpp`
- `src/z4c/z4c_calcrhs.cpp`
- `src/z4c/z4c_newdt.cpp`
- `src/pgen/z4c/z4c_one_puncture.cpp`
- `src/pgen/z4c/z4c_one_puncture_gauge_diagnostics.hpp`
- `tst/unit/z4c_shift_gauge_oracle_test.cpp`
- `inputs/z4c/onepuncture/z4c_candidate_ac_qualification.athinput`
- `scripts/run_z4c_candidate_ac_qualification.py`
- the committed dated report, remote-review prompt, and compact evidence directory named
  by that report.

Review the exact RHS operation ordering. Verify that Candidate A and Candidate C include
the unchanged advective term and damping exactly once. Verify tensor-index placement,
signs, powers of alpha and chi, and whether `g_uu` is the inverse conformal metric rather
than the physical inverse metric. Check all host/device seams and ensure no allocation,
host dispatch, or branch-dependent hidden copy was added inside the Kokkos RHS loop.

Independently derive Candidate C from the stated formula and test the following without
assuming the production helper is right:

- `G_C(1)=1`;
- `G_C -> 1` as `Q -> 0+`;
- `min_{Q>0} G_C = 2(sqrt(2)-1)` at `Q=sqrt(2)-1`;
- positivity and finiteness for every finite `alpha>0`, `chi>0`;
- the exact Candidate-A and Candidate-C pointwise source decomposition;
- zero modified source on exact Minkowski data;
- whether any overflow/cancellation corner remains for large or tiny `Q`.

Audit parser behavior for default selection, unknown values, legacy-key conflicts, and
restart reconstruction. Review the opt-in `reset_dt_from_cfl_on_restart` behavior: exact
terminal landing can persist a tiny remainder timestep, so qualification restarts reset
only the next proposal to `float::max` and let the normal AthenaK CFL computation restore
the spatial step. Decide whether this is scientifically neutral and correctly scoped,
and state how it affects split-versus-uninterrupted reproducibility.

## Runtime matrix and preserved preflights

The immutable primary campaign contains three gauges crossed with three coherent spatial
resolutions:

| ID | root points per axis | MeshBlock points | static refinement levels | dx_min/M |
|---|---:|---:|---:|---:|
| R0 | 8 | 8 | 5 | 0.15625 |
| R1 | 16 | 8 | 5 | 0.078125 |
| R2 | 24 | 8 | 5 | 0.0520833333333333 |

The domain is `[-10M,10M]^3`; the innermost fixed refined cube is
`[-0.6M,0.6M]^3`; `nghost=4`; RK4; CFL 0.1; KO/dissipation 0.02;
`kappa1=0.02`; `eta=2`; raw-chi floor disabled (`chi_div_floor=-1000`,
`floor_chi=false`); identical outer boundaries, initial data, horizon settings, output
cadence, one rank, and eight OpenMP threads. Common exact accepted targets are
`0.1, 0.25, 0.5, 1, 2, 5, 10 M`. The runner schedules target-major, then resolution,
then gauge; only one application can own the global lock.

The report also records two preserved orchestration/preflight roots that are not science
results:

1. A 4-point-MeshBlock attempt stopped at standard R0 T=0.1 because AthenaK's constraint
   history emitted nonfinite values even though accepted lapse/chi/metric telemetry and
   the horizon were finite. This motivated returning to native 8-point blocks, not
   changing gauge physics.
2. A prior target-major root demonstrated that exact landing could persist a subnormal
   remainder timestep across restart; the solver then spent cycles doubling it without
   advancing physical time. This motivated the explicit CFL-recompute-on-restart policy.
3. Two incomplete attempts at Candidate-A R0 T=1 in the primary evidence root were caused
   by API process/session cleanup. Their partial stdout is preserved and they were retried
   with the same source, binary, input, checkpoint, and schedule item under a persistent
   transient service. Do not classify these orchestration interruptions as PDE failures.

Verify that the compact evidence faithfully and explicitly distinguishes all three from
application numerical terminals.

## Primary terminal facts to verify, not assume

The primary terminal records 62 of 63 scheduled segments: 61 complete and one failed.
Standard and Candidate C R0/R1, plus standard R2, reached exact T=10. Candidate A R2's
application reached exact T=10 and returned status 0, but the accepted diagnostic row had
minimum raw chi `-5.555811974786313e-6` and eight invalid points. The runner therefore
classified sequence 61 as `failed` with reason `nonpositive accepted lapse or raw chi`.
It did not launch sequence 62, Candidate C R2 T=10.

There is also one deliberate evidence-integrity disclosure: Candidate-A R2's sequence-61
stderr was empty when the runner hashed it, but MPI appended a 187-byte segmentation-fault
message after the application had otherwise printed normal T=10 termination and returned.
The compact artifact index retains the recorded empty SHA-256, the later SHA-256, and
`hash_valid=false`; it must not rewrite or conceal the mismatch. Determine how this
affects what can be concluded from the failed terminal. All completed-outcome artifacts
must still match their recorded hashes.

The local report classifies Candidate A as failed and the Candidate-C qualification gate
as failed: Candidate C lacks R2 T=10, its R0-to-R1 T=10 horizon mass moves from about
0.9052 to 0.8527 rather than toward one, and gauge/horizon quantities have not settled by
T=10. Treat that as a claim to audit independently. Candidate C's close tracking of the
baseline through T=5 does not override the preregistered all-requirements gate.

## Evidence and artifact audit

Start with the compact evidence directory cited by the report. It must include at least:

- immutable `manifest.json`;
- `terminal.json` and complete outcome index;
- SHA-256 inventory;
- runtime table for every attempted segment;
- accepted-state table at every common time;
- raw Z4c constraint table;
- regional gauge-source table;
- horizon table with finder authority/convergence classification;
- resolution-comparison tables and plots;
- exact validation/build/legacy-equivalence/restart-smoke commands and results;
- selected stdout/stderr/resource/command records sufficient to audit success and any
  terminal failure;
- an index of bulk local-only checkpoints/binaries and their hashes, without committing
  those bulk files.

Fail closed if hashes, row counts, schedule order, source identity, accepted time, or gauge
identity disagree. The mutable cumulative history/gauge/horizon files were appended over
segments. Earlier outcome hashes may therefore represent historical prefixes rather than
the final whole file. Verify them using the committed prefix-hash ledger; do not compare
every historical hash to the final full-file hash.

For every segment and case determine:

- application exit and exact accepted terminal time/cycle;
- wall time, peak RSS, MeshBlocks and effective dt history;
- restart checkpoint path, size, SHA-256, and whether the next segment restored it;
- minimum lapse, raw chi, and physical metric principal minor;
- maximum shift and conformal Gamma; invalid/nonfinite count;
- raw AthenaK C/H/M/Z/Theta history values and their exact normalization semantics;
- coordinate- and proper-volume regional RMS values for Gamma force, chi-gradient force,
  lapse-gradient force, and damping in the global domain, puncture core `r<=0.25M`, AH
  shell `0.25<r<=1M`, and exterior `2<=r<=8M`;
- horizon mass, area, expansion residuals, coordinate radii, and whether FastFlow actually
  set `ah_found` at that accepted target. A numeric summary alone is not sufficient if the
  finder did not converge; use shape/verbose evidence and classify candidate/not-found
  honestly.

Inspect the line profiles of lapse, shift, chi, conformal metric, and K where present.
Separate coordinate gauge expansion from proper-geometry behavior and slice stretching.
Do not call a larger coordinate horizon healthier by itself.

## Convergence and causality questions

At exact common accepted times:

1. Does R0 -> R1 -> R2 improve constraints monotonically? State which norms do and do not.
2. Do same-gauge radial profiles approach one another under refinement? Quantify pairwise
   differences on a common coordinate grid and avoid fitting an order where interpolation,
   puncture nonsmoothness, or outer boundaries dominate.
3. Does horizon mass approach 1, and do area/radius/residual trends improve?
4. Is Candidate A's near-core `alpha^2 chi Gamma` force overwhelmed by `eta beta`? Quantify
   the ratio without dividing by numerical zero and relate it to shift decay or slice
   stretching.
5. Does Candidate C preserve an O(1) Gamma force near the core while its two gradient
   forces remain controlled? Which term balances damping in each region and time window?
6. Does Candidate C remain asymptotically baseline-like in the exterior?
7. Are any observed differences plausibly temporal, boundary, horizon-finder, restart-
   landing, AMR/topology, or finite-duration artifacts rather than gauge effects?
8. If any case terminates early, identify the first accepted evidence of deterioration and
   the exact terminal mechanism. Do not infer causality merely from the last scalar.

Use three resolutions only for a credible trend; do not claim a continuum convergence
order unless the data and common-grid comparison support it. Reaching T=10 is
finite-resolution survival, not a proof of continuum well-posedness or long-time
stability.

## Qualification decision

Independently apply this gate to Candidate C. It passes only if all are supported:

A. all R0/R1/R2 cases reach at least T=10 without NaN/Inf, nonpositive raw chi,
   inadmissible metric/lapse, timestep collapse, or gauge-caused horizon loss;
B. same-time spatial behavior credibly improves with refinement;
C. horizon mass/geometry are consistent with Schwarzschild and improve with resolution;
D. lapse, shift, chi, conformal metric, K, and coordinate horizon size approach a
   stationary or slowly converging state without secular slice stretching;
E. the core avoids Candidate A's vanishing-Gamma-force pathology;
F. the exterior stays asymptotically baseline-like without a persistent outgoing gauge
   pathology;
G. no compensating CFL, dissipation, eta, boundary, refinement, or filter change was used.

Return exactly one independent verdict:

```text
CANDIDATE_C_ATHENAK_GATE = PASS
```

or `FAIL` or `INCONCLUSIVE`, with a requirement-by-requirement table. PASS alone would
authorize a later, separate IrisK design phase. This commit must not contain an IrisK
Candidate-C implementation. If the evidence is incomplete or any gate is unproved, choose
INCONCLUSIVE rather than filling gaps with assumptions.

## Proposed fixes and falsification work

After the verdict, propose ranked next actions. Keep formulation changes separate from
numerical compensation. For each proposed fix or experiment give:

- the concrete equation or code seam;
- why it addresses an observed mechanism;
- what it might break;
- smallest independent source/oracle test;
- smallest bounded runtime falsification case;
- required resolutions/common times/observables;
- fail-closed acceptance and rejection thresholds;
- whether it belongs in AthenaK, IrisK, or both.

Explicitly consider, but do not assume, these possibilities: Candidate-C coefficient or
gradient-term correction; different typed gauge; constraint-preserving source treatment;
outer-boundary/gauge compatibility; horizon-observer artifacts; refinement/topology
effects; timestep/event/restart semantics; and whether evidence instead supports leaving
the baseline unchanged. Do not recommend tuning dissipation, eta, floors, CFL, or filter
strength as a substitute for diagnosing a formulation failure.

List concrete independent tests that could falsify your preferred diagnosis. If you
cannot prove a point from committed source/evidence, state exactly what is missing.

## Required response format

1. Exact repository and evidence identity
2. Source equations independently reconstructed
3. Source/oracle and portability review
4. Runtime artifact integrity audit
5. Case-by-case terminal table
6. Three-resolution constraint/profile/horizon analysis
7. Candidate A mechanism assessment
8. Candidate C mechanism assessment
9. Requirement-by-requirement Candidate-C gate
10. Ranked alternative fixes and falsification experiments
11. Risks and missing proof/evidence
12. Exact final verdict line

Cite exact repository paths and line numbers for source claims, and exact evidence paths,
table rows, and hashes for runtime claims. Clearly label analytic derivation, source fact,
unit-test evidence, runtime observation, inference, and speculation. Do not treat this
prompt or the committed report as proof.
