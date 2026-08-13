# AthenaK Z4c Candidate-A/C single-puncture qualification — 2026-08-13

Status: **final fail-closed AthenaK report.**

Claim labels used below are **analytic derivation**, **source fact**, **unit/oracle
result**, **runtime observation**, **inference**, and **hypothesis**. A label applies only
to the sentence or paragraph carrying it.

## 1. Repository identities

- **Source fact:** AthenaK task branch
  `codex/z4c-candidate-a-c-gauge-qualification-20260813` was created from upstream
  `5f1993109bcb2e5d588ba41b4efc897408e9959a`.
- **Source fact:** the immutable science campaign used source
  `39e6372fcc7c2ba166a8498a007c858cfde73b6c` and Kokkos submodule
  `6739bc623081648af9e752b616d9671527922cbf`.
- **Runtime evidence:** the executable SHA-256 was
  `c321afba740e65b524b3e2a213d68543e2d7d2b20775496fa9360ed4bc22824f`; the input
  SHA-256 was `35afbc6055067147011013758b6c2481cd176ab317f5834c5df434a295c83268`.
- **Runtime evidence:** GCC 13.3, IEEE double precision, MPI enabled, Kokkos OpenMP,
  one MPI rank, and exactly eight OpenMP threads were frozen in `manifest.json`.

The final documentation/evidence commit is a descendant of the campaign source commit.
The manifest continues to identify the exact source and binary that produced the data.

## 2. AthenaK baseline gauge

**Source fact:** the unchanged default driver is

\[
  D_0\beta^i=\widetilde\Gamma^i-\eta\beta^i,
  \qquad D_0=\partial_t-\beta^j\partial_j,
\]

with \(\eta=2/M\) in this campaign. The source/oracle validation compared a baseline
one-cycle run against the unmodified upstream binary and found byte-identical `co_0.txt`
and Z4c history output. The new profiles are opt-in typed values; the default branch keeps
the original arithmetic ordering.

## 3. AthenaK Candidate-A implementation

**Source fact:** Candidate A is

\[
  D_0\beta^i=\alpha^2\chi\widetilde\Gamma^i-\eta\beta^i.
\]

It uses raw positive \(\chi\). This is the originally proposed lapse/chi-suppressed
Gamma driver and is retained as a diagnostic comparator, not presumed healthy.

## 4. AthenaK Candidate-C implementation

**Source fact:** define \(Q=\alpha^2\chi\) and

\[
 G_C(Q)=Q+\frac{1-Q}{1+Q}=\frac{1+Q^2}{1+Q}.
\]

Candidate C is

\[
 D_0\beta^i=G_C\widetilde\Gamma^i-\eta\beta^i
 +\frac{\alpha^2}{2}\widetilde\gamma^{ij}\partial_j\chi
 -\alpha\chi\widetilde\gamma^{ij}\partial_j\alpha.
\]

Here `g_uu` is the inverse conformal metric. Candidate C also uses raw positive \(\chi\).
The typed parser rejects unknown profiles and conflicting legacy coefficient overrides.

## 5. AthenaK source/oracle verification

**Analytic derivation:** \(G_C(1)=1\), \(G_C\to1\) as \(Q\to0^+\), and
\(G_C>0\) for \(Q>0\). Differentiation gives its minimum
\(2(\sqrt2-1)\) at \(Q=\sqrt2-1\).

**Unit/oracle result:** `z4c_shift_gauge_oracle_test` independently checks the exact
Candidate-A/C source formulas, coefficient extrema/limits, Minkowski behavior, finite
positive inputs, parser identity, invalid values, and the preserved baseline expression.
The application, oracle, default-equivalence check, candidate smokes, terminal horizon,
and split restart were validated before the immutable campaign.

## 6. AthenaK single-hole resolution campaign

The common configuration is a Schwarzschild wormhole puncture on
\([-10M,10M]^3\), MeshBlocks of \(8^3\) points, five fixed-refinement levels, inner
refined cube \([-0.6M,0.6M]^3\), four ghost zones, RK4, CFL 0.1, KO/dissipation 0.02,
\(\kappa_1=0.02\), \(\eta=2\), no raw-chi floor, identical boundaries and outputs.

| level | root grid | minimum spacing |
|---|---:|---:|
| R0 | 8^3 | 0.15625 M |
| R1 | 16^3 | 0.078125 M |
| R2 | 24^3 | 0.0520833333333333 M |

All nine gauge-resolution cases were scheduled target-major at exact accepted times
0.1, 0.25, 0.5, 1, 2, 5, and 10 M. Only one foreground evolution owned the global lock.

### Runtime terminal table

The 62 recorded segments consumed 13,817.4 s (3.838 h) of serialized application wall
time. There were 61 validated complete segments, one rejected accepted-state terminal,
and one schedule item not attempted after the fail-closed stop.

| case | T=10 classification | min lapse | min raw chi | max shift | max Gamma | AH mass | AH mean radius | wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard R0 | complete | 5.9668e-2 | 1.0959e-2 | 0.21191 | 0.42180 | 0.90679 | 0.87064 | 241.4 |
| Candidate A R0 | complete | 1.5499e-2 | 1.3453e-3 | 0.05765 | 9.95861 | 0.76129 | 0.50740 | 254.4 |
| Candidate C R0 | complete | 5.8061e-2 | 1.1005e-2 | 0.21180 | 0.44901 | 0.90521 | 0.89039 | 236.7 |
| standard R1 | complete | 3.3875e-2 | 2.9482e-3 | 0.21178 | 0.42152 | 0.86328 | 0.75411 | 630.4 |
| Candidate A R1 | complete | 1.5058e-2 | 3.8135e-5 | 0.05765 | 9.93041 | 0.63073 | 0.59574 | 616.3 |
| Candidate C R1 | complete | 3.3092e-2 | 2.9545e-3 | 0.21168 | 0.44880 | 0.85269 | 0.74576 | 623.4 |
| standard R2 | complete | 2.5379e-2 | 1.3398e-3 | 0.21178 | 0.42153 | 0.85881 | 0.74268 | 1707.8 |
| Candidate A R2 | rejected terminal | 2.1614e-3 | **-5.5558e-6** | 0.05766 | 9.92702 | not authoritative | not authoritative | 1655.7 |
| Candidate C R2 | not attempted after stop | — | — | — | — | — | — | — |

**Runtime observation:** Candidate A R2's application reached exact T=10 and returned
status 0, but the runner rejected the accepted slice because raw chi was negative at eight
points. No T=10 Candidate-A R2 checkpoint or horizon is promoted. MPI subsequently wrote
a 187-byte segmentation-fault message to stderr after the runner had hashed the initially
empty stream; this is retained as one explicit outcome-artifact hash drift, not hidden or
repaired. Sequence 62 (Candidate C R2 to T=10) was never launched.

## 7. AthenaK gauge-relaxation comparison

At T=5, Candidate C was numerically very close to the baseline at every resolution. For
example, `(baseline, Candidate C)` AH masses were `(0.889215, 0.889257)` at R0,
`(0.884680, 0.884738)` at R1, and `(0.882111, 0.882170)` at R2. Maximum shifts differed by
less than 0.001 and maximum conformal Gamma by about 0.0065. Candidate A instead had
maximum Gamma about 4.83–4.94 and maximum shift about 0.031, versus about 0.35 and 0.154
for the baseline.

**Runtime observation:** native constraint histories improved enormously from R0 to R1
and again to R2 at T=5 for all gauges. Candidate C tracked the baseline closely. Candidate
A was qualitatively different: at R0 its T=5 collective/Hamiltonian native values were
33.48/27.13 versus baseline 0.583/0.387, while at R1 its Hamiltonian value was 2.15e-4
versus baseline 4.57e-5. This is evidence of strong resolution sensitivity, not a fitted
continuum order.

**Runtime observation:** by T=10, Candidate C R0/R1 remained close to their same-resolution
baseline in state and native constraints, but neither was approaching unit horizon mass
under refinement: Candidate-C mass changed from 0.9052 at R0 to 0.8527 at R1. The lapse,
shift, and coordinate horizon radius also continued to change substantially from T=5 to
T=10. Candidate C therefore did not demonstrate the required stationary, convergent black-
hole geometry in the tested window.

The compact bundle contains exact state/constraint/source/horizon tables, common-grid
profile differences, resolution trends, gauge-versus-baseline differences, and four PNG
plots. The last common full-matrix profile plot is T=5 because Candidate C R2 T=10 is
absent.

The committed compact evidence preserves native AthenaK constraint labels and
normalizations. It does not relabel them as continuum norms. Coordinate horizon radius is
reported separately from mass/area and is never used alone as a health criterion.

## 8. AthenaK Candidate-A verdict

**FAIL.** Candidate A's Gamma forcing is strongly suppressed near the puncture relative to
the baseline, its shift remains much smaller, conformal Gamma grows to about 10, native
constraints and horizon estimates degrade, and R2 develops negative raw chi by T=10.

## 9. AthenaK Candidate-C verdict

**FAIL as a qualification candidate.** Candidate C survives at R0/R1 to T=10 and follows
the baseline closely through T=5 at all three resolutions. Nevertheless, the required
gate is not met: Candidate C R2 T=10 is missing, R0→R1 T=10 horizon mass worsens away from
one, and gauge/horizon profiles have not relaxed to a stationary state by T=10. These are
gate failures, not evidence that Candidate C's source formula itself caused the baseline-
like mass drift.

## 10. AthenaK moving-hole result, if reached

No moving-hole test is authorized unless the stationary Candidate-C gate is a clear PASS.

## 11. ATHENAK GATE: PASS / FAIL / INCONCLUSIVE

| gate | result | evidence-based reason |
|---|---|---|
| A survival | **not met** | Candidate C R2 T=10 was not attempted after the Candidate-A failure. |
| B spatial behavior | **not met at final target** | T=5 constraints improve, but no complete three-resolution T=10 Candidate-C comparison exists. |
| C black-hole geometry | **failed** | Candidate-C AH mass is 0.9052 at R0 and 0.8527 at R1 at T=10, worsening away from unit mass. |
| D gauge relaxation | **failed/incomplete** | lapse, shift, chi, and coordinate AH radius remain strongly time-dependent from T=5 to T=10. |
| E puncture core | **provisionally met through T=5** | Candidate C avoids Candidate A's vanishing Gamma coefficient and tracks baseline source magnitudes; no R2 T=10 evidence. |
| F exterior | **provisionally met through available times** | Candidate C remains close to baseline in the exterior diagnostics; the finest final target is absent. |
| G no compensation | **met** | all gauges use identical grid hierarchy, CFL, RK4, dissipation, damping, boundaries, outputs, and resources. |

Because every item is required, the result is FAIL. It is not promoted based on survival
of two resolutions.

## 12. IrisK implementation, only if AthenaK gate = PASS

No IrisK source was changed during the AthenaK experiment. A FAIL or INCONCLUSIVE gate
is a hard stop.

## 13. IrisK Candidate-C continuum derivation

Not reached in this AthenaK-only phase.

## 14. IrisK characteristic/symmetrizer/SAT verification

Not reached in this AthenaK-only phase.

## 15. IrisK timestep/CFL verification

Not reached in this AthenaK-only phase.

## 16. IrisK R0/R1/R2 evolution evidence

Not reached in this AthenaK-only phase.

## 17. IrisK stationary-trumpet result

Not reached in this AthenaK-only phase.

## 18. IrisK moving-hole result

Not reached in this AthenaK-only phase.

## 19. Remaining mathematical and numerical risks

- **Limitation:** T=10 is a finite-resolution survival screen, not proof of continuum
  well-posedness, asymptotic stationarity, or long-time stability.
- **Limitation:** three resolutions support a trend only where same-time observables and
  common-grid profile differences improve consistently; survival alone is not convergence.
- **Limitation:** exact-target landing plus restart intentionally recomputes the ordinary
  spatial CFL timestep rather than persisting the tiny landing remainder. This changes the
  split step sequence and is not bitwise split/uninterrupted equivalence.
- **Limitation:** FastFlow numerical summaries must be paired with actual found/shape
  evidence; a number without finder convergence is not an authoritative horizon.
- **Limitation:** the compact commit excludes multi-gigabyte restarts and full binary
  volumes. Their paths, sizes, and SHA-256 hashes remain indexed.

## 20. Exact files changed

Source and qualification files before the evidence commit:

- `src/z4c/z4c_shift_gauge.hpp`
- `src/z4c/z4c.hpp`
- `src/z4c/z4c.cpp`
- `src/z4c/z4c_calcrhs.cpp`
- `src/pgen/z4c/z4c_one_puncture.cpp`
- `src/pgen/z4c/z4c_one_puncture_gauge_diagnostics.hpp`
- `tst/unit/z4c_shift_gauge_oracle_test.cpp`
- `inputs/z4c/onepuncture/z4c_candidate_ac_qualification.athinput`
- `scripts/run_z4c_candidate_ac_qualification.py`
- `CMakeLists.txt`

The final evidence commit additionally adds the analyzer, this report, the remote review
prompt, and the compact dated evidence bundle.

## 21. Exact commands executed

See `validation/commands_and_results.md`, the manifest, and every segment `command.txt`.
The immutable campaign command was:

```bash
python3 scripts/run_z4c_candidate_ac_qualification.py \
  --repo /home/hzhu/athenak-candidate-a-c-20260813 \
  --binary /home/hzhu/build-athenak-candidate-a-c-openmp/src/athena \
  --run-root /home/hzhu/athenak-candidate-ac-evidence-r3-20260813 --all
```

## 22. Exact tests and outcomes

The pre-campaign build/oracle, default-equivalence, invalid parser, Candidate-C cycles,
terminal horizon, and restart-dt-reset evidence is committed. Post-campaign validation
ran the hash-bound analyzer, repeated it deterministically, reran the focused oracle,
rebuilt the application, and ran `git diff --check`; exact commands/results are in the
validation record.

## 23. Evidence locations

Committed compact evidence:
`docs/evidence/2026-08-13/z4c_candidate_ac_qualification/`.

Local bulk authority (not committed):
`/home/hzhu/athenak-candidate-ac-evidence-r3-20260813/`.

The compact bundle includes manifest, terminal, outcomes, raw text evidence, selected
profiles, derived tables/plots, prefix-hash ledger, artifact index, and SHA-256 inventory.
It excludes restart checkpoints and full binary volume dumps while preserving their hashes.

Two earlier preflight/orchestration roots are retained locally and described in the
validation record; they are not promoted as science results.

## 24. Final recommendation

Do not implement Candidate C in IrisK from this evidence. Preserve it as a typed AthenaK
experiment. The next work should first explain the baseline-like horizon-mass drift and
continued gauge relaxation, then run a separately immutable Candidate-C R2 completion
only if justified. Candidate A should not advance without addressing its negative-chi
terminal and suppressed core forcing. The committed remote prompt asks an independent
read-only reviewer to audit the equations, artifacts, convergence evidence, failure
mechanisms, and rank alternative fixes with falsification tests.

```text
CANDIDATE_C_ATHENAK_GATE = FAIL
```
