# Read-only review prompt: Aurora shock-gauge Figure-3 campaign

Please perform a skeptical, source-and-artifact-only review of the following
AthenaK campaign. You have read access only; do not assume that you can run the
code.

Repository: <https://github.com/HengruiZhu99/athenak>

Branch: `codex/z4c-vc-reference-shock-gauge-figure3-20260828`

Immutable evidence commit:
`e8bb8a8658c193294cce4ab096a3de5fb0c6a2a5`

Source-fix commit:
`f8303c6be7eb214fa1e91b646123ee0d434b3698`

## Governing artifacts

- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/REPORT.md`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/EVIDENCE_MANIFEST.json`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/AURORA_PVC_TESTS.md`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/N256_REPRODUCTION.md`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/CONVERGENCE.md`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n256/n256_summary.json`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n256/figures/figure3_with_constraint_validity.png`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n256/figures/constraints_and_rho5_mode.png`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/analysis/aurora_n256/figures/timestep_refinement_meshblocks.png`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/evidence/aurora/qualification_8789659/`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/evidence/aurora/bisect_8789460/`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/evidence/aurora/n256_reference_shock_seg000_retry1/`
- `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/evidence/aurora/authority/n256_reference_shock_authority.jsonl`

Relevant source includes `src/z4c/z4c.cpp`,
`src/z4c/cartoon_meridional_sampler.hpp`, the native vertex-centered Z4c AMR
transfer/boundary code, and the refinement-history machinery.

## Established observations

Aurora qualification job `8789659` passed 11 focused tests and a two-cycle
shock-avoiding/prescribed-zero-shift smoke on one PVC. A preceding device
bisect, job `8789460`, found that the state-admissibility Kokkos lambda had
implicitly accessed host-owned `Z4c::opt`; commit `f8303c6b` copies the needed
boolean policy before the lambda.

N256 science job `8789703` used one PVC charged to `CompactBinaryMerger` and
reached coordinate time `30 M` in 5791 cycles. It used native VC Cartoon SO(2),
O4 bulk derivatives, q6 prolongation, RK4/CFL 0.15, KO 0.5,
`dchi_max=0.02`, derefinement at `0.25*dchi_max`, shock-avoiding lapse with
`kappa=1`, prescribed zero shift, and no Z4c damping.

Key N256 values:

- imported ADM mass: `2.6606354586228815`;
- final central proper time: `11.2863067801 M`;
- minimum sampled axis lapse: `0.1434111145`, no zero crossings;
- apparent curvature peak: proper time `10.30333 M`,
  `log10(abs(Kretschmann))=5.01349`;
- published first peaks: proper time `10.30683--10.31384 M`, amplitude
  `5.47778--5.48688`;
- maximum C and H squared-integrals: `48.2330` and `41.1314`;
- C first crosses `0.01`, `0.1`, `1`, and `10` at coordinate times
  `20.12825`, `21.99447`, `23.37376`, and `24.40678 M`;
- the curvature maximum occurs during this constraint catastrophe;
- the hierarchy reaches 212 MeshBlocks at physical level 5 in the first two
  cycles, then remains fixed through `t=30 M`;
- later refinement requests are zero while derefinement requests are rejected;
- the run ends before the published deep minimum/rebound region;
- no horizon finder was enabled.

The Cartoon constraint-history measure is already
`2*pi*rho*dx1*dx2*sqrt(abs(det gamma))`, with native-VC trapezoid weights and
canonical shared-node ownership. The observed growth is not a fictitious
collapsed-y normalization artifact.

N128/N512 replay was intentionally not run because N256 failed the scientific
Figure-3 gate. There is no convergence result.

## Questions

1. Audit the device-capture repair narrowly. Is copying the gauge-dependent
   admissibility boolean before the Kokkos lambda sufficient, or is another
   host-owned option/reference still reachable on the PVC path?
2. Given that the hierarchy becomes fixed almost immediately, what mechanisms
   best explain late constraint/curvature growth: bulk O4/gauge evolution, a
   persistent coarse-fine or block-interface mode, a short scale missed by the
   chi-only sensor, or an interaction among them?
3. Why can the central-curvature curve agree closely at early proper time and
   peak at almost the published time while the peak itself is constraint
   invalid? Does any scientifically useful Figure-3 statement survive beyond
   descriptive early-time agreement?
4. Inspect the refinement and VC Cartoon boundary/transfer source for a
   concrete coherence, ownership, parity, stale-stage, or indexing defect that
   is consistent with the artifacts. Separate an actual source finding from a
   plausible mechanism.
5. Does the absence of later chi-triggered refinement plausibly indicate that
   another Z4c variable becomes under-resolved first? If so, which fields and
   which cheap local indicator should be audited first?
6. Propose the single smallest decisive next diagnostic or source-level
   correction. Prefer a bounded replay or offline/location-resolved diagnostic
   on an existing pre-runaway state, with an explicit decision table.

Please return:

- a concise verdict;
- observations, inferences, and hypotheses in separate sections;
- any specific source locations and equations supporting a bug claim;
- the smallest decisive next diagnostic and its pass/fail interpretation;
- the narrowest correction only if the existing code establishes it.

Do not recommend chi floors, clipping, weakened positivity/admissibility gates,
broad gauge/KO/CFL/AMR parameter sweeps, or another long production run as the
first step. Do not claim convergence, a full Figure-3 reproduction, critical
behavior, horizon formation/absence, or a unique formulation instability from
the current evidence.
