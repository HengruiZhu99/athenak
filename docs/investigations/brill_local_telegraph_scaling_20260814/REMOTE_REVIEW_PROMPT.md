# Remote review prompt: scale-invariant Brill gauge damping

You are an expert read-only reviewer of a numerical-relativity experiment in
AthenaK.  You have repository access but cannot compile or run anything.  Use
source inspection, dimensional analysis, PDE reasoning, and the committed
plots/tables to audit the result and recommend the smallest mathematically
discriminating next step.  The bundle now contains four telegrapher-scale
comparisons and a one-parameter fixed-shift control.  Do not propose a blind
parameter sweep.

## Repository identity

Repository: `https://github.com/HengruiZhu99/athenak.git`

Branch: `codex/cartoon-allbulk-brill-scaleinv-20260813`

The telegrapher implementation is commit
`2a8ad80e02279769a99fe279b7a33516bc6c8d0d`, tree
`67709c405a1169a15643cb933eec5353cd216243`, parent
`ae3a817e26f153c135f924648594b8639e7d60bf`.  Kokkos is
`6739bc623081648af9e752b616d9671527922cbf`.

Read the bundle at
`docs/investigations/brill_local_telegraph_scaling_20260814/`, beginning with:

1. `figure3_shift_damping_report.pdf`
2. `RESULTS.md`
3. `data/analysis_summary.json`
4. `data/fixed_shift_eta2_comparison.json`
5. `figures/figure3_local_telegraph_mu_comparison.png`
6. `figures/figure3_fixed_shift_eta2_overlay.png`
7. `figures/fixed_shift_eta2_stiffness_and_gauge.png`
8. `figures/fixed_shift_eta2_constraint_comparison.png`
9. `figures/fixed_shift_eta2_trajectory_and_amr.png`
10. the original lapse, mu-extrema, and spatial-profile plots

The complete reduced temporal series used by the plots is
`data/history_curves.csv`; selected raw-AthenaK spatial profiles are in
`data/selected_mu_profiles.json`.  The fixed-shift reduced history, sampled
gauge extrema, AMR events, exact command, and final interpretation are the
`data/fixed_shift_eta2_*` files.  The overlay can be regenerated with
`scripts/plot_fixed_eta_control.py`.

Then inspect these source and test files:

- `src/z4c/telegraph_damping.hpp`
- the telegrapher block in `src/z4c/z4c_calcrhs.cpp`
- option migration in `src/z4c/z4c.cpp` and `src/z4c/z4c.hpp`
- mu history/output plumbing in `src/outputs/history.cpp`,
  `src/outputs/basetype_output.cpp`, and `src/outputs/outputs.hpp`
- `tst/unit/z4c/z4c_telegraph_damping_test.cpp`
- `tst/unit/z4c/z4c_telegraph_damping_static_test.py`
- `tst/inputs/z4c_telegraph_damping.athinput`
- the exact production input in
  `docs/investigations/brill_local_telegraph_scaling_20260814/data/figure3_input.athinput`

For broader context, inspect the earlier tests and investigation material
already in this branch, especially:

- `docs/investigations/cartoon_half_plane_kerr_default_gauge_20260813/`
- `docs/z4c_cartoon_half_plane_design.md`
- `tst/unit/z4c/cartoon_axis_boundary_test.cpp`
- `tst/unit/z4c/cartoon_axis_centered_derivatives_test.cpp`
- `tst/unit/z4c/cartoon_axis_parity_test.cpp`
- `tst/unit/z4c/cartoon_regular_functionals_test.cpp`
- `tst/unit/z4c/z4c_kerr_half_plane_*`
- `tst/test_suite/z4c/cartoon_half_plane_kerr_campaign.py`
- `tst/test_suite/z4c/cartoon_half_plane_kerr_convergence.py`
- `tst/unit/z4c/z4c_shock_avoiding_gauge_static_test.py`

## Fixed experiment

All four cases use the same Figure-3 Brill data and evolution settings:

- Brill amplitude `A=-0.047`, ADM mass `2.660301967997158`.
- Direct interpolation in AthenaK from the included IrisK global 48x32
  coefficient payload; no M0 intermediate-domain wiring.
- Pre-collapsed lapse `alpha=psi^-2`.
- Half-plane `rho>=0` Cartoon SO(2), N128, O6, RK4, CFL 0.15.
- Nine AMR levels total, `dchi_max=0.02`, regrid every cycle.
- KO dissipation 0.02, no chi floor, no clipping.
- Telegrapher lapse plus scale-invariant Gamma-driver shift.
- Z4c damping remains enabled with the existing scale-invariant max-K rule.
- `telegraph_tau=1`, `telegraph_kappa=1`, target `t=20M`.
- One A100 40 GB, one rank, identical source/executable/input except for the
  runtime telegrapher damping prescription.

The four physical inverse-length damping fields were fixed prospectively:

1. `mu=max_domain |K|` (legacy/fresh baseline)
2. `mu(x)=|K(x)|`
3. `mu(x)=sqrt(K_ij K^ij)`
4. `mu(x)=sqrt(gamma^ij partial_i chi partial_j chi)`

## Scale-invariant algebra to audit

Let `Kstar=max_domain |K|`.  The implemented conceptual parameters are

```text
Q(x)       = mu(x)/Kstar
tau_eff    = tau/Kstar
kappa_eff  = kappa/Kstar.
```

The evolution kernel evaluates the algebraically cancelled coefficients

```text
Q/tau_eff          = mu/tau
kappa_eff/tau_eff  = kappa/tau,
```

so the auxiliary covector obeys, schematically,

```text
(partial_t - L_beta) B_i = -(mu/tau) B_i + (kappa/tau) partial_i alpha,
```

while the lapse equation contains `chi div(B)` in addition to the existing
gauge terms.  Scaling tau and kappa together preserves their ratio and hence
the principal telegrapher speed.  The cancelled form supplies a finite
extension through a time-symmetric slice where `Kstar=0`.

Audit all of the following rather than accepting the comments at face value:

- the dimensions of alpha, B_i, chi, K, tau, kappa, Q, and coordinate time;
- whether this is genuinely invariant under a global mass/length rescaling;
- whether the `chi` factor in the lapse equation changes the characteristic
  speed or the proper interpretation of tau and kappa;
- whether `sqrt(K_ij K^ij)` is formed correctly from conformal A, conformal
  inverse metric, and K, with no missing chi factor;
- whether the physical chi-gradient norm uses the correct inverse physical
  metric for the configured `chi_psi_power` convention;
- whether evaluating the cancelled ratios at `Kstar=0` is a legitimate
  continuous extension for all four mu choices;
- whether a spatially varying, potentially vanishing damping field preserves
  strong hyperbolicity and admits a useful energy estimate for the coupled
  lapse-B subsystem.

## Authenticated observations

Job `56955603` completed its configure, build, and focused test steps.  All
four scientific cases then stopped at the unchanged strict-positive chi
boundary-prolongation gate:

| mu prescription | terminal t/M | terminal central tau/M | rejected parents |
|---|---:|---:|---:|
| domain max|K| | 10.162940 | 6.201069 | 6,412 |
| local |K| | 8.907275 | 5.472376 | 13,184 |
| local sqrt(KijKij) | 9.193945 | 5.751011 | 12 |
| local |grad chi| | 8.358252 | 5.016554 | 7,320 |

The baseline has 854 rows and reproduces the preceding run's 22 primary
history columns bit-for-bit, including the same terminal time and same 6,412
parent failure.  At the nearest `t=8M` rows, combined constraint norms are
approximately 101, 2.61e4, 3.20e3, and 77.7 respectively in the table order.
The last value is temporarily smaller but that case fails about `0.36M` later.

Selected spatial mu outputs contain no negative or nonfinite values.  Local
`|K|` is visibly jagged around moving K-zero surfaces and has a noisy tiny
domain minimum; the extrinsic norm is smoother; the chi-gradient norm develops
the earliest strong localized maximum.  Treat these as observations, not a
causal proof.

The native output manifest and campaign summary hashes are recorded in
`RESULTS.md` and `data/analysis_summary.json`.  The Git bundle deliberately
does not include multi-gigabyte native binaries.

## Fixed-shift control

After the four-mode campaign, one controlled diagnostic tested whether the
scale-invariant Gamma-driver shift damping, rather than the telegrapher lapse
alone, generated the baseline stiffness.  Job `56966125` reused the exact
source, executable, Figure-3 input, initial-data payload, hardware class, and
all numerical settings.  The sole physics change was:

```text
z4c/shift_eta_max_K=true -> false
```

Thus the baseline has `eta_shift=2 max|K|`, and the control has fixed
`eta_shift=2`.  The telegrapher remains `max_domain_abs_K` with
`tau=kappa=1`; scale-invariant Z4c damping remains enabled.

Define `S_beta=dt eta_shift`.  The baseline's final recorded value is
`5.55889`, above the approximate RK4 negative-real-axis limit `2.8`; the fixed
control never exceeds `0.075`.  At the baseline failure time the fixed control
has combined constraint norm `136.568` rather than `2.4013e11`, `max|K|=3.49`
rather than `1.8974e4`, sampled `max|Gamma|=0.237` rather than `1288`, and 32
rather than 290 MeshBlocks.

The fixed control avoids the baseline chi rejection and reaches
`t=16.73892M`, central proper time `10.37631M`, a `64.7%` coordinate-time
extension.  It then fails a different strict post-RK axis-central diagnostic.
The last finite row already has `max|K|=1.0689e4`, axis Kretschmann
`7.6154e15`, and combined constraint norm `8.3439e7`.  The frozen telemetry
does not distinguish whether lapse, constraint density, or curvature
reconstruction becomes invalid first.

Interpret this carefully: the control strongly implicates max-K-scaled shift
damping as a major contributor to the baseline stiffness, but fixed eta is
dimensionful, is not the target algorithm, and is not robust.  It is neither a
Figure-3 reproduction nor a qualified replacement gauge.

## Hard review constraints

- Do not recommend a chi floor, clipping, relaxed finite/positivity gate, or
  retrospectively increased tolerance.
- Do not recommend tuning tau/kappa after selecting a preferred outcome.
- Do not call reaching a later failure time convergence or qualification.
- Do not treat different rejected-parent counts as a stability ordering; the
  cases reach the fatal check at different states.
- Do not infer that local damping caused the strict-chi failure merely from
  temporal correlation.
- Keep the original domain-max result as the accepted comparator.
- Do not promote fixed `eta=2` merely because it runs farther.
- Do not equate later coordinate-time failure with convergence or reproduction.
- Do not assume the late fixed-eta central failure has the same cause as the
  baseline chi rejection.
- Prefer a source-level or mathematical discriminator that a future agent can
  test prospectively in one bounded run.

## Questions to answer

1. **Scaling audit.** Is the simultaneous max-K scaling of Q, tau, and kappa
   dimensionally and mathematically correct?  Derive the characteristic speeds
   and damping rates of the frozen-coefficient lapse-B subsystem, including
   the chi factor and the variable-mu case.

2. **Invariant definitions.** Verify the two local norms implemented in
   `telegraph_damping.hpp`.  Identify any missing conformal factor, index
   contraction error, sign issue, or coordinate-versus-physical norm mismatch.

3. **Why local choices lose.** Rank explanations supported by the evidence:
   vanishing damping at K-zero surfaces, sharp coefficient variation and AMR
   sampling, insufficient damping across most of the domain, excessive local
   stiffness near peaks, coupling to the scale-invariant shift/Z4c damping,
   or an unrelated chi/AMR instability that gauge changes merely shift in time.

4. **Smooth scale design.** Is there a better local or quasi-local inverse
   length that is smooth, positive without a dimensionful floor, and invariant
   under mass rescaling?  Consider curvature invariants, expansion/shear
   combinations, covariant spacetime norms, or a normalized spatial average.
   State explicitly how zeros and time symmetry are handled.  Do not merely
   propose `sqrt(mu^2+epsilon^2)` with a dimensionful arbitrary epsilon.

5. **Energy/hyperbolicity.** Determine whether spatially varying damping can
   affect only lower-order energy decay or can indirectly damage the principal
   system through the full gauge/Z4c coupling.  State which conclusion follows
   analytically and which remains empirical.

6. **Smallest next step.** Recommend exactly one bounded next diagnostic or
   experiment.  Specify the fields, derived quantities, times/locations, and
   prospective pass/fail rule.  Prefer a diagnostic that distinguishes
   gauge-scale failure from the known strict-chi/AMR failure without another
   four-case sweep.

7. **Go/no-go decision.** State whether the code should retain only the
   domain-max prescription for production, keep the local modes as
   experimental diagnostics, revise one mathematically defective definition,
   or stop with `NO FURTHER TELEGRAPHER CHANGE YET`.

8. **Shift-stiffness causality.** Audit the inference from the fixed-eta
   control.  Is `S_beta=dt eta_shift` the correct frozen-mode explicit-RK
   stiffness measure for the implemented Gamma-driver subsystem?  Derive the
   relevant eigenvalues, including the coupled shift/Gamma principal terms,
   and say whether crossing `2.8` is a sufficient, necessary, or merely
   suggestive instability condition.

9. **Scale-invariant shift redesign.** Propose at most one smooth,
   positive, mass-rescaling-invariant replacement for global `max|K|` in the
   shift damping.  Explain its dimensions, behavior at time symmetry and
   local zeros, spatial regularity, and how it avoids both a global curvature
   spike and an arbitrary dimensionful floor.  If the evidence is insufficient
   to choose one, say so and instead specify the one diagnostic needed.

10. **Late central failure.** Rank the plausible causes of the fixed-eta
    terminal event using only the committed histories and source: lapse
    pathology, constraint growth, curvature reconstruction, AMR interaction,
    or a shared underlying collapse instability.  Be explicit that the current
    telemetry cannot identify the first invalid field.

## Required response format

Return:

1. **Source and dimensional audit** with exact paths and line references.
2. **Frozen-coefficient PDE analysis** with derived lapse and shift
   speeds/damping eigenvalues.
3. **Evidence interpretation** separating facts, inferences, and unknowns.
4. **Ranked hypotheses** for the earlier local-mode failures.
5. **One minimum discriminating next step** with strict stop conditions and no
   coefficient sweep.
6. **Recommendation** and the precise qualification boundary.

You must reason from the committed repository and artifacts only.  Explicitly
say when the available evidence cannot distinguish two mechanisms.
