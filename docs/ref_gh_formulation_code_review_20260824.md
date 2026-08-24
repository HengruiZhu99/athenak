# Ref-GH formulation and code review, 2026-08-24

## Review verdict

The fixed-core continuation campaign is paused.  The medium feedback run T4
failed closed at t=3.70126M, and the required aggressive prescribed run T5
became ill-conditioned at stage time 3.18677M.  T6 was not run and must not be
run from this evidence.  No trumpet-stability, long-time-stability, or
convergence claim is supported.

The most important formulation difference from the mature first-order GH
system is now explicit: Ref-GH implements gauge-constraint damping through
`gamma0`, but it has no `gamma2` reduction-constraint damping.  Its
`phi_ordering=standard` option changes only the curl-equivalent principal
ordering; it is not the standard gamma2-damped first-order GH system.  The
compatible update is consistent on the reduction-constraint surface, but
moving-frame terms, component-wise dissipation, and independent SMR transfer
provide off-surface channels that are not damped.  These are strong candidates
for a separate formulation investigation, not a proven single cause of the
observed failure.

This review made no evolution-equation, reference-profile, numerical-method,
or production-source change.  The only code touched after T5 is compact
analysis/reporting code.

## Scope and evidence base

The review is against branch `codex/ref-gh-feedback-continuation-20260823`,
based on exact parent `9c438dc619aa742404530c953243d71b2a01d8e6`.  The T5
executable was built from `0189d495058e09a53ca36a31f08ab924bc496582` and has
SHA-256 `7496ae64fe8a57c4f3111884bd04be9439e1d520acba60876aadca5944ded742`.

The numerical evidence reviewed here is:

- T0--T2 source, moving-reference, algebra, manufactured-controller, prescribed
  equivalence, PVC, and restart gates;
- the successful prescribed tau-8 medium calibration through t=4M;
- T3 feedback plumbing through xi approximately 0.5;
- T4 feedback job `8777607` on eight Aurora nodes and 96 PVC tiles;
- T5 prescribed-tau-4 job `8777824` on the identical eight-node, 96-tile,
  328-MeshBlock `[-24M,24M]^3` medium SMR mesh;
- native GH, reduction, and curl histories, max-location histories, controller
  histories, checkpoint continuity, mesh evidence, and exact logs.

Large output and restart files are retained on Aurora.  Git contains only
compact evidence.

## T5 outcome and feedback comparison

T5 passed rank-to-tile mapping, mesh reproduction, build, all segments through
t=3M, and checkpoint/restart continuity.  The segment targeting t=3.5M failed
because `MeasureControllerAtTime` found invalid relative conditioning at stage
time `3.18677M`, generation 4689.  Kokkos finalization errors followed the
intentional process exit and are aftermath, not the initiating failure.

The last T5 history point is:

| quantity | value |
|---|---:|
| time | 3.150078211271024M |
| xi | 0.7875195528177561 |
| activation S(xi) | 0.9320458622144755 |
| condition number | 7.700589313786415 |
| v2 max | 3.947033492622919 |
| GH L2 | 4.499013219443283e-2 |
| reduction L2 | 3.611535758786963e-3 |
| curl L2 | 2.365672088208585e-1 |
| minimum physical lapse | 4.530766696034445e-3 |
| accumulated characteristic travel | 3.0030709492631567M |
| distance remaining to outer face | 20.996929050736842M |

All recorded fields remained finite and the physical lapse remained positive
through the last history output.  The actual stage failure was nevertheless an
invalid relative metric/conditioning state, so T5 did not reach xi=1 or t=4M.
The frozen thresholds were diagnostic only in prescribed mode and did not
alter its trajectory.

At a common coordinate time of 3.150078211271024M, feedback had activated less
of the reference (`xi=0.63497` rather than `0.78752`) and therefore looked much
better: T5/T4 maxima ratios were 2.20 for conditioning, 7.78 for v2, 14.86 for
GH, 9.93 for reduction, and 13.91 for curl.  This establishes that feedback
delayed the time-domain growth.

That comparison is not sufficient to say feedback improved the formulation.
At equal activation the two histories are close through xi about 0.60, and the
controlled history becomes worse after it spends longer near the unstable
part of the path.  At xi=0.675:

| quantity | feedback T4 | prescribed T5 |
|---|---:|---:|
| time | 3.50030M | 2.70000M |
| condition number | 6.4819 | 3.5563 |
| v2 max | 1.7117 | 0.9850 |
| GH L2 | 1.4082e-2 | 4.8002e-3 |
| reduction L2 | 1.6242e-3 | 5.1338e-4 |
| curl L2 | 9.0562e-2 | 3.1148e-2 |

Thus feedback delayed failure in coordinate time but did not establish a
better state at fixed activation, did not reach a higher stable activation,
and did not satisfy T4 or T5 success criteria.

## Formulation map to standard first-order GH

For a coordinate frame, the current primary equations have the effective
first-order choices `gamma1=-1` and `gamma2=0`:

```
Psi_t = -alpha Pi + beta^i Phi_i
Pi_t  = beta^i d_i Pi - alpha gamma^ij d_i Phi_j + lower-order source
```

The compatible Phi equation is obtained by differentiating the Psi RHS.  In a
time-dependent non-coordinate spatial frame `E_I` it is implemented as

```
Phi_I,t = E_I(Psi_t) + (partial_t E_I^p) theta^J_p Phi_J.
```

This is visible in
[`ref_gh_calcrhs.cpp`](../src/ref_gh/ref_gh_calcrhs.cpp#L215-L242).  The
non-coordinate curl is

```
C_IJ = E_I(Phi_J) - E_J(Phi_I) - c^K_IJ Phi_K,
```

and the optional standard ordering subtracts `beta^J C_IJ`; see
[`phi_ordering.hpp`](../src/ref_gh/phi_ordering.hpp#L12-L45) and
[`ref_gh_calcrhs.cpp`](../src/ref_gh/ref_gh_calcrhs.cpp#L243-L329).

The standard Lindblom--Scheel--Kidder--Owen--Rinne first-order GH system adds
constraint multiples controlled by `gamma1` and `gamma2`.  With
`gamma1=-1`, a positive `gamma2` gives leading reduction-constraint evolution

```
(partial_t - beta^k partial_k) C_iab approximately -alpha gamma2 C_iab,
```

and damps short-wavelength reduction violations.  The complete equations also
contain the matching `gamma1 gamma2` terms in the Pi equation and `gamma2`
terms in Phi and the characteristic fields.  The primary reference is
[Lindblom et al., *A New Generalized Harmonic Evolution System*](https://arxiv.org/html/gr-qc/0512093v3),
especially equations 35--37 and the discussion immediately following them.

Ref-GH exposes only `gamma0` in its options
([`ref_gh.hpp`](../src/ref_gh/ref_gh.hpp#L36-L84)).  That parameter damps the
four GH gauge constraints in the second-order metric equation through
[`covariant_gh_source.hpp`](../src/ref_gh/covariant_gh_source.hpp#L126-L137),
but it does not damp `Phi-E(Psi)` or its curl.  No `gamma2` or corresponding
Pi/Phi/characteristic term exists anywhere in `src/ref_gh`.

Consequently, the already completed `phi_ordering=standard` discriminator was
only an ordering test at gamma2=0.  It cannot be cited as evidence that a
standard gamma2-damped GH formulation would also fail.

## Ranked code and formulation findings

### F1 — Blocker: no reduction/curl damping or full standard-GH comparator

The module evolves 50 fields but implements only `gamma0`.  Numerically seeded
reduction errors have no designed exponential damping.  This matters because
the standard first-order GH paper identifies the raw first-order reduction
constraint as a mode that can grow and introduces positive gamma2 specifically
to suppress it.

This is a formulation gap, not a one-line bug.  A valid gamma2 implementation
must be derived in the moving non-coordinate frame and applied consistently to
Psi/Pi/Phi, characteristic variables, and any boundary treatment.  Adding only
`-alpha*gamma2*Phi` or changing only the Phi ordering would not be a faithful
comparison.

### F2 — Blocker: `frozen` means zero command, not fixed xi

The controlling specification says that after a constraint warning the code
must set `v_cmd=0` and continue at fixed xi to test recovery.  The implementation
sets only the command to zero.  `xi_rhs` remains `xi_dot`, while `xi_dot`
relaxes on `tau_v`; see
[`feedback_continuation.hpp`](../src/ref_gh/feedback_continuation.hpp#L87-L96).
The `controller_frozen` flag is nevertheless set when `v_cmd==0` in
[`ref_gh_tasks.cpp`](../src/ref_gh/ref_gh_tasks.cpp#L643-L675).

The discrepancy was active in T4.  The risk stop occurred at
`t=2.901290467804224M`, `xi=0.5993743631850005`, and
`xi_dot=0.2166307313667841/M`.  The constraint veto did not begin until t=3.5M,
when xi had reached 0.674981458568367 and still had positive rate.  The system
therefore did not perform the specified fixed-xi recovery experiment.

This controller issue can exacerbate the closed-loop failure, but it is not a
complete explanation: the independent prescribed T5 path also became invalid.

### F3 — High: compatible preservation is conditional and dissipation is not compatible

The two-pass compatible kernel differentiates the discrete Psi RHS, which is a
strong design on a uniform grid.  However, its moving-frame term reconstructs
the coordinate gradient from Phi rather than independently differentiating
Psi:

[`ref_gh_calcrhs.cpp`](../src/ref_gh/ref_gh_calcrhs.cpp#L224-L238).

This is exact when `Phi_I=E_I(Psi)`.  Off that surface, the time-dependent frame
acts algebraically on the existing reduction error.  With no gamma2, there is
no competing designed damping channel.

After that compatible kernel, Kreiss--Oliger dissipation is added independently
to every Psi, Pi, and Phi component
([`ref_gh_calcrhs.cpp`](../src/ref_gh/ref_gh_calcrhs.cpp#L332-L345)).  In a
spatially varying frame, component-wise dissipation does not commute with
`Phi=E(Psi)`.  Therefore the full discrete RHS is not an exact compatible
gradient update even on a uniform grid.  This is a code-level fact; whether it
dominates the present failure requires an RHS-contribution diagnostic.

The observed localization makes this channel important to test.  At the last
T4 output, maximum GH, reduction, curl, source-frame-correction, and shift/lapse
diagnostics all lie on the finest level at radii 0.45--0.56M, where the moving
reference frame changes most strongly.  T5 shows the same localization.

### F4 — High for SMR qualification: transfers do not preserve Phi=E(Psi)

Every RK stage restricts, exchanges, and prolongates all 50 components
independently
([`ref_gh_tasks.cpp`](../src/ref_gh/ref_gh_tasks.cpp#L212-L248) and
[`ref_gh_tasks.cpp`](../src/ref_gh/ref_gh_tasks.cpp#L1434-L1454)).  Passing
`true` selects AthenaK's high-order Z4c cell-centered transfer, which is better
than the generic low-order transfer, but it does not impose the Ref-GH
derivative relation between separately transferred Psi and Phi.

Unlike FO-GH, Ref-GH has no post-AMR gradient repair.  The only topology repair
hook in [`mesh_refinement.cpp`](../src/mesh/mesh_refinement.cpp#L158-L179)
calls `FoGh::RepairGradients`; there is no corresponding Ref-GH method.
Static-SMR ghost filling also independently prolongates Phi every stage, so the
issue is not limited to dynamic regrids.

This is a production/SMR qualification blocker.  It is not evidence that an
SMR interface triggered T4/T5: the failure maxima are near r=0.5M, whereas the
finest Cartesian box extends to +/-4M.  An interface mechanism remains a
secondary hypothesis for these particular runs.

### F5 — Medium: current physical boundary data are not constraint preserving

The outer boundary fills the regular-frame state with `Psi=eta`, `Pi=Phi=0`
([`ref_gh_tasks.cpp`](../src/ref_gh/ref_gh_tasks.cpp#L1456-L1538)).  It is not a
characteristic or constraint-preserving GH boundary condition.  This precludes
production long-time qualification.

It did not cause T4 or T5.  Their measured characteristic travel was only
3.48M and 3.00M respectively with outer faces at 24M.

### F6 — Medium: failure instrumentation is too aggregated for a formulation decision

The native histories provide global L2 norms and max-location records for the
combined tensor magnitudes.  They do not provide:

- per-component reduction and curl modes;
- a split of the compatible core, moving-frame term, dissipation term, and SMR
  transfer contribution to constraint growth;
- fixed shells around each SMR interface for native reduction/curl;
- the exact cell/component that first makes the relative metric invalid at an
  RK stage.

The fatal conditioning path prints only stage time and controller generation
([`ref_gh_tasks.cpp`](../src/ref_gh/ref_gh_tasks.cpp#L676-L684)).  The last
history output precedes the actual invalid state.  The evidence is sufficient
to reject the campaign, but not to distinguish the leading formulation
mechanism.

## What the current evidence does and does not establish

Established observations:

- both feedback and aggressive prescribed paths fail on the same enlarged,
  causally clean medium grid;
- feedback delays growth in coordinate time but is not better at fixed xi;
- v2 is the first frozen risk channel in T4;
- curl and GH subsequently grow, while reduction stays below its warning;
- all large maxima are localized on the finest grid near the 0.30--0.60M
  fixed-core transition shell;
- the outer boundary and restart discontinuities are excluded;
- no threshold, floor, clipping, stuffing, or field reset concealed the result.

Not established:

- that missing gamma2 alone causes the failure;
- that SMR causes the failure;
- that the covariant source contractions contain an index/sign bug;
- that the reference family is mathematically incapable of continuation;
- that any resolution converges, or that a trumpet evolution is stable.

The covariant source and coordinate-oracle tests, dynamic spatial-reference
tests, stationary trumpet tests, compatible/standard ordering algebra tests,
and restart tests passed.  This review found no direct sign or index mismatch in
the on-constraint moving-frame identity, structure coefficient, or standard
curl correction.  The unresolved problem is the off-constraint propagation
and its interaction with a rapidly changing reference.

## Ranked hypotheses for a future investigation

1. **Reduction/curl propagation gap.**  Numerical errors seeded by a varying
   frame, KO dissipation, or transfer are undamped at gamma2=0 and can be
   amplified while the reference moves.  Strong code rationale; not yet isolated
   numerically.
2. **Large cancellation/stiffness in the moving reference source.**  The
   source-frame-correction maximum reaches 164 in T4 and 901 in T5 near the
   transition shell before invalid conditioning.  This may amplify truncation
   error even if every contraction is algebraically correct.
3. **Controller inertia and false freeze semantics.**  This worsens T4 and
   invalidates the requested fixed-xi recovery test, but cannot explain the
   independent T5 failure.
4. **SMR transfer injection.**  Structurally real and a qualification blocker,
   but the observed maxima are far inside the nearest refinement interface.
5. **Outer boundary contamination.**  Ruled out for T4/T5 by measured causal
   distance.

## Recommended review and correction order

No item below was executed in this paused campaign.

1. Add equation-preserving instrumentation before changing the formulation:
   compute per-component `C_IAB`, `C_IJAB`, their RHS, and separate compatible,
   moving-frame, KO, and post-transfer contributions.  Capture the exact first
   invalid cell and relative-metric eigenvalues at stage time.
2. Add manufactured reduction-error propagation tests in a nontrivial spatially
   and temporally varying reference frame.  Cover uniform mesh, static SMR,
   dynamic regrid, and dissipation on/off.  A zero-constraint test alone cannot
   discriminate off-surface stability.
3. Audit the fixed-xi veto semantics separately from the GH formulation.  Make
   the state transition and restart behavior explicit and test that a warning
   truly holds xi constant if that remains the requirement.
4. Derive a complete moving-frame gamma2-damped system on paper or in a symbolic
   oracle before editing production code.  Include matching Pi, Phi,
   characteristic, and boundary terms; compare it against the gamma2=0
   compatible system in a separate branch.
5. Design a constraint-compatible Ref-GH SMR transfer or post-transfer repair,
   and prove its order on interface-crossing waves.  Do not infer correctness
   from use of the Z4c high-order component-wise interpolator.
6. Only after those focused gates pass should the bounded continuation and
   resolution campaign be reconsidered.  Do not resume T6 from the current
   checkpoints.

## Artifact map and paused state

Compact T4 evidence:

```
docs/fo_gh_artifacts/ref_gh_feedback_continuation_20260823/aurora/
  job_8777607_t4_fail_closed/
```

Compact T5 evidence:

```
docs/fo_gh_artifacts/ref_gh_feedback_continuation_20260823/aurora/
  job_8777824_t5_open_loop_fail/
```

The direct feedback/open-loop comparison is
`docs/fo_gh_artifacts/ref_gh_feedback_continuation_20260823/feedback_vs_open_loop_comparison.json`.

Large T4 output/checkpoints:

```
/lus/flare/projects/CompactBinaryMerger/hzhu/
  refgh_feedback_continuation_20260823_2466879e_v1/runs/
  t4_feedback_outer24_8777607.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
```

Large T5 output/checkpoints:

```
/lus/flare/projects/CompactBinaryMerger/hzhu/
  refgh_feedback_continuation_20260823_2466879e_v1/runs/
  t5_prescribed_outer24_8777824.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
```

PBS reports no active jobs owned by the campaign user after job 8777824 ended.
The campaign is paused and no T6 or follow-on run was submitted.
