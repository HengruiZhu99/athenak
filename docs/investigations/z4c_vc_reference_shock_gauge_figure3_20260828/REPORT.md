# Reference shock-avoiding Figure-3 campaign handoff

## Final verdict

`SOURCE_PATCH_HOST_QUALIFIED; PERLMUTTER_RUNTIME_AND_SCIENCE_UNEXECUTED`

The exact literature gauge was identified and the AthenaK mapping was audited.
The existing lapse RHS already implements

```text
partial_t alpha = beta^i partial_i alpha
                - (alpha^2 + kappa) (Khat + 2 Theta),
kappa = 1, beta^i = 0.
```

The base source nevertheless rejected `alpha <= 0` in its timestep and stage
admissibility checks, even though the reference shock-avoiding slicing permits
the lapse to cross zero. The narrow source correction allows any finite lapse
only when `lapse_shock_avoiding=true`, retains positive-lapse requirements for
all other gauge modes, and uses `abs(alpha)` only in the physical light-cone
CFL speed magnitude. The gauge characteristic speed remains
`sqrt((alpha^2+kappa) gamma^nn)`.

Five focused host unit/static tests pass. A fresh, exact-source Perlmutter CUDA
build also completed, but the A100 runtime qualification could not execute
because the GPU partition was down. Qualification job `57670461` was cancelled
at wrap-up with zero elapsed time. No N256 production evolution was submitted.
The prepared qualification, N256, replay, and analysis files are configuration
and workflow evidence only.

## Published reference gauge

The Figure-3 paper states that sphGR used shock-avoiding Bona-Masso slicing and
vanishing shift. The supporting sphGR literature gives

```text
(partial_t - beta^i partial_i) alpha = -alpha^2 f(alpha) K,
f(alpha) = 1 + 1/alpha^2,
alpha(t=0) = 1,
beta^i = 0.
```

With AthenaK's standard extrinsic-curvature convention and Z4c variable
`Khat = K - 2 Theta`, the source is `-(alpha^2+1)(Khat+2 Theta)`.
`REFERENCE_GAUGE.md` records the authority chain and the isolated sign
ambiguity in one 2026 source.

## Source audit and correction

At base commit `c67edf54ce97536ad9410adcf6029a4478038139`:

- the shock-avoiding lapse RHS and `Khat+2*Theta` mapping were already correct;
- `shift_mode=prescribed_zero` already zeroed the evolved shift;
- the gauge CFL speed already used `sqrt((alpha^2+kappa) gamma^nn)`;
- two timestep scans and the stage-state gate unconditionally required
  positive lapse.

The current patch changes only the last item and the physical light-speed CFL
magnitude. It does not floor, clip, regularize, or modify the lapse evolution;
it does not alter telegraph lapse, Gamma-driver shift, damping, KO, AMR
transfer, or boundary conditions.

## Verification

The focused host test selection passed 5/5 on 2026-08-28:

```text
athena.z4c_state_admissibility
athena.z4c_state_admissibility_static
athena.z4c_timestep_contract
athena.z4c_timestep_contract_static
athena.z4c_shock_avoiding_gauge_static
```

These checks establish the narrow source contract only. They do not establish
GPU correctness, evolution stability, Figure-3 agreement, or convergence.

## Perlmutter disposition

A clean checkout of commit `00e66dfa5e0e7f7f2c711166998806a05decbd55`
was configured and built with CUDA, MPI, IrisK, unit tests, and Ampere80 on
Perlmutter. The resulting executable SHA-256 is
`c5153a5c25c4b2aba22737061baa00628badcd6eade2c66461ac182708677e55`.

The scheduler rejected shared-interactive and normal-interactive access while
the GPU partition was down. Debug qualification job `57670461` never allocated
and was cancelled at wrap-up; `sacct` reported `CANCELLED by 102811` and
`00:00:00`. No GPU runtime evidence or science data exists. Build success must
not be presented as CUDA qualification.

## N256 and convergence disposition

The fresh N256 run was not submitted or executed. Consequently:

- no fresh N256 AMR authority exists;
- there is no central Kretschmann-versus-proper-time comparison;
- the N256 reproduction gate was neither passed nor failed;
- N128/N512 replay was correctly not attempted;
- no convergence order can be reported.

No active Perlmutter campaign job remained at wrap-up, and no production job
was submitted.

## Exact artifacts

- `REFERENCE_GAUGE.md`: literature equation and parameter authority.
- `GAUGE_IMPLEMENTATION_AUDIT.md`: source mapping and defect boundary.
- `brill_vc_reference_shock_gauge.athinput`: unexecuted N256 configuration.
- `N256_REPRODUCTION.md`: explicit unexecuted disposition.
- `CONVERGENCE.md`: explicit gate-not-reached disposition.
- `HOST_TESTS.txt`: focused host verification record.
- `PERLMUTTER_CUDA_TESTS.md`: exact build and unexecuted runtime disposition.
- `PERLMUTTER_SCHEDULER_DISPOSITION.txt`: terminal scheduler evidence.
- `perlmutter_cuda_qualification.sbatch`: unexecuted bounded A100 gate.
- `perlmutter_run_reference_n256.sh` and companion sbatch: unexecuted fresh
  N256 workflow.
- `perlmutter_run_reference_replay.sh` and companion sbatch: unexecuted,
  gate-protected N128/N512 replay workflow.
- `analyze_reference_n256.py` and `figure3_published_curves.csv`: offline
  direct-comparison analysis and authenticated curve data; no fresh N256 data
  were supplied to them.
- `EVIDENCE_MANIFEST.json`: strict file hashes and claim limits.

## Remaining limitations

The source correction still requires Perlmutter A100 runtime qualification.
The reference-gauge N256 run, comparison against the authenticated curves,
terminal failure localization if needed, fresh AMR recording, conditional
N128/N512 replay, and constraint convergence are all outstanding. This handoff
must not be cited as GPU qualification, Figure-3 reproduction, numerical
failure, or convergence evidence.
