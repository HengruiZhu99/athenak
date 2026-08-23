# Ref-GH fixed-core tau-8 completion and causal gate

## Controlling status

This campaign is in progress.  No enlarged-domain evolution has started, so
neither the success nor fail-closed conclusion is established.  In particular,
there is no claim yet of full activation, stability through `t=10M`,
three-resolution convergence through `t=8M`, or removal of the old common-ADM
Hamiltonian reversal.

The branch is `codex/ref-gh-tau8-completion-causal-gate-20260823`, created from
the required parent `9c438dc619aa742404530c953243d71b2a01d8e6`.  Before branch
creation, the parent local HEAD, upstream-tracking ref, and remote ref were all
verified to equal that SHA.  The parent branch and dirty primary AthenaK
checkout were not modified.

The diagnostic/input/launcher implementation is commit
`9869a1481e2af487b70fff5215182db8d7a6cb3d`.  The campaign
report remains explicitly incomplete until the required Aurora results exist.

The frozen candidate remains exactly:

```text
transition_path = fixed_core
r_core = 0.30 M
tau_transition = 8 M
phi_ordering = compatible
controller_enabled = false
delta_q = delta_p = 0
```

No 50-field equation, source term, reference interpolation, FD/RK/KO/CFL
setting, smoothstep, core geometry, or controller rule has changed.

## Phase 0 regression evidence

A fresh Kokkos-Serial Debug build of the current worktree passed the compiled
source/Phi algebra, exact Minkowski, time-dependent lapse frame,
time-dependent spatial frame, and one-step stationary-trumpet gates.  The
source gate reported:

| check | result |
|---|---:|
| Phi-ordering algebra max error | `6.93889e-16` |
| flat covariant source max error | `6.93889e-17` |
| nonflat covariant source max error | `3.33067e-16` |
| dynamic-spatial source oracle max error | `4.996e-16` |
| exact source-unit Minkowski error | `0` |

The evolved exact Minkowski error remained zero; both genuinely time-dependent
frame tests ended with evolved errors `1.665335e-15`.  The stationary-trumpet
one-step gate reached `t=0.02090302M` normally.  The standalone reference-frame,
trumpet-reference, standard-source, and binary64 stationary-source audits also
passed.  The host `/usr/bin/python3` does not provide `pytest`; the failed
wrapper invocation is retained as `pytest_unavailable.log`, and the same four
underlying audit programs were run directly with exit status zero.

The diagnostic-only current binary before the final duplicate-row correction
had SHA-256
`b21bf5aab8d4849a05a037844d9f34d8c6022bb67425f23d555cfc37cf8f47c1`.
The duplicate-row correction changes no gate arithmetic and has a separate
post-build smoke below.  The final post-correction local binary used for that
smoke has SHA-256
`98edf96984f413ef3848d3032b706fa00739e3451949d5ae2c238d92be478508`.

## Existing `[-6M,6M]^3` feature audit

The compact retained three-resolution gate contains fixed-shell norms but not
the full histories, common-ADM maximum locations, or common-ADM field dumps.
Within the available evidence:

| fixed shell / metric | first retained resolution-reversed time |
|---|---:|
| `2M <= r < 4M`, Hamiltonian L2 | `2.0M` |
| `4M <= r < 8M`, Hamiltonian L2 | `0.5M` |
| `2M <= r < 4M`, momentum L2 | `2.2M` (transient in retained samples) |
| `4M <= r < 8M`, momentum L2 | `0.5M` |

At `t=2M`, the old 2--4M Hamiltonian L2 values are
`7.83555e-5, 9.69317e-5, 1.16269e-4` from M/16 through M/32.  At `t=4M`
they are `7.36146e-3, 1.34953e-2, 1.89982e-2`.  The old-domain
classification is therefore currently:

```text
D. unresolved pending enlarged-domain location/onset comparison
```

It is not yet labeled boundary-driven.  The old full outputs are retained on
Aurora under the prior campaign roots recorded in the parent report, but the
current noninteractive SSH session needs renewed Aurora authentication before
they can be inspected.

## Diagnostic-only additions

Common reconstructed ADM history now retains all previous broad and interface
regions and adds six fixed coordinate shells:

```text
0 <= r < 1M
1 <= r < 2M
2 <= r < 3M
3 <= r < 4M
4 <= r < 6M
6 <= r < 8M
```

When Ref-GH maximum-location diagnostics are enabled, a new compact
`*.adm_common_maxloc.tsv` records the global unmasked maxima and locations of
`|H|` and `|M|`.  Duplicate calls at the same time/cycle are suppressed.  The
controlled-transition user history also records `ds/dt` and `d2s/dt2` from the
same production `QuinticSmoothstep` jet helper, allowing the `t=8M` and hold
segments to prove `s=1`, `ds/dt=0`, and `d2s/dt2=0` directly.

A one-block current-source smoke emitted all nine common-ADM history chunks,
exactly one H and one momentum max-location record at `t=0`, and 29 user-history
fields with `s=feedback=ds/dt=d2s/dt2=0`.  The causal analyzer parsed the
result.  A prior attempt to instantiate the full 272-block tree on this
30-GiB local host was killed by the host memory limit before evolution or
history output.  This is classified as a local resource limit, not a numerical
failure; the actual distributed startup remains part of the PVC preflight.

## Authoritative enlarged mesh

The committed input uses the requested domain, root grid, and intended static
boxes:

```text
domain: [-12M,12M]^3
root grid: 3 x 3 x 3 MeshBlocks
level 1 request: [-4M,4M]^3
level 2 request: [-2M,2M]^3
nghost: 4
```

AthenaK mesh-only mode gives the same logical tree for all three cell tuples:

| active physical level | logical level | MeshBlocks | active coverage | block width |
|---:|---:|---:|---|---:|
| 1 | 3 | 208 | `[-12M,12M]^3` | `4M` |
| 2 | 4 | 64 | `[-4M,4M]^3` | `2M` |
| total | | 272 | | |

There are no active physical-level-0 leaves.  This is not an input-parser
mistake: vertex-centered symmetric level-2 refinement covers the eight central
level-1 parents through `[-4M,4M]^3`; 2:1 balancing then promotes every
surrounding root block to physical level 1.  The exact 272-row tree is in the
compact TSV artifact.

| resolution | cells/MB | level-1 dx | finest dx | puncture vertex audit |
|---|---:|---:|---:|---|
| coarse | 32 | `M/8` | `M/16` | exact on all active levels |
| medium | 48 | `M/12` | `M/24` | exact on all active levels |
| fine | 64 | `M/16` | `M/32` | exact on all active levels |

The full `0.30M < r < 0.60M` transition shell lies on the finest level.  No
unused static-refinement capacity parameter remains in the input.

## Causal accounting and pending gate

The analyzer uses the maximum coordinate characteristic speed stored at every
history output and computes a conservative upper Riemann sum,

```text
D_n = sum_i max(v_i, v_{i-1}) (t_i - t_{i-1}).
```

For the default `0.5M` safety buffer, a shell with outer radius `r_a` is marked
clean only when `r_a < 12M - D(t) - 0.5M`.  No enlarged-domain causal distance
exists yet, so no shell has been qualified.

The focused Aurora order is:

1. current-source PVC build, 12 distinct tile mapping, local-equivalent unit
   gates, authoritative 272-block mesh audit, and distributed diagnostic smoke;
2. medium M/24 segments `0->4M`, `4->8M`, and `8->10M`;
3. causal/outer-feature audit and fail-closed medium decision;
4. only after a satisfactory medium decision, coarse and fine `0->4M` and
   `4->8M` segments;
5. three-resolution causal convergence analysis at `t/M=1,2,4,6,8`.

The segment launcher requires checkpoint hashes on every continuation and uses
all 12 PVC tiles.  It rejects nonzero controller variables, controller
generation, nonpositive lapse/eigenvalue margins, bad state, missing
diagnostics, wrong full-activation derivatives, or a nonzero final status.

## External blocker and job state

The saved Aurora control connection expired.  A read-only BatchMode attempt on
2026-08-23 returned `Permission denied (keyboard-interactive,hostbased)` before
any scheduler command ran.  Consequently this report does not claim a current
Aurora queue state or that no jobs remain.  No job was submitted, cancelled,
or altered in this campaign from the current session.

Once `ssh aurora` is reauthenticated interactively, the next commands are to
inspect only this user's live queue, create a fresh unique Flare campaign root,
fetch this branch, submit the single debug preflight, and then remain idle with
30-minute checks while it is queued, as previously requested.

## Reproduction commands

Local build and mesh audit:

```bash
cmake -S . -B build_tau8_cpu \
  -DCMAKE_BUILD_TYPE=Debug -DPROBLEM=built_in_pgens \
  -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_SYCL=OFF
cmake --build build_tau8_cpu --target athena --parallel 4
build_tau8_cpu/src/athena -m \
  -i inputs/ref_gh/ref_gh_tau8_completion_causal.athinput
python3 scripts/ref_gh/analyze_tau8_mesh_tree.py mesh_structure.dat \
  --output-prefix authoritative_mesh_tree
```

Aurora preflight after interactive authentication and source staging:

```bash
qsub -q debug -l walltime=01:00:00 -N refgh_t8_pre \
  -v CAMPAIGN_ROOT=<fresh-root>,EXPECTED_COMMIT=<full-sha> \
  scripts/ref_gh/aurora_tau8_causal_preflight12.pbs
```

The exact production command is generated by
`scripts/ref_gh/aurora_tau8_causal_segment12.pbs` into each segment's
`command.txt`; no production segment is authorized before the PVC preflight
passes.

## Completion audit

| requirement | current evidence | status |
|---|---|---|
| medium reaches `t=10M` | no enlarged evolution | pending |
| `s(8)=1`, derivatives zero, hold time independent | runtime columns added; no `t=8` row | pending |
| all three reach `t=8M` through restarts | no enlarged evolution | pending |
| native constraints converge | no enlarged ladder | pending |
| causal common-ADM shells improve | analyzer ready; no enlarged data | pending |
| old outer feature classified A/B/C/D with comparison | old side is D; comparison absent | pending |
| no invalid state/intervention/controller activation | no enlarged evolution | pending |
