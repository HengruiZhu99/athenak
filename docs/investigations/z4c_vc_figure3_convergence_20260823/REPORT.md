# Native-VC Figure-3 common-tree qualification

Date: 2026-08-23

Branch: `codex/z4c-vc-figure3-convergence-20260823`

Source under test: `6ad9cf4048af6a93aa73cf9940fc78c3b439c8fe`

## Verdict

```text
VC_AMR_INTERFACE_LIMITED
```

The repaired native vertex-centered Cartoon path is not qualified for the
Figure-3 collapse campaign. The three-resolution early discriminator fails
inside the causally protected domain immediately after a replayed AMR
derefinement, and the injection worsens strongly with resolution. N512 later
fails its unmodified metric-SPD admissibility gate in the protected interior.
The full Figure-3 endpoint was therefore not run.

This is a qualification disposition, not proof of one unique defective source
line. The evidence isolates the failure to the AMR-enabled path relative to the
previous fixed-grid core control and strongly localizes its first catastrophic
effect to an AMR transaction/interface representation.

## Frozen production configuration

- Brill data: `A=-0.047`, `rho0=5`, `z0=0`, both widths 1.
- Corrected 128x32 IrisK coefficients, ADM mass
  `2.6606354586228815`, SHA-256
  `1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10`.
- Native VC, Cartoon SO(2), O4, q6 transfer, `extrap_order=3`.
- RK4, CFL 0.15, KO 0.02.
- Max-|K|-scaled telegraph lapse, `tau=kappa=1`; Gamma driver, `eta=2`.
- No Z4 damping, no chi floor, pre-collapsed `psi^-2` initial lapse.
- Domain `rho=[0,16]`, `z=[-16,16]`, physical 4x8 root-MeshBlock lattice.
- N256 records the authority; N128/N512 replay its physical-time leaf history.
- Derefinement threshold is `0.25*dchi_max`, with `dchi_max=0.02`.

The CUDA executable SHA-256 is
`87b86be33725ddb0d55dbd3484fdb36cf570f3436e7677c0e6bbcae823773204`.

## Run outcomes

| Case | Cells/physical MB | Outcome | Last science state |
|---|---:|---|---|
| N128 | 16x16 | exact replay through event 114; reached authority cutoff | `t=3.8697650691`, `tau_c=2.3748996330`, 899 MBs, physical level 20 |
| N256 | 32x32 | authority run deliberately stopped after protected-interior runaway | last history `t=3.8697653346`, `C=1.5798e14`; exit 143 records operator cancellation |
| N512 | 64x64 | exact replay through event 24; fail-closed state rejection | failure at `t=2.4953913377`, `tau_c=1.5368600101` |

N512 rejected `nonpositive_metric_pivot_2` after RK stage 1 at
`(rho,z)=(3.421875,-0.9140625)`, relative level 3, one fine spacing from a
MeshBlock edge. Its `chi=0.1160377` remained positive; the conformal metric
determinant was `-0.163833`. No floor or weakened gate was used.

## Decisive AMR event

Authority event 3 at coordinate time `0.2979919496568637` is a derefinement:

```text
leaf count: 50 -> 44
maximum logical level: 5 -> 4
deleted leaves: 6
requested derefine: 12
```

The N256 central proper time at the event is approximately
`0.1835262604`. Bracketing history samples give the following post/pre ratios:

| Norm | N128 | N256 | N512 |
|---|---:|---:|---:|
| C | 4.7147e5 | 4.9416e8 | 2.4240e11 |
| H | 1.2326e6 | 9.5207e8 | 3.3816e11 |
| M | 2.2078e3 | 2.5943e6 | 1.3585e8 |
| Z | 3.7296e4 | 1.5172e7 | 4.8229e9 |

For N512 the post-event sample precedes event 4, so its jump isolates event 3.
For the other resolutions the history cadence is only a bracket, so those
ratios must not be represented as zero-PDE point measurements. The common
trend is nevertheless resolution-worsening by many orders of magnitude.

## Localization and controls

- At `tau_c≈0.1268`, before event 3, global C/H/M/Z effective orders are
  `5.84/3.58/9.44/7.64`.
- By `tau_c≈0.2497`, they are `-3.26/-3.57/-2.62/-0.54`.
- Common-vertex field analysis finds trusted-core order loss in chi, K, Theta,
  evolved/metric Gamma incompatibility, constraints, and Kretschmann by
  coordinate times 0.5--1.0.
- At `t=1`, the axis remains positively convergent for C/H/M/Z and
  Kretschmann, while block interiors, seams, and coarse/fine neighborhoods are
  resolution-worsening. Outer layers also mostly retain positive order.
- VC axis regularity corrections remain roundoff-sized: maxima
  `9.23e-15`, `3.86e-14`, `1.48e-13` for N128/N256/N512, with no nonfinite
  rows. Sampled shared-node spreads are exactly zero.
- The N256 maximum-speed integral leaves a trusted radius of `9.0027` even at
  the later `t=3.8698`; the N512 failure radius is only `3.5419`.
- The previous fixed-grid three-resolution control using the same repaired
  source/configuration family retained positive inner-core order through this
  window. That rules against the symmetry axis or outer boundary as the first
  cause in this AMR campaign.

The history normalization is not responsible. Cartoon history uses the
axisymmetric proper ring measure `2*pi*rho*dx1*dx2*sqrt(det(gamma))`, with VC
endpoint weights; there is no fictitious collapsed-y width.

## Regression status

- Complete enabled host suite: 131/131 passed; two CUDA-required tests were
  disabled in the host build as intended.
- Selected current-source CUDA matrix on one Perlmutter A100: 14/14 passed,
  including the device production kernel, VC Cartoon restart/output, q6 and
  axis contracts.
- No current-source SYCL runtime was available; SYCL is not qualified.

## Evidence boundary and next step

Established: exact common-tree replay, catastrophic resolution-worsening at a
specific derefinement, protected-interior field degradation, and the N512
state failure.

Supported inference: repeated AMR transaction/interface handling is the
leading mechanism. Event 3 removes a level and event 4 immediately begins a
refinement cascade, consistent with representation damage followed by a
sensor response.

Not established: whether the first bad value is written by restriction,
post-derefinement cache/ghost reconstruction, a q6 edge closure, or the next
RHS evaluation. The smallest next experiment is a bounded zero-PDE replay of
authority event 3 with writer/stage-resolved state and derivative provenance,
including pre/post restriction, redistributed active state, coarse-cache
refresh, physical/axis ghosts, and first post-event RHS. Do not tune the
boundary, gauge, KO, damping, or admissibility gates before that writer is
identified.

## Limitations

- The Figure-3 endpoint was not reached and no horizon/critical behavior is
  claimed.
- The AthenaK gauge is an analogue, not an exact gauge identity with every
  published code.
- The accepted coefficient bytes came from a dirty IrisK exporter worktree;
  this is not an upstream exporter qualification.
- Field convergence is available at common coordinate times through `t=2`;
  later raw output is intentionally not treated as convergent science.
- The history cadence identifies event 3 but does not retain a stage-resolved
  first-writer location or per-variable high-k spectrum at the transaction.
  High-k interface growth is therefore a hypothesis for the bounded replay,
  not a measured claim from this campaign.
- Exact replay is hierarchy control, not convergence evidence.
- No unique AMR source bug has yet been isolated.
