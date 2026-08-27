# Read-only external review prompt

Please perform a skeptical, read-only review of this AthenaK investigation.
You cannot run code. Do not propose changes without tying them to retained
source or numerical evidence.

Repository: https://github.com/HengruiZhu99/athenak

Branch: `codex/z4c-vc-ko05-bjorhus-figure3-20260826`

Primary source-fix commit:
`d39822c6522688749fe5ead8025907bc055f02f8`

Read first:

- `docs/investigations/z4c_vc_ko05_bjorhus_figure3_20260826/REPORT.md`
- `docs/investigations/z4c_vc_ko05_bjorhus_figure3_20260826/CONVERGENCE.md`
- `docs/investigations/z4c_vc_ko05_bjorhus_figure3_20260826/CPBC_BOUNDARY_COMPARISON.md`
- `docs/investigations/z4c_vc_ko05_bjorhus_figure3_20260826/BJORHUS_DERIVATION.md`
- `docs/investigations/z4c_vc_ko05_bjorhus_figure3_20260826/EVIDENCE_MANIFEST.json`

Core evidence:

1. N256 recorded 142 accepted hierarchy snapshots; N128/N512 replayed all
   events exactly. The common stop is t=14.405106M.
2. Regional constraints are near fourth order early. By t about 8, convergence
   degrades first in r<=8/r<=12 while r<=4 remains near fourth order.
3. The late mode is concentrated around rho=4.85-4.91. Terminal amplitudes are
   nonmonotone: N256 is catastrophically worst; N512 is still severely unstable;
   N128 is much milder.
4. N256 fails at POST_RK_UPDATE with an indefinite conformal metric, not a
   negative chi child created by prolongation.
5. The proper Cartoon history measure is already
   `2*pi*rho*sqrt(gamma)*drho*dz`; this is not a collapsed-y normalization bug.
6. The Figure-3 curves stop at central proper time 8.827, before the published
   first peak near 10.31 and minimum near 12.62.
7. A CPBC frame bug was found and fixed: the raised normal used incompletely
   initialized covariant components for non-diagonal metrics. A regression test
   now covers this.
8. The CPBC discriminator is incomplete. Rout16 original reaches t=6.5; Rout16
   CPBC develops a corner-local constraint runaway and fails at t=3.244461;
   Rout128 original reaches t=6.5; Rout128 CPBC was intentionally cancelled at
   t=1.508791 and is excluded.

Please answer:

1. What source-level mechanism best explains early fourth-order central behavior
   followed by nonmonotone rho approximately 5 failure on an exactly replayed
   hierarchy?
2. Which specific AMR transfer, shared-vertex synchronization, RK-stage cache,
   or block-edge operations remain plausible, and which are contradicted by the
   evidence?
3. Does the fixed normal-construction bug suggest similar initialization-order
   defects elsewhere in native-VC geometry/AMR code? Cite exact source locations.
4. Why might N256 be much worse than N128 and N512 on the same physical tree?
   Consider phase/error cancellation and interface-localized modes without
   treating nonmonotonicity as convergence.
5. Is the sparse Theta/Gamma CPBC mathematically well posed given that it cannot
   preserve all paired outgoing rates? Could that limitation explain the Rout16
   corner instability?
6. What is the smallest decisive next diagnostic or source-level correction?
   Prefer a bounded zero-PDE/restart-window discriminator with a falsifiable
   outcome over a long run.

Do not suggest chi floors, clipping, weakened positivity/metric gates, broad
parameter sweeps, or unsupported convergence/critical-phenomena claims. Do not
claim the physical boundary is ruled out; current evidence only strongly
deprioritizes it relative to the bulk/AMR instability.
