# Outer-boundary investigation — 20 September 2026

**One concrete stencil defect is repaired. Long-time boundary stability is not
established; production must remain paused.** The production G=1 volume gauge,
constraint damping, and campaign artifacts were not changed. See
[PROVENANCE.md](PROVENANCE.md) for the byte-verified production baseline.

## Findings and focused repair

1. The characteristic boundary differentiated configuration RHS through
   tangential ghost cells at internal meshblock edges. Those RHS ghosts are
   never computed or exchanged. An off-diagonal full metric raises the physical
   face normal to a vector with nonzero tangential components, activating the
   invalid read. The conformally flat zero-background test hides it.
   `BoundaryCoordinateDerivative2` now uses an active-only one-sided stencil
   at local block edges, for both residual state and its RHS. The stencil is
   centered elsewhere. No synchronization, clipping, resets or new sponge
   were added. The full normal and unique face/edge/corner ownership remain.

2. This is **not the entire growing-mode problem**. With zero-filled RHS ghosts
   its erroneous contribution is generally quadratic about the conformally flat
   background. A single-block displaced weak-field vacuum patch, without the
   black-hole interior or AMR, also fails. The original frozen continuum
   boundary admits growing scalar and vector constraint modes with the actual
   damping and gauge coefficients. [The portable half-space calculation](halfspace/README.md)
   verifies the boundary and volume residuals independently. The reported
   continuum modes and discrete run growth rates are distinct diagnostics;
   they are not a fit identifying a particular production eigenfunction.
   At the tangentially resolved frequency k=pi/(8*32), positive growth rates
   0.003358,0.005538 and0.007258/M are independently verified. The earlier
   k=0.1/M example is slightly above the dx32 tangential Nyquist frequency;
   neither example proves the complete spatial eigenfunction is resolved.

3. `zero_rate` overwrites incoming characteristic rates using the principal
   decomposition but does not supply compatible lower-order damping data.
   The isolated constraint-wave equations imply the leading outgoing residuals
   (sigma=alpha*kappa1, c=alpha*sqrt(chi), Q=Gamma_evolved-Gamma_metric):

   ```text
   F_Q     = D0 Q     + c Dn Q     + sigma Q
   F_Theta = D0 Theta + c Dn Theta + sigma Theta - sigma sqrt(chi) Q_n/2.
   ```

   An opt-in `damped_constraint_radiation` control implements this approximation
   with identical full/background evaluations and the metric-Gamma time
   derivative. **It is a rejected stability candidate**, retained for
   reproducibility and announced as experimental at startup. The complete
   gauge/radiation boundary still has growing modes. It is not an all-angle
   nonlinear constraint-preserving condition or a recommended production setting.

4. Corner kernels have disjoint writers and a combined physical normal. No
   in-place stencil write race, missed receive completion or stale background
   cache was demonstrated. The confirmed invalid RHS input must not be confused
   with a race. See [the source review](halfspace/BOUNDARY_STENCIL_REVIEW.md).

## First injection versus later amplification

In the instrumented weak-field pulse control, at cycle 0 / RK stage 1, the
volume Theta RHS is at most 1.66e-26 near the pulse. The characteristic boundary
creates 1.68e-16 at (1776,1552,-1008)M on the patch's lower-y active face.
The RK update gives Theta=1.01e-16 there; linear fourth-ghost extrapolation
raises the maximum to 5.05e-16. The composed derivative stencil reaches the
pulse. This is not nonlocal propagation or a race. Subsequent sampled maxima
move to edge/corner bands; at 1500M and 2100M they lie near
(1968,1968,-1200)M. [First-stage evidence](results/first-injection.json) and
[late samples](results/late-amplification.json) retain ranks, blocks and stages.

The production records do not contain equivalent early stage dumps. They
establish first *recorded* invalid primitive metrics after 2088.45M at
(2160,2160,-944)M: two physical boundary directions plus an internal z block
ghost, not three physical faces. The active failure occurs at 2120.025M.
The global Hamiltonian maximum was already near (2040,2040,-936)M at 1552.01M.
AMR level changes may modulate that trace, but are not necessary for the
isolated boundary instability. Interior pointwise Theta maxima, exterior
integrated growth and Hamiltonian maxima must be kept separate.

## Evolution results

All rows below retain G=1, kappa1=0.1, kappa2=0, lapse damping=0.1, RK3 and
sixth-order volume differences. Weak-field patch spacing is32M, dt=0.6M;
it is an affordable control, not the production time step or geometry.

| Control | Outcome | Interpretation |
|---|---|---|
| Centered trumpet, original linear ghosts | Invalid at788.025M | Exponential Theta growth, gamma=0.03374/M at400–600M |
| Weak-field patch, original linear ghosts | Invalid at2336.4M | No interior or AMR needed; gamma=0.01285/M at1500–2000M |
| Same cells, eight blocks, original stencil | Invalid at2336.4M | Partitioning is not the source of that mode |
| Eight blocks, repaired active stencil | Invalid at2337.0M | Ghost-read repair does not cure the independent mode |
| Damped physical constraints, linear ghosts | Invalid at3767.4M | Delays failure; still exponential |
| Damped physical constraints, quadratic ghosts | Invalid at4980.0M | Still exponential |
| Damped physical constraints, cubic ghosts | Reaches5000M | Theta=4.28e-8; late gamma=0.00271/M, no saturation |
| Original boundary +128M layer, rate0.02/M | Reaches5000M | Theta=9.33e-12; residual growth remains |
| Original boundary +256M layer, rate0.02/M | Reaches5000M | Theta=8.25e-15 in this small patch only |

The sponge uses the **existing** smooth residual relaxation before CPBC.
The wide layer covers most of the small patch. A longer discrete strip with
an undamped center still has growing modes: at tangential k*dx=pi/8,
gamma=0.001585/M and0.001245/M for rates0.02/M and0.1/M. The peak moves just
inside the layer onset. It is mitigation, not a cure; increasing damping is
not monotonically helpful. The [discrete audit](discrete/README.md) includes
negative tests of ghost orders, source retention, SAT/SBP, localized kappa
taper, and configuration-field boundary controls. Finite completion and small
Theta alone can hide gauge growth.

## Equilibrium and execution checks

- Compiled skew-normal test:1536 manufactured derivatives,98 configurations;
  old stencil returns NaN in72 poisoned-ghost cases, repaired stencil in0.
  Quadratic derivative errors are below3.5e-14; exact-zero error is0.
- Long zero vacuum reaches5000M /8334cycles. Every one of345600 saved residual
  entries, including all ghosts, is exactly zero; this is not a perturbation
  stability pass.
- Oblique-metric integration on1/2/4 MPI ranks passes all nineRK stages.
  The test actually activates tangential raised-normal components; all ten
  independently checked boundary-rate equations agree within1.39e-17. All288
  active stage/block arrays and72 complete post-RK residual block payloads
  agree bitwise across rank counts for the same block layout.
- Diagnostics off/on, one/four CPU MPI ranks: eight zero/pulse controls pass;
  residual checkpoint payloads, including ghosts, are bitwise identical for a
  fixed block layout. This does not require equivalence after changing block sizes.
- One/four-rank uniform/refined vacuum restarts preserve zero exactly. Perturbed
  checkpoint restoration is bitwise exact on active cells; subsequent split
  versus continuous differences are at most3.10e-13 in these short tests.
- The guarantee is zero **residual evolution** when identical finite inputs
  are evaluated and subtracted, not zero discrete truncation error in the
  analytic background's Hamiltonian constraint. No small residual is erased.
- The active-only stencil preserves second-order boundary consistency. Its
  accuracy across moving refinement interfaces and full stellar evolution is
  not established by the short restart tests.

Aurora job8841948 is an independent one-node/four-rank debug control charged
to the registered MHDTidal account. It uses the exact original production
binary for a larger-domain baseline/sponge comparison, target5000M,24-minute
application caps per case and one-hour allocation. It is **not** a GPU test of
the repaired stencil. Its results must be reported separately after execution.

## Reproduction and remaining work

Build problem `z4c_tov_ks` in double precision, with OpenMP or MPI as desired.
`run_controls.py --exe EXE --output NEW_DIR --cases outer-linear outer-zero`
records inputs, binary hashes and exits. `analyze_runs.py RUN_DIR... --output DIR`
fits successive windows and plots point maxima, volume-normalized constraint
RMS values and gauge amplitudes. Input seeds are in `inputs/`; compact histories
and regression results are in `results/`. Complete raw records remain in the
sibling `outer-boundary-fix-20260920` directory recorded in the manifests.

Run the two compiled point tests under `tst/unit/`, and the MPI diagnostics,
oblique-boundary and restart regressions under `tst/regression/`. The half-space
and discrete directories contain their own quick reproduction commands.

The next required development is a complete damped constraint/gauge/radiation
boundary condition **together with** a compatible discrete closure. Removing
one continuum branch or damping a small box is insufficient. Atmosphere,
star/AMR, reflection and conservation validation remain gated on a perturbed
vacuum stability pass. No such pass has been obtained, so production is not
cleared to resume. No campaign restart or monitor resumption was performed.
