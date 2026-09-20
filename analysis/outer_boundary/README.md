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
gamma=0.001585/M and0.001245/M for rates0.02/M and0.1/M. The weighted full-state
peak moves just inside the layer onset, whereas Theta itself peaks within the
layer. It is mitigation, not a cure; increasing damping is
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
- Oblique-metric integration on1/2/4 MPI ranks passes all nineRK stages
  (Minkowski principal-part fixture, with kappa/eta/lapse damping zero).
  The test actually activates tangential raised-normal components; all ten
  independently checked boundary-rate equations agree within1.39e-17. All288
  active stage/block arrays and72 complete post-RK residual block payloads
  agree bitwise across rank counts for the same block layout.
- The same fixture with eight32³ blocks also passes on1/4 MPI ranks: all
  12.8 million saved residual entries, including ghosts, are exactly zero in
  the zero control and bitwise equal across ranks in both zero/pulse controls.
  All checkpoint payloads are finite and full metrics positive. The pulse
  activates an off-diagonal raised normal; it is not another zero-only test.
  This three-step test reaches0.01875M and is not a production-damping or
  stability pass. [Results and reproduction](results/block32-smoke/README.md).
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

Aurora job8841948 completed an independent one-node/four-rank debug control
charged to MHDTidal, using the **original production binary**, not the repaired
stencil. Its larger weak-field box has four32³ blocks and no BH interior or AMR.
The original boundary failed at2375.4M; the first printed invalid metric lies
in the fourth+x ghost at(2160,2000,-1040)M between neighboring log records at
2361M and2362.2M. Its only saved checkpoint is initial data.

The matched256M/rate0.02 sponge reached the5000M target in497 application-wall
seconds, with all four final checkpoints finite and ghost-inclusive metrics
positive. It did **not** stop at the walltime limit. Theta RMS still grows
exponentially: gamma=0.001386/M over4000–5000M, an e-fold time of722M.
The final active Theta and lapse peaks lie at(1616,1904,-1008)M,144M from+y,
**inside** the sponge;99.13% of the proper-volume Theta² integral is in that
layer. The strip model also places its Theta maximum144M from the face, with
99.12% of its Theta² integral in the layer. This close agreement concerns the
Theta profile, not the weighted full-state peak near the layer onset. Different
geometry, variable coefficients and volume measures prevent equating the full
modes, and a final profile does not establish initial injection. See
[GPU comparison and spatial profile](results/gpu-8841948/README.md).

Separate repaired-stencil GPU verification job8841975 stopped before evolution:
the zero-control input disabled pulse amplitude but left pulse family enabled,
which the problem generator rejects. The original record is preserved. The
two zero inputs were corrected and passed local execution preflight. Replacement
job8842005 **passed**, with PBS exit0 in63seconds. Exact-zero controls remain
zero on1/2/4 GPU MPI ranks and the longer four-rank zero run reaches20code-time
units. Oblique boundary responses and signed stage arrays agree bitwise across
GPU rank counts. All360 CPU/GPU array comparisons pass the stated tolerance;
maximum absolute error is2.3863e-13, and no full array is CPU/GPU bitwise equal.
All-ten-mode boundary-rate errors are below1.74e-17. This verifies the tested
stencil portability, not long-time perturbation stability. See
[GPU verification](results/gpu-8842005/README.md). The repaired executable SHA256
is`a6c3af79571819fba5dc2252ceb9abacb31279feec440f2c542e342dfee43639`.

## Longer source and boundary study

The [long-sponge study](long_sponge/README.md) extends the investigation beyond
5000M. A wider, smoother layer still fails with the original positive kappa;
in the matched large flat cube it delays failure from5650.2M to6420M. Reducing
kappa to zero and reducing shift damping to0.02 is the leading tested candidate:
one displaced weak-curvature control reaches50000M with valid full/ghost
metrics and Theta RMS7.56e-17. A separate GPU compact-lapse control with kappa0
and the original eta2 reaches20000M with decaying metric perturbations.
These are specific vacuum controls, not strong-field or stellar validation.

The incoming-state audit also identifies retained nonzero initial Gaussian
boundary data under `zero_rate`; this is separate from the growing positive
source-coupled boundary mode. With access renewed, three direct constraint-pulse
GPU controls are now verified through50000M, with finite eight-rank payloads
and valid full/ghost metrics. Their final exterior Theta RMS values are
1.26e-13,1.27e-13 and5.26e-15. Small late gauge/Hamiltonian drifts remain;
these are not an all-field asymptotic stability pass. Strong-field SMR
job8843091 is submitted in debug on two nodes/24ranks, with an exact-zero gate
before a fresh lapse pulse. It uses the existing boundary/reduced damping,
not the coupled exterior prototype. The study records actual stopping reasons
and prototype limitations separately.

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
vacuum stability pass. The reduced-source vacuum passes do not establish stability for the
strong-field production configuration, so production is not cleared to resume. No campaign restart or monitor resumption was performed.
