# Production interface and separated-speed controls

These are explicit opt-in RF2 policies. Existing configuration identities and
frozen Aurora snapshots are unchanged. The new large-center stationary control
uses three SMR levels, twelve radial shells, a constant central Omega out to the
inscribed sphere r=.4, and gauge blending .62–.78. It has not yet demonstrated
long-time accuracy or convergence. The geometry, conformal-frame transformation,
and native initial slice remain consistent with the same configuration.

## Cartesian-to-shell interpolation

`halo_frame=legacy` transforms chi, all six BSSN tracefree-curvature components,
and lapse to the original conformal frame before the existing interpolation,
then back at the target. With rho=Omega_new/Omega_legacy,

    chi_legacy = rho^2 chi_new,
    A_legacy = rho A_new,
    alpha_legacy = alpha_new/rho.

Stored lapse deviations are converted to full lapse before interpolation.
The ratio is exactly one outside the changed-Omega region, including scri.
Only rows with a spherical target and exclusively Cartesian active donors are
changed. There is no analytic solution injection, radial diagnostic mask,
new donor request, or modification of active values. Metric, Khat, Theta,
Lambda, shift and auxiliary fields keep their original interpolation.

The CPU plan and the existing device halo kernel use the same factors and
weights. Device factors are uploaded only on partitions needing them; no
additional communication or kernel launch is introduced. This is an avoidance
of unnecessary overhead, not a measured GPU speedup.

The manufactured test has nonzero curvature and varying lapse, with polynomial
fields in the legacy frame. Its exact-polynomial error is9.77e-15 versus an
uncorrected .00323166; unaffected components/rows are bit-identical and the
Serial device path matches the CPU path exactly. Two-step stationary N12
MPI1/2 native arrays and entire checkpoints are byte-identical, with finite
payload, positive chi/metric and native checksum verified. See frame-halo-control/.
These are local Serial tests, not SYCL or long-time evidence.

The N16 radial12 production initial Hamiltonian norm is .00044628679, matching
the earlier read-only interpolation control, versus .04218909677 with stored-
frame interpolation. This approximately94.5-fold reduction establishes a
specific initial interface discretization error. It does not establish that
this error is the sole cause of subsequent growth.

## Independent spatial gauge check

`check_spatial_gauge_block.py` derives the constant-frame inner scalar gauge
block and compares it against the independently extracted full RF2 RHS.
For U=(d_x alpha_phys,d_x B_x,Khat,d_x beta_x,Lambda_x), the restricted block is

    P = [0, chi_guard, -2alpha_phys, 0, 0;
         kappa/tau, 0, 0, 0, 0;
         -chi_phys, 0, 0, 0, 0;
         0, 0, 0, 0, Gamma;
         0, 0, -4alpha_phys/3, 4/3, 0].

The two characteristic squared speeds are
2alpha_phys chi_phys + chi_guard kappa/tau and4Gamma/3. The nonzero Khat-to-
Lambda coupling produces the defective coincident block. Both previously
identified inner roots reproduce this matrix within4.45e-16. The full-symbol
rank audit is separate; this five-variable restriction alone is not a full
characteristic decomposition.

The actual telegrapher reference input g1_bhahaha_L5.athinput at
b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a has tau=.1,kappa=.2,Gamma=1,eta=2.
Its ratio is2, matching the mass-scaled RF2 inner equations. Thus the same
inner coincidence can also arise in the corresponding spatial formulation.
It is not uniquely a hyperboloidal stitching defect and does not prove the
cause of either finite-resolution evolution. Preserve that distinction from
the known successful spatial-reference runs.

## Optional separated inner shift coefficient

`shift_speed_policy=separated` changes only the inner Lambda source coefficient:

    g = g0 + (3/4)[2 Omega alpha_bar chi_bar + c kappa/tau],
    c = max(Omega^2 chi_bar, chi_guard),  g0 = k^2.

The inner shift RHS remains beta.grad(beta) + g Lambda - eta beta. The source
coefficient is not a conservative flux, so no gradient of g is added. Damping,
auxiliary-D, lapse, restoration and the outer/scri equations are unchanged.
The policy is admitted only for the configured direct-outer formulation with
ratio2. It differs from constant-Gamma spatial Z4c and does not claim exact
inner equivalence. The chi guard gives a small positive correction even at
vanishing physical chi.

Let s be the inner weight, F=alpha_bar^2+4Omega^2 alpha_bar and
Q=gtilde^{ij} n_i n_j>0. Then

    c_lapse^2/Q = chi_bar[(1-s)F+2s Omega alpha_bar]+s^2 c kappa/tau,
    c_shift^2/Q = (4/3)[s g+(1-s)(3/2)chi_bar F],

and hence

    (c_shift^2-c_lapse^2)/Q
      = (4/3)s g0 + (1-s)chi_bar F + s(1-s)c kappa/tau > 0.

This separates the lapse and longitudinal-shift speeds for positive lapse,
chi and metric. It is not a proof of full strong hyperbolicity: other repeated
modes, arbitrary backgrounds and nonlinear regularity remain to be examined.
No singular division by Omega or s is introduced; the existing s=0 branch
returns the original outer RHS exactly.

The6363-case manufactured test includes nonzero shift gradients, Lambda, B,
and transport, log-spaced lapse/chi and all blend weights. The actual RHS and
speed identity agree within2.36e-16 when errors are normalized by the operands
being subtracted; unrelated source terms and outer values are bit-identical.
The first test used an inappropriate normalization by a tiny difference of
large terms; its2.72e-12 residual was roundoff, not an equation discrepancy.
Both logs are retained. The Serial CPU/device paths agree exactly.

The full26-variable algebraic-trace-tangent symbol was sampled at361 radii for
both central configurations. The minimum positive lapse/shift gaps are .06716
and .01365; sampled imaginary parts are below4.46e-14 and the expected gauge
speeds match eigenvalues within7.05e-14. Outer symbols are bit-identical.
See separated-shift-control/. These are initial radial-symbol checks, not a
nonlinear well-posedness or evolved-convergence result.

## Next evolution comparison

Freeze one new source/build for the frame-corrected radial12 stationary runs,
with constant and separated inner drivers distinguished by their complete
configuration identity. Start the separated control fresh on four Aurora
nodes at N16,dt=.000125,target10M; no checkpoint from another formulation.
Use the constant-driver companion to separate any gauge effect from the shared
interface/mesh changes. Compare fixed-domain histories and unsmoothed spatial
profiles before interpreting survival. Boosted and resolution controls follow
stationary accuracy assessment; keep existing completed runs as preserved
context rather than matched one-parameter baselines.


## Late-time read-only control

Applying the production frame interpolation only in postprocessing to the
preserved stationary6 10M checkpoint leaves the exterior r>.3 H/M norms
essentially unchanged (relative changes7.1e-9 and1.4e-10). This control uses
actual stored donors, changes no active fields and retains the original
identity/admission qualification. The modified Omega frame ends at.30 in
that older configuration, so most exterior halos are unchanged by construction.
It rules out this specific postprocessing correction as the explanation of
the large late exterior norms; it does not rule out all diagnostic error,
undo errors accumulated in evolution, or establish a slicing pathology.
The paired fresh production evolutions remain necessary. Reproducer and
inspected radial overlay: analyze_late_frame_control.py and late-frame-control/.
