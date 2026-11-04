# Initial radial resolution and conformal-frame interpolation

All cases here use the same native stationary20–40M geometry, flat Omega
.40–.54/.42, gauge.62–.78 with Gamma=k², core half-width.4 and three SMR levels
with criterion radius.025. N16 Cartesian/angular resolution is unchanged. Only
shell radii differ, and all radii are checkpoint-identity-bound. Production
configuration validation now admits6–24 shells, outer radius1 and penultimate
radius>=.875. This retains the original derivative *operator family*, but does
not hold the final shell spacing or penalty CFL fixed. No old scientific
snapshot was modified. The frozen six-shell10M run is a different configuration.

The six-shell initial Hamiltonian/momentum norms .0572033/.2387504 are dominated
by r~.90–.94. Refining .54–.875 (eleven-shell negative control) does not help.
Targeting .875–1 (nine-shell control) reduces them to .00320683/.00143734.
The native areal radii at compact .90,.91,.92,.93 are approximately
29.87M,32.09M,35.94M,41.30M. Thus the latter cells resolve the compressed final
part of the initial20–40M slicing transition. This explains the initial-data
improvement, not the late10M error. The native core fields/topology are identical.

The twelve-shell control additionally refines .35–.54 and changes first-shell
spacing. Hamiltonian instead worsens to .0421891, while momentum is unchanged.
At t=0, without altering a single active field, replacing interpolated ghosts
in .33<=r<=.60 by analytic values lowers H to .00044628. Restricting replacement
to shell targets with exclusively Cartesian donors gives the same result:
316272 replaced ghosts, max field change~3.0771e-4. This isolates Cartesian-to-
shell interpolation as the dominant excess initial Hamiltonian error in this
control. It does not prove that all evolved angular growth has this cause.

The coarse Cartesian donor spacing is .0125 in the outer core. Some refined
shell spacings are near.002; second derivatives amplify small interpolant errors
on that target grid. More shell resolution alone is therefore not sufficient.
This is distinct from unresolved puncture structure, which the inner SMR resolves
at the same physical spacing in every control. Coordinate singularity inside the
horizon is not used to explain this exterior interpolation error.

## A practical analysis-only repair

With stationary conformal frame Omega, the stored fields obey

    chi_phys = Omega² chi_bar,
    A_phys = Omega A_bar,
    alpha_phys = alpha_bar / Omega,

Here A_phys denotes the tracefree BSSN variable of the physical-spacetime
conformal decomposition, not the unrescaled extrinsic-curvature tensor. The
determinant-one metric and Lambda are unchanged. Interpolate these
transformed donor values with the *same existing weights*, then transform back
with the target Omega. Lapse storage uses alpha_bar-(1+r²)/2, so remove/add the
reference at each donor/target explicitly. This control uses no analytic
solution and gives essentially the exact-ghost Hamiltonian result at t=0.
It neither changes active states nor integrates the transformed fields.

A less intrusive canonical-frame version uses
rho=Omega_new/Omega_legacy, where Omega_legacy=(1-r²)/2:

    chi_legacy = rho² chi_new,
    A_legacy = rho A_new,
    alpha_legacy = alpha_new / rho.

Interpolate these values, then invert with target rho. For r>=Omega_join_end,
rho is assigned exactly1 without evaluating0/0 at scri. Thus a production
version can become exactly the original interpolation wherever all donors and
targets are outside the changed conformal frame. The current reader only tests
this on t0 Cartesian-to-shell ghosts in the stated annulus; no production path
has yet changed. The measured canonical-frame H is .000446286794 (fixed r>.3: .000264273845),
with M unchanged. See initial-radial-controls/comparison.json.

The initial stationary control predominantly tests chi interpolation; it does
not qualify evolved/nontrivial A, boosted data, coarse/fine transfer, GPU parity
or long-time stability. Those require separate tests before a production repair.
The original derivative stencils, surfaces, scri closure, floors and evolved
active values remain unchanged in every diagnostic control. All modes check
CPU integrals against the native kernel using a separate scratch halo buffer.
Only t=0 checkpoints are accepted when an override is requested.

Reproduce from a fresh local output namespace with the checked-in configurations
and native driver. Initial command pattern:

    mpiexec -n 2 build-refined-convergence/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_refined_amr 16 boost 0 .000125 PREFIX - smr CONFIG

Build the separate reader with build_endpoint_diagnostic.py, then run
run_initial_halo_control.py and analyze_initial_radial_controls.py. Existing
reader outputs reject overwrite. Checkpoint framing/finite/SPD/FNV and exact
report identities were separately verified for all five local controls. Reports,
histories, checksums, analysis metadata and inspected plots are preserved under
initial-radial-controls/. Intermediate radial11 binary hash was not recorded;
that case remains exploratory local evidence, not a frozen production admission.
