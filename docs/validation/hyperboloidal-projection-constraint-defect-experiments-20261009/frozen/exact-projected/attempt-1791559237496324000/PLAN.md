# Analytically seeded projected-constraint point gate (HELD)

This new source candidate awaits root review and a hash-bound release. It uses
the unchanged Release API executable, J0/N8/rb=.98 operator, physical seed/lift,
Minkowski reference, C0 equations and spatial-norm gauge. It compiles nothing
and computes no spectrum or propagation. Both prior ordinary-FD attempts remain
failed, with all five levels retained. No threshold or failed classification
is changed.

For each of the same 21 centers and all eight J0 real m=0 channels, query the
existing manufactured-rate API with envelopes0,1,2. Their WJets are respectively
`(1,0,0)`, `(rho,1,0)`, `(rho^2,2rho,2)`. The actual API returns the RHS22 and
physical initial constraints8 after complete analytic `SeedPhysical` and
`LiftPhysical`, including the actual reference coefficient jets. Pointwise
duality makes both outputs linear in `(W,W_rho,W_rhorho)`. Recover the three
maps with

    map_v  = output0
    map_d  = output1 - rho output0
    map_dd = (output2 - rho^2 output0 - 2rho map_d)/2.

This is an algebraic identity for the analytically seeded point action, evaluated
in binary64; it is not an exact-arithmetic claim about floating kernel output.
No radial or angular interpolation of the maps occurs.

Direct held envelopes3,4,5,6 must match the reconstructed maps at every point
and channel. Compare the full30 outputs and the constraints8 separately, with
both global and per-row scaled errors <=5e-11. This prevents a large RHS from
masking a constraint-map defect. Independently compare the recovered RHS22 maps
to the saved source-batch WJet maps at every center at the same global/per-row
threshold. These are 1,176 rows in the existing manufactured API, including
672 held rows. Floating warnings and exceptions remain errors; all stdout,
stderr, inputs, sources and failures are retained.

The eight constraint rows are H, Cartesian Mx/My/Mz, Cartesian Zx/Zy/Zz,
physical Theta. They have no Omega rescaling or moving-frame convention.
The underlying `EvolvedConstraints` uses metric/chi/A/Lambda and P+2Theta;
its physical eight constraints are independent of live lapse/shift. Reference
metric/chi jets through second order and A through first order suffice. No
third/fourth reference jets are invented.

After the source controls pass, reconstruct the same 14 polynomial-interpolated
modal seeds X used in both FD attempts. Evaluate the exact Jacobi envelope
jets for X, Jbulk X, Jsat X and their sum, and contract each with the recovered
constraints8 map. The result is an analytically seeded point value of
C_ref[Phi(X)] or C_ref[Phi(Y)], including every reference/angular coefficient
jet. It is not a value-only RJB restriction or a frozen-coefficient derivative.
No extra r^L is included; the angular basis already contains that factor.
Saved operator matrices must be finite real64x64; the original Riesz/T and
modal interpolation checks retain their old thresholds. Check total=bulk+SAT
at5e-11 and pure-gauge initial constraints at5e-11. Keep all projected/SAT
constraint magnitudes as measurements, not zero gates or physical energies.

Compare every available ordinary-FD projected/initial vector and Richardson
vector with these analytically seeded vectors. Record those errors without
changing or rerunning either failed FD gate. Neither prior failed run is called
passing because a new analytic point gate passes.

The general nongauge continuum comparator C_ref[L_actual Phi(X)] is NOT
evaluated here and remains unresolved for the complete 14-witness projection
defect gate. The final receipt must say so explicitly even if the point-map
gate passes. For the four gauge-only seeds only, the continuum constraint-rate
zero can be used as a derived positive-Omega Einstein-sector identity: the
physical constraints are independent of lapse/shift, ADM lapse/shift variations
at an Einstein reference are tangent to H=M=0, and C0 additions vanish when
Theta=Z=0. This relies on the previously audited actual/ADM source equivalence
and the frozen gauge-rate controls, not a numerical pass of the failed FD
sequences. The frozen constraint-rate index is
`3d4c613a814a8a3325a7f980c2e20dcabf3ea08ddbcb10d42026ddca732d4e2f`;
the readable context is `docs/hyperboloidal-continuum-constraint-rate-audit.md`.
Rows using that identity are explicitly labeled derived gauge projection
defects. No analogous replacement is made for nongauge seeds.

Rehash all pinned sources, original maps/schema/receipts and relevant FD logs
before and after. The driver defaults to HELD and requires an authorization
with `analytic_projected_constraint_points_admitted=true` and exact source,
executable and operator hashes. Use CommandLineTools Python with the saved
`PYTHONPATH=build-layer-research/boundary/python-deps` and
`OPENBLAS_NUM_THREADS=1`. Preserve all failure receipts additively.

This gate cannot establish CPBC, integrated constraint energy, constraint-wave
stability, nonlinear closure, a scri boundary prescription, or later BH
formation. The single-BH wormhole-to-trumpet requirement retains the Minkowski
hyperboloidal reference and remains unresolved.
