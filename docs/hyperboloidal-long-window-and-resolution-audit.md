# Longer Cartesian tangent screens and closest-shell phase

The original C0 Cartesian operators show substantial sustained growth through t6 for both tested seeds. The live damping coefficient changes that conclusion little. Higher resolution and a controlled closest-shell phase narrow the observation; they do not establish a continuum instability, a converged eigenvalue, a native nonlinear result or a stability theorem.

The N16/span2.2 screen uses the actual projected generator J20=P_ref J22 Lift, original centered/mixed/upwind stencils, strict interior nonrecursive symmetric quadratic continuation, KO.1, S1,a.5, geometry transition(.05,.95), κinput10 and physical-P lapse. Its continuous exponential differs from native final-only projected SSPRK3, P_ref R3(dtJ22)Lift. Original actual-RHS/Jv/final-step controls remain pinned. The spatial-norm shift is the previously failed private baseline; the production gauge is recorded separately.

Several early constraint peaks were already falling near t2. Fresh propagation through t6 distinguishes that transient from later growth. Two-pass restarted Arnoldi m50/m80 uses local pair tolerance1e−10, maximum step.1 and sample cadence.025. Original C0 t≤2 prefixes agree with the independent canonical control to1.76e−13. All saved states remain finite and the Euclidean1e12 guard is not hit. Actual Krylov-curve defects are recorded; they are not accumulated nonnormal forward-error bounds. No independent long canonical or long native evolution is claimed.

| Unit-Euclidean seed/operator at t6 | H | M | Z | Component amplification |
| --- | ---: | ---: | ---: | ---: |
| Production gauge pulse | 9120.2315 | 14017.4931 | 3647.6150 | 174957.508 |
| Spatial-norm gauge pulse | 5801.4543 | 7449.8893 | 2504.3881 | 97061.850 |
| Live damping gauge pulse | 5685.4425 | 7398.6319 | 2523.0921 | 97124.086 |
| Spatial-norm shell seed | 5.160924 | 5.191349 | 1.381657 | 23.852524 |
| Live damping shell seed | 4.813243 | 4.746406 | 1.255922 | 22.828965 |

H is the actual physical Hamiltonian diagnostic; M/Z are conformal covectors contracted with the stationary Penrose inverse metric. RMS uses the active-cell mean. The component norm combines configuration H1 and momentum L2 with reference weights and field units; it is not a proved physical energy. At t6, most gauge M/Z squared norm lies at r≥.9 while H is mainly interior. Oscillatory curves, all sampled extrema and radial fractions remain in the archive. t6 is about8.045 outward reference crossing times. C0 production/norm actions cost187.0/189.0s; the live action costs179.6s. The live t≤2 prefix matches its frozen previous action to5.79e−14.

A fresh original C0 spatial-norm N20 operator passes the actual full22/Jv/final-only-step gate. Comparison uses the same pointwise analytic lapse.1/shift.02 pulse, with each unit-propagated state rescaled by its exact initial Euclidean norm. This avoids changing physical seed amplitude with resolution.

| Grid | Active cells | h | Ωmin/h² | Pointwise H/M/Z at t2 |
| --- | ---: | ---: | ---: | --- |
| N16, span2.2 | 1640 | .1375 | .142562 | 1.2551548/.6986007/.2116346 |
| N20, span2.2 | 3112 | .11 | 1.894628 | 1.1443791/.6791779/.1878106 |
| N20, controlled span2.1967074064860954 | 3184 | .1098353703 | .142562 | 1.1321651/.7707442/.2306894 |

On even centered grids r²/h² is a sum of three squared half-integers. Choosing the controlled span with h=(82.75+69/484)^−1/2 matches the N16 closest-shell ratio69/484; independent cube enumeration and actual native metadata agree. The tiny span adjustment materially affects M/Z. Default N20 therefore cannot be treated as clean uniform convergence evidence. The phase control has an independent short canonical similarity check to1.879e−14; it has no t6 action.

Default N20 reaches t6 with pointwise H/M/Z2458.1905/2638.7390/772.5266, versus N16 3320.9525/4264.5736/1433.5982. Its component amplification84786.566 remains large. The initial discrete constraint-tangent defect C_h J_h has Hdot/Mdot/Zdot=.824747/1.872265/.082188 at N16, .323254/1.730908/.043013 at default N20 and .290273/1.840843/.040620 at controlled N20. Momentum source reduction is weak. These are discrete source measurements, not a continuum gauge-constraint violation or asymptotic-order proof.

The [archive](validation/hyperboloidal-live-damping-and-scri-hierarchy-experiments-20261009/README.md) retains exact matrices' hashes, source/compile provenance, epsilon checks, propagation commands, samples, errors and failures. The N20 v1 table transcription and corrected v2 are explicit; array data never changed. Production src/CMake remains byte-identical to implementation27c19d20. The next numerical test uses matching composed Hessians in RHS and Hamiltonian diagnostics, motivated by the independently derived flat discrete Bianchi defect; it has not yet passed a global gate.
