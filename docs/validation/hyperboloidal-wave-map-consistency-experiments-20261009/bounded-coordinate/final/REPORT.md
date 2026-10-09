The bounded inertial-coordinate witness family passes the declared finite-Omega geometry, actual C0 gauge, constraints, algebraic-normal, and directional derivative controls in Release and ASan/UBSan Debug. This is a point-action validation of the existing physical-P/spatial-norm equations using complete analytic reference jets. It supplies no discrete radial/Cartesian operator, Cdot, eigenvalue, evolution, or scri-class preservation result.

The new family is a deliberate change of witness, with four independent polynomial radial fields. For rho=x.x, prescribed physical inertial displacements Xi^T=tau(rho), Xi^I=x^I zeta(rho) pull back to xi^t=tau-(b/alpha)r zeta and xi^i=(Omega^2/L)x^i zeta; the same stationary adapter applies to velocities v,w. The exact core is the identity adapter and has regular polynomial limits at the origin. In the exact outer branch its compactified field tangents and required spatial jets have only powers of 1+rho in their denominators. ADAPTER-BOUNDEDNESS.md gives the explicit canceled formulas. The earlier unbounded compact-coordinate polynomial family, including both failed geometry attempts, remains frozen separately at index 9f4ec9f106ec95f27d912a33b29e7dae7056025bf5b9d1f8d92bdf57809095df. No old failed gate is relabeled as an accepted subset.

The fixed scientific cases are 629 witnesses: sixteen individual fields tau,zeta,v,w times rho^0..3 and one mixed field, evaluated at the origin and at twelve positive radii through .995 along three Cartesian directions. The 420 raw22/free20 binding cases retain both reference and nontrivial finite SPD backgrounds. The original geometry threshold is 5e-10; the final directional FD threshold is 2e-7 using the fixed five-epsilon sequence, with every sequence retained and classified as truncation decrease or already within the absolute floor. Nothing was accepted by selecting a best epsilon.

Final006 results, identical in Release and Debug:

| Check | Largest reported error |
| --- | ---: |
| Entrywise geometry, absolute and scaled | 3.001332515850663e-11 |
| Gauge row, entrywise scaled / absolute | 6.359296331984441e-16 / 7.275957614183426e-12 |
| Physical H, M_cov, Z_cov, Theta input, absolute | 2.2737367544323206e-13 |
| Input/output algebraic normals, absolute | 2.842170943040401e-14 / 2.2879476091475226e-11 |
| Spatial differentiated normals, absolute | 1.2789769243681803e-13 |
| Analytic reference RHS / physical constraints, absolute | 3.1086244689504383e-15 / 8.881784197001252e-15 |
| Exact core full jets / acceleration, scaled | 3.51540321247563e-16 / 1.1271656825341395e-16 |
| Physical versus factored lift at moderate radii <=.95, scaled | 2.260401088581119e-12 |
| Coordinate directional FD, final epsilon | 7.970187216432285e-8 |
| Raw22/free20 directional FD, final epsilon | 8.78982898853158e-9 |

The independent kinematic geometry and gauge acceleration comparison uses complete coefficient derivatives and the fixed external Omega convention. Physical constraint order is H,M_cov xyz,Z_cov xyz,Theta_phys; there is no Omega rescaling. The stored A lift includes both the metric-induced trace tangent and delta-chi*Aref term. The actual generic dual spatial-norm gauge is bound directly, with a double-only wrapper negative control whose derivative discrepancy is 31.578941457304698. This test cannot be bypassed by allowing that wrapper to fall back to zero derivatives. Actual C0 damping and existing analytic reference subtraction are retained.

Initial003 passed these scientific controls but failed its additional direct inverse-embedding identity, with a third-order temporal Taylor-jet residual 1.1209007944046334e-8. Its spatial residual was at most 1.6082052356480858e-11. Both Release and sanitizer failed statuses remain unchanged. Source-only004 was never compiled because it reassociated the original arithmetic graph in its diagnostic decomposition. Source-only005 was never compiled because its initial corruption comparator did not test the independently expected binding relation. Both preparations and corrections are retained.

Root-reviewed final006 changes diagnostics only. It binds the actual returned spatial Taylor jets against Omega^2 times the prescribed inertial spatial displacement divided by L, then checks the algebraically canceled spatial and temporal forward identities. Their maximum Taylor-coefficient residuals are 5.91062173119204e-16, 1.1601830607332886e-14, and 1.865174681370263e-14 respectively. All 629 deliberately corrupted returned third jets fail the same binding comparator; the smallest corruption residual is 7.583459476670696e-8. The original direct multiplication graph is also evaluated and still fails at 1.1209007944046334e-8. Only the stable equivalent identities are accepted.

The reproducible saved-data readback proves that all common scientific fields in all 629 initial003 and final006 output rows are exactly equal. It also verifies all four final006 Release/Debug scientific stdout files are byte-identical. Thus neither the adapter nor lift nor actual source action was tuned to cure the diagnostic arithmetic failure. The final Release compile/gate took 2.22147/.63754 seconds; Debug ASan/UBSan took .94504/10.64639 seconds. Compiler, flags, complete dependency/archive hashes, commands, immutable executable identities, recipes, input and output hashes are retained in the build and run receipts. Final006 launch HEAD is a0f8fc8464665db3104fdbdaf142661259a6a399; compiled production inputs remain 27c19d20696ea6dd4704032c51dfd026218f64f2.

Higher-reference context is additive. The earlier independent 100/130-digit radial oracle passes 2052 field/derivative comparisons, maximum scaled error 4.21005e-12, with 139 endpoint-tail checks and 183 representable nonzero tails. Its independent Cartesian composition is a separate mathematical consistency control. Root's later saved-only comparison binds the actual owner C++ Cartesian exports: 53 fields times 20 ordinary Cartesian jets times eight saved points gives 8480 comparisons at each precision, maximum native scaled error 3.568423714179294e-13 and precision difference 6.202786e-98. This resolves the old historical pending C++ readback additively; historical run receipts are not rewritten.

Input-tangent smoothness does not imply bounded gauge sources or preservation of a scri-compatible class. For the outer pure inertial velocity v=1 with tau=zeta=w=0, delta-alpha=alpha, delta-beta=beta and the geometric tangent is zero. At S=1, curvature radius .5, xi=2, spatial-norm eta=6, let rho=x.x and A=1+rho. The actual C0 control obeys

    Omega F_alpha = Omega*(-8rho-1.5A)-4A*(1+3rho) -> -32,
    Omega F_beta_i = (12+10Omega)*x_i -> 12*x_i.

The sources therefore diverge like 1/Omega despite the bounded input. The saved-data readback checks these analytic formulas against nine already-saved rows at .95/.98/.995, maximum absolute error 3.588240815588506e-13; it makes no new API-query claim. A forthcoming exact localized wave-map family is a separate compatibility control. This checkpoint does not admit that candidate or any operator, spectrum, Cdot, propagation, boundary treatment, or production edit.
