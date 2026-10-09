# Private physical reference wave-map gauge: fixed local release recipe

The reviewed parent DERIVATION.md SHA256 8616d5d48e7d1cf84e4d7223b3331a20cdb7940ab6c9c7a649424471166dc656 is authoritative. This fresh prototype retains physical P=Kphysical-2Theta physical and the complete Lambda=GammaTilde+2*gtildeInv*Zcov convention. It changes only lapse/shift definitions, with a global harmonic core for diagnosis. C0 remains a separate unmodified production baseline. No moving-puncture blend, BH core choice, black-hole RHS subtraction, Omega floor or exact-scri extension is selected.

The source is g^bc(Gamma[g]-Gamma[ghat])^a_bc+2 Zphysical^a=0. Hhat=bar(g)^bc Gamma[ghat]^a_bc and Fbar=Hhat-2bar(g)^ai Omega_i/Omega. The independent conformal form contracts the entire reference four-metric, with s=bar(g)^bc bar(ghat)_bc and four spacetime dimensions. Actual C0 geometry equations supply complete metric time derivatives for the independent four-dimensional source identity. This checks full Z coupling; it is not a claim that C0 has the covariant Z4 subsidiary system.

For the stationary physical embedding Yhat^0=t+height(r), Yhat^I=x^I/Omega, define ah=alpha_hat (never the height), L=Omega-r Omega', b as the actual reference. All temporal connection entries vanish. The candidate evaluates G=Omega Gammahat with

    G^k_ij=-delta^k_i Omega_j-delta^k_j Omega_i-x^k Omega Omega_ij/L,
    G^0_ij=L(Omega b'-b Omega') n_i n_j/ah^3
             +b(delta_ij-n_i n_j)/(ah*r).

At the origin, isotropy gives G^0_ij=-Kbar_hat delta_ij/3. b'=-L*Kbar_hat-2b/r away from the origin. The independent oracle instead evaluates the full embedding Jacobian and Hessians with direct radial height derivatives in long-double arithmetic, and separately obtains the physical reference connection from the stationary ADM four-metric and all its first derivatives. Neither oracle uses the simplified G formulas.

Let V=alpha^2 chi gtildeInv and Llive=V-beta beta; these Llive components are not the radial reference L. The direct retained-P lapse numerator is -alpha[alpha P+beta.dOmega+Llive:G0]. The stationary identity ah Phat+betahat.dOmega+Lhat:G0=Omega*(betahat.dah)/ah is checked independently. It permits factored lapse regular and pole terms without a full Q subtraction or numerical reference-RHS counterterm. The shift numerator is 2V.dOmega-Llive:(Gi+beta^i G0). Its exact deviation expansion and regular reference identities are checked against direct rows. Inverse-metric deviations use -inverse(g)*(g-ghat)*inverse(ghat). Both beta and alpha poles must be assembled explicitly once. The candidate gauge rows have no division by live alpha; the diagnostic Fbar calculation still uses alpha^-2 at ordinary positive alpha and is not claimed uniformly bounded near zero.

Fixed test grid: S=1; a=.5,.75,1,2; geometric layers (.05,.95),(.2,.8), plus pure-CMC reference disabled-layer control; r=0,.01,.049,.05,.050001,.1,.2,.200001,.44,.45,.5,.65,.799999,.8,.85,.9,.949999,.95,.98,.999,.99999; orientations ex,(.36,-.48,.8),(-.48,.64,.6). Finite off-constraint SPD/determinant-one jets use the pinned deterministic construction. Each point checks embedding/ADM physical reference connection, conformal transformation, both Fbar forms, reference stationarity, factored/raw rows, and actual C0 full-Z source identities. Exact Cauchy-core rows are compared to harmonic lapse/integrated shift. Tiny positive alpha=1e-4,1e-100,1e-200,1e-300,1e-320 at r=0,.5,.9,.99999 checks only finite assembled gauge values, with reference/generic and zero shift.

Directional grid: a as above; r=.01,.3,.5,.75,.9,.98; all three orientations; 20 free det/trace-compatible jet directions, each with nonzero spatial value/first/second direction jets. AD derivatives of assembled gauge and Omega*Fbar are compared with symmetric FD eps=1e-4,5e-5,2.5e-5. This is a local source/jet derivative check, not an operator or principal-symbol assembly.

Fixed acceptance thresholds (each scaled error is |x-y|/max(1,|x|,|y|)):
- embedding/ADM scaled connections, full conformal transform, both source forms, factored/raw gauge and actual C0 full-Z source identities: <=2e-7;
- stationary B identity <=2e-12; raw reference gauge <=2e-7; raw reference C0 geometry <=2e-6;
- factored reference gauge exactly zero; exact core harmonic error <=2e-14;
- final directional errors <=2e-7; all three FD levels reported without requiring a rate when roundoff dominates;
- every reported metric finite/valid; all tiny gauge values finite; Release/ASan/UBSan numeric JSON exact equality.

The first compile/run is released by the parent's local-prototype task, contingent on saving the exact source/plan/recipe pins before compilation. Each fresh attempt copies the pinned source and recipe, records launch HEAD separately from production implementation, saves commands/flags/compiler and executable hashes, full compiler dependency paths/hashes, stdout/stderr, timing and all failures. No warnings are filtered. No spectra, eigenanalysis, operators, propagation or evolution may be run in this stage.

Finite-amplitude inverse-map sign: Y=X+epsilon a phi inverted at fixed target Y gives delta g=-Lie_(a phi) eta. The active convention delta g=Lie_xi ghat therefore uses xi=-a phi. A later nonlinear oracle is separate and not implemented by this local source gate.

Arithmetic-only second attempt: the first direct binary64 embedding oracle lost cancellation accuracy at small Omega (2.850070010831953e-6) while independent binary64 ADM agreed with the factored connection to6.994405055138486e-15. Its complete failed recipe/source/output is preserved under attempts/1791563251078338000. The direct embedding oracle now uses long-double intermediates with the same formulas, grid and2e-7 threshold; the candidate helper is unchanged.
