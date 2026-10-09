# Regular total-J flat-core envelope action

The local core gate passes. For J=0,1,2, it derives and validates the 8-, 16- and 20-amplitude C0 physical-P/spatial-norm core action as

```
W_t = B0(rho) W + B1(rho) W_rho + B2(rho) W_rhorho,   rho=x.x.
```

Every entry is a polynomial in rho of degree at most two. There are 188 nonzero coefficient terms: 28 for J=0, 69 for J=1 and 91 for J=2. The exact algebraic coefficients are in `symbolic-report.json`; numerical coefficient arrays are in `core-envelope-blocks.npz`; the generated division-free C++ evaluator is `core_envelope.hpp`. All independent envelopes are free. No r/rho denominator, cross-L compatibility workaround or origin excision is used.

This is the continuum linearized action at the exact flat Cauchy core of the pinned reference, r<=0.05, with Omega=alpha=chi=1, g=I and beta=A=P=Theta=Lambda=0. It supplies a cancellation-safe origin formula for this core. It does not construct a radial PDE discretization or global matrix, adopt a boundary/SAT, compute eigenvalues, evolve data, or prove stability. In particular these core coefficients do not apply to the variable-coefficient transition or outer layer.

## Physical lift and independent core equations

The channel ordering is the frozen basis ordering: alpha, metric_trace, P, Theta_phys; then beta and Lambda with each allowed orbital L; then metric_STF and independent_A with each allowed orbital L. The normalized physical metric trace is tau=tr(delta bar-gamma)/sqrt(3). Thus

```
delta bar-gamma = h_STF + tau I/sqrt(3),
delta chi = -tau/sqrt(3),   delta g = h_STF,   delta A = independent_A.
```

The complete metric-to-chi/g conversion and the differentiated A trace condition are retained through the pinned lift. At the exact core A_ref=0, so the latter reduces to tracefree A. The outputs retain all 22 raw native components in order chi,g6,P,A6,Lambda3,Theta,alpha,beta3. The input space is the algebraic tangent space; this is not a claim of 22 independent algebraic-normal inputs. Native input/output metric determinant and A trace normals are checked before any output projection.

The independent frozen Cartesian core oracle, rewritten in these physical fields, gives

```
alpha_t = -3 P,                         beta_t = (3/8) Lambda,
tau_t = -2(P+2Theta-div beta)/sqrt(3),
h_STF,t = -2 A + sym(grad beta) - (2/3)I div beta,
P_t = -Delta alpha + 10 Theta,
Theta_t = -Delta tau/sqrt(3) + (1/2)div Lambda - 20 Theta,
A_t = STF[-Hess alpha - Hess tau/(2sqrt(3))
          + (1/2)sym(grad Lambda)] - (1/2)Delta h_STF,
Lambda_t = Delta beta + (1/3)grad div beta - (4/3)grad P
           - (2/3)grad Theta - 10(Lambda-div h_STF).
```

Here sym(grad v)_ij=partial_i v_j+partial_j v_i. The physical-P gauge uses f=3 and the shift coefficient is mu=3/8. The actual kappa_input=10/alpha argument is differentiated at the stationary reference, including the +10Theta trace term. The connection damping is -10(Lambda-Gamma); because Z=(Lambda-Gamma)/2 in this core, replacing this with -20(Lambda-Gamma) would be incorrect. No C1, Q lapse, profile, extra source, upwind/KO or finite-h ghost closure is added.

## Exact polynomial derivation and origin evaluation

The derivation imports the byte-pinned exact solid-CG Cartesian basis. It inserts independent W, W_rho and W_rhorho into every input channel, applies the independent core equations, separates the resulting homogeneous Cartesian polynomials and projects exactly on the unit sphere. A homogeneous degree d contribution can appear as rho^p times an output solid polynomial of degree L only when d-L=2p>=0. This extraction uses integer degrees and exact angular integrals, with no division by a radius. The full Cartesian polynomial is then reconstructed and its coefficients compared exactly, rather than accepting angular projection alone.

All 44 primary input columns and all 156 all-m input columns pass, representing 468 independent radial-jet actions. Every m=-J,...,J is reconstructed directly, including negative m; the frozen i^(L+spin-J) phase is inherited once. All symbolic Cartesian residuals are exactly zero. The largest nonzero rho power is two. The exact derivation took 13.57 seconds.

The generated header stores binary64 approximations to these exact radical coefficients. `core_envelope::Apply(J,rho,jets)` takes derivative-first arrays jets[0]=W, jets[1]=W_rho, jets[2]=W_rhorho, each with room for 20 amplitudes; it returns the J-specific amplitude action. It evaluates only 1,rho,rho^2. At rho=0 it therefore gives the regular amplitude action directly. It does not recover amplitudes by dividing vanishing Cartesian fields by r^L. Nonzero regular envelopes whose Cartesian value vanishes at the origin are retained.

## Declared compiled gates

The original plan and root review precede the symbolic/compiled runs. The standalone evaluator is newly compiled in Release and ASan/UBSan modes. The actual C0 point kernel is the unchanged, previously frozen full dual bridge in its Release and ASan/UBSan builds; before reuse, all 1055/1057 compiler dependencies and four link archives per build are reverified by hash. Exact compile commands, compiler version, dependencies, executable hashes and run commands are retained.

The declared 19,500 queries cover J=0,1,2, all nonnegative m real/imaginary phases, four oblique directions, the origin and r=1e-8,1e-5,0.001,0.01,0.025,0.049, with envelopes 1,rho,rho^2,rho^3 and a mixed cubic polynomial. Every native query has Omega exactly one. The main tolerance was 5e-11 scaled; raw algebraic normals use 5e-12. Results are:

| Check | Largest scaled action error | Largest absolute entry error |
|---|---:|---:|
| Full22 native RHS versus polynomial action | 3.98240e-16 | 1.77636e-15 |
| Full22 input lift/layout | 5.55112e-17 | 2.77556e-17 |
| Native Release versus ASan RHS | 0 | 0 |
| Polynomial Release versus ASan RHS | 0 | 0 |

The maximum native raw algebraic-normal residual is 4.44090e-16 scaled. Both Release/ASan output pairs are byte-identical. All four processes exit zero with empty stderr. Native Release/ASan queries took 0.5465/26.8215 seconds; standalone polynomial Release/ASan queries took 1.4792/20.4319 seconds. These are local print-heavy checks, not evolution benchmarks.

Absolute and near-zero readbacks are preserved: the polynomial RHS contains 1254 exactly zero actions and 5395 nonzero actions with norm below 1e-12; the smallest nonzero norm is 1.90731e-81. The native count of exact zeros differs by two because of roundoff cancellation; those entries remain in the residual. The error scale is norm(residual)/max(1,norm(expected)), and should not be read as relative accuracy of each near-zero action.

The polynomial blocks also match all 18 saved fitted B0/B1/B2 core matrices at r=0.025 and the flat endpoint r=0.05. The worst scaled block discrepancy is 8.40736e-11, below the declared 2e-9 threshold. Its largest entry discrepancy is 3.30216e-9 in J=2/B1/r=0.025, where the older angular fit has raw condition number 3.16742e6. This residual is stated explicitly; exact symbolic blocks and floating fitted matrices are not represented as bitwise identical.

## Independent held-out Cartesian polynomials

A separately predeclared driver constructs raw Cartesian monomials of every total degree zero through four in each of the 20 physical tangent columns at the origin and three nonzero core points. Its scalar/vector and five orthonormal STF tensor seeds do not use CG harmonics. It compares the unchanged frozen `FlatFormula` and `FlatConstraints` bodies with actual full22 `ActualDual` and physical eight-constraint diagnostics. The formula extraction is byte-pinned and unmodified.

All 2800 cases pass in Release and ASan/UBSan. Both modes give full22 RHS error 4.17262e-17 scaled / 5.20418e-17 maximum absolute, and physical eight-constraint error 2.77556e-17 scaled and absolute. This validates the independent core Cartesian oracle and lift, including origin data. It does not assert that every arbitrary Cartesian monomial belongs to the J<=2 subspace; the exact all-m and envelope-materialization gates validate the derived J blocks separately.

## Reviews, preserved failure and limits

Root and sibling-agent independent read-only source reviews found no correction to the physical normalization, damping, STF Hessian terms, exact homogeneous projection/reconstruction or origin scope. They did not perform a new numerical rerun.

The first symbolic attempt tested a candidate rho degree before computing whether its angular coefficient was zero. A J=2/L=4 to L=0 candidate therefore triggered the degree assertion even though its coefficient vanishes exactly. The complete source/plan/stdout/stderr are retained in `history/001-zero-projection-degree-check`. The assertion now follows the exact zero-coefficient test. No core formula, basis, coefficient, tolerance, radial division or envelope constraint changed. All accepted sources and compiled files are separately hashed.

The origin formula completes this local core action gate only. A finite-rb radial control still needs its explicit lower-order reduction, normalized boundary map, source-matched dense mass/bulk/trace identities, conditioning and variable-coefficient estimates, manufactured Einstein/constraint rates and separately scoped incoming SAT. No such control is run or admitted here.
