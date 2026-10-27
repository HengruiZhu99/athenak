# Frozen wormhole scri-pole localization

This read-only diagnostic samples the exact archived initial data and the completed AMR/log-halo pilot's accepted midpoint checkpoint at t=.0001. It does not solve a new shift, evolve, reset a field, change SAT terms, impose spherical symmetry, or weaken any gate. All216 scri points are retained, including face edges/corners. Both six- and seven-point radial reconstructions use the same N6 volume data and unchanged six-point angular reconstruction.

The diagnostic compiles its new driver against `amr-log-halo-coupled-20261027/source.tar.gz`, then reads that pilot's paired checkpoint and source tag. No live PDE header precedes the frozen include directory. The independent local pole evaluator agrees with the full tensor kernel's Axx pole to1.34e-15. It reconstructs the original fixed-angle/null shift value from each sampled metric and chi; no beta derivative or new shift solve is needed by these pole sources. The scri reconstruction uses zero halo terms: its stencil consists only of active cells in the outer shell and explicit surface values. Thus the measured endpoint defect does not directly read Cartesian/coarse-fine halo values, though those may have influenced the previously evolved fields.

## Initial radial error

For this isotropic M=1/3 slice, the entire outer shell lies in the s=1 hyperboloidal branch: tilde-g=identity, Atilde=0, Khat=-3, Theta=Lambda=0. Writing y=1/R and delta(y)=integral[1/sqrt(1+y²-2My³)-1/sqrt(1+y²)]dy gives delta=M y⁴/4+O(y⁶). The isotropic map obeys -log r=asinh(y)+delta. Consequently chi=1-(M/2)y³+O(y⁴), with y=(1-r)+O((1-r)²); chi'(1)=chi''(1)=0. Likewise alpha-Aref=-(3M/4)y³+O(y⁴). These are endpoint limits of this particular initial slice, not universal puncture or evolution assumptions.

The sampled initial normal chi derivative is2.04903e-6 with six radial nodes and−6.18772e-7 with seven nodes (magnitudes reported in JSON); tangential chi derivative stays below6e-17. The largest pole is Axx,2.51636e-6 or7.59895e-7 respectively, near a face center. Replacing only the initial chi first derivative by its independently derived zero endpoint limit in a temporary diagnostic jet reduces all initial poles to7.41e-16. This localizes the initial defect to radial endpoint truncation, rather than angular patch matching or an incompatibility of the physical initial slice. It is not an admissible evolution fix to force chi'=0 later without deriving the evolving closure.

## Evolved angular connection defect

At t=.0001 the largest pole is LambdaY=7.285255309e-6 at face1,j0,k3, Cartesian point(-.7620007620011431,-.6350006350009526,.12700012700019056), adjacent to a face edge. Six radial nodes give7.284409409e-6, a change of only8.46e-10. The A components improve with seven radial nodes, but the dominant connection pole does not.

For stationary Omega, the original Lambda pole is

    S_Lambda^i = -alpha(4K/3+2kappa1)(Lambda^i-Gamma_contract^i)/2
                 -(2alpha/3) g^ij(2 d_j Khat+d_j Theta)
                 -4alpha A^ij d_j Omega.

Its exact first-derivative directional split at the worst point is:

| Contribution to S_LambdaY | Value |
|---|---:|
| Metric connection from normal-direction metric derivatives | +6.539420853e-7 |
| Metric connection from tangential-direction metric derivatives | +5.782864471e-6 |
| Independent evolved Lambda damping | +9.174027025e-7 |
| Normal trace-gradient term | −1.422691757e-7 |
| Tangential trace-gradient term | −6.014980945e-9 |
| Atilde-gradient-Omega term | +7.933020729e-8 |

The decomposition reconstructs the full pole within2.55e-21. Setting metric first derivatives to zero in a temporary jet reduces this component to8.48449e-7; removing trace gradients increases it to7.43354e-6. Chi derivatives do not enter this Lambda pole. These ablations isolate terms; they are not proposed projections or a change to the independent Lambda equation.

Tangential differentiation of Cartesian tensor components can be nonzero for an exactly spherical tensor. Therefore tangential dominance alone does not prove a faulty angular derivative stencil. As an additional diagnostic, fitting all216 evolved metric values to the spherical form delta-g=a I+b nn gives a=−1.11302646e-7 and b=3.33908004e-7. The full metric deviation is4.81776e-7, while the non-spherical fit residual is2.76745e-7 maximum and8.20506e-8 RMS. The fitted spherical tensor's linearized tangential contribution to the worst LambdaY pole would be−8.48127e-7; the measured contribution is+5.78286e-6. There is substantial evolved angular structure beyond that spherical fit. The fit is never applied to the state or included as an exclusion in acceptance.

The next falsifiable diagnostic should compare the initial full vector shift's first derivatives and resulting metric rate on the outer shell/surface against an independent radial shift oracle, then propagate only the measured discretization error into the connection contraction. In particular, measure angular scatter of the initial metric rate and its normal derivative at the same edge-adjacent locations. Increasing only radial scri order cannot repair the dominant evolved Lambda defect. This audit does not yet establish whether the nonspherical structure originated in the global elliptic source/operator, boundary derivative reconstruction, or the regular tensor scri rate.

## Reproduction and preserved attempts

```sh
python3 tst/elliptic_shift_3d/run_scri_pole_diagnostic.py --output NEW_DIRECTORY
python3 tst/elliptic_shift_3d/analyze_scri_pole.py NEW_DIRECTORY
```

The qualified complete output is `evidence/scri-pole-angular-audit-20261027`, with raw surface samples, all source/input/library hashes, frozen headers, commands and stdout/stderr. Build1.8s and diagnostic1.7s, with a180-second process bound. Earlier `scri-pole-localization-qualified-20261027` and `scri-pole-decomposition-20261027` retain the successive diagnostic detail. The first runner attempt under `scri-pole-localization-20261027` failed before compilation because the system Python tarfile lacked the newer filter keyword; explicit checked regular-file extraction fixed the runner. No evolution result was lost or replaced.

## Cross-face angular interpolation probes (not adopted)

`CenteredSurfaceAtlas` evaluates a constant-on-radial-rays extension of each global Cartesian stored component. Centered fourth-order differences in the original patch's two angular coordinates can query a neighboring physical owner face. First and second Cartesian derivatives use both J and H, including the full mixed angular chain rule. The frozen-field comparison retains original values and radial first derivatives; it changes only temporary diagnostic angular jets. The original all-field pole maximum7.28526e-6 decreases to1.02599e-6, still failing the unchanged1e-7 reference gate. A metric-only angular replacement gives1.74660e-6 maximum. This is diagnostic sensitivity, not an accepted solver change.

The independent nonradial AD manufactured function is a cubic polynomial in the unit direction n=x/r. Tests cover every surface node plus exact face seams/corners at N6,12,24. Constants preserve value/first/second derivatives exactly. Global first-derivative errors decrease .067351,.001613,7.674e-5; global second-derivative errors decrease2.72741,.069980,.00314936. Exact seams are poorly resolved at N6. See `evidence/scri-centered-atlas-20261027`.

### Same-field mixed derivative construction

The standalone `CrossFaceBoundaryProbe` implements this precise possible integration into `ScriSurface::Sample`:

1. Retain the existing six/seven radial Lagrange nodes and coefficients, including the exact Omega=0 node. The pure q0 and q00 derivatives retain the original radial fit.
2. Pull back the same stored field F through the target chart at all tensor-product sample positions `(s_i,u+jh,v+kh)`, j,k=−2..2. Every off-face physical query resolves its global owner; interior queries interpolate active volume values and endpoint queries interpolate explicit surface values. Cartesian field components never rotate.
3. Differentiate that single pullback: qA and qAB use centered fourth-order angular weights, q0A uses radial-first times angular-first weights, and q00 uses the original radial-second weights. Transform all first/second derivative directions using the full J/H chain rule. Pure-normal derivatives alone cannot supply the mixed normal-angular derivatives.
4. Reconstruct physical positive-field jets through the existing chart interface after stored-component differentiation. Apply the same optional sampling contract to the algebraic delta-beta components required by the tensor-limit closure. No field projection, gauge/source change, puncture pin, or normal-data substitution is implied.

A future implementation should precompute these linear geometry/transfer weights, coalesce identical active/surface donors, and cache by mesh revision and radial/angular scheme. Creating interpolation searches during every source evaluation is unnecessary. The flag must default to the existing behavior; restart metadata must bind it and rebuild plans. It must remain separate from the physics operator and actual boundary equations. No such engine flag or checkpoint change has been made here.

Fourteen bounded MMS configurations cover a one-layer cube-to-sphere transition (whose physical radial spacing depends on angle), two and seven layers, subdivided faces, six/seven radial nodes, and nonradial fields with nonzero normal-angular Hessians. All surface nodes, edge/corner bands and exact patch seams/corners are reported separately without exclusions. Constants remain exact; pure-normal first/second derivatives differ from the old sampler by at most1.64e-11 roundoff. Each configuration stays below50,000 volume cells and runs in under0.7s. The three related resolutions are N6/12/18 for one shell, N6/10/14 for two shells, and N6/8/10 for seven shells. The subdivided seven-shell N6 case is a separate topology control.

These checks expose a transfer error rather than qualifying the proposed sampler. For the complete manufactured set at exact seams, one-shell N6/12/18 second-derivative errors are22.4492,.550867,.163684. Replacing only off-node interpolation by an analytic MMS oracle gives .211763,.0151151,.00299210. Normal-tangential Hessian errors similarly decrease1.74980,.0350365,.00366230 with stored-field transfer versus .118849,.00759830,.00150089 with the analytic pullback. The analytic oracle is used only to localize the MMS error; no physical run uses it. Cross-face owner-polynomial extrapolation remains a substantial obstruction to using the full mixed sampler at coarse resolution. Evidence: `scri-full-boundary-jet-probe-20261027` and `scri-full-boundary-jet-oracle-20261027`.

### Guarded global-donor endpoint MLS

`SurfaceMLSProbe` is a second, endpoint-only diagnostic. It uses the48 globally nearest surface nodes in a query-centered gnomonic tangent chart and a degree5 two-dimensional polynomial basis (21 terms). A pivoted twice-reorthogonalized QR solve constructs geometry-only value weights. It requires rank21, infinity condition number of the triangular QR factor below1e8, weight L1 norm at most8, and scaled polynomial reproduction below1e-10. Nodal queries return the original nodal value exactly; no state values are fitted or overwritten. Off-node MLS weights depend only on geometry. Constants are evaluated through anchored differences. This is an interpolation prototype, not a derivative filter applied to a completed RHS or a projection of the evolved field.

The N6/12/24 MMS suite has exact constant preservation and polynomial reproduction at most6.32e-16; maximum weight L1=1.689. The final manifest reports actual infinity conditions of the triangular factor, replacing the earlier diagonal-ratio proxy preserved in the first MLS attempt. Exact-seam/corner Hessian errors improve to .223757,.00470086,.000100192. However, the N6 native-node corner-band Hessian error .235678 is worse than the original owner-face .101124; that counterexample is retained. Improved seam or pole values do not establish uniformly improved accuracy or stability for unresolved angular data.

On the same frozen midpoint field, all216 points give maximum pole9.14445e-7 (Axx), with Lambda components below7.36934e-7. The original worst LambdaY falls to5.95319e-7. The pole gate still fails. The frozen-field weights have L1 at most1.5191, QR-factor infinity condition at most114.486, reproduction error4.45e-16 and unchanged nodal values within2.17e-19. Endpoint MLS has **not** replaced the off-face volume interpolation needed by q0A. Therefore the mixed-jet obstruction remains and no evolution adoption is proposed yet. An admissible next test is a guarded local three-dimensional active/surface-donor interpolation (degree5,56 terms), with anisotropic geometric scaling, identical field-independent rank/condition/L1/reproduction checks, and the same transition/subdivision/mixed MMS controls.

Reproduce these disjoint probes with `run_centered_surface_atlas.py`, `run_cross_face_boundary_probe.py`, and `run_surface_mls_probe.py`. Final endpoint evidence is `evidence/scri-surface-mls-conditioned-20261027`; `analyze_surface_mls_probe.py` recomputes its summary and leaves the reference1e-7 pole gate false. No engine, shared checkpoint, physical equation, SAT, solve tolerance, or evolution acceptance has been changed by these probes.

The subsequent volume-transfer and zero-boundary-trace falsifiers, final physical-normal Hermite construction,20-case6/7-node MMS matrix, and frozen-pole/Minkowski checks are recorded in [boundary-transfer-probes.md](boundary-transfer-probes.md). That standalone candidate resolves the interpolation-consistency obstruction without an engine adoption; the frozen wormhole scri pole remains above the unchanged scientific gate.
