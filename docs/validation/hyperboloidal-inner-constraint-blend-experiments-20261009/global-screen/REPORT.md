# Prescribed bulk C1 Cartesian tangent: small mixed exploratory screen

The exact shared helper multiplies every mechanical C1 term and the derived covector Lambda repair by C(r)=1−W_gauge. It is the prescribed bulk blend, not the full covariant system. The scientific coefficient-gradient/principal/Einstein/outer-pole gate passed before compiling this separate actual Cartesian full22/native20 experiment. The short independent propagation gate passes; the t2 screen gives only small mixed changes and does not justify a long native/Taylor run. No production option or stabilization is adopted.

Parameters match the frozen C0 and full-C1 comparisons: N16,span2.2,h=.1375,1640 active nodes,32800 free20/36080 raw22,S1,a.5,wide reference(.05,.95),gauge(.45,.85),kappa10,symmetric quadratic ghosts,nativeKO.1. Theta remains unrestricted on the strict interior grid. Source helperBulkC1Additions is byte-pinned4c7b6637fc9c38339134d5d5824e589986110edc9efa55489ebe0fc941bafbe2, common C1 math908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7. All27 frozen gate-index files were checked against index1bb5f697691404c7a8ed19c20fc77aa79c5aa3fa31a81248d10c7e958d3ac2f0 before compilation.

The copied CartesianPatch calls the live shared helper after existing C0 analytic-reference roundoff subtraction. No blended C1 reference subtraction is added. Cached Point makes the identical call with p.radius and the same layer_gauge. No lapse/shift, stencil, ghost, projector, mask or KO change is made. Mathematical dC terms belong to the derived constraint propagation gate; the actual global operator multiplies live pointwise RHS additions by C and its native spatial constraint differentiation samples that variation. No value-only interpolation of subsidiary matrices is used here.

## Native implementation and exact support

Prepare change0, referenceRHS6.08357e-12 and H/M/Z3.23947e-14/1.33057e-15/6.64311e-16 match C0. All63576 donor references remain strict interior/nonrecursive, constant-weight error1.33e-15. Raw CSR/cached native stencil agreement is1.11e-15. Six-amplitude actual native22 RHS derivatives agree to worst best7.25e-10; actual final-only SSPRK3 derivative versus P_refR3(dtJ22)Lift agrees to1.93e-10 across tested seeds and dt/2,dt,2dt. Nominaldt=.03minOmega=8.0859375e-5. Twelve retrospective implementation checks pass; they are not physical acceptance criteria.

The actual projected sparse matrix provides a direct support check: all20 rows at each of672 nodes r>=.85 are numerically EXACTLY equal to frozen C0. Chi/g/alpha/beta rows are exactly equal everywhere. Initial pure-gauge pulse J20 action is exactly equal to C0. The maximum changed coefficient is7.7524414 in inner geometry/momentum rows. These are matrix support identities, not a global boundary energy or stability theorem. In particular, the original instantaneous native gauge constraint source is not removed by this lower-order blend.

## Verified short and exploratory long propagation

The propagated generator is J20=P_ref J22Lift, continuously projected semidiscrete evolution. It is distinct from actual native finite final-only RK3. Direct Taylor expm_multiply and independent two-pass Arnoldi m50/80 agree at t=.025,.05 within2.86e-14 for both seeds and both gauges. Taylor short controls cost72.59/70.85s. At t=.05 smooth H/M/Z ratios to C0 are.997807/1.002004/1.000494 production and.997810/1.002099/1.000663 spatialnorm; shell differences are<=.33%. The chosen component/derivative norm is likewise nearly unchanged.

Long t2 Arnoldi costs77.66/77.63s, with local coarse/fine differences<=1e-10. Independent long canonical comparison remains pending by design: the small mixed screen does not justify that cost. This local empirical truncation test is not a rigorous nonnormal forward-error bound. The following results are exploratory, single-grid, finite-window evidence, not exact native t2 histories or a theorem of all-time failure/success. No eigenvalue/Ritz search is used.

| Gauge/seed at t2 | Blend H/M/Z | Blend/C0 H/M/Z | Blend cfgH1+momL2 amplification(C0) |
| --- | --- | --- | --- |
| Production gauge |1.691591/1.218061/.264725|.98322/1.00027/.97123|46.7621(47.0983)|
| Spatialnorm gauge |2.209822/1.268840/.352081|1.00783/1.03969/.95232|35.9798(35.7290)|
| Production shell |.00345536/.00385030/.00097727|.99308/.99395/1.00843|.075721(.074887)|
| Spatialnorm shell |.00272555/.00261090/.00116520|1.08830/1.14278/1.16982|.018220(.016994)|

Seed arrays and normalization exactly match prior frozen controls: native width.5 angular gauge pulse and controlled RNG690 radial-shell random vector, each normalized to unit Euclidean free20 L2. Generation sources/hashes are retained. H is physical Hamiltonian; M/Z use native Penrose inverse contractions and unweighted active-cell RMS. The field units diagnostic is h^3sqrtgamma quadrature of configuration{chi,g,alpha,beta} H1 plus momenta{P,A,Lambda,Theta} L2 at S1, counting stored upper tensor components once. It is not invariant tensor energy, symmetrizer or a proven bound. Both shell sampled maxima in that norm are1 initially. No field positivity/SPD claim is made for finite states from a linear tangent vector.

Smooth final squared outer r>=.9 H/M/Z fractions are.06917/.39923/.60230 production and.03471/.48848/.72904 spatialnorm. PeakH remains in the bulk (r=.22802/.35724); peakZ near r=.99865. These are localizations of a measured error, not its boundary-causation proof. Native signed constraint amplitude sweeps and all finite-output checks are preserved. Diagnostic shell-pole columns remain C0 in this tangent-only copied header and are neither used nor claimed; the comparator uses actual EvolvedConstraints H/M/Z. Root's independent native build adds separate complete pole diagnostics without changing evolution.

## Provenance and limits

Builds launch at aef47b0a with production headers byte-identical27c19d20 plus the explicit blend overlay/math. Freeze HEAD is recorded separately after parent documentation commit80b83eab. Exact commands, AppleClang21 arm64 flags, four native/tangent compiler dependency inventories and Kokkos static archives are rehashed unchanged. All build outputs remain in this fresh blend tree. The prior C1 command-path correction stays in its immutable archive; both original C0 scratch oracle hashes remain unchanged here. The initial SOURCE_PREPARATION_HOLD.json is retained as historical source-only status, superseded by gate-bound-sources.json and compiled build-provenance.json.

Sources/overlay diffs, gate-index identity, all native stage/sweep observations, short canonical checks, long empirical receipts, exact sparse support audit and native constraint/field diagnostics are copied. Large CSR/state/seed/diagnostic arrays and executables are metadata-only with path/shape/size/SHA256. Existing C0/global/fullC1/stage/wave frozen archives are unchanged. This screen accepts no nonlinear scri closure, complete covariant formulation, uniform energy estimate, finite angular pulse or black-hole transition. Any damping-profile follow-up stays in a separate source tree and requires its own coefficient-gradient/pole/principal gates.
