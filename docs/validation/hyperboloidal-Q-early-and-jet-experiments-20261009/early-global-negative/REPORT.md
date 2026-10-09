# Earlier Wgauge null feedback: negative global attribution control

Replacing the late feedback onset with Wgauge does not rescue the catastrophic
N16 projected-continuous t2 result. Every endpoint H/M/Z norm is larger than
the late-onset control, and both systems remain orders of magnitude worse
than the original C0 spatialnorm control. No t6 or long native run is made.
This is one fixed finite-Omega discrete-operator screen, not a certified
eigenvalue, continuum instability, energy estimate or native trajectory.

## Exact single change and admitted inputs

The helper is the independently frozen earlynf::Gauge(p,u,g,{5,0}) followed
by qnf::Assemble. Mode0 uses v=Wgauge(.45,.85), replacing
SmoothCutoff(.85,.95) in only the null-residue beta pole. The physical-inner
P/Q alpha blend and preferred-Q regular shift source are unchanged. Inputs
explicitly remain physical_trace_lapse=false, preferred_source=true,
scri_lapse_damping=2. No helper silently toggles input flags. No C1, kappa2
profile or other candidate is combined.

The grid/reference/stencils/initial data are exactly the late-Q and original
C0 controls: N16/span2.2, 1640 active points, h=.1375, Omega_min=
.0026953124999994555, S1/a.5/referencewide(.05,.95), C0 kappa_input10/kappa2zero,
symd2 strictly interior nonrecursive ghost interpolation, native centered,
mixed and upwind derivatives and KO=.1. The original gauge/shell shapes are
normalized to unit free20 Euclidean norm identically in all three systems.
Their seed archive is byte-identical. This linear comparison is separate
from root's finite-amplitude native preflights.

The scientific index is
902d875e6da80bf3b10e7103aa59cf123c4d8a034862f0a5d9be6f9e220c56c0;
the independent review is
273fdab75ae847d4a54a6cd8b50334a215a839969de39c5bb7378b50fa24b19a.
Both were read/hash-verified before binding and compilation. Runtime helper
562d01b4382c5f9c36afc83b29208f5db3d90ca37f12890c92c03b101eb36a69 includes
the unchanged core/base f3f3acfe.../216bb80c... sources. All 36 scientific and
11 review files, 376 input paths, eight successful commands and the separately
pinned postprocessor are checked. Full hashes, compiler/flags, dependencies,
link archives and executable identities are in the source/build receipts.

Only the copied forced-include wrapper changes; production CartesianPatch and
implementation 27c19d20696ea6dd4704032c51dfd026218f64f2 are untouched. The
analytic geometric RHS subtraction, reference reconstruction, ghost filling
and actual native projection lifecycle remain unchanged. The beta pole is
included exactly once. Strict positive gradient norm is required on support;
there is no floor or SPD repair. Admission is the reviewed fixed geometry,
not arbitrary gauge/geometry parameters.

## Source, full22 and native-stage checks

Prepared reference change is exactly zero. Maximum reference RHS is
6.0836e-12 (small-Omega roundoff retained); reference H/M/Z are
3.2395e-14/1.3306e-15/6.6431e-16. All 63576 donor references are strictly
interior/nonrecursive, with weight-sum error1.3323e-15. The six inherited
implementation-consistency gates pass: raw CSR/cache relative1.1059e-15,
worst best-vector native22 RHS centered-amplitude error8.9865e-10,
actual final-only one-step derivative error1.8937e-10, PL maximum error
2.9950e-16, reference/support and server-exit checks. These are consistency
thresholds, not physical acceptance.

An independent analytic reference expansion predicts the difference from
the late operator:

    delta beta_dot^i = (Wgauge−Vlate) sigma Omega_i/(|dOmega|² Omega)
                      [h² deltaG+2Bh((Bh/h)deltaAlpha−dOmega.deltaBeta)]
    deltaG = dOmega_i [deltaChi ginv−chi ginv deltaMetric ginv]^ij dOmega_j.

Here h=alpha_ref and Bh=beta_ref.dOmega. This changes only local chi/g/alpha/
beta value columns in beta rows. The actual raw22/projected20 differences
match it within 2.0010e-10/2.0309e-10 absolute versus maximum predicted
coefficient94.9370. Every non-beta matrix row is bitwise unchanged from late.
The independent full difference versus original C0 also passes within
2.252e-9/2.175e-9. Tiny cached finite-difference coefficient discrepancies
are retained without coefficient dropping. Initial native constraints and
C_h J_h component RMS match late/C0 exactly for all four validation seeds.

Raw22 matrix SHA is
a861623ecac66bc9dcd17338f77949433bc1ccd8890b53c12c79b37dfdf7bc7c;
projected20 SHA is
714e6e509a8b7ad22c8fde4e95c8bbb5f5d7cad77c94d262b6501407c5fb5880
(15111165 nonzeros). Exact lift/restrict, raw CSR and native RHS comparison
amplitude sweeps are retained. The finite-stage oracle is
P_ref R3(dt J22) Lift, with final-stage-only projection, compared directly
against actual nonlinear native one-step centered differences. Its dt
halving/quartering comparison with the distinct R3(dt P_ref J22 Lift) is
preserved. The histories below evolve the continuous projected generator,
not the exact finite native RK3 map.

## Action verification and matched t2 results

Independent canonical Taylor actions at .025/.05, using an exact constant
configuration 1/h similarity then undoing it, agree with Arnoldi within
4.241e-14 relative. Canonical cost is94.971s. Exploratory t2 costs67.660s,
6400 matrix-vector plus240 direct curve-residual products. Maximum local
50/80 truncation-pair difference is9.956e-11; direct curve defect divided
by state norm is<=2.074e-10 per unit time. The independently checked short
prefix agrees within2.192e-14. These are empirical controls, not a rigorous
nonnormal forward-error bound; no long independent canonical action is made.
All saved states and JSON values are finite and the1e12 guard does not trigger.
Thirty BLAS/scipy floating-status RuntimeWarnings in the original pilot/t2
logs remain retained without an assigned cause.

H is native physical Hamiltonian RMS; M/Z are native conformal covectors
contracted with the reference Penrose spatial inverse and use unweighted
cell RMS. Field-unit amplification uses the unchanged reference-volume
configuration-value/derivative plus momentum-value component norm. It is not
an invariant tensor energy or a proved symmetrizer.

| Seed | Early W H/M/Z | Early/late H/M/Z | Early/C0 H/M/Z |
|---|---|---|---|
| Gauge | 11471.899 / 37398.250 / 8004.284 | 1.05576 / 1.20300 / 1.40095 | 5232 / 30644 / 21650 |
| Shell | 3176.536 / 10248.979 / 2303.739 | 2.00942 / 2.25958 / 2.73852 | 1.268e6 / 4.486e6 / 2.313e6 |

Gauge field-unit amplification is647824.3 versus late423893.1/C0 35.7290;
shell56729.9 versus18730.0/.0169943. Free20 Euclidean amplifications are
1013361.6 and298415.4. Full histories, sampled maxima, Theta/component groups,
radial localization and native diagnostic amplitude controls are in summary
and the saved analysis payloads. At t2, gauge r>=.9 squared H/M/Z fractions
are .0742/.7677/.9441 and shell .0710/.7317/.9403. Peak radii are
.89902/.91981/.99865 for both. The constraint-bearing shell result prevents
attributing the entire observation to growth from initially zero constraints.

The earlier support substantially reduced some sampled frozen primitive
roots in the independent local gate but did not improve this actual global
endpoint. This controlled result shows that the late onset gap is not
necessary for the observed large finite-time amplification of this candidate
on this grid. It does not establish its cause, a global rightmost eigenvalue,
or a continuum limit. Below-coordinate-Nyquist k16 in the separate local gate
does not certify finite-difference accuracy or an actual discrete growth mode.
Full preferred Box identities still apply only where W=1.

Before freezing, all2312 new build dependency references and callbacks,
original71 frozen small files, late107 frozen/105 working small counterparts
and both late compiled dependency/executable identities are rechecked. The
two omitted working counterparts are the collector's open stdout and a
snapshot-generated metadata file; their immutable copies are verified.
The initial verifier assumption that the latter also existed in the working
directory is preserved as a reader-only correction under history. No old
scientific source, executable, array or frozen evidence is modified.
