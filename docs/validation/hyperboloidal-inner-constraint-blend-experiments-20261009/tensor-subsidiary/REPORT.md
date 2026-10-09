# C1 and bulk-blend constraint gate (scratch, reference tangent)

All10 compile/run/check commands pass, including Release and ASan/UBSan tensor
gates. The380 recorded source inputs are unchanged:365 tracked src/CMake files
and15 audit/helper/versioned-test inputs. LaunchHEAD is aef47b0a; compiled
production remains27c19d20. No tracked file or existing frozen receipt is edited.

The actual physical-P norm-gauge full20 dual tangent and separately derived
physical subsidiary equations agree on1000 samples each for repaired C1 and
c=1-W_gauge bulk blend, kappa10. There are ten radii .3 through the actualN48
outer radius, ten frequencies0..256, radial/oblique directions and five
coefficient-difference refinements. The finest relative closure error is
7.454e-9 for each. The difference step shrinks near scri so every sampled
coefficient point remains strict interior. The largest coarse error is about
5.29e-4; it decreases by about16 per step-halving in the outer refinement.

Static physical constraint gauge columns are exactly zero. Corrected gauge
Cdot converges toward zero, with fine outer absolute residual up to4.351e-4
on large constraint matrices; its scaled maximum is recorded. The absolute
residual is not presented as roundoff. Frozen QL instead remains nonzero.

All200 directly derived local eight-constraint generators in each kappa10 set
have negative eigenvalues. The largest real part is -1.15132096 at r=.3,k2.
At the primitive C1 positive-root sample r=.9983726994,k128,oblique, the separate
C1 constraint generator has maxRe=-1335.503; the blend has -1486.279. This
distinguishes the local operators. It does not assign a primitive eigenvector
to a subsidiary branch, prove a global spectrum, or explain all native growth.
Full C1 native/global negative experiments are separate parent-owned evidence.

The blend's analytic grad-c term is required: omitting it gives relative
closure error .01290836 at r=.65,k0, versus the correct refined agreement.
The negative control is retained with all matrices. This check uses the actual
spatially varying coefficient, not a value-only interpolation of frozen8
matrices. Its derivative terms remain homogeneous in constraints.

The new byte-pinned `bulk_c1_additions.hpp` multiplies ALL mechanical C1 parts
and the covector repair by the same existing1-W_gauge coefficient. Its actual
360 principal cases preserve the complete basis: symbol error3.553e-15,
left-field error8.882e-16, normalized basis condition11.5292. In4004 nonlinear
tensor/dual rows, exact Einstein-sector additions, unrestricted outer additions
and outer cutoff value/first-three jets are all zero. Release/Debug results
agree exactly. Reference additions are <=1.40e-15. The support Omega lower
bound passes for a=.5/.75/1/2; at a=.5 the analytic bound is.2775.

The blend is therefore eligible for a separate fixed-finite-Omega actual RHS/Jv/
final-RK and short global/native exploratory screen. This gate does not accept
it as stabilization or authorize production adoption. The existing C0 outer
pole certificate is unchanged because the added helper is exactly zero there.
It does not establish nonlinear scri closure, a uniform energy estimate or a
later black-hole transition. Physical Theta remains unrestricted; no floor,
falloff forcing, reference-RHS subtraction or gauge change is introduced.

DERIVATION.md states the full tensor and blend equations; ENERGY.md records
the actual damping/normal terms and the remaining uniform energy obstruction.
The earlier renamed-main compiler warning is retained in exploratory-first-pass.
The independent asymptotic experiment is separate from this completed gate;
its exploratory failures do not modify these frozen results.
