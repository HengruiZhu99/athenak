# Next shared initial-profile implementation

**Implemented and checked in v14.** See [COMPACT-PROFILE.md](COMPACT-PROFILE.md)
for the shared C++ implementation, independent geometry checks, matched 10M
results and unresolved differential boundary gate. The list below records the
design that motivated the implementation; it is no longer an unimplemented
proposal.

The v11/v12 evidence localizes the remaining 10M error, rather than establishing
an admitted candidate. At N=256, the areal 20–80M map has peak exterior H/M near
compact r=.953/.957 (reference R/M=62.5/68.2). The Lambda/D gradient half-maximum
covers only two grid points, at r≈.949. Central refinement improves the horizon
mass but loses outer accuracy; the two-end grid improves both at fixed N, but
its weak-penalty variant has a scri-localized growth eigenvalue ≈401.7 and fails
near .207M even with a four-times smaller timestep.

The next slicing candidate should therefore implement the **compact-profile**
construction already supplied in the parent handoff's `COMPACT-SLICING-STUDY.md`,
not just broaden the areal S(R) switch again. The latter changes the central
compact allocation and still leaves a sharp outer feature.

Required implementation and checks:

1. Preserve the native 20–40M baseline's inner compact radius a and isotropic
   scale k for the same user-selected central reference radius. Prescribe
   S(r)=I_z(p,7), z=(r-a)/(b-a), with b=.97 as the first explicit control, and
   solve p so the backward map y'=-sqrt(y²-2My³+S²)/r, y(1)=0, satisfies
   y(a)=1/(20M). Validate ordering, monotonicity, finite values and both plateaus.
   This is an initial-coordinate/reference construction, not an elliptic
   evolution solve or a reset of evolved constraints.
2. Put the profile/map and derivatives in shared C++ code. For integer second
   beta parameter 7, the incomplete-beta value can be evaluated as
   z^p sum_{j=0}^6 [(p)_j/j!] (1-z)^j; its density is
   z^(p-1)(1-z)^6 / B(p,7), with 1/B(p,7)=p...(p+6)/720.
   Keep the existing native areal profile bitwise unchanged when not selected.
3. Derive the geometry from this same map: chi=(r y/Omega)^2,
   R S_R=r S_r/a_s, K=-3S-r S_r/a_s and Ar=-2 r S_r/(3 Omega a_s).
   Retain the actual free initial lapse prescription, not an accidentally
   stationary replacement in the central region. Distinguish it from the
   stationary reference lapse Omega*a_s and beta=-Sr used for gauge sources.
4. Make `StationaryGaugeReference` and the spherical initializer use precisely
   the selected map/profile. Keep the single central configuration and its
   conformal/gauge joins derived from a. Check the complete actual RHS rather
   than assuming the background is stationary under the inner telegrapher.
5. Audit NativeBoostedWormhole's Tail/Fields/Radial paths before exposing the
   option there. They currently reconstruct the areal switch independently.
   Explicitly reject the new option in unsupported boosted paths; do not let
   an enum fall through to an old formula. No boosted claim or Aurora job is
   authorized. Unboosted shared C++ parity and a future small spherical 3D
   comparison remain separate from boosted validation.
6. Compare the new C++ initial data, derivatives and physical constraints with
   the supplied analysis-only Python construction, including tolerance
   refinement and changing-geometry/off-constraint evolution parity. Then run
   the cheap Release-build 1M and 10M controls on common reference coordinates.
   Keep earlier failures, timestep guards, actual characteristic speeds and
   differential D checks. Reference-tail options must be matched explicitly.

Current endpoint derivative work is already available as an independent control:
`NativeWormhole::OuterRadialDerivatives`, `--reference-tail scri|outer`, with
70-digit verification. It lowers the small 1M D-defect floor but does not fix
the large 10M error. The compact-profile implementation and its new evolution
evidence are now recorded separately in COMPACT-PROFILE.md.
