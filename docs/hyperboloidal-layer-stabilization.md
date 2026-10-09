# Hyperboloidal layer stabilization and wormhole data

Ongoing work, 2026-10-08–09 (America/New_York), branch
`z4c_hyperboloidal_layer`, following `5c0951e7`. The first formulation and its
negative results remain in [the formulation](hyperboloidal-layer.md) and
[the initial validation receipt](hyperboloidal-layer-validation.md).
This stage's exact builds, checks and experiments are recorded in
[the stabilization validation](hyperboloidal-stability-validation.md).

The acceptance objective is stable evolution of finite radial and angular gauge
pulses on Minkowski over several gauge/layer crossing times, followed by a single
black hole that survives the inner wormhole-to-trumpet transition **with a
Minkowski hyperboloidal reference throughout**. Neither stage has passed.
Short runs, a negative frozen pole spectrum and constraint-consistent initial
data establish distinct checks; none establishes global evolution stability.

## Physical-trace lapse alternative

The geometric equations, stabilized physical trace `P=Kphys-2*Thetaphys`,
Cartesian connection convention, storage and complete coupled-shift principal
part are unchanged. An independent opt-in
`hyperboloidal_physical_trace_lapse=true` selects a cancellation-aware Cartesian
extension of [2408.08952, Eq. 9](https://arxiv.org/html/2408.08952v2). That paper's
mixed trace is precisely the stored `P`, including off-constraint Theta.
The extension to the nonflat transition and combination with our coupled shift
are derived here, rather than a published stable 3D formulation.

Let `da=alpha-alpha_hat`, `db=beta-beta_hat`, `W` be the independent gauge
cutoff, and `alpha2f=alpha^2+2*(1-W)*alpha`. The new lapse is

```
dt(alpha) = R_alpha + S_alpha/Omega
R_alpha = beta.d(alpha)-beta_hat.d(alpha_hat)-alpha*nu*log(alpha/alpha_hat)
S_alpha = -alpha2f*(P-P_hat)
          -W*xi*(alpha+alpha_hat)*da
          -W*Omega_i*(alpha*db^i+beta_hat^i*da).
```

`xi=hyperboloidal_scri_lapse_damping` defaults to 1.5 and must be finite and
nonnegative. Both reference advection and reference derivatives are analytic.
The last term is the exact expansion of
`W*Omega_i*(alpha_hat*beta_hat^i-alpha*beta^i)`; the restoring product is the
expansion of `W*xi*(alpha_hat^2-alpha^2)`. The reference is a continuum fixed
point. In the Cauchy interior, the lapse reduces to the same modified
Bona–Massó slicing used by the original layer mode. The independent finite
logarithmic restoring term remains available.

The log ratio uses `log1p(da/alpha_hat)` for small relative deviations and
`log(alpha)-log(alpha_hat)` otherwise. This avoids rejecting a positive
collapsed lapse when `da` rounds to `-alpha_hat`. A scaled-RHS test checks both
lapse modes, zero/nonzero restoring rate and lapse values through `1e-300`,
without floors. It is not a claim of uniform hyperbolicity at the exact puncture.

The `1/Omega` numerator is explicit. Its stability is audited below, rather
than assumed from a restoring sign. The implementation still evolves only
strictly interior staggered points; it provides no nonlinear evaluation at
exact scri and adds no floors. The live gauge-speed timestep restriction now
also applies when this mode is used with a pure CMC reference.

`hyperboloidal_curvature_radius` exposes the analytic reference's `a` in the
native adapter. `S=1` remains a native runtime restriction. For an enabled
layer, the sufficient positivity condition remains `a>=S/2`.
The curvature control requires the layer or physical-trace lapse mode, which
uses a live gauge-speed bound. The legacy pure CMC mode retains `a=1` and its
original timestep behavior; other curvature radii are explicitly rejected there.

## Complete principal symbol and frozen scri residue

`kernel_symbol.cpp` extracts the actual constrained 20-field principal matrix
for both lapse modes: 360 coefficient, lapse, chi, metric and propagation
direction cases. The physical-trace replacement changes only lower-order
coefficients. The existing canceled scalar/vector/tensor characteristic fields
and harmonic endpoint therefore apply to both modes; the checked matrix
identity and basis completeness remain independent of the pole audit.

For the new lapse, the complete constrained frozen residue is checked using
the actual geometric/gauge kernel at exact scri, without assembling a live
singular RHS. At `S=1`, normalize time by `c=1/a^2`, relative lapse and shift by
`1/a`, and trace/Theta by `1/a`. Define

```
k = hyperboloidal_kappa1*a^2
d = 1+2*xi*a.
```

The scalar `(relative alpha, log chi, a*P, a*Theta)` subblock is

```
[-d,  0,    -1,        0;
 -2,  0,   2/3,      4/3;
  0, -3,    -2,      k-4;
  0, -3,    -2,    -1-2k].
```

Its characteristic polynomial is

```
lambda^4+(d+2k+3)*lambda^3+(2dk+3d+6k)*lambda^2
        +6k*(d+1)*lambda+6*(d+3)*(k-1).
```

The Routh-Hurwitz determinants are strictly positive for `xi>=0`, `k>1`.
The longitudinal A/Lambda block has polynomial
`lambda^2+k*lambda+2k+4/3`; each transverse block has
`lambda^2+k*lambda+2k`. They are stable for `k>0`. The two tensor curvature
roots are `-2`. In the full matrix, there are 12 negative roots and eight
semisimple zero roots for `k>1`. Checking Lambda alone would give an incorrect
condition, because its curvature coupling is essential.

The compiled extraction checks varying damping/restoring strength and
`a=.5,.75,1,2`. At runtime damping 5 and xi 1.5, the largest nonzero real pole
roots in unnormalized time are respectively
`-1.81818`, `-1.60161`, `-1.01583`, `-.332600`. At `S=a=1` the scalar roots are
`-10.2978`, `-4.6706`, `-1.01583 +/- 1.56878 i`.
The original conformal-Q lapse retains its positive root as a negative control.

These are frozen lower-order statements at the asymptotic reference, not a
variable-coefficient PDE estimate, discrete spectral estimate or nonlinear
regularity closure. In particular, the existing finite-Q counterexample to the
Theta/trace regularity conditions survives this lapse replacement.

## Preferred source remains unresolved for the new lapse

The adapter rejects the combination
`physical_trace_lapse=true, preferred_source=true`. Reusing the original source
would misidentify the new lapse's spacetime coordinate source. The exact
off-constraint ADM identity remains

```
F0 = Gamma4^0+2*Zbar^0 = -D0(alpha)/alpha^3-Q/alpha
Fi+beta^i*F0 = chi*Lambda^i+gtilde^{ij}*chi_j/2
              -chi*gtilde^{ij}*partial_j(log(alpha))-D0(beta^i)/alpha^2.
```

For the physical-trace lapse, its time source has pole numerator

```
F0_pole = -S_alpha/alpha^3-(P-3*omega_n)/alpha.
```

Imposing the full preferred spatial projection introduces
`S_beta^i=-alpha^2*V^i*(beta.dOmega)*F0_pole`, where `V` uses the Euclidean
gradient norm. This hypothetical correction is not enabled in production.
At `S=a=1`, its scalar-plus-radial-shift characteristic polynomial factors as

```
lambda*(lambda+d+1)*(lambda^3+2k*lambda^2-9lambda-12k).
```

It restores the original positive `2.57170949` pole at `k=5`, independently of
xi. An actual 20-field extraction and independent four-dimensional Christoffel
audit verify both the pole and

```
Box_bar(Omega) = Omega*W_hat
                +chi*(Lambda-Gamma_tilde).dOmega+2*omega_n*Theta_phys/Omega.
```

In the frozen scalar subsystem, the projection forces the pole derivative of
`da+db_radial` to be `3*(da+db_radial)-dP`, independently of the lapse pole
coefficients. Retuning the lapse alone cannot remove this unstable cubic while
retaining that exact projection and the current geometric equations.

The positive mode is normal to the necessary leading regularity conditions.
For `u=da+db_radial`, `C=dchi+2*u`, `T=dP-3*u`, the closed frozen normal block is

```
[ 0, -4/3,       4/3;
 -3,    1,       k-4;
 -3,   -2,    -1-2k].
```

Its polynomial is the unstable cubic above. The compatible leading tangent
`(u,dchi,dP,Theta)=(1,-2,3,0)` has zero pole. At `k=5`, the positive mode with
`u=1` has `C=1.37252`, `T=-2.57171`, `Theta=.075587` and leading physical
Hamiltonian variation 1.44700. This is a local frozen scalar check with spatial
perturbation derivatives held fixed, not a nonlinear constraint-manifold
theorem. Merely reweighting residues by Omega leaves the interior eigenvalue
and produces new weighted singular couplings; it does not close their evolution.

Turning off this projection is a documented experimental change, not a claim
that the new lapse satisfies preferred conformal gauge or preserves a live
null boundary. The observed null residual is still measured. A compatible
source/constraint closure is required before claiming a regular scri system.

## Optional symmetric spherical continuation

`hyperboloidal_symmetric_ghosts=true` selects a separate plan that canonicalizes
the target under Cartesian cube reflections/permutations and averages its
stabilizer. Every donor is strictly interior; there is no recursive ghost
dependency. The original plan and defaults remain available.

The option requires a centered, isotropic cubic grid and invariant spherical
mask. Both are checked. Reflection/permutation weight mismatch is exactly zero
for tested even grids 24/36/48 and odd grid 25, degrees 2–4. Polynomial errors
are at most `1.14e-14`. The existing fixed support capacity rejects symmetric
degree five at N48, where 238 donors would exceed 216.

Symmetry does not give an energy bound: cubic and quartic maximum weight L1
norms remain about 25 and 66. Actual shift-upwind plus interior KO transport
passes the targeted long runs; centered-advection negative controls still fail
with either plan. A conformal-wave dipole test improves RMS error moderately,
but the maximum boundary error remains comparable. The full Z4c closure and
transition errors require separate evidence.

## Constraint-consistent black-hole initial geometry

`LayerWormhole` constructs mass `0<M<2*r0` data, placing the Schwarzschild throat
strictly inside the exact Cauchy region. The supplied `LayerReference` remains
Minkowski. It does not define a black-hole gauge fixed point or subtract a
black-hole evolution RHS.

With isotropic physical radius `R=r/Omega`, define

```
m=M/(2R); psi=1+m; N=(1-m)/(1+m); v=b/alpha_hat.
```

Use the mass-corrected height derivative `h'_M=psi^2*v/N` only where the layer
boost is nonzero; set it exactly zero in the throat neighborhood. Then

```
gamma_bar_BH=psi^4*gamma_bar_hat
gtilde_BH=gtilde_hat; Lambda_BH=Lambda_hat; chi_BH=chi_hat/psi^4
alpha_initial=alpha_hat*((1-w)/psi^2+w*N)
beta_initial=(N/psi^2)*beta_hat.
```

The outer height has `R+2M*log(R)` asymptotics. Using the uncorrected Minkowski
height on Schwarzschild would instead leave a divergent radial compactified
metric. The lapse is positive away from the puncture, including the second
wormhole end; it is not the signed static Schwarzschild lapse in the interior.

Let `eta=r*Omega*w'/L`. The physical curvature eigenvalues are

```
kR=-(w+eta+2*w*m/(1-m^2))/(a*psi^2)
kT=-w*N/(a*psi^2); P=kR+2*kT; Theta=0.
(kR-kT)/Omega
 =-[r*w'/L+w*M*(2-m)/(r*(1-m^2))]/(a*psi^2).
```

The last identity supplies factored finite conformal shear without subtracting
nearly equal curvature values. All consumed Cartesian jets are analytic. The
exact Cauchy branch avoids the apparent static throat singularity. Near the
puncture, `r/(r+M*Omega/2)` is used instead of a large psi; no floor is added.
At the exact puncture `alpha=chi=0`, so the actual evolution grid must exclude
that coordinate point.

Independent symbolic ADM checks prove Hamiltonian, momentum and Schwarzschild
mass identities for an arbitrary smooth boost. Compiled tests verify the
throat, both layer endpoints, small Omega, changed S/a, the mass-zero limit,
Cartesian jets and independent stationary-height curvature. Maximum analytic
Hamiltonian and momentum residuals are `3.50e-14` and `1.22e-15`; outer geometric
stationary residual is `8.84e-12`. This verifies the initial geometry. It does
not establish gauge stationarity, trumpet formation or black-hole stability.

In the native adapter, an enabled layer with `problem/mass>0` selects these
data; the non-layer CMC-trumpet initializer remains available. Even cell counts
exclude the exact puncture. Analytic initial jets plus finite differences of
evolving deviations provide fixed-profile derivative reconstruction, while all
gauge sources and the sole geometric roundoff RHS subtraction remain Minkowski.
Native checks recover Schwarzschild mass from dumped physical ADM fields,
verify initial constraints, actual gauge/geometry changes and restart agreement.

## Evolution evidence and pending gates

The new physical lapse with a 10% lapse / 2% shift angular pulse reaches
`t=.2` on N24 for both layer and pure CMC controls, beyond the original layer's
failure at `.0271`. At `a=1`, the layer still has Hamiltonian norm about 2.12;
the pure CMC control is about .0064 with cubic ghosts. Hamiltonian error is
concentrated in the nonflat transition, while outer momentum and Z errors
depend strongly on continuation degree.

At `a=.5`, symmetric ghosts and `t=.5`, the original transition `.35–.75` gives
Hamiltonian norms 1.058 (degree three) and .983 (degree two). Broadening it to
`.2–.8` reduces them to .300 and .156. The broad quadratic run has momentum
.239, Z .0934 and maximum null deviation .0472. Fields remain positive but
constraints grow, so these are diagnostic runs, not accepted stable pulses.
Their source was a dirty research workspace; executable hashes and inputs are
retained. New run harness snapshots explicitly distinguish launch workspace
from proven executable build provenance.

The broad stationary N24 reference remains at Hamiltonian `6.48e-14`, momentum
`7.21e-14` and null deviation `5.96e-15` at `t=.5`. Refining the angular pulse
to N36 gives `(H,M,Z)=(.0314,.0252,.00503)` at `t=.2`, compared with N24 history
interpolation `(.0551,.0565,.0223)`. This is an improvement at two resolutions,
not yet an asymptotic convergence estimate. Reducing the actual step from
`.00057216` to `.00042773` changes the N24 `t=.5` Hamiltonian norm by less than
.05%. The nominal global `cfl_number` cancels in this adapter: its timestep is
the minimum of the explicit `.025*h/max_speed` and pole-CFL bounds. Receipts
report actual steps rather than interpreting the requested global CFL as a
step ratio.

An independent constraint-tangent audit differentiates the analytic continuum
RHS at reference geometry with finite angular lapse/shift jets. Its sampled
Hamiltonian derivative is below `3.6e-6`. Applying the actual native spatial
stencils to the same initial disturbance produces RMS first-step Hamiltonian
derivatives about `1.938,1.604,1.282` at N24/36/48 for `.35–.75`; the bulk peak
is about 19 and is unchanged by cubic versus quadratic ghosts. Broadening to
`.2–.8` reduces the RMS sequence to `.7971,.3702,.2163,.1098,.08118` at
N24/36/48/64/72. The final observed orders are 2.35 and 2.56, approaching but
not reaching the fourth-order asymptotic regime. These are inexpensive
instantaneous spatial audits, not evolutions at N64/72. They directly expose
a transition discretization defect before nonlinear growth. A matrix-free
linearized spectrum attempt did not converge eigenpairs; no spectrum conclusion
is inferred from that attempt.

Pending Minkowski gates include stationary controls, radial and nonradial
finite pulses, several crossing times, independent timestep and resolution
refinement, bulk/transition/boundary convergence, and compatible scri residues.
The later black-hole gate requires actual Cartesian wormhole-to-trumpet
evolution, resolved inner profiles and mass/constraint/boundary convergence
over multiple crossing times. Runtime scope remains CPU/Serial, one uniform
vacuum MeshBlock. MPI, AMR, matter, GPU evolution and binaries remain rejected
until independently implemented and validated.

Later [longer projected-discrete screens and grid-phase controls](hyperboloidal-long-window-and-resolution-audit.md)
retain substantial growth through t6. The [live damping candidate](hyperboloidal-live-damping-audit.md)
does not suppress it. The [linear scri hierarchy audit](hyperboloidal-scri-linear-hierarchy-audit.md)
identifies the missing next-jet compatibility condition; no exact-scri boundary
closure or wormhole-to-trumpet evolution is accepted by those experiments.
