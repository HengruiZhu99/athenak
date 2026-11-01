# Inner telegrapher gauge, N16 toward 10M

User authorization: match the puncture gauge to the tested telegrapher_lapse
branch while preserving the outer hyperboloidal gauge; evolve fresh N16 to10M.
The later10M request replaces24M. No N24 run is part of this experiment.
Reference: project/telegrapher_lapse commit
b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a, z4c_calcrhs.cpp and unboosted
experiments/telegrapher_aurora_20260929/inputs/g1_bhahaha_L5.athinput.

## Equations and scope

IT1 is an opt-in target, hyperbolic_shift_3d_mpi_inner_telegrapher, derived from
GD1. Old targets retain their gauge. Direct chi and lapse RK/KO and existing
floors remain: physical lapse1e-10, positive conformal chi1e-8 at r<.03,
and finite nonpositive chi rescue everywhere. Count signed candidates before
clipping. Nonfinite arithmetic and indefinite metrics remain failures.

Let D0=partial_t-beta.grad, a=alpha_bar/Omega, c=Omega^2 chi_bar,
p=K_phys-2Theta, and g the determinant-one spatial metric. Omega=(1-r^2)/2
is stationary. On r<=.2 the gauge is exactly:

    D0 a = -2 a p + max(c,1e-5) g^{ij} partial_i B_j
    D0 B_i = (kappa partial_i a - B_i)/tau
    D0 beta^i = Lambda^i - eta_shift beta^i

The branch uses a partial derivative of B, not its covariant divergence. B is
interpreted as a physical covector in the inner plateau, in the same global
Cartesian component slots as donor B outside. Original initial B and D are
zero, so no initial conversion is necessary. For M=1/3 in our coordinates,
tau=.1M=1/30, kappa=.2M=1/15, eta_shift=2/M=6. No slow-start term is added.
The branch's chi_guard is a coefficient guard, distinct from our state floor;
we apply it to the telegrapher contraction, not to the geometric equations.

In stored variables the inner lapse RHS is

    partial_t alpha_bar = beta.grad(alpha_bar)
       - a beta.grad(Omega) - 2 alpha_bar p
       + Omega max(c,1e-5) g^{ij} partial_i B_j.

The physical lapse gradient in B's source is
partial_i a=(partial_i alpha_bar-a partial_i Omega)/Omega. Omitting either
Omega-gradient term would fail even inside the Cartesian core.

Let s=Smoothstep9((.56-r)/.36). Blend complete local lapse, B and shift RHSs
as s*inner+(1-s)*donor. s=1 at r<=.2 and0 at r>=.56, with four vanishing
endpoint derivatives. This does not differentiate s or add a derivative-of-s
source. It finishes before the existing donor weight transition .56--.8;
that weight, harmonic collar .8--.9 and scri closure remain unchanged.
The old D auxiliary continues its equation but is decoupled from beta on the
inner plateau. This preserves its outer role without pretending D is Lambda.

Constraint-damping coefficients are also interpolated with s. Inside,
kappa1=.01/M=.03 and kappa2=0, giving the physical coefficients
+alpha_phys*kappa1*Theta in p, -2 alpha_phys*kappa1*Theta in Theta and
-2 alpha_phys*kappa1*(Lambda-Gamma_metric) in Lambda. Outside, donor damping
is unchanged. Other off-constraint geometric terms are NOT replaced by the
production Cauchy Z4c implementation. FD4, KO, mesh, interpolation and projection
are unchanged. This is a gauge/damping experiment, not an assertion of complete
Cauchy-kernel equivalence.

The speed estimate now multiplies the blended lapse/B principal coefficients
rather than interpolating their old squared speeds. It is a frozen-coefficient
estimate, not a proof of strong hyperbolicity. Outer pointwise RHS identity
does not imply identical outer trajectories: gauge waves can propagate outward.
Minkowski geometry with the old stationary CMC lapse/shift is not a stationary
solution of the new inner gauge; reference-subtracting its RHS would change
the intended physical Cauchy gauge and is deliberately not done.

## Execution and restart

Fresh N16 original generator input SHA256
35c1b65aeaaefac34552f9a63b279cabe839b09326adc9af1d2c767ecc29c8ca.
Keep dt=.0005, dt/dx=.0444444, target coordinate10/3=10M, horizon lmax12/
ntheta24/tolerance.001, diagnostics interval.05, angular epsilon2 and scri
penalty scale1. Use CompactBinaryMerger/debug, one node,12 ranks/GPU tiles,
SYCL/MPI, make -j64. Keep the home campaign below5,000,000,000 bytes including
output and atomic checkpoint reservations. Preserve unrelated jobs.

IT1 has its own HYP3D_MPI_IT1 checkpoint magic,168-byte header/scheme3;
GD1/GW1/AC1 checkpoints must be rejected. Source/binary bindings freeze the
blend, mass-scaled constants and equations. inner_telegrapher_campaign.py is
restricted to fresh N16 first, then verified healthy IT1 continuations toward
10M. Never resume a numerical failure or change timestep/floors automatically.
Copy and SHA256/length verify evidence before purging large remote artifacts.
The report identifies inner_gauge=telegrapher_b99e27f_M1over3_r020_056_v1.

## Verification

Native tests compare the transformed RHS with an independent physical-variable
oracle on anisotropic, off-diagonal metrics, with nonradial gradients and chi
above/below the coefficient guard. They check inner damping, endpoint plateaus,
monotonic blending, outer/scri RHS identity, principal-speed limits and CPU/device
partition parity. Campaign tests reject wrong schemas/configurations, unadmitted
checkpoints and changed targets. Short native evolution and restart checks are
setup tests; the10M run is the first duration test of this formulation.

Local fresh N16 completed two steps to coordinate.001; its IT1 restart completed
one more step to.0015. Maximum null residual across these checks was1.12e-15.
A native attempt to load GD1 was rejected with checkpoint schema/configuration/
admission mismatch. The physical oracle and CPU/device partition tests passed;
the existing tensor-scri test also passed. See checks/.
