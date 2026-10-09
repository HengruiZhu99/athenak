# Exploratory spatial-norm feedback: independent kernel and native gates

No tracked runtime file or default is changed. The private native configuration
is S=1, a=.5, rho=1.5, physical-P lapse, preferred source off, xi=1/a=2,
eta=rho*S/a^2=6, C=(S/a)*(1-1/rho)=2/3. The only additional shift numerator is

```text
S_beta^i = -eta*W*(beta^i-beta_ref^i + C*n^i*(G-Ghat)/Ghat)
G = chi*gtilde_inverse^ij*Omega_i*Omega_j,
Ghat = the same spatial norm evaluated on the analytic reference,
n = outward Euclidean radial unit vector.
```

The implementation factors G-Ghat before division by the strictly positive
spatial reference norm. It never divides by the vanishing four-dimensional
null norm. W remains the production .45–.85 gauge cutoff; the reference is
the production .05–.95 layer. Both geometric equations and derivative/boundary
operators remain the compiled production implementation 27c19d20. No Omega
floor, prescribed Theta falloff, preferred-Box projection or weighted storage
is added.

All eight actual full20 scaled-rho pole matrices at a=.5,.75,1,2 and kappa=5,10
have 15 negative nonzero roots and five semisimple zeros. Their maximum error
against the independent exact matrix is 2.24e-9. In units k=kappa*a^2,
rho=eta*a^2/S at S=1, the closed scalar/radial-beta quintic is

```text
lambda^5 + (2*k+rho+6)*lambda^4
 + (2*k*rho+12*k+4*rho+11)*lambda^3
 + (8*k*rho+28*k-5*rho+14)*lambda^2
 + (-4*k*rho+64*k-12*rho-24)*lambda
 - 12*(k-1)*(rho-4).
```

For k>1 and 1<=rho<4, exact Hurwitz determinants Delta1..3 and the constant
are positive; Delta4>0 is the remaining condition. The uniform sufficient
interval is 1<=rho<=5/2 for every k>1. At rho>5/2 the fourth determinant fails
as k approaches 1; rho>4 gives a positive real root. Keeping eta=6 at every a
would violate this condition at a=1,2. The native experiment uses scaled rho,
and this parameter dependence is preserved explicitly in report.json.

The actual wide a=.5,kappa10 continuum Fourier matrices have no positive
sampled outer roots at r=.85,.9,.95,.98, k=0,1,2,4,8,16,32,64,128,256, radial
and oblique. The r=.75,k=0 frozen geometric root is still +.491960. At kappa5
the outer derivative-coupled root still grows, +6.31754 at r=.98,k256. These
are local, constant-coefficient generators, not global PDE eigenvalues.
The epsilon convergence error is 4.49e-8; exact value-only operator-update
error is 7.02e-9. An independent full principal extraction passes all 360
radial/oblique/SPD cases with error 3.55e-15. The actual native include wrapper
matches the audited full20/Fourier candidate bit-for-bit.

At a fixed live state, relative to the old xi=1.5 source-off gauge, write the
added pole numerators as f_alpha and f_beta. The independent off-constraint
four-dimensional Christoffel gate verifies

```text
Delta F^0 = -f_alpha/(alpha^3*Omega),
Delta F^i = -f_beta^i/(alpha^2*Omega) - beta^i*Delta F^0,
Delta Box(Omega) = (Omega_i*f_beta^i + omega_n*f_alpha)/(alpha^2*Omega).
```

The Box change is generally nonzero. Reference/source/Box errors are
4.44e-16/4.26e-14/6.22e-15 on nonflat, oblique, off-constraint SPD states.
The finite-Q counterexample at a=1 is retained: P-Pref=Omega*.01 with its
consistent derivative jets gives Omega*Qdot -> .02 and ThetaDot -> -.02.
This source does not establish a nonlinear regularity manifold.

The separate independent BH initial-data gate is preserved at
build-layer-research/detached-wormhole/spatialnorm-gate/receipt.json. For the
original mass-corrected geometric first jets the leading lapse, shift, G and
null time rates vanish. The original complete a=.5,S1,M=.5 jets still give
NrawDot=-29*Omega+...; adding (29/40)*Omega^2 to initial radial beta cancels
that first coefficient without changing the mass/ADM data. The corrected
rate is -92.7*Omega^2+..., and the nonzero LambdaDot=116/15 agrees with the
contracted metric-connection derivative. The original BH receipt contained a failed symbolic structural-factor
assertion despite both compiled kernel checks passing. That failure was
detected before the t2 launch. Its complete receipt, failing source and
stdout/stderr are preserved under BH-gate-complete-first-failure-20261009/.
The corrected receipt passes all five Release/ASan/UBSan/leading/second/oracle
checks, and every recorded source hash is verified. The passing receipt is
preserved as BH-gate-receipt-pass-20261009.json. This is initial compatibility
only; no BH integration or closure claim follows.

The native build/launch receipts identify the actual compiled source separately
from documentation HEAD f615acf4. The private executable is
build-layer-spatial-norm-native/src/athena, SHA256
dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d.
Reference t=.05 passes the actual-array gate: three regular snapshots, maximum
array drift 1.14e-13, H/M/Z=8.48e-14/2.69e-14/1.25e-15. The short and long
finite-pulse outcomes are recorded separately in native-experiment-report.json;
local gates do not predict acceptance of those runs.

The actual short t=.02 pulse passes the native array admission checks with
H/M/Z=.00311063/.00489405/.00119800. The actual N24 long pulse completes
t=2 with all81 saved snapshots finite, positive lapse/chi and SPD physical
metric. Final H/M/Z=1.222186892/1.348372697/.375766082; matching production
ratios=1.018963/.683838/.847257. Thus H is1.9% worse while M/Z reduce, and
all constraints still grow: the candidate is rejected as pulse stabilization.
Theta=.0218962, final alpha/chi minima=.772240/.611262, and physical metric
eigenvalues=.650948..3.88644. Final H/M/Z maxima occur at r=.929108, while
H squared budget is88.61% inside .9 and M/Z squared budgets are69.93%/91.41%
outside .9. Exact matched histories and additional radial budgets are in
native-summary.json and constraint-budgets.json.

The unchanged live characteristic bound remains active. Candidate and control
share the configured pole-CFL .03 and initial dt=.000427734375, but their
final cycles are4676 and4679; individual later steps need not match. The
comparison uses matched physical times. Root owns a separate N36 t=.2
spatial/timestep control using the same frozen executable and distinct output
directory; no conclusion from that control is included here before its actual
array audit completes. No native BH integration is authorized by this outcome.
