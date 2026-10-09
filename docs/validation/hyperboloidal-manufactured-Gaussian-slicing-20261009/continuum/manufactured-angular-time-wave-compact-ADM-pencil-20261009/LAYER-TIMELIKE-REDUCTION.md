# Additive Gaussian time-domain and layered-height reduction

This is a pencil-only cross-check of the parent's additional analytic reduction.
It changes neither the frozen general-f note nor the Gaussian seed plan.
No numerical/CAS call, implementation or timelike-domain proof is performed.

For the proposed even Gaussian seed, G(T,R) and F(T,X) are odd in T, so F(0,X)=0.
The already established global J>0 makes T+epsilon F strictly increasing at
fixed X and maps T=0 to Y0=0. Therefore sign(T)=sign(Y0). The fixed reference
height has H(0)=0 and H_R>=0, hence H>=0. Native t>=0 consequently implies
Y0=t+H>=0 and T>=0. This reduction is specific to the even seed and the
declared height normalization; it is not a generic-f or arbitrary-time claim.

For any fixed Euclidean unit m, the exact sphere formula gives

    F_T +/- m.grad F
      =-(1/(2pi)) integral_S2 n1 n2 (1 +/- m.n) f4(T+n.X) dOmega.

The factors 1 +/- m.n are nonnegative and |n1 n2| is antipodally even, so
the odd integral of |n1 n2| m.n vanishes. Since integral|n1 n2|=8/3,

    |F_T +/- m.grad F| <= 4 M4/(3pi).

For either sign of epsilon choose m parallel to grad F (the zero-gradient
case is immediate). This gives

    J-|epsilon| |grad F| >= 1-|epsilon|4M4/(3pi).

Thus the Cauchy-height endpoint h=0 has D>0 globally whenever
|epsilon|4M4/(3pi)<1. For the proposed Gaussian M4=3 and epsilon<=3/4,
the lower bound is at least1-3/pi>0. This joint bound is sharper than adding
separate FT and gradient bounds; it certifies the endpoint h=0 only.

At fixed physical (T,R,n), write

    D(h)=J^2-h^2+2epsilon h F_R-epsilon^2|grad F|^2.

It is a concave quadratic in h, with D''(h)=-2. The actual reference-height
boost with cutoff0<=w<=1 is

    h_layer=wR/sqrt(a^2+w^2 R^2),
    0<=h_layer<=h_CMC=R/sqrt(a^2+R^2).

For R>0 and theta=h_layer/h_CMC in[0,1],

    D(h_layer)>=(1-theta)D(0)+theta D(h_CMC).

At R=0 both endpoints coincide and the regular Cartesian value applies.
Therefore strict positivity at the already certified h=0 endpoint and at
h_CMC for every physical R,T>=0 and angle would establish positivity for
every such layered cutoff. The cutoff need not have constant derivatives:
native-slice timelikeness uses its actual height gradient, not its higher jets.
Higher jets remain necessary for the independent consumed-data/source checks.

The exact fixed-physical-event angular quadratic minimum from the pinned plan
can reduce the remaining CMC endpoint question to the two-dimensional (R,T)
domain with T>=0. If instead only a bounded T interval is proved, it must
independently enclose all native t in[0,6] under consideration. The implicit
inverse still depends on angle at fixed native t, so a single angle's T is
never used for the entire sphere. No CMC endpoint minimum, finite-annulus
bound or weighted outer remainder has been evaluated or established here.

For clarity, this reduction would provide a sufficient admissibility proof
for the manufactured flat family if its missing CMC endpoint bound passed.
It would not prove native discretization stability, original-pulse coverage,
the coupled-inner gauge equation or a BH wormhole-to-trumpet transition.
