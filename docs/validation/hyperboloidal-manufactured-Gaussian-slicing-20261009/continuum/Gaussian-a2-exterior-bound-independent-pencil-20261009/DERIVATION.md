This source-only pencil verifies the parent's practical exterior estimate for
the Gaussian manufactured time-map. No numerical, CAS, kernel, compiler,
scientific import, inverse solve or evolution was performed. Units are S=1,
a=2, 0<sigma<=1/2, |epsilon|<=3/4, T>=0. The seed is
f(t)=sigma^4 exp(-t^2/(2 sigma^2)),
F=partial_X partial_Y[(f(T-R)-f(T+R))/R],
Y0=T+epsilon F, Yi=Xi. The prescribed reference remains Minkowski.

Write z=1/R, p=n_x n_y, tau=|grad_S p|^2=n_x^2+n_y^2-4p^2.
Then |p|<=1/2 and 0<=tau<=1. For a radial reference height slope h=H_R,
J=1+epsilon F_T, w_r=h-epsilon F_R, and
D=J^2-w_r^2-epsilon^2 z^4 Phi^2 tau. The compact profiles in the pinned
Gaussian source satisfy exactly

  F=p z Phi,
  F_T+F_R=p z^2 Aplus,
  F_T-F_R=p z Aminus.

For the pure CMC endpoint, s=sqrt(1+a^2 z^2), h=1/s, and
eta=(1-h)/z^2=a^2/[s(s+1)]. Thus

  R^2 D=Delta,
  Delta=(eta+epsilon p Aplus)(1+h+epsilon p z Aminus)
         -epsilon^2 z^2 Phi^2 tau.

The symbol h here denotes the physical height slope, not the conformal lapse.

Advanced tail for R>=4

Put v=T+R and x=v/sigma. Then v>=R and x>=8. For j=0,1,2,3,
f_j(v)=sigma^(4-j) times the corresponding Hermite polynomial times
exp(-x^2/2). For m=0 or1, R^m<=v^m gives

  R^m |f_j(v)| <= (x^4+3x^2) exp(-x^2/2).

Every monomial times this Gaussian decreases for x>=8. At8 the polynomial is
4288<8192=2^13, and exp(-32)<2^-32, so all eight products are <b=2^-19.
No evaluation of the exponential or of a sample is needed for this bound.
The advanced pieces therefore satisfy

  |Phi_advanced| <= (31/16)b,
  |Aplus_advanced| <= (201/16)b,
  |Aminus_advanced| <= (49/64)b.

In particular each is <delta=2^-14=32b. The 2 f_3(v)/z term in Aplus
uses the m=1 bound; this factor has not been dropped.

Retarded global norms

The Gaussian derivative norms used by the parent are valid:
M0<=1/16, M1<=1/8, M2<=1/4, M3<=2. For example the standard Gaussian
Fourier integral bounds the unit-Gaussian first/second derivatives by its
absolute first/second moments, <=1 and1; its absolute third moment is <=sqrt3
by Cauchy-Schwarz between moments2 and4. Restoring sigma^(4-j) gives an even
stronger third-derivative bound, so the stated2 is conservative.
For z<=1/4 these imply

  |Phi| <= 1/4+3/32+3/256+delta <3/8,
  |Aplus| <= 1/4+3/16+9/256+delta <1/2,
  |Aminus| <= 4+7/16+3/32+9/1024+delta <37/8.

CMC endpoint positivity

For a=2, s<=sqrt(5/4)<9/8, so h>8/9 and
eta>256/153>5/3. Consequently

  eta+epsilon p Aplus >5/3-(3/8)(1/2)=71/48,
  1+h+epsilon p z Aminus >1+8/9-(3/8)(1/4)(37/8)>23/16.

The subtracted angular term is at most81/16384. Hence, for every angle,
retarded argument and future T>=0 at R>=4,

  R^2 D_CMC=Delta >1633/768-81/16384
                    =2+5965/49152 >2.

This verifies the parent's constants and signs. It is an analytic endpoint
bound, not a sampled consistency or a floating interval result.

Transfer to the actual layer

The wide layer in the pinned independent reference has b=w*r/a,
alpha_hat=sqrt(Omega^2+b^2), R=r/Omega, and0<=w<=1. Therefore its physical
height slope is

  h_layer=b/alpha_hat=w*R/sqrt(a^2+w^2 R^2),

which lies in[0,h_CMC]. The layer is not asserted to be pure CMC at R=4.
For fixed event/source derivatives D(h)=J^2-|h n-epsilon grad F|^2 is
concave in h. Positivity at h=0 and h=h_CMC therefore gives positivity
throughout that interval. It does NOT automatically transfer the quantitative
lower bound R^2 D>2 from the CMC endpoint to every layer height.

For completeness, global Cauchy-endpoint positivity follows from the exact
sphere identity

  F=-(1/(2pi)) integral n_x n_y f'''(T+n.X) dOmega.

For any unit e, the weights1+/-e.n are nonnegative, and their odd contribution
integrates to zero against |n_x n_y|. Since integral |n_x n_y| dOmega=8/3,

  |F_T|+|grad F| <=4 M4/(3pi).

The Gaussian Fourier fourth moment is3, giving M4<=3 uniformly in sigma.
Thus J-|epsilon grad F|>=1-3/pi>0 for |epsilon|<=3/4, and D(0)>0.
This proves the Cauchy endpoint required by the layer concavity transfer.
It does not by itself establish positivity at arbitrary nonzero height slope.

High-T tail on R<=4

If T>=4+8sigma and R<=4, every sphere argument obeys
(T+n.X)/sigma>=8. Its fourth derivative satisfies

  |f''''(T+n.X)| <=(x^4+6x^2+3) exp(-x^2/2)
                  <=4483 exp(-32)<2^-19.

The same directional sphere estimate yields
|F_T|+|grad F|<2^-19. At a=2 and R<=4,
h_layer<=h_CMC<=2/sqrt5<9/10. The triangle inequality then gives

  J-|h_layer n-epsilon grad F| >1/10-(3/4)2^-19 >0.

Therefore the high-T tail is positive directly, with no small-R radial
cancellation or denominator. Since4+8sigma<=8, T>=8 is a common sufficient
tail for the whole stated sigma interval. Together with the R>=4 exterior
and a separately justified small-R/origin bound, the remaining compact
positivity problem may be reduced to .02<=R<=4 and0<=T<=8. This note does
not certify that remaining box or silently assume its small-R premise.

Limits

These statements concern the exact manufactured Gaussian map and reference
height endpoints. They do not establish actual native-pulse regularity,
embedding jets, a full22 gauge/RHS gate, BH data, moving-puncture compatibility,
full nonlinear evolution or Z4 stability. J positivity is distinct from D
positivity, and target-native-time inversion remains distinct from physical T.
