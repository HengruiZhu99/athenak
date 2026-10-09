# Future CMC timelikeness: explicit Gaussian exterior, core and time-tail bounds

This is a fresh mathematical pencil, with no numerical/CAS evaluation,
scientific import, compiler, source query, matrix, inverse solve or evolution.
The fixed source context is recorded separately. No helper or prior receipt is
changed. The remaining compact physical-event box is **not certified here**.

Take a=1/2, 0<sigma<=1/2 (in particular sigma=7/20 or1/2),
f(T)=sigma^4 exp[-T^2/(2sigma^2)], and |epsilon|<=3/4. For X=R n define

    G=[f(T-R)-f(T+R)]/R, F=partial_1 partial_2 G,
    J=1+epsilon F_T, w=h n-epsilon grad_X F,
    D=J^2-|w|^2, h=h_CMC=R/sqrt(R^2+a^2).

All inequalities below concern the physical event (T,X), with T>=0 where
specified. They do not use one angular inverse event for an entire native
time slice. The layers in the fixed source have 0<=h_layer<=h_CMC; the already
established concavity D''(h)=-2 transfers positivity of the two endpoints to
that interval, once the missing CMC endpoint region is proved.

## Elementary Gaussian constants and the joint estimate

For x=T/sigma the first profiles are

    f0=sigma^4 exp(-x^2/2),
    f1=-sigma^3 x exp(-x^2/2),
    f2=sigma^2(x^2-1)exp(-x^2/2),
    f3=sigma(3x-x^3)exp(-x^2/2),
    f4=(x^4-6x^2+3)exp(-x^2/2).

Thus the global suprema admit the bounds

    M0<=1/16, M1<=1/8, M2<=1/4, M3<=2, M4=3.

Here M1<=sigma^3 follows from max|x|exp(-x^2/2)=exp(-1/2)<1.
The extrema for |(x^2-1)exp(-x^2/2)| are at x=0 and x^2=3;
2exp(-3/2)<1, so M2=sigma^2. For M3, separate the two monomials:
3sqrt(3)exp(-3/2)<2 follows from exp(3)>8>27/4, and
3exp(-1/2)<2 follows from e>5/2>9/4. Their sum is less than4,
giving M3<=4sigma<=2.

For completeness, f4 has stationary points x=0 or x^2=5+/-sqrt(10).
At y=x^2=5+/-sqrt(10), its polynomial is4(y-3). For the smaller root,
sqrt(10)<16/5 and y>9/5 give
4(sqrt(10)-2)exp(-y/2)<(24/5)/(19/10)=48/19<3.
For the larger root y>8 and4(y-3)<104/5, while
exp(y/2)>exp(4)>1+4+8+64/6=71/3, again giving a value below3.
The value at x=0 is3 and the tails vanish, proving M4=3.

The regular sphere representation gives, for each Euclidean unit m,

    F_T +/- m.grad F
      =-(1/(2pi)) integral_S2 nu1 nu2(1+/-m.nu)f4(T+nu.X)dOmega.

Since 1+/-m.nu>=0, integral|nu1 nu2|=8/3, and the weighted odd term
integrates to zero, |F_T+/-m.grad F|<=4M4/(3pi). Choose the sign and m
to obtain the joint lower bound

    J-|epsilon| |grad F| >=1-|epsilon|4M4/(3pi)>=1-3/pi>0.       (1)

In particular J>0 everywhere and the Cauchy-height endpoint h=0 is positive.
This joint estimate does not by itself prove positivity at h_CMC near infinity.

## Explicit exterior bound: R>=32, T>=0

Put z=1/R, u=T-R, v=T+R, p=n1*n2 and

    t_ang=n2 e1+n1 e2-2p n, tau=|t_ang|^2<=1, |p|<=1/2.

Use the exact source identities, with all advanced terms present:

    Phi=f2(u)-f2(v)+3z[f1(u)+f1(v)]+3z^2[f0(u)-f0(v)],
    Aplus=-f2(u)-6z f1(u)-9z^2 f0(u)
          -2f3(v)/z+7f2(v)-12z f1(v)+9z^2 f0(v),
    Aminus=2f3(u)+7z f2(u)+12z^2 f1(u)+9z^3 f0(u)
           -z f2(v)+6z^2 f1(v)-9z^3 f0(v),
    eta=a^2/[sqrt(1+a^2z^2)(sqrt(1+a^2z^2)+1)],
    kminus=eta+epsilon p Aplus,
    jplus=1+h+epsilon p z Aminus,
    R^2 D=Delta=kminus*jplus-epsilon^2 z^2 Phi^2 tau.             (2)

The future restriction is essential for a uniform advanced-tail bound:
v>=R>=32, so x_v=v/sigma>=64 and R<=sigma x_v. For j=0,1,2,3 and
m=0,1,

    R^m |f_j(v)| <=2^-2000.                                    (3)

To see this without a rounded Gaussian evaluation, the sigma factors are at
most1, and each resulting polynomial is bounded by x^4+3x^2 for x>=64.
Each monomial x^k exp(-x^2/2), k<=4, decreases on this range. At x=64,
x^4+3x^2<2^25 and exp(-2048)<2^-2048 because e>2. This gives a bound
below2^-2023, hence (3). In particular (3) controls the normalized term
R f3(v) in Aplus; it is not dropped as an underflowed seed value.

The sums of the absolute advanced contributions to Phi, Aplus and Aminus are
at most7,30,16 times2^-2000, respectively, all smaller than delta=2^-100.
Using z<=1/32 and the retarded global norms above therefore gives

    |Phi| <=1/4+3/256+3/16384+delta <17/64,
    |Aplus|<=1/4+3/128+9/16384+delta <9/32,
    |Aminus|<=4+7/128+3/2048+9/524288+delta <65/16.              (4)

Let x=a^2z^2<=1/4096 and s=sqrt(1+x)<=1+x/2. Then

    h>=1-x/2>=1-1/8192,
    eta=1/[4s(s+1)]
        >=1/[8+6x+x^2]>1/[8+1/512]>1/8-1/32768.              (5)

As |epsilon p|<=3/8, equations (4)-(5) imply the positive factor bounds

    kminus>1/8-1/32768-(3/8)(9/32)=639/32768,
    jplus>2-1/8192-(3/8)(1/32)(65/16)>31/16.

Insert these into (2), retaining the angular subtraction:

    R^2D > (639/32768)(31/16)-(9/16)(1/1024)(17/64)^2
          =2532951/67108864 >3/80.                              (6)

This is a finite-radius uniform lower bound, not only a leading scri limit.
It holds for every R>=32,T>=0 and angle, for both declared sigma values and
indeed all0<sigma<=1/2, and both signs of epsilon within the displayed bound.

## Explicit core bound: 0<=R<=1/50

Here h<=R/a=2R<=1/25. A rational lower bound pi>157/50 suffices.
For example the exact Machin identity and alternating arctangent bounds give

    pi=16atan(1/5)-4atan(1/239)
       >16(1/5-1/375)-4/239=281476/89625>157/50.

Thus 1-3/pi>7/157. From (1) and the triangle inequality,

    J-|w|>=J-|epsilon| |grad F|-h
          >7/157-1/25=18/3925>0,
    J+|w|>=J>7/157,
    D>126/616225.                                              (7)

This bound does not need T>=0 and includes both epsilon signs. At R=0 the
regular Cartesian representation additionally gives the exact value D=1.

## Explicit high-time bound at finite radius

Take 0<=R<=32 and T>=32+8sigma. Every argument T+nu.X in the joint sphere
formula is at least8sigma. For x>=8,

    |f4(sigma x)|<=(x^4+6x^2+3)exp(-x^2/2)
                 <=4483 exp(-32)<4483*2^-32<2^-19.             (8)

Each monomial in this upper bound decreases for x>=8. Applying the same
joint estimate to this restricted argument range yields

    J-|epsilon| |grad F|>1-2^-19/pi>1-2^-20.                   (9)

At R=32, with x=1/4096 and s=sqrt(1+x),
s(s+1)<=2+(3/2)x+x^2/4<2+1/2048. Consequently

    1-h(R)>=1-h(32)=1/[4096 s(s+1)]>1/8194.

Combining this with (9),

    J-|w|>1/8194-2^-20>1/16388,
    J+|w|>=J>1/2,
    D>1/32776.                                                (10)

This also holds for both epsilon signs. It is a physical high-time estimate;
no native-time inverse or finite sampled event is used to prove it.

## Remaining compact box and a rigorous interval strategy

The only unresolved CMC endpoint region is

    1/50<=R<=32, 0<=T<=32+8sigma, sigma in{7/20,1/2}.            (11)

For the requested nonnegative amplitude range0<=epsilon<=3/4 it is enough
to certify epsilon=3/4 there. Indeed, at fixed physical event and angle,

    g(epsilon)=1+epsilon F_T-|h n-epsilon grad F|

is concave in epsilon, g(0)=1-h>0, and J>0 by (1). Positivity of g(3/4)
therefore implies positivity for the entire amplitude interval; D>0 and g>0
are equivalent when J>0. This argument does not assert D is monotone in
epsilon. A future claim including negative amplitudes would also need the
epsilon=-3/4 endpoint in (11); only the exterior/core/high-time bounds above
have already covered that sign.

There is an exact angular reduction with no angular sampling. Write
F=R^2p C(T,R), s_ang=n1^2+n2^2. With

    d0=a^2/(R^2+a^2),
    A=C_T+h(C_R+2C/R), B=C_T^2-C_R^2-4CC_R/R,
    Llin=2epsilon R^2 A, Q=epsilon^2 R^4 B,
    c0=d0-epsilon^2 R^2 C^2,

the full sphere minimum is exactly

    min_n D=c0+min_{|p|<=1/2}(Llin*p+Q*p^2).                  (12)

For each p in this interval s_ang=1 is attainable, and its coefficient in D
is nonpositive, which proves (12). Set ell=|Llin| and define

    gquad(Q,ell)=Q/4-ell/2                     if Q<=0,
                 -ell^2/(4Q)                 if Q>0,ell<=Q,
                 Q/4-ell/2                   if Q>0,ell>Q.

This function is nondecreasing in Q and nonincreasing in ell: it is the
minimum of Q*q^2-ell*q over0<=q<=1/2. Hence an interval box with
Q>=Qlo and |Llin|<=ellmax has the rigorous lower bound

    min_n D>=c0_lo+gquad(Qlo,ellmax).                          (13)

The Qlo<=0 branch avoids division by an interval containing zero. Square and
sign bounds for the Gaussian and its derivatives must be outward enclosures,
not rounded finite-event samples. A possible future certificate uses rational
R/T box endpoints, rationally enclosed square roots and exponentials, and
adaptive bisection until (13)>0 on every covered box. Exponentials can be
range-reduced and enclosed with Taylor remainders using rational arithmetic.

For small R, direct advanced/retarded differences can widen intervals severely.
The regular identity

    C=-(1/8) integral_-1^1(1-q^2)^2 f5(T+Rq)dq,
    partial_T^m partial_R^n C
      =-(1/8) integral_-1^1(1-q^2)^2 q^n f^(5+m+n)(T+Rq)dq

provides a cancellation-free interval/Taylor alternative with an explicit
remainder. The certificate must enclose C,C_T,C_R and all quadrature/Taylor
errors. Its final report would need exhaustive box coverage, exact rational
lower bounds and unresolved-box/failure records. No such calculation has
been executed here, and termination or positivity on (11) is not asserted.

## Scope

Equations (6), (7), and (10) rigorously remove the far exterior, the core and
the finite-radius high-time tail from the future CMC endpoint question.
They do not prove the remaining compact box, globally admit the manufactured
family, establish binary64 arithmetic robustness, or prove PDE/native/boundary
stability. The even-seed time-sign and layered-height concavity reductions
remain conditional on finishing that CMC endpoint certificate. A finite
physical-event screen cannot replace the missing certificate.
