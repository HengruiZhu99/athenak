# A conservative height feasibility control for the manufactured Gaussian family

This is an independent analytical feasibility pencil, with no numerical/CAS
evaluation, scientific import, compiler, source query, inverse solve, operator,
eigensolve or evolution. The old a=1/2 note, its counterexample addendum and
all helper/source failures remain unchanged. No new height is implemented.

Let

    f(T)=sigma^4 exp[-T^2/(2 sigma^2)], 0<sigma<=1/2,
    G(T,X)=[f(T-R)-f(T+R)]/R, R=|X|,
    F=partial_X1 partial_X2 G, |epsilon|<=3/4,
    J=1+epsilon F_T, D(h)=J^2-|h n-epsilon grad_X F|^2.

The regular Cartesian/sphere representation defines the value at R=0.
Here h is the physical radial height slope, n=X/R away from the origin,
and the actual layer height has H(0)=0 and 0<=H'(R)<=h_CMC(R;a).
This note changes the hypothetical reference parameter to a>=1024 at S=1;
it does not reclassify the fixed a=1/2 native family or repair its failures.

## The two uniform ingredients

The joint sphere-integral estimate is worth stating explicitly. For every
Euclidean unit m,

    F_T +/- m.grad F
      =-(1/(2 pi)) integral_S2 nu1 nu2(1+/-m.nu) f4(T+nu.X) dOmega.

The nonnegative factor 1+/-m.nu, the identity
integral_S2 |nu1 nu2| dOmega=8/3 and oddness of the weighted linear term
give |F_T +/- m.grad F|<=4 sup|f4|/(3 pi). For this Gaussian
sup|f4|=3. Thus, at every T and X,

    J-|epsilon| |grad F| >=1-3/pi >7/157=:g.                 (1)

The last rational inequality follows from pi>157/50. In particular J>g>0.
At the Cauchy-height endpoint,

    D(0)=(J-|epsilon| |grad F|)(J+|epsilon| |grad F|)>g^2.   (2)

The separately preserved regional note proves the uniform exterior bound

    R^2 D(h_CMC(R;1/2))>3/80, R>=32, T>=0.                 (3)

That proof explicitly retains all advanced v=T+R terms, including
R f3(v), and bounds them using v>=R. Its envelope bounds cover all
0<sigma<=1/2 and both signs |epsilon|<=3/4. It is a finite-R estimate,
not merely a leading scri limit. The exact source/note pins are recorded
in source-context.json; this note does not substitute finite sampling for (3).

## Finite physical radius: a>=1024 gives a direct positive gap

For 0<=R<=32 and a>=1024,

    0<=h_layer<=h_CMC(R;a)=R/sqrt(R^2+a^2)<=R/a<=1/32.

The triangle inequality and (1) give

    J-|h_layer n-epsilon grad F|>7/157-1/32=67/5024,
    J+|h_layer n-epsilon grad F|>=J>7/157.

Consequently

    D(h_layer)>469/788768>0, 0<=R<=32.                     (4)

This part holds for every real physical T. It requires no angular inverse
solve, radial sampling, high-time truncation or amplitude endpoint argument.
At the origin use the regular representation and h_layer(0)=0.

## Exterior: concavity transfers the a=1/2 endpoint certificate

At each fixed physical event and angle,

    D(h)=J^2-epsilon^2|grad F|^2
         +2 h epsilon n.grad F-h^2,  D''(h)=-2.

Therefore D is concave as a function of the slope and is bounded below on
an interval by the smaller of its endpoint values. Since a>=1024>1/2,

    0<=h_layer<=h_CMC(R;a)<=h_CMC(R;1/2).

Equations (2)-(3) yield

    D(h_layer)>min(g^2,3/(80 R^2))=3/(80 R^2),
    R>=32, T>=0.                                         (5)

For the equality of the displayed minimum, it is enough that
g^2=49/24649>3/81920>=3/(80R^2). No assumption that D is monotone in h
is used. The actual layer cutoff/compactification changes with a, but its
physical slope stays in the certified interval; the argument covers every
physical R and is independent of where a compactified cutoff reaches it.

## Why native future time stays in the proved physical domain

The Gaussian seed is even, so G and F are odd in physical T and F(0,X)=0.
For fixed X the map Y0(T,X)=T+epsilon F(T,X) has derivative J>0. It is
strictly increasing, with Y0(0,X)=0. The layer coordinates satisfy

    t_native=Y0(T,X)-H(R), H(0)=0, H'(R)>=0.

Thus t_native>=0 implies Y0=t_native+H(R)>=0 and hence T>=0. This is
an eventwise sign implication for the unique time inverse, not the use of
one angle's inverse time for an entire slice. Combining (4)-(5) proves
timelikeness of every such future-native event for the hypothetical a>=1024
height, indeed throughout the larger domain T>=0.

The earlier T=0,R=2sigma negative CMC examples at a=1/2 do not contradict
this result: they use a larger slope. The newly observed a=1/2 future-native
failure is also not removed or relabeled by a proof about a different height.

## Feasibility and limits

The production layer admission a>=S/2 permits a>=1024 for S=1. The exact
outer CMC outgoing speed at scri is 2S/a, hence 1/512 at a=1024. This is
only the scri value; the core has outgoing speed 1 and no crossing-time or
uniform speed claim is made. Such a conservative height may be inefficient.

The proof supplies an existence/feasibility control for the manufactured
Gaussian family, all 0<sigma<=1/2 and |epsilon|<=3/4. It does not choose a
practical height, implement a new candidate, admit the fixed a=1/2 family,
validate binary64 arithmetic or preserved scri falloff, solve the coupled
inner-helper problem, provide BH initial data, or establish continuum/native
gauge, constraint, discretization or long-time stability.
