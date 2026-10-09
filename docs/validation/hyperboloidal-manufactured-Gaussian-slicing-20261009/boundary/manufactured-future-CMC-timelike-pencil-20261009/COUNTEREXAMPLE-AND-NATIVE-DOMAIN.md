# Addendum: the unrestricted future-physical CMC claim is false

This additive pencil preserves DERIVATION.md unchanged. That file proves
exterior/core/high-time positivity and explicitly leaves its compact box
unresolved. The proposed certificate of that entire box cannot succeed:
the all-physical-T>=0 CMC sufficient condition is false. No numerical/CAS
evaluation, import, source query, inverse solve or evolution is used below.
The parent's finite-event result motivated this check; the displayed
counterexample is independently derived from the defining Gaussian.

## Exact counterexample, with a rational sign proof

For even f the defining G and F are odd in T. At T=0,

    F=0, grad_X F=0,
    G_T=[f1(-R)-f1(R)]/R=2sigma^2 exp[-R^2/(2sigma^2)].

For any radial function q, partial_1 partial_2 q=X1X2(q_RR/R^2-q_R/R^3).
Consequently, with p=n1*n2,

    F_T(0,X)=2R^2p/sigma^2 exp[-R^2/(2sigma^2)].

At R=2sigma and p=-1/2 this is exactly -4exp(-2). The CMC endpoint is

    D_CMC=(1-4epsilon exp(-2))^2
          -4sigma^2/(a^2+4sigma^2), a=1/2.                    (A1)

All relevant J are positive by the joint estimate in DERIVATION.md. There
is no ambiguity from squaring a negative time Jacobian.

A fully rational bound suffices to prove the indicated negative signs.
The exponential series gives e<8/3+5/96=87/32<11/4, by bounding the
n>=4 tail by (1/24)sum_{k>=0}(1/5)^k. Thus e^2<121/16<8 and exp(-2)>1/8.

For sigma=1/2, the reference h^2=4/5. For epsilon in{1/4,1/2,3/4},

    0<J=1-4epsilon exp(-2)<7/8,
    D_CMC<49/64-4/5<0.                                      (A2)

For sigma=7/20, h^2=49/74. For epsilon in{1/2,3/4},

    0<J<3/4, D_CMC<9/16-49/74<0.                            (A3)

These are five exact negative parameter cases in the proposed family. They
do not contradict any exterior/core/high-time inequality already proved:
R=2sigma lies in the previously unresolved finite-radius box and T=0 is
outside the high-time tail.

## Why this is not a future-native counterexample

At these events Y0=T+epsilon F=0, so native t=Y0-H(R)=-H(R).
For CMC height normalized at the origin,
H_CMC(R)=sqrt(R^2+a^2)-a>0 for R>0, hence their native time is strictly
negative. For the actual layer the same conclusion holds wherever its
normalized height is positive. If a wider exact Cauchy core instead gives
H=0 at such an event, its actual height gradient is also zero there and the
CMC endpoint calculation is not its actual D. No future-native failure
follows from (A1)-(A3).

The even seed and global J>0 imply native t>=0 gives physical T>=0.
The converse was never established and is false at these events. A
mathematical future-native domain must retain

    T+epsilon F(T,X)>=H(R),                                  (A4)

using the actual height H, not only T>=0. A finite inverse/event screen
remains diagnostic until this entire restricted domain is certified.

## Exact restricted angular domain at a fixed physical event

At fixed T,R>0, F=R^2p C(T,R). Therefore the future-native angular set is

    I(T,R)={p in[-1/2,1/2]: epsilon R^2 C(T,R)p>=H(R)-T}.       (A5)

It is a closed interval, the whole interval or the empty set. If the
coefficient epsilon R^2C is positive it imposes a lower endpoint; if
negative it imposes an upper endpoint. If it is zero, the set is the whole
interval exactly when T>=H(R), and is otherwise empty. No division by a
zero coefficient is permitted.

For every p in[-1/2,1/2], s_ang=n1^2+n2^2=1 is attainable. Condition (A5)
depends only on p and does not alter that attainability. Thus the exact
restricted angular minimum, with the c0,Llin,Q defined in DERIVATION.md,
is

    min_{future-native angles}D
      =c0+min_{p in I(T,R)}(Llin*p+Q*p^2).                    (A6)

If I is nonempty, evaluate its two endpoints and also the vertex
p=-Llin/(2Q) when Q>0 and the vertex is inside I. The symmetric |Llin|
shortcut in the earlier unrestricted sphere formula cannot be used on an
asymmetric interval. Empty intervals contain no future-native event and
contribute no positivity condition. A rigorous interval implementation
would need outward enclosures of H, C and (A5), retaining their dependence;
using an interval superset of admissible p is conservative but may be too
loose. Subdivision must preserve all unresolved/empty/boundary boxes.

Restricting the CMC endpoint to (A4) is only a possible sufficient strategy.
Since the actual H_layer can be smaller than H_CMC, it can still admit
events where D_CMC<=0 while D(h_layer)>0. No claim that this changed CMC
strategy succeeds is made. A direct certificate of actual D(h_layer) on
(A4), or a separately assessed change of the Cauchy-height extent, may be
necessary. Neither is executed or admitted by this note.

## Amplitude-domain caution

The earlier concavity in epsilon proves an amplitude interval only at a
fixed physical event and angle where its chosen endpoint is positive.
Here the admissible angular set (A5) itself depends on epsilon. It is
therefore not valid to certify only epsilon=3/4 on its own future-native
domain and silently infer all smaller amplitudes: those smaller amplitudes
can have different admissible events. A later uniform amplitude certificate
must either retain epsilon as an interval variable with (A4), or certify
positivity at endpoint amplitudes on a common domain containing the union
of the requested future-native domains. The exterior/core/high-time bounds
already cover the entire amplitude range, independently of (A4).

## Current mathematical status

The explicit bounds R^2D>3/80 for R>=32,T>=0, D>126/616225 for R<=1/50,
and D>1/32776 for R<=32,T>=32+8sigma are retained. The unrestricted
all-physical-T>=0 CMC sufficient condition is rigorously false. Future-native
timelikeness, the restricted finite-box certificate and actual-layer
admissibility remain unresolved. These results give no PDE/native/boundary
stability or coupled-inner-helper acceptance.
