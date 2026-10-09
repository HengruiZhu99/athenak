from pathlib import Path
import json, hashlib
root=Path('/Users/hz0693/research/hyperboloidal')
out=root/'build-layer-research/manufactured-angular-seed-admissibility-plan-held-20261009'
out.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
text="""# Held Gaussian seed and slicing admissibility plan

This is a source-only next-step plan. No scalar evaluation, inverse solve,
quadrature, CAS, compiler, kernel query or evolution is admitted here. The
original general-f pencil and independent cross-review remain immutable.
The manufactured solution is for full physical-reference wave-map gauge;
it generally does not solve the new coupled inner gauge additions.

Choose the dimensionally consistent seed

    f(T)=sigma^4 exp[-T^2/(2 sigma^2)]

and use F=partial_1 partial_2([f(T-R)-f(T+R)]/R),
Y^0=T+epsilon F, Y^I=X^I, with the same complete Minkowski hyperboloidal
reference. Proposed fixed profiles are sigma=7/20 and sigma=1/2;
epsilon=0,1/4,1/2,3/4. These are predeclared probes, not admitted amplitudes.
Use S=1, a=1/2, geometry cutoff r0=1/20,r1=19/20, H(0)=0,
and bounded future native time 0<=t<=6. No BH or puncture conclusion follows.

## Exact bounds available before numerical work

Writing x=T/sigma, f''''=(x^4-6x^2+3) exp(-x^2/2).
Its exact supremum absolute value is3: differentiation gives extrema
x=0 and x^2=5+-sqrt(10); both latter extrema have magnitude less than3.
Consequently the sphere integral bound gives

    J=1+epsilon F_T >= 1-4 |epsilon|/pi.

For the largest epsilon=3/4 this is1-3/pi>0. Since f and its derivatives
are bounded, F is bounded and the pure-time inverse is globally unique
at each fixed physical spatial X. This does not certify native slicing.
A coarse explicit bound |f'''|<=4 sigma follows by separately maximizing
|x|^3 exp(-x^2/2) and3|x|exp(-x^2/2). Hence

    |F|<=16 sigma/(3 pi),
    |T-(t+H)|<=|epsilon|16 sigma/(3 pi).

This supplies a genuine global inverse bracket; later root solving must
verify residual and bracket width independently, not just Newton steps.

Also f''=sigma^2(x^2-1) exp(-x^2/2), with exact supremum sigma^2.
The leading outgoing native slicing margin therefore satisfies

    lim R^2 D >= a^2-|epsilon|sigma^2.

At sigma=1/2,epsilon=3/4 this lower bound is1/16. At native time
 t=-C_H, the exact leading lapse relative to reference has angular extrema

    [1+epsilon sigma^2/a^2]^-1/2,
    [1-epsilon sigma^2/a^2]^-1/2.

The largest profile has the second factor2, so its proposed conformal lapse
can differ by100percent at scri while keeping a positive leading margin.
These expressions concern the future leading coefficient only. Neither
this leading bound nor global J>0 proves D>0 at finite radius.

## Exact fixed-event angular minimization

Let p=n1*n2, s=n1^2+n2^2, F=R^2 p C(T,R), h=H_R,
d0=1-h^2. At fixed physical (T,R),

    |grad F|^2=R^2 C^2 s+4R^3 p^2 C C_R+R^4 p^2 C_R^2,
    A=C_T+h(C_R+2C/R),
    B=C_T^2-C_R^2-4C C_R/R,
    D=d0+2epsilon R^2 A p
         +epsilon^2[R^4 B p^2-R^2 C^2 s].

For every p in[-1/2,1/2], s=1 is geometrically attainable and minimizes
D. Thus the exact sphere minimum at this physical event is

    d0-epsilon^2 R^2 C^2
       +min_{p in[-1/2,1/2]}[2epsilon R^2 A p+epsilon^2 R^4 B p^2].

Evaluate both endpoints and the interior vertex when B>0. At R=0 use
the analytic regular origin values, with D=J=1. At fixed native t the
implicit T depends on p: minimizing over an independently enclosing
physical T interval gives a sufficient bound, while substituting one
angle's inverse T for every p is invalid.

## Required next gates

First derive and independently review cancellation-free compact D/Omega^2,
metric/ADM values and derivative pathways, including advanced terms and the
exact core. Preserve the fixed-event versus fixed-native-time distinction.
Then prepare a separate source-pinned diagnostic recipe before execution:
analytic Gaussian derivatives, regular origin representation, the exact
outer CMC branch, interval or rigorous weighted outer remainder control,
and finite-annulus/core timelike bounds over the stated native-time window.
A finite sample can screen an amplitude but cannot prove global positivity.
Record failed profiles unchanged; no lapse/radius/Omega floors or adjusted
thresholds. Only admitted profiles proceed to full consumed jets, vacuum
constraints, physical-RWM source identity, and native bulk/boundary controls.
The live actual-pulse full derivative gate remains a separate experiment.
"""
(out/'PLAN.md').write_text(text)
pins=[]
for rel in ['build-layer-research/continuum/manufactured-angular-time-wave-pencil-20261009/index.json','build-layer-research/continuum/manufactured-angular-time-wave-pencil-20261009/DERIVATION.md','build-layer-research/continuum/manufactured-angular-time-wave-independent-review-20261009/index.json','build-layer-research/continuum/manufactured-angular-time-wave-independent-review-20261009/receipt.json','src/z4c/hyperboloidal/layer_reference.hpp']:
 p=root/rel;pins.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
(out/'context.json').write_text(json.dumps(dict(source_only=True,pins=pins),indent=2)+'\n')
files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())]
(out/'index.json').write_text(json.dumps(dict(source_only=True,execution_admitted=False,files=files),indent=2)+'\n')
print(json.dumps(dict(path=str(out),index_sha256=sha(out/'index.json'),files=files)))
