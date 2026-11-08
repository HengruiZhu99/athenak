from decimal import Decimal as D, localcontext
from math import factorial, log2
import json

def trig(x, cosine=False):
    term = D(1) if cosine else x
    total = term
    for k in range(1,100):
        term *= -x*x / D((2*k-1)*(2*k) if cosine else (2*k)*(2*k+1))
        total += term
        if abs(term) < D('1e-55'): break
    return total

def dimension(n, omega, cosine):
    sources=[-1+(D(i)+D('.5'))*2/n for i in range(n)]
    vals=[trig(omega*x,cosine) for x in sources]
    result=[]
    for i in range(2*n):
        x=-1+(D(i)+D('.5'))/n
        t=(D(i)/2-D('.25'))
        first=max(0,min(n-6,int(t//1)-(1 if t<0 else 0)-2))
        weights=[]
        for j in range(first,first+6):
            w=D(1)
            for k in range(first,first+6):
                if j!=k: w *= (t-k)/D(j-k)
            weights.append(w)
        approx=sum(w*vals[first+j] for j,w in enumerate(weights))
        product=D(1)
        for node in sources[first:first+6]: product *= abs(x-node)
        bound=omega**6*product/D(factorial(6))
        result.append((trig(omega*x,cosine),approx,bound,sum(abs(w) for w in weights)))
    return result

with localcontext() as ctx:
    ctx.prec=60
    result=[]
    for n in [12,24,48,96,192]:
        us=dimension(n,D('5.1'),False);vs=dimension(n,D('4.7'),True)
        error=max(abs(a[0]*b[0]-a[1]*b[1]) for a in us for b in vs)
        bound=max(a[2]+a[3]*b[2] for a in us for b in vs)
        result.append(dict(n=n,error=str(error),remainder_bound=str(bound),
                           error_over_bound=str(error/bound)))
    for a,b in zip(result,result[1:]): b['order_from_previous']=log2(float(a['error'])/float(b['error']))
print(json.dumps(dict(precision_decimal_digits=60, waveform='sin(5.1u)*cos(4.7v)',
    method='independent Decimal sine/cosine Taylor series, rational Lagrange weights',
    bound='Bu + Lu*Bv; Ba=omega_a^6/6!*product_i abs(x-x_i)',results=result),indent=2))
