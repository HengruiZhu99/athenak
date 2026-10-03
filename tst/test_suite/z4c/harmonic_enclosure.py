"""Outward interval range of a retained orthonormal real SH surface.

The zonal projection about x/y/z is bounded as a one-dimensional cosine
polynomial. The orthogonal remainder uses the addition theorem. This certifies
the ideal finite expansion with the retained binary coefficients, not a PDE
horizon or its truncation error. Standard-library Decimal arithmetic only.
"""
from decimal import Context,Decimal,ROUND_CEILING,ROUND_FLOOR
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
import numpy as np

PRECISION=100
DOWN=Context(prec=PRECISION,rounding=ROUND_FLOOR)
UP=Context(prec=PRECISION,rounding=ROUND_CEILING)
ZERO=(Decimal(0),Decimal(0))
ONE=(Decimal(1),Decimal(1))


def point(value):
    value=Decimal.from_float(value) if isinstance(value,float) else Decimal(value)
    return value,value


def rational(value):
    value=Fraction(value);a,b=Decimal(value.numerator),Decimal(value.denominator)
    return DOWN.divide(a,b),UP.divide(a,b)


def add(a,b):return DOWN.add(a[0],b[0]),UP.add(a[1],b[1])
def neg(a):return a[1].copy_negate(),a[0].copy_negate()
def sub(a,b):return add(a,neg(b))
def mul(a,b):
    return (min(DOWN.multiply(x,y) for x in a for y in b),
            max(UP.multiply(x,y) for x in a for y in b))
def div(a,b):
    if b[0]<=0:raise ValueError('positive interval denominator required')
    return mul(a,(DOWN.divide(Decimal(1),b[1]),UP.divide(Decimal(1),b[0])))
def square(a):
    lo=Decimal(0) if a[0]<=0<=a[1] else min(DOWN.multiply(x,x) for x in a)
    return lo,max(UP.multiply(x,x) for x in a)
def root(a):
    if a[0]<0:raise ValueError('nonnegative square-root interval required')
    # Decimal.sqrt is correctly rounded HALF_EVEN regardless of context.
    lo=Decimal(0) if a[0]==0 else DOWN.next_minus(DOWN.sqrt(a[0]))
    hi=Decimal(0) if a[1]==0 else UP.next_plus(UP.sqrt(a[1]))
    return lo,hi
def magnitude(a):return max(x.copy_abs() for x in a)


@lru_cache(maxsize=1)
def pi_interval():
    # Machin: tan(4 atan(1/5)-atan(1/239))=1 in the first quadrant.
    # Exact rational alternating sums bracket both arctangents; no stored
    # decimal digits or platform transcendental functions enter the proof.
    def arctan(den):
        x=Fraction(1,den);term=x;total=Fraction(0);terms=160
        for k in range(terms):
            total+=(-1 if k%2 else 1)*term/(2*k+1);term*=x*x
        return total,total+term/(2*terms+1)
    a,b=arctan(5),arctan(239)
    lo=rational(16*a[0]-4*b[1])[0];hi=rational(16*a[1]-4*b[0])[1]
    return lo,hi


def double_factorial(n):
    value=1
    for k in range(n,0,-2):value*=k
    return value


@lru_cache(maxsize=771)
def axis_weights(l,axis):
    if axis not in ('x','y','z'):raise ValueError('Cartesian axis x/y/z required')
    weights=[ZERO]*(2*l+1)
    if axis=='z':weights[0]=ONE;return tuple(weights)
    for m in range(l+1):
        if (l+m)%2:continue
        ratio=Fraction(double_factorial(l+m-1),double_factorial(l-m))
        sign=(-1)**((l+m)//2)
        w=(rational(sign*ratio) if m==0 else root(rational(
            2*Fraction(math.factorial(l-m),math.factorial(l+m))*ratio*ratio)))
        if m and sign<0:w=neg(w)
        if m==0:weights[0]=w
        elif axis=='x':weights[2*m-1]=w
        else:
            c,s=(1,0,-1,0)[m%4],(0,1,0,-1)[m%4]
            if c:weights[2*m-1]=w if c>0 else neg(w)
            if s:weights[2*m]=w if s>0 else neg(w)
    return tuple(weights)


@lru_cache(maxsize=4)
def cosine_grid(intervals):
    """cos(j*pi/N), outward on the pi interval, with exact quadrant reduction."""
    values=[];pi=pi_interval()
    for j in range(intervals+1):
        if j==0:values.append(ONE);continue
        if j==intervals:values.append(neg(ONE));continue
        if 2*j==intervals:values.append(ZERO);continue
        k=min(j,intervals-j);x=mul(pi,rational(Fraction(k,intervals)))
        x2=square(x);term=ONE;total=ONE
        for n in range(1,80):
            term=div(mul(term,x2),point((2*n-1)*(2*n)))
            total=add(total,neg(term) if n%2 else term)
        next_term=div(mul(term,x2),point(159*160))[1]
        # The alternating tail is bounded by this next term. Widen both ends
        # to include its sign and every interval/rounding uncertainty.
        total=(DOWN.subtract(total[0],next_term),UP.add(total[1],next_term))
        values.append(total if 2*j<intervals else neg(total))
    return tuple(values)


def axial_range(coefficients,axis='x',intervals=1024):
    c=np.asarray(coefficients,dtype=float)
    degree=math.isqrt(c.size)-1
    if (c.ndim!=1 or degree<0 or degree>256 or (degree+1)**2!=c.size
        or not np.isfinite(c).all()):raise ValueError('finite complete real SH coefficients through degree256 required')
    if not isinstance(intervals,int) or intervals<16 or intervals>16384 or intervals%2:
        raise ValueError('even theta interval count in[16,16384] required')
    pi=pi_interval();cosine=[ZERO]*(degree+1);remainder=ZERO
    for l in range(degree+1):
        coefficients_l=[point(float(x)) for x in c[l*l:(l+1)**2]]
        weights=axis_weights(l,axis);alpha=ZERO;norm2=ZERO
        for value,weight in zip(coefficients_l,weights):
            alpha=add(alpha,mul(value,weight));norm2=add(norm2,square(value))
        rem2=sub(norm2,square(alpha))
        if rem2[1]<0:raise ArithmeticError('zonal projection violates unit weight norm')
        rem2=(max(Decimal(0),rem2[0]),rem2[1])
        normalization=root(div(point(2*l+1),mul(point(4),pi)))
        remainder=add(remainder,mul(normalization,root(rem2)))
        a=mul(alpha,normalization)
        # Generating-function factorization gives positive exact Fourier
        # coefficients beta=2*A_k*A_(l-k), with the central term undoubled.
        for m in range(l+1):
            if (l-m)%2:continue
            k=(l-m)//2;j=(l+m)//2
            beta=Fraction(math.comb(2*k,k)*math.comb(2*j,j),4**l)*(2 if m else 1)
            cosine[m]=add(cosine[m],mul(a,rational(beta)))
    first=second=Decimal(0)
    for m,value in enumerate(cosine):
        first=UP.add(first,UP.multiply(Decimal(m),magnitude(value)))
        second=UP.add(second,UP.multiply(Decimal(m*m),magnitude(value)))
    step=div(pi,point(intervals))
    lipschitz=mul(point(first),div(step,point(2)))[1]
    interpolation=mul(point(second),div(square(step),point(8)))[1]
    gap=min(lipschitz,interpolation)
    grid=cosine_grid(intervals);minimum=Decimal('Infinity');maximum=Decimal('-Infinity')
    for j in range(intervals+1):
        value=ZERO
        for m,coefficient in enumerate(cosine):
            k=(j*m)%(2*intervals);k=min(k,2*intervals-k)
            value=add(value,mul(coefficient,grid[k]))
        minimum=min(minimum,value[0]);maximum=max(maximum,value[1])
    lower=DOWN.subtract(DOWN.subtract(minimum,gap),remainder[1])
    upper=UP.add(UP.add(maximum,gap),remainder[1])
    lower_float=math.nextafter(float(lower),-math.inf)
    upper_float=math.nextafter(float(upper),math.inf)
    if not math.isfinite(lower_float) or not math.isfinite(upper_float):raise ArithmeticError('finite certificate range required')
    return dict(schema='axial_real_sh_interval_v1',axis=axis,degree=degree,
        basis='orthonormal real spherical harmonics; Condon-Shortley; m0,cos(m),sin(m)',
        coefficient_binary_sha256=hashlib.sha256(c.astype('<f8').tobytes()).hexdigest(),
        theta_intervals=intervals,decimal_precision=PRECISION,pi_interval=list(map(str,pi)),
        axial_sample_min_lower=str(minimum),axial_sample_max_upper=str(maximum),
        orthogonal_remainder_upper=str(remainder[1]),theta_first_derivative_upper=str(first),
        theta_second_derivative_upper=str(second),grid_error_lipschitz_upper=str(lipschitz),
        grid_error_interpolation_upper=str(interpolation),grid_error_used=str(gap),
        continuous_min_lower=str(lower),continuous_max_upper=str(upper),
        radius_lower_bound=lower_float,radius_upper_bound=upper_float,
        scope='retained ideal finite harmonic surface; no PDE/truncation/modified-ball acceptance transfer')
