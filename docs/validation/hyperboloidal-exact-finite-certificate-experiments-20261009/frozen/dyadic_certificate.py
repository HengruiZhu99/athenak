"""Exact binary64-input complex similarity certificate; standard library only.

No floating arithmetic is used in certificate matrix products, norms, Neumann
bound or disc counting. Floating point is only an exact input/output encoding.
"""
from dataclasses import dataclass
from fractions import Fraction
import math,struct

def bits(value):return struct.unpack('>Q',struct.pack('>d',value))[0]
def from_bits(value):return struct.unpack('>d',struct.pack('>Q',value))[0]
MAX_FINITE=from_bits(0x7fefffffffffffff)

def _dyadic_float(n,e):
    """Encode exactly representable nonnegative n*2**e, including overflow inf."""
    if n==0:return 0.
    while n%2==0:n//=2;e+=1
    k=n.bit_length()-1+e
    if k>1023:return math.inf
    if k>=-1022:
        assert n.bit_length()<=53
        significand=n<<(53-n.bit_length())
        return from_bits(((k+1023)<<52)+(significand-(1<<52)))
    shift=e+1074;assert shift>=0
    return from_bits(n<<shift)

def outward_binary64(value):
    """Directed enclosing binary64 endpoints of an exact rational, by integers."""
    value=Fraction(value)
    if value<0:
        lo,hi=outward_binary64(-value);return -hi,-lo
    if value==0:return 0.,0.
    if value>Fraction.from_float(MAX_FINITE):return MAX_FINITE,math.inf
    n,d=value.numerator,value.denominator
    k=n.bit_length()-d.bit_length()
    below_power=(n<(d<<k)) if k>=0 else ((n<<(-k))<d)
    if below_power:k-=1
    e=max(k-52,-1074)
    quotient,remainder=divmod(n,d<<e) if e>=0 else divmod(n<<(-e),d)
    return _dyadic_float(quotient,e),_dyadic_float(quotient+(remainder!=0),e)

def nearest_binary64(value):
    """Exact round-to-nearest/ties-even for finite-range exact rationals."""
    value=Fraction(value);lo,hi=outward_binary64(value)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError('nearest encoder only admits finite-range directed endpoints')
    lower,upper=Fraction.from_float(lo),Fraction.from_float(hi)
    dl,du=value-lower,upper-value
    if dl<du:return lo
    if du<dl:return hi
    return lo if bits(lo)%2==0 else hi

def fraction_record(value):
    value=Fraction(value);lo,hi=outward_binary64(value)
    return {'numerator':str(value.numerator),'denominator':str(value.denominator),
            'outward_binary64_hex':[lo.hex(),hi.hex()]}

@dataclass(frozen=True)
class DyadicMatrix:
    # Each exact complex entry is (real+i*imag)/2**exponent.
    real:tuple
    imag:tuple
    exponent:int

    @property
    def rows(self):return len(self.real)
    @property
    def columns(self):return len(self.real[0])

    @classmethod
    def from_binary64(cls,rows):
        rows=[list(row) for row in rows]
        if not rows or not rows[0] or any(len(r)!=len(rows[0]) for r in rows):raise ValueError('ragged/empty matrix')
        ratios=[];exponent=0
        for row in rows:
            rr=[]
            for z in row:
                z=complex(z)
                if not(math.isfinite(z.real) and math.isfinite(z.imag)):raise ValueError('nonfinite binary64 entry')
                a,b=z.real.as_integer_ratio(),z.imag.as_integer_ratio()
                assert a[1]&(a[1]-1)==0 and b[1]&(b[1]-1)==0
                exponent=max(exponent,a[1].bit_length()-1,b[1].bit_length()-1);rr.append((a,b))
            ratios.append(rr)
        real=[];imag=[]
        for row in ratios:
            real.append(tuple(a[0]<<(exponent-(a[1].bit_length()-1)) for a,b in row))
            imag.append(tuple(b[0]<<(exponent-(b[1].bit_length()-1)) for a,b in row))
        return cls(tuple(real),tuple(imag),exponent)

    def entry(self,i,j):
        denominator=1<<self.exponent
        return Fraction(self.real[i][j],denominator),Fraction(self.imag[i][j],denominator)

    def multiply(self,other):
        if self.columns!=other.rows:raise ValueError('matrix shape mismatch')
        # Exact arbitrary-precision integer multiply/add, with shared dyadic scale.
        br=list(zip(*other.real));bi=list(zip(*other.imag));real=[];imag=[]
        for ar,ai in zip(self.real,self.imag):
            real.append(tuple(sum(x*y-u*v for x,u,y,v in zip(ar,ai,colr,coli)) for colr,coli in zip(br,bi)))
            imag.append(tuple(sum(x*v+u*y for x,u,y,v in zip(ar,ai,colr,coli)) for colr,coli in zip(br,bi)))
        return DyadicMatrix(tuple(real),tuple(imag),self.exponent+other.exponent)

    def subtract(self,other):
        if(self.rows,self.columns)!=(other.rows,other.columns):raise ValueError('matrix shape mismatch')
        exponent=max(self.exponent,other.exponent);a=exponent-self.exponent;b=exponent-other.exponent
        real=tuple(tuple((x<<a)-(y<<b) for x,y in zip(r,s)) for r,s in zip(self.real,other.real))
        imag=tuple(tuple((x<<a)-(y<<b) for x,y in zip(r,s)) for r,s in zip(self.imag,other.imag))
        return DyadicMatrix(real,imag,exponent)

    def right_diagonal(self,diagonal):
        if diagonal.rows!=1 or diagonal.columns!=self.columns:raise ValueError('diagonal shape mismatch')
        dr,di=diagonal.real[0],diagonal.imag[0]
        real=tuple(tuple(x*y-u*v for x,u,y,v in zip(ar,ai,dr,di)) for ar,ai in zip(self.real,self.imag))
        imag=tuple(tuple(x*v+u*y for x,u,y,v in zip(ar,ai,dr,di)) for ar,ai in zip(self.real,self.imag))
        return DyadicMatrix(real,imag,self.exponent+diagonal.exponent)

    def upper_infinity_norm(self):
        # |z| <= |Re z|+|Im z|; sum is an exact rational upper bound.
        numerator=max(sum(abs(x)+abs(y) for x,y in zip(ar,ai)) for ar,ai in zip(self.real,self.imag))
        return Fraction(numerator,1<<self.exponent)

def identity(n):
    return DyadicMatrix(tuple(tuple(int(i==j) for j in range(n)) for i in range(n)),tuple((0,)*n for _ in range(n)),0)

def disc_components(centers,radius):
    """Exact connected components of common-radius CLOSED complex discs."""
    n=len(centers);unseen=set(range(n));components=[];diameter2=(2*radius)**2
    while unseen:
        first=min(unseen);unseen.remove(first);component=[first];todo=[first]
        while todo:
            i=todo.pop();neighbours=[]
            for j in sorted(unseen):
                dx=centers[i][0]-centers[j][0];dy=centers[i][1]-centers[j][1]
                if dx*dx+dy*dy<=diameter2:neighbours.append(j)
            for j in neighbours:unseen.remove(j);component.append(j);todo.append(j)
        components.append(sorted(component))
    return components

def certify(J,V,W,eigenvalues):
    """Certificate for exact rounded J, using arbitrary binary64 approximate V,W,D."""
    J=DyadicMatrix.from_binary64(J);V=DyadicMatrix.from_binary64(V);W=DyadicMatrix.from_binary64(W)
    D=DyadicMatrix.from_binary64([eigenvalues]);n=J.rows
    if J.columns!=n or (V.rows,V.columns)!=(n,n) or(W.rows,W.columns)!=(n,n) or D.columns!=n:
        raise ValueError('square full basis and one center per dimension required')
    N=W.multiply(V);defect=identity(n).subtract(N);nu=defect.upper_infinity_norm()
    result={'dimension':n,'arithmetic':'Exact arbitrary-precision integer dyadic products; exact rational bounds; complex modulus bounded by |Re|+|Im|.',
      'nu_upper':fraction_record(nu),'invertibility_proved':nu<1,
      'certified_positive_eigenvalues_at_least':0,'certified_negative_eigenvalues_at_least':0,
      'scope':'Eigenvalues of the exact supplied rounded finite matrix J only; no continuum, nonlinear, CPBC or subsidiary classification.'}
    if nu>=1:
        result['status']='inconclusive: ||I-WV|| bound is not strictly below1';return result
    M=W.multiply(J.multiply(V));R=M.subtract(N.right_diagonal(D));rho=R.upper_infinity_norm();radius=rho/(1-nu)
    centers=[D.entry(0,i) for i in range(n)];groups=disc_components(centers,radius);clusters=[]
    for group in groups:
        left=min(centers[i][0] for i in group)-radius;right=max(centers[i][0] for i in group)+radius
        positive=left>0;negative=right<0
        # Strict separation follows because all touching/overlapping discs merged.
        for i in group:
            for j in set(range(n))-set(group):
                dx=centers[i][0]-centers[j][0];dy=centers[i][1]-centers[j][1]
                assert dx*dx+dy*dy>(2*radius)**2
        clusters.append({'center_indices':group,'eigenvalues_counted_with_algebraic_multiplicity':len(group),
            'real_lower':fraction_record(left),'real_upper':fraction_record(right),
            'entire_cluster_strictly_positive':positive,'entire_cluster_strictly_negative':negative})
        if positive:result['certified_positive_eigenvalues_at_least']+=len(group)
        if negative:result['certified_negative_eigenvalues_at_least']+=len(group)
    result.update({'residual_upper':fraction_record(rho),'common_radius_upper':fraction_record(radius),'clusters':clusters,
      'status':'certified positive finite-matrix eigenvalue' if result['certified_positive_eigenvalues_at_least'] else 'no positive cluster certified; inconclusive about existence'})
    return result
