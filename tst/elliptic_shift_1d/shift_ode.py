"""Banded discrete integration of the metric-derived g=1 shift equation.

D_r v = 3 alpha Ar/(2r), v(scri)=v_outer, with beta=r(-1+v).
Regular-center even parity supplies the bounded homogeneous branch. No evolved
connection or differential Einstein constraint enters this solve.
"""
import numpy as np


class BandedShift:
    def __init__(self,op,refinements=2,bias="centered"):
        if refinements not in (0,1,2,4):raise ValueError('Unsupported refinement count')
        if bias not in ('centered','right'):raise ValueError('Unsupported shift derivative bias')
        indices=op.index if bias=='centered' else op.aindex[1]
        coefficients=op.d1 if bias=='centered' else op.ad1[1]
        self.n=len(op.r);self.bandwidth=op.width;self.scale=float(op.r[1]-op.r[0]);self.refinements=refinements
        # The centered even-parity origin derivative is an identically zero
        # row and must be omitted. The biased reflected row is nonzero and is
        # part of that discrete closure; omitting it selects a parasitic mode.
        self.skip=int(op.origin_on_grid and not op.puncture and bias=='centered');b=self.bandwidth;n=self.n
        self.matrix=np.zeros((n,2*b+1));w=op.width
        for i in range(n-1):
            row=i+self.skip
            for k,coefficient in zip(indices[row],coefficients[row]):
                physical=(w-k if op.origin_on_grid else w-1-k) if k<w else k-w
                offset=physical-i+b
                if not 0<=offset<2*b+1:raise ValueError('Derivative stencil exceeds shift bandwidth')
                self.matrix[i,offset]+=self.scale*coefficient
        self.matrix[-1,b]=1.;self.lu=self.matrix.copy()
        for k in range(n):
            diagonal=self.lu[k,b]
            if not np.isfinite(diagonal) or abs(diagonal)<1e-13:
                raise ValueError('Discrete shift matrix needs pivoting; unsupported grid/operator')
            right=min(b,n-k-1)
            for i in range(k+1,min(n,k+b+1)):
                slot=b+k-i;multiplier=self.lu[i,slot]/diagonal;self.lu[i,slot]=multiplier
                self.lu[i,slot+1:slot+right+1]-=multiplier*self.lu[k,b+1:b+right+1]
        self.minimum_pivot=float(min(abs(self.lu[:,b])))
        self.factor_growth=float(np.max(abs(self.lu))/np.max(abs(self.matrix)))
        self.last_residual=0.;self.last_backward_error=0.

    def triangular_solve(self,rhs):
        x=np.array(rhs,dtype=float,copy=True);n=self.n;b=self.bandwidth
        for i in range(n):
            left=min(b,i);x[i]-=self.lu[i,b-left:b]@x[i-left:i]
        for i in range(n-1,-1,-1):
            right=min(b,n-1-i);x[i]=(x[i]-self.lu[i,b+1:b+right+1]@x[i+1:i+right+1])/self.lu[i,b]
        return x

    def residual(self,rhs,x):
        n=self.n;b=self.bandwidth;result=np.asarray(rhs,dtype=np.longdouble).copy()
        for offset in range(-b,b+1):
            lo=max(0,-offset);hi=min(n,n-offset)
            result[lo:hi]-=self.matrix[lo:hi,b+offset].astype(np.longdouble)*x[lo+offset:hi+offset].astype(np.longdouble)
        return result

    def solve(self,source,outer):
        if len(source)!=self.n:raise ValueError('Shift source/grid mismatch')
        rhs=np.empty(self.n);rhs[:-1]=self.scale*source[self.skip:self.n-1+self.skip];rhs[-1]=outer
        x=self.triangular_solve(rhs)
        for _ in range(self.refinements):x+=self.triangular_solve(self.residual(rhs,x))
        residual=self.residual(rhs,x);self.last_residual=float(max(abs(residual[:-1]))/self.scale)
        denominator=np.max(np.sum(abs(self.matrix),axis=1))*max(abs(x))+max(abs(rhs))
        self.last_backward_error=float(max(abs(residual))/denominator) if denominator else 0.
        if not np.all(np.isfinite(x)):raise FloatingPointError('Nonfinite elliptic ODE solution')
        return x
