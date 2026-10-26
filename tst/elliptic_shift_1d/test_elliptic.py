#!/usr/bin/env python3
"""Independent manufactured, differential, boundary and tensor-kernel checks."""
import argparse,json,unittest
import numpy as np
from elliptic import solve,residual,connection
from numerics import RadialOperators
from kernel import Kernel,PARITY

LIBRARY=None
class EllipticTests(unittest.TestCase):
    def test_polynomial_operators(self):
        r=(np.arange(48)+.5)/47.5;op=RadialOperators(r)
        for degree in range(1,7):
            d,dd=op.derivatives(r**degree,(-1)**degree)
            np.testing.assert_allclose(d,degree*r**(degree-1),atol=2e-12,rtol=0)
            expected=degree*(degree-1)*r**max(degree-2,0)
            np.testing.assert_allclose(dd,expected,atol=3e-10,rtol=0)
            np.testing.assert_allclose(op.integral(r**degree),r**(degree+1)/(degree+1),atol=3e-14,rtol=0)

    def test_nonlinear_manufactured(self):
        errors=[]
        for n in (24,48,96):
            r=(np.arange(n)+.5)/(n-.5);op=RadialOperators(r)
            op.elliptic_quadrature='inward'
            a=np.exp(.3*r*r);da=.6*r*a;dda=(.6+.36*r*r)*a
            alpha=(1+r*r)/2;dalpha=r
            b=-r+.2*r*(r*r-1)+.01*r**7
            db=-1+.2*(3*r*r-1)+.07*r**6;ddb=1.2*r+.42*r**5
            num=b*da+4*a*(db-b/r)/3
            arr=num/(2*alpha)
            dnum=db*da+b*dda+4*da*(db-b/r)/3+4*a*(ddb-db/r+b/r**2)/3
            darr=dnum/(2*alpha)-num*dalpha/(2*alpha**2)
            got=solve(op,a,da,dda,alpha,dalpha,arr,darr,b[-1])
            errors.append(float(max(abs(got[0]-b))))
            np.testing.assert_allclose(residual(r,a,da,dda,alpha,dalpha,arr,darr,*got[:3],0),0,atol=8e-12)
            self.assertEqual(got[0][-1],b[-1])
        self.assertLess(errors[-1],2e-11)
        self.assertGreater(errors[0]/max(errors[1],1e-16),20)
        print('manufactured_solution_errors',errors)

    def test_inward_quadrature_polynomial_and_dynamic_range(self):
        import math
        for grid in ['half','nodal','cutoff']:
            r=(np.arange(129)+.5)/128.5 if grid=='half' else np.linspace(0 if grid=='nodal' else .03,1,129)
            op=RadialOperators(r,puncture=grid=='cutoff')
            for degree in range(7):
                got=op.tail_integral(r**degree,origin=float(degree==0))
                np.testing.assert_allclose(got,(1-r**(degree+1))/(degree+1),atol=2e-14,rtol=0)
        r=(np.arange(513)+.5)/512.5;op=RadialOperators(r)
        f=1e12*np.maximum(1-r/.4,0.)**8+1e-12*r*r
        inc=np.sum(op.qweights*np.r_[1e12,f][op.qindex],axis=1)
        i=480;reference=math.fsum(inc[i+1:]);inward=op.tail_integral(f,origin=1e12)[i]
        prefix=op.integral(f,origin=1e12);old=prefix[-1]-prefix[i]
        self.assertGreater(reference,0)
        self.assertLess(abs(inward-reference),2e-15*reference)
        self.assertGreater(abs(old-reference),.9*reference)

    def test_damped_connection_independent_bvp(self):
        errors=[]
        for n in (24,48,96):
            r=(np.arange(n)+.5)/(n-.5);op=RadialOperators(r)
            a=np.exp(.15*r*r);da=.3*r*a;dda=(.3+.09*r*r)*a
            alpha=(1+r*r)/2;dalpha=r;arr=.01*r*r;darr=.02*r
            b,db,ddb,Q=solve(op,a,da,dda,alpha,dalpha,arr,darr,-1.,eta=.7)
            # An independent finite-difference boundary-value solve of expanded
            # second-order equation. The integral solver's derivatives aren't used.
            D1,D2=op.derivatives(np.eye(n),-1)
            P=-2*da/a+(a**1.5+2)/r;A=4*a/3;B=da-4*a/(3*r)
            dB=dda-4*da/(3*r)+4*a/(3*r*r)
            M=A[:,None]*D2+(4*da/3+B+P*A)[:,None]*D1+np.diag(dB+P*B)
            rhs=2*(dalpha*arr+alpha*darr)+2*P*alpha*arr-.7*a*a*connection(r,a,da)
            M[-1]=0;M[-1,-1]=1;rhs[-1]=-1
            fd=np.linalg.solve(M,rhs)
            errors.append(float(max(abs(fd-b))))
            np.testing.assert_allclose(residual(r,a,da,dda,alpha,dalpha,arr,darr,b,db,ddb,.7),0,atol=1e-10)
        self.assertLess(errors[-1],2e-9)
        self.assertGreater(errors[0]/errors[1],10)
        print('independent_bvp_errors',errors)

    def test_tensor_connection_freezing_and_scalar_audit(self):
        if LIBRARY is None: self.skipTest("Pass --library")
        r=(np.arange(64)+.5)/63.5;op=RadialOperators(r);kernel=Kernel(LIBRARY)
        q=np.zeros((len(r),8));q[:,1]=np.expm1(.1*r*r);q[:,2]=.01*r*r
        q[:,4]=1e-5*(1-r*r)**4;q[:,5]=.002*r*(1-r*r)**3
        q[:,6]=.02*(1-r*r)**4
        d,dd=op.derivatives(q,PARITY);a=1+q[:,1];alpha=(1+r*r)/2+q[:,6]
        b,db,ddb,Q=solve(op,a,d[:,1],dd[:,1],alpha,r+d[:,6],q[:,2],d[:,2],-1.,eta=.4)
        R,S,diag=kernel.parts(r,q,d,dd,b,db,ddb)
        np.testing.assert_allclose(R[:,1],Q,atol=8e-14,rtol=0)
        # Finite differences of the actual kernel metric RHS verify V_t=-eta V.
        dQ,_=op.derivatives(R[:,1],1)
        Vt=dQ/a**2+(-2*d[:,1]/a**3+1/(r*np.sqrt(a))+2/(r*a*a))*R[:,1]
        np.testing.assert_allclose(Vt,-.4*connection(r,a,d[:,1]),atol=3e-8,rtol=0)
        R0,S0,_=kernel.parts(r,q,d,dd,b,db,ddb,appendix_c=False)
        w=b*r/alpha
        np.testing.assert_allclose(S[:,4]-S0[:,4],3*alpha*w*q[:,4],atol=8*np.finfo(float).eps,rtol=0)
        keep=[0,1,2,3,5,6,7]
        np.testing.assert_array_equal(S[:,keep],S0[:,keep]);np.testing.assert_array_equal(R,R0)
        # Lambda errors remain independent of the metric-derived shift.
        q2=q.copy();q2[:,5]+=.001*r*(1-r*r)**3
        R2,S2,diag2=kernel.parts(r,q2,d,dd,b,db,ddb)
        self.assertGreater(max(abs(diag2[:,3]-diag[:,3])),1e-6)
        self.assertGreater(max(abs(S2[:,5]-S[:,5])),1e-7)

    def test_unsubtracted_minkowski(self):
        if LIBRARY is None: self.skipTest("Pass --library")
        r=(np.arange(64)+.5)/63.5;q=np.zeros((64,8));z=np.zeros_like(r)
        R,S,diag=Kernel(LIBRARY).parts(r,q,q,q,-r,-np.ones_like(r),z)
        assembled=R[:-1]+S[:-1]/((1-r[:-1]**2)/2)[:,None]
        self.assertLess(max(abs(assembled).flat),4e-11)
        self.assertLess(max(abs(S[-1])),4e-14)
        self.assertLess(max(abs(diag[:,0])),4e-14)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library');args,other=p.parse_known_args();LIBRARY=args.library
    unittest.main(argv=['test_elliptic',*other])
