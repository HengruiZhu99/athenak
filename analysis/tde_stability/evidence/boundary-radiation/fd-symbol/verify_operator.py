"""Bounded independent consistency/regression checks for the frozen model."""
import json
from pathlib import Path
import numpy as np
from fd_boundary import matrix,vector_matrix,operators
from fourier_boundary import full_matrix

checks=[]
for degree,expected in [(1,-7/30),(2,2.),(3,2.)]:
    D,Dxx,*_=operators(16,.125,degree)
    x=np.arange(16)*.125
    value=float(Dxx[0]@(x*x));assert abs(value-expected)<1e-11
    checks.append(dict(test='boundary_quadratic_moment',degree=degree,value=value,expected=expected))
    for power in range(7):
        truth=0 if power==0 else power*x[8]**(power-1)
        assert abs(D[8]@(x**power)-truth)<1e-10
        truth=0 if power<2 else power*(power-1)*x[8]**(power-2)
        assert abs(Dxx[8]@(x**power)-truth)<1e-9

xyz=np.array([1.875,-.125,-.125]);r=np.linalg.norm(xyz);a=r/(r+1);beta=xyz/(r+1)**2
for mode in ['zero_rate','radiation']:
    n=12;L,_=full_matrix(n=n,alpha=a,chi=a*a,beta=(beta[0],0,0),mode=mode,damping=True)
    for sector,builder,ids,nv in [('scalar',matrix,[6,7,8,13,0,1,16,17],8),('vector',vector_matrix,[10,14,3,18],4)]:
        A,_=builder(n=n,h=.25,alpha=a,chi=a*a,beta=beta[0],mode=mode,damping=True)
        T=np.zeros((20*n,nv*n))
        for j,i in enumerate(ids):T[i*n:(i+1)*n,j*n:(j+1)*n]=np.eye(n)
        if sector=='scalar':T[2*n:3*n,5*n:6*n]=-.5*np.eye(n);T[9*n:10*n,2*n:3*n]=-.5*np.eye(n)
        err=float(np.max(abs(L@T-T@A)));assert err<1e-10
        checks.append(dict(test='independent_sector_reduction',sector=sector,mode=mode,error=err))
    Lt,_=full_matrix(n=n,alpha=a,chi=a*a,beta=(beta[0],0,0),mode=mode,damping=True,gauge_target='tangential',tt_target='tangential')
    err=float(np.max(abs(L-Lt)));assert err<1e-10
    checks.append(dict(test='tangential_target_normal_limit',mode=mode,error=err))

for mode,expected in [('zero_rate',0.),('radiation',1.2127989530042136)]:
    L,data=full_matrix(alpha=a,chi=a*a,beta=tuple(beta),mode=mode,damping=True,angle_y=np.pi/2)
    e,v=np.linalg.eig(L);j=np.argmax(e.real);growth=float(e[j].real)
    assert abs(growth-expected)<1e-8
    residual=float(np.linalg.norm(L@v[:,j]-e[j]*v[:,j])/max(1,np.linalg.norm(L@v[:,j])))
    assert residual<1e-10
    checks.append(dict(test='shifted_oblique_regression',mode=mode,gamma=growth,eigen_residual=residual))
Path('verification.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
print(f'PASS: {len(checks)} reported checks plus 42 interior derivative moments')
