"""Independent physical spherical constraints from dl²=A dr²+B² dOmega².

No conformal-Ricci or tensor-kernel formulas enter. The full geometric trace
Kr+2Kt is retained, including any algebraic trace-A defect in free-field data.
The last node (scri) is excluded from this finite-interior chart.
"""
import numpy as np

def evaluate(r,q,d,dd,omega):
    r=r[:-1];q=q[:-1];d=d[:-1];dd=dd[:-1];o=omega[:-1,0];op=omega[:-1,1];opp=omega[:-1,2]
    chi,gr,gt,Ar,At,K,theta,Lambda=q[:,:8].T
    u=o*np.sqrt(chi/gr);B=r*np.sqrt(gt/chi)/o
    L=1/r+.5*(d[:,2]/gt-d[:,0]/chi)-op/o
    Lp=-1/r**2+.5*(dd[:,2]/gt-(d[:,2]/gt)**2-dd[:,0]/chi+(d[:,0]/chi)**2)-opp/o+(op/o)**2
    up=u*(op/o+.5*d[:,0]/chi-.5*d[:,1]/gr)
    scalar=2/B**2-6*u*u*L*L-4*u*up*L-4*u*u*Lp
    Kr=o*Ar/gr+(K+2*theta)/3;Kt=o*At/gt+(K+2*theta)/3
    Ktp=op*At/gt+o*d[:,4]/gt-o*At*d[:,2]/gt**2+(d[:,5]+2*d[:,6])/3
    H=scalar+(Kr+2*Kt)**2-Kr**2-2*Kt**2
    M=2*u*(L*(Kr-Kt)-Ktp)
    Gamma=d[:,1]/(2*gr**2)+2*(gr-gt)/(r*gr*gt)-d[:,2]/(gr*gt)
    Z=u*.5*gr*(Lambda-Gamma)
    return np.column_stack((H,M,theta,Z))
