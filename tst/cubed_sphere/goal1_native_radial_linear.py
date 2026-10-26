"""Independent MP linearization of native tensor Z4c/RWM on outer Minkowski.

Scalar radial amplitudes are analysis inputs, not new evolved fields.
Order: chi, g_nn_TF, alpha, beta_n, Omega*A_nn_TF, P, Theta,
Omega*Lambda_n. The returned matrix satisfies Omega*Y_t=M(E;Omega)Y
on a normal power Y=Omega**s*v, with E=Omega*d/dOmega.
"""
import mpmath as mp


def action(s, omega, v, damping=10):
    s, o, d = mp.mpf(s), mp.mpf(omega), mp.mpf(damping)
    c, h, a, b, ar, p, theta, lam = map(mp.mpf, v)
    r = mp.sqrt(1-o)
    L = 2-o
    w = -4*r*r/L
    d1 = lambda x: -2*r*s*x
    d2 = lambda x: (4*r*r*s*(s-1)-2*s*o)*x
    dc, dh, da, db, dp, dt = map(d1, [c,h,a,b,p,theta])
    ddc, ddh, dda, ddb = map(d2, [c,h,a,b])
    dar = -2*r*(s-1)*ar
    dlam = -2*r*(s-1)*lam
    z1 = (lam-dh-3*o*h/r)/2
    dz = (dlam-ddh-3*o*dh/r+3*o*o*h/(r*r))/2
    ell1 = -6*o*c+2*r*dh+6*o*h+r*dc
    ricci = ddh+5*o*dh/r+3*o*o*h/(r*r)+2*ddc+4*o*dc/r
    lap_alpha = dda+2*o*da/r+6*o*o*c-2*r*o*dh-6*o*o*h-r*o*dc
    cross1 = -4*r*r*o*(c-h)-2*r*da
    dw = (2*r*b-w*a)/L
    fchi = -2*r*dc-mp.mpf(2)/3*(db+2*o*b/r)-4*o*a/L+2*L*(p+2*theta-3*dw)/3
    fh = -2*L*ar-2*r*dh+mp.mpf(4)/3*(db-o*b/r)
    fa = -2*r*da+2*r*o*b+4*r*r*o*a/L-(6*L*a+6*L*L*c+L*L*p+(-2*L+8)*r*b)
    fb = (-2*r*db-2*o*b+L*L*lam+L*L*dc/2-L*da-2*r*o*a
          -2*L*r*o*c+2*L*r*o*h-4*L*r*a-2*L*L*r*c
          +8*L*L*r*h+(-6*L-8*r*r+8*r*r/L)*b)
    fA = (-mp.mpf(2)/3*(dda-o*da/r)+r*o*(dh-4*dc)/3
          +L*(ddh/6+o*dh/(3*r)-o*o*h/(r*r)+ddc/3-o*dc/(3*r)
              +mp.mpf(4)/3*(dz-o*z1/r))-2*r*dar
          +2*L*r*(dh-4*dc-8*z1)/3+L*(-6-w)*ar)
    fp = -lap_alpha+3*cross1+L*ell1+6*o*a-2*r*dp-4*L*p-8*L*theta-12*L*r*r*(c-h)+d*theta
    ft = L*(ricci+2*dz+4*o*z1/r)/2+2*L*ell1-2*r*dt-4*L*p-L*(8+3*w)*theta-12*L*r*r*(c-h)-2*d*theta
    fl = (-4*o*lam-2*r*dlam+2*o*dh+6*o*o*h/r-4*r*o*ar
          +mp.mpf(4)/3*(ddb+2*o*db/r-2*o*o*b/(r*r))
          +(8*L-2*d)*z1-mp.mpf(2)/3*L*(2*dp+dt)+8*L*r*ar)
    return mp.matrix([fchi,fh,fa,fb,fA,fp,ft,fl])


def constraints(s, omega, v):
    s, o = mp.mpf(s), mp.mpf(omega)
    c,h,a,b,ar,p,theta,lam = map(mp.mpf,v)
    r=mp.sqrt(1-o)
    d1=lambda x:-2*r*s*x
    d2=lambda x:(4*r*r*s*(s-1)-2*s*o)*x
    dc,dh,dp,dt=map(d1,[c,h,p,theta])
    R=d2(h)+5*o*dh/r+3*o*o*h/(r*r)+2*d2(c)+4*o*dc/r
    ell1=-6*o*c+2*r*dh+6*o*h+r*dc
    H=R-8*p-16*theta+4*ell1-24*r*r*(c-h)
    M=(-2*r*(s-3)+3*o/r)*ar-mp.mpf(2)/3*(dp+2*dt)
    Z=(lam-dh-3*o*h/r)/2
    return mp.matrix([H,M,theta,Z])


def matrix(s, omega, damping=10):
    eye=mp.eye(8)
    return mp.matrix([[action(s,omega,eye[:,j],damping)[i]for j in range(8)]for i in range(8)])


def clock(s):
    s=mp.mpf(s)
    return mp.matrix([-mp.mpf(1)/2,1,-mp.mpf(3)/2,3,-s,mp.mpf(3)/2*(3-s),0,-2*s])
