"""Exact copied operand functions; modules supplied only after admission."""
import math

def build_mass_operand(np,dgemm,weighting,radii,weights,refs,maps,angles,ell,ir):
    nc,N,nd,rb = 8,8,64,.98
    na = len(angles)
    qidx = [0,1,2,8,12,16,18,7,11,15]
    vidx = [3,4,5,9,13,17,19,6,10,14]
    r = radii[ir]
    ref = refs[ir]
    c,cr = ref[9:11]
    def mm(a, b):
        return np.einsum('ik,kj->ij', a, b, optimize=False)

    def bilinear(a, b, weights):
        root = np.sqrt(weights)[:, None, None]
        aa = weighting.mul(root, a, "bilinear:left_root_times_a").reshape(-1, a.shape[-1])
        bb = weighting.mul(root, b, "bilinear:right_root_times_b").reshape(-1, b.shape[-1])
        result = dgemm(1., aa.T, bb)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite independent contraction')
        return result

    def action(m, a):
        return np.einsum('ij,ajk->aik', m, a, optimize=False)

    def jacobi_jet(k, beta, x):
        p0 = np.ones_like(x); d0 = np.zeros_like(x); dd0 = d0.copy()
        if k == 0:
            return p0, d0, dd0
        p1 = (-beta+(beta+2)*x)/2
        d1 = np.full_like(x, (beta+2)/2); dd1 = np.zeros_like(x)
        for n in range(2, k+1):
            q = 2*n+beta
            den = 2*n*(n+beta)*(q-2)
            linear = (q-1)*q*(q-2)
            constant = -(q-1)*beta*beta
            back = 2*(n-1)*(n+beta-1)*q
            mult = linear*x+constant
            p = (mult*p1-back*p0)/den
            d = (linear*p1+mult*d1-back*d0)/den
            dd = (2*linear*d1+mult*dd1-back*dd0)/den
            p0,d0,dd0,p1,d1,dd1 = p1,d1,dd1,p,d,dd
        return p1, d1, dd1

    def modal_jet(rho, ell):
        rho = np.atleast_1d(rho)
        B = rb*rb; x = 2*rho/B-1
        result = np.empty((3, len(rho), N))
        for k in range(N):
            norm = math.sqrt(2*(2*k+ell+1.5)/B**(ell+1.5))
            p,d,dd = jacobi_jet(k, ell+.5, x)
            result[:, :, k] = np.stack((norm*p, norm*2*d/B, norm*4*dd/B**2))
        return result

    M = np.zeros((20, 20))
    M[:8, :8] = [[0,0,0,-1,0,0,0,0], [0,0,0,2/3,4/3,0,0,-2/3],
        [0,0,0,0,0,-2,0,4/3], [-1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,1/2,0], [-2/3,1/3,-1/2,0,0,0,2/3,0],
        [0,0,0,-4/3,-2/3,0,0,4/3], [-1,1/2,0,0,0,0,1,0]]
    for first in (8, 12):
        M[first:first+4, first:first+4] = [[0,-2,0,1],[-.5,0,.5,0],[0,0,0,1],[0,0,1,0]]
    for first in (16, 18):
        M[first:first+2, first:first+2] = [[0,-2],[-.5,0]]
    H = np.eye(20)+mm(M.T, M)

    fieldbasis=np.zeros((8,3,nd))
    for channel in range(nc):
        fieldbasis[channel,:,channel*N:(channel+1)*N]=modal_jet([r*r],ell[channel])[:,0]
    point=np.einsum('acjf,cjd->afd',maps[ir],fieldbasis,optimize=False)
    u,ur,urr,v,vr=[point[:,start:start+10] for start in (0,10,20,30,40)]
    y=np.zeros((na,20,nd));dy=y.copy();y[:,qidx]=c*ur;y[:,vidx]=v
    hy=action(H,y)
    measure=weights[ir]/c
    mass_sum=bilinear(y,hy,angles)+bilinear(u,u,angles)
    return measure,mass_sum
