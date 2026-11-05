"""Analysis-only exact spherical slicing/map controls; no production mutation.

Run with Python + numpy/scipy/matplotlib. The output directory must be new.
The native baseline is independently checked against preserved C++ samples.
These one-dimensional derivative tests are not native constraints/evolutions.
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, platform
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import betainc, betaln

D = Path(__file__).resolve().parent
M = 1 / 3
R0, R1 = 20*M, 40*M


def cdf(z, p=7., q=7.):
    return betainc(p, q, np.clip(z, 0, 1))


def pdf(z, p=7., q=7.):
    z = np.asarray(z)
    t = np.clip(z, 1e-100, 1-1e-15)
    return np.where((z > 0) & (z < 1),
                    np.exp((p-1)*np.log(t)+(q-1)*np.log1p(-t)-betaln(p,q)), 0.)


def native_s(y):
    return float(cdf((1/y-R0)/(R1-R0))) if y else 1.


def integral(y):
    return quad(lambda z: 1/np.sqrt(z*z-2*M*z**3+native_s(z)**2),
                0, y, points=[1/R1] if y > 1/R1 else None,
                epsabs=2e-13, epsrel=2e-13)[0]


A = np.exp(-integral(1/R0))
B0 = np.exp(-integral(1/R1))
K_SCALE = A / ((R0-M+np.sqrt(R0*(R0-2*M)))/2)


class Profile:
    def __init__(self, b=None, p=None, q=7., tolerance=2e-13):
        self.b, self.p, self.q = b, p, q
        self.name = 'native_R20_40' if b is None else f'compact_b{b:.2f}'
        def rhs(r, Y):
            y = Y[0]
            s = native_s(y) if b is None else float(cdf((r-A)/(b-A), p, q))
            return [-np.sqrt(y*y-2*M*y**3+s*s)/r]
        self.sol = solve_ivp(rhs, (1, A-.001), [0.], method='DOP853',
                             rtol=tolerance, atol=tolerance/100,
                             max_step=.001, dense_output=True)
        assert self.sol.success

    def fields(self, r):
        r = np.asarray(r)
        y = self.sol.sol(r)[0]
        R = 1/y
        if self.b is None:
            s = cdf((R-R0)/(R1-R0))
            sr_R = pdf((R-R0)/(R1-R0))/(R1-R0)
            radicand = 1-2*M*y+R*R*s*s
            ar = np.sqrt(radicand)
            Rprime = R*ar/r
            sp = sr_R*Rprime
        else:
            s = cdf((r-A)/(self.b-A), self.p, self.q)
            sp = pdf((r-A)/(self.b-A), self.p, self.q)/(self.b-A)
            radicand = 1-2*M*y+R*R*s*s
            ar = np.sqrt(radicand)
            Rprime = R*ar/r
        omega = (1-r*r)/2
        w = r*y
        wprime = y-y*ar
        denom = np.sqrt(y*y-2*M*y**3+s*s)
        wsecond = (-denom+y-3*M*y*y)/r-s*sp/denom
        lr = -s-r*sp/ar
        lt = -s
        trace = lr+2*lt
        alpha = omega*ar  # actual stationary-reference lapse, not free lapse blend
        aprime = -r*ar+omega*((M/R**2+R*s*s)*Rprime+R*R*s*sp)/ar
        chi = (w/omega)**2
        chiprime = 2*chi*(wprime/w+r/omega)
        return dict(r=r, R=R, S=s, chi=chi, alpha=alpha, beta=-s*r,
                    trace=trace, lr=lr, lt=lt, w=w, wprime=wprime,
                    wsecond=wsecond, alpha_prime=aprime, chi_prime=chiprime,
                    beta_prime=-s-r*sp, dr_dR=1/Rprime,
                    spacelike_margin=(1-2*M*y)/radicand,
                    H_exact=4*w*(wsecond+2*wprime/r)-6*wprime*wprime
                            +trace*trace-lr*lr-2*lt*lt)


def solve_shape(b):
    # Preserve both compact inner radius and areal R0, hence inner isotropic k.
    def residual(p):
        return Profile(b,p,tolerance=2e-11).sol.sol(A)[0]-1/R0
    return brentq(residual, 7, 150, xtol=2e-10)


def fd_metrics(profile, h):
    # Equal coordinate sample domain for all profiles and all spacings.
    r = np.linspace(.65, .995, 2049)
    f = [profile.fields(r+j*h) for j in [-2,-1,0,1,2]]
    def d1(key):
        return (f[0][key]-8*f[1][key]+8*f[3][key]-f[4][key])/(12*h)
    def d2(key):
        return (-f[0][key]+16*f[1][key]-30*f[2][key]+16*f[3][key]-f[4][key])/(12*h*h)
    w=f[2]['w']; wp=d1('w'); lr=f[2]['lr']; lt=f[2]['lt']; tr=f[2]['trace']
    H=4*w*(d2('w')+2*wp/r)-6*wp*wp+tr*tr-lr*lr-2*lt*lt
    Mr=w*(d1('lr')+2*(1/r-wp/w)*(lr-lt)-d1('trace'))
    def l2(v):return float(np.sqrt(np.trapezoid(v*v*r*r,r)/np.trapezoid(r*r,r)))
    return dict(h=h, H_coordinate_L2=l2(H), M_orthonormal_coordinate_L2=l2(Mr),
                H_max=float(np.max(abs(H))), M_max=float(np.max(abs(Mr))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    baseline=Profile()
    candidates=[Profile(b,solve_shape(b)) for b in [.93,.95,.97,.99]]
    # Exact native C++ oracle cross-check; exterior uses identical conformal frame.
    oracle=D/'reference-profile-regularity/initial.csv.gz'
    with gzip.open(oracle,'rt') as f:rows=[x for x in csv.DictReader(f) if A+.001<float(x['r'])<.998]
    r=np.array([float(x['r']) for x in rows]);v=baseline.fields(r)
    checks={}
    for key,col,scale in [('R','R_over_M',M),('S','S',1),('chi','chi',1),
                          ('alpha','alpha',1),('trace','K',1),
                          ('alpha_prime','dr_alpha',1),('chi_prime','dr_chi',1)]:
        ref=np.array([float(x[col])*scale for x in rows])
        checks[key]=float(np.max(abs(v[key]-ref)/(1+abs(ref))))
    assert max(checks[k] for k in ['R','S','chi','alpha','trace'])<2e-10,checks
    assert max(checks.values())<2e-7,checks
    r=np.linspace(A+.00001,.999,8193)
    summary={};profiles={}
    for p in [baseline]+candidates:
        v=p.fields(r);profiles[p.name]=v
        grad=abs(v['alpha_prime']);half=r[grad>=max(grad)/2]
        hs=[1/640,1/960,1/1280,1/1920]
        fd=[fd_metrics(p,h) for h in hs]
        inner_error=float(p.sol.sol(A)[0]*R0-1)
        assert abs(inner_error)<2e-9
        summary[p.name]=dict(b=p.b or B0,p=p.p,q=p.q,inner_y_relative_error=inner_error,
             end_areal_radius_M=float(p.fields(np.array([p.b or B0]))['R'][0]/M),
             alpha_gradient_max=float(max(grad)),alpha_halfmax_width=float(half[-1]-half[0]),
             chi_gradient_max=float(max(abs(v['chi_prime']))),
             beta_gradient_max=float(max(abs(v['beta_prime']))),
             K_max=float(max(abs(v['trace']))),
             analytic_H_max=float(max(abs(v['H_exact']))),
             min_spacelike_margin=float(min(v['spacelike_margin'])),
             min_dr_dR=float(min(v['dr_dR'])),fd=fd)
        for key in ['H_coordinate_L2','M_orthonormal_coordinate_L2']:
            summary[p.name][key+'_fit_order']=float(np.polyfit(np.log(hs),np.log([x[key] for x in fd]),1)[0])
        with gzip.open(out/(p.name+'.csv.gz'),'wt') as f:
            writer=csv.writer(f);keys=list(v);writer.writerow(keys)
            writer.writerows(zip(*(v[k] for k in keys)))
    # Tightened integration independently checks selected broader-profile fields.
    chosen=candidates[2];tight=Profile(chosen.b,chosen.p,tolerance=3e-14).fields(r)
    numerical={k:float(max(abs(tight[k]-profiles[chosen.name][k])/(1+abs(tight[k]))))
               for k in ['R','alpha','chi','trace','alpha_prime']}
    assert max(numerical.values())<2e-10,numerical
    result=dict(inner_compact_radius=A,inner_areal_radius_M=20,inner_isotropic_scale=K_SCALE,
        native_outer_compact_radius=B0,native_oracle_scaled_errors=checks,
        tightened_ode_scaled_errors=numerical,profiles=summary,
        qualification='Analysis-only spherical exact-data controls. Fixed central k and R0; varying outer transition radius and slicing. FD4 physical H and orthonormal radial momentum, coordinate r^2 weighted L2 on .65-.995; not native constraint norms, not boosted/evolved/SMR/interface/penalty validation or stability/CFL/cost evidence.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_sha256=hashlib.sha256(oracle.read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for name,v in profiles.items():
        label='native R=20–40M' if name==baseline.name else 'compact S, b='+name[-4:]
        for a,key in zip(ax.flat[:4],['S','alpha','alpha_prime','trace']):
            a.plot(r,v[key],label=label);a.set_xlabel('compact radius r');a.set_ylabel(key)
        d=summary[name]['fd']
        for a,key in zip(ax.flat[4:],['H_coordinate_L2','M_orthonormal_coordinate_L2']):
            a.loglog([1/x['h'] for x in d],[x[key] for x in d],'-o',label=label)
            a.set_xlabel('1 / radial derivative spacing');a.set_ylabel(key)
    for a in ax.flat:a.grid(alpha=.2)
    ax[0,0].legend(fontsize=8)
    fig.suptitle('Initial slicing/map control: identical central scale; no evolution')
    for ext in ['png','pdf']:fig.savefig(out/('profiles.'+ext),dpi=170)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
