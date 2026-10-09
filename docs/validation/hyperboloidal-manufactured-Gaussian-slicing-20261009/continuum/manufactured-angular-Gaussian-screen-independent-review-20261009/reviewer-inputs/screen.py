"""Source-only until exact external authorization; finite physical-event screen."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(1048576),b''):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify(pins):
    for p,d in pins.items():
        if sha(p)!=d:raise RuntimeError('changed pinned input '+p)

def run(recipe,out):
    import mpmath as mp
    def rat(s):
        f=Fraction(s);return mp.mpf(f.numerator)/f.denominator
    records=[]
    def derivatives(t,sigma,n):
        x=t/sigma;e=mp.exp(-x*x/2);h0=mp.mpf(1);h1=x
        values=[sigma**4*e]
        if n:values.append(-sigma**3*h1*e)
        for k in range(1,n):
            h0,h1=h1,x*h1-k*h0
            values.append((-1)**(k+1)*sigma**(3-k)*h1*e)
        return values
    def radial(t,r,sigma,terms):
        if r==0:
            f=derivatives(t,sigma,6)
            return -2*f[5]/15,-2*f[6]/15,mp.mpf(0),'origin'
        if r<=sigma/8:
            f=derivatives(t,sigma,2*terms+6)
            c=mp.mpf(0);ct=mp.mpf(0);cr=mp.mpf(0)
            for j in range(terms):
                factor=-8*(j+2)*(j+1)/mp.factorial(2*j+5)
                c+=factor*f[2*j+5]*r**(2*j)
                ct+=factor*f[2*j+6]*r**(2*j)
                if j:cr+=factor*f[2*j+5]*(2*j)*r**(2*j-1)
            return c,ct,cr,'origin_series'
        u=derivatives(t-r,sigma,3);v=derivatives(t+r,sigma,3)
        c=(u[2]-v[2])/r**3+3*(u[1]+v[1])/r**4+3*(u[0]-v[0])/r**5
        ct=(u[3]-v[3])/r**3+3*(u[2]+v[2])/r**4+3*(u[1]-v[1])/r**5
        cr=-(u[3]+v[3])/r**3-6*(u[2]-v[2])/r**4-15*(u[1]+v[1])/r**5-15*(u[0]-v[0])/r**6
        return c,ct,cr,'advanced_retarded'
    for digits,terms in recipe['precision_and_terms']:
        mp.mp.dps=digits
        for sigma_s in recipe['sigma']:
            sigma=rat(sigma_s);a=rat(recipe['a'])
            for rratio_s in recipe['radius_over_sigma']:
                r=sigma*rat(rratio_s)
                ratio_exact=Fraction(rratio_s)
                times={Fraction(s) for s in recipe['time_over_sigma']}
                times.update(ratio_exact+Fraction(s) for s in recipe['retarded_time_over_sigma'] if ratio_exact+Fraction(s)>=0)
                for time_ratio in sorted(times):
                    t=sigma*rat(str(time_ratio))
                    c,ct,cr,branch=radial(t,r,sigma,terms)
                    for eps_s in recipe['epsilon']:
                        eps=rat(eps_s)
                        if r==0:
                            minimum=mp.mpf(1);scaled=minimum;p=mp.mpf(0)
                            direct=minimum;j=mp.mpf(1)
                        else:
                            h=r/mp.sqrt(r*r+a*a);d0=a*a/(r*r+a*a)
                            aa=ct+h*(cr+2*c/r);bb=ct*ct-cr*cr-4*c*cr/r
                            linear=2*eps*r*r*aa;quadratic=eps*eps*r**4*bb
                            candidates=[-mp.mpf('0.5'),mp.mpf('0.5')]
                            if quadratic>0:
                                vertex=-linear/(2*quadratic)
                                if abs(vertex)<=mp.mpf('0.5'):candidates.append(vertex)
                            def val(p):return d0-eps*eps*r*r*c*c+linear*p+quadratic*p*p
                            p=min(candidates,key=val);minimum=val(p);scaled=minimum/d0
                            # Real sphere point with s=1 and chosen p; independently
                            # evaluate the original vector gradient and D definition.
                            n1=mp.sqrt((1+mp.sqrt(1-4*p*p))/2)
                            n2=p/n1
                            ft=r*r*p*ct
                            grad=[r*c*n2+r*r*p*cr*n1,r*c*n1+r*r*p*cr*n2,mp.mpf(0)]
                            w=[h*n1-eps*grad[0],h*n2-eps*grad[1],mp.mpf(0)]
                            j=1+eps*ft;direct=j*j-sum(v*v for v in w)
                        residual=abs(direct-minimum)/max(1,abs(direct),abs(minimum))
                        records.append(dict(digits=digits,terms=terms,sigma=sigma_s,epsilon=eps_s,
                            radius_over_sigma=rratio_s,time_over_sigma=mp.nstr(t/sigma,digits),branch=branch,
                            D=mp.nstr(minimum,digits),D_over_reference_D=mp.nstr(scaled,digits),
                            minimizing_p=mp.nstr(p,digits),J_at_minimum=mp.nstr(j,digits),
                            angular_identity_residual=mp.nstr(residual,digits),passed_positive=bool(minimum>0)))
            write(out/('progress-'+str(digits)+'-'+sigma_s.replace('/','_')+'.json'),
                  dict(completed_precision=digits,sigma=sigma_s,records=len(records)))
    write(out/'samples.json',records)
    low=[q for q in records if q['digits']==recipe['precision_and_terms'][0][0]]
    high=[q for q in records if q['digits']==recipe['precision_and_terms'][1][0]]
    if len(low)!=len(high):raise RuntimeError('precision grid count differs')
    mp.mp.dps=recipe['precision_and_terms'][1][0]
    max_precision=mp.mpf(0);max_identity=mp.mpf(0)
    keys=['sigma','epsilon','radius_over_sigma','branch']
    for x,y in zip(low,high):
        if any(x[k]!=y[k] for k in keys) or abs(mp.mpf(x['time_over_sigma'])-mp.mpf(y['time_over_sigma']))>mp.mpf('1e-70'):
            raise RuntimeError('precision grid keys differ')
        for key in ['D','D_over_reference_D','J_at_minimum']:
            xx=mp.mpf(x[key]);yy=mp.mpf(y[key]);max_precision=max(max_precision,abs(xx-yy)/max(1,abs(xx),abs(yy)))
        max_identity=max(max_identity,mp.mpf(x['angular_identity_residual']),mp.mpf(y['angular_identity_residual']))
    groups=[]
    for sigma in recipe['sigma']:
        for eps in recipe['epsilon']:
            group=[q for q in high if q['sigma']==sigma and q['epsilon']==eps]
            worst=min(group,key=lambda q:mp.mpf(q['D_over_reference_D']))
            groups.append(dict(sigma=sigma,epsilon=eps,samples=len(group),all_sampled_D_positive=all(q['passed_positive'] for q in group),worst=worst))
    passed=bool(max_precision<=mp.mpf(recipe['precision_tolerance']) and max_identity<=mp.mpf(recipe['identity_tolerance']))
    result=dict(scope='finite physical-event Gaussian screen only; no global timelike, native inverse, jets, PDE or stability admission',
        checks_passed=passed,all_sampled_D_positive=all(q['passed_positive'] for q in high),
        samples_per_precision=len(high),precision_comparison_max=mp.nstr(max_precision,110),
        angular_identity_max=mp.nstr(max_identity,110),profiles=groups,
        sampled_negativity_is_not_a_native_time_failure_without_inverse_domain_check=True,
        global_positivity_proven=False,continuum_or_native_stability_accepted=False)
    write(out/'result.json',result)
    if not passed:raise RuntimeError('identity or precision gate failed')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authorization',type=Path,required=True)
    ap.add_argument('--authorization-sha256',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve()
    if out.parent!=(HERE/'attempts').resolve():raise ValueError('fresh direct attempt child required')
    out.mkdir(parents=True,exist_ok=False)
    pins={};started=time.monotonic();receipt=dict(completed=False,returncode=1,scientific_scope='finite Gaussian physical-event screen')
    try:
        if sys.flags.optimize!=0 or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('unoptimized bytecode-off runtime required')
        if sha(args.authorization)!=args.authorization_sha256:raise RuntimeError('authorization digest differs')
        auth=load(args.authorization);recipe=load(HERE/'recipe.json');index=load(HERE/'source-index.json')
        if not(auth.get('finite_Gaussian_screen_authorized') is True and auth.get('recipe_sha256')==sha(HERE/'recipe.json') and auth.get('source_index_sha256')==sha(HERE/'source-index.json') and auth.get('screen_source_sha256')==sha(__file__)):
            raise RuntimeError('exact root source/math release required')
        pins.update(recipe['pins'])
        for row in index['files']:pins[row['path']]=row['sha256']
        pins[str(HERE/'source-index.json')]=sha(HERE/'source-index.json')
        pins[str(args.authorization.resolve())]=args.authorization_sha256
        verify(pins);write(out/'pins-before.json',pins)
        result=run(recipe,out)
        receipt.update(completed=True,returncode=0,checks_passed=result['checks_passed'],all_sampled_D_positive=result['all_sampled_D_positive'])
    except BaseException as e:
        receipt['failure']=type(e).__name__+': '+str(e);(out/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:verify(pins);receipt['inputs_unchanged']=True
        except BaseException as e:receipt.update(inputs_unchanged=False,post_pin_failure=str(e))
        receipt['seconds']=time.monotonic()-started;write(out/'receipt.json',receipt)
    print(json.dumps(receipt));return receipt['returncode']
if __name__=='__main__':raise SystemExit(main())
