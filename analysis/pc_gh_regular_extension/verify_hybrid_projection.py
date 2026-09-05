"""Independent symbolic identities and ghost-inclusive production projection checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import sympy as s
from make_inputs import parse
from verify_smooth_pulses import rate_gradient


def identities():
    x,y,z=s.symbols('x y z', real=True)
    xyz=[x,y,z]
    P=s.Function('P')(x,y,z)
    E=[s.Function('E'+str(i))(x,y,z) for i in range(3)]
    for i,j in [(0,1),(0,2),(1,2)]:
        old=s.diff(E[j],xyz[i])-s.diff(E[i],xyz[j])
        new=s.diff((1-P)*E[j],xyz[i])-s.diff((1-P)*E[i],xyz[j])
        assert s.simplify(new-(1-P)*old+s.diff(P,xyz[i])*E[j]-s.diff(P,xyz[j])*E[i])==0
    w,rho,p,L,dw,drho,a=s.symbols('w rho p L dw drho a')
    pp=p-a*(p-dw); LL=L-a*(L-2*(w*drho+rho*dw))
    assert s.expand(LL-2*(w*drho+rho*pp)-(1-a)*(L-2*(w*drho+rho*p)))==0
    # Any linear discrete curl D gives C(G-P E)=C(G)-C(P E), without Leibniz.
    D=s.Matrix([[1,2,-3],[-2,0,2],[3,-1,-2]])
    pv=s.diag(*s.symbols('P0:3')); ev=s.Matrix(s.symbols('E0:3'))
    assert s.expand(D*((s.eye(3)-pv)*ev)-(D*ev-D*pv*ev))==s.zeros(3,1)
    assert D*pv-pv*D != s.zeros(3,3)
    print('PASS: continuum mask curl, exact Ralpha scaling, discrete commutator identity')


def check(run):
    params=parse((run/'used_input.athinput').read_text())
    pc=params['pc_gh']; order=int(pc['spatial_order'])
    coefficients={2:np.array([-.5,0,.5]),
                  4:np.array([1/12,-2/3,0,2/3,-1/12]),
                  6:np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60])}[order]
    dim=sum(int(params['mesh']['nx'+str(d)])>1 for d in [1,2,3])
    worst=0.; limits={'zero':0,'full':0,'taper':0}; curl_error=0.
    for path in sorted(run.glob('projection-before-rank*.csv')):
        before=np.genfromtxt(path,delimiter=',',names=True)
        after=np.genfromtxt(path.with_name(path.name.replace('before','after')),delimiter=',',names=True)
        assert len(before)==len(after)
        for block in np.unique(before['block']):
            b=before[before['block']==block]; out=after[after['block']==block]
            shape=tuple(int(b[key].max())+1 for key in ['k','j','i'])
            u=np.stack([b['u'+str(n)].reshape(shape) for n in range(55)])
            v=np.stack([out['u'+str(n)].reshape(shape) for n in range(55)])
            active=b['active'].reshape(shape).astype(bool)
            def derivative(field,d):
                if d>=dim: return np.zeros_like(field)
                return sum(c*np.roll(field,-offset,axis=2-d)
                           for c,offset in zip(coefficients,range(-order//2,order//2+1)))/b['d'+'xyz'[d]][0]
            target=u.copy()
            for d in range(3):
                dw=derivative(u[0],d); drho=derivative(u[18],d)
                target[22+d]=dw
                target[43+d]=2*(u[0]*drho+u[18]*dw)
                for n in range(6): target[25+6*d+n]=derivative(u[1+n],d)
                for n in range(3): target[46+3*d+n]=derivative(u[19+n],d)
            if pc['reduction_projection_profile']=='global':
                weight=np.ones(len(b))
            else:
                # Independent logistic mask, with the oracle's known final center shift.
                policy=dict(pc,reduction_profile='smooth_core',reduction_rate='0',reduction_inner_rate='1')
                if pc.get('reduction_follow_trackers','false')=='true':
                    n=0
                    while policy.get('co_'+str(n),'false')=='true':
                        key=f'co_{n}_x'; policy[key]=str(float(policy.get(key,0))+.125); n+=1
                weight=rate_gradient(np.column_stack([b[a] for a in 'xyz']),policy,0,0)[0]
            weight=weight.reshape(shape)*active
            expected=u.copy()
            for n in range(22,55):
                expected[n]=np.where(weight==1,target[n],np.where(weight==0,u[n],u[n]+weight*(target[n]-u[n])))
            error=float(abs(v-expected).max()); worst=max(worst,error)
            assert error<2e-12,(run,block,error)
            assert np.array_equal(u[:22],v[:22])
            for label,mask in [('zero',weight==0),('full',weight==1),('taper',(weight>0)&(weight<1))]:
                limits[label]+=int((mask&active).sum())
            assert np.array_equal(v[22:,weight==0],u[22:,weight==0])
            # Includes target curl and all available ghost values. Never assume
            # the factorized L target is a discrete gradient, or that D obeys Leibniz.
            for indexes in [[22+d for d in range(3)],[43+d for d in range(3)]]+[
                [25+6*d+n for d in range(3)] for n in range(6)]+[
                [46+3*d+n for d in range(3)] for n in range(3)]:
                for i,j in [(0,1),(0,2),(1,2)]:
                    actual=derivative(v[indexes[j]],i)-derivative(v[indexes[i]],j)
                    exact=derivative(expected[indexes[j]],i)-derivative(expected[indexes[i]],j)
                    curl_error=max(curl_error,float(abs(actual[active]-exact[active]).max()))
            assert curl_error<1e-10,(run,curl_error)
    assert sum(limits.values())>0,run
    result=dict(run=str(run),max_component_error=worst,max_curl_identity_error=curl_error,cells=limits)
    print('PASS:',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',type=Path,nargs='*'); ap.add_argument('--output',type=Path)
    args=ap.parse_args(); identities()
    results=[check(run) for run in args.runs]
    if args.output: args.output.write_text(json.dumps(results,indent=2)+'\n')
