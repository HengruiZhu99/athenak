"""Native 26-variable principal audit; candidate mode never changes evolution."""
import argparse,ctypes as C,hashlib,json,struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def balance(p):
    b=p.copy();d=np.ones(len(p))
    for _ in range(60):
        changed=False
        for i in range(len(p)):
            r=np.sum(np.abs(b[i]))-abs(b[i,i]);c=np.sum(np.abs(b[:,i]))-abs(b[i,i])
            if r<=0 or c<=0:continue
            f=2.**round(.5*np.log2(r/c))
            if r/f+c*f<.95*(r+c):b[i]/=f;b[:,i]*=f;d[i]*=f;changed=True
        if not changed:break
    return b,d

def spaces(p,expected):
    b,d=balance(p);scale=max(1.,np.linalg.norm(b,2));tol=2e-9*scale
    groups=[]
    for x in sorted(expected):
        if groups and abs(x-groups[-1][0])<2e-8:groups[-1][1]+=1
        else:groups.append([x,1])
    info=[];vectors=[]
    for x,m in groups:
        _,sv,vh=np.linalg.svd(b-x*np.eye(26));k=int(sum(sv<tol))
        info.append(dict(speed=float(x),algebraic=m,geometric=k,smallest_required_singular=float(sv[-m])))
        if k:vectors.extend(vh[-k:])
    complete=all(x['algebraic']==x['geometric'] for x in info)
    return dict(complete=complete,balanced_basis_condition=float(np.linalg.cond(np.array(vectors).T)) if complete else None,spaces=info)

def bisect(f,a,b):
    fa=f(a);assert fa*f(b)<0
    for _ in range(65):
        m=(a+b)/2;fm=f(m)
        if b-a<2e-13:return m
        if fa*fm<=0:b=m
        else:a=m;fa=fm
    return (a+b)/2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--snapshot',type=Path);ap.add_argument('--coordinates',type=Path);ap.add_argument('--snapshot-sha256');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lib=C.CDLL(str(a.library.resolve()));fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.inner_gauge_symbol.argtypes=[C.c_int]+[C.c_double]*3+[fp,fp,fp]
    lib.hyperbolic_symbol.argtypes=[C.c_double,C.c_double,fp,fp]+[C.c_double]*6+[fp]
    lib.candidate_outer_rhs_error.restype=C.c_double
    outer_rhs_error=float(lib.candidate_outer_rhs_error());assert outer_rhs_error<1e-12
    rng=np.random.default_rng(20261101)
    def symbol(mode,r,alpha,chi,g,n):
        p=np.zeros((26,26));assert lib.inner_gauge_symbol(mode,r,alpha,chi,np.ascontiguousarray(g),np.ascontiguousarray(n),p)==0
        return p
    def speeds(p,alpha,chi,g,n):
        q=n@np.linalg.inv(g)@n;mu=p[17,13]+p[17,23]
        l=-chi*p[16,11]*q+p[16,20:23]@p[20:23,16]
        assert l>0 and mu>0
        return np.array([alpha*np.sqrt(chi*q),np.sqrt(l),np.sqrt(4*mu*q/3),np.sqrt(mu*q)])
    def assess(mode,r,alpha,chi,g,n):
        p=symbol(mode,r,alpha,chi,g,n);cp,cl,cs,ct=speeds(p,alpha,chi,g,n)
        ex=np.array([0.]*6+[-cp,cp]*6+[-cl,cl,-cs,cs]+[-ct,ct]*2)
        ev=np.linalg.eigvals(p)
        error=max(float(max(abs(np.sort(ev.real)-np.sort(ex)))),float(max(abs(ev.imag))))
        z=spaces(p,ex);z.update(spectrum_error=error,gap=float(cs*cs-cl*cl));return p,z
    n=np.array([1.,0,0]);g=np.eye(3)
    # Independent original native-symbol equivalence; no numeric eigenproblem
    # is used to choose the expected speeds.
    donor_error=0.;outer_error=0.;inner_error=0.
    for r in np.linspace(.01,1,51):
        alpha=(1+r*r)/2;chi=1.;o=(1-r*r)/2
        def smooth(x):
            if x<=0:return 0.
            if x>=1:return 1.
            t=min(x,1-x);v=t**5*(126+t*(-420+t*(540+t*(-315+70*t))))
            return 1-v if x>.5 else v
        w=smooth((.8-r)/.24);t=(r-.8)/.1
        collar=0 if t<=0 else 1 if t>=1 else t**8/(t**8+(1-t)**8)
        F=alpha**2+4*o*o*alpha;eta=1.0001*alpha/(alpha+.0001*(1+r*r)/2);mu=(1-collar)*4+collar*1.5*chi*F
        old=np.zeros((26,26));assert lib.hyperbolic_symbol(alpha,chi,g,n,F,eta,w,.005,.02,mu,old)==0
        p=symbol(0,r,alpha,chi,g,n);donor_error=max(donor_error,float(np.max(abs(old-p))))
        if r>=.9:outer_error=max(outer_error,float(np.max(abs(p-symbol(1,r,alpha,chi,g,n)))),float(np.max(abs(p-symbol(2,r,alpha,chi,g,n)))))
        if r<=.2:inner_error=max(inner_error,float(np.max(abs(symbol(1,r,alpha,chi,g,n)-symbol(2,r,alpha,chi,g,n)))))
    assert donor_error<1e-8 and outer_error<1e-8 and inner_error<1e-8
    stats={str(m):dict(cases=0,incomplete=[],max_spectrum_error=0,max_balanced_condition=0,min_gap=1e99) for m in range(3)}
    def record(mode,case,r,alpha,chi,g,n):
        p,z=assess(mode,r,alpha,chi,g,n);v=stats[str(mode)];v['cases']+=1
        v['max_spectrum_error']=max(v['max_spectrum_error'],z['spectrum_error']);v['min_gap']=min(v['min_gap'],z['gap'])
        if not z['complete']:v['incomplete'].append(dict(case=case,r=r,alpha=alpha,chi=chi,metric=g.tolist(),normal=n.tolist(),audit=z))
        else:v['max_balanced_condition']=max(v['max_balanced_condition'],z['balanced_basis_condition'])
        assert z['spectrum_error']<2e-7,(mode,case,z)
        return p,z
    radii=np.linspace(0,1,201);profiles=np.zeros((3,len(radii),4));profile_audits={str(m):[] for m in range(3)}
    for i,r in enumerate(radii):
        for mode in range(3):
            p,z=record(mode,'reference',float(r),(1+r*r)/2,1.,g,n);profiles[mode,i]=speeds(p,(1+r*r)/2,1,g,n);profile_audits[str(mode)].append(z)
    for sample in range(160):
        m=rng.normal(size=(3,3));metric=m@m.T+np.eye(3);metric/=np.linalg.det(metric)**(1/3)
        normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
        r=float(rng.uniform(0,1));alpha=float(np.exp(rng.uniform(np.log(.03),np.log(1.5))));chi=float(np.exp(rng.uniform(np.log(1e-4),np.log(3))))
        for mode in range(3):record(mode,f'random-{sample}',r,alpha,chi,metric,normal)
    # Verified old N16 GD1 state: a state sample, NOT a trajectory of either
    # proposed gauge. Exact coordinates regenerated from ContinuousCoreMesh.
    snapshot_stats={str(m):dict(cases=0,min_gap=1e99,max_balanced_condition=0,incomplete=0) for m in range(3)}
    if a.snapshot:
        assert a.coordinates and a.snapshot_sha256
        cp=a.snapshot
        assert hashlib.sha256(cp.read_bytes()).hexdigest()==a.snapshot_sha256
        with cp.open('rb') as f:header=struct.unpack('<16s4i3Q2d',f.read(72))
        assert header[0] in [b'HYP3D_MPI_GD1\0\0\0',b'HYP3D_MPI_IT1\0\0\0'] and header[1:5]==(16,31,2,1)
        values=np.memmap(cp,dtype='<f8',offset=168,mode='r',shape=(header[5],31))
        coords=a.coordinates
        xyz=np.genfromtxt(coords,delimiter=',',names=True);assert len(xyz)==header[5]
        candidates=np.flatnonzero(xyz['volume']>0)
        selected=np.unique(np.r_[rng.choice(candidates,160,replace=False),np.argsort(xyz['r'])[:16]])
        for cell in selected:
            r=float(xyz['r'][cell]);v=values[cell];alpha=float(v[18]+(1+r*r)/2);chi=float(v[0])
            metric=np.array([[1+v[1],v[2],v[3]],[v[2],1+v[4],v[5]],[v[3],v[5],1+v[6]]])
            assert np.linalg.eigvalsh(metric)[0]>0
            radial=np.array([xyz[k][cell] for k in ['x','y','z']])/r
            tangent=np.cross(radial,[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
            oblique=(radial+tangent)/np.sqrt(2)
            for normal in [radial,tangent,oblique]:
                for mode in range(3):
                    _,z=record(mode,f'snapshot-N16-cell-{cell}',r,alpha,chi,metric,normal);ss=snapshot_stats[str(mode)];ss['cases']+=1;ss['min_gap']=min(ss['min_gap'],z['gap']);ss['incomplete']+=not z['complete']
                    if z['complete']:ss['max_balanced_condition']=max(ss['max_balanced_condition'],z['balanced_basis_condition'])
    assert all(not z['incomplete'] for z in stats.values()), 'Unexpected rank defect in generic test cases'
    collisions={}
    # Genuine pure-inner longitudinal collision shared by IT1 and candidate.
    for mode in [1,2]:
        def f(chi):
            p=symbol(mode,.1,.5,chi,g,n);sp=speeds(p,.5,chi,g,n);return sp[1]**2-sp[2]**2
        chi=bisect(f,.1,3);p,z=assess(mode,.1,.5,chi,g,n);assert not z['complete']
        collisions[f'inner-{mode}']=dict(r=.1,alpha=.5,chi=chi,audit=z)
        np.savez(a.output/f'inner-collision-{mode}.npz',P=p,expected=speeds(p,.5,chi,g,n))
    # Tune metric anisotropy while keeping the radial direction well separated.
    # At r=.4 the legacy Euclidean B term remains active in IT1.
    r=.4;alpha=.58;chi=1.
    def anisotropy(T):return np.diag([T**-2,T,T])
    nt=np.array([0.,1.,0.])
    def f(T):
        gt=anisotropy(T);p=symbol(1,r,alpha,chi,gt,nt);sp=speeds(p,alpha,chi,gt,nt);return sp[1]**2-sp[2]**2
    T=bisect(f,1,100);gt=anisotropy(T)
    for mode in range(3):
        p,z=assess(mode,r,alpha,chi,gt,nt);_,rad=assess(mode,r,alpha,chi,gt,n)
        np.savez(a.output/f'tangential-collision-{mode}.npz',P=p,metric=gt,normal=nt)
        collisions[f'tangential-{mode}']=dict(r=r,alpha=alpha,chi=chi,T=T,metric=gt.tolist(),radial_gap=rad['gap'],audit=z)
    assert not collisions['tangential-1']['audit']['complete']
    assert collisions['tangential-1']['radial_gap']>0 and collisions['tangential-2']['audit']['complete']
    # Scri speed sign convention in the original audit: 1-eigenvalue for
    # outward-unit n, reference beta.n=-1. Counts must stay identical.
    scri={}
    for mode in range(3):
        p=symbol(mode,1.,1.,1.,g,n);ev=np.linalg.eigvals(p).real;v=1-ev
        scri[str(mode)]=dict(incoming=int(sum(v<-1e-7)),outgoing=int(sum(v>1e-7)),tangent=int(sum(abs(v)<=1e-7)))
        assert scri[str(mode)]==dict(incoming=3,outgoing=16,tangent=7)
    report=dict(modes={'0':'original smooth donor','1':'current IT1 (two transitions)','2':'audit-only direct-to-outer C4 .2--.9'},library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),candidate_outer_full_rhs_error=outer_rhs_error,donor_symbol_error=donor_error,outer_symbol_error=outer_error,inner_symbol_error=inner_error,stats=stats,snapshot=snapshot_stats,snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest() if a.snapshot else None,coordinates_sha256=hashlib.sha256(a.coordinates.read_bytes()).hexdigest() if a.coordinates else None,collisions=collisions,scri=scri,scope='Frozen interior principal symbol with normalized p/Omega,Theta/Omega; positive metric/chi/lapse only. Balanced conditions are basis dependent. No uniform puncture/scri estimate, boundary well-posedness proof or stability/convergence claim for candidate.')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(a.output/'reference-speeds.npz',r=radii,speeds=profiles)
    colors=['#777777','#1769aa','#b45418'];labels=['Previous donor','Current IT1','Direct blend (proposal)']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for mode in range(3):
        axes[0].plot(radii,profiles[mode,:,1],color=colors[mode],label=labels[mode]);axes[1].plot(radii,profiles[mode,:,2],color=colors[mode]);axes[2].plot(radii,profiles[mode,:,2]**2-profiles[mode,:,1]**2,color=colors[mode])
    for ax,title in zip(axes,['Lapse speed','Longitudinal shift speed','Squared-speed separation']):ax.set(xlabel='Compactified radius r',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Frozen characteristic speeds on the reference state: alpha_bar=(1+r²)/2, chi_bar=1, metric=I')
    fig.tight_layout(rect=(0,.1,1,.92));fig.text(.5,.015,'Comoving speeds; this reference profile is not stationary under the new inner gauge.\nPositive separation here does not rule out collisions on other states.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(a.output/f'characteristics.{ext}',dpi=170)
    print(json.dumps(dict(stats={k:{j:v for j,v in val.items() if j!='incomplete'}|{'incomplete_count':len(val['incomplete'])} for k,val in stats.items()},snapshot=snapshot_stats,scri=scri,collision_chi=collisions['inner-1']['chi'],tangential_metric_T=T),indent=2))
if __name__=='__main__':main()
