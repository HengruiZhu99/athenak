from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parent
SHAPE=(25,24,24,24)
ACT=(slice(None),slice(4,20),slice(4,20),slice(4,20))
pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def load(case,op,cycle=0,stage=1,bg=False):
    tag='.background' if bg else ''
    p=ROOT/'validation'/case/f'z4c_snapshot_{op}_rank0_cycle{cycle}_stage{stage}{tag}.bin'
    return np.fromfile(p).reshape(SHAPE)
def source(q):
    v=q[ACT].reshape(25,-1).T
    N=len(v);d=np.zeros((N,3,25))
    for a in range(3):
        for offset,weight in [(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)]:
            sl=list(ACT);sl[3-a]=slice(4+offset,20+offset)
            d[:,a,:]+=weight/.25*q[tuple(sl)].reshape(25,-1).T
    g=np.zeros((N,3,3));A=np.zeros_like(g);dg=np.zeros((N,3,3,3))
    for k,(a,b) in enumerate(pairs):
        g[:,a,b]=g[:,b,a]=v[:,1+k];A[:,a,b]=A[:,b,a]=v[:,8+k]
        dg[:,:,a,b]=dg[:,:,b,a]=d[:,:,1+k]
    gi=np.linalg.inv(g)
    def christoffel(inv,derivative):
        C=np.zeros((N,3,3,3))
        for c in range(3):
            for a in range(3):
                for b in range(3):
                    for e in range(3):
                        C[:,c,a,b]+=.5*inv[:,c,e]*(derivative[:,a,b,e]+derivative[:,b,a,e]-derivative[:,e,a,b])
        return C
    C=christoffel(gi,dg)
    metric_Gamma=np.einsum('nab,ncab->nc',gi,C)
    Q=v[:,14:17]-metric_Gamma
    chi=v[:,0];alpha=v[:,18];Theta=v[:,17];K=v[:,7]+2*Theta
    physical=g/chi[:,None,None];pi=gi*chi[:,None,None]
    dp=dg/chi[:,None,None,None]-g[:,None,:,:]*d[:,:,0,None,None]/chi[:,None,None,None]**2
    Cp=christoffel(pi,dp)
    Z=.5*np.einsum('nij,nj->ni',g,Q)
    dZ=.5*np.einsum('nkij,nj->nki',dg,Q)
    DZ=dZ-np.einsum('ncab,nc->nab',Cp,Z)
    E=DZ+DZ.swapaxes(1,2)  # dQ=0 in this independent identity evaluation.
    trace=np.einsum('nab,nab->n',pi,E)
    Zu=np.einsum('nij,nj->ni',pi,Z)
    Zdalpha=np.einsum('ni,ni->n',Zu,d[:,:,18])
    out=np.zeros_like(v)
    out[:,7]=2*Zdalpha
    out[:,17]=.5*alpha*trace-alpha*K*Theta-Zdalpha
    db=d[:,:,19:22];divbeta=np.trace(db,axis1=1,axis2=2)
    out[:,14:17]=-2*Theta[:,None]*np.einsum('nij,nj->ni',gi,d[:,:,18])
    out[:,14:17]+=(-(2/3)*alpha*K+(2/3)*divbeta)[:,None]*Q
    out[:,14:17]-=np.einsum('nji,nj->ni',db,Q)
    out[:,14:17]+=.3*Q  # Published Gamma damping: old -2sigma Q -> -sigma Q.
    arhs=(alpha*chi)[:,None,None]*(E-physical*trace[:,None,None]/3)-2*(alpha*Theta)[:,None,None]*A
    for k,(a,b) in enumerate(pairs):out[:,8+k]=arhs[:,a,b]
    return out.T.reshape((25,16,16,16))
report={}
zero_files=[p for p in (ROOT/'validation/zero').glob('z4c_snapshot_*.bin') if '.background.' not in p.name and 'rhs_full_vs_bg' not in p.name]
zeros=[np.fromfile(p) for p in zero_files]
report['zero']=dict(arrays=len(zeros),max_abs=float(max(abs(q).max() for q in zeros)),finite=all(np.isfinite(q).all() for q in zeros))
off=[]
for p in (ROOT/'validation/off_new').glob('z4c_snapshot_*.bin'):
    old=ROOT/'validation/off_old'/p.name
    off.append(p.read_bytes()==old.read_bytes())
report['default_off']=dict(arrays=len(off),all_bitwise_equal=all(off))
mf=load('matter_on_low','rhs_full_vs_bg');mb=load('matter_off_low','rhs_full_vs_bg')
rf=load('matter_on_low','volume_rhs');rb=load('matter_off_low','volume_rhs')
report['matter_initial_response']=dict(input_bitwise_equal=bool(np.array_equal(mf.view(np.uint64),mb.view(np.uint64))),
    rhs_bitwise_equal=bool(np.array_equal(rf.view(np.uint64),rb.view(np.uint64))),
    Theta_rhs_max=float(abs(rf[17,4:20,4:20,4:20]).max()),
    finite=bool(np.isfinite(rf).all()),description='Fresh atmosphere rho=1e-14 with normal matter feedback, existing CPBC ceiling unchanged')
full=load('late_on','rhs_full_vs_bg',6667);bg=load('late_on','rhs_full_vs_bg',6667,bg=True)
other=load('late_off','rhs_full_vs_bg',6667)
actual=(load('late_on','volume_rhs',6667)-load('late_off','volume_rhs',6667))[ACT]
expected=source(full)-source(bg)
error=actual-expected
report['late_matched_rhs']=dict(input_bitwise_equal=bool(np.array_equal(full.view(np.uint64),other.view(np.uint64))),
    max_absolute_error=float(abs(error).max()),relative_l2=float(np.linalg.norm(error)/np.linalg.norm(expected)),
    max_expected=float(abs(expected).max()),
    unchanged_fields_max=float(abs(actual[[0,1,2,3,4,5,6,18,19,20,21,22,23,24]]).max()),
    fields={str(i):dict(expected_max=float(abs(expected[i]).max()),error_max=float(abs(error[i]).max())) for i in [7,8,9,10,11,12,13,14,15,16,17]})
assert report['zero']['max_abs']==0 and report['zero']['finite']
assert report['default_off']['all_bitwise_equal']
assert report['matter_initial_response']['input_bitwise_equal'] and report['matter_initial_response']['rhs_bitwise_equal']
assert report['matter_initial_response']['Theta_rhs_max']>1e-14
assert report['late_matched_rhs']['input_bitwise_equal']
assert report['late_matched_rhs']['max_absolute_error']<2e-15
assert report['late_matched_rhs']['unchanged_fields_max']==0
report['passed']=True
report['binary_sha256']=hashlib.sha256((ROOT/'athena-covariant-sources').read_bytes()).hexdigest()
(ROOT/'source-regression.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
