#!/usr/bin/env python3
"""Frozen second-order right-port SAT, actual native symbols and actual SBP D/H."""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--commutator',action='store_true');a=p.parse_args();out=a.output
assert (out/('protocol-v2.json' if a.commutator else 'protocol-v1.json')).exists()
root=Path('docs/validation/goal1-minkowski-20261018')
# Reuse archived parsing and geometric sensors; never import their main programs.
for filename,names in [('check_constraint_incoming_lift.py',['mm','symbols']),('check_goal1_incoming_hierarchy.py',['gram_tt','rank','scaled_rows'])]:
    path=Path('tst/cubed_sphere')/filename
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
def gram(qb,inv):
    # MetricChart's trace-free orthonormal frame induces tr(g^-1 a g^-1 b).
    packed=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    tensors=[]
    for i,j in packed:
        v=np.zeros((3,3));v[i,j]=v[j,i]=1;tensors.append(v)
    H=np.zeros((11,11));H[0,0]=H[7,7]=1;H[8:11,8:11]=np.eye(3)
    for i,x in enumerate(tensors):
        for j,y in enumerate(tensors):H[1+i,1+j]=np.trace(mm(mm(mm(inv,x),inv),y))
    return mm(mm(qb.T,H),qb)
old=Path('docs/validation/cubed-sphere-balanced-20261013/finite-radius-long/pulse-t02-controls/pulse-boundary-symbol-t02.txt')
raw=root/'oblique-symbol.txt';meta=json.loads((root/'oblique-review-classified.json').read_text())['records']
operators=[json.loads(s) for s in (root/'canonical-ko-v2/radial-matrices.jsonl').read_text().splitlines() if s.startswith('{')]
commutators=[]
if a.commutator:
 # H0 D+D^T H0=B0 determines the unweighted diagonal norm, without new exports.
 # The offdiagonal SBP graph gives H0_j/H0_0=-D_0j/D_j0.
 for op in operators:
    N=op['nodes'];D=np.array(op['D']);H=np.array(op['H']);H0=np.full(N,1/N)
    H0[0]=-1/(2*D[0,0])
    for j in range(1,8):H0[j]=-H0[0]*D[0,j]/D[j,0]
    H0[-8:]=H0[:8][::-1];B0=np.zeros((N,N));B0[0,0]=-1;B0[-1,-1]=1
    sbp=float(np.linalg.norm(H0[:,None]*D+D.T*H0[None,:]-B0))
    mu=H/H0;C=H[:,None]*D+D.T*H[None,:]-mu[:,None]*B0
    eigen=np.linalg.eigvalsh(C/np.sqrt(H[:,None]*H[None,:]));bound=float(max(abs(eigen)))
    stretch=op['stretch'];rinner=.25*np.sqrt(2.16);dr=.98-rinner
    tprime0=1 if stretch==0 else stretch/(-np.expm1(-stretch));tprime1=tprime0*np.exp(-stretch)
    logmu=max(abs(2*dr*tprime0/rinner-stretch),abs(2*dr*tprime1/.98-stretch))
    record={'nodes':N,'stretch':stretch,'unweighted_SBP_error':sbp,'commutator_min_eigenvalue':float(eigen[0]),'commutator_max_eigenvalue':float(eigen[-1]),'commutator_norm':bound,'analytic_max_abs_log_volume_derivative':float(logmu),'pilot_upper_bound':10*(1+logmu),'passed':bool(sbp<=1e-10 and bound<=10*(1+logmu))}
    commutators.append(record);op.update(H0=H0,mu=mu,commutator=C)
 for stretch in [0,2.8592697712828787]:
    rows=sorted([r for r in commutators if r['stretch']==stretch],key=lambda r:r['nodes']);ratio=rows[-1]['commutator_norm']/rows[-2]['commutator_norm'];rows[-1]['fine_pair_norm_ratio']=ratio;rows[-1]['passed'] &= ratio<=1.5
records=[]
for path in [old,raw]:
 for index,s in enumerate(symbols(path)):
    M=s.pop('matrix');C=s.pop('constraints');basis=s.pop('basis');inv=s.pop('inverse');n=s.pop('normal');qb=basis[:11,:10]
    lo,hi=s['lo'],s['hi'];beta=-(lo+hi)/2;c=(hi-lo)/2;ain=c+beta;aout=c-beta
    kind={'kind':'reused_evolved_boundary_normal','boundary_normal':True} if path==old else meta[index]['pilot_case']
    A=M[:10,:10];B=M[:10,10:];Ae=A-beta*np.eye(10);T=np.block([[np.eye(10),np.zeros((10,10))],[Ae,B]])
    canonical=np.linalg.solve(T.T,mm(T,M).T).T
    target=np.block([[beta*np.eye(10),np.eye(10)],[c*c*np.eye(10),beta*np.eye(10)]])
    wave=float(np.linalg.norm(canonical-target)/max(1,np.linalg.norm(target)))
    Ji=C[4:,:10]+mm(C[4:,10:],np.linalg.solve(B,(beta+c)*np.eye(10)-A))
    Jo=C[4:,:10]+mm(C[4:,10:],np.linalg.solve(B,(beta-c)*np.eye(10)-A))
    Fraw=np.zeros((6,11));Fraw[0,7]=1;Fraw[1:4,8:11]=np.eye(3)
    for row,V in enumerate(gram_tt(np.linalg.inv(inv),inv,n),4):Fraw[row,1:7]=[V[0,0],2*V[0,1],2*V[0,2],V[1,1],2*V[1,2],V[2,2]]
    F=mm(Fraw,qb);K=np.vstack([Ji,F]);scale=np.linalg.norm(K,axis=1);K/=scale[:,None]
    R=np.linalg.solve(K,np.vstack([-Jo,np.zeros((6,10))])/scale[:,None]);G=gram(qb,inv)
    # Boundary energy divided by2c^2 in the incoming/outgoing q_n amplitudes.
    Flux=np.block([[-ain*G,ain*mm(G,R+np.eye(10))],[ain*mm(R.T+np.eye(10),G),-aout*G-ain*(mm(G,R)+mm(R.T,G))]])
    schur=-aout*G+ain*(G+mm(mm(R.T,G),R))
    # Complete-square identity: Flux = -ain|u-(I+R)v|_G^2 + v^T schur v.
    U=np.hstack([np.eye(10),-np.eye(10)-R]);reconstructed=-ain*mm(mm(U.T,G),U);reconstructed[10:,10:]+=schur
    defect=float(np.linalg.norm(Flux-reconstructed)/max(1,np.linalg.norm(Flux)))
    L=np.linalg.cholesky(G);W=np.block([[L,np.zeros((10,10))],[np.zeros((10,10)),L]])
    left=np.linalg.solve(W,Flux);white=np.linalg.solve(W,left.T).T
    largest=float(np.linalg.eigvalsh((white+white.T)/2)[-1]/max(1,abs(lo),abs(hi)))
    tests=[]
    # Right-port identity includes the unclosed left flux explicitly.
    if kind.get('boundary_normal'):
     for op in operators:
        N=op['nodes'];H=np.array(op['H']);D=np.array(op['D'])
        q=np.sin(.17*np.arange(N*10)).reshape(N,10);pi=np.cos(.11*np.arange(N*10)).reshape(N,10)
        z=mm(D,q);qdot=beta*z+pi;pidot=c*c*mm(D,z)+beta*mm(D,pi)
        mismatch=mm((pi[-1]+c*z[-1])[None,:]+mm((pi[-1]-c*z[-1])[None,:],R.T),np.eye(10))[0]
        weight=op['mu'][-1] if a.commutator else 1.
        pidot[-1]-=ain*weight/H[-1]*mismatch
        direct=float(2*np.einsum('n,ni,ij,nj->',H,pi,G,pidot)+2*c*c*np.einsum('n,ni,ij,nj->',H,z,G,mm(D,qdot)))
        def port(i):return float(beta*(pi[i].dot(G.dot(pi[i]))+c*c*z[i].dot(G.dot(z[i])))+2*c*c*pi[i].dot(G.dot(z[i])))
        left_weight=op['mu'][0] if a.commutator else 1.
        sat=-2*ain*weight*pi[-1].dot(G.dot(mismatch));volume=0.
        if a.commutator:
            Cvol=op['commutator']
            volume=float(beta*np.einsum('ni,nm,ij,mj->',pi,Cvol,G,pi)+beta*c*c*np.einsum('ni,nm,ij,mj->',z,Cvol,G,z)+2*c*c*np.einsum('ni,nm,ij,mj->',pi,Cvol,G,z))
        expected=weight*port(-1)-left_weight*port(0)+sat+volume
        rel=abs(direct-expected)/max(1,abs(direct),abs(expected),abs(weight*port(-1)),abs(left_weight*port(0)),abs(sat),abs(volume))
        E=float(np.einsum('n,ni,ij,nj->',H,pi,G,pi)+c*c*np.einsum('n,ni,ij,nj->',H,z,G,z))
        volume_bound=(abs(beta)+c)*next((r['commutator_norm'] for r in commutators if r['nodes']==N and r['stretch']==op['stretch']),0.)
        tests.append({'nodes':N,'stretch':op['stretch'],'direct_energy_rate':direct,'unclosed_left_flux':-left_weight*port(0),'closed_right_flux':weight*port(-1)+sat,'volume_commutator_rate':volume,'volume_rate_bound':volume_bound,'volume_rate_over_energy':volume/E,'relative_identity_defect':rel,'passed':bool(rel<=1e-10 and (not a.commutator or abs(volume)/E<=volume_bound+1e-10))})
    rec=dict(**s,case_index=index,pilot_case=kind,ain=ain,aout=aout,live_metric_Gram_condition=float(np.linalg.cond(G)),wave_identity_error=wave,
        boundary_complete_square_error=defect,normalized_closed_right_maximum=largest,no_SAT_incoming_positive_rate=ain,
        boundary_normal_admission_required=bool(kind.get('boundary_normal')),actual_SBP_tests=tests,
        passed=bool(wave<=1e-10 and defect<=1e-10 and largest<=1e-10 and all(t['passed'] for t in tests)))
    records.append(rec)
required=[r for r in records if r['boundary_normal_admission_required']]
report={'scope':__doc__,'variant':'mapped commutator' if a.commutator else 'omitted mapped commutator (failed)', 'boundary_normal_port_pass':all(r['passed'] for r in required) and all(r['passed'] for r in commutators),'required_normal_cases':len(required),'commutator_checks':commutators,
    'reused_evolved_normal_cases':sum(r['pilot_case']['kind']=='reused_evolved_boundary_normal' for r in records),'records':records,
    'non_boundary_failed_controls':[{'source':r['source'],'case':r['case_index']} for r in records if not r['boundary_normal_admission_required'] and not r['passed']],
    'full_method_adopted':False,'exact_scri_admitted':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [old,raw,root/'canonical-ko-v2/radial-matrices.jsonl']}}
(out/('review-v2.json' if a.commutator else 'review.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print({k:v for k,v in report.items() if k not in ['records','input_sha256','non_boundary_failed_controls']})
print('normal failures',[(r['source'],r['case_index'],r['normalized_closed_right_maximum'],max(t['relative_identity_defect'] for t in r['actual_SBP_tests'])) for r in required if not r['passed']])
