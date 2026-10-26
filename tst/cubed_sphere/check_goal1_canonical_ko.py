#!/usr/bin/env python3
"""Actual native frozen20-mode canonical KO lift and constraint consistency.

Uses native symbol/pole conventions and actual exported radial kernels. This
does not evolve a prescribed surrogate or admit live native boundary closure.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--tapered',action='store_true');a=p.parse_args();out=a.evidence
protocol=json.loads((out/('tapered-protocol-v1.json' if a.tapered else 'protocol-v1.json')).read_text())
raw=Path('docs/validation/goal1-minkowski-20261018/oblique-symbol.txt')
meta=json.loads(Path('docs/validation/goal1-minkowski-20261018/oblique-review-classified.json').read_text())['records']
matrixfile=out/('tapered-radial-matrices.jsonl' if a.tapered else 'radial-matrices.jsonl')
operators=[json.loads(s) for s in matrixfile.read_text().splitlines() if s.startswith('{')]
def mm(x,y):return np.einsum('ij,jk->ik',x,y)
def symbols():
    lines=iter(raw.read_text().splitlines());index=-1
    for line in lines:
        if not line.startswith('case '):continue
        index+=1;_,family,radius,label,omega,lo,hi=line.split()
        inverse=np.array(list(map(float,next(lines).split()[1:]))).reshape(3,3)
        normal=list(map(float,next(lines).split()[1:]));assert next(lines)=='constraints'
        C=np.array([list(map(float,next(lines).split())) for _ in range(8)])
        matrices=[]
        for step in [1.,.5]:
            assert next(lines)==f'matrix {step:g}'
            matrices.append(np.array([list(map(float,next(lines).split())) for _ in range(22)]))
        assert np.linalg.norm(matrices[0]-matrices[1])/max(1,np.linalg.norm(matrices[0]))<1e-11
        if meta[index]['pilot_case']['kind']!='evolved_native_t0.2':continue
        assert family=='0'
        trace=np.array([inverse[0,0],2*inverse[0,1],2*inverse[0,2],inverse[1,1],2*inverse[1,2],inverse[2,2]])
        cq=np.zeros((1,11));cq[0,1:7]=trace;cp=np.zeros((1,11));cp[0,:6]=trace
        qb=np.linalg.svd(cq,full_matrices=True)[2][1:].T;pb=np.linalg.svd(cp,full_matrices=True)[2][1:].T
        basis=np.zeros((22,20));basis[:11,:10]=qb;basis[11:,10:]=pb
        yield index,dict(radius=float(radius),omega=float(omega),normal=normal,source_case=meta[index]['pilot_case']),mm(mm(basis.T,matrices[0]),basis),mm(C,basis),float(lo),float(hi)

def constraints(q,p,D,C):
    dq=mm(D,q);ddq=mm(D,dq);dp=mm(D,p)
    return np.column_stack([np.einsum('nj,j->n',ddq,C[0,:10]),
        mm(dp,C[1:4,10:].T),mm(dq,C[4:,:10].T)+mm(p,C[4:,10:].T)])
def rms(x,h):return np.sqrt(np.einsum('n,nj,nj->j',h,x,x)/np.sum(h))
records=[];consistency=[];negative_max=0;max_initial=0
for index,case,M,C,lo,hi in symbols():
    beta=-(lo+hi)/2;c=(hi-lo)/2;Ae=M[:10,:10]-beta*np.eye(10);B=M[:10,10:]
    Binv=np.linalg.inv(B);T=np.block([[np.eye(10),np.zeros((10,10))],[Ae,B]])
    canonical=np.linalg.solve(T.T,mm(T,M).T).T
    target=np.block([[beta*np.eye(10),np.eye(10)],[c*c*np.eye(10),beta*np.eye(10)]])
    canonical_error=float(np.linalg.norm(canonical-target)/max(1,np.linalg.norm(target)))
    rownorm=np.linalg.norm(C,axis=1);Cn=C/rownorm[:,None];_,singular,V=np.linalg.svd(Cn,full_matrices=True)
    assert np.count_nonzero(singular>1e-9)==8
    kernel=V[8:].T;coefficient_leak=float(np.max(np.abs(mm(Cn,kernel))));bygeometry={}
    for op in operators:
        n=op['nodes'];h=np.array(op['H']);D=np.array(op['D']);Q=np.array(op['Q']);x=(np.arange(n)+.5)/n
        q=np.sin(np.arange(n*10)*.17).reshape(n,10);p0=np.cos(np.arange(n*10)*.11).reshape(n,10)
        Pi=mm(mm(D,q),Ae.T)+mm(p0,B.T);desired=mm(Q,Pi);ppen=mm(desired,Binv.T);recovered=mm(ppen,B.T)
        lift_error=float(np.linalg.norm(recovered-desired)/max(1,np.linalg.norm(desired)))
        direct=float(2*np.einsum('n,ni,ni->',h,Pi,recovered))
        S=h[:,None]*Q+Q.T*h[None,:];expected=float(np.einsum('ni,nm,mi->',Pi,S,Pi))
        energy_error=abs(direct-expected)/max(1,abs(direct),abs(expected))
        largest=float(np.linalg.eigvalsh((S+S.T)/2)[-1]/max(1,np.linalg.norm(S)))
        naive=mm(mm(Q,p0),B.T)
        negative=float(np.linalg.norm(naive-desired)/max(1,np.linalg.norm(desired)));negative_max=max(negative_max,negative)
        # Actual q penalty is zero: the derivative variable is never reset.
        qpen=np.zeros_like(q);reduction=float(np.max(np.abs(mm(D,qpen))))
        penalties=[];raw_injections=[];conditioned_injections=[];initial=[];zero_coefficients=[]
        for mode in range(12):
            v=kernel[:,mode];scale=math.e*max(np.max(np.abs(v[:10])),np.max(np.abs(v[10:])))
            f=np.exp(x)/scale;dqf=np.einsum('ij,j->i',D,f)
            qq=f[:,None]*v[:10];pp=dqf[:,None]*v[10:]
            initial.append(rms(constraints(qq,pp,D,C),h))
            pi=mm(mm(D,qq),Ae.T)+mm(pp,B.T);pen=mm(mm(Q,pi),Binv.T)
            penalties.append(rms(pen,h));raw_injections.append(rms(constraints(np.zeros_like(qq),pen,D,C),h))
            conditioned_injections.append(rms(constraints(np.zeros_like(qq),pen,D,Cn),h))
            # Frozen analytic coefficient oracle separates exact zero sectors
            # from small measured values without dropping their raw samples.
            pc=np.einsum('ij,j->i',Binv,np.einsum('ij,j->i',Ae,v[:10])+np.einsum('ij,j->i',B,v[10:]))
            zero_coefficients.append(np.r_[0,np.einsum('ij,j->i',Cn[1:4,10:],pc),np.einsum('ij,j->i',Cn[4:,10:],pc)])
        initial=np.array(initial);max_initial=max(max_initial,float(initial.max()))
        raw_injections=np.array(raw_injections);conditioned_injections=np.array(conditioned_injections);zero_coefficients=np.array(zero_coefficients)
        # Aggregate each physical row over all12 constraint-free normal modes.
        rawmax=raw_injections.max(axis=0);conditionedmax=conditioned_injections.max(axis=0)
        exactzero=np.max(np.abs(zero_coefficients),axis=0)<1e-12
        test={'case':index,**case,'nodes':n,'stretch':op['stretch'],'canonical_matrix_error':canonical_error,
            'B_condition':float(np.linalg.cond(B)),'kernel_coefficient_leak':coefficient_leak,
            'lift_error':lift_error,'energy_identity_error':energy_error,'dissipative_energy_rate':direct,
            'normalized_max_dissipation_energy_eigenvalue':largest,'configuration_and_reduction_penalty':reduction,
            'omitted_cross_term_relative_defect':negative,'initial_raw_constraints_rms_max':float(initial.max()),
            'momentum_penalty_rms_max':float(np.max(penalties)),'constraint_injection_raw_rms':rawmax.tolist(),
            'constraint_injection_conditioned_rms':conditionedmax.tolist(),'algebraically_zero_constraint_rows':exactzero.tolist(),
            'algebra_energy_pass':bool(canonical_error<=1e-10 and lift_error<=1e-10 and energy_error<=1e-10 and largest<=1e-10 and reduction==0)}
        records.append(test);bygeometry.setdefault(op['stretch'],[]).append(test)
    for stretch,runs in bygeometry.items():
        runs.sort(key=lambda r:r['nodes']);assert [r['nodes'] for r in runs]==[24,48,96]
        values=[r['momentum_penalty_rms_max'] for r in runs];order=math.log2(values[1]/values[2])
        checks=[{'row':'momentum_penalty','values':values,'fine_order':order,'passed':values[0]>values[1]>values[2] and order>=1.5}]
        for row,name in enumerate(['H','Mx','My','Mz','Theta','Zx','Zy','Zz']):
            values=[r['constraint_injection_raw_rms'][row] for r in runs]
            exactzero=all(r['algebraically_zero_constraint_rows'][row] for r in runs)
            order=None if exactzero else math.log2(values[1]/values[2])
            checks.append({'row':name,'values':values,'algebraically_zero':exactzero,'fine_order':order,
                'passed':bool(exactzero or (values[0]>values[1]>values[2] and order>=1.5))})
        consistency.append({'case':index,'stretch':stretch,'checks':checks,'passed':all(r['passed'] for r in checks)})
report={'scope':__doc__,'states':len(set(r['case'] for r in records)),'algebra_energy_pass':all(r['algebra_energy_pass'] for r in records),
    'constraint_consistency_pass':all(r['passed'] for r in consistency),'omitted_cross_term_detected':negative_max>=1e-3,
    'maximum_initial_raw_constraint_rms':max_initial,'records':records,'consistency':consistency,
    'native_or_live_adoption_admitted':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [raw,matrixfile]}}
(out/('tapered-review.json' if a.tapered else 'review.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['scope','records','consistency','input_sha256']},indent=2))
print('failed consistency',[(r['case'],r['stretch'],[(c['row'],c['fine_order']) for c in r['checks'] if not c['passed']]) for r in consistency if not r['passed']])
