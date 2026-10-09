from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
p=Path(__file__).resolve().parent;base=p.parent;repo=p.parents[4]
spec=importlib.util.spec_from_file_location('physical',repo/'tst/hyperboloidal/check_physical_gauge.py');physical=importlib.util.module_from_spec(spec);spec.loader.exec_module(physical)
t=np.eye(20);t[10,7]=.5;t[15,12]=.5;ti=np.linalg.inv(t)
report={'scope':'Scratch actual full20 frozen continuum operator. Stable sampled roots do not prove a global PDE spectrum or nonlinear scri closure. A separate ignored eta10 native overlay is being tested; this Fourier receipt does not certify its finite-duration result.', 'poles':[],'fourier':[]}
maxpole=0
for s in json.loads((p/'shift-poles.json').read_text()):
    a=s['a'];sc=np.ones(20);sc[[0,2,3,4,5,6]]=1/a
    expected=physical.expected_matrix(s['kappa']*a*a,1+3*a,True)*sc[:,None]/sc[None,:]/(a*a)
    for i in range(3):expected[4+i,4+i]=-s['eta']
    expected=np.einsum('ij,jk,kl->il',ti,expected,t)
    m=np.asarray(s['M']);error=float(np.max(abs(m-expected)));maxpole=max(maxpole,error);assert error<2e-8,(s['a'],s['eta'],error)
    ev=np.linalg.eigvals(m);nonzero=ev[abs(ev)>1e-6];assert len(nonzero)==15 and max(nonzero.real)<0
    rank=np.linalg.matrix_rank(m,1e-6);rank2=np.linalg.matrix_rank(np.einsum("ij,jk->ik",m,m),1e-5);assert rank==rank2==15
    report['poles'].append({k:v for k,v in s.items() if k!='M'}|{'error':error,'max_nonzero_real':float(max(nonzero.real)),'semisimple_zero_count':5})
original={}
for name in ('fourier-matrices.json','fourier-mode-matrices.json'):
    for s in json.loads((base/name).read_text()):
        if s['candidate']==0 and s['eps']==5e-7:original[(s['wide'],s['r'],s['kappa'],s['oblique'],s['k'])]=np.asarray(s['B'])+1j*np.asarray(s['C'])
raw=json.loads((p/'shift-fourier.json').read_text());keys=('wide','r','kappa','eta','oblique','k');groups={}
for s in raw:groups.setdefault(tuple(s[k] for k in keys),[]).append(s)
maxeps=0;maxqeps=0;maxidentity=0
for key,rows in groups.items():
    rows.sort(key=lambda s:s['eps'],reverse=True);aa,bb=rows
    m0=np.asarray(aa['B'])+1j*np.asarray(aa['C']);m=np.asarray(bb['B'])+1j*np.asarray(bb['C'])
    q0=np.asarray(aa['HB'])+1j*np.asarray(aa['HC']);q=np.asarray(bb['HB'])+1j*np.asarray(bb['HC'])
    err=float(np.max(abs(m-m0)/(1+abs(m))));qe=float(np.max(abs(q-q0)/(1+abs(q))));maxeps=max(maxeps,err);maxqeps=max(maxqeps,qe)
    assert err<2e-6 and qe<2e-6,(key,err,qe)
    # The production cutoff is C-infinity, using an exponential logistic.
    x=max(0.,min(1.,(bb['r']-.45)/(.85-.45)))
    if x==0:W=0.
    elif x==1:W=1.
    else:
        g=-1/x+1/(1-x);e=np.exp(-abs(g));W=e/(1+e) if g<=0 else 1/(1+e)
    zero=original[(key[0],key[1],key[2],key[4],key[5])];expected=zero.copy()
    for i in range(3):expected[4+i,4+i]-=bb['eta']*W/bb['omega']
    error=float(np.max(abs(m-expected)/(1+abs(m))));maxidentity=max(maxidentity,error);assert error<3e-6,(key,error)
    roots,vectors=np.linalg.eig(m);ix=int(np.argmax(roots.real));lam=roots[ix];v=vectors[:,ix];v/=max(abs(v));residue=q@v
    phase=None if bb['k']==0 else float(lam.imag/bb['k'])
    report['fourier'].append(dict(zip(keys,key))|{'max_real':float(lam.real),'root_imag':float(lam.imag),'phase_generator_speed':phase,'beta_n':bb['beta_n'],'light_speed':bb['light_speed'],
        'positive_count':int((roots.real>1e-6).sum()),'constraint_abs':abs(residue).tolist(),'constraint_real':residue.real.tolist(),'constraint_imag':residue.imag.tolist(),'mode_real':v.real.tolist(),'mode_imag':v.imag.tolist()})
report['max_fullpole_error']=maxpole;report['max_operator_epsilon_scaled_change']=maxeps;report['max_constraint_epsilon_scaled_change']=maxqeps;report['max_exact_lower_order_identity_error']=maxidentity
(p/'shift-report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS 40 full20 pole cases and Fourier epsilon/constraint/exact lower-order identity',maxpole,maxeps,maxqeps,maxidentity)
for wide in (0,1):
    for kappa in (5,10):
        for eta in (1,2,5,10,20):
            s=[x for x in report['fourier'] if x['wide']==wide and x['kappa']==kappa and x['eta']==eta]
            worst=max(s,key=lambda x:x['max_real']);outer=max([x for x in s if x['r']>=.85],key=lambda x:x['max_real'])
            print('wide/kappa/eta',wide,kappa,eta,'worst',worst['r'],worst['k'],worst['oblique'],worst['max_real'],'outer',outer['r'],outer['k'],outer['oblique'],outer['max_real'])
receipt=json.loads((p/'receipt.json').read_text());receipt['artifacts_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('*.json') if f.name!='receipt.json'};receipt['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
