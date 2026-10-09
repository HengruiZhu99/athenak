from pathlib import Path
import importlib.util
import json
import numpy as np
p=Path(__file__).resolve().parent
raw=json.loads((p/'fourier-mode-matrices.json').read_text());keys=('wide','r','kappa','candidate','oblique','k')
groups={}
for s in raw:groups.setdefault(tuple(s[k] for k in keys),[]).append(s)
report={'scope':'Frozen local Fourier roots and local linearized constraint residues at one point. Im(lambda)/k is the phase-generator speed; physical coordinate propagation has the opposite sign. Pointwise residues do not prove a global constraint subspace or PDE instability.',
        'constraint_order':['H_phys','M_cov_x','M_cov_y','M_cov_z','Z_cov_x','Z_cov_y','Z_cov_z','Theta_phys','null_residue'],
        'mode_normalization':'max absolute independent stored-field component equals one','cases':[]}
maxerr=0;maxqerr=0;maxrooterr=0
for key,rows in groups.items():
    rows.sort(key=lambda s:s['eps'],reverse=True);lo,hi=rows
    m0=np.asarray(lo['B'])+1j*np.asarray(lo['C']);m=np.asarray(hi['B'])+1j*np.asarray(hi['C'])
    q0=np.asarray(lo['HB'])+1j*np.asarray(lo['HC']);q=np.asarray(hi['HB'])+1j*np.asarray(hi['HC'])
    err=float(np.max(abs(m-m0)/(1+abs(m))));qe=float(np.max(abs(q-q0)/(1+abs(q))))
    maxerr=max(maxerr,err);maxqerr=max(maxqerr,qe);assert err<2e-6 and qe<2e-6,(key,err,qe)
    roots,vectors=np.linalg.eig(m);ix=int(np.argmax(roots.real));lam=roots[ix];v=vectors[:,ix];v/=np.max(abs(v))
    residue=q@v;conev=np.array([hi['beta_n']-hi['light_speed'],hi['beta_n'],hi['beta_n']+hi['light_speed']])
    phase=float(lam.imag/hi['k']);nearest=int(np.argmin(abs(conev-phase)))
    old=max(np.linalg.eigvals(m0).real);rooterr=abs(float(lam.real-old));maxrooterr=max(maxrooterr,rooterr)
    assert rooterr<2e-4,(key,rooterr)
    report['cases'].append(dict(zip(keys,key))|{'root_real':float(lam.real),'root_imag':float(lam.imag),'phase_generator_speed':phase,
        'physical_coordinate_propagation_speed':-phase,'beta_n':hi['beta_n'],'light_speed':hi['light_speed'],'principal_phase_speeds':conev.tolist(),
        'nearest_principal_branch':['fast_outgoing','advective','slow_incoming'][nearest],'phase_distance':float(abs(conev[nearest]-phase)),
        'constraint_residues':[{'real':float(z.real),'imag':float(z.imag),'abs':float(abs(z))} for z in residue],
        'mode_real':v.real.tolist(),'mode_imag':v.imag.tolist(),'root_epsilon_difference':rooterr})
report['max_operator_epsilon_scaled_change']=maxerr;report['max_constraint_epsilon_scaled_change']=maxqerr;report['max_root_epsilon_difference']=maxrooterr
# k=0 full assembled operator must tend to the independently derived full pole.
repo=p.parents[3];spec=importlib.util.spec_from_file_location('physical',repo/'tst/hyperboloidal/check_physical_gauge.py');physical=importlib.util.module_from_spec(spec);spec.loader.exec_module(physical)
a=.5;d=1+3*a;scale=np.ones(20);scale[[0,2,3,4,5,6]]=1/a
t=np.eye(20);t[10,7]=.5;t[15,12]=.5;ti=np.linalg.inv(t)
limits=[]
for s in json.loads((p/'fourier-k0-poles.json').read_text()):
    expected=physical.expected_matrix(s['kappa']*a*a,d,True)
    if s['candidate']:
        sigma=5;expected[4,0]=d+3-2*sigma;expected[4,1]=-sigma;expected[4,4]=2-2*sigma;expected[4,7]=sigma
    expected=expected*scale[:,None]/scale[None,:]/(a*a);expected=np.einsum("ij,jk,kl->il",ti,expected,t)
    assert np.isfinite(expected).all()
    error=float(np.max(abs(s['omega']*np.asarray(s['B'])-expected)));limits.append({k:v for k,v in s.items() if k!='B'}|{'pole_error':error})
    if s['r']==.9999999:assert error<2e-5,(s['kappa'],s['candidate'],error)
report['k0_to_fullpole_limits']=limits
(p/'fourier-mode-report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS high-k Fourier/root/constraint FD and independent k0-pole limit',maxerr,maxqerr,maxrooterr)
for wide,r,candidate in [(0,.95,0),(1,.85,1)]:
    for kappa in (5,10):
        print('TARGET wide/r/candidate/kappa',wide,r,candidate,kappa)
        for s in report['cases']:
            if s['wide']==wide and s['r']==r and s['candidate']==candidate and s['kappa']==kappa and not s['oblique']:
                residue=[round(z['abs'],6) for z in s['constraint_residues']]
                print(s['k'],round(s['root_real'],8),round(s['phase_generator_speed'],8),s['nearest_principal_branch'],'|H,Mxyz,Zxyz,Theta,C|',residue)
