from pathlib import Path
import json
import numpy as np
p=Path(__file__).resolve().parent
raw=json.loads((p/'fourier-matrices.json').read_text())
keyfields=('wide','r','kappa','candidate','oblique','k')
groups={}
for s in raw:groups.setdefault(tuple(s[k] for k in keyfields),[]).append(s)
matrices={};maxeps=0;worsteps=None;maxeig40=0
for key,rows in groups.items():
    rows.sort(key=lambda s:s['eps'],reverse=True)
    samples=[np.asarray(s['B'])+1j*np.asarray(s['C']) for s in rows]
    errors=[np.max(abs(a-b)/(1+abs(b))) for a,b in zip(samples[:-1],samples[1:])]
    error=float(max(errors))
    if error>maxeps:maxeps=error;worsteps=key
    assert error<2e-6,(key,error)
    matrices[key]=samples[-1]
    if key[-1]==0:assert np.max(abs(samples[-1].imag))==0
    b,c=samples[-1].real,samples[-1].imag
    real40=np.block([[b,-c],[c,b]])
    e20=np.linalg.eigvals(samples[-1]);e40=np.linalg.eigvals(real40)
    difference=abs(float(max(e40.real))-float(max(e20.real)))
    maxeig40=max(maxeig40,difference)
    assert difference<5e-5,(key,difference)
report={'scope':'Frozen local continuum Fourier spectra about the analytic reference, not global continuum or native grid spectra. The 20 free fields reconstruct gzz/Azz and derivative jets algebraically. Re(L),Im(L) correspond to consistent cos/sin jets; real40=[[B,-C],[C,B]].',
        'cases':[],'max_epsilon_scaled_difference':maxeps,'worst_epsilon_case':worsteps,'max_real40_spectral_abscissa_difference':maxeig40}
maxpoly=0;maxderivativechange=0
for prefix in sorted(set(key[:-1] for key in matrices)):
    zero=matrices[prefix+(0.,)];high=matrices[prefix+(64.,)]
    # Use the largest wave to avoid multiplying k=1 cancellation noise by 4096.
    b2=(high.real-zero.real)/(64.*64.);c1=high.imag/64.
    for k in (0.,1.,2.,4.,8.,16.,32.,64.):
        m=matrices[prefix+(k,)];estimate=zero.real+k*k*b2+1j*k*c1
        error=float(np.max(abs(m-estimate)/(1+abs(m))))
        maxpoly=max(maxpoly,error);assert error<3e-5,(prefix,k,error)
        roots,vectors=np.linalg.eig(m);index=int(np.argmax(roots.real));z=roots[index]
        mode=vectors[:,index];mode/=np.max(abs(mode))
        report['cases'].append(dict(zip(keyfields,prefix+(k,)))|{'max_real':float(z.real),'max_root_imag':float(z.imag),
            'positive_count':int((roots.real>1e-6).sum()),'dominant_mode_real':mode.real.tolist(),'dominant_mode_imag':mode.imag.tolist(),
            'roots':[{'real':float(z.real),'imag':float(z.imag)} for z in roots]})
        if prefix[3]:
            baseprefix=prefix[:3]+(0,)+prefix[4:]
            base=matrices[baseprefix+(k,)];delta=m-base
            deltazero=zero-matrices[baseprefix+(0.,)]
            error=float(np.max(abs(delta-deltazero)/(1+abs(m))))
            maxderivativechange=max(maxderivativechange,error)
            assert error<3e-6,(prefix,k,error)
report['max_wave_polynomial_scaled_error']=maxpoly
report['max_candidate_derivative_scaled_change']=maxderivativechange
(p/'fourier-report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS Fourier FD/phase/real40/polynomial checks',maxeps,maxeig40,maxpoly,maxderivativechange,flush=True)
for wide in (0,1):
    for kappa in (5.,10.):
        for candidate in (0,1):
            cases=[s for s in report['cases'] if s['wide']==wide and s['kappa']==kappa and s['candidate']==candidate]
            worst=max(cases,key=lambda s:s['max_real'])
            print('wide',wide,'kappa',kappa,'candidate',candidate,'worst',worst['r'],worst['k'],worst['oblique'],worst['max_real'],worst['max_root_imag'],flush=True)
