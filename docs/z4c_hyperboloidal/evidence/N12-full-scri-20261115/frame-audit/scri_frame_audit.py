"""Read-only decomposition of an existing native scri extraction."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=10);a=p.parse_args()
sys.path.insert(0,str(a.source/'vis/python'))
import scri_frame as frame
from scri_geometry import spacetime_derivatives,derivative_weights,ricci
from scri_news import stencil
from scri_angular import spin_basis,scalar_basis,weighted_projection
record=json.loads((a.cache/'records.json').read_text());times=np.array([r['time'] for r in record]);grid=np.load(a.cache/'grid.npz');x=grid['x'];w=grid['weights'];archive=np.load(a.cache/'metric.npy',mmap_mode='r')
result=np.load(a.result/'news.npz');states=result['frame_coefficients'];assert np.array_equal(times,result['frame_times'])
summary=json.loads((a.result/'summary.json').read_text());mass=summary['mass_M0'];lmax=summary['angular_lmax'];transport=frame.FrameTransport(x,w,lmax)
original=frame.conformal_ricci_projection;captured={}
def split(R,G,m,grad,hess,factor):
 d=m/factor[:,None];cov=hess-np.einsum('qcab,qc->qab',G,grad)
 terms=np.column_stack((-np.einsum('qa,qb,qab->q',d,d,R),2*np.einsum('qa,qb,qab->q',d,d,cov),-2*np.einsum('qa,qa->q',d,grad)**2))
 captured.update(terms=terms,dyad=d,cov=cov)
 answer=original(R,G,m,grad,hess,factor)
 np.testing.assert_allclose(terms.sum(axis=1),-answer,rtol=2e-12,atol=2e-13)
 return answer
frame.conformal_ricci_projection=split
mu=x[:,0];sn=np.sqrt(1-mu*mu);e=(x*mu[:,None]-np.array([1.,0,0]))/sn[:,None];f=np.column_stack((np.zeros(len(x)),-x[:,2],x[:,1]))/sn[:,None]
native_spin,labels=spin_basis(x,lmax);mode20=labels.index((2,0));native_project=weighted_projection(native_spin,w)
rows=[];allmodes=[];sample=[]
selected=sorted(set(range(3,len(times)-3,a.stride))|{int(np.argmin(abs(times/mass-t))) for t in [1,5,10,20,30,40,50,59]} )
for q in selected:
 if not 3<=q<len(times)-3:continue
 window,center=stencil(times,q);g,dg,ddg=spacetime_derivatives(archive[window],times[window],center);R,G,inv=ricci(g,dg,ddg)
 flow,source,nt,H=frame.generator_geometry(x,inv,G)
 first=np.einsum('t,tkf->kf',derivative_weights(times[window],times[q],1),states[window]);second=np.einsum('t,tkf->kf',derivative_weights(times[window],times[q],2),states[window])
 observed=transport.observe(states[q],g,R,G,flow,source,nt,second,first)
 spin,_=spin_basis(observed['direction'],lmax);project=weighted_projection(spin,observed['angular_weights']);terms=captured['terms'];coeff=project@terms;total=coeff.sum(axis=1)
 np.testing.assert_allclose(total,result['native_cut_news_coefficients'][q-3],rtol=2e-10,atol=3e-12)
 gamma=g[:,1:,1:];ee=np.einsum('qi,qij,qj->q',e,gamma,e);ef=np.einsum('qi,qij,qj->q',e,gamma,f);ff=np.einsum('qi,qij,qj->q',f,gamma,f)
 native_d=np.column_stack((np.zeros(len(x)),(e/np.sqrt(ee)[:,None]-1j*(f-(ef/ee)[:,None]*e)/np.sqrt(ff-ef*ef/ee)[:,None])/np.sqrt(2)))
 native=-np.einsum('qa,qb,qab->q',native_d,native_d,R)
 Htf=H-g*(nt*source)[:,None,None]
 screen_trace=2*np.einsum('qa,qb,qab->q',native_d,native_d.conj(),H).real
 scale_rate=screen_trace/nt-2*source
 vals=transport.y@states[q];mean_lambda=np.sum(w*vals[:,0])/sum(w)
 n=np.column_stack((nt,nt[:,None]*flow));d=captured['dyad'];factor=observed['factor']
 # Adding k*Omega to log(omega) changes no boundary frame values, but changes
 # the formal Ricci projection by 2*k*H_ab*m'^a*m'^b if regularity fails.
 extension=2*np.einsum('qa,qb,qab->q',d,d,H)
 # Trace-free Hessian transforms homogeneously at scri: H'_TF=omega H_TF.
 error_norm=float(np.max(abs(Htf)))
 weights=observed['angular_weights']
 rms=lambda z:float(np.sqrt(np.sum(weights*abs(z)**2)/sum(weights)))
 row=dict(cycle=record[q]['cycle'],native_time_M0=float(times[q]/mass),formal_N20_real=float(total[mode20].real),formal_N20_imag=float(total[mode20].imag),ricci_N20_real=float(coeff[mode20,0].real),hessian_N20_real=float(coeff[mode20,1].real),quadratic_N20_real=float(coeff[mode20,2].real),native_ricci_N20_real=float((native_project@native)[mode20].real),raw_rms=rms(terms[:,0]),correction_rms=rms(terms[:,1]+terms[:,2]),total_rms=rms(terms.sum(axis=1)),omega_min=float(min(factor)),omega_max=float(max(factor)),lambda_mean=float(mean_lambda),lambda_angular_max=float(max(abs(vals[:,0]-mean_lambda))),angular_label_displacement_max=float(max(np.linalg.norm(observed['direction']-x,axis=1))),clock_angular_range_M0=float(np.ptp(observed['bondi_time'])/mass),conformal_hessian_tf_max=error_norm,scale_rate_mean=float(np.sum(w*scale_rate)/sum(w)),unit_normal_extension_N20=float((project@extension)[mode20].real),unit_normal_extension_max=float(max(abs(extension))),**observed['diagnostics'])
 rows.append(row);allmodes.append(coeff);sample.append(q)
a.output.mkdir(parents=True,exist_ok=False)
np.savez(a.output/'decomposition.npz',sample_indices=sample,native_times=times[sample],ell_m=labels,modes_by_term=np.array(allmodes),terms=np.array(['minus_Ricci','plus_2_covariant_hessian_lambda','minus_2_gradient_lambda_squared']))
report=dict(kind='Same-dyad conformal Ricci decomposition; all curves unvalidated radiation diagnostics',baseline_result=str(a.result),baseline_sha256=hashlib.sha256((a.result/'news.npz').read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),samples=rows)
(a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
for t in [1,5,10,20,30,40,50,59]:
 r=min(rows,key=lambda r:abs(r['native_time_M0']-t));print(json.dumps(r),flush=True)
