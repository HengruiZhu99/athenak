"""Bounded exact-oracle pilots of actual native radial8 linear evolution.

No additional evolved fields. This runner preserves failed finite-boundary
spectral criteria and does not claim nonlinear or mapped3D exact-scri admission.
"""
from pathlib import Path
import sys,json,math,time,hashlib
import numpy as np
import mpmath as mp
import goal1_native_constraint_outflow as candidate
import goal1_native_constraint_finite as finite
import goal1_native_outgoing_gauge_modes as oracle

R=Path(__file__).resolve().parents[2]
P=R/'docs/validation/goal1-minkowski-20261018/native-constraint-boundary-v1'
W=R/'build-goal1-minkowski-20261018/native-constraint-boundary-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def targets(model,kind):
 cache=W/f'oracle-{kind}-N{model.N}.npz';receipt=cache.with_suffix('.json')
 if cache.exists():
  j=json.loads(receipt.read_text());assert sha(cache)==j['sha256'];d=np.load(cache);assert np.array_equal(d['Omega'],model.o);return d['jets'],j
 mp.mp.dps=100;out=np.zeros((3,model.M,8));checks=[]
 for i,o in enumerate(model.o[1:],1):
  x=mp.mpf(float(o));J=oracle.jets(x,mp.mpf(1),kind);F,C=finite.matrices(x);src=sum([F[k]*J[k]for k in range(3)],mp.matrix(8,1));con=sum([C[k]*J[k]for k in range(3)],mp.matrix(4,1));err=mp.norm(src-J[0],p=mp.inf)/max(1,mp.norm(J[0],p=mp.inf));cerr=mp.norm(con,p=mp.inf);checks.append({'Omega':float(o),'source_identity_error':float(err),'raw_weighted_constraints_error':float(cerr)})
  out[:,i]=np.array([list(v)for v in J],float)
 extrap=[]
 for delta in ['1e-8','1e-9']:
  J=[oracle.jets(mp.mpf(delta)*i,mp.mpf(1),kind)for i in range(1,5)];extrap.append(np.array([list(sum([c*j[k]for c,j in zip([4,-6,4,-1],J)],mp.matrix(8,1)))for k in range(3)],float))
 out[:,0]=extrap[-1];known=np.array([8/3,-16/3,8,-16,0,-24,0,0]);leading_error=float(abs(out[0,0]-known).max());out[0,0]=known
 uncertainty=float(abs(extrap[1]-extrap[0]).max());assert max(q['source_identity_error']for q in checks)<1e-45;assert max(q['raw_weighted_constraints_error']for q in checks)<1e-45;assert leading_error<1e-12
 np.savez_compressed(cache,jets=out,Omega=model.o);j={'kind':kind,'lambda':1,'N':model.N,'precision':100,'source':sha(Path(oracle.__file__)),'sha256':sha(cache),'endpoint_jet_delta_comparison_absolute':uncertainty,'endpoint_known_leading_error':leading_error,'finite_source_and_constraint_checks':checks};receipt.write_text(json.dumps(j,indent=2)+'\n');return out,j

def run(kind,N,factor):
 label=f'physical-outflow-{kind}-N{N}-dt{factor}';case=W/label;assert not case.exists(),'Do not replay a compatible trajectory';case.mkdir();model=candidate.Model(N,constraint_frame='physical',gauge='metric');J,oj=targets(model,kind);eps=1e-6;A=32*eps;rho=float(abs(np.linalg.eigvals(model.A)).max());cycles=math.ceil(.001*rho/.5)*factor;dt=.001/cycles
 # All active/imported numerical research modules are pinned before evolution.
 sources={str(Path(v.__file__).resolve().relative_to(R)):sha(v.__file__)for k,v in list(sys.modules.items())if k.startswith('goal1_native_')and hasattr(v,'__file__')};sources[str(Path(__file__).relative_to(R))]=sha(__file__)
 sig={'kind':'native8-physical-constraint-metric-gauge-CK03','profile':kind,'N':N,'H':model.H,'dt':dt,'factor':factor,'epsilon':eps,'amplitude_normalization':A,'native_free_DOF':model.dim,'source_sha256':sources,'Jacobian_sha256':hashlib.sha256(model.A.tobytes()).hexdigest(),'boundary_data':'exact nonzero entering retarded gauge; physical constraints zero','oracle_sha256':oj['sha256'],'new_evolved_fields':0,'scientific_admission':False};(case/'descriptor.json').write_text(json.dumps(sig,indent=2)+'\n')
 for name,h in sources.items():
  dst=P/f'executed-source-{h[:12]}-{Path(name).name}'
  if not dst.exists():dst.write_bytes((R/name).read_bytes())
 bc0=model.data_from_jets(model.H,*J[:,-1]);bc0[:2]=0
 inflow=lambda t:eps*math.exp(t)*bc0
 z=model.pack(eps*J[0]);t=0.;cycle=0;history=[];stage_min=np.full(4,np.inf)
 edges=np.r_[0,(model.o[:-1]+model.o[1:])/2,model.H];mass=2/3*((1-edges[:-1])**1.5-(1-edges[1:])**1.5)
 regions={'all':np.ones(model.M,bool),'interior':(model.o>0)&(model.o<model.H),'scri':model.o==0,'collar8':np.arange(model.M)<9,'inner3':np.arange(model.M)>=model.M-3}
 C=[None]+[np.array([M.tolist()for M in finite.matrices(mp.mpf(float(o)))[1]],float)for o in model.o[1:]]
 def norm(v,mask):
  val=np.asarray(v)[mask];w=mass[mask];sq=np.sum(val*val,axis=1)if val.ndim>1 else val*val;return {'rms':float(np.sqrt(np.einsum('i,i->',w,sq)/w.sum())),'linf':float(np.sqrt(sq.max()))}
 def check_stage(z,at):
  u=model.expand(z,inflow(at));mins=np.array([(1+u[:,0]).min(),(1+u[:,1]).min(),(1-u[:,1]/2).min(),(2-model.o+u[:,2]).min()]);stage_min[:]=np.minimum(stage_min,mins);assert np.isfinite(u).all()and(mins>0).all()
 def observe(initial=False):
  u=model.expand(z,inflow(t));flat=u.reshape(-1);d=np.einsum('ij,j->i',model.first_operator,flat).reshape(model.M,8);dd=np.einsum('ij,j->i',model.second_operator,flat).reshape(model.M,8);exact=eps*math.exp(t)*J;err=u-exact[0];de=d-exact[1];dde=dd-exact[2];dr=-2*model.r[:,None]*de;drr=4*model.r[:,None]**2*dde-2*de
  arrays={'fields':err/A,'first_Omega':.08*de/A,'config_Hessian_Omega':.08**2*dde[:,:4]/A,'first_radial':.04*dr/A,'config_Hessian_radial':.04**2*drr[:,:4]/A};errors={r:{k:norm(v,mask)for k,v in arrays.items()}for r,mask in regions.items()}
  rawerr=err[1:].copy();rawerr[:,[4,7]]/=model.o[1:,None];rawscale=rawerr/A
  V=np.array([np.einsum('kij,kj->i',C[i],np.array([u[i],d[i],dd[i]]))for i in range(1,model.M)]);raw=V.copy();raw[:,[1,3]]/=model.o[1:,None];cnorm=raw*np.array([.04**2,.04**2,.04,.04])/A
  # Independent whole native RHS compatibility at algebraically dependent boundary and endpoint components.
  rate=np.einsum('ij,j->i',model.expand_matrix,model.rhs(z,inflow(t)))+np.einsum('ij,j->i',model.boundary_full,inflow(t));full=np.einsum('ij,j->i',model.full_operator,flat);f0=np.einsum('ij,j->i',model.endpoint_full_operator,flat);defect=rate[-4:]-full[-4:];endpoint=rate[:8]-f0
  report={'time':t,'cycle':cycle,'errors':errors,'all_regional_amplitude_budgets_pass':all(q['rms']<=.01 and q['linf']<=.05 for row in errors.values()for q in row.values()),'raw_native_positive_field_error':norm(rawscale,np.ones(model.M-1,bool))if False else {'rms':float(np.sqrt(np.einsum('i,ij,ij->',mass[1:],rawscale,rawscale)/mass[1:].sum())),'linf':float(np.linalg.norm(rawscale,axis=1).max())},'constraints_raw_positive_max':np.max(abs(raw),axis=0).tolist(),'constraints_normalized_positive_max':np.max(abs(cnorm),axis=0).tolist(),'natural_constraints_positive_max':np.max(abs(V/model.o[1:,None]**2),axis=0).tolist(),'native_inner_p_rate_defect_over_A':float(abs(defect).max()/A),'native_endpoint_rate_defect_over_A':float(abs(endpoint).max()/A),'algebraic_boundary_residual':float(abs(np.einsum('ij,j->i',model.boundary_operator,flat)-inflow(t)).max()),'stage_min_chi_grr_gtt_alpha':stage_min.tolist(),'endpoint_native_leading':u[0].tolist(),'general_endpoint_constraint_limits_admitted':False,'full_nonlinear_or_mapped3D_admission':False}
  fname='initial.npz'if initial else'last.npz';np.savez_compressed(case/fname,z=z,u=u,d=d,dd=dd,Omega=model.o,time=t,cycle=cycle);history.append(report);(case/'history.json').write_text(json.dumps(history,indent=2)+'\n');return report
 def checkpoint():
  with(case/'checkpoint.tmp').open('wb')as f:np.savez_compressed(f,z=z,Omega=model.o,time=t,cycle=cycle,descriptor=np.array(json.dumps(sig)))
  (case/'checkpoint.tmp').replace(case/'checkpoint.npz')
 receipt={'label':label,'descriptor':sig,'protocol_sha256':sha(P/'exact-gauge-pilot-protocol-v10.json'),'restart':None,'rho':rho,'dt_rho':dt*rho,'cycles':cycles,'target':.001,'oracle':oj};started=time.time()
 try:
  check_stage(z,0);observe(True);checkpoint()
  for k in range(cycles):
   a=z+dt*model.rhs(z,inflow(t));check_stage(a,t+dt);b=.75*z+.25*(a+dt*model.rhs(a,inflow(t+dt)));check_stage(b,t+dt/2);z=z/3+2/3*(b+dt*model.rhs(b,inflow(t+dt/2)));cycle=k+1;t=cycle*dt;check_stage(z,t)
   if cycle%32==0 or cycle==cycles:observe();checkpoint()
  receipt.update(exit=0,seconds=time.time()-started,final_report=history[-1])
 except Exception as exc:
  checkpoint();receipt.update(exit=1,seconds=time.time()-started,error=repr(exc),last_cycle=cycle,last_time=t)
 receipt['outputs']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size}for p in case.iterdir()if p.is_file()and p.name!='receipt.json'};(case/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(P/(label+'-run.json')).write_text(json.dumps(receipt,indent=2)+'\n');print({'label':label,'exit':receipt['exit'],'cycles':cycle,'time':t,'final':receipt.get('final_report',{}).get('errors',{}).get('all',{}),'boundary_rate_defect':receipt.get('final_report',{}).get('native_inner_p_rate_defect_over_A')},flush=True);return receipt['exit']
if __name__=='__main__':sys.exit(run(sys.argv[1],int(sys.argv[2]),int(sys.argv[3])))
