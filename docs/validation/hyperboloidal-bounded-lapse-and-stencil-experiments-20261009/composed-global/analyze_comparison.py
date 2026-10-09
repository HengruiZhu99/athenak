"""Read frozen original states; apply fresh common diagnostics without rerunning old sources."""
from pathlib import Path
import hashlib,json,struct,subprocess,time
import numpy as np
from scipy.sparse import load_npz
w=Path(__file__).resolve().parent;old=w.parents[1]/'boundary/full-tensor-propagator/full22-v2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();newfile=w/'spatialnorm-projected-krylov-m50-80-h0.1-t2.0.npz';oldfile=old/'spatialnorm-projected-expm-t2.0.npz';new=np.load(newfile);baseline=np.load(oldfile);meta=json.loads((w/'spatialnorm-cache0.0001-metadata.json').read_text());N=meta['points'];coords=np.asarray(meta['xyz_omega_volume_ginv_chi']);r=np.linalg.norm(coords[:,:3],axis=1);weights=meta['spacing']**3*coords[:,4];G=np.zeros((N,3,3));ix=[0,0,0,1,1,2];iy=[0,1,2,1,2,2]
for k,(i,j) in enumerate(zip(ix,iy)):G[:,i,j]=G[:,j,i]=coords[:,5]*coords[:,6+k]
L=np.fromfile(w/'spatialnorm-cache0.0001-lift.bin',dtype='<f8').reshape(N,22,20);assert np.allclose(new['times'],baseline['times'],rtol=0,atol=1e-14);assert new['names'].tolist()==baseline['names'].tolist();assert np.max(abs(new['values'][0]-baseline['values'][0]))<1e-13
f=(w/'comparison-diagnostics.stderr').open('w');p=subprocess.Popen([str(w/'constraint-server')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=f);assert json.loads(p.stdout.readline())['points']==N
def apply(v,mode,eps=1e-5):
 p.stdin.write(mode.encode()+struct.pack('dd',eps,0)+np.asarray(v,dtype=np.float64).tobytes());p.stdin.flush();n=N*(2 if mode=='e' else 7);b=bytearray()
 while len(b)<n*8:
  x=p.stdout.read(n*8-len(b))
  if not x:raise RuntimeError('diagnostic server terminated')
  b.extend(x)
 return np.frombuffer(b,dtype=np.float64).copy().reshape(N,-1)
def parts(c):return np.column_stack([c[:,0]**2,np.einsum('pi,pij,pj->p',c[:,1:4],G,c[:,1:4]),np.einsum('pi,pij,pj->p',c[:,4:7],G,c[:,4:7])])
def row(v,t):
 c=apply(v,'d');s=apply(v,'c');ns=apply(v,'e');cp=parts(c);sp=parts(s);full=np.einsum('pij,pj->pi',L,v.reshape(N,20));field2=(full*full).sum(axis=1)*weights;total=field2.sum();return {'time':float(t),'matching_composed_H_M_Z_rms':np.sqrt(cp.mean(axis=0)).tolist(),'standard_H_M_Z_rms':np.sqrt(sp.mean(axis=0)).tolist(),'Theta_phys_rms':float(np.sqrt(np.mean(v.reshape(N,20)[:,15]**2))),'euclidean_component_l2':float(np.linalg.norm(v)),'reference_volume_component_l2':float(np.sqrt(np.sum(weights*ns[:,0]))),'reference_volume_component_H1':float(np.sqrt(np.sum(weights*ns.sum(axis=1)))),'outer_r09_squared_component_fraction':float(field2[r>=.9].sum()/total) if total else 0,'outer_r09_squared_constraints_fraction':[float(q[r>=.9].sum()/q.sum()) if q.sum() else 0 for q in cp.T],'M_Z_standard_vs_composed_max_difference':float(abs(c[:,1:]-s[:,1:]).max())},c,s
out={'semantics':'N16/span2.2 finiteΩ exploratory exp(t P_ref J22 Lift); fresh actual radius4 composed RHS and matching composed Hamiltonian; M/Z first derivatives and physicalTheta unchanged. Original baseline states/diagnostics are read-only frozen inputs, never rerun. Common-diagnostic comparison reapplies fresh ng4 continuation to both saved trajectories; original diagnostics retained separately. No native finite-pulse, continuum stability, nonlinear closure or production adoption claim.','inputs':{str(x):sha(x) for x in [newfile,oldfile,old/'spatialnorm-projected-expm-analysis.json',w/'spatialnorm-projected-J20.npz',w/'spatialnorm-cache0.0001-metadata.json',w/'constraint-server',Path(__file__)]},'outward_reference_crossing_time':.7457643839234269,'comparison':[],'initial_gauge_constraint_source':[]};olds=json.loads((old/'spatialnorm-projected-expm-analysis.json').read_text());t0=time.monotonic();Cs={};count=0
for col,name in enumerate(new['names'].tolist()):
 histories={};
 for tag,data in [('composed',new),('original_states_common_ng4_diagnostic',baseline)]:
  hist=[];ds=[];ss=[]
  for it,t in enumerate(data['times']):
   rr,d,s=row(data['values'][it,:,col],t);hist.append(rr);ds.append(d);ss.append(s);count+=1
  histories[tag]=hist;Cs[tag+'_'+name+'_composed']=np.asarray(ds);Cs[tag+'_'+name+'_standard']=np.asarray(ss)
 oldhist=next(x['history'] for x in olds['histories'] if x['name']==name);newlast=histories['composed'][-1];oldlast=oldhist[-1];commonlast=histories['original_states_common_ng4_diagnostic'][-1]
 ratio=lambda a,b:[float(x/y) if y else None for x,y in zip(a,b)]
 report={'name':name,'histories':histories,'frozen_original_native_H_M_Z_rms':oldlast['native_H_M_Z_rms'],'candidate_matching_vs_original_native_H_M_Z_ratio':ratio(newlast['matching_composed_H_M_Z_rms'],oldlast['native_H_M_Z_rms']),'candidate_vs_original_common_composed_H_M_Z_ratio':ratio(newlast['matching_composed_H_M_Z_rms'],commonlast['matching_composed_H_M_Z_rms']),'candidate_standard_vs_original_native_H_M_Z_ratio':ratio(newlast['standard_H_M_Z_rms'],oldlast['native_H_M_Z_rms']),'candidate_vs_original_Euclidean_component_ratio':newlast['euclidean_component_l2']/oldlast['euclidean_component_l2'],'candidate_vs_original_reference_component_H1_ratio':newlast['reference_volume_component_H1']/oldlast['reference_volume_component_H1'],'original_common_standard_vs_frozen_native_final_H_M_Z_relative_error':ratio(np.asarray(commonlast['standard_H_M_Z_rms'])-oldlast['native_H_M_Z_rms'],oldlast['native_H_M_Z_rms'])};out['comparison'].append(report);print(name,'final',json.dumps({k:v for k,v in report.items() if k!='histories'}),flush=True)
# Initial pure gauge source is zero initially, but C_h J_h v need not vanish on variable coefficients.
v=new['values'][0,:,0]
for tag,matrix in [('composed',w/'spatialnorm-projected-J20.npz'),('original',old/'spatialnorm-projected-J20.npz')]:
 J=load_npz(matrix);jv=J@v;rr,_,_=row(jv,0);rr['operator']=tag;rr['matrix_sha256']=sha(matrix);out['initial_gauge_constraint_source'].append(rr)
canonical=np.load(w/'spatialnorm-canonical-short.npz');a=new['values'][:len(canonical['times'])];b=canonical['values'];out['short_canonical_vs_Arnoldi_relative_l2']=[float(np.linalg.norm(a[it,:,col]-b[it,:,col])/np.linalg.norm(b[it,:,col])) for it in range(len(b)) for col in range(2)];out['max_short_canonical_difference']=max(out['short_canonical_vs_Arnoldi_relative_l2']);out['M_Z_callbacks_exact_equal']=max(x['M_Z_standard_vs_composed_max_difference'] for c in out['comparison'] for h in c['histories'].values() for x in h)==0;out['seconds']=time.monotonic()-t0;p.stdin.close();p.wait();f.close();out['server_returncode']=p.returncode;out['all_diagnostics_finite']=all(np.isfinite(x).all() for x in Cs.values());np.savez_compressed(w/'comparison-constraint-arrays.npz',times=new['times'],**Cs);(w/'comparison-report.json').write_text(json.dumps(out,indent=2)+'\n');print('COMPLETE',out['seconds'],out['max_short_canonical_difference'],flush=True);assert out['all_diagnostics_finite'] and out['M_Z_callbacks_exact_equal'] and p.returncode==0 and out['max_short_canonical_difference']<1e-9
