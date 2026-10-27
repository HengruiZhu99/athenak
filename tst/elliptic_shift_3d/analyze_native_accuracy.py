#!/usr/bin/env python3
"""All-cell accuracy budgets for completed frozen-v14 native controls.
Independent reference, unnormalized L2, volume changes, overlapping regions and
actual-node angular scatter are retained; no regional exclusion or order claim.
"""
import argparse,hashlib,json,os,re,sys,tempfile,time
from pathlib import Path
import numpy as np
from compare_coupled_family import FIELDS,interpolate,norms,topology


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--baseline',type=Path,default=Path('docs/elliptic-shift-3d/evidence/native-static8-controls-20261027'));p.add_argument('--reference',type=Path,default=Path('docs/elliptic-shift-3d/evidence/selected-taper-pilot-20261027'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False);start=time.monotonic();hashes={}
 def track(p):hashes[str(p)]=sha(p);return p
 manifest=json.loads(track(a.controls/'manifest.json').read_text());old=json.loads(track(a.baseline/'manifest.json').read_text());ref=np.load(track(a.reference/'radial-n2049.npz'));coarse=np.load(track(a.reference/'radial-n1025.npz'));parameters=json.loads(track(a.reference/'reference-parameters.json').read_text());assert parameters['time']==.0002 and parameters['telegraph_width']==.24 and parameters['reference']=='Minkowski'
 scenarios=[('baseline',a.baseline/'full/native_wormhole.00002.csv',old['native_trace'][-1],6,old)]
 for name,c in manifest['cases'].items():
  if c.get('execution_failed') or 'final_folder' not in c:continue
  scenarios.append((name,a.controls/c['final_folder']/f'native_wormhole.{c["cycles"]:05d}.csv',c['trace'][-1],c['n'],c))
 followup=a.controls/'restart-followup.json';restart_followup=json.loads(track(followup).read_text()) if followup.exists() else None
 summary={'restart_followup':restart_followup,'scope':'Fixed qualified native executable/PDE capsule; short t=.0002 accuracy controls only. Every scientific acceptance remains false.','binary_sha256':manifest['binary_sha256'],'production_archive_sha256':manifest['production_archive_sha256'],'reference_parameters':parameters,'weight_definition':'Each case supplied Penrose proper-volume quadrature map_volume*sqrt(det(g_tilde))/chi^(3/2); all active cells retained. RMS, unnormalized L2, point max and total volume reported. Same-grid controls additionally use baseline weights.','regions':'All includes every active cell. Other diagnostic masks overlap and never replace acceptance. Cartesian inferred spacing tiers are not native AMR levels.','angular_method':'All actual nodal directions and values retained, grouped by round(r,14) only for diagnostic means/scatter; no projection into evolution or interpolated ray claim.','temporal_limit':'Actual native Ketcheson2S RK4 controls compare dt and dt/2. Independent stored spherical profiles use classical RK4; no equivalence assumed at finite dt.','spatial_limit':'N6/8/10 Cartesian AND shell logical N change. Penrose volume and puncture minimum radius change; a uniform order inferred from volume RMS would be misleading. No apparent RMS-order fit.','unavailable':'CSV contains scalar H/M/Theta/Z diagnostic norms; vector momentum/Z component residual differences and regional original elliptic residuals require derivative-complete data, not reconstructed here.','cases':{},'failed_or_incomplete_cases':{k:v for k,v in manifest['cases'].items() if v.get('execution_failed') or v.get('matched_reference_incomplete')}}
 cache={}
 for name,path,trace,n,record in scenarios:
  q=np.genfromtxt(track(path),delimiter=',',names=True);assert len(q)==trace['cells'] and all(np.all(np.isfinite(q[f])) for f in q.dtype.names);xyz=np.column_stack([q[f] for f in 'xyz']);r=np.linalg.norm(xyz,axis=1);assert np.all((r>0)&(r<1));normal=xyz/r[:,None];w=q['weight'];assert np.all(w>0)
  def contract(start,identity=False):
   c=np.column_stack([q[f'state_{start+i}'] for i in range(6)]);v=(1 if identity else 0)+sum(c[:,i]*normal[:,a]*normal[:,b]*(1 if a==b else 2) for i,(a,b) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]));return v
  beta=np.column_stack([q['beta_'+c] for c in 'xyz']);physical={'alpha':q['alpha'],'chi':q['chi'],'g_rr':contract(1,True),'A_rr':contract(7),'Khat':q['state_13']-3,'Theta':q['Theta'],'Lambda_r':sum(q[f'state_{15+i}']*normal[:,i] for i in range(3)),'beta_r':np.sum(beta*normal,axis=1)}
  derived=np.empty(len(q),dtype=list(q.dtype.descr)+[('r','f8'),('M_norm','f8')]);
  for f in q.dtype.names:derived[f]=q[f]
  derived['r']=r;derived['M_norm']=q['M'];masks,mesh=topology(derived,n,[.45,.55,.60,.65,.70,.75,1]);truth={f:interpolate(ref['r'],ref[f],r) for f in FIELDS};error={f:physical[f]-truth[f] for f in FIELDS};referror={f:interpolate(coarse['r'],coarse[f],r)-truth[f] for f in FIELDS};betaerror=beta-truth['beta_r'][:,None]*normal;tangent=np.linalg.norm(beta-physical['beta_r'][:,None]*normal,axis=1)
  logs=[a.baseline/'full.stderr'] if name=='baseline' else [a.controls/(name+'-'+command['name']+'.stderr') for command in record['commands'] if command['name'] in ['full','segment1','segment2']]
  solve_rows=[]
  for log in logs:
   text=track(log).read_text()
   for match in re.finditer(r'shift_solve iterations=(\d+) seconds=([\d.eE+-]+) physical_rms=([\d.eE+-]+) point_max=([\d.eE+-]+)',text):
    solve_rows.append({'iterations':int(match[1]),'seconds':float(match[2]),'physical_RMS':float(match[3]),'point_max':float(match[4])})
  cost={'source':'Actual native full-run stderr; includes warmstart/accepted-state checks, all scalar/vector original-equation solves, not a rank-scaling claim','calls':len(solve_rows),'iteration_sum':sum(r['iterations'] for r in solve_rows),'solve_wall_seconds_sum':sum(r['seconds'] for r in solve_rows),'maximum_iterations':max((r['iterations'] for r in solve_rows),default=None),'rows':solve_rows}
  row={'execution_control_history':record.get('commands',record.get('cases',[])),'matched_actual_RK':record.get('matched_actual_RK',record.get('matched_actual_RK_comparison',{})),'restart_control':restart_followup if name=='dt_half' else record.get('restart',record.get('accepted_restart')),'solve_cost':cost,'n':n,'cells':len(q),'time':trace['time'],'dt':record.get('dt',1e-4),'tolerance':record.get('tolerance',1e-11),'minimum_radius':float(r.min()),'Penrose_volume':float(w.sum()),'physical_errors':{f:norms(error[f],w) for f in FIELDS},'reference_N1025_minus_N2049':{f:norms(referror[f],w) for f in FIELDS},'regions':{},'angular':{},'original_vector_residual':{'stage_RMS_max':trace['shift_original_rms_max'],'stage_point_max':trace['shift_point_max'],'regional_reconstruction_available':False},'gates':{f:{'value':trace[f],'unchanged_target':tol,'passes':trace[f]<=tol} for f,tol in [('scri_tensor_pole',1e-7),('scri_ghost_clock',1e-13),('scri_ghost_null',1e-12),('scri_ghost_tangent',1e-12)]},'scientific_acceptance':False}
  for region,mask in masks.items():
   if not mask.any():row['regions'][region]={'cells':0,'unavailable':True};continue
   ww=w[mask];row['regions'][region]={'cells':int(mask.sum()),'Penrose_volume':float(ww.sum()),'diagnostics':{f:norms(q[f][mask],ww) for f in ['H','M','Theta','Z']},'physical_errors':{f:norms(error[f][mask],ww) for f in FIELDS},'beta_vector_error':norms(np.linalg.norm(betaerror[mask],axis=1),ww),'beta_tangent':norms(tangent[mask],ww)}
  for f in ['H','M','Theta','Z']:assert abs(row['regions']['all']['diagnostics'][f]['penrose_volume_rms']-trace[f+'_rms'])<5e-12*max(1,abs(trace[f+'_rms']))
  groups=np.round(r,14);radii=np.unique(groups)
  for f in FIELDS:
   mean=[];scatter=[];counts=[];angular_rms=[];small=[];large=[];exact=interpolate(ref['r'],ref[f],radii)
   for rr,value in zip(radii,exact):
    select=groups==rr;v=physical[f][select];mean.append(float(v.mean()));scatter.append(float(v.std()));counts.append(len(v));small.append(float(v.min()));large.append(float(v.max()));angular_rms.append(float(np.sqrt(np.mean((v-value)**2))))
   row['angular'][f]={'radii':radii.tolist(),'counts':counts,'mean':mean,'scatter':scatter,'minimum':small,'maximum':large,'angular_RMS_error':angular_rms,'mean_error':(np.array(mean)-exact).tolist(),'maximum_scatter':max(scatter)}
  if name=='baseline':baseline=(q,physical,masks)
  elif n==6:
   b,bp,bm=baseline;assert np.array_equal(xyz,np.column_stack([b[f] for f in 'xyz']));fields=[f'state_{i}' for i in range(22)]+[f'delta_beta_{i}' for i in range(3)];row['same_grid_control_difference']={'regions':{},'maximum_stored_or_shift_difference':max(float(np.max(abs(q[f]-b[f]))) for f in fields),'volume_change':float(w.sum()-b['weight'].sum()),'qualification':'All stored coefficients, shift corrections, physical fields and scalar diagnostics; both own and frozen baseline volume weights'}
   for region,mask in masks.items():
    if not mask.any():continue
    assert np.array_equal(mask,bm[region]);ww=w[mask];bw=b['weight'][mask]
    def both(delta):return {'current_weights':norms(delta,ww),'baseline_weights':norms(delta,bw)}
    row['same_grid_control_difference']['regions'][region]={'stored_and_delta_beta':{f:both(q[f][mask]-b[f][mask]) for f in fields},'physical':{f:both(physical[f][mask]-bp[f][mask]) for f in FIELDS},'scalar_diagnostics':{f:both(q[f][mask]-b[f][mask]) for f in ['H','M','Theta','Z']}}
  np.savez_compressed(out/(name+'-all-cell-and-angular.npz'),points=xyz,radius=r,angular_group_radius=groups,directions=normal,weights=w,fields=np.array(FIELDS),physical=np.column_stack([physical[f] for f in FIELDS]),reference=np.column_stack([truth[f] for f in FIELDS]),physical_errors=np.column_stack([error[f] for f in FIELDS]),H=q['H'],M=q['M'],Theta=q['Theta'],Z=q['Z'],beta=beta,beta_vector_error=betaerror,beta_tangent=tangent)
  summary['cases'][name]=row;cache[name]=(q,physical)
 with tempfile.TemporaryDirectory(prefix='native-accuracy-plot-',dir=out) as directory:
  os.environ['MPLCONFIGDIR']=directory;os.environ['XDG_CACHE_HOME']=directory
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=plt.rcParams['axes.prop_cycle'].by_key()['color'];regions=['all','core_r_lt_01','cartesian','cartesian_shell_stencil_neighbors','radial_shell_stencil_neighbors','scri_r_ge_09']
  for (name,row),color in zip(summary['cases'].items(),colors):
   axes[0,0].semilogy(range(len(FIELDS)),[row['physical_errors'][f]['penrose_volume_rms'] for f in FIELDS],'.-',label=name,color=color);axes[0,1].semilogy(range(len(FIELDS)),[row['angular'][f]['maximum_scatter'] for f in FIELDS],'.-',color=color)
   for ax,f in zip(axes[1],['H','M']):ax.semilogy(range(len(regions)),[row['regions'][r]['diagnostics'][f]['penrose_volume_rms'] for r in regions],'.-',color=color)
  axes[0,0].legend(fontsize=8);axes[0,0].set_title('All-cell volume RMS versus spherical');axes[0,1].set_title('Maximum actual-node angular scatter')
  for ax in axes[0]:ax.set_xticks(range(len(FIELDS)),FIELDS,rotation=35,ha='right');ax.grid(alpha=.2)
  for ax,f in zip(axes[1],['H','M']):ax.set_xticks(range(len(regions)),['all','r<.1','Cartesian','cube-shell','shell radial','r>=.9'],rotation=30,ha='right');ax.set_title(f+' RMS; overlapping regions');ax.grid(alpha=.2)
  fig.suptitle('Frozen native accuracy controls at t=.0002\nAll scientific FAIL; volume/endpoint sampling changes preclude RMS-order claim');fig.savefig(out/'accuracy-regions.png',dpi=150);plt.close(fig)
  selected=[(k,summary['cases'][k]) for k in ['baseline','n8','n10'] if k in summary['cases']]
  if len(selected)>1:
   ns=[r['n'] for _,r in selected];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
   for metric in ['penrose_volume_rms','weighted_l2','point_max']:axes[0,0].semilogy(ns,[r['regions']['all']['diagnostics']['H'][metric] for _,r in selected],'.-',label=metric)
   for f in ['alpha','chi','beta_r']:axes[0,1].semilogy(ns,[r['physical_errors'][f]['penrose_volume_rms'] for _,r in selected],'.-',label=f)
   axes[1,0].plot(ns,[r['Penrose_volume'] for _,r in selected],'.-',label='Penrose volume');axis=axes[1,0].twinx();axis.plot(ns,[r['minimum_radius'] for _,r in selected],'x--',color='tab:orange');axis.set_ylabel('minimum sampled radius',color='tab:orange')
   for f in ['scri_tensor_pole']:axes[1,1].semilogy(ns,[r['gates'][f]['value'] for _,r in selected],'.-',label=f)
   axes[1,1].axhline(1e-7,color='k',ls=':',label='unchanged pole target')
   for ax in axes.flat:ax.set_xticks(ns);ax.set_xlabel('logical active cells per block axis N');ax.grid(alpha=.2);ax.legend(fontsize=8)
   axes[0,0].set_title('All-cell Hamiltonian metrics');axes[0,1].set_title('All-cell physical-field RMS error');axes[1,0].set_title('Changing domain quadrature and endpoint sampling');axes[1,1].set_title('Tensor pole remains above target')
   fig.suptitle('Related frozen native spatial controls at t=.0002\nNo RMS-order fit; same physical gauge and every active cell retained');fig.savefig(out/'resolution-controls.png',dpi=150);plt.close(fig)
 summary['input_hashes']=hashes;summary['inputs_unchanged']=all(sha(Path(p))==v for p,v in hashes.items());assert summary['inputs_unchanged'];summary['script_sha256']=sha(Path(__file__));summary['command']=sys.argv;summary['seconds']=time.monotonic()-start;(out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({name:{'H_RMS':row['regions']['all']['diagnostics']['H']['penrose_volume_rms'],'H_L2':row['regions']['all']['diagnostics']['H']['weighted_l2'],'volume':row['Penrose_volume'],'beta_error':row['physical_errors']['beta_r']['penrose_volume_rms']} for name,row in summary['cases'].items()}))

if __name__=='__main__':main()
