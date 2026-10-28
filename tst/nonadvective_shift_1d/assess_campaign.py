"""Verify terminal trial evidence and render the requested diagnostic comparisons."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def time_rms(t,v):return float(np.sqrt(np.trapezoid(v*v,t)/(t[-1]-t[0]))) if len(t)>1 and t[-1]>t[0] else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--supplement',type=Path);p.add_argument('--constraint-audit',type=Path,required=True);p.add_argument('--reference-sampling',type=Path,required=True);a=p.parse_args()
 c=json.loads((a.campaign/'campaign.json').read_text());assert c['active_child'] is None and c['status']=='completed_campaign'
 for rel,sha in c['frozen_hashes'].items():assert digest(a.campaign/'source'/rel)==sha
 events=c['finished'].copy();supplement=None
 if a.supplement:
  supplement=json.loads(a.supplement.read_text());assert supplement['status']=='completed' and supplement['active_child'] is None;events+=supplement['finished']
 runs=[]
 for event in events:
  case=a.campaign/event['name'];m=json.loads((case/'manifest.json').read_text());assert m['status'] not in ['running','launch_failed'];assert m['frozen_hashes']==c['frozen_hashes']
  for name,sha in m['checkpoint_hashes'].items():assert digest(case/name)==sha
  if m.get('parent'):
   for name,sha in m['parent']['hashes'].items():assert digest(Path(m['parent']['path'])/name)==sha
  h=json.loads((case/'history.json').read_text());runs.append(dict(name=event['name'],case=str(case),manifest=m,history=h))
 a.output.mkdir(parents=True,exist_ok=False);figdir=a.output/'figures';figdir.mkdir()
 # Retain terminal leaves of each physical trajectory, including failed leaves.
 groups={}
 for run in runs:
  cfg=run['manifest']['configuration'];key=(cfg['reference_damping'],cfg['puncture'],cfg['n'])
  if key not in groups or run['manifest']['target']>groups[key]['manifest']['target']:groups[key]=run
 punctures=[v for k,v in groups.items() if k[1]];pulses=[v for k,v in groups.items() if not k[1]]
 refh=json.loads((a.reference/'history.json').read_text());refm=json.loads((a.reference/'manifest.json').read_text())
 assert refm['status']=='completed'
 for name,sha in refm['checkpoint_hashes'].items():assert digest(a.reference/name)==sha
 sampling=json.loads((a.reference_sampling/'manifest.json').read_text());assert sampling['status']=='completed' and sampling['exact_checkpoint_recovery']['state'] and sampling['exact_checkpoint_recovery']['compensation']
 for name,sha in sampling['checkpoint_hashes'].items():assert digest(a.reference_sampling/name)==sha
 sampleh=json.loads((a.reference_sampling/'history.json').read_text());merged={round(r['t']/refm['dt']):r for r in refh}
 for row in sampleh:
  key=round(row['t']/refm['dt'])
  if key in merged:
   for field in ['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms','apparent_horizon_mass','apparent_horizon_r']:assert row[field]==merged[key][field],field
  merged[key]=row
 rates=json.loads((a.reference_sampling/'diagnostic-rates.json').read_text());assert rates['replay_manifest_sha256']==digest(a.reference_sampling/'manifest.json')
 for row in rates['rows']:merged[round(row['t']/refm['dt'])].update({k:v for k,v in row.items() if k!='t'})
 refh=[merged[k] for k in sorted(merged)]
 fd=json.loads(a.constraint_audit.read_text());fd_by_case={}
 for row in fd['rows']:fd_by_case.setdefault(row['case'],[]).append(row)
 colors={'Advective reference':'tab:blue','Direct control':'tab:orange','Adaptive reference':'tab:green'}
 def label(run):return 'Direct control' if run['manifest']['configuration']['reference_damping']==0 else 'Adaptive reference'
 def line(ax,run,key,log=False,reference=False,end_time=None):
  if not reference and key=='Z_exterior_rms':
   data=fd_by_case[run['name']];t=3*np.array([row['t'] for row in data]);v=np.array([row['fd_exterior_norms']['Z'] for row in data]);name=label(run)
   ax.semilogy(t,np.ma.masked_less_equal(v,0),'-o',ms=3,color=colors[name],label=name)
   if run['manifest']['status']!='completed':ax.plot(t[-1],v[-1],'x',ms=7,color=colors[name])
   return
  h=refh if reference else run['history']
  if end_time is not None:h=[row for row in h if 3*row['t']<=end_time+1e-12]
  if reference and key in ['exterior_field_rate_rms','exterior_beta_rate_rms']:h=[row for row in h if key in row]
  name='Advective reference' if reference else label(run);t=3*np.array([r['t'] for r in h]);v=np.array([r.get(key,np.nan) for r in h],dtype=float)
  if key in ['trace_chi','trace_shear','trace_connection']:v=np.array([row['scri_traces'][{'trace_chi':0,'trace_shear':2,'trace_connection':5}[key]] for row in h])
  if key=='mass_error':v=abs(3*np.array([r['apparent_horizon_mass'] for r in h])-1)
  if key=='mass_drift':v=abs(np.array([r['apparent_horizon_mass'] for r in h])/h[0]['apparent_horizon_mass']-1)
  if not reference and run['manifest']['status']!='completed' and len(v):ax.plot(t[-1],v[-1],'x',color=colors[name],ms=7,zorder=5)
  (ax.semilogy if log else ax.plot)(t,np.ma.masked_less_equal(v,0) if log else v,color=colors[name],label=name,linewidth=1.25,linestyle='--' if name=='Direct control' else '-')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  for ext in ['png','pdf']:fig.savefig(figdir/(name+'.'+ext),dpi=180)
  plt.close(fig)
 def panel(keys,titles,name,logs=None,with_reference=True):
  fig,axes=plt.subplots(2,2,figsize=(10,6.7));fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.94,hspace=.38,wspace=.28)
  end=max((3*r['history'][-1]['t'] for r in punctures),default=.6)
  for j,(ax,key,title) in enumerate(zip(axes.flat,keys,titles)):
   if with_reference:line(ax,None,key,logs[j] if logs else False,True,end_time=end)
   for run in punctures:line(ax,run,key,logs[j] if logs else False,end_time=end)
   ax.set(title=title,xlabel='Time / M',xlim=(0,max(end,.01)));ax.grid(alpha=.2)
   if key=='max_floor_radius' and all(row.get(key,0)==0 for run in punctures for row in run['history']):ax.set_ylim(0,.001);ax.text(.5,.5,'No floor activation',ha='center',transform=ax.transAxes)
  axes.flat[0].legend(fontsize=8);save(fig,name)
 panel(['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms'],['Exterior Hamiltonian RMS','Exterior momentum RMS','Exterior Theta RMS','Exterior connection-constraint RMS'],'exterior-constraints',[True]*4)
 panel(['mass_error','mass_drift','apparent_horizon_r','exterior_field_rate_rms'],['Absolute relative AH mass error','Relative AH mass drift from initial value','Outermost compact horizon radius','Exterior field time-derivative RMS'],'mass-horizon-settling',[True,True,False,True])
 panel(['max_relative_null','pole','raw_clock_rate','driver_boundary_residual'],['Maximum stagewise relative scri nullness','Maximum absolute scri numerator','Raw lapse-clock compatibility','Driver-boundary compatibility'],'boundary-health',[True,True,False,False])
 panel(['max_trace','trace_chi','trace_shear','trace_connection'],['Maximum stagewise regularity trace','Scri chi minus g','Shear-like regularity trace','Evolved-jet Lambda minus V trace'],'regularity-traces',[True,False,False,False])
 panel(['nonadvective_gap_min','g_min','alpha_min','max_floor_radius'],['New characteristic separation (times g)','Minimum radial metric (accepted samples)','Conformal lapse minimum (accepted samples)','Maximum floor activation radius'],'gauge-positivity-floor',[False,False,True,False],with_reference=False)
 panel(['H_interior_max','M_interior_max','Theta_interior_max','Z_interior_max'],['Interior max Hamiltonian error','Interior max momentum error','Interior max Theta','Interior max connection constraint'],'interior-errors',[True]*4)
 panel(['max_boundary_reduction_defect','reference_gap_max','exterior_beta_rate_rms','exterior_Lambda_rms'],['Maximum boundary derivative-reduction defect','Maximum adapting-reference gap','Exterior shift time-derivative RMS','Exterior independent connection RMS'],'reduction-and-settling',[True]*4,with_reference=False)
 fig,axes=plt.subplots(1,2,figsize=(10,3.8));fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.90,wspace=.3)
 for run in punctures:
  name=label(run);h=run['history'];data=fd_by_case[run['name']]
  axes[0].semilogy([3*r['t'] for r in h],np.ma.masked_less_equal([r['Z_exterior_rms'] for r in h],0),'--',color=colors[name],label=name+', evolved boundary jet')
  axes[0].semilogy([3*r['t'] for r in data],np.ma.masked_less_equal([r['fd_exterior_norms']['Z'] for r in data],0),'-o',color=colors[name],label=name+', metric finite difference')
  axes[1].plot([3*r['t'] for r in data],[r['endpoint_reduction_defect'][1] for r in data],'-o',color=colors[name],label=name)
 axes[0].set(title='Exterior Z: independent diagnostic check',ylabel='Exterior RMS')
 axes[1].set(title='Boundary metric derivative inconsistency',ylabel='Independent g_r minus FD g_r')
 for ax in axes:ax.set_xlabel('Time / M');ax.grid(alpha=.2);ax.legend(fontsize=7)
 save(fig,'constraint-definition')
 # Pulse tests are plotted in coordinate time, without an artificial BH mass.
 fig,axes=plt.subplots(1,2,figsize=(10,3.8));fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.90,wspace=.3)
 for run in pulses:
  h=run['history'];t=[r['t'] for r in h];name=label(run)+f", N={run['manifest']['configuration']['n']}"
  for ax,key in zip(axes,['pole','max_boundary_reduction_defect']):
   v=[r[key] for r in h];ax.semilogy(t,np.ma.masked_less_equal(v,0),label=name)
   if run['manifest']['status']!='completed':ax.plot(t[-1],v[-1],'kx',ms=7)
 for ax,title in zip(axes,['Regular-center pulse: scri numerator','Regular-center pulse: reduction defect']):ax.set(title=title,xlabel='Coordinate time');ax.grid(alpha=.2);ax.legend(fontsize=7)
 save(fig,'pulse-gates')
 fig,axes=plt.subplots(1,2,figsize=(10,3.8));fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.90,wspace=.3)
 time=3*np.array([row['t'] for row in refh]);radius=np.array([row['apparent_horizon_r'] for row in refh]);mass=abs(3*np.array([row['apparent_horizon_mass'] for row in refh])-1);late=time>=80
 axes[0].plot(time,radius,color='tab:blue',label='Advective N=257');fit=np.polyfit(time[late],radius[late],1);axes[0].plot(time[late],np.polyval(fit,time[late]),'k--',label='Late 20M linear fit')
 axes[1].semilogy(time,np.ma.masked_less_equal(mass,0),color='tab:blue',label='Advective N=257')
 axes[0].set(title='Reference horizon coordinate through 100M',ylabel='Compact outermost horizon radius');axes[1].set(title='Reference mass error through 100M',ylabel='Absolute relative AH mass error')
 for ax in axes:ax.set_xlabel('Time / M');ax.grid(alpha=.2);ax.legend(fontsize=8)
 save(fig,'advective-duration-context')
 # Find exact saved profile times in the verified reference ancestry.
 ancestry=[];case=a.reference
 while True:
  m=json.loads((case/'manifest.json').read_text());assert m['status']=='completed'
  for name,sha in m['checkpoint_hashes'].items():assert digest(case/name)==sha
  ancestry.append(case)
  if not m.get('parent'):break
  case=Path(m['parent']['path'])
 ref_profiles={}
 for case in ancestry:
  with np.load(case/'profiles.npz') as z:
   for i,t in enumerate(z['t']):ref_profiles.setdefault(round(float(t),12),(case,i))
 with np.load(a.reference_sampling/'profiles.npz') as z:
  for i,t in enumerate(z['t']):ref_profiles[round(float(t),12)]=(a.reference_sampling,i)
 # Recover terminal physical profiles algebraically from last accepted states;
 # this is not an evolution or a restart of the failed trajectory.
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hyperbolic_shift_1d'))
 from puncture import PunctureEvolution
 import inspect
 physical_maps={}
 for run in punctures:
  cfg=run['manifest']['configuration'];kwargs={k:v for k,v in refm['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
  ev=PunctureEvolution(cfg['n'],a.reference/'source'/Path(refm['arguments']['library']).name,**kwargs)
  with np.load(Path(run['case'])/'profiles.npz') as z:
   records={round(float(t),12):(z['r'].copy(),q.copy()) for t,q in zip(z['t'],z['physical'])};last=z['last'].copy();last_time=round(float(z['last_time']),12)
   if last_time not in records:
    chart=last[:,:8].copy();chart[1:,6]=np.log(last[1:,6]/ev.chart.refs['alpha'][1:,0]);q,chi,alpha,_=ev.chart.reconstruct(chart)
    records[last_time]=(ev.r.copy(),np.c_[chi,1+last[:,1],q[:,2:6],alpha,last[:,7]-ev.r,last[:,8],last[:,9]-ev.r])
  physical_maps[run['name']]=records
 shared=set(ref_profiles)
 for run in punctures:
  shared&=set(physical_maps[run['name']])
 shared=sorted(shared);profile_time=max(shared) if shared else None
 if profile_time is not None:
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hyperbolic_shift_1d'))
  from puncture import PunctureEvolution
  import inspect
  case,index=ref_profiles[profile_time];m=json.loads((case/'manifest.json').read_text());args=m.get('arguments',refm['arguments']);lib=case/'source'/Path(args['library']).name
  kwargs={k:v for k,v in args.items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
  e=PunctureEvolution(args['n'],lib,**kwargs)
  with np.load(case/'profiles.npz') as z:q=z['q'][index]
  e.rhs(q);x=e.last;profiles=[('Advective reference',e.r,x['alpha'],x['beta'],x['g'])]
  for run in punctures:
   grid,phys=physical_maps[run['name']][profile_time];profiles.append((label(run),grid,phys[:,6],phys[:,7],phys[:,1]))
  fig,axes=plt.subplots(1,3,figsize=(12,3.8));fig.subplots_adjust(left=.07,right=.98,bottom=.18,top=.83,wspace=.32)
  for name,r,alpha,beta,g in profiles:
   for ax,v in zip(axes,[alpha,beta,g]):ax.plot(r[1:],v[1:],label=name,color=colors[name])
  for ax,title in zip(axes,['Conformal lapse','Radial shift','Radial conformal metric']):ax.set(title=title,xlabel='Compact radius r');ax.grid(alpha=.2)
  axes[0].legend(fontsize=7);fig.suptitle(f'Exact matched profiles at {3*profile_time:g}M');save(fig,'matched-profiles')
 assessments=[]
 for run in punctures:
  h=run['history'];m=run['manifest'];mass=np.array([r.get('apparent_horizon_mass',np.nan) for r in h],dtype=float);t=np.array([r['t'] for r in h]);steps={round(r['t']/refm['dt']):r for r in refh};matched=[]
  for r in h:
   key=round(r['t']/refm['dt'])
   if key in steps and abs(steps[key]['t']-r['t'])<1e-12:matched.append(dict(t=r['t'],new={k:r[k] for k in ['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms','apparent_horizon_mass','apparent_horizon_r']},advective={k:steps[key][k] for k in ['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms','apparent_horizon_mass','apparent_horizon_r']}))
  for match in matched:
   row=next((row for row in fd_by_case[run['name']] if abs(row['t']-match['t'])<1e-12),None)
   match['connection_constraint_definition']='metric finite difference' if row else 'evolution-used boundary jet; no independent profile at this time'
   if row:match['new']['Z_exterior_rms']=row['fd_exterior_norms']['Z']
  assessments.append(dict(name=label(run),case=run['case'],status=m['status'],t_over_M=3*m['t'],target_over_M=3*m['target'],error=m['error'],peak_relative_mass_drift=float(np.nanmax(abs(mass/mass[0]-1))),final_absolute_relative_mass_error=float(abs(3*mass[-1]-1)),last_report_t_over_M=3*t[-1],exact_matched_diagnostic_samples=matched,stage_stats=m['stage_stats']))
 summary=dict(constraint_audit_sha256=digest(a.constraint_audit),reference_sampling=str(a.reference_sampling),reference_sampling_manifest_sha256=digest(a.reference_sampling/'manifest.json'),supplement=str(a.supplement) if a.supplement else None,supplement_sha256=digest(a.supplement) if a.supplement else None,campaign=str(a.campaign),campaign_sha256=digest(a.campaign/'campaign.json'),runs=[dict(name=r['name'],case=r['case'],status=r['manifest']['status'],t_over_M=3*r['manifest']['t'],error=r['manifest']['error'],manifest_sha256=digest(Path(r['case'])/'manifest.json')) for r in runs],puncture_assessments=assessments,matched_profile_t_over_M=3*profile_time if profile_time is not None else None,reaches_100M=any(r['manifest']['status']=='completed' and r['manifest']['t']>=33.3333984375-1e-12 for r in punctures),scope='No failed pulse candidate is promoted to long evolution. Crosses mark last reported diagnostics, which can precede the actual failure time in the manifest. Matched profiles use exact saved times; no interpolation. All interior errors remain separate.')
 (a.output/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
