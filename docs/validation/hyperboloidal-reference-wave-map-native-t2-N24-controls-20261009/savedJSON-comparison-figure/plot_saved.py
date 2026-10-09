"""Figure from completed saved observer/manual JSON and failure stderr only."""
from pathlib import Path
import hashlib,json,math,os,re,subprocess,sys,time
P=Path(__file__).resolve().parent;R=P.parents[2]
def pin(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify(rows):
 for row in rows:assert pin(row['path'])==row,row['path']
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
index=load(P/'source-index.json');verify(index['files']+index['inputs'])
recipe=load(P/'recipe.json');verify(load(P/'runtime-inputs.json')['files'])
D=P/'attempt001';D.mkdir(exist_ok=False);started=time.monotonic()
protected=index['files']+index['inputs']+load(P/'runtime-inputs.json')['files']+[pin(P/'source-index.json')]
dump(D/'pins-before.json',protected);(D/'source.py').write_bytes(Path(__file__).read_bytes())
record={'completed':False,'source':pin(__file__),'recipe':pin(P/'recipe.json'),'scope':'Saved scalar plotting only; no arrays, source queries, native steps, operator or causal stability inference.','launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()}
try:
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 report=load(recipe['comparison_report']);assert report['passed_scalar_readback'] is True and report['all_saved_independent_field_pairs']==105
 specs=load(recipe['comparison_recipe'])['cases'];data={};derived={}
 for spec in specs:
  tag=spec['tag'];q=load(spec['owner_observations']);manual=load(spec['manual_receipt']);owner=load(spec['owner_receipt'])
  assert manual['diagnostic_protocol_completed'] and manual['before_after_equal'] and manual['saved_guard_failures']==0
  assert owner['observer_completed'] and owner['protected_before_after_equal'] and owner['accepted_native_run'] is False
  assert len(q)==manual['arrays']==owner['saved_restart_files']
  stderr=Path(spec['native_stderr']).read_text();match=re.search(r'mesh_time=([^ ]+) cycle=(\d+) xyz=\(([^)]+)\) Omega=([^ ]+)',stderr);assert match
  expected=next(x for x in report['failure_metadata'] if x['tag']==tag)
  assert match[1]==expected['time_decimal'] and int(match[2])==expected['cycle'] and match[3]==expected['xyz'] and match[4]==expected['Omega']
  times=np.array([x['time'] for x in q]);norms=np.array([x['native_diagnostics']['rms_H_Mcon_Zcon_Theta'] for x in q])
  fraction=[]
  for x in q:
   n=x['native_diagnostics'];z=n['rms_H_Mcon_Zcon_Theta'][2];s=n['shell_rms_H_Mcon_Zcon_Theta'][2]
   fraction.append(n['shell_r_ge_09_count']/n['active_count']*(s/z)**2 if z>1e-12 else None)
  assert np.isfinite(times).all() and np.isfinite(norms).all()
  data[tag]=(times,norms,fraction,float(match[1]))
  derived[tag]={'saved_states':len(q),'last_saved_time':float(times[-1]),'last_norms':norms[-1].tolist(),'last_Z_shell_squared_fraction':fraction[-1],'root_reported_abort_time_decimal':match[1],'abort_state_is_saved':False,'actual_first_history_dt':q[0]['history_dt'],'last_restart_header_dt':q[-1]['restart_header_dt']}
 # Fixed figure design and limits were declared in recipe before importing plotting modules.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 fig,axs=plt.subplots(2,2,figsize=(11.5,8.2));fig.subplots_adjust(left=.08,right=.98,top=.89,bottom=.17,hspace=.42,wspace=.26)
 colors=['#0072B2','#D55E00','#009E73','#CC79A7'];labels=['H','M','Z',r'$\Theta$']
 for ax,tags,title in [(axs[0,0],['standard','half'],'A  Large pulse: standard and half timestep'),(axs[0,1],['small'],'B  Small pulse: standard timestep')]:
  for tag in tags:
   times,norms,fraction,abort=data[tag];active=times>0
   for field,(color,label) in enumerate(zip(colors,labels)):
    ax.semilogy(times[active],norms[active,field],color=color,lw=1.8 if tag!='half' else 1.35,ls='--' if tag=='half' else '-',label=label if tag!= 'half' else None)
   ax.axvspan(times[-1],abort,color='#777777',alpha=.11,zorder=0)
  abort=data[tags[0]][3];ax.axvline(abort,color='#444444',lw=1,ls=':')
  ax.set(xlim=(0,1.06),ylim=(8e-6,.6),xlabel='Coordinate time t',ylabel='Native constraint RMS',title=title);ax.grid(True,which='major',alpha=.22)
  ax.legend(loc='lower right',ncol=4,frameon=False)
 axs[0,0].text(.03,.97,'standard abort 0.792773\nhalf-step abort 0.792806',transform=axs[0,0].transAxes,va='top',fontsize=9)
 axs[0,0].add_artist(axs[0,0].legend(handles=[Line2D([0],[0],color='#333333',lw=1.8,label='standard'),Line2D([0],[0],color='#333333',ls='--',lw=1.4,label='half dt')],loc='center left',frameon=False))
 axs[0,1].text(.03,.97,'small-pulse abort 1.018034\ninitial amplitudes 10× smaller',transform=axs[0,1].transAxes,va='top',fontsize=9)
 # These are the root's already-computed matched relative differences; no re-fitting/interpolation.
 pairs=[x for x in report['matched_standard_half_saved_pairs'] if all(y is not None for y in x['relative_differences'])]
 t=np.array([x['standard_time'] for x in pairs]);rel=100*np.array([x['relative_differences'] for x in pairs])
 for field,(color,label) in enumerate(zip(colors,labels)):axs[1,0].semilogy(t,rel[:,field],color=color,lw=1.8,label=label)
 axs[1,0].set(xlim=(0,.8),ylim=(1e-6,.2),xlabel='Matched saved time t',ylabel='Relative difference (%)',title='C  Half vs standard: saved RMS differences')
 axs[1,0].grid(True,which='major',alpha=.22);axs[1,0].legend(ncol=4,loc='upper left',frameon=False)
 axs[1,0].text(.03,.05,'At t≈0.775: H 0.0532%, M 0.0489%,\nZ 0.0756%, Θ 0.0841%',transform=axs[1,0].transAxes,fontsize=9)
 for tag,color,style,label in [('standard','#333333','-','large, standard'),('half','#0072B2','--','large, half dt'),('small','#D55E00','-','small, standard')]:
  times,norms,fraction,abort=data[tag];idx=[i for i,x in enumerate(fraction) if x is not None]
  axs[1,1].plot(times[idx],[100*fraction[i] for i in idx],color=color,ls=style,lw=1.8,label=label)
 axs[1,1].set(xlim=(0,1.06),ylim=(0,101),xlabel='Coordinate time t',ylabel=r'Squared Z norm in $r\geq0.9$ (%)',title='D  Saved Z localization')
 axs[1,1].grid(True,alpha=.22);axs[1,1].legend(loc='lower right',frameon=False)
 fig.suptitle('N24 native reference-wave-map controls — stopped-run observations',fontsize=15,x=.08,ha='left')
 fig.text(.08,.925,'Same Cartesian spherical grid, wide reference (0.05–0.95), a=0.5, κ=10, symmetric degree-2 ghosts, KO=0.1',fontsize=10)
 fig.text(.08,.095,'All three targeted t=2 and aborted at the same first reported cell (Ω=0.00217014). Curves stop at saved times;\nshading marks the interval to the unsaved abort. All 105 saved states passed the recorded field guards.\nLog panels omit t=0 near-roundoff norms. Shell fractions use unchanged native contractions and active counts.',fontsize=9,va='top',linespacing=1.5)
 fig.savefig(D/'N24-controls.png',dpi=180);fig.savefig(D/'N24-controls.pdf');plt.close(fig)
 dump(D/'plotted-values-summary.json',{'source_data_only':True,'cases':derived,'last_matched_relative_differences':report['matched_standard_half_saved_pairs'][-1]['relative_differences'],'scope':record['scope'],'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__})
 verify(protected);dump(D/'pins-after.json',protected)
 record.update(completed=True,inputs_unchanged=True,outputs=[pin(D/name) for name in ['N24-controls.png','N24-controls.pdf','plotted-values-summary.json']],scientific_native_or_probe_calls=0)
except BaseException as exc:record['failure']=type(exc).__name__+': '+str(exc);raise
finally:
 record['seconds']=time.monotonic()-started;dump(D/'receipt.json',record)
print(json.dumps(record,indent=2))
