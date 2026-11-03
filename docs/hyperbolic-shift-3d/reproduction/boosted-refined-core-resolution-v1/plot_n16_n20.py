"""Compare preserved BC2 N16/N20 histories, without evolving or admitting N20."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'n16-n20-histories';OUT.mkdir(exist_ok=True)
INPUTS={'N16':HERE.parent/'boosted-refined-core-v1/completed','N20':HERE/'n20-failure'}
COLORS={'N16':'#2166ac','N20':'#d95f02'}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
hist={};reports={};provenance={}
for label,p in INPUTS.items():
    worker=read(p/'worker-status.json');binding=read(p/'binding.json');submission=read(p/'submission.json')
    assert worker['job_id']==submission['job_id']
    assert sha(p/'plan.json')==binding['plan_sha256']==worker['plan_sha256']==submission['plan_sha256']
    assert binding['binaries']==worker['binaries']
    checks={}
    for f in ['puncture.history.csv','puncture.report.json']:
        candidates=[v for k,v in worker['cases'][0]['artifacts'].items() if k.endswith('/'+f)]
        assert len(candidates)==1
        digest=sha(p/f);assert digest==candidates[0]['sha256'] and (p/f).stat().st_size==candidates[0]['bytes']
        checks[f]=digest
    report=read(p/'puncture.report.json');reports[label]=report
    assert report['checkpoint_schema']=='HYP3D_MPI_BC2'
    assert report['n']==int(label[1:])
    assert report['dt']==(.00025 if label=='N16' else .0002)
    assert report['status']==('completed' if label=='N16' else 'failed')
    assert report['checkpoint_admitted']==(label=='N16')
    hist[label]=np.genfromtxt(p/'puncture.history.csv',delimiter=',',names=True)
    assert np.all(np.diff(hist[label]['t_over_M'])>0)
    provenance[label]=dict(path=str(p),job_id=worker['job_id'],plan_sha256=binding['plan_sha256'],
        binaries=binding['binaries'],source_manifest_sha256=binding['source_manifest_sha256'],input_hashes=checks)
assert provenance['N16']['source_manifest_sha256']==provenance['N20']['source_manifest_sha256']
assert provenance['N16']['binaries']==provenance['N20']['binaries']
failure=reports['N20']['t_over_M']
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9})

def decorate(ax,log=False):
    ax.axvline(failure,color='#555555',ls='--',lw=1,label='N20 stop: 6.2082M')
    ax.axvspan(failure,10,color='#777777',alpha=.055)
    ax.set(xlabel=r'$t/M$',xlim=(0,10))
    ax.grid(alpha=.2)
    if log:ax.set_yscale('log')

fig,axes=plt.subplots(2,3,figsize=(13.4,7.4))
panels=[('H_exterior_L2','Exterior Hamiltonian',r'$\|H\|_2$'),
        ('M_exterior_L2','Exterior conformal momentum',r'$\|M\|_2$'),
        ('Mr_exterior_L2','Exterior radial momentum',r'$\|M_r\|_2$'),
        ('H_L2','Full-domain Hamiltonian',r'$\|H\|_2$'),
        ('M_L2','Full-domain conformal momentum',r'$\|M\|_2$'),
        ('Mr_L2','Full-domain radial momentum',r'$\|M_r\|_2$')]
for ax,(field,title,ylabel) in zip(axes.flat,panels):
    for label,a in hist.items():
        ax.plot(a['t_over_M'],a[field],'.-',ms=3.5,lw=1.6,color=COLORS[label],label=label)
    decorate(ax,True);ax.set(title=title,ylabel=ylabel)
axes[0,0].legend(loc='best')
fig.suptitle('Boosted refined-core BC2: N16 versus N20\nExterior norms use each run\'s moving candidate cutoff; no fixed-domain convergence inference',fontsize=13)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'constraints.png',dpi=180);fig.savefig(OUT/'constraints.pdf');plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(13.4,7.5))
for label,a in hist.items():
    t=a['t_over_M'];color=COLORS[label]
    axes[0,0].plot(t,a['exterior_cutoff'],'.-',ms=3,color=color,label=label)
    axes[0,1].plot(t,a['puncture_x'],'.-',ms=3,color=color,label=label)
    axes[0,2].plot(t,a['horizon_trial_expansion_rms'],'.-',ms=3,color=color,label=label)
    mass=np.where((a['horizon_found']==1)&np.isfinite(a['horizon_mass']),abs(3*a['horizon_mass']-1),np.nan)
    axes[1,0].plot(t,mass,'o-',ms=3,color=color,label=label)
    axes[1,1].plot(t,np.maximum(a['null_surface'],a['null_ghost']),'.-',ms=3,color=color,label=label)
    for field,style,description in [('pole_max','-','pole'),('clock_max','--','clock')]:
        axes[1,2].plot(t,a[field],style,lw=1.5,color=color,label=label+' '+description)
axes[0,2].axhline(.01,color='black',ls='-.',lw=1.2,label=r'Post RMS: $10^{-2}$')
axes[0,2].axhline(.001,color='#888888',ls=':',lw=1.2,label=r'Online RMS: $10^{-3}$')
titles=['Moving exterior cutoff','Puncture trajectory','Complete candidate expansion RMS',
        'Recorded horizon mass error; missing values stay gaps','Max(surface, ghost) null residual','Cumulative pole / clock maxima']
ylabels=['Compact radius',r'$x_p$',r'Expansion RMS',r'$|M_{AH}/M-1|$','Null residual','Cumulative maximum']
for i,(ax,title,ylabel) in enumerate(zip(axes.flat,titles,ylabels)):
    decorate(ax,i>=2);ax.set(title=title,ylabel=ylabel)
axes[0,0].legend();axes[0,2].legend(fontsize=8);axes[1,2].legend(fontsize=8)
fig.suptitle('Gauge, horizon and scri diagnostics\nPost RMS selection does not recover unsaved masses or change the recorded exterior domains',fontsize=13)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'diagnostics.png',dpi=180);fig.savefig(OUT/'diagnostics.pdf');plt.close(fig)

# Restrict numerical ratios to actual matching samples, without interpolation.
pairs=[]
for row in hist['N20']:
    indices=np.flatnonzero(abs(hist['N16']['t_over_M']-row['t_over_M'])<1e-9)
    if len(indices)==1:
        other=hist['N16'][indices[0]]
        pairs.append(dict(t_over_M=float(row['t_over_M']),time_mismatch_M=float(abs(other['t_over_M']-row['t_over_M'])),
            cutoff16=float(other['exterior_cutoff']),cutoff20=float(row['exterior_cutoff']),
            **{key:dict(N16=float(other[key]),N20=float(row[key]),ratio_N20_N16=float(row[key]/other[key]))
               for key in ['H_exterior_L2','M_exterior_L2','Mr_exterior_L2']}))
summaries={}
for label,a in hist.items():
    accepted=(a['horizon_found']==1)&np.isfinite(a['horizon_mass'])
    post=(a['horizon_complete_surface']==1)&np.isfinite(a['horizon_trial_radius'])&(a['horizon_trial_radius']>0)&np.isfinite(a['horizon_trial_expansion_rms'])&(a['horizon_trial_expansion_rms']>=0)&(a['horizon_trial_expansion_rms']<=.01)
    r=reports[label]
    summaries[label]=dict(last_accepted_step_t_M=r['t_over_M'],last_diagnostic_t_M=float(a['t_over_M'][-1]),
        recorded_rows=len(a),online_accepted_rows=int(accepted.sum()),post_expansion_pass_rows=int(post.sum()),
        last_recorded_mass_time_M=float(a['t_over_M'][accepted][-1]),
        native_seconds=r['seconds'],ranks=r['mpi_size'],
        final_recorded={f:float(a[f][-1]) for f in ['H_exterior_L2','M_exterior_L2','Mr_exterior_L2','exterior_cutoff']})
save('comparison.json',dict(inputs=provenance,cases=summaries,matched_times=pairs,
    qualifications=['Only saved diagnostics plotted; no resampling/smoothing or extrapolation to failed stage.',
    'N20 last accepted evolution step6.2082M; last diagnostic6.15M. Failed checkpoint is unadmitted.',
    'Exterior cuts are own-candidate moving spheres and differ between runs; ratios are not fixed-domain convergence.',
    'Space and time refined together and node count changed2to4; no isolated spatial/scaling inference.',
    'Online1e-3 mass gaps retained. Post1e-2 expansion quality selection is not full native finder acceptance.']))
print(json.dumps(dict(cases=summaries,matched_rows=len(pairs),last_matched=pairs[-1]),indent=2))
