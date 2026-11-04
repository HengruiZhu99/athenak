"""Compare preserved BN1 and BC2 N16 histories without changing admission."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'n16-comparison'
OUT.mkdir(exist_ok=True)
INPUTS = {'BC2 gamma=1': HERE.parent/'boosted-refined-core-v1/completed',
          'BN1 gamma=k²': HERE/'n16-completed'}
hist = {}; reports = {}; provenance = {}
for label, p in INPUTS.items():
    worker = json.loads((p/'worker-status.json').read_text())
    binding = json.loads((p/'binding.json').read_text())
    submission = json.loads((p/'submission.json').read_text())
    assert worker['status'] == 'passed'
    assert worker['job_id'] == submission['job_id']
    assert hashlib.sha256((p/'plan.json').read_bytes()).hexdigest() == binding['plan_sha256'] == worker['plan_sha256'] == submission['plan_sha256']
    assert worker['binaries'] == binding['binaries']
    hashes = {}
    for name in ['puncture.history.csv', 'puncture.report.json']:
        expected = [v for k,v in worker['cases'][0]['artifacts'].items() if k.endswith('/'+name)]
        assert len(expected) == 1
        data = (p/name).read_bytes(); digest = hashlib.sha256(data).hexdigest()
        assert digest == expected[0]['sha256'] and len(data) == expected[0]['bytes']
        hashes[name] = digest
    r = json.loads((p/'puncture.report.json').read_text())
    assert r['n'] == 16 and r['dt'] == .00025 and r['status'] == 'completed' and r['checkpoint_admitted']
    assert r['checkpoint_schema'] == ('HYP3D_MPI_BN1' if label.startswith('BN1') else 'HYP3D_MPI_BC2')
    reports[label] = r
    hist[label] = np.genfromtxt(p/'puncture.history.csv', delimiter=',', names=True)
    provenance[label] = dict(job=worker['job_id'], source_manifest_sha256=binding['source_manifest_sha256'], binaries=binding['binaries'], input_hashes=hashes)

fig, axes = plt.subplots(2,3,figsize=(13,7))
fields = [('H_exterior_L2','Candidate exterior Hamiltonian', True),
          ('M_exterior_L2','Candidate exterior momentum', True),
          ('exterior_cutoff','Moving exterior cutoff', False),
          ('horizon_trial_expansion_rms','Candidate expansion RMS', True),
          ('puncture_x','Puncture x', False),
          ('horizon_mass','Recorded relative mass error', True)]
for ax,(field,title,log) in zip(axes.flat,fields):
    for label,a in hist.items():
        y = a[field]
        if field == 'horizon_mass': y = np.where(a['horizon_found']==1,abs(3*y-1),np.nan)
        ax.plot(a['t_over_M'],y,'.-',ms=3,label=label)
    if log: ax.set_yscale('log')
    ax.set(title=title,xlabel='t/M',xlim=(0,10));ax.grid(alpha=.25)
    if field == 'horizon_trial_expansion_rms':
        ax.axhline(.01,color='black',ls='--',lw=1,label='Post 1e-2')
        ax.axhline(.001,color='gray',ls=':',lw=1,label='Online 1e-3')
axes[0,0].legend();axes[1,0].legend()
fig.suptitle('N16 normalized inner shift versus original driver\nMoving candidate domains differ; missing masses remain gaps')
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'histories.png',dpi=160);fig.savefig(OUT/'histories.pdf');plt.close(fig)
summary = {'provenance':provenance,'qualification':'Own-candidate moving domains; no common-cutoff or convergence claim. Post 1e-2 expansion selection does not recover masses.'}
for label,a in hist.items():
    r=reports[label];passed=(a['horizon_complete_surface']==1)&np.isfinite(a['horizon_trial_expansion_rms'])&(a['horizon_trial_expansion_rms']<=.01)&(a['horizon_trial_radius']>0)
    accepted=a[a['horizon_found']==1]
    summary[label] = dict(rows=len(a), online_accepted=len(accepted),post_expansion_pass=int(passed.sum()),last_online_mass_time_M=float(accepted[-1]['t_over_M']),final_candidate_rms=float(a[-1]['horizon_trial_expansion_rms']),final_H=float(a[-1]['H_exterior_L2']),final_M=float(a[-1]['M_exterior_L2']),final_cut=float(a[-1]['exterior_cutoff']),final_tracker_x=float(a[-1]['puncture_x']),chi_floor_stages=r['chi_floor_stage_interventions'],chi_floor_final=r['chi_floor_acceptance_interventions'],lapse_floor_nodes=r['lapse_floor_node_interventions'],null_surface=r['max_null_surface'],null_ghost=r['max_null_ghost'],cumulative_pole=r['max_pole'],cumulative_clock=r['max_clock'])
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
