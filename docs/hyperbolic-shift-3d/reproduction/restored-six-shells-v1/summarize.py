"""Reproduce penalty, runtime, and endpoint qualifications without changing science."""
from pathlib import Path
import hashlib,json,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'src/z4c/hyperboloidal/hyperbolic_scri_device.hpp'
cases={}
for key,raw,campaign in [('A','native-shared-transition','n16-native-transition-v2-a-evolution-segment01'),('NB1','restored-wide-smooth','n16-restored-wide-smooth-v1-a-evolution-segment01'),('N61','restored-six-shells','n16-restored-six-shells-v1-a-evolution-segment01')]:
 path=ROOT/f'output/{raw}/aurora/campaigns/{campaign}'
 r=json.loads((path/'output/puncture.report.json').read_text());w=json.loads((path/'worker-status.json').read_text())
 hist=np.genfromtxt(path/'output/puncture.history.csv',delimiter=',',names=True)
 cases[key]={'native_elapsed_seconds':w['cases'][0]['elapsed_seconds'],'worker_elapsed_seconds':w['elapsed_seconds'],'max_bulk_cfl':r['max_cfl'],'penalty_scale_min':r['scri_penalty_scale_min'],'penalty_cfl_max':r['scri_penalty_cfl_max'],'unlimited_penalty_cfl_max':r['scri_unlimited_penalty_cfl_max'],'history_columns':list(hist.dtype.names),'fixed_cutoff_history_available':False,'history_cutoff_min_max':[float(min(hist['exterior_cutoff'])),float(max(hist['exterior_cutoff']))]}
initial=ROOT/'output/restored-six-shells/initial-check/state.report.json';r=json.loads(initial.read_text())
assert abs(r['scri_penalty_scale_min']-.5*(1-1e-12)/.768)<1e-14
penalty={'formula':'c=12 dt/h; scale=min(1,0.5(1-1e-12)/c); p4=4 scale/h, p8=8 scale/h','source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'source_lines':[204,212],'dt':.0005,'three_shell_outer_h':.25/16,'six_shell_outer_h':.125/16,'unclipped_dt_upper_bound':(.125/16)*.5*(1-1e-12)/12,'three_shell_effective_p4_p8':[4/(.25/16),8/(.25/16)],'six_shell_effective_p4_p8':[4*r['scri_penalty_scale_min']/(.125/16),8*r['scri_penalty_scale_min']/(.125/16)],'initial_report':str(initial.relative_to(ROOT)),'initial_report_sha256':sha(initial),'initial_penalty_scale_min':r['scri_penalty_scale_min'],'interpretation':'Already active in local initial-state check, not an onset of late numerical instability. Preflight review missed this existing field. Frozen campaign acceptance remains failed. No timestep/equation changes or restart authorized automatically.'}
(P/'penalty-limiter-analysis.json').write_text(json.dumps(penalty,indent=2)+'\n')
results={'cases':cases,'six_over_three_native_runtime':cases['N61']['native_elapsed_seconds']/cases['NB1']['native_elapsed_seconds'],'six_over_A_native_runtime':cases['N61']['native_elapsed_seconds']/cases['A']['native_elapsed_seconds'],'history_limitation':'Only full-domain and own-candidate cutoff histories were emitted. A fixed-cutoff time series cannot be reconstructed from the single terminal checkpoint; none is fabricated. Common-cutoff reconstruction is available at t0 and10M only. Pole/clock histories are running maxima, not instantaneous residuals.'}
(P/'comparison/runtime-and-history-scope.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
