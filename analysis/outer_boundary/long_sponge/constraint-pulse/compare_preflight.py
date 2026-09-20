"""Compare already-written MPI checkpoints; never run or change the solver."""
from pathlib import Path
import json,os,re,sys
import numpy as np
sys.path.insert(0,os.environ.get('ATHENA_REGRESSION_PATH','/Users/hz0693/research/TDE/athenak-outer-boundary-fix/tst/regression'))
from z4c_background_restart import checkpoint,cohort
root=Path(__file__).resolve().parent
def active(name,cycle=None):
    run=root/'runs'/name
    if cycle is None:cycle=max(checkpoint(p)['cycle'] for p in (run/'rst/rank_00000000').glob('*.rst'))
    _,recs=cohort(run,8,cycle)
    blocks=[np.asarray(a).reshape(25,40,40,40)[:,4:36,4:36,4:36] for rec in recs for a in rec['state']]
    return np.stack(blocks),recs[0]
a,qa=active('theta_loweta_early_dt3p2');b,qb=active('theta_loweta_early_dt1p6')
assert abs(qa['time']-320)<1e-9 and abs(qb['time']-320)<1e-9
initial1,qi1=active('theta_loweta_early_dt3p2',0);initial2,qi2=active('theta_loweta_early_dt1p6',0)
delta=a-b
def norm(x):return float(np.linalg.norm(x.ravel()))
weighted_a=a.copy();weighted_b=b.copy();weighted_a[:,7:18]*=64;weighted_b[:,7:18]*=64
result=dict(time=qa['time'],ranks=8,blocks=8,initial_active_bitwise_equal=bool(np.array_equal(initial1,initial2)),coarse_cycle=qa['cycle'],fine_cycle=qb['cycle'],coarse_checkpoint_dt=qa['dt'],fine_checkpoint_dt=qb['dt'],relative_active_state_L2=norm(delta)/norm(b),relative_derivative_scaled_state_L2=norm(weighted_a-weighted_b)/norm(weighted_b),theta_relative_L2=norm(delta[:,17])/norm(b[:,17]),theta_max_abs_difference=float(abs(delta[:,17]).max()),all_fields_max_abs_difference=float(abs(delta).max()),coarse_theta_max=float(abs(a[:,17]).max()),fine_theta_max=float(abs(b[:,17]).max()),scope='Two timesteps at one resolution and one early time; relative difference, not estimated error or convergence order. Derivative scaling multiplies Khat/A/Gamma/Theta by dx=64.')
assert result['initial_active_bitwise_equal']
result['gates']={}
for name in ['zero_standard','pulse_gate_standard','zero_loweta','pulse_gate_loweta','theta_loweta_early_dt3p2','theta_loweta_early_dt1p6','zero_loweta_lapse01','pulse_gate_loweta_lapse01','pulse_gate_loweta_coremask','zero_loweta_coremask','zero_loweta_lapse01_coremask']:
    val=json.loads((root/'runs'/name/'checkpoint-validation.json').read_text());prov=json.loads((root/'runs'/name/'provenance.json').read_text())
    result['gates'][name]={k:val[k] for k in ['time_code','cycle','passed','all_payload_finite','invalid_metric_cells_including_ghosts','all_residuals_zero','stopping_reason','target_reached','minimum']}
    result['gates'][name]['wall_seconds']=prov['wall_seconds']
(root/'preflight-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
