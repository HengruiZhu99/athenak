"""Three distinct frozen scientific identities; no checkpoint relabeling."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
import compare_histories as h
from frozen_campaign import load
from audit_completed import load_completed
p=argparse.ArgumentParser();p.add_argument('--plans',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert a.plans==['campaigns/n16-restored-six-shells-v1-a-evolution-segment01/plan.json'], 'Only the audited terminal plan is supported'
old=ROOT/'output/native-shared-transition/aurora';broad=ROOT/'output/restored-wide-smooth/aurora';new=ROOT/'output/restored-six-shells/aurora'
cases={
 'A':h.load_native(old,'A',['campaigns/n16-native-transition-v2-a-evolution-segment01/plan.json']),
 '3 shells':h.load_native(broad,'A',['campaigns/n16-restored-wide-smooth-v1-a-evolution-segment01/plan.json'],load(broad/'launcher-restored-wide-smooth-v1/native_transition_campaign.py')),
 '6 shells*':load_completed()[0]}
h.LABELS['3 shells']='Broad + smooth: 3 shells';h.COLORS['3 shells']='#b02871'
h.LABELS['6 shells*']='6 shells: penalty limiter active';h.COLORS['6 shells*']='#008060'
result=h.render(cases,a.output,title='N16 to 10M: six-shell run completed, but penalty-policy check failed',
 footer='Exterior histories use each run’s candidate horizon; accepted mass gaps are retained.\n* Six-shell boundary penalty scale reached 0.651 (required 1). This is not a matched-parameter convergence result.')
print(json.dumps({k:{key:v[key] for key in ['status','last_accepted_t_over_M','accepted_horizon_rows','final_accepted_mass_error']} for k,v in result['cases'].items()},indent=2))

result['cases']['6 shells*']['campaign_accepted']=False
result['cases']['6 shells*']['status']='native_completed_but_penalty_policy_failed'
result['cases']['6 shells*']['penalty_scale_min']=load_completed()[0]['reports'][0]['scri_penalty_scale_min']
(a.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
