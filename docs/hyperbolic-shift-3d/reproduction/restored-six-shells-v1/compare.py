"""Three distinct frozen scientific identities; no checkpoint relabeling."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
import compare_histories as h
from frozen_campaign import load
p=argparse.ArgumentParser();p.add_argument('--plans',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
old=ROOT/'output/native-shared-transition/aurora';broad=ROOT/'output/restored-wide-smooth/aurora';new=ROOT/'output/restored-six-shells/aurora'
cases={
 'A':h.load_native(old,'A',['campaigns/n16-native-transition-v2-a-evolution-segment01/plan.json']),
 'A-broad-smooth':h.load_native(broad,'A',['campaigns/n16-restored-wide-smooth-v1-a-evolution-segment01/plan.json'],load(broad/'launcher-restored-wide-smooth-v1/native_transition_campaign.py')),
 'A-six-shells':h.load_native(new,'A',a.plans,load(new/'launcher-restored-six-shells-v1/native_transition_campaign.py'))}
h.LABELS['A-broad-smooth']='Broad + smooth: 3 shells';h.COLORS['A-broad-smooth']='#b02871'
h.LABELS['A-six-shells']='Broad + smooth: 6 shells';h.COLORS['A-six-shells']='#008060'
result=h.render(cases,a.output,title='N16 restored broad-smooth A: radial shell refinement',
 footer='Exterior histories use each run’s candidate horizon; accepted mass gaps are retained.\nSix shells double radial sampling, keeping core/angular N16. Added interfaces and refined boundary nodes preclude a uniform-resolution convergence claim.')
print(json.dumps({k:{key:v[key] for key in ['status','last_accepted_t_over_M','accepted_horizon_rows','final_accepted_mass_error']} for k,v in result['cases'].items()},indent=2))
