"""Three restored-A histories; each formulation keeps its own frozen identity."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
import compare_histories as h
from frozen_campaign import load
p=argparse.ArgumentParser();p.add_argument('--plans',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
old=ROOT/'output/native-shared-transition/aurora'
wide=ROOT/'output/wide-collar-a/aurora'
new=ROOT/'output/restored-wide-smooth/aurora'
cases={
 'A':h.load_native(old,'A',['campaigns/n16-native-transition-v2-a-evolution-segment01/plan.json']),
 'A-wide':h.load_native(wide,'A',['campaigns/n16-wide-collar-a-v1-a-evolution-segment01/plan.json'],load(wide/'launcher-wide-collar-a-v1/native_transition_campaign.py')),
 'A-broad-smooth':h.load_native(new,'A',a.plans,load(new/'launcher-restored-wide-smooth-v1/native_transition_campaign.py'))}
h.LABELS['A-wide']='A: collar 0.7–0.9, original slice';h.COLORS['A-wide']='#1769aa'
h.LABELS['A-broad-smooth']='A: collar 0.6–0.9, C6 slice 4M–12M';h.COLORS['A-broad-smooth']='#b02871'
result=h.render(cases,a.output,title='N16 retained restoration: broader collar and improved initial slice')
print(json.dumps({k:{key:v[key] for key in ['status','last_accepted_t_over_M','accepted_horizon_rows','final_accepted_mass_error']} for k,v in result['cases'].items()},indent=2))
