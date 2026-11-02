"""Compare A and A-width histories with separate frozen helpers and raw roots."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
import compare_histories as h
from frozen_campaign import load
p=argparse.ArgumentParser();p.add_argument('--plans',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
old=ROOT/'output/native-shared-transition/aurora';new=ROOT/'output/wide-collar-a/aurora'
helper=load(new/'launcher-wide-collar-a-v1/native_transition_campaign.py')
cases={'A':h.load_native(old,'A',['campaigns/n16-native-transition-v2-a-evolution-segment01/plan.json']),
       'A-wide':h.load_native(new,'A',a.plans,helper)}
h.LABELS['A-wide']='A, restoring collar widened to 0.7–0.9';h.COLORS['A-wide']='#1769aa'
result=h.render(cases,a.output,title='N16 A: original versus wider restoring collar')
print(json.dumps({k:{key:v[key] for key in ['status','last_accepted_t_over_M','accepted_horizon_rows','final_accepted_mass_error']} for k,v in result['cases'].items()},indent=2))
