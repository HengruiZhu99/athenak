from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/tst/regression'))
from z4c_background_restart import cohort
p=Path(sys.argv[1]);_,records=cohort(p,4,3)
assert all(v==0 for r in records for block in r['state'] for v in block)
result={'passed':True,'ranks':4,'blocks':records[0]['total'],'cycles':3,'time_M':records[0]['time'],'residual_max':0,'checkpoint_ghost_metric_validation':'passed separately by verify_checkpoint.py','stage_validity_csv_available':False,'note':'Short exact-zero check only, not perturbation stability.'}
(p/'zero-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
