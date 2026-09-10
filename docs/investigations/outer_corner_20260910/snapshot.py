from pathlib import Path
import json,sys
sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-t200-20260909')
from criterion import read_history
root=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 d=root/name
 files=list(d.glob('*.hst'))
 out={'case':name,'run_status':(d/'run-status').read_text().strip() if (d/'run-status').exists() else None}
 if files:
  try:
   rows=read_history(files[0]);last=rows[-1]
   out.update({k:last[k] for k in ['time','dt','minLapse','maxAbsKret','C-Linf','C-rho','C-z','Z-Linf','Z-rho','Z-z','nmb_total','maxRefLev']})
   out['peak_kretschmann']=max(r['maxAbsKret'] for r in rows)
  except Exception as e:out['history_error']=str(e)
 print(json.dumps(out))
