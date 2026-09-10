from pathlib import Path
import numpy as np,json,subprocess
r=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910')
a=np.loadtxt(next((r/'validation_v3/dchi_old').glob('*.hst')));b=np.loadtxt(next((r/'validation_v3/dchi_new').glob('*.hst')));assert np.array_equal(a,b)
rows=[]
for n in ['te_1e2','te_1e3','te_1e4']:
 d=r/'validation_v4'/n;assert (d/'run-status').read_text().strip()=='0'
 v=np.loadtxt(next(d.glob('*.hst')));assert np.isfinite(v).all() and v[-1,0]==6
 assert np.array_equal(v[v[:,0]<5],a[a[:,0]<5])
 rows.append(dict(case=n,final_time=float(v[-1,0]),blocks=int(v[-1,12]),maxlevel=int(v[-1,15]),minlapse=float(v[-1,14]),C2=float(v[-1,2]),C_Linf=float(v[-1,58])))
acct=subprocess.check_output(['sacct','-j','58148291','-X','-n','-P','--format=JobID,State,ExitCode'],text=True);assert '58148291|COMPLETED|0:0' in acct
result=dict(status='PASS',dchi_history_exact=True,bootstrap_histories_exact=True,calibration_job='58148291',cases=rows,selected_reference_threshold=1e-4,reason='Tightest tested tolerance reaching t6 within short calibration cap;317blocks versus254dchi. Selected before late-time outcomes.',scope='Short startup/switch qualification only; not long-time stability or spatial convergence.')
(r/'CALIBRATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
