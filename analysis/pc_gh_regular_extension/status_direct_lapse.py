"""Read a small live status snapshot without modifying or interpreting a run as passed."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
run=a.root/'stress/core256-R16'
result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),run=str(run),completed=(run/'completed.json').exists())
if (a.root/'stress-exit.json').exists():result['driver_exit']=json.loads((a.root/'stress-exit.json').read_text())
logs=sorted(run.glob('segment-*.log'));result['segments']=len(logs)
if logs:
    tail=subprocess.check_output(['tail','-n','20',str(logs[-1])],text=True)
    result['latest_log']=str(logs[-1]);result['fatal']=[x for x in tail.splitlines() if 'FATAL ERROR' in x]
    progress=[x for x in tail.splitlines() if x.startswith('elapsed=')]
    if progress:result['progress']=progress[-1]
mon=list(run.glob('*.hybrid.csv'))
if mon:
    tail=subprocess.check_output(['tail','-n','8000',str(mon[0])],text=True)
    steps={}
    for v in csv.reader(tail.splitlines()):
        if len(v)!=16 or v[3]!='3' or v[4]!='-2' or v[5]!='before':continue
        key=(int(v[0]),float(v[1]));steps.setdefault(key,{})[(v[6],v[7])]=v
    complete=[key for key,rows in steps.items() if len(rows)==100]
    if complete:
        rows=steps[max(complete)];row=next(iter(rows.values()))
        result['diagnostic_time']=float(row[1])+float(row[2]);result['all_extrema']={}
        for q in ['Rw','RQ','Ralpha','RL_direct','curl_p','curl_Q','curl_L','curl_B','min_eigenvalue','min_w','min_rho','abs_H']:
            row=rows[('all',q)]
            result['all_extrema'][q]=dict(value=float(row[8]),xyz=[float(x) for x in row[10:13]],level=int(row[13]))
print(json.dumps(result,indent=2))
