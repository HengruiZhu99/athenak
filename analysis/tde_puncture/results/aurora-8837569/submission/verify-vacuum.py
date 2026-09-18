#!/usr/bin/env python3
import csv,json,math,re,sys
from pathlib import Path
run=Path(sys.argv[1]);ranks=int(sys.argv[2]);target=float(sys.argv[3]);result={'passed':True,'ranks':ranks,'target':target}
for pattern in ['z4c_balance_rank*.csv','z4c_geometry_rank*.csv']:
 files=sorted(run.glob(pattern));rows=[r for f in files for r in csv.DictReader(f.open())]
 ok=len(files)==ranks and bool(rows) and all(math.isfinite(float(r['max_abs'])) and float(r['max_abs'])==0 and int(r['bit_mismatch'])==0 and int(r['nonfinite'])==0 for r in rows)
 result[pattern]={'files':len(files),'rows':len(rows),'exact_zero':ok};result['passed'] &= ok
hist={}
for f in run.glob('*.hst'):
 ls=f.read_text().splitlines();names=re.findall(r'\[\d+\]=([^\s]+)',next((x for x in ls if '[1]=' in x),''));rows=[[float(x) for x in l.split()] for l in ls if l.strip() and not l.startswith('#')]
 ok=bool(rows) and all(math.isfinite(x) for r in rows for x in r)
 if 'Theta-max' in names:
  ok &= all(r[names.index('Theta-max')]==0 and r[names.index('bad-metric')]==0 for r in rows)
  ok &= all(r[names.index(n)]>0 for r in rows for n in ['alpha-min','chi-min','detg-min'])
 hist[f.name]={'last_time':rows[-1][0] if rows else None,'valid':ok};result['passed'] &= ok
result['histories']=hist;result['passed'] &= len(hist)==3
log=(run/'run.log').read_text(errors='replace');result['no_fatal']='### FATAL ERROR' not in log;result['passed'] &= result['no_fatal']
result['numerical_valid']=bool(result['passed'])
result['wallclock_stop']='Terminating on wall clock limit' in log
if target>0:
 result['target_reached']=bool(hist) and all(v['last_time'] is not None and abs(v['last_time']-target)<1e-8 for v in hist.values())
 result['passed'] &= result['target_reached']
else:
 result['three_cycles']='Terminating on cycle limit' in log and 'cycle=3' in log;result['passed'] &= result['three_cycles']
(run/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] or (result['numerical_valid'] and result['wallclock_stop']) else 1)
