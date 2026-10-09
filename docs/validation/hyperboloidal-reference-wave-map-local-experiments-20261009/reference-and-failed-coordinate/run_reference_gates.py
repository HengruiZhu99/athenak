"""One fixed local reference/synthetic readback; independent oracle follows."""
from pathlib import Path
import datetime,hashlib,json,math,subprocess,time
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
recipe=json.loads((P/'reference-local-recipe.json').read_text());assert sha(P/'reference-local-recipe.json')=='3e7107cdb0b3904a8881a3ee8bb25b1e8b25e0dba2b266f50ac2238f664cad36'
D=P/'reference-gate-attempt001';assert not D.exists();D.mkdir();records=[]
for target in ('synthetic','reference'):
 for mode in ('release','debug'):
  build=P/f'build-attempts/{target}-{mode}-001';br=json.loads((build/'receipt.json').read_text());exe=Path(br['executable_path']);assert br['exit_code']==0 and sha(exe)==br['executable_sha256']
  data=(P/'reference-radii.txt').read_bytes() if target=='reference' else b''
  start=time.monotonic();run=subprocess.run([str(exe)],input=data,capture_output=True)
  name=target+'-'+mode;(D/(name+'.stdout')).write_bytes(run.stdout);(D/(name+'.stderr')).write_bytes(run.stderr)
  r={'target':target,'mode':mode,'command':[str(exe)],'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':run.returncode,'seconds':time.monotonic()-start,'executable_sha256':sha(exe),'build_receipt_sha256':sha(build/'receipt.json'),'stdout_sha256':sha(D/(name+'.stdout')),'stderr_sha256':sha(D/(name+'.stderr'))}
  if run.returncode==0 and target=='reference':
   rows=[[float(x) for x in line.split()] for line in run.stdout.decode().splitlines()];r.update(rows=len(rows),columns=len(rows[0]),finite=all(math.isfinite(x) for row in rows for x in row),old_consumed_max_scaled=max(row[1] for row in rows),old_consumed_max_absolute=max(row[2] for row in rows))
   r['passed']=len(rows)==len(recipe['radii']) and all(len(row)==recipe['output_columns'] for row in rows) and r['finite'] and r['old_consumed_max_scaled']<=recipe['gates']['old_consumed_scaled_max']
  elif run.returncode==0:
   output=json.loads(run.stdout);r.update(output=output,passed=output['algebra_scaled']<=5e-13 and output['tail_relative']<=5e-13 and output['tail_zero_value'] and output['tail_fourth_coefficient']>0 and output['derivative_coefficients']==0)
  else:r['passed']=False
  assert sha(exe)==br['executable_sha256'];records.append(r)
r={'kind':'local complete-reference/synthetic owner gates','recipe_sha256':sha(P/'reference-local-recipe.json'),'driver_sha256':sha(Path(__file__)),'results':records,'passed_owner_reference_and_synthetic':all(x['passed'] for x in records),'independent_high_precision_reference_gate':'pending separate oracle','coordinate_lift_or_gauge_query':False}
(D/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'receipt':str(D/'receipt.json'),'sha256':sha(D/'receipt.json'),'results':records},indent=2));assert r['passed_owner_reference_and_synthetic']
