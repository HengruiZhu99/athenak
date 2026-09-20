from pathlib import Path
import re,json,os,subprocess,concurrent.futures,hashlib,datetime,sys
b=Path(__file__).resolve().parent
exe=b/'athena-covariant-sources'
assert json.loads((b/'source-regression.json').read_text())['passed']
assert hashlib.sha256(exe.read_bytes()).hexdigest()==json.loads((b/'source-regression.json').read_text())['binary_sha256']
base=(b.parent/'evolution/small_dx025.athinput').read_text()
def setv(s,block,key,value):
 m=re.search(rf'(<{block}>\n)(.*?)(?=\n<|\Z)',s,re.S);assert m
 body=m[2];pat=rf'(?m)^{key}\s*=.*$'
 body=re.sub(pat,f'{key} = {value}',body) if re.search(pat,body) else body+f'\n{key} = {value}\n'
 return s[:m.start()]+m[1]+body+s[m.end():]
for a in (1,2,3):base=setv(base,'meshblock',f'nx{a}',16)
for block,k,v in [('time','tlim',300),('time','cfl_number',.15),
                  ('output4','dt',100),('mhd','debug_metric_before_c2p','true'),
                  ('z4c','ccz4_covariant_sources','true')]:base=setv(base,block,k,v)
cases=[('covariant_alpha01',False,.1),('covariant_const01',True,.1),('covariant_const03',True,.3),('covariant_const10',True,1.0)]
if len(sys.argv)>1:cases=[c for c in cases if c[0] in sys.argv[1:]]
def run(case):
 name,scaled,kappa=case;d=b/name;d.mkdir(exist_ok=False);s=setv(base,'z4c','damp_lapse_scaled',str(scaled).lower());s=setv(s,'z4c','damp_kappa1',kappa)
 (d/'input.athinput').write_text(s)
 args=[str(exe),'-i',str(d/'input.athinput'),'-d',str(d),'-t','00:20:00']
 record={'case':name,'command':args,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'OMP_NUM_THREADS':2,'status':'running','binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
 (d/'manifest.json').write_text(json.dumps(record,indent=2))
 with (d/'run.log').open('w') as f:p=subprocess.run(args,env={**os.environ,'OMP_NUM_THREADS':'2','OMP_PROC_BIND':'false'},stdout=f,stderr=subprocess.STDOUT)
 record.update(status='completed',exit_code=p.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(d/'manifest.json').write_text(json.dumps(record,indent=2));(d/'exit_code.txt').write_text(str(p.returncode)+'\n');return record
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(run,cases):print(json.dumps(r),flush=True)
