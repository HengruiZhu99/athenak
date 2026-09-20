from pathlib import Path
import importlib.util,hashlib,json,datetime,os,subprocess
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/hz0693/research/TDE/athenak-review')
spec=importlib.util.spec_from_file_location('stage',REPO/'tst/regression/z4c_constraint_radiation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=(ROOT/'pilots-v2/pulse_oneblock/input.athinput').read_text()
folder=ROOT/'wormhole-control';folder.mkdir(exist_ok=False)
for block,key,value in [('problem','bh_background','schwarzschild_puncture'),('z4c','extrap_order',4),('time','tlim',100),('time','ndiag',100),('output4','dt',100)]:base=m.set_value(base,block,key,value)
(folder/'input.athinput').write_text(base)
exe=ROOT/'bin/athena-radiation-v2'
cmd=[str(exe),'-i',str(folder/'input.athinput'),'-d',str(folder),'-t','00:05:00']
record={'purpose':'Nonspinning zero-background-shift vacuum discriminator, not high-spin evolution or a physical stationary Einstein solution','source_base':'74b7691e','binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'command':cmd,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','radiation_weight_scope':'Uses unchanged r+1 leading model; not exact wormhole areal radius'}
(folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
with (folder/'run.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OMP_PROC_BIND':'false'})
record.update(status='completed',exit_code=r.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
