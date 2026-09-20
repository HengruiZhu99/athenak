from pathlib import Path
import concurrent.futures,datetime,hashlib,importlib.util,json,os,subprocess
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/hz0693/research/TDE/athenak-boundary-radiation')
spec=importlib.util.spec_from_file_location('stage_tools',REPO/'tst/regression/z4c_constraint_radiation.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
base=(ROOT/'pilots-v2/pulse_oneblock/input.athinput').read_text()
cases=[('second_plm_nofofc_trumpet',False,2,2),('quadratic_trumpet',False,4,3)]
exe=ROOT/'bin/athena-radiation-v2'
def run(case):
 name,flat,ng,order=case
 folder=ROOT/'isolation-v2'/name;folder.mkdir(parents=True,exist_ok=False)
 edits=[('mesh','nghost',ng),('z4c','extrap_order',order),('time','tlim',50),('time','ndiag',50)]
 if ng==2:edits += [('mhd','reconstruct','plm'),('mhd','fofc','false')]
 if flat:edits += [('problem','bh_mass',0),('problem','bh_background','kerr_schild'),('coord','minkowski','true'),('z4c','characteristic_radiation_areal_shift',0),('z4c','history_excise_ks_horizon','false'),('z4c','history_excise_ks_radius',0)]
 text=base
 for block,key,value in edits:text=mod.set_value(text,block,key,value)
 (folder/'input.athinput').write_text(text)
 cmd=[str(exe),'-i',str(folder/'input.athinput'),'-d',str(folder),'-t','00:08:00']
 record={'case':name,'command':cmd,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'}
 (folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 with (folder/'run.log').open('w') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OMP_PROC_BIND':'false'})
 record.update(status='completed',exit_code=r.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:list(p.map(run,cases))
