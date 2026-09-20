from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/hz0693/research/TDE/athenak-boundary-radiation')
spec=importlib.util.spec_from_file_location('stage_tools',REPO/'tst/regression/z4c_constraint_radiation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
setv=module.set_value
exe=ROOT/'bin/athena-radiation-v1'
assert json.loads((ROOT/'stages-v1/results.json').read_text())['checks']['zero_exact_including_ghosts']
base=(ROOT.parent/'stability-modes-20260919/evolution/small_dx025.athinput').read_text()
cases=[('zero',0,4,2,20),('pulse_cubic',1e-8,4,2,100),('pulse_linear',1e-8,2,2,100),('pulse_wide',1e-8,4,4,100)]
if len(sys.argv)>1:cases=[c for c in cases if c[0] in sys.argv[1:]]
assert cases
def run(case):
    name,amplitude,order,radius,target=case
    folder=ROOT/'pilots-v1'/name;folder.mkdir(parents=True,exist_ok=False)
    text=base
    for block,key,value in [('z4c','characteristic_bc_source','physical_constraint_radiation'),
                            ('z4c','extrap_order',order),('time','tlim',target),
                            ('time','ndiag',100),('problem','vacuum_gauge_pulse_amplitude',amplitude),
                            ('mhd','debug_metric_before_c2p','true'),
                            ('z4c','characteristic_bc_diagnostics','true'),
                            ('z4c','characteristic_bc_diagnostic_interval',100),
                            ('output4','dt',25)]:text=setv(text,block,key,value)
    for axis in (1,2,3):
        text=setv(text,'mesh',f'nx{axis}',int(8*radius))
        text=setv(text,'mesh',f'x{axis}min',-radius)
        text=setv(text,'mesh',f'x{axis}max',radius)
        text=setv(text,'meshblock',f'nx{axis}',int(4*radius))
    (folder/'input.athinput').write_text(text)
    cmd=[str(exe),'-i',str(folder/'input.athinput'),'-d',str(folder),'-t','00:08:00']
    record={'case':name,'target_M':target,'command':cmd,'source_base':'74b7691e',
            'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
            'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running',
            'OMP_NUM_THREADS':2}
    (folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    with (folder/'run.log').open('w') as stream:
        p=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,
                         env={**os.environ,'OMP_NUM_THREADS':'2','OMP_PROC_BIND':'false'})
    record.update(status='completed',exit_code=p.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (folder/'exit_code.txt').write_text(str(p.returncode)+'\n')
    (folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(run,cases):print(json.dumps(result),flush=True)
