"""Small finite-radius infrastructure control after the larger AMR suite cap."""
import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tst/cubed_sphere'))
from check_evolution import make_input
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
a.output.mkdir(parents=True,exist_ok=True);binary=a.binary.resolve();result={}
for data in ['minkowski','gauge_wave']:
    folder=(a.output/data).resolve();folder.mkdir(exist_ok=True)
    changes={('mesh','nx'+str(i)):16 for i in [1,2,3]}
    changes.update({('time','nlim'):3,('cubed_sphere','nr'):12,
        ('cubed_sphere','nangular'):8,('problem','initial_data'):data})
    inp=folder/'input.athinput';inp.write_text(make_input(changes))
    command=[str(binary),'-i',str(inp),'-d',str(folder)]
    with (folder/'run.log').open('w') as log:
        run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    assert run.returncode==0,(data,run.returncode)
    with (folder/'cubed_sphere_errors.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2
    tolerance=1.e-11 if data=='minkowski' else 1.e-4
    assert max(float(x['linf']) for x in rows)<tolerance,rows
    result[data]=dict(command=command,returncode=run.returncode,metrics=rows,tolerance=tolerance)
result['scope']='Uniform finite-radius regular-Z4c infrastructure smoke only; no hyperboloidal or new-gauge evolution.'
result['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
