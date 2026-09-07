#!/usr/bin/env python3
"""Verify production component histories against independent global primary jets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart,global_state
from intrinsic_diagnostics import diagnostics,norms,component_names

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--launcher',default='');p.add_argument('--ranks',type=int,default=1)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);runs=[];records=[]
for dim in [2,3]:
    for order in [2,4,6]:
        case=f'fd{order}-{dim}d';seed=a.fixtures/(case+'-seed-multi.rst');folders=[]
        text=(a.fixtures/(case+'-seeded-multi')/'used.athinput').read_text().replace('nlim = 3','nlim = 1').replace('tlim = 0.0003','tlim = 0.0001')
        for enabled in [False,True]:
            folder=a.output/(case+('-diagnostic' if enabled else '-control'));folder.mkdir();folders.append(folder)
            inp=folder/'used.athinput';inp.write_text(text.replace('formulation = intrinsic_clean',f'formulation = intrinsic_clean\nintrinsic_diagnostics = {str(enabled).lower()}'))
            command=shlex.split(a.launcher)+[str(a.binary.resolve()),'-i',str(inp.resolve()),'-r',str(seed.resolve())]
            with (folder/'run.log').open('w') as log:r=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            runs.append(dict(command=command,returncode=r.returncode,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),restart_sha256=hashlib.sha256(seed.read_bytes()).hexdigest()))
            (a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
            assert r.returncode==0,(folder,(folder/'run.log').read_text())
            assert re.findall(r'Number of parallel ranks = (\d+)',(folder/'run.log').read_text())==[str(a.ranks)]
        baseline=read_restart(sorted((folders[0]/'rst').glob('*.rst'))[-1])
        final=read_restart(sorted((folders[1]/'rst').glob('*.rst'))[-1])
        assert np.array_equal(baseline['state'],final['state'])
        files=sorted(folders[1].glob('intrinsic-diagnostics-*.csv'));assert len(files)==2
        assert not list(folders[0].glob('intrinsic-diagnostics-*.csv'))
        observations=[]
        for file,data in zip(files,[read_restart(seed),final]):
            with file.open() as f:rows=list(csv.DictReader(f))
            assert len(rows)==89 and len({r['component'] for r in rows})==89
            state=global_state(data);spacing=data['domain'][6:9]
            fields=diagnostics(state,spacing,order,physical_operator='direct')
            expected=norms(fields,float(np.prod(data['domain'][3:6]-data['domain'][:3])))
            lookup={name:(key,n) for key,names in component_names().items() for n,name in enumerate(names)}
            error=0.;location_error=0.
            for row in rows:
                assert row['region']=='full' and int(row['cycle'])==data['cycle'] and float(row['time'])==data['time']
                key,n=lookup[row['component']];stats=expected[key]
                assert abs(float(row['volume'])-stats['volume'])<2e-12 and int(row['cells'])==stats['cells']
                for name in ['L1_integral','L2_integral','RMS','maximum']:
                    error=max(error,abs(float(row[name])-stats[name][n])/(1+abs(stats[name][n])))
                gid=int(row['gid']);x,y,z,level=data['locations'][gid];bx,by,bz=data['mb'][1:4]
                k,j,i=[int(row[v]) for v in ['k','j','i']]
                assert int(row['logical_level'])==level
                index=(z*bz+k,y*by+j,x*bx+i)
                signed=fields[key][(n,*index)]
                error=max(error,abs(float(row['signed_at_max'])-signed)/(1+abs(signed)),abs(abs(signed)-stats['maximum'][n])/(1+stats['maximum'][n]))
                coordinates=data['domain'][:3]+(np.array(index[::-1])+.5)*spacing
                location_error=max(location_error,float(np.max(abs(coordinates-np.array([float(row[v]) for v in ['x','y','z']])))))
            observations.append(dict(cycle=data['cycle'],time=data['time'],component_error=error,location_error=location_error))
            assert error<=2e-12 and location_error<=2e-12,observations[-1]
        record=dict(case=case,ranks=a.ranks,status='PASS',neutral_bitwise=True,observations=observations)
        records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
(a.output/'summary.json').write_text(json.dumps(dict(status='PASS',cases=6,ranks=a.ranks,components=89,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),scope='actual full-domain component history, initial/final synchronized states; independent direct-primary-jet physical oracle and all reduction/curl norms/locations'),indent=2)+'\n')
