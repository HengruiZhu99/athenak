#!/usr/bin/env python3
"""Check unchanged initial values and quantify a two-step reference-jet change."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
import numpy as np
from restart_roundtrip import parse,payload

def run(before,after,original,out):
    out.mkdir(exist_ok=False);report={'executables':{},'comparisons':{}}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for tag,binary in [('before',before),('after',after)]:
        report['executables'][tag]={'path':str(binary),'sha256':sha(binary)}
    for steps in (0,2):
        arrays=[]
        for tag,binary in [('before',before),('after',after)]:
            case=out/f'{tag}-{steps}';case.mkdir()
            text=re.sub(r'^nlim = \d+$',f'nlim = {steps}',original.read_text(),flags=re.M)
            text=text.replace('dcycle=1','dcycle=0')
            inp=case/'input.athinput';inp.write_text(text)
            cmd=['mpiexec','-n','2',str(binary),'-i',str(inp)]
            with (case/'run.log').open('w') as log:
                subprocess.run(cmd,cwd=case,stdout=log,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'OMP_NUM_THREADS':'1'})
            checkpoint=sorted((case/'rst').glob('*.rst'))[-1]
            parsed=parse(checkpoint);assert parsed['cycle']==steps
            values=payload(parsed);assert np.isfinite(values).all();arrays.append(values)
            report[f'{tag}-{steps}']={'command':cmd,'input_sha256':sha(inp),
                'checkpoint_sha256':sha(checkpoint),'checkpoint':str(checkpoint)}
        equal=np.array_equal(arrays[0].view('u8'),arrays[1].view('u8'))
        if steps==0:assert equal,'reference-jet correction changed initial state values'
        report['comparisons'][str(steps)]={'bitwise_equal':equal,'scalar_values':int(arrays[0].size),
            'field_max_differences':[float(np.max(abs(arrays[0][:,:,f]-arrays[1][:,:,f]))) for f in range(39)]}
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['comparisons'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    run(a.before.resolve(),a.after.resolve(),a.input.resolve(),a.output.resolve())
