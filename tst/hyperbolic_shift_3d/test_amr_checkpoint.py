#!/usr/bin/env python3
"""Collective rejection tests against an actual completed AMR checkpoint."""
import argparse, hashlib, json, pathlib, subprocess
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
raw=pathlib.Path(a.checkpoint).read_bytes();results=[]
cases=[('truncated',raw[:-24],'wave','.00025','truncated'),
       ('identity',b'X'+raw[1:],'wave','.00025','identity mismatch'),
       ('field-layout',raw[:raw.index(b'\0')+1]+b'X'+raw[raw.index(b'\0')+2:],'wave','.00025','field layout mismatch'),
       ('checksum',raw[:-100]+bytes([raw[-100]^1])+raw[-99:],'wave','.00025','checksum mismatch'),
       ('timestep',raw,'wave','.0005','configuration mismatch'),
       ('initial-data',raw,'boost','.00025','configuration mismatch')]
for name,data,kind,dt,message in cases:
    checkpoint=out/(name+'.checkpoint');checkpoint.write_bytes(data)
    cmd=['mpiexec','-n','2',str(pathlib.Path(a.binary).resolve()),'12',kind,'6',dt,str(out/(name+"-out")),str(checkpoint),'dynamic']
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,text=True)
    (out/(name+'.log')).write_text(r.stdout)
    record=dict(name=name,command=cmd,returncode=r.returncode,expected_message=message,passed=r.returncode==2 and message in r.stdout,sha256=hashlib.sha256(data).hexdigest())
    results.append(record);checkpoint.unlink()
(out/'manifest.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['passed'] for r in results),results
print('All six collective checkpoint rejection tests passed.')
