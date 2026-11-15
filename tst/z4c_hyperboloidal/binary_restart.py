#!/usr/bin/env python3
"""Actual zero-step native binary restart with rank change and policy rejection."""
import argparse,hashlib,json
from pathlib import Path
from restart_roundtrip import run,compare
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
fixture=out/'input.athinput';fixture.write_text(a.input.read_text()+
    '\n<time>\nintegrator = rk4_classical\ntlim = 1\nnlim = 0\nstart_time = 0\n'
    '<output1>\nfile_type = rst\ndt = 100\nsingle_file_per_rank = false\n')
binary=a.binary.resolve()
original=run(binary,out,'write2',2,['-i',str(fixture)])
restored=run(binary,out,'read4',4,['-r',original['path'],'output1/last_time=-1',
    'problem/binary_directory=/export-not-needed-on-restart'])
result={'roundtrip':compare(original,restored),'steps':0,'ranks':[2,4],
    'executable_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
for name,option in [('weight','z4c_hyperboloidal/weight_inner=.29'),
                    ('archive','problem/binary_identity='+'0'*64)]:
    result['reject_'+name]=run(binary,out,'reject_'+name,2,
       ['-r',original['path'],option],'mathematical parameters')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
