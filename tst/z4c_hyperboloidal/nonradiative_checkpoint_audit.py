#!/usr/bin/env python3
"""Stream-check completed native checkpoints: hashes, endpoint and all 39 fields."""
import argparse,hashlib,json,time
from pathlib import Path
from nonradiative_wide import require_completed_run
import numpy as np
import restart_roundtrip
from restart_roundtrip import parse

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def run(root,out):
    start=time.monotonic();result={'scope':'Read-only streaming audit; no evolution or extraction. All 31 fields and eight persistent chi-record scalars are checked.',
        'script_sha256':sha(Path(__file__)),'reader_sha256':sha(Path(restart_roundtrip.__file__)),'checkpoints':{}}
    for folder in sorted((root/'runs').iterdir()):
        if not folder.is_dir():continue
        require_completed_run(folder)
        receipt=json.loads((folder/'receipt.json').read_text())
        records=[(c['name'],Path(c['final']),c,c['cycles']) for c in receipt.get('cases',[])]
        if 'reference_probe' in receipt:
            c=receipt['reference_probe'];p=sorted((folder/'reference-control-N12/rst').glob('*.rst'))[-1]
            records.append(('reference-control-N12',p,c,c['cycles']))
        if folder.name.startswith('halfdt-'):
            p=sorted((folder/'gauge-N12-halfdt/rst').glob('*.rst'))[-1]
            records.append(('gauge-N12-halfdt',p,receipt,receipt['cycle']))
        for name,path,record,cycles in records:
            info=parse(path);assert info['cycle']==cycles and abs(info['time']-record['time'])<1e-12
            rows=info['record_blocks']*info['stride']//(39*8);maximum=np.zeros(39);h=hashlib.sha256()
            with path.open('rb') as stream:
                h.update(stream.read(info['payload_at']))
                for begin in range(0,rows,65536):
                    count=min(65536,rows-begin);block=stream.read(count*39*8)
                    assert len(block)==count*39*8
                    h.update(block);chunk=np.frombuffer(block,dtype='=f8').reshape(-1,39)
                    assert np.all(np.isfinite(chunk)),(path,begin)
                    maximum=np.maximum(maximum,np.max(abs(chunk),axis=0))
            digest=h.hexdigest();assert digest==record['final_sha256'],path
            result['checkpoints'][folder.name+'/'+name]={'path':str(path),'sha256':digest,'cycle':cycles,
                'time_M':3*info['time'],'finite_scalar_values':rows*39,'field_abs_maxima':maximum.tolist()}
    assert len(result['checkpoints'])==8,'Require six principal cases, reference control and half-step control'
    result['seconds']=time.monotonic()-start
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checkpoints':len(result['checkpoints']),'seconds':result['seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.root.resolve(),a.output.resolve())
