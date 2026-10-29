#!/usr/bin/env python3
"""Bounded matched oblique interface probe; failed cases remain failed evidence."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--mpiexec',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
for label,layers,kappa,shifted in [('control',3,0,0),('thin-zero',1,0,1),('thin-paired',1,1,1)]:
 prefix=out/label;cmd=[str(a.mpiexec.resolve()),'-n','1',str(a.binary.resolve()),'12',str(prefix),'oblique','.04','.0005','-','-','350','.01','.001','6','12','2']
 env=dict(os.environ,HYPERBOLIC_TRACE_LAYERS=str(layers),HYPERBOLIC_TRACE_KAPPA=str(kappa),HYPERBOLIC_TRACE_SHIFTED=str(shifted))
 t=time.monotonic()
 with prefix.with_suffix('.log').open('w') as log:
  r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=410)
 record=dict(name=label,command=cmd,parameters=dict(layers=layers,kappa=kappa,shifted=shifted),returncode=r.returncode,seconds=time.monotonic()-t)
 f=prefix.with_suffix('.report.json')
 if f.exists():record['report']=json.loads(f.read_text())
 rows.append(record);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(label,record.get('report',{}).get('status'),record.get('report',{}).get('t'),record['seconds'],flush=True)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
