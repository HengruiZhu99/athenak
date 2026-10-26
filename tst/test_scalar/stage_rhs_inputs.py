#!/usr/bin/env python3
"""Hard-link verified native states for a read-only RHS audit across restarts."""
import argparse,hashlib,json,os
from pathlib import Path
from scalar_evidence import parent
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
if a.output.exists():raise RuntimeError('Existing staged audit inputs are immutable')
if not (run/'complete.txt').exists() or 'time=100 ' not in (run/'complete.txt').read_text():raise RuntimeError('A completed t100 lineage is required')
files={}
for t in [2,6,12,100]:
 name=f'fields-t{t:.3f}.f64';current=run
 while not (current/name).exists():
  current=parent(current)
  if current is None:raise FileNotFoundError(name)
 files[name]=current/name
files['checkpoint.f64']=run/'checkpoint.f64'
verified={}
for name,source in files.items():
 receipt=source.parent/('restart-receipt.json' if (source.parent/'restart-receipt.json').exists() else 'receipt.json');record=json.loads(receipt.read_text());digest=sha(source)
 if record['returncode']!=0 or record['files'].get(name)!=digest:raise RuntimeError('Native state/receipt hash mismatch: '+str(source))
 verified[name]=dict(source=str(source),sha256=digest,receipt_sha256=sha(receipt))
a.output.mkdir(parents=True)
for name,source in files.items():os.link(source,a.output/name)
(a.output/'input-provenance.json').write_text(json.dumps(dict(run=str(run),files=verified,native_states_added=0,scope='Hard links to unchanged verified native data; read-only RHS evaluation only.'),indent=2)+'\n')
print(json.dumps(dict(files=len(files),native_states_added=0)))
