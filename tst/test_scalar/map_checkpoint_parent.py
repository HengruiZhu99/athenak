#!/usr/bin/env python3
"""Map collected checkpoint ancestry without editing any native evidence."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--child',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();child=a.child.resolve();parent=a.parent.resolve();record=child/'restart-invocation.json';inv=json.loads(record.read_text());output=child/'analysis-parent.json'
if output.exists():raise RuntimeError('Existing ancestry mapping is immutable')
if sha(parent/'checkpoint.f64')!=inv['checkpoint_sha256']:raise RuntimeError('Parent checkpoint SHA256 differs from the native restart input')
parent_inv=parent/('restart-invocation.json' if (parent/'restart-invocation.json').exists() else 'invocation.json');pi=json.loads(parent_inv.read_text());binary=pi['binary_sha256'] if 'binary_sha256' in pi else pi['binary']
if binary!=inv['binary_sha256']:raise RuntimeError('Parent executable differs from continuation executable')
old=json.loads((parent/'parameters.json').read_text());new=json.loads((child/'parameters.json').read_text())
if any(new.get(key)!=value for key,value in old.items() if key!='tlim'):raise RuntimeError('Parent scalar/geometry parameters differ')
pg=json.loads((parent/'frozen-integrity.json').read_text());cg=json.loads((child/'frozen-integrity.json').read_text());keys=['physical_before','reference_before','physical_derivatives_before','reference_derivatives_before']
if not pg['unchanged'] or not cg['unchanged'] or any(pg[key]!=cg[key] for key in keys):raise RuntimeError('Parent and child frozen coefficients differ')
result=dict(native_parent=inv['parent'],local_parent=str(parent),checkpoint_sha256=inv['checkpoint_sha256'],binary_sha256=binary,restart_invocation_sha256=sha(record),geometry={key:pg[key] for key in keys},scope='Analysis-only path translation. Original restart metadata, receipt and checkpoint bytes remain unchanged.')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
