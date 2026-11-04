#!/usr/bin/env python3
"""Preserve small receipts with hashes; leave all large original states in place."""
import hashlib
import json
from pathlib import Path
import sys
import struct
import tarfile
from storage import require_capacity

root=Path(sys.argv[1]).resolve()
number=int(sys.argv[2]) if len(sys.argv)>2 else 1
assert number in (1,2)
job='job'+str(number)
state=json.loads((root/job/'state.json').read_text())
if state['status']=='running':
    raise RuntimeError('Do not archive mutable running outputs')
require_capacity(root,planned=100_000_000)
archive=root/(job+'-evidence.tar.gz')
if archive.exists():
    raise RuntimeError('Evidence archive already exists; verify rather than overwrite')
selected=[]
for directory in [job,'launcher']:
    for path in sorted((root/directory).rglob('*')):
        if path.is_file() and path.suffix in ['.json','.jsonl','.txt','.csv','.log','.py','.pbs']:
            selected.append(path)
for name in ['source-manifest.json','launcher-manifest.json','local-qualification.json','plan1.json',
             'submission1.json','pre-submit-jobs.txt','pre-submit-queue.txt','build-storage-before.json',
             'build-storage-after.json','build-modules.txt','job1-modules.txt','job1-pbs.log',
             'configure.log','build.log','plan2.json','submission2.json','pre-submit2-jobs.txt','pre-submit2-queue.txt','job2-modules.txt','job2-pbs.log','binaries.sha256','build/CMakeCache.txt','build/compile_commands.json']:
    path=root/name
    if path.exists():selected.append(path)
# Independently verify RF1 framing and admission before recording receipts.
# Frozen first-source reports the 208-byte fixed prefix; the true RF1 payload
# additionally follows a uint64 length and its exact configuration identity.
checks={}
for name in [p.name.removesuffix('.report.json') for p in (root/job).glob('*n16.report.json')]:
    checkpoint=root/job/(name+'.checkpoint')
    report_path=root/job/(name+'.report.json')
    if not checkpoint.exists() or not report_path.exists():
        continue
    report=json.loads(report_path.read_text())
    with checkpoint.open('rb') as stream:
        header=struct.unpack('<16s4i3Q2d',stream.read(72))
        stream.seek(208)
        length=struct.unpack('<Q',stream.read(8))[0]
        if length>8192:
            raise RuntimeError('Unexpected RF1 identity size')
        identity=stream.read(length).decode()
    magic,n,nf,kind,admitted,cells,surface,steps,time_value,dt=header
    expected_size=216+length+8*(2*nf*cells+2*surface)
    valid=(magic.rstrip(b'\0')==b'HYP3D_MPI_RF1' and admitted==1 and nf==31 and n==16
           and identity==report['formulation_identity'] and checkpoint.stat().st_size==expected_size)
    if report.get('checkpoint_admitted') and not valid:
        raise RuntimeError('RF1 checkpoint framing/admission mismatch: '+name)
    checks[name]=dict(valid=valid,payload_offset=216+length,bytes=expected_size,
                      steps=steps,time=time_value,dt=dt,identity=identity)
(root/(job+'-checkpoint-framing.json')).write_text(json.dumps(checks,indent=2)+'\n')
selected.append(root/(job+'-checkpoint-framing.json'))
manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected}
with tarfile.open(archive,'w:gz') as out:
    for path in selected:out.add(path,arcname=str(path.relative_to(root)),recursive=False)
receipt=dict(archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             files=manifest,storage=require_capacity(root),large_artifacts_retained=job+'/state.json contains checkpoint and failure-capture hashes; original files remain in '+job)
(root/(job+'-evidence-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))
