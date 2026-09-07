import hashlib,json,tarfile
from pathlib import Path
root=Path('/scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-mesh-001')
folders=['oracle-cuda-001','decomposition-cuda-serial-001','decomposition-cuda-mpi-001','snapshot-diagnostics-cuda-001','oracle-cuda-repeat-fix-001','decomposition-cuda-seeded-001','repeat-fix-controls-001','memcheck-001','memcheck-ob1-001','memcheck-ob1-pt2pt-001']
paths=[]
for name in folders:
    paths.extend(p for p in (root/name).rglob('*') if p.is_file())
paths.extend(p for p in root.iterdir() if p.is_file() and p.suffix in ['.json','.log','.sh','.sha256','.csv','.exit'])
paths.extend([root/'build/CMakeCache.txt',root/'build/src/athena',root/'before-repeat-fix/athena',root/'repeat-fix/run.sh',root/'repeat-fix/tests/compare_intrinsic_backends.py'])
paths=sorted(set(paths))
manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
(root/'final-artifact-manifest-001.json').write_text(json.dumps(dict(root=str(root),files=manifest),indent=2)+'\n')
with tarfile.open(root/'compact-evidence-001.tar.gz','w:gz') as tar:
    tar.add(root/'final-artifact-manifest-001.json',arcname='final-artifact-manifest-001.json')
    for p in paths:
        if p.suffix in ['.json','.log','.sh','.sha256','.csv','.exit','.athinput'] or p.name=='CMakeCache.txt':
            tar.add(p,arcname=str(p.relative_to(root)))
print('archived',len(paths),'manifest entries; archive bytes', (root/'compact-evidence-001.tar.gz').stat().st_size)
