#!/usr/bin/env python3
"""Collect new bounded-run records without editing earlier evidence capsules."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[2]
    work=root/'build-cubed-sphere/research-20261012'
    capsule=root/'docs/validation/cubed-sphere-refinement-20261012'
    destination=capsule/'mapped-probe'; destination.mkdir(exist_ok=True)
    for directory in sorted(work.glob('conformal-*')):
        receipt=directory/'receipt.json'
        if not receipt.exists():
            continue
        target=destination/directory.name; target.mkdir(exist_ok=False)
        metadata={}
        for p in sorted(directory.iterdir()):
            if not p.is_file():continue
            metadata[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
            if p.stat().st_size<1024*1024 and p.name!='athena_frozen':
                shutil.copy2(p,target/p.name)
        (target/'payload-index.json').write_text(json.dumps(metadata,indent=2)+'\n')
    smoke=capsule/'static-minkowski-smoke'; smoke.mkdir(exist_ok=False)
    for p in (work/'static-minkowski-smoke2').iterdir():
        if p.is_file() and p.stat().st_size<1024*1024:shutil.copy2(p,smoke/p.name)
    shutil.copy2('/tmp/cubed-sphere-static-smoke2.log',smoke/'run.log')
    rejected=capsule/'rejected-static-input';rejected.mkdir(exist_ok=False)
    shutil.copy2(work/'static-minkowski-smoke/input.athinput',rejected/'input.athinput')
    shutil.copy2('/tmp/cubed-sphere-static-smoke.log',rejected/'run.log')
    logs=capsule/'build-and-tests';logs.mkdir(exist_ok=False)
    for name in ['cubed-sphere-conformal-build.log','cubed-sphere-conformal-build-repair1.log',
                 'cubed-sphere-conformal-tested-build.log','cubed-sphere-conformal-degree-build.log',
                 'cubed-sphere-conformal-profile-build.log','cubed-sphere-final-ctest.log']:
        shutil.copy2(Path('/tmp')/name,logs/name)
    source=capsule/'current-source';source.mkdir(exist_ok=False)
    paths=[*root.glob('src/mesh/cubed_sphere/*'),*root.glob('tst/cubed_sphere/*.py'),
           *root.glob('tst/cubed_sphere/*.cpp'),*root.glob('tst/cubed_sphere/*.pbs'),
           root/'CMakeLists.txt',root/'src/CMakeLists.txt',
           root/'inputs/z4c/cubed_sphere_conformal_probe.athinput']
    hashes={}
    for p in sorted(paths):
        if not p.is_file():continue
        rel=p.relative_to(root); hashes[str(rel)]=sha(p)
        out=source/rel; out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    (capsule/'current-source-index.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Collected local probe records; large reference payloads and executables are hash-only.')


if __name__=='__main__':
    main()
