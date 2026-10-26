#!/usr/bin/env python3
"""Collect completed local controls; retain hashes instead of large payloads."""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def copy_case(source,dest):
    dest.mkdir(parents=True,exist_ok=False);index={}
    for p in sorted(source.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(source)
        if p.name in ['athena_frozen','probe_checkpoint.bin'] or p.name.startswith(('cubed_sphere_reference_','cubed_sphere_initial_profile_')):
            index[str(rel)]={'sha256':sha(p),'bytes':p.stat().st_size,'retained_at':str(p.resolve())}
            continue
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if p.stat().st_size>2*1024*1024:
            target=target.with_name(target.name+'.gz');target.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
        else:shutil.copy2(p,target)
    (dest/'payload-index.json').write_text(json.dumps(index,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',type=Path)
    ap.add_argument('--destination',type=Path)
    args=ap.parse_args()
    if args.case or args.destination:
        assert args.case and args.destination
        copy_case(args.case.resolve(),args.destination.resolve())
        return
    root=Path(__file__).resolve().parents[2]
    src=root/'build-cubed-sphere/research-20261013'
    cap=root/'docs/validation/cubed-sphere-spherical-comparison-20261013/cpu'
    for name,dest in [('spherical-CPU-B16-S12-k10','wide-collar-capped'),
        ('device-CPU-parity','device-port-parity'),('CPU-matched-one-step-admitted','matched-one-step'),
        ('CPU-matched-one-step','rejected-binary-override')]:
        copy_case(src/name,cap/dest)


if __name__=='__main__':main()
