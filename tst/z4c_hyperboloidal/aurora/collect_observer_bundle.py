#!/usr/bin/env python3
"""Archive one terminal successful observer qualification, preserving all outputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    status = subprocess.run(['qstat','-xf',str(args.job)],capture_output=True,text=True,check=True).stdout
    assert 'job_state = F' in status and 'Exit_status = 0' in status
    case = ROOT/f'runs/step-observer-nodes{args.nodes}'
    assert (case/'pipeline-complete.txt').is_file()
    for name in (['fresh8','prefix2'] if args.nodes == 2 else ['fresh8','resume8']):
        report = json.loads((case/name/'operational.json').read_text())
        assert report['operational_passed'] and report['ranks'] == args.nodes*12
        assert (case/name/'exit-code.txt').read_text().strip() == '0'
    out = ROOT/f'validation/step-observer-artifacts-{args.job}'
    out.mkdir(exist_ok=False)
    (out/'job-final.txt').write_text(status)
    validation = ROOT/'validation/step-observer'
    files = [p for p in case.rglob('*') if p.is_file() and 'cache' not in p.relative_to(case).parts]
    files += [p for p in validation.rglob('*') if p.is_file() and
              p.suffix != '.o' and p.name != 'athena-step-observer']
    files += [ROOT/p for p in ['build-sycl/runtime-object-sha256.json',
               'build-sycl/source-sha256.json','source/source-revision.json']]
    files.append(out/'job-final.txt')
    files = sorted(set(files))
    manifest = {str(p.relative_to(ROOT)):dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
    (out/'files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in files: tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    report=dict(schema='native-step-observer-collection-v1',job_id=str(args.job),
        nodes=args.nodes,ranks=args.nodes*12,job_state='F',exit_status=0,files=len(files),
        archive=str(archive),archive_bytes=archive.stat().st_size,sha256=sha(archive),
        total_uncompressed_bytes=sum(x['size'] for x in manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_comparison_pending=True)
    (out/'package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes',type=int,choices=(2,8),required=True)
    p.add_argument('--job',type=int,required=True)
    main(p.parse_args())
