#!/usr/bin/env python3
"""Collect one successful AMR candidate allocation without a science claim."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    status=subprocess.check_output(['qstat','-xf',str(args.job)],text=True)
    assert 'job_state = F' in status and 'Exit_status = 0' in status
    case=ROOT/f'runs/amr-candidate-nodes{args.nodes}'
    assert (case/'pipeline-complete.txt').is_file()
    for name in (['lifecycle','evolution','prefix'] if args.nodes==2 else
                 ['lifecycle','evolution','continuous','resumed']):
        proof=json.loads((case/name/'operational.json').read_text())
        assert proof['operational_passed'] and proof['scientific_comparison']=='pending'
        assert proof['ranks']==12*args.nodes and proof['case']==name and proof['fixed_dt']==.0002
        assert proof['job_id'].split('.')[0]==str(args.job)
        assert (case/name/'exit-code.txt').read_text().strip()=='0'
        assert sha(case/name/'command.txt')==proof['command_sha256']
    out=ROOT/f'validation/amr-candidate-artifacts-{args.job}'
    out.mkdir(exist_ok=False)
    (out/'job-final.txt').write_text(status)
    validation=ROOT/'validation/amr-admission-candidate-v1'
    files=[p for p in case.rglob('*') if p.is_file() and 'cache' not in p.relative_to(case).parts]
    # The physically separate CMake tree and object files stay on Aurora. Bind
    # the exact recipe/provenance and reviewed package, not another huge copy.
    files += [validation/p for p in ['coherent/provenance.json',
        'coherent/build/compile_commands.json','coherent/build/src/CMakeFiles/athena.dir/link.txt',
        'lifecycle-build/provenance.json','evolution-build/provenance.json',
        'run-amr-candidate-validation.sh','amr_run_audit.py','amr-candidate-two.pbs',
        'amr-candidate-eight.pbs','launchers.sha256','staging-origin.json']]
    files += [p for p in (validation/'package').rglob('*') if p.is_file()
              and '__pycache__' not in p.parts]
    files += [ROOT/p for p in ['build-sycl/runtime-object-sha256.json',
        'build-sycl/source-sha256.json','source/source-revision.json']]
    files += [out/'job-final.txt',Path(__file__).resolve()]
    files=sorted(set(files))
    manifest={str(p.relative_to(ROOT)):dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
    (out/'files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for path in files:
            tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    report=dict(schema='native-amr-candidate-collection-v1',job_id=str(args.job),
        nodes=args.nodes,ranks=12*args.nodes,job_state='F',exit_status=0,
        files=len(files),archive=str(archive),archive_bytes=archive.stat().st_size,
        sha256=sha(archive),total_uncompressed_bytes=sum(v['size'] for v in manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_comparison_pending=True)
    (out/'package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes',type=int,choices=(2,8),required=True)
    parser.add_argument('--job',type=int,required=True)
    main(parser.parse_args())
