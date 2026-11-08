#!/usr/bin/env python3
"""Collect a terminal common-time segment; never grant scientific continuation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
CYCLES = [0,356,711,1066,1421,1776,2131,2486,2841,3196,3551]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    status=subprocess.run(['qstat','-xf',str(args.job)],capture_output=True,text=True,check=True).stdout
    assert 'job_state = F' in status and 'Exit_status = 0' in status
    case=ROOT/f'runs/native-n12-common-{args.segment:02d}'
    assert (case/'exit-code.txt').read_text().strip() == '0'
    structural=json.loads((case/'structural-cadence.json').read_text())
    assert structural['schema']=='native-n12-common-segment-v1' and structural['structural_cadence_passed']
    assert (structural['segment'],structural['start_cycle'],structural['end_cycle']) == (
        args.segment,CYCLES[args.segment-1],CYCLES[args.segment])
    assert structural['continuation_admitted'] is False
    stage=json.loads((case/'native-step-observer.json').read_text())
    assert stage['passed'] and stage['backend']=='SYCL' and stage['ranks']==24
    assert (stage['initial_cycle'],stage['final_cycle'],stage['stage_rows']) == (
        CYCLES[args.segment-1],CYCLES[args.segment],4*(CYCLES[args.segment]-CYCLES[args.segment-1]))
    out=ROOT/f'validation/common-artifacts-{args.job}'
    out.mkdir(exist_ok=False);(out/'job-final.txt').write_text(status)
    files=[p for p in case.rglob('*') if p.is_file() and 'cache' not in p.relative_to(case).parts]
    files += [p for p in (ROOT/'validation/full-n12').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    predecessor_proof = None
    if args.segment > 1:
        proof_directory = ROOT/f'validation/common-next-{args.segment:02d}'
        acceptance = json.loads((proof_directory/'acceptance.json').read_text())
        assert acceptance['segment'] == args.segment
        predecessor = ROOT/f'runs/native-n12-common-{args.segment-1:02d}'
        science = predecessor/'continuation-admission.json'
        review = Path(acceptance['independent_review']['path'])
        assert sha(science) == acceptance['science_admission_sha256']
        assert sha(review) == acceptance['independent_review']['sha256']
        checkpoint = predecessor/'rst'/f'native_n12_5M.{args.segment-1:05d}.rst'
        assert sha(checkpoint) == acceptance['checkpoint_sha256']
        files += [p for p in proof_directory.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
        files += [science, review]
        predecessor_proof = dict(science_sha256=sha(science),review_sha256=sha(review),
            checkpoint_sha256=acceptance['checkpoint_sha256'],
            proof_directory=str(proof_directory))
    files += [ROOT/p for p in ['build-sycl/runtime-object-sha256.json','build-sycl/source-sha256.json',
        'source/source-revision.json','validation/step-observer/build/provenance.json',
        'validation/step-observer/native_step_observer.py','validation/step-observer/native_step_observer.hpp',
        'validation/step-observer/native_step_observer_contract.md']]
    files.extend([out/'job-final.txt',Path(__file__).resolve()]);files=sorted(set(files))
    manifest={str(p.relative_to(ROOT)):dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
    (out/'files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    report=dict(schema='native-n12-common-segment-collection-v1',job_id=str(args.job),segment=args.segment,
        nodes=2,ranks=24,job_state='F',exit_status=0,start_cycle=CYCLES[args.segment-1],end_cycle=CYCLES[args.segment],
        archive=str(archive),archive_bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),
        total_uncompressed_bytes=sum(x['size'] for x in manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        predecessor_proof=predecessor_proof,
        endpoint_scientific_comparison_pending=True,continuation_admitted=False)
    (out/'package.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=int,required=True)
    parser.add_argument('--segment',type=int,choices=range(1,11),required=True)
    main(parser.parse_args())
