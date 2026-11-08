#!/usr/bin/env python3
"""Admit only the prepared356-to711 restart from exact completed science proofs.

This validates artifacts and the frozen runtime. It does not submit a job,
evolve, change a checkpoint, or create scientific acceptance.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

FIRST_JOB = '9051541'
SHORT_ADMISSION = 'deb26a9eca22bed36370725662531ef54ac3868d6cca4344bbda7d2fdc7db66c'
FROZEN_HELPERS = {
    'run-common-segment.sh':'7c13e2467cc11a7dde98b95026628cdf53a986b9a690684834b0a6abfaa1011f',
    'common_segment_audit.py':'e89427a8a98dc93b504924b57a55bf444d589de41538fe674fe0da7b6f43e94f',
    'longrun_admission.py':'2887b479590c51d473564f9e62d5c6c042ac0b92b520126d6acb421195dddf7f'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def check_science(science, review, science_sha, checkpoint_sha):
    require(science['schema']=='native-n12-common-science-admission-v1' and
            science['passed'] is True and science['segment']==1 and science['end_cycle']==356 and
            science['through_5m_complete'] is False and
            science['checkpoint_sha256']==checkpoint_sha,
            'missing or mismatched completed first-endpoint scientific admission')
    require(set(science['comparisons'])=={'0','356'} and
            all(x['passed'] is True and x['writer_mpi_ranks']==24
                for x in science['comparisons'].values()) and
            science['history_completed_prefix_passed'] is True and
            science['structural']['structural_cadence_passed'] is True,
            'incomplete first-endpoint fields/history/structural coverage')
    metric=science['stage_metrics']
    require(metric['schema']=='native-step-metrics-v1' and metric['passed'] is True and
            metric['through_step']==356 and metric['stage_rows']==1424 and
            metric['ranks']==24 and metric['gate']=='long' and metric['failed_values']==0,
            'missing complete first-endpoint stage metrics')
    require(review['schema']=='native-n12-common-independent-review-v1' and
            review['passed'] is True and review['end_cycle']==356 and
            review['science_admission_sha256']==science_sha and
            review['checkpoint_sha256']==checkpoint_sha,
            'independent review is missing or bound to another science report/checkpoint')


def main(args):
    root=args.root.resolve();scripts=root/'validation/full-n12'
    require(not args.output.exists(), 'refuse to overwrite preflight evidence')
    acceptance=json.loads(args.acceptance.read_text())
    require(acceptance['schema']=='native-n12-continuation-input-v1' and
            (acceptance['segment'],acceptance['start_cycle'],acceptance['end_cycle'])==(2,356,711),
            'this wrapper only admits356-to711')
    require(sha(scripts/'admission.json')==SHORT_ADMISSION, 'initial short admission changed')
    for name,digest in FROZEN_HELPERS.items():
        require(sha(scripts/name)==digest, 'frozen first launcher/helper changed: '+name)
    first=root/'runs/native-n12-common-01'
    checkpoint=first/'rst/native_n12_5M.00001.rst'
    science_path=first/'continuation-admission.json'
    review_path=Path(acceptance['independent_review']['path'])
    require(sha(science_path)==acceptance['science_admission_sha256'] and
            sha(review_path)==acceptance['independent_review']['sha256'], 'science/review proof changed')
    checkpoint_sha=sha(checkpoint)
    require(checkpoint_sha==acceptance['checkpoint_sha256'], 'accepted predecessor checkpoint changed')
    science=json.loads(science_path.read_text());review=json.loads(review_path.read_text())
    check_science(science,review,sha(science_path),checkpoint_sha)
    status=subprocess.run(['qstat','-xf',FIRST_JOB],capture_output=True,text=True,check=True).stdout
    require('job_state = F' in status and 'Exit_status = 0' in status,
            'first job did not complete normally')
    spec=importlib.util.spec_from_file_location('frozen_common_audit',scripts/'common_segment_audit.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    reference=root/'runs/n12-native-gpu-eight-two/rst/native_n12_5M.00000.rst'
    header=audit.validate(audit.checkpoint(checkpoint),audit.checkpoint(reference),1,356)
    audit.verify_runtime(root)
    observer=root/'validation/step-observer'
    verify=subprocess.run(['/usr/bin/python3.11',str(observer/'native_step_observer.py'),
        'verify','--output',str(observer/'build')],capture_output=True,text=True,check=True)
    report=dict(schema='native-n12-continuation-preflight-v1',passed=True,
        segment=2,start_cycle=356,end_cycle=711,predecessor_job=FIRST_JOB,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=header,science_admission_sha256=sha(science_path),
        independent_review_sha256=sha(review_path),short_admission_sha256=SHORT_ADMISSION,
        acceptance_sha256=sha(args.acceptance),frozen_helpers=FROZEN_HELPERS,
        observer_verify_stdout=verify.stdout,observer_verify_stderr=verify.stderr,
        restart_entry='unchanged run-common-segment.sh2; ordinary-r, no-i, no-t',
        scope='Only356-to711; no later continuation or full3551 success claim')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
