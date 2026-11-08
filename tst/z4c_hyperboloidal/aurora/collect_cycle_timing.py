#!/usr/bin/env python3
"""Collect a successful future timing pair; no scientific/performance admission."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from cycle_timing_run_audit import (ROOT, BASELINE, TIMING, admission, binding,
                                    read_json, require, sha, terminal)


def collect(args):
    root = ROOT;validation = root/'validation/cycle-timing-run'
    case = root/f'runs/cycle-timing-nodes{args.nodes}'
    status = subprocess.check_output(['qstat','-xf',str(args.job)],text=True)
    terminal(status,args.job,args.nodes)
    require((case/'job-id.txt').read_text().strip().split('.')[0] == str(args.job),
            'case belongs to another PBS job')
    require((case/'pipeline-complete.txt').is_file(), 'pair did not complete guards')
    before,after = [read_json(case/f'guard-{when}.json') for when in ('before','after')]
    require(before == after and before['passed'] is True, 'before/after guards differ')
    bound = admission(validation/'admission.json')
    for key,value in bound.items():
        require(before[key] == value, 'admission changed after execution: '+key)
    for when in ('before','after'):
        require(sha(case/f'timing-verify-{when}.txt') == before['timing_verify_log_sha256'],
                'timing verification log changed')
    binding(case/'binding.txt',args.nodes)
    binding_argument = '--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
    for name,executable,digest in (
            ('normal',root/'build-sycl/src/athena',BASELINE),
            ('timed',root/'validation/cycle-timing/build/athena-cycle-timing',TIMING)):
        directory = case/name
        expected = ['mpiexec','-n',str(12*args.nodes),'-ppn','12',binding_argument,
            before['gpu_wrapper'],str(executable),'-r',bound['source_checkpoint_remote'],
            '-d',str(directory),'time/nlim=3681','time/tlim=2','time/ndiag=1000000000']
        require(shlex.split((directory/'command.txt').read_text()) == expected,
                'recorded command differs from bounded qualification')
        require(sha(executable) == digest and (directory/'exit-code.txt').read_text().strip() == '0',
                'failed or changed executable')
        require(sorted(p.name for p in (directory/'rst').glob('*.rst')) == ['native_n12_5M.00011.rst'],
                'missing or extra restart output')
        require((directory/'environment.json').is_file() and (directory/'run.log').is_file(),
                'missing execution provenance')
    timing = read_json(case/'timed/native-cycle-timing.json')
    require(timing['schema'] == 'native-cycle-timing-v1' and timing['passed'] is True and
            timing['initial_cycle'] == 3551 and timing['final_cycle'] == 3681 and
            timing['complete_steps'] == 130 and timing['ranks'] == 12*args.nodes,
            'incomplete timing run')
    out = root/f'validation/cycle-timing-artifacts-{args.job}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(status)
    files = {str(p.relative_to(root)):p for p in case.rglob('*') if p.is_file()
             and 'cache' not in p.relative_to(case).parts}
    # Never collect an arbitrary private checkpoint/archive placed in validation.
    for name in ('cycle_timing_run_audit.py','collect_cycle_timing.py',
                 'run-cycle-timing-validation.sh','cycle-timing-two.pbs',
                 'cycle-timing-eight.pbs','admission.json','launchers.sha256'):
        p = validation/name
        require(p.is_file(), 'missing prepared timing tool: '+name)
        files[str(p.relative_to(root))] = p
    controls = validation/'cycle_timing_run_controls.py'
    if controls.is_file():
        files[str(controls.relative_to(root))] = controls
    descriptor = read_json(validation/'admission.json')
    for key in ('science_admission','independent_review'):
        p = Path(descriptor[key]['path'])
        p = p if p.is_absolute() else validation/p
        files[str(p.resolve().relative_to(root))] = p
    proof = root/'validation/cycle-timing/build/provenance.json'
    files[str(proof.relative_to(root))] = proof
    files['terminal-qstat.txt'] = out/'terminal-qstat.txt'
    manifest = {name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())}
    (out/'files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive = out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name,p in sorted(files.items()):
            tar.add(p,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    report = dict(schema='native-cycle-timing-collection-v1',job_id=str(args.job),
        nodes=args.nodes,ranks=12*args.nodes,job_state='F',exit_status=0,files=len(files),
        source_checkpoint_remote=bound['source_checkpoint_remote'],
        source_checkpoint_sha256=bound['source_checkpoint_sha256'],
        science_admission_sha256=bound['science_admission_sha256'],
        independent_review_sha256=bound['independent_review_sha256'],
        archive=str(archive),archive_bytes=archive.stat().st_size,sha256=sha(archive),
        total_uncompressed_bytes=sum(x['size'] for x in manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_comparison_pending=True,performance_admission=False)
    (out/'package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes',type=int,choices=(2,8),required=True)
    parser.add_argument('--job',type=int,required=True)
    collect(parser.parse_args())
