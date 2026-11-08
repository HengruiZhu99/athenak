#!/usr/bin/env python3
"""Collect terminal native controls; do not admit scientific results or timings."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import tarfile

from native_two_node_timing_audit import (ROOT,WORK,CASE,EXECUTABLES,VERIFY,admission,
    binding,completed_case,read_json,require,sha,terminal,write_json)


def collect(job):
    status=subprocess.check_output(['qstat','-xf',str(job)],text=True);terminal(status,job)
    require((CASE/'job-id.txt').read_text().strip().split('.')[0]==str(job),'wrong case job')
    require((CASE/'pipeline-complete.txt').is_file(),'pipeline did not finish')
    before,after=[read_json(CASE/f'guard-{suffix}.json') for suffix in ('before','after')]
    require(before==after and before['passed'] is True,'runtime guard differs')
    bound=admission(WORK/'admission.json')
    require(all(before[k]==v for k,v in bound.items()),'admission changed after run')
    for suffix in ('before','after'):
        for name,(work,builder,digest,text) in VERIFY.items():
            log=CASE/f'{name}-verify-{suffix}.txt'
            require(log.read_text()==text and sha(log)==before[name.replace('-','_')+'_verify_log_sha256'],
                    'verifier output differs')
    binding(CASE/'binding.txt')
    for kind,(exe,digest) in EXECUTABLES.items():
        require(sha(exe)==digest,'executable changed after run')
        require(read_json(CASE/kind/'operational.json')==completed_case(CASE/kind,kind),
                'operational case record differs')
    out=ROOT/f'validation/native-two-node-timing-artifacts-{job}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(status)
    files={str(p.relative_to(ROOT)):p for p in CASE.rglob('*') if p.is_file()
           and 'cache' not in p.relative_to(CASE).parts}
    for name in ('native_two_node_timing_audit.py','collect_native_two_node_timing.py',
                 'run-native-two-node-timing.sh','native-two-node-timing.pbs','admission.json',
                 'launchers.sha256','science-admission.json','independent-review.json'):
        p=WORK/name;require(p.is_file(),'missing prepared input: '+name);files[str(p.relative_to(ROOT))]=p
    for work,builder,digest,text in VERIFY.values():
        p=work/'build/provenance.json';files[str(p.relative_to(ROOT))]=p
    files['terminal-qstat.txt']=out/'terminal-qstat.txt'
    manifest={name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())}
    write_json(out/'files.json',manifest)
    archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    report=dict(schema='native-n12-two-node-timing-collection-v1',job_id=str(job),nodes=2,ranks=24,
        job_state='F',exit_status=0,files=len(files),**bound,archive=str(archive),
        archive_bytes=archive.stat().st_size,sha256=sha(archive),
        total_uncompressed_bytes=sum(v['size'] for v in manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_comparison_pending=True,performance_admission=False)
    write_json(out/'package.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=int,required=True)
    collect(parser.parse_args().job)
