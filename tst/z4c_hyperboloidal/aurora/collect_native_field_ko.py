#!/usr/bin/env python3
"""Collect the completed, state-qualified two-node field-KO screen."""
import argparse
import datetime
import re
import subprocess
import tarfile
from pathlib import Path

from native_field_ko_audit import (ROOT,WORK,TIMING_WORK,LAUNCH,CASE,ORDER,binding,completed,
    read_json,write_json,require,sha)


def collect(job):
    status=subprocess.check_output(['qstat','-xf',str(job)],text=True)
    match=re.search(r'^Job Id:\s*(\S+)',status,re.M)
    fields=dict(re.findall(r'^\s+([A-Za-z_.]+)\s*=\s*(\S+)\s*$',status,re.M))
    require(match and match[1].split('.')[0]==str(job) and all(fields.get(k)==v for k,v in
        [('job_state','F'),('Exit_status','0'),('Job_Name','native-field-ko'),
         ('Resource_List.nodect','2')]),'not the expected successful terminal job')
    require((CASE/'job-id.txt').read_text().strip().split('.')[0]==str(job)
            and (CASE/'pipeline-complete.txt').is_file(),'wrong/incomplete case')
    binding()
    for stem in ('guard','candidate-guard'):
        require(read_json(CASE/f'{stem}-before.json')==read_json(CASE/f'{stem}-after.json'),
                'before/after provenance guard differs')
    screen=read_json(CASE/'first-pair-screen.json')
    names=ORDER[:2] if screen['clearly_slower'] else ORDER
    require((CASE/'early-rejected.txt').is_file()==screen['clearly_slower'],'screen branch differs')
    for name in names:
        require(read_json(CASE/name/'operational.json')==completed(name),'case proof changed')
    for name in set(ORDER)-set(names):require(not (CASE/name).exists(),'unexpected case after rejection')
    out=ROOT/f'validation/native-field-ko-artifacts-{job}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(status)
    files={str(p.relative_to(ROOT)):p for p in CASE.rglob('*') if p.is_file()
           and 'cache' not in p.relative_to(CASE).parts}
    for folder in (LAUNCH,WORK/'package'):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(ROOT))]=p
    for folder in (TIMING_WORK/'timing-build',WORK/'build'):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.log','.mk','.cpp','.d'):
                files[str(p.relative_to(ROOT))]=p
    files['terminal-qstat.txt']=out/'terminal-qstat.txt'
    manifest={name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())}
    write_json(out/'files.json',manifest);archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    report=dict(schema='native-field-ko-collection-v1',job_id=str(job),nodes=2,ranks=24,
        job_state='F',exit_status=0,cases=list(names),files=len(files),
        archive=str(archive),archive_bytes=archive.stat().st_size,sha256=sha(archive),
        total_uncompressed_bytes=sum(v['size'] for v in manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state_and_diagnostic_checks_passed=True,independent_timing_decision_pending=True)
    write_json(out/'package.json',report);print(__import__('json').dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True)
    collect(p.parse_args().job)
