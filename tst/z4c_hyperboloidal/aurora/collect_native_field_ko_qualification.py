#!/usr/bin/env python3
"""Collect the five existing KO qualification cases; independent science stays pending."""
import argparse
import datetime
import re
import subprocess
import tarfile
from pathlib import Path

from native_field_ko_qualification import (ROOT,WORK,LAUNCH,CASE,CAPS,manifest,
    read_json,write_json,require,sha)


def collect(job):
    status=subprocess.check_output(['qstat','-xf',str(job)],text=True)
    match=re.search(r'^Job Id:\s*(\S+)',status,re.M)
    fields=dict(re.findall(r'^\s+([A-Za-z_.]+)\s*=\s*(\S+)\s*$',status,re.M))
    require(match and match[1].split('.')[0]==str(job) and all(fields.get(k)==v for k,v in
        [('job_state','F'),('Exit_status','0'),('Job_Name','native-ko-qual'),
         ('Resource_List.nodect','2')]),'not the expected successful terminal job')
    require((CASE/'job-id.txt').read_text().strip().split('.')[0]==str(job)
            and (CASE/'pipeline-complete.txt').is_file(),'wrong/incomplete case')
    m=manifest()
    for stem in ('guard','qualification-guard'):
        require(read_json(CASE/f'{stem}-before.json')==read_json(CASE/f'{stem}-after.json'),
                'before/after provenance guard differs')
    summary=read_json(CASE/'qualification-operational.json')
    require(summary['passed'] and set(summary['cases'])==set(CAPS),
            'five qualification cases incomplete')
    for name in CAPS:
        require((CASE/name/'exit-code.txt').read_text().strip()=='0' and
                read_json(CASE/name/'qualification-case.json')==summary['cases'][name] and
                summary['cases'][name]['passed'],'qualification case proof changed')
    out=ROOT/f'validation/native-field-ko-qualification-artifacts-{job}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(status)
    files={str(p.relative_to(ROOT)):p for p in CASE.rglob('*') if p.is_file()
           and 'cache' not in p.relative_to(CASE).parts}
    for folder in (LAUNCH,WORK/'qualification-package'):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(ROOT))]=p
    for folder in (WORK/'build',WORK/'qualification-build'):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.log','.mk','.cpp','.d'):
                files[str(p.relative_to(ROOT))]=p
    for s in m['cases']:
        p=Path(s['provenance']);require(sha(p)==s['provenance_sha256'],'candidate proof changed')
        files[str(p.relative_to(ROOT))]=p
    files['terminal-qstat.txt']=out/'terminal-qstat.txt'
    file_manifest={name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())}
    write_json(out/'files.json',file_manifest);archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    report=dict(schema='native-field-ko-qualification-collection-v1',job_id=str(job),nodes=2,ranks=24,
        job_state='F',exit_status=0,cases=list(CAPS),files=len(files),
        command_manifest_sha256=sha(LAUNCH/'qualification-commands.json'),
        archive=str(archive),archive_bytes=archive.stat().st_size,sha256=sha(archive),
        total_uncompressed_bytes=sum(v['size'] for v in file_manifest.values()),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        operational_checks_passed=True,independent_scientific_review_pending=True,
        performance_admission=False)
    write_json(out/'package.json',report);print(__import__('json').dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True)
    collect(p.parse_args().job)
