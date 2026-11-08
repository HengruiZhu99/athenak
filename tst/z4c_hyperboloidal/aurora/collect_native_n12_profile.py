#!/usr/bin/env python3
"""Collect compact profile proof after terminal success; raw scientific data stay remote."""
import argparse
from pathlib import Path
import subprocess
import tarfile

from native_n12_profile_audit import CASE
from native_two_node_timing_audit import ROOT,read_json,require,sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True);args=p.parse_args()
    import re
    text=subprocess.check_output(['qstat','-xf',str(args.job)],text=True)
    job=re.search(r'^Job Id:\s*(\S+)',text,re.M)
    require(job and job[1].split('.')[0]==str(args.job)==(CASE/'job-id.txt').read_text().strip().split('.')[0],
            'actual/terminal job differs')
    for key,value in [('job_state','F'),('Exit_status','0'),('Job_Name','native-n12-profile-two'),('Resource_List.nodect','2')]:
        match=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',text,re.M)
        require(match and match[1]==value,'terminal field differs: '+key)
    report=read_json(CASE/'qualification.json')
    require(report['passed'] and report['scientific_qualification_passed'],'profile qualification failed')
    # Rehash every retained raw image. No raw trace/checkpoint is copied or deleted.
    for rel,v in report['artifact_manifest'].items():
        path=CASE/rel;require(path.stat().st_size==v['bytes'] and sha(path)==v['sha256'],'raw artifact changed: '+rel)
    out=ROOT/f'validation/native-n12-profile-compact-{args.job}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(text)
    selected={str(p.relative_to(ROOT)):p for p in CASE.rglob('*') if p.is_file()
        and 'cache' not in p.relative_to(CASE).parts and
        (p.name in ('qualification.json','job-id.txt','modules.txt','nodes.txt','launchers-after.txt',
                    'guard-before.json','guard-after.json','command.txt','exit-code.txt','launch.json','runtime.json')
         or '-verify-' in p.name)}
    work=ROOT/'validation/perf-two-node-v1/profile'
    for path in work.iterdir():
        if path.is_file():selected[str(path.relative_to(ROOT))]=path
    selected['terminal-qstat.txt']=out/'terminal-qstat.txt'
    require(sum(p.stat().st_size for p in selected.values())<=20*1024*1024,'compact proof exceeds20MiB; raw retained')
    manifest={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(selected.items())}
    write_json(out/'files.json',manifest)
    archive=out/'compact.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in sorted(selected.items()):tar.add(path,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    package=dict(schema='native-n12-profile-compact-collection-v1',job_id=args.job,nodes=2,ranks=24,
        job_state='F',exit_status=0,raw_root=str(CASE),raw_files=len(report['artifact_manifest']),
        compact_files=len(selected),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        raw_traces_and_checkpoints_retained_remote=True,performance_measurement=False)
    write_json(out/'package.json',package);print(package)

if __name__=='__main__':main()
