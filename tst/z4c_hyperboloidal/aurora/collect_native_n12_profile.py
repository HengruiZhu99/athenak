#!/usr/bin/env python3
"""Collect compact profile proof; retain actual batch status and any offline reanalysis."""
import argparse
from pathlib import Path
import subprocess
import tarfile

from native_n12_profile_audit import CASE,WORK
from native_two_node_timing_audit import ROOT,read_json,require,sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True)
    p.add_argument('--reaudit',type=Path)
    p.add_argument('--audit-package',type=Path)
    args=p.parse_args()
    require((args.reaudit is None)==(args.audit_package is None),'reanalysis requires its isolated audit package')
    import re
    text=subprocess.check_output(['qstat','-xf',str(args.job)],text=True)
    job=re.search(r'^Job Id:\s*(\S+)',text,re.M)
    require(job and job[1].split('.')[0]==str(args.job)==(CASE/'job-id.txt').read_text().strip().split('.')[0],
            'actual/terminal job differs')
    exit_status=2 if args.reaudit is not None else 0
    for key,value in [('job_state','F'),('Exit_status',str(exit_status)),('Job_Name','native-n12-profile-two'),('Resource_List.nodect','2')]:
        match=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',text,re.M)
        require(match and match[1]==value,'terminal field differs: '+key)
    report_path=args.reaudit or CASE/'qualification.json'
    report=read_json(report_path)
    require(report['passed'] and report['scientific_qualification_passed'],'profile qualification failed')
    if args.reaudit is not None:
        require(report_path.resolve()==(CASE/'qualification-v2.json').resolve(),'unexpected reanalysis report path')
        r=report['reanalysis'];old=CASE/'qualification.json';terminal=Path(r['terminal_record'])
        require(r['previous_report']==str(old) and r['previous_report_sha256']==sha(old) and
                r['previous_failure']==read_json(old)['failure'] and read_json(old)['passed'] is False,
                'original audit failure changed')
        require((r['job_id'],r['job_state'],r['batch_exit_status'],r['solver_exit_statuses'])==
                (str(args.job),'F',2,{'normal':0,'profiled':0}) and r['terminal_sha256']==sha(terminal),
                'reanalysis hides actual batch/solver status')
        for mode in ('normal','profiled'):
            require((CASE/mode/'exit-code.txt').read_text().strip()=='0','actual solver exit differs')
        require(args.audit_package.resolve()==ROOT/'validation/perf-two-node-v1/profile-audit-v2',
                'unexpected reanalysis package path')
        for name,digest in report['source_sha256'].items():
            require(Path(name).name==name and sha(args.audit_package/name)==digest,'reanalysis source changed')
    # Rehash every retained raw image. No raw trace/checkpoint is copied or deleted.
    for rel,v in report['artifact_manifest'].items():
        path=CASE/rel;require(path.stat().st_size==v['bytes'] and sha(path)==v['sha256'],'raw artifact changed: '+rel)
    suffix='-reaudit-v2' if args.reaudit is not None else ''
    out=ROOT/f'validation/native-n12-profile-compact-{args.job}{suffix}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(text)
    selected={str(p.relative_to(ROOT)):p for p in CASE.rglob('*') if p.is_file()
        and 'cache' not in p.relative_to(CASE).parts and
        (p.name in ('qualification.json','qualification-v2.json','job-id.txt','modules.txt','nodes.txt','launchers-after.txt',
                    'guard-before.json','guard-after.json','command.txt','exit-code.txt','launch.json','runtime.json')
         or '-verify-' in p.name or p.name=='run.log' or p.suffix=='.hst' or '.horizon' in p.name)}
    work=WORK
    for path in work.iterdir():
        if path.is_file():selected[str(path.relative_to(ROOT))]=path
    for name,v in report.get('external_profile_artifacts',{}).items():
        path=Path(name);require(path.stat().st_size==v['bytes'] and sha(path)==v['sha256'],'external profile text changed')
        selected[str(path.relative_to(ROOT))]=path
    if args.audit_package is not None:
        for path in args.audit_package.iterdir():
            if path.is_file():selected[str(path.relative_to(ROOT))]=path
        selected[str(terminal.relative_to(ROOT))]=terminal
    selected['terminal-qstat.txt']=out/'terminal-qstat.txt'
    require(sum(p.stat().st_size for p in selected.values())<=20*1024*1024,'compact proof exceeds20MiB; raw retained')
    manifest={name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in sorted(selected.items())}
    write_json(out/'files.json',manifest)
    archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in sorted(selected.items()):tar.add(path,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    package=dict(schema='native-n12-profile-compact-collection-v1',job_id=args.job,nodes=2,ranks=24,
        job_state='F',exit_status=exit_status,raw_root=str(CASE),raw_files=len(report['artifact_manifest']),
        files=len(selected),compact_files=len(selected),archive=str(archive),sha256=sha(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        qualification_report=str(report_path),qualification_sha256=sha(report_path),
        offline_reanalysis=report.get('reanalysis'),
        raw_traces_and_checkpoints_retained_remote=True,performance_measurement=False)
    write_json(out/'package.json',package);print(package)

if __name__=='__main__':main()
