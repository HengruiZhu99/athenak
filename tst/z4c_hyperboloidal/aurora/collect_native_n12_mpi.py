#!/usr/bin/env python3
"""Retain complete APS data and compact qualification after the actual terminal job."""
import argparse
from pathlib import Path
import re
import subprocess
import tarfile

from native_n12_mpi_audit import CASE
from native_two_node_timing_audit import ROOT,read_json,require,sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True)
    p.add_argument('--reaudit',type=Path);p.add_argument('--audit-package',type=Path);args=p.parse_args()
    require((args.reaudit is None)==(args.audit_package is None),'reanalysis needs its isolated source package')
    terminal=subprocess.check_output(['qstat','-xf',str(args.job)],text=True)
    job=re.search(r'^Job Id:\s*(\S+)',terminal,re.M)
    require(job and job[1].split('.')[0]==str(args.job)==(CASE/'job-id.txt').read_text().strip().split('.')[0],
            'actual terminal job differs')
    exit_status=2 if args.reaudit is not None else 0
    for key,value in [('job_state','F'),('Exit_status',str(exit_status)),('Job_Name','native-n12-mpi-two'),('Resource_List.nodect','2')]:
        match=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',terminal,re.M)
        require(match and match[1]==value,'terminal field differs: '+key)
    report_path=args.reaudit or CASE/'qualification.json';report=read_json(report_path)
    require(report['passed'] and report['scientific_qualification_passed'] and report['mpi_interception_verified'],
            'MPI qualification failed')
    if args.reaudit is not None:
        require(report_path.resolve()==CASE/'qualification-v2.json','unexpected reanalysis path')
        old=CASE/'qualification.json';r=report['reanalysis'];saved_terminal=Path(r['terminal_record'])
        require(r['previous_report']==str(old) and r['previous_report_sha256']==sha(old) and
            read_json(old)['passed'] is False and r['previous_failure']==read_json(old)['failure'],
            'original failure changed')
        require((r['job_id'],r['job_state'],r['batch_exit_status'],r['solver_exit_statuses'])==
            (str(args.job),'F',2,{'normal':0,'profiled':0}) and sha(saved_terminal)==r['terminal_sha256'],
            'reanalysis hides actual batch/solver status')
        for mode in ('normal','profiled'):
            require((CASE/mode/'exit-code.txt').read_text().strip()=='0','actual solver exit differs')
        require(args.audit_package.resolve()==ROOT/'validation/perf-two-node-v1/mpi-profile-audit-v3',
            'unexpected reanalysis package path')
        for name,digest in report['source_sha256'].items():
            require(Path(name).name==name and sha(args.audit_package/name)==digest,'reanalysis source changed')
    for rel,record in report['artifact_manifest'].items():
        path=CASE/rel
        require(path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],'raw artifact changed: '+rel)
    suffix='-reaudit-v2' if args.reaudit is not None else ''
    out=ROOT/f'validation/native-n12-mpi-compact-{args.job}{suffix}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(terminal)
    selected={str(path.relative_to(ROOT)):path for path in CASE.rglob('*') if path.is_file()
        and 'cache' not in path.relative_to(CASE).parts and path.suffix!='.rst'}
    work=ROOT/'validation/perf-two-node-v1/mpi-profile-v2'
    for path in work.iterdir():
        if path.is_file():selected[str(path.relative_to(ROOT))]=path
    if args.audit_package is not None:
        for path in args.audit_package.iterdir():
            if path.is_file():selected[str(path.relative_to(ROOT))]=path
        selected[str(saved_terminal.relative_to(ROOT))]=saved_terminal
    selected['terminal-qstat.txt']=out/'terminal-qstat.txt'
    require(sum(path.stat().st_size for path in selected.values())<=50*1024*1024,
            'compact MPI data exceed50MiB; raw retained for offline review')
    manifest={name:dict(size=path.stat().st_size,sha256=sha(path)) for name,path in sorted(selected.items())}
    write_json(out/'files.json',manifest)
    archive=out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in sorted(selected.items()):tar.add(path,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    package=dict(schema='native-n12-mpi-compact-collection-v1',job_id=args.job,nodes=2,ranks=24,
        job_state='F',exit_status=exit_status,raw_root=str(CASE),raw_files=len(report['artifact_manifest']),
        files=len(selected),archive=str(archive),sha256=sha(archive),archive_bytes=archive.stat().st_size,
        qualification_report=str(report_path),qualification_sha256=sha(report_path),
        offline_reanalysis=report.get('reanalysis'),raw_checkpoints_retained_remote=True,
        performance_measurement=False)
    write_json(out/'package.json',package);print(package)


if __name__=='__main__':main()
