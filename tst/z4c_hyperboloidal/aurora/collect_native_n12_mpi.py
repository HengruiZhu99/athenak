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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True);args=p.parse_args()
    terminal=subprocess.check_output(['qstat','-xf',str(args.job)],text=True)
    job=re.search(r'^Job Id:\s*(\S+)',terminal,re.M)
    require(job and job[1].split('.')[0]==str(args.job)==(CASE/'job-id.txt').read_text().strip().split('.')[0],
            'actual terminal job differs')
    for key,value in [('job_state','F'),('Exit_status','0'),('Job_Name','native-n12-mpi-two'),('Resource_List.nodect','2')]:
        match=re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',terminal,re.M)
        require(match and match[1]==value,'terminal field differs: '+key)
    report=read_json(CASE/'qualification.json')
    require(report['passed'] and report['scientific_qualification_passed'],'MPI qualification failed')
    for rel,record in report['artifact_manifest'].items():
        path=CASE/rel
        require(path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],'raw artifact changed: '+rel)
    out=ROOT/f'validation/native-n12-mpi-compact-{args.job}';out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(terminal)
    selected={str(path.relative_to(ROOT)):path for path in CASE.rglob('*') if path.is_file()
        and 'cache' not in path.relative_to(CASE).parts and path.suffix!='.rst'}
    work=ROOT/'validation/perf-two-node-v1/mpi-profile'
    for path in work.iterdir():
        if path.is_file():selected[str(path.relative_to(ROOT))]=path
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
        job_state='F',exit_status=0,raw_root=str(CASE),raw_files=len(report['artifact_manifest']),
        files=len(selected),archive=str(archive),sha256=sha(archive),archive_bytes=archive.stat().st_size,
        qualification_sha256=sha(CASE/'qualification.json'),raw_checkpoints_retained_remote=True,
        performance_measurement=False)
    write_json(out/'package.json',package);print(package)


if __name__=='__main__':main()
