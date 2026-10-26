#!/usr/bin/env python3
"""Collect completed pulse segments and independently recount their continuity.

Does not submit jobs or modify the solver. Existing evidence is never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from analyze_spherical import analyze_case, rows
from continue_pulse_jobs import remote, HOST, CONTROL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--parent-summary',type=Path)
    args=parser.parse_args()
    record_bytes=args.record.read_bytes();record=json.loads(record_bytes)
    destination=args.destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    record_sha=hashlib.sha256(record_bytes).hexdigest()
    snapshot=destination/('input-record-'+record_sha+'.json')
    if snapshot.exists():assert snapshot.read_bytes()==record_bytes
    else:snapshot.write_bytes(record_bytes)
    previous={};review=[];parent_sha=None
    if args.parent_summary:
        parent_bytes=args.parent_summary.read_bytes()
        previous=json.loads(parent_bytes)['cases']
        parent_sha=hashlib.sha256(parent_bytes).hexdigest()
    for job in record['jobs']:
        if 'summary' not in job:continue
        result=job['result_path'];job_id=job['job_id'];number=job_id.split('.')[0]
        target=destination/number
        if not target.exists():
            archived=record['root']+'/compact-evidence-'+number
            script='from pathlib import Path;print(Path('+repr(archived)+').exists())'
            exists=remote('/usr/bin/python3.11 -c '+shlex.quote(script)).strip()=='True'
            if not exists:
                collector='/lus/flare/projects/CompactBinaryMerger/hzhu/cubed-sphere-20261013/collect_checkpoint_evidence.py'
                remote('/usr/bin/python3.11 '+shlex.quote(collector)+' --case '+shlex.quote(result)+' --destination '+shlex.quote(archived))
            subprocess.run(['scp','-q','-r','-o','BatchMode=yes','-o','ControlPath='+CONTROL,
                            HOST+':'+archived,str(target)],check=True,timeout=180)
        scheduler=target/'scheduler.txt'
        # Finished accounting is immutable evidence. Reuse it instead of
        # re-querying and replacing every old segment on each collection.
        accounting=scheduler.read_text() if scheduler.exists() else remote('qstat -xf '+shlex.quote(job_id))
        assert re.search(r'^Job Id: '+re.escape(job_id)+r'$',accounting,re.M)
        for field,value in [('job_state','F'),('queue','capacity'),
                            ('Account_Name','CompactBinaryMerger'),
                            ('Resource_List.select','1'),('Resource_List.walltime','00:30:00'),
                            ('Exit_status',str(job['PBS_exit_status']))]:
            assert re.search(r'^\s*'+re.escape(field)+r' = '+re.escape(value)+r'$',accounting,re.M),field
        if not scheduler.exists():scheduler.write_text(accounting)
        summary=json.loads((target/'summary.json').read_text());cases={}
        assert summary['binary_sha256']==job['summary']['binary_sha256']
        for name,receipt in summary['cases'].items():
            if receipt.get('continued_from_previous_terminal_case'):
                cases[name]={'retained_terminal_case':receipt['continued_from_previous_terminal_case']}
                continue
            case=target/name;data=rows(case/'cubed_sphere_conformal.csv')
            local_receipt=json.loads((case/'receipt.json').read_text())
            assert local_receipt==receipt
            assert receipt['reference_unchanged']
            assert len({v['sha256'] for v in receipt['reference_payloads'].values()})==1
            cases[name]=analyze_case(case)
            if name in previous:
                old=previous[name]
                assert receipt['restart_payload']==old['checkpoint_payload']
                assert data[:2]==old['last_rows'],'changed diagnostics at restart boundary'
                assert float(data[-1]['time'])>float(old['last_rows'][-1]['time'])
            previous[name]=receipt
        review.append({'job_id':job_id,'cases':cases})
        print(number,{n:receipt['outcome'] for n,receipt in summary['cases'].items()},flush=True)
    output={'record_sha256':record_sha,'record_snapshot':snapshot.name,'parent_summary_sha256':parent_sha,
            'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Finite diagnostic recount, radial norm partition, reference equality and exact restart-boundary metrics. Does not establish convergence or stability.',
            'segments':review}
    (destination/'chain-review.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
