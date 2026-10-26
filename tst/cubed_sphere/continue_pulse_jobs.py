#!/usr/bin/env python3
"""Submit sequential short capacity allocations; never overlap this task's nodes."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

HOST='hzhu@aurora.alcf.anl.gov'
CONTROL='/Users/hz0693/.ssh/codex-control/%C'


def remote(command):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ControlPath='+CONTROL,HOST,command],
                     capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root');p.add_argument('record',type=Path)
    p.add_argument('--wait-for');p.add_argument('--previous');p.add_argument('--max-stages',type=int,default=16)
    a=p.parse_args();assert a.root.startswith('/lus/flare/projects/CompactBinaryMerger/hzhu/')
    record={'root':a.root,'jobs':[],'policy':{'queue':'capacity','nodes':1,'allocation':'00:30:00','solver_wall_seconds':300,'one_allocation_at_a_time':True}}
    def save():a.record.write_text(json.dumps(record,indent=2)+'\n')
    def wait(job):
        previous=None;failures=0
        while True:
            try:text=remote('qstat -xf '+shlex.quote(job));failures=0
            except Exception as e:
                failures+=1
                if failures>=3:raise
                print('scheduler retry',str(e),flush=True);time.sleep(30);continue
            state=re.search(r'^\s*job_state = (\S+)',text,re.M)[1]
            if state!=previous:print(job,'state',state,flush=True);previous=state
            if state=='F':return text
            time.sleep(30)
    if a.wait_for:
        print('Waiting for preceding allocation',a.wait_for,flush=True);wait(a.wait_for)
    previous=a.previous
    for stage in range(a.max_stages):
        env='PULSE_MODE=evolution,PULSE_AFFINITY=tile'
        if previous:env+=',PULSE_PREVIOUS='+previous
        command='cd '+shlex.quote(a.root)+' && qsub -q capacity -l select=1 -l walltime=00:30:00 -v '+shlex.quote(env)+' run_checkpoint_pulse_aurora.pbs'
        result=remote(command);matches=re.findall(r'(?m)^(\d+\.aurora-pbs-\S+)$',result)
        assert len(matches)==1,result
        job=matches[0];row={'stage':stage+1,'job_id':job,'submission':command,'previous':previous};record['jobs'].append(row);save()
        print('Submitted',job,flush=True)
        accounting=wait(job)
        assert re.search(r'^\s*queue = capacity$',accounting,re.M)
        assert re.search(r'^\s*Resource_List.select = 1$',accounting,re.M)
        assert re.search(r'^\s*Resource_List.walltime = 00:30:00$',accounting,re.M)
        row['PBS_exit_status']=int(re.search(r'^\s*Exit_status = (\d+)',accounting,re.M)[1])
        path=a.root+'/results/'+job
        script='import json;d=json.load(open('+repr(path+'/summary.json')+'));print(json.dumps({"binary_sha256":d["binary_sha256"],"all_terminal":d["all_terminal"],"cases":{k:{"outcome":r["outcome"],"time":r["last_rows"][-1]["time"] if r["last_rows"] else None,"checkpoint":r.get("checkpoint_payload"),"reference_unchanged":r["reference_unchanged"],"terminal_messages":r["terminal_messages"]} for k,r in d["cases"].items()}}))'
        summary=json.loads(remote('/usr/bin/python3.11 -c '+shlex.quote(script)))
        row['result_path']=path;row['summary']=summary;save();print(json.dumps(summary),flush=True)
        if summary['all_terminal']:
            record['all_terminal']=True;save();return
        assert all(r['outcome'] in ['checkpointed','complete','failed','timeout'] for r in summary['cases'].values())
        assert any(r['outcome']=='checkpointed' for r in summary['cases'].values())
        previous=path
    record['stage_cap_reached']=True;save()


if __name__=='__main__':main()
