#!/usr/bin/env python3
"""Release the already-queued, never-started first job after diagnostic revision."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import advance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',action='store_true')
    args=parser.parse_args()
    root=advance.STATE_ROOT
    with (root/'advance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'state.json').read_text())
        revision=state['diagnostic_revision']
        if revision['status']=='released':
            print('Revision already released; no action.');return
        jobs=advance.scheduler(revision['job_id'])
        job_id,job=next(iter(jobs.items()))
        assert job['job_state']=='H' and not job.get('stime'), 'Job must be held and never started.'
        assert job['Account_Name']=='CompactBinaryMerger'
        assert job['queue'] in ('debug','capacity')
        assert job['Variable_List']['CASE']=='g5_tel_b32'
        build=advance.EXE.parent.parent
        marker=build/'executable.sha256'
        if not marker.exists():
            print('Build marker absent; keep the queued job held.');return
        binary_hash=hashlib.sha256(advance.EXE.read_bytes()).hexdigest()
        assert binary_hash==marker.read_text().split()[0]
        source=(build/'source_commit.txt').read_text().strip()
        assert source.startswith(revision['source_commit'])
        spec=json.loads((advance.WORKFLOW/'cases.json').read_text())
        case=next(c for c in spec['cases'] if c['name']=='g5_tel_b32')
        input_hash=hashlib.sha256((advance.WORKFLOW/'inputs/g5_tel_b32.athinput').read_bytes()).hexdigest()
        assert input_hash==case['input_sha256']
        evidence=dict(job_id=job_id,source_commit=source,executable_sha256=binary_hash,
                      input_sha256=input_hash,time=advance.now())
        print(json.dumps(evidence,indent=2))
        if not args.release: return
        attempt=next(a for a in state['attempts'] if a['job_id']==job_id)
        attempt.setdefault('original_executable_sha256',attempt['executable_sha256'])
        attempt['executable_sha256']=binary_hash
        attempt['input_sha256']=input_hash
        attempt['diagnostic_revision']=evidence
        revision.update(status='release_intent',validation=evidence)
        advance.save(state)
        result=advance.command(['qrls',job_id])
        revision['release_receipt']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode==0: revision['status']='released'
        else: revision['status']='release_needs_reconciliation'
        advance.save(state)
        print(json.dumps(revision['release_receipt']))

if __name__=='__main__': main()
