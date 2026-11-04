#!/usr/bin/env python3
"""One-shot task-only submission with an enduring receipt; never auto-retry."""
import argparse,json,pathlib,subprocess,time,getpass
p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('receipt');p.add_argument('--nodes',type=int,required=True);a=p.parse_args()
assert 1<=a.nodes<=2
receipt=pathlib.Path(a.receipt)
# Exclusive creation means an ambiguous prior submission requires reconciliation.
with receipt.open('x') as f:json.dump({'status':'preparing','script':str(pathlib.Path(a.script).resolve()),'nodes':a.nodes,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},f)
query=subprocess.run(['qstat','-u',getpass.getuser()],capture_output=True,text=True,check=True)
receipt.with_suffix('.jobs-before.txt').write_text(query.stdout)
# This task always runs allocations sequentially, even across different queues.
selected=subprocess.run(['qselect','-u',getpass.getuser(),'-N','hyp-perf-v1'],capture_output=True,text=True)
if selected.returncode not in [0,153]:raise RuntimeError('scheduler reconciliation failed')
if selected.stdout.strip():raise RuntimeError('task already has scheduler jobs; inspect before submitting')
result=subprocess.run(['qsub',a.script],capture_output=True,text=True)
receipt.write_text(json.dumps({'status':'submission_returned','returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'nodes':a.nodes,'script':str(pathlib.Path(a.script).resolve()),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
if result.returncode:raise RuntimeError('qsub failed; receipt preserved, do not resubmit without diagnosis')
print(result.stdout.strip())
