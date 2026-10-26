#!/usr/bin/env python3
"""Inject a disconnected SSH read into the real read-only collection watcher."""
import argparse,contextlib,hashlib,io,json,runpy,subprocess,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing recovery instrument is immutable')
 source=Path(__file__).with_name('wait_collect_aurora.py');checks=[]
 with tempfile.TemporaryDirectory(prefix='scalar-watcher-instrument-') as temporary:
  root=Path(temporary);journal=root/'scheduler.jsonl';output=root/'collection';job='existing.scalar.job';state={'Jobs':{job:{'job_state':'E','Exit_status':0}}};journal.write_text(json.dumps(state)+'\n');calls=[];sleeps=[];stdout=io.StringIO()
  def invoke(command,**kwargs):
   calls.append(command)
   if command[0]=='ssh':
    if sum(c[0]=='ssh' for c in calls)==1:return subprocess.CompletedProcess(command,255,'','Permission denied (keyboard-interactive,hostbased).')
    return subprocess.CompletedProcess(command,0,json.dumps({'Jobs':{job:{'job_state':'F','Exit_status':0}}}),'')
   if Path(command[1]).name!='collect_aurora.py':raise AssertionError('Unexpected mutation command')
   return subprocess.CompletedProcess(command,0)
  arguments=[str(source),'--root','/isolated/existing','--job',job,'--output',str(output),'--journal',str(journal),'--resume-watcher']
  with patch.object(sys,'argv',arguments),patch('subprocess.run',side_effect=invoke),patch('time.sleep',side_effect=sleeps.append),contextlib.redirect_stdout(stdout):runpy.run_path(str(source),run_name='__main__')
  records=[json.loads(line) for line in journal.read_text().splitlines()];access=[json.loads(line) for line in journal.with_suffix('.access.jsonl').read_text().splitlines()]
  checks.append(dict(name='SSH outage followed by completed existing allocation',passed=len(calls)==3 and calls[-1][0]==sys.executable and len(access)==1 and sleeps==[45] and records[0]==state and records[-1]['Jobs'][job]['job_state']=='F' and not output.exists()))
  for name,changes,create in [('existing journal requires explicit resume',{'--resume-watcher':None},False),('another allocation journal refused',{'--job':'another.scalar.job'},False),('partial collection never overwritten',{},True)]:
   if create:output.mkdir()
   args=list(arguments)
   for key,value in changes.items():
    index=args.index(key)
    if value is None:args.pop(index)
    else:args[index+1]=value
   refused=False
   with patch.object(sys,'argv',args),patch('subprocess.run',side_effect=AssertionError('Guard must precede SSH')):
    try:runpy.run_path(str(source),run_name='__main__')
    except RuntimeError:refused=True
   checks.append(dict(name=name,passed=refused))
 result=dict(passed=all(c['passed'] for c in checks),time=time.time(),checks=checks,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_trajectories_added=0,jobs_submitted=0,scope='Failure injection into the actual scheduler watcher, with every subprocess mocked. Existing journal retained, one SSH failure retried, existing finished allocation collected exactly once; wrong-job and partial-output guards refuse before SSH. Scientific source and native evidence unchanged.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
