#!/usr/bin/env python3
"""Replay the fixed joint-plane diagnostic in a fresh directory, bounded385s."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
root=next(x for x in Path(__file__).resolve().parents if (x/'src/z4c').is_dir())
cmd=[sys.executable,str(Path(__file__).with_name('v2')/'analyze.py'),str(a.output/'result.json')]
start=time.monotonic()
try:
 r=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=385)
 record=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.monotonic()-start)
except subprocess.TimeoutExpired:
 record=dict(command=cmd,status='bounded',seconds=time.monotonic()-start)
(a.output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['stdout','stderr']}))
raise SystemExit(record.get('returncode',1))
