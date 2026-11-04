#!/usr/bin/env python3
"""Check same-rank matched histories exactly, independently of final checkpoints."""
import argparse,json,pathlib,hashlib
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.directory);seen={};checks=[]
for receipt in json.loads((root/'results.json').read_text()):
 c=receipt['case'];key=c.get('equivalence')
 if not key or receipt['returncode']!=0 or c.get('restart','-')!='-':continue
 key=(key,c['ranks'],c.get('ppn',c['ranks']))
 path=root/(c['name']+'.history.jsonl');raw=path.read_bytes();prior=seen.setdefault(key,(c['name'],raw));equal=raw==prior[1]
 checks.append(dict(name=c['name'],reference=prior[0],bitwise_history_equal=equal,sha256=hashlib.sha256(raw).hexdigest()))
pathlib.Path(a.output).write_text(json.dumps(checks,indent=2)+'\n');assert all(x['bitwise_history_equal'] for x in checks)
print(len(checks),'same-rank histories checked exactly')
