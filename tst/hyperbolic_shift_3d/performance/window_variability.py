#!/usr/bin/env python3
"""Audit the preselected 4-step warmup/28-step scaling window, without tuning it."""
import argparse,json,pathlib,statistics,re
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.directory);rows=[]
for receipt in json.loads((root/'results.json').read_text()):
 name=receipt['case']['name']
 if not name.startswith('scaling-'):continue
 rank=[]
 for f in root.glob(name+'.rank*.samples.jsonl'):
  samples=[json.loads(x) for x in f.read_text().splitlines()];s=[x['seconds'] for x in samples if x['phase']=='step'];d=[x['seconds'] for x in samples if x['phase']=='diagnostics'];v=[x+y for x,y in zip(s[4:],d[5:])];assert len(v)==28
  rank.append({'rank':int(re.search(r'\.rank(\d+)\.',f.name).group(1)),'mean':statistics.mean(v),'coefficient_of_variation':statistics.stdev(v)/statistics.mean(v),'range':[min(v),max(v)],'second_half_over_first_half':statistics.mean(v[14:])/statistics.mean(v[:14])})
 critical=max(rank,key=lambda x:x['mean']);rows.append({'name':name,'critical_rank':critical,'all_ranks':sorted(rank,key=lambda x:x['rank'])})
pathlib.Path(a.output).write_text(json.dumps(rows,indent=2)+'\n')
