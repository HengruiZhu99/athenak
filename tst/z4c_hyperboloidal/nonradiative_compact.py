#!/usr/bin/env python3
"""Package completed summaries, small plots/cuts and run receipts; omit bulk data."""
import argparse,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
p.add_argument('--analysis',action='append',default=[]);p.add_argument('--runs',action='append',default=[])
a=p.parse_args();paths=set()
def add(base,names):
    for name in names:
        paths.update(p for p in base.glob(name) if p.is_file())
for name in a.analysis:
    base=a.root/'analysis'/name
    add(base,['*.json','*.txt','*.pbs','run.log'])
    groups=[base]+([base/'analytic-followup'] if (base/'analytic-followup').exists() else [])
    for group in groups:
        add(group,['*.json'])
        for case in group.iterdir():
            if not case.is_dir() or case.name=='source':continue
            add(case,['summary.json','series.json','controls.json','pointwise-summary.json',
                      'halfdt-pointwise-comparison.json','cuts.json','cuts.npz','nodes.npz'])
            for news in case.glob('news*'):
                if news.is_dir():add(news,['*.json','news.npz'])
for name in a.runs:
    base=a.root/'runs'/name
    add(base,['*.json','*.txt','*.py','*.pbs','*.log','*/input.athinput','*/run.log',
              '*/seed/run.log','*/*.hst','initial-constraints/*/*.hst','initial-constraints/*/run.log'])
with tarfile.open(a.output,'w:gz') as archive:
    for path in sorted(paths):archive.add(path,arcname=str(path.relative_to(a.root)),recursive=False)
print(len(paths),'files;',a.output.stat().st_size,'bytes')
