#!/usr/bin/env python3
"""Collect receipts and compact products from completed wider-pulse jobs."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from nonradiative_wide import require_completed_run
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('run');p.add_argument('output',type=Path);a=p.parse_args()
base=a.root/'runs'/a.run
require_completed_run(base)
paths=set()
for pattern in ['*.json','*.txt','*.pbs','*.log','*/input.athinput','*/run.log','*/seed/run.log','*/*.hst',
                'analysis/*/*.json','analysis/*/news/*.json','analysis/*/news/news.npz',
                'analysis/*/cadence3/*.json','analysis/*/cadence3/news.npz',
                'reference-analysis/*/*.json','reference-analysis/*/news/*.json','reference-analysis/*/news/news.npz']:
    paths.update(p for p in base.glob(pattern) if p.is_file())
manifest={str(p.relative_to(a.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
with tarfile.open(a.output,'w:gz') as archive:
    for path in sorted(paths):archive.add(path,arcname=str(path.relative_to(a.root)),recursive=False)
a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(paths),'files;',a.output.stat().st_size,'bytes;',hashlib.sha256(a.output.read_bytes()).hexdigest())
