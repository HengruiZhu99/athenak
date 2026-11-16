#!/usr/bin/env python3
"""Summarize archived native constraint and signed scri histories, without fits."""
import argparse,json,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();report={}
for path in sorted(a.root.glob('**/*.hst')):
    if 'seed' in path.parts:continue
    labels=[]
    with path.open() as stream:
        for line in stream:
            if not line.startswith('#'):break
            found=re.findall(r'\[(\d+)\]=(\S+)',line)
            if found:labels=[name for index,name in sorted(found,key=lambda p:int(p[0]))]
    data=np.loadtxt(path,ndmin=2)
    assert data.shape[1]==len(labels),(path,data.shape,labels)
    assert np.isfinite(data).all(),path
    report[str(path.relative_to(a.root))]={'rows':len(data),'first_record':dict(zip(labels,data[0].tolist())),
        'final':dict(zip(labels,data[-1].tolist())),'max_abs':dict(zip(labels,np.max(abs(data),axis=0).tolist()))}
a.output.write_text(json.dumps(report,indent=2)+'\n')
