"""Read source-bound checkpoint ancestry and aggregate complete stage extrema."""
import hashlib,json
from pathlib import Path
import numpy as np


def read_trajectory(folder,seen=None):
    folder=Path(folder).resolve();seen=set() if seen is None else seen
    if folder in seen:raise ValueError('Checkpoint ancestry cycle')
    seen.add(folder);m=json.loads((folder/'manifest.json').read_text())
    rows=[];stages={};segments=[]
    if m.get('checkpoint_parent'):
        parent=m['checkpoint_parent'];checkpoint=Path(parent['path'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=parent['sha256']:raise ValueError('Parent checkpoint hash changed')
        with np.load(checkpoint,allow_pickle=False) as z:metadata=json.loads(str(z['metadata']))
        pm,rows,stages,segments=read_trajectory(checkpoint.parent,seen)
        if pm['status']!='finished' or pm['binding']!=m['binding'] or metadata['binding']!=m['binding']:
            raise ValueError('Continuation is not an exact completed-parent source binding')
    history=folder/'history.jsonl';local=[json.loads(line) for line in history.read_text().splitlines()]
    if rows and local and local[0]['t']<rows[-1]['t']-1e-10:raise ValueError('Nonmonotonic continuation time')
    rows=rows+local
    for key,value in m.get('all_RK_stages',{}).items():stages[key]=(min if key.startswith('min_') else max)(stages.get(key,value),value)
    # k1 samples every accepted step input, but the last updated state has no
    # subsequent k1. Include its independent report (and any other reports).
    # Never promote old, sparsely sampled histories to all-step coverage.
    if 'max_accepted_H_rms' in m.get('all_RK_stages',{}):
        for name in ['H','M','Theta','Z']:
            for suffix,report_suffix in [('rms','rms'),('bulk','bulk_max'),('endpoint','endpoint_max')]:
                key='max_accepted_'+name+'_'+suffix
                if key not in stages:continue
                stages[key]=max(stages[key],max(row[name+'_'+report_suffix] for row in local))
    segments.append(dict(folder=str(folder),status=m['status'],time=m.get('time'),
        manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        history_sha256=hashlib.sha256(history.read_bytes()).hexdigest()))
    return m,rows,stages,segments
