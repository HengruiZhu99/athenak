"""Read source-bound checkpoint ancestry and aggregate complete stage extrema."""
import hashlib,json
from pathlib import Path
import numpy as np


def read_trajectory(folder,seen=None,allow_domain_transfer=False):
    folder=Path(folder).resolve();seen=set() if seen is None else seen
    if folder in seen:raise ValueError('Checkpoint ancestry cycle')
    seen.add(folder);m=json.loads((folder/'manifest.json').read_text())
    transition=m.get('domain_transfer_parent')
    if transition:
        if not allow_domain_transfer:raise ValueError('Domain transition requires explicit retained-domain assessment and a separate original full-prefix assessment')
        source_manifest=Path(transition['source_manifest']);profile=Path(transition['profile'])
        if hashlib.sha256(source_manifest.read_bytes()).hexdigest()!=transition['source_manifest_sha256']:raise ValueError('Domain source manifest hash changed')
        if hashlib.sha256(profile.read_bytes()).hexdigest()!=transition['profile_sha256']:raise ValueError('Domain source profile hash changed')
        if json.loads(source_manifest.read_text())['binding']!=transition['source_binding']:raise ValueError('Domain source binding changed')
    rows=[];stages={};segments=[]
    if m.get('checkpoint_parent'):
        parent=m['checkpoint_parent'];checkpoint=Path(parent['path'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=parent['sha256']:raise ValueError('Parent checkpoint hash changed')
        with np.load(checkpoint,allow_pickle=False) as z:metadata=json.loads(str(z['metadata']))
        pm,rows,stages,segments=read_trajectory(checkpoint.parent,seen,allow_domain_transfer)
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
        scope='retained-domain-only; original full AF prefix required separately' if transition else 'full-trajectory',
        domain_transfer_parent=transition,
        manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        history_sha256=hashlib.sha256(history.read_bytes()).hexdigest()))
    return m,rows,stages,segments


def prefix_stage_extrema(rows,until):
    """Use only recorded cumulative stages through an exact saved report.

    A later manifest's extrema cannot be subtracted to recover a prefix.
    Every included report must carry its actual all-stage instrumentation;
    sparse old diagnostics are never upgraded to all-step coverage.
    """
    selected=[row for row in rows if row['t']<=until+1e-10]
    if not selected or abs(selected[-1]['t']-until)>1e-10:raise ValueError('Prefix endpoint requires an exact recorded report')
    result={}
    for row in selected:
        for key,value in row.get('all_RK_stages',{}).items():result[key]=(min if key.startswith('min_') else max)(result.get(key,value),value)
    if 'max_accepted_H_rms' not in result:raise ValueError('Prefix lacks every-accepted-step constraint instrumentation')
    for row in selected:
        for name in ['H','M','Theta','Z']:
            for suffix,report_suffix in [('rms','rms'),('bulk','bulk_max'),('endpoint','endpoint_max')]:
                key='max_accepted_'+name+'_'+suffix
                if key in result:result[key]=max(result[key],row[name+'_'+report_suffix])
        for key,field in [('max_scri_pole','scri_pole'),('max_lapse_pole','scri_lapse_pole'),
          ('max_regularity_trace','regularity_trace_max'),('max_scri_null','scri_null'),
          ('max_scri_clock','scri_clock'),('max_Bondi_nonaffinity','Bondi_frame_nonaffinity')]:
            if field in row:result[key]=max(result.get(key,0),abs(row[field]))
    return result
