#!/usr/bin/env python3
"""Retain signed, point-matched witnesses from completed native diagnostics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

TERMS=['clock','2 alpha_n(P+3)','P_n','2 kappa(alpha_n-1)',
       '-beta_nn','s','-2','alpha_n','raw s_t','projected s_t',
       'identity residual','raw-minus-chart gzz_t']

def run(base):
    data=np.load(base/'pointwise.npz')
    times=data['times'];x=data['x'];rhs=data['T_rhs'];fd=data['T_fd']
    lapse=data['lapse_terms'];penalty=data['penalty_rates'];pole=data['pole_over_h']
    digest=hashlib.sha256()
    with (base/'pointwise.npz').open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    report={'source':str(base/'pointwise.npz'),'pointwise_sha256':digest.hexdigest(),
            'summary_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'witnesses':{},'windows':{}}
    for label,mask in [('interior',np.arange(len(times))>=3),
                       ('early',3*times<=.15),('late',3*times>=.5)]:
        mask=mask & (np.arange(len(times))>=3) & (np.arange(len(times))<len(times)-3)
        indices=np.flatnonzero(mask)
        if not len(indices):continue
        local,point,a,b=np.unravel_index(np.argmax(abs(rhs[indices])),rhs[indices].shape)
        q=indices[local]
        report['witnesses'][label]={'time_M':float(3*times[q]),'point_index':int(point),
            'x':x[point].tolist(),'largest_component':[int(a),int(b)],
            'T_rhs':rhs[q,point].tolist(),'T_fd':fd[q,point].tolist(),
            'lapse':dict(zip(TERMS,lapse[q,point].tolist())),
            'penalty_rates':penalty[q,point].tolist(),'pole_over_h':pole[q,point].tolist()}
        report['windows'][label]={'time_M':[float(3*times[indices[0]]),float(3*times[indices[-1]])],
            'T_rhs_max':float(np.max(abs(rhs[indices]))),
            'T_path_difference_max':float(np.max(abs(rhs[indices]-fd[indices]))),
            'clock_max':float(np.max(abs(lapse[indices,:,0])))}
    q,point=np.unravel_index(np.argmax(abs(lapse[:,:,0])),lapse[:,:,0].shape)
    report['witnesses']['clock']={'time_M':float(3*times[q]),'point_index':int(point),
        'x':x[point].tolist(),'T_rhs':rhs[q,point].tolist(),'T_fd':fd[q,point].tolist(),
        'lapse':dict(zip(TERMS,lapse[q,point].tolist())),
        'penalty_rates':penalty[q,point].tolist(),'pole_over_h':pole[q,point].tolist()}
    report['initial']={'T_rhs_max':float(np.max(abs(rhs[0]))),
        'clock_max':float(np.max(abs(lapse[0,:,0]))),'lapse_identity_error':float(np.max(abs(lapse[:,:,10])))}
    (base/'pointwise-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def compare_half(baseline,half):
    a=np.load(baseline/'pointwise.npz');b=np.load(half/'pointwise.npz')
    assert np.array_equal(a['x'],b['x'])
    time_difference=float(np.max(abs(a['times']-b['times'])))
    assert time_difference<1e-13,time_difference
    report={'native_time_mismatch':time_difference}
    for key in ('T_rhs','T_fd','lapse_terms','penalty_rates','pole_over_h'):
        report[key+'_difference_max']=float(np.max(abs(a[key][3:-3]-b[key][3:-3])))
    (half/'halfdt-pointwise-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);a=p.parse_args();run(a.analysis)
