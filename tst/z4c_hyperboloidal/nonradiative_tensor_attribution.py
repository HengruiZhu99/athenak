#!/usr/bin/env python3
"""Instantaneous full-T attribution at saved extrema, never an evolved ablation.

At fixed metric values the tracefree Hessian is affine in first metric
derivatives. Split its actual time derivative into geometric and chi-penalty
parts, and its spatial derivative into unlifted and normal-lift parts. This
does not re-solve compatibility, change a state, or establish a stable closure.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vis/python'))
from scri_reader import read
from scri_geometry import endpoint_jets,spacetime_metric
from nonradiative_native_analysis import metric_rate

def tensor(metric,derivatives,x):
    inverse=np.linalg.inv(metric)
    normal=np.einsum('qab,qb->qa',inverse,np.column_stack((np.zeros(len(x)),-x)))
    # Direct contracted Koszul formula, independent of Ricci/connection assembly.
    h=np.diag([0.,-1.,-1.,-1.])[None,:,:]-.5*(
        np.einsum('qd,qadb->qab',normal,derivatives)
        +np.einsum('qd,qbda->qab',normal,derivatives)
        -np.einsum('qd,qdab->qab',normal,derivatives))
    return h-metric*np.einsum('qab,qab->q',inverse,h)[:,None,None]/4

def run(base,output=None,peak_only=False):
    records=json.loads((base/'cache/records.json').read_text())
    pointwise=np.load(base/'pointwise.npz',mmap_mode='r')
    witnesses=json.loads((base/'pointwise-summary.json').read_text())['witnesses']
    selections={name:int(np.argmin(abs(pointwise['times']-w['time_M']/3))) for name,w in witnesses.items()}
    selections['initial']=0
    if peak_only:selections={'interior':selections['interior']}
    report={}
    for label,q in selections.items():
        record=records[q];path=Path(record['source']);parts=[]
        for rank in range(record['ranks']):
            f=path.with_name(path.name.replace('r00000000.scri',f'r{rank:08d}.scri'))
            state=read(f);epoch=read(f.parent/state['epoch']);parts.append((state,epoch))
        ids=np.concatenate([e['surface_ids'][:,[0,2,3]] for s,e in parts]);order=np.lexsort(ids.T[::-1])
        x=np.concatenate([e['surface_geometry'][:,:3] for s,e in parts])[order]
        jets=np.concatenate([endpoint_jets(s,e) for s,e in parts])[order]
        rates=np.concatenate([s['projected_metric_rates'] for s,e in parts])[order]
        penalties=np.concatenate([s['boundary_penalty_rates'] for s,e in parts])[order]
        lifts=np.concatenate([s['normal_lift_jets'] for s,e in parts])[order]
        assert lifts.shape[1]==3 and np.max(abs(penalties[:,1:7]))==0
        native=spacetime_metric(jets,x);g=native[...,0]
        dg=np.empty((len(x),4,4,4));dg[:,0]=metric_rate(jets,x,rates)
        dg[:,1:]=np.moveaxis(native[...,1:4],-1,-3)
        full=tensor(g,dg,x)
        # Exact chi-penalty tangent of beta=-sqrt(chi*Q)*n at fixed gtilde.
        gamma=g[:,1:,1:];Q=np.einsum('qi,qij,qj->q',x,np.linalg.inv(gamma),x)/jets[:,0,0]
        speed=np.sqrt(jets[:,0,0]*Q)
        rate_p=np.zeros_like(rates);rate_p[:,0]=penalties[:,0]
        rate_p[:,8:]=-x*(penalties[:,0]*Q/(2*speed))[:,None]
        no_penalty=dg.copy();no_penalty[:,0]-=metric_rate(jets,x,rate_p)
        p_piece=full-tensor(g,no_penalty,x)
        bare=jets.copy();bare[:,22:25]-=lifts
        bare_metric=spacetime_metric(bare,x)
        no_lift=dg.copy();no_lift[:,1:]=np.moveaxis(bare_metric[...,1:4],-1,-3)
        l_piece=full-tensor(g,no_lift,x)
        neither=no_penalty.copy();neither[:,1:]=no_lift[:,1:]
        remainder=tensor(g,neither,x)
        stored=pointwise['T_rhs'][q]
        point=int(np.unravel_index(np.argmax(abs(full)),full.shape)[0])
        component=np.unravel_index(np.argmax(abs(full[point])),(4,4))
        report[label]={'time_M':3*record['time'],'point_index':point,'x':x[point].tolist(),
            'component':list(map(int,component)),
            'signed_witness':{name:float(a[(point,)+component]) for name,a in
               [('full',full),('chi_penalty',p_piece),('shift_normal_lift',l_piece),('remainder',remainder)]},
            'full_surface_maxima':{name:float(np.max(abs(a))) for name,a in
               [('full',full),('chi_penalty',p_piece),('shift_normal_lift',l_piece),('remainder',remainder)]},
            'stored_tensor_difference':float(np.max(abs(full-stored))),
            'additivity_error':float(np.max(abs(full-p_piece-l_piece-remainder)))}
        assert report[label]['stored_tensor_difference']<1e-9
        assert report[label]['additivity_error']<1e-9
    result={'scope':__doc__,'analysis_case':str(base),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cuts':report}
    (output or base/'tensor-attribution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'case':base.name,'cuts':report}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',type=Path);p.add_argument('--peak-only',action='store_true');a=p.parse_args();run(a.case.resolve(),a.output,a.peak_only)
