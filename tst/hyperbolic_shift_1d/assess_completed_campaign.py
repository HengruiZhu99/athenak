"""Audit every shared saved profile of completed spatial/temporal trajectories.

Read-only PDE evaluation; no evolution, interpolation, excision or restart.
"""
import argparse, inspect, json
from pathlib import Path
import numpy as np
from continue_series import verify
from continue_puncture import digest
from assess_refinement import PHYSICS, SOURCES, FIELDS
from puncture import PunctureEvolution


def load(final):
    chain=[]; seen=set(); p=final.resolve(); states={}; base=None
    while True:
        assert p not in seen; seen.add(p)
        m=verify(p)
        if base is None: base=m
        for key in PHYSICS+['n','cfl']:
            assert m['arguments'].get(key)==base['arguments'].get(key),(p,key)
        assert m['dt']==base['dt'] and m['library_sha256']==base['library_sha256']
        for name in SOURCES:
            key='tst/hyperbolic_shift_1d/'+name
            assert m['sources'][key]==base['sources'][key]
        with np.load(p/'profiles.npz') as z:
            for t,q in zip(z['t'],z['q']):
                k=round(float(t)*1e12)
                if k in states: assert np.array_equal(states[k][1],q),('overlap',p,t)
                else: states[k]=(float(t),q.copy())
        chain.append(dict(case=str(p),manifest_sha256=digest(p/'manifest.json'),checkpoints=m['checkpoint_hashes']))
        if 'parent' not in m: break
        p=Path(m['parent']['path'])
    for rel,sha in base['sources'].items():
        local=Path(rel)
        if local.suffix=='.py' and local.name not in ['assess_refinement.py','assess_short_trajectory.py','plot_gamma_speed_progress.py']:
            assert local.exists() and digest(local)==sha,('local evaluation source mismatch',rel)
    kwargs={k:v for k,v in base['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
    lib=final/'source'/Path(base['arguments']['library']).name
    e=PunctureEvolution(base['arguments']['n'],lib.resolve(),**kwargs)
    return dict(e=e,states=states,manifest=base,chain=chain)


def physical(run,k):
    t,q=run['states'][k];e=run['e'];report=e.report(q,t);v=e.last
    x=v['physical'][:,:7].copy();x[:,0]=v['chi'];x[:,1]=v['g'];x[:,6]=v['alpha'];x=np.c_[x,v['beta'],q[:,8],q[:,9]]
    assert np.isfinite(x).all()
    return x,report


def compare(runs,keys,kind):
    rows=[]
    for k in keys:
        data=[physical(run,k) for run in runs];pairs=[]
        for i in range(2):
            e=runs[i]['e'];ef=runs[i+1]['e'];r=e.r;w=r*r*e.jac;w[[0,-1]]*=.5
            if kind=='spatial': assert np.array_equal(r,ef.r[::2]);d=data[i][0]-data[i+1][0][::2]
            else: assert np.array_equal(r,ef.r);d=data[i][0]-data[i+1][0]
            ah=max(data[i][1]['apparent_horizon_r'],data[i+1][1]['apparent_horizon_r'])
            masks=dict(full=r>=0,exterior=r>=ah,interior=r<ah,puncture=r<=.02,scri_collar=r>=.9)
            pair={}
            for name,mask in masks.items():
                rms=np.sqrt(np.sum(w[mask,None]*d[mask]**2,axis=0)/np.sum(w[mask]))
                pair[name]=dict(field_rms=rms.tolist(),mixed_rms=float(np.linalg.norm(rms)),nine_field_mixed_rms=float(np.linalg.norm(rms[:-1])))
            pairs.append(pair)
        orders={name:float(np.log2(pairs[0][name]['mixed_rms']/pairs[1][name]['mixed_rms'])) if pairs[0][name]['mixed_rms']>0 and pairs[1][name]['mixed_rms']>0 else None for name in pairs[0]}
        rows.append(dict(t=runs[0]['states'][k][0],t_over_M=3*runs[0]['states'][k][0],pairs=pairs,orders=orders,
                         diagnostics=[{key:d[1][key] for key in ['H_exterior_rms','M_exterior_rms','Theta_exterior_rms','Z_exterior_rms','apparent_horizon_mass','apparent_horizon_r','g_min','H_interior_max']} for d in data]))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.cases.read_text()); cache={}
    for name in set(config['spatial']+config['temporal']):cache[name]=load(Path(name))
    groups={k:[cache[x] for x in config[k]] for k in ['spatial','temporal']}
    base=groups['spatial'][0]['manifest']
    for run in cache.values():
        m=run['manifest']
        for key in PHYSICS: assert m['arguments'].get(key)==base['arguments'].get(key),key
        assert m['library_sha256']==base['library_sha256']
        for name in SOURCES: assert m['sources']['tst/hyperbolic_shift_1d/'+name]==base['sources']['tst/hyperbolic_shift_1d/'+name]
        assert m['t']==33.3333984375
    for i in range(2):
        x,y=groups['spatial'][i:i+2];assert len(y['e'].r)==2*len(x['e'].r)-1 and x['manifest']['dt']==2*y['manifest']['dt']
        assert x['manifest']['arguments']['cfl']==y['manifest']['arguments']['cfl']
        x,y=groups['temporal'][i:i+2];assert len(x['e'].r)==len(y['e'].r) and x['manifest']['dt']==2*y['manifest']['dt']
    a.output.mkdir(parents=True,exist_ok=False);result={};provenance={}
    for kind,runs in groups.items():
        keys=sorted(set.intersection(*(set(x['states']) for x in runs)))
        assert keys[-1]==round(33.3333984375*1e12)
        rows=compare(runs,keys,kind);result[kind]=rows
        provenance[kind]=[dict(chain=x['chain'],saved_samples=len(x['states']),unshared_times=[v[0] for k,v in x['states'].items() if k not in keys]) for x in runs]
        (a.output/(kind+'.json')).write_text(json.dumps(dict(fields=FIELDS+['D'],rows=rows),indent=2)+'\n')
    spatial={round(r['t']*1e12):r for r in result['spatial']};ratios=[]
    for r in result['temporal']:
        if r['t']<=0:continue
        k=round(r['t']*1e12)
        if k in spatial:
            ratios.append(dict(t=r['t'],ratios={name:r['pairs'][0][name]['mixed_rms']/spatial[k]['pairs'][0][name]['mixed_rms'] for name in ['full','exterior']}))
    summary={}
    for kind,rows in result.items():
        positive=[r for r in rows if r['t']>0]
        summary[kind]=dict(samples=len(rows),first_t=rows[0]['t'],last_t=rows[-1]['t'],order_ranges={name:[min(r['orders'][name] for r in positive if r['orders'][name] is not None),max(r['orders'][name] for r in positive if r['orders'][name] is not None)] for name in rows[0]['orders']},nonpositive_order_samples={name:[dict(t=r['t'],order=r['orders'][name]) for r in positive if r['orders'][name] is not None and r['orders'][name]<=0] for name in rows[0]['orders']},endpoint=rows[-1])
    summary['temporal_spatial_comparison']=dict(samples=len(ratios),max_ratios={name:max(r['ratios'][name] for r in ratios) for name in ['full','exterior']},rows=ratios)
    summary['scope']='Every exact shared saved time, without time interpolation. Mixed norms contain ten unscaled fields; regional norms use their own r^2 Jacobian weights. Puncture endpoint is an asymptotic-coefficient slot with zero radial weight; all positive-radius nodes are retained. Finite-grid orders are empirical, not formal order proofs.'
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:{key:v for key,v in s.items() if key in ['samples','first_t','last_t','order_ranges']} for k,s in summary.items() if isinstance(s,dict)},indent=2))

if __name__=='__main__':main()
