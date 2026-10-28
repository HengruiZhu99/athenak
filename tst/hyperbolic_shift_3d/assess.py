"""Export bounded donor reproduction without relabeling it as 3D evolution."""
import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_1d'))
from puncture import PunctureEvolution
from numerics import weights

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interpolation(r,q,target):
    out=[]
    for x in target:
        i=max(0,min(np.searchsorted(r,x)-3,len(r)-7))
        out.append(weights(r[i:i+7],x)@q[i:i+7])
    return np.asarray(out)
def norm(q,r,mask):
    w=r*r;w=w*mask
    if not np.any(w):return None
    return float(np.sqrt(np.sum(w[:,None]*q*q)/sum(w)))
def compare(profiles):
    r=profiles[0][0]
    q=[interpolation(x,y,r) for x,y in profiles]
    masks=dict(full=np.ones(len(r),bool),puncture=r<=.02,bulk=(r>.02)&(r<.8),scri=r>=.9)
    out={}
    for name,mask in masks.items():
        d=[norm(q[i]-q[i+1],r,mask) for i in range(2)]
        out[name]=dict(differences=d,order=float(np.log2(d[0]/d[1])) if all(x and x>0 for x in d) else None)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    library=(a.build/'libspherical_z4c.so').resolve()
    records={};profiles={};histories={}
    for family in ['gamma4-reproduction','gamma4-time','gamma4-puncture-short']:
        for source in sorted((a.build/family).glob('*/manifest.json')):
            run=source.parent;m=json.loads(source.read_text());name=family+'/'+run.name
            assert m['status']!='running'
            for path,sha in m['sources'].items():assert digest(run/'source'/path)==sha,(name,path)
            for path,sha in m['checkpoint_hashes'].items():assert digest(run/path)==sha,(name,path)
            assert digest(library)==m['library_sha256']
            dest=a.output/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
            for f in ['manifest.json','history.json','profiles.npz','failed-stage.npz','speed-crossing-state.npz']:
                if (run/f).exists():shutil.copy2(run/f,dest/f)
            h=json.loads((run/'history.json').read_text());z=np.load(run/'profiles.npz')
            r=z['r'];q=z['last'].copy()
            record=dict(status=m['status'],t=m['t'],error=m['error'],final=h[-1],source_and_checkpoint_hashes_verified=True)
            if m['arguments']['puncture']:
                kw={k:v for k,v in m['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
                e=PunctureEvolution(len(r),library,**kw);e.rhs(q,diagnostics=True);v=e.last
                physical=v['physical'].copy();physical[:,0]=v['chi'];physical[:,1]=v['g']
                physical[:,6]=v['alpha'];physical[:,7]=v['beta'];q=np.c_[physical,q[:,8:]]
                w=r[1:]**2;d=v['diag'][1:,:4]
                assert np.isfinite(d).all()
                record['all_positive_radius_constraint_rms']=np.sqrt(np.sum(w[:,None]*d*d,axis=0)/sum(w)).tolist()
                record['all_positive_radius_constraint_max']=np.max(abs(d),axis=0).tolist()
                record['r0_constraint_status']='undefined physical point: donor r=0 stores AF coefficients; no positive-radius point omitted'
            else:
                q[:,0]+=1;q[:,1]+=1;q[:,3]-=3;q[:,6]+=(1+r*r)/2;q[:,7]-=r
            records[name]=record;profiles[name]=(r,q);histories[name]=h
    spatial=compare([profiles[f'gamma4-reproduction/pulse-n{n}'] for n in [65,129,257]])
    temporal=compare([profiles[f'gamma4-time/{n}'] for n in ['dt','half-dt','quarter-dt']])
    puncture=compare([profiles[f'gamma4-puncture-short/puncture-n{n}'] for n in [257,513,1025]])
    result=dict(runs=records,regular_spatial_t1=spatial,regular_temporal_t015=temporal,
        puncture_spatial_t005=puncture,fields=['chi','g','Arr','p','Theta','Lambda','alpha','beta','B','D'],
        norm='Euclidean aggregate of ten unscaled radial r^2-weighted RMS fields; seven-point interpolation onto coarse grid; regional weights renormalized',
        limitations=['All evolutions here are the unchanged spherical donor.',
            'No full 3D hyperboloidal evolution was run.',
            'Short puncture refinement does not repair the archived 100M local failure or establish gauge stationarity.',
            'AF coefficient endpoint is not an ordinary Cartesian grid point.',
            'Spatial t=1 and temporal t=.15 are different intervals; no temporal/spatial dominance ratio is asserted.'])
    (a.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for n in [65,129,257]:
        h=histories[f'gamma4-reproduction/pulse-n{n}']
        for ax,key in zip(axs[0],['H','M']):
            ax.semilogy([v['t'] for v in h],[v[key] for v in h],label=f'N={n}')
    for ax,title in zip(axs[0],['Hamiltonian constraint','Momentum constraint']):
        ax.set(xlabel='Coordinate time',ylabel='Radial-weighted RMS',title=title);ax.legend()
    for label,record in [('Pulse t=1',spatial),('Puncture t=.05',puncture)]:
        axs[1,0].plot(list(record),[v['order'] for v in record.values()],'o-',label=label)
    axs[1,0].axhline(0,color='gray',lw=.7);axs[1,0].legend();axs[1,0].set(ylabel='Observed ten-field order',title='Three-grid regional self-convergence')
    diffs=temporal['full']['differences'];axs[1,1].loglog([.05,.025],diffs,'o-')
    axs[1,1].set(xlabel='Larger dt / h in pair',ylabel='Successive time-step difference',
        title=f"Fixed N=65, t=.15; order {temporal['full']['order']:.3f}")
    fig.suptitle('Spherical donor reproduction — not 3D evolution')
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(a.output/'donor-convergence.png',dpi=170)
    plt.close(fig)
    print(json.dumps({k:result[k] for k in ['regular_spatial_t1','regular_temporal_t015','puncture_spatial_t005']},indent=2))

if __name__=='__main__':main()
