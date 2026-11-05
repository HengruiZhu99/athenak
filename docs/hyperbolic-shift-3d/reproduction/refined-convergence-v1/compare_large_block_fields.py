"""Sample admitted distinct-grid endpoints at common coordinates, two orders."""
from pathlib import Path
import argparse,json,gzip,csv
import numpy as np
from sample_rf2_fields import Checkpoint,Geometry,sample,FIELDS,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    campaign=a.root/'campaigns/boosted-block12-block24-pair-v1'
    terminal=json.loads((campaign/'terminal-verification.json').read_text())
    local=json.loads((Path(__file__).parent/'large-block-results/local-verification.json').read_text())
    assert terminal['worker_status']=='completed' and terminal['scheduler_exit']==0 and local['all_verified']
    radii=np.linspace(.301,.99,200)
    directions=np.array([[1,0,0],[-1,0,0],[0,1,0],[1,1,1]],dtype=float)
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    points=np.concatenate([radii[:,None]*d for d in directions])
    samples={};metas={};records={}
    for name,v,cfg in zip(['block12','block24'],terminal['cases'],
                         [a.root/'candidate.cfg',a.root/'large-block-pair-v1/candidate.cfg']):
        cp=Checkpoint(campaign/'output'/name/'state.checkpoint',sha256=v['sha256'],identity=v['identity'])
        geom=Geometry(cp.leaves,cp.subdivisions,cfg)
        samples[name]={};metas[name]={}
        for order in [6,8]:
            samples[name][order],metas[name][order]=sample(cp,geom,points,stencil=order)
        records[name]=dict(checkpoint_sha256=v['sha256'],configuration_sha256=digest(cfg),
            interpolation_max_scaled_difference=float(np.max(abs(samples[name][8]-samples[name][6])/(1+abs(samples[name][8])))),
            max_weight_l1=max(x['weight_l1'] for x in metas[name][8]))
    comparison={}
    for j,f in enumerate(FIELDS):
        diff=samples['block24'][8][:,j]-samples['block12'][8][:,j]
        interp=max(float(np.max(abs(samples[n][8][:,j]-samples[n][6][:,j]))) for n in samples)
        comparison[f]=dict(max_abs_difference=float(np.max(abs(diff))),
                          rms_difference=float(np.sqrt(np.mean(diff*diff))),
                          max_6_8_interpolation_difference=interp,
                          max_difference_xyz=points[int(np.argmax(abs(diff)))].tolist())
    for name in samples:
        records[name]['theta_ray_bands']={}
        for lo,hi in [(.3,.6),(.6,.78),(.78,.875),(.875,.95),(.95,.975),(.975,1.)]:
            mask=(np.tile(radii,len(directions))>=lo)&(np.tile(radii,len(directions))<hi)
            theta=samples[name][8][mask,14]
            xyz=points[mask][int(np.argmax(abs(theta)))]
            records[name]['theta_ray_bands'][f'{lo}-{hi}']=dict(max_abs=float(max(abs(theta))),xyz=xyz.tolist())
    with gzip.open(out/'samples.csv.gz','wt') as f:
        w=csv.writer(f);w.writerow(['case','order','ray','r','x','y','z','block','weight_l1']+FIELDS)
        for name in samples:
            for order in [6,8]:
                for i,(x,q,meta) in enumerate(zip(points,samples[name][order],metas[name][order])):
                    w.writerow([name,order,i//len(radii),radii[i%len(radii)],*x,meta['block'],meta['weight_l1'],*q])
    result=dict(records=records,fields=comparison,points=len(points),
        qualifier='Common coordinate samples outside r=.3; stored fields, not constraint interpolation. Different grids/volumes/interfaces. Both six/eight-point tensor interpolation preserved; interpolation sensitivity is not a rigorous error bound. No reference truth, accuracy/convergence/stability or 10M claim.',
        source_sha256=digest(Path(__file__)))
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,key in zip([0,18,22,13,14,25],['chi','lapse','shift x','P','Theta','D_x']):
        aplot=ax.flat[[0,18,22,13,14,25].index(j)]
        for name,style in [('block12','-'),('block24','--')]:
            v=samples[name][8][:len(radii),j].copy()
            if j==18:v+=(1+radii*radii)/2
            if j==22:v-=radii
            if j==13:v-=3
            aplot.plot(radii,v,style,label=name)
        aplot.set_xlabel('compact r on +x ray');aplot.set_ylabel(key);aplot.grid(alpha=.2)
    ax[0,0].legend();fig.suptitle('Common-coordinate fields at .075M; interpolated, no smoothing')
    for ext in ['png','pdf']:fig.savefig(out/('fields.'+ext),dpi=170)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
