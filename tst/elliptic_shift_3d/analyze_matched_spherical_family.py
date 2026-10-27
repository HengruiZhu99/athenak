#!/usr/bin/env python3
"""All-cell radial comparison, exact-radius angular scatter, and honest gates."""
import argparse,gzip,hashlib,io,json,os,tempfile
from pathlib import Path
import numpy as np
from radial_snapshot import RadialSnapshot
from radial_reference import interpolate
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
root=a.input.resolve();ref=a.reference.resolve();manifest=json.loads((root/'manifest.json').read_text())
profiles=sorted((ref/'short-n1025').glob('profile-*.npz'))
assert len(profiles)>=2
references={'initial':RadialSnapshot(profiles[0]),'final':RadialSnapshot(profiles[-1])}
fine_reference=RadialSnapshot(sorted((ref/'short-n2049').glob('profile-*.npz'))[-1])
assert fine_reference.time==references['final'].time
fields={'chi':0,'alpha':1,'A_rr':2,'Khat':3,'Theta':4,'Lambda_r':5,'beta_r':6,'B':7}
def profile_bytes(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.csv.gz').read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'scope':'Matched bulk equations/gauge and RK variable; spatial stencil, KO and interface differences retained. No wormhole-to-trumpet validation.',
        'fine_reference_sha256':sha(fine_reference.path),'reference_time':references['final'].time,'cases':{},'reference_hashes':{k:sha(r.path) for k,r in references.items()},
        'all_cells_retained':True,'angular_scatter':'Actual nodes grouped by radius rounded to 12 decimals, no angular projection or symmetry enforcement',
        'input_manifest_sha256':sha(root/'manifest.json')}
def norms(x,w):return {'rms':float(np.sqrt(np.average(x*x,weights=w))),'max':float(np.max(abs(x)))}
raw={}
for name,case in manifest['cases'].items():
    if 'final' not in case:continue
    result['cases'][name]={}
    for label,refdata in references.items():
        f=root/name/(label+'.csv');payload=profile_bytes(f);data=np.genfromtxt(io.BytesIO(payload),delimiter=',',names=True);raw[(name,label)]=data
        assert len(data)==case[label]['cells'] and np.isfinite(data.view(float)).all()
        x=data['r'];w=data['weight'];report={'fields':{},'regions':{},'cells':len(data)}
        for field,column in fields.items():
            truth=interpolate(refdata.r[1:],refdata.values[1:,column],x)
            report['fields'][field]=norms(data[field]-truth,w)
            if label=='final':
                fine=interpolate(fine_reference.r[1:],fine_reference.values[1:,column],x)
                report.setdefault('N1025_minus_N2049_reference_control',{})[field]=norms(truth-fine,w)
        report['fields']['g_rr']=norms(data['g_rr']-1,w)
        for region,mask in [('all',x>=0),('puncture_r_lt_01',x<.1),('core_r_lt_025',x<.25),('cube_transition',(x>=.25)&(x<.45)),('shells',x>=.45)]:
            if not mask.any():report['regions'][region]={'cells':0};continue
            report['regions'][region]={'cells':int(mask.sum()),'H':norms(data['H'][mask],w[mask]),'M':norms(data['M_norm'][mask],w[mask]),'freeze':norms(np.sqrt(sum(data['freeze_'+axis][mask]**2 for axis in 'xyz')),w[mask])}
        radii,inverse=np.unique(np.round(x,12),return_inverse=True)
        count=np.bincount(inverse);angular={'r':radii.tolist(),'count':count.tolist()}
        for field in fields:
            sums=np.bincount(inverse,weights=data[field]);mean=sums/count
            sd=np.sqrt(np.bincount(inverse,weights=(data[field]-mean[inverse])**2)/count)
            angular[field]={'mean':mean.tolist(),'std':sd.tolist()}
        report['angular']=angular;report['profile_sha256']=hashlib.sha256(payload).hexdigest()
        result['cases'][name][label]=report
if ('n6','final') in raw and ('n6-floor-off','final') in raw:
    result['floor_off_control']={label:profile_bytes(root/'n6'/(label+'.csv'))==profile_bytes(root/'n6-floor-off'/(label+'.csv')) for label in ['initial','final']}
(root/'radial-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
with tempfile.TemporaryDirectory(prefix='matched-plots-') as config:
    os.environ.setdefault('MPLCONFIGDIR',config)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.5),layout='constrained')
    refdata=references['final'];mask=(refdata.r>0)&(refdata.r<1)
    for ax,field in zip(axes[0],['alpha','beta_r']):
        ax.plot(refdata.r[mask],refdata.values[mask,fields[field]],'k-',lw=1.5,label='1D N1025 reference')
        for name,color in [('n6','tab:blue'),('n8','tab:orange'),('n10','tab:green')]:
            if name not in result['cases']:continue
            row=result['cases'][name]['final']['angular'];r=np.array(row['r']);mean=np.array(row[field]['mean']);sd=np.array(row[field]['std'])
            ax.plot(r,mean,'.',ms=2.6,color=color,label=name.upper()+' 3D angular mean')
            ax.fill_between(r,mean-sd,mean+sd,color=color,alpha=.2)
        ax.set(xlabel='Compact radius r',ylabel='Conformal lapse α' if field=='alpha' else 'Radial shift βʳ')
    axes[0,0].legend(fontsize=8)
    for name,color in [('n6','tab:blue'),('n8','tab:orange'),('n10','tab:green')]:
        if name not in result['cases']:continue
        data=raw[(name,'final')];r=data['r'];truth=interpolate(refdata.r[1:],refdata.values[1:,1],r)
        axes[1,0].scatter(r,np.maximum(abs(data['alpha']-truth),1e-18),s=1,alpha=.18,color=color,rasterized=True,label=name.upper())
        angular=result['cases'][name]['final']['angular'];ur=np.array(angular['r']);idx=np.searchsorted(ur,np.round(r,12));fv=np.sqrt(sum(data['freeze_'+axis]**2 for axis in 'xyz'))
        peak=np.zeros(len(ur));np.maximum.at(peak,idx,fv)
        axes[1,1].plot(ur,peak,'.',ms=3,color=color,label=name.upper())
    axes[1,0].set(yscale='log',xlabel='Compact radius r',ylabel='|α₃ᴅ − α₁ᴅ|, every active cell')
    axes[1,1].set(yscale='log',xlabel='Compact radius r',ylabel='max angular |Vₕ,t|')
    axes[1,1].axhline(1e-8,color='k',ls='--',lw=1,label='RMS gate (scale reference)')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(f'Matched spherical gauge in 3D, t={references["final"].time:.8f}\nShort coarse pilot: discrete gauge gate fails; no collapse validation',fontsize=12)
    fig.savefig(root/'matched-radial-comparison.png',dpi=170);fig.savefig(root/'matched-radial-comparison.pdf');plt.close(fig)
print(json.dumps({k:{f:v['final']['fields'][f] for f in ['alpha','beta_r','g_rr']} for k,v in result['cases'].items()},indent=2))
