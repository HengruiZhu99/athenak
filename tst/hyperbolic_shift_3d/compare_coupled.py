"""Matched-time 3D/1D diagnostics; no spherical projection of evolved states."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from horizon import apparent_horizons

def read(p):return np.genfromtxt(p,delimiter=',',names=True)
def horizon_rays(data):
    results=[]
    for ray in np.array_split(data,14):
        r=ray['r'];effective_chi=(r/(((1-r*r)/2)*ray['area_radius']))**2
        good=np.isfinite(effective_chi)&np.isfinite(ray['expansion'])
        estimate=apparent_horizons(r[good],effective_chi[good],ray['expansion'][good])
        if 'apparent_horizon_r' in estimate:results.append(estimate)
    if not results:return dict(valid_rays=0)
    radii=np.array([x['apparent_horizon_r'] for x in results]);mass=np.array([x['apparent_horizon_mass'] for x in results])
    return dict(valid_rays=len(results),coordinate_radius_mean=float(np.mean(radii)),coordinate_radius_spread=float(np.ptp(radii)),
                mass_mean=float(np.mean(mass)),mass_fractional_error=float(np.mean(mass)*3-1),mass_spread=float(np.ptp(mass)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--reanalyzed',type=Path);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    reference=read(a.baseline/'final.csv');report=json.loads((a.baseline/'report.json').read_text());t=report['t']
    history=json.loads((a.baseline/'history.json').read_text());rh=history[-1].get('apparent_horizon_r',0.)
    fields=['chi','alpha','beta_r','g_rr','H','M_r'];summary=dict(t=t,t_over_M=3*t if report.get('kind','puncture')=='puncture' else None,reference_horizon=rh,runs=[],
        horizon_method='14 ray coordinate-sphere expansion zeros; spherical diagnostic, not a general 3D apparent-horizon solve',
        normalization='static compactified Cartesian coordinate volume; exterior normalized, r > contemporaneous 1D horizon')
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,field in zip(axes.flat,fields):
        ax.plot(reference['r'][1:-1],reference[field][1:-1],color='black',lw=1.5,label='1D N=513')
        ax.set(xlabel='compact radius r',ylabel=field,xlim=(0,1));ax.axvline(rh,color='gray',ls=':',lw=.7)
    for run in a.runs:
        m=json.loads((run/'evolution.json').read_text());assert m['status']=='completed' and abs(m['t']-t)<1e-12
        prefix=a.reanalyzed/f'n{m["n"]}' if a.reanalyzed else run/'evolution'
        v=read(str(prefix)+'.diagnostics.csv');rays=read(str(prefix)+'.rays.csv');coords=read(run/'evolution.coordinates.csv')
        exterior=v['r']>rh;w=coords['volume'][exterior];w=w/sum(w)
        entry=dict(n=m['n'],cells=m['cells'],dt=m['dt'],null_max=m['null_max'],pole_max=m['pole_max'],
                   compatibility_max=m['compatibility_max'],horizon=horizon_rays(rays),rms_error={},constraint_rms={},angular_rms={})
        for field in ['chi','alpha','beta_r','g_rr','A_rr','P','Theta','Lambda_r','B_r','D_r']:
            ref=np.interp(v['r'][exterior],reference['r'],reference[field]);entry['rms_error'][field]=float(np.sqrt(sum(w*(v[field][exterior]-ref)**2)))
        for field in ['H','M_r','Z_r','Theta','M_norm','Z_norm','det_error','tracefree_error']:
            entry['constraint_rms'][field]=float(np.sqrt(sum(w*v[field][exterior]**2)))
        for ax,field in zip(axes.flat,fields):
            samples=rays[field].reshape(14,-1);r=rays['r'][:199];valid=np.any(np.isfinite(samples),axis=0);samples=samples[:,valid];r=r[valid];mean=np.nanmean(samples,axis=0)
            ax.plot(r,mean,lw=.9,label=f"3D n={m['n']}")
            ax.fill_between(r,np.nanmin(samples,axis=0),np.nanmax(samples,axis=0),alpha=.13)
            mask=r>rh;entry['angular_rms'][field]=float(np.sqrt(np.nanmean((samples[:,mask]-mean[None,mask])**2)))
        summary['runs'].append(entry)
    axes[0,0].legend(fontsize=8);time_label=f't/M={3*t:g}' if report.get('kind','puncture')=='puncture' else f't={t:g}'
    fig.suptitle(f'Hyperboloidal {report.get("kind","puncture")}: {time_label}; bands show angular range')
    fig.savefig(a.output/'comparison.png',dpi=180);fig.savefig(a.output/'comparison.pdf');plt.close(fig)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
