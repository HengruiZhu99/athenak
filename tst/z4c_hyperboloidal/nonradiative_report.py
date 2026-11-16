#!/usr/bin/env python3
"""Compact numerical tables and publication-exportable nonradiative plots."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullLocator,StrMethodFormatter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vis/python'))
from scri_news import interpolate_time

def load(path):return json.loads(path.read_text())
def savefig(fig,out,name):
    for suffix in ('png','pdf'):fig.savefig(out/(name+'.'+suffix),dpi=180)
    plt.close(fig)
def resolution_axis(ax):
    ax.xaxis.set_major_locator(FixedLocator([12,16,20]))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_locator(NullLocator())
def aligned(archives,limit=None):
    labels=set(map(tuple,archives[0]['ell_m']))
    for a in archives[1:]:labels &= set(map(tuple,a['ell_m']))
    labels=sorted(pair for pair in labels if limit is None or pair[0]<=limit)
    begin=max(a['bondi_time'][0] for a in archives);end=min(a['bondi_time'][-1] for a in archives)
    assert end>begin
    # Include every original cut in the common interval, so sharp sampled
    # maxima are not lost by downsampling the comparison grid.
    times=np.unique(np.concatenate([np.linspace(begin,end,401)]+[a['bondi_time'][(a['bondi_time']>=begin)&(a['bondi_time']<=end)] for a in archives]));values=[]
    for a in archives:
        indices={tuple(pair):i for i,pair in enumerate(a['ell_m'])}
        values.append(np.column_stack([interpolate_time(a['bondi_time'],a['news_modes'][:,indices[pair]],times) for pair in labels]))
    return times,values,labels

def native(roots,out):
    cases={}
    for root in roots:
        for path in root.glob('*/summary.json'):
            summary=load(path)
            if 'T_rhs_max' in summary:cases[path.parent.name]=(path.parent,summary)
    if not cases:return {}
    report={'cases':{},'convergence':{},'controls':{}}
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for row,family in enumerate(('control','gauge')):
        archives=[];resolutions=[]
        for n in (12,16,20):
            name=f'{family}-N{n}'
            if name not in cases:continue
            path,summary=cases[name];data=np.load(path/'news/news.npz');series=load(path/'series.json')
            diagnostics=load(path/'news/diagnostics.json');archives.append(data);resolutions.append(n)
            color={12:'C0',16:'C1',20:'C2'}[n]
            axes[row,0].semilogy(data['bondi_time_over_M0'],np.max(abs(data['news_modes']),axis=1),color=color,label=f'N={n}')
            axes[row,1].semilogy(3*np.array(series['time']),series['T_rhs'],color=color)
            axes[row,2].semilogy(3*data['native_times'],[d['round_cut_max'] for d in diagnostics],color=color)
            report['cases'][name]={key:summary[key] for key in ('records','end_M','T_rhs_max','T_fd_max','T_path_difference','clock_max','lapse_identity_error','metric_rate_difference_max','derivative_reconstruction_error','seconds')}
            report['cases'][name].update(all_mode_peak=float(np.max(abs(data['news_modes']))),diagnostics=summary['news']['diagnostics_max_abs'])
            for filename,key in [('pointwise-summary.json','pointwise'),('halfdt-pointwise-comparison.json','halfdt_pointwise')]:
                if (path/filename).exists():report['cases'][name][key]=load(path/filename)
            pieces=data['native_cut_decomposition']
            report['cases'][name]['native_cut_piece_maxima']=np.max(abs(pieces),axis=(0,2)).tolist()
            report['cases'][name]['native_cut_piece_sum_error']=float(np.max(abs(np.sum(pieces,axis=1)-data['native_cut_news_coefficients'])))
            report['cases'][name]['endpoint_T_rhs']=series['T_rhs'][-1]
            report['cases'][name]['pole_over_h_max']=np.max(series['pole_over_h_max'],axis=0).tolist()
            report['cases'][name]['normal_lift_jet_max']=np.max(series['lift_jet_max'],axis=0).tolist()
            report['cases'][name]['penalty_rate_max']=float(np.max(series['penalty_max']))
        if archives:
            times,values,labels=aligned(archives)
            errors=[float(np.max(abs(v))) for v in values]
            orders=[float(np.log(errors[k]/errors[k+1])/np.log(resolutions[k+1]/resolutions[k])) for k in range(len(errors)-1)]
            report['convergence'][family]={'N':resolutions,'common_Bondi_interval_M':[float(3*times[0]),float(3*times[-1])],'all_mode_absolute_error':errors,'adjacent_effective_order':orders}
            for label,mask in [('early',3*times<=.15),('late',3*times>=.5)]:
                if np.any(mask):report['convergence'][family][label+'_all_mode_error']=[float(np.max(abs(v[mask]))) for v in values]
            tensor_errors=[report['cases'][f'{family}-N{n}']['T_rhs_max'] for n in resolutions]
            report['convergence'][family]['T_rhs_error']=tensor_errors
            report['convergence'][family]['T_rhs_adjacent_order']=[float(np.log(tensor_errors[k]/tensor_errors[k+1])/np.log(resolutions[k+1]/resolutions[k])) for k in range(len(tensor_errors)-1)]
            np.savez_compressed(out/(family+'-common-cuts.npz'),time_M=3*times,N=resolutions,ell_m=labels,news_modes=values)
        for col,metric in enumerate(('All-mode false news, $\\ell\\leq6$','Full $T_{ab}$ from projected rates','Transported cut-metric defect')):
            axes[row,col].set_title(('Control: ' if row==0 else 'Gauge: ')+metric,fontsize=10)
            axes[row,col].set_xlabel('$u_B/M$' if col==0 else '$t/M$');axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
    axes[0,0].set_ylabel('$\\max_{\\ell m}|N_{\\ell m}|$');axes[1,0].set_ylabel('$\\max_{\\ell m}|N_{\\ell m}|$')
    axes[0,1].set_ylabel('$\\max_{q,a,b}|T_{ab}|$');axes[1,1].set_ylabel('$\\max_{q,a,b}|T_{ab}|$')
    savefig(fig,out,'native-errors')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,family in zip(axes,('control','gauge')):
        available=[n for n in (12,16,20) if f'{family}-N{n}' in cases]
        if not available:continue
        n=max(available);path,_=cases[f'{family}-N{n}'];data=np.load(path/'news/news.npz')
        for k,label in enumerate(('Ricci','Hessian','Gradient')):
            ax.semilogy(3*data['native_times'],np.max(abs(data['native_cut_decomposition'][:,k]),axis=1),label=label)
        ax.semilogy(3*data['native_times'],np.max(abs(data['native_cut_news_coefficients']),axis=1),'k--',label='Sum')
        ax.set_title(f'{family.capitalize()}, N={n}: identical cuts and dyads',fontsize=10)
        ax.set_xlabel('$t/M$');ax.set_ylabel('All-mode magnitude');ax.grid(alpha=.2);ax.legend(fontsize=8)
    savefig(fig,out,'native-decomposition')
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for n in (12,16,20):
        name=f'gauge-N{n}'
        if name not in cases:continue
        path,_=cases[name];series=load(path/'series.json');d=load(path/'news/diagnostics.json')
        t=3*np.array(series['time']);color={12:'C0',16:'C1',20:'C2'}[n]
        clock=np.maximum(abs(np.array(series['lapse_min'])[:,0]),abs(np.array(series['lapse_max'])[:,0]))
        values=[clock,series['T_difference'],None,np.array(series['lift_jet_max'])[:,2],
                np.max(series['pole_over_h_max'],axis=1),np.max(series['penalty_max'],axis=1)]
        for panel,(ax,value) in enumerate(zip(axes.flat,values)):
            if value is not None:
                cut=slice(3,-3) if panel==1 else slice(None)
                ax.semilogy(t[cut],np.asarray(value)[cut],color=color,label=f'N={n}')
        data=np.load(path/'news/news.npz');cut=3*data['native_times']
        axes[0,2].semilogy(cut,[v['null_max'] for v in d],color=color)
        axes[0,2].semilogy(cut,[v['shear_max'] for v in d],':',color=color)
    titles=['Signed lapse-equation residual: max absolute','Full tensor: projected-rate minus time-record path',
            'Nullness (solid) and tangential shear (dotted)','Normal lift: largest second derivative',
            'Largest pole/h among chi, P, Theta, lapse','Largest boundary penalty rate']
    for ax,title in zip(axes.flat,titles):ax.set_title(title,fontsize=9);ax.set_xlabel('$t/M$');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);savefig(fig,out,'native-regularity-diagnostics')
    if 'gauge-N12' in cases:
        baseline=cases['gauge-N12'][0];ref=np.load(baseline/'news/news.npz')
        if (baseline/'controls.json').exists():
            control_data=load(baseline/'controls.json')
            report['cadence_T']={stride:{k:v for k,v in row.items() if k!='news'} for stride,row in control_data['cadence'].items()}
        if 'gauge-N12-halfdt' in cases:
            halfpath=cases['gauge-N12-halfdt'][0]
            if (halfpath/'halfdt-pointwise-comparison.json').exists():report['halfdt_pointwise']=load(halfpath/'halfdt-pointwise-comparison.json')
        controls=[]
        if 'gauge-N12-halfdt' in cases:controls.append(('half evolution timestep',cases['gauge-N12-halfdt'][0]/'news/news.npz'))
        for name in ('news-L4','news-L8','news-stride2','news-stride4'):
            path=baseline/name/'news.npz'
            if path.exists():controls.append((name,path))
        for name,path in controls:
            candidate=np.load(path);times,values,labels=aligned([ref,candidate])
            report['controls'][name]={'common_Bondi_interval_M':[float(3*times[0]),float(3*times[-1])],
                'retained_lmax':int(max(l for l,m in labels)),'complex_mode_difference_max':float(np.max(abs(values[0]-values[1]))),
                'baseline_absolute_error':float(np.max(abs(values[0]))),'control_absolute_error':float(np.max(abs(values[1])))}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for family,values in report['convergence'].items():
        axes[0].loglog(values['N'],values['all_mode_absolute_error'],'o-',label=family)
    axes[0].set_xlabel('Boundary resolution N');axes[0].set_ylabel('Absolute all-mode false-news error');axes[0].legend();axes[0].grid(alpha=.2)
    resolution_axis(axes[0])
    if report['controls']:
        names=list(report['controls']);labels={'half evolution timestep':'Half evolution dt','news-L4':'Angular L=4','news-L8':'Angular L=8','news-stride2':'Cadence ×2','news-stride4':'Cadence ×4'}
        axes[1].barh([labels[n] for n in names],[report['controls'][n]['complex_mode_difference_max'] for n in names]);axes[1].set_xscale('log');axes[1].set_xlabel('Common-cut complex-mode difference\nfrom N12 gauge baseline');axes[1].grid(axis='x',alpha=.2)
    savefig(fig,out,'native-convergence-controls')
    return report

def analytic(roots,out):
    cases={}
    for root in roots:
        for path in root.glob('*/summary.json'):cases[path.parent.name]=(path.parent,load(path))
    if not cases:return {}
    report={name:s for name,(p,s) in cases.items()}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for family,param,height,label in [('rotation',.1,1,'rotation 0.1'),('rotation',.8,1,'rotation 0.8'),('retarded_bounce',2,1,'retarded bounce'),('gauge',.01,1,'gauge 0.01, H=1'),('gauge',.01,16,'gauge 0.01, H=16'),('gauge',10,16,'gauge 10, H=16')]:
        rows=[]
        for n in (12,16,20):
            name=f'rows-N{n}-{family}-{param}'+('-H16' if height==16 else '')
            if name in cases:rows.append((n,cases[name][1]))
        if rows:
            axes[0,0].loglog([n for n,s in rows],[s['all_mode_max'] for n,s in rows],'o-',label=label)
            axes[0,1].loglog([n for n,s in rows],[s['T_max'] for n,s in rows],'o-',label=label)
    for col,title in enumerate(('Actual donor rows: all-mode false news','Actual donor rows: full $T_{ab}$')):
        axes[0,col].set_title(title,fontsize=10);axes[0,col].set_xlabel('Boundary resolution N');axes[0,col].grid(alpha=.2)
        resolution_axis(axes[0,col])
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=8)
    for dt in (.01,.005):
        name=f'conformal-{dt}-L4'
        if name in cases:
            data=np.load(cases[name][0]/'cuts.npz');axes[1,0].semilogy(3*data['times'],np.max(abs(data['modes']),axis=1),label=f'dt={dt} code')
            if dt==.01:
                for k,label in enumerate(('Ricci','Hessian','Gradient')):axes[1,1].semilogy(3*data['times'],np.max(abs(data['pieces'][:,k]),axis=1),label=label)
                axes[1,1].semilogy(3*data['times'],np.max(abs(data['modes']),axis=1),'k--',label='Sum')
    axes[1,0].set_title('Exact jets: nontrivial conformal-frame cancellation',fontsize=10)
    axes[1,1].set_title('Separate terms on identical cuts and dyads',fontsize=10)
    for ax in axes[1]:ax.set_xlabel('$t/M$');ax.set_ylabel('All-mode magnitude');ax.grid(alpha=.2);ax.legend(fontsize=8)
    savefig(fig,out,'analytic-errors')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,action='append',default=[]);p.add_argument('--analytic',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report={'native':native(a.native,a.output),'analytic':analytic(a.analytic,a.output)}
    (a.output/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
