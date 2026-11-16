#!/usr/bin/env python3
"""Common-cut wide-pulse convergence and isolated reference-jet comparison."""
import argparse,json
from pathlib import Path
import numpy as np
from nonradiative_report import aligned,savefig,plt,resolution_axis

def load(p):return json.loads(p.read_text())

def report(root,legacy,out,partial=False):
    out.mkdir(parents=True,exist_ok=True)
    paths=list(root.glob('runs/N*-*/analysis/*/summary.json'))
    cases={p.parent.name:p.parent for p in paths}
    assert len(cases)==len(paths),'duplicate principal analyses; select the authoritative run explicitly'
    if not partial:assert set(cases)=={f'{f}-N{n}' for f in ('control','gauge') for n in (12,16,20)}
    result={'complete_spatial_sequence':len(cases)==6,'cases':{},'common_cuts':{},'reference':{}}
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for row,family in enumerate(('control','gauge')):
        archives=[];ns=[];native_series=[]
        for n in (12,16,20):
            name=f'{family}-N{n}'
            if name not in cases:continue
            case=cases[name];s=load(case/'summary.json');a=np.load(case/'news/news.npz');series=load(case/'series.json');diagnostics=load(case/'news/diagnostics.json')
            archives.append(a);ns.append(n)
            stride=3 if n==12 and family=='gauge' else 1
            native_series.append({k:np.asarray(series[k])[::stride] for k in ('time','T_rhs')})
            color={12:'C0',16:'C1',20:'C2'}[n]
            axes[row,0].semilogy(a['bondi_time_over_M0'],np.max(abs(a['news_modes']),axis=1),color=color,label=f'N={n}')
            axes[row,1].semilogy(3*np.array(series['time'])[3:-3],np.array(series['T_rhs'])[3:-3],color=color)
            axes[row,2].semilogy(3*a['native_times'],[d['round_cut_max'] for d in diagnostics],color=color)
            result['cases'][name]={k:s[k] for k in ['records','end_M','T_rhs_max','T_path_difference','clock_max','lapse_identity_error','derivative_reconstruction_error']}
            result['cases'][name].update(analysis_source=str(case),all_mode_peak=float(np.max(abs(a['news_modes']))),diagnostics=s['news']['diagnostics_max_abs'])
            result['cases'][name]['boundary_diagnostic_maxima']={
                'scope':'Matching nominal native cadence; omit three common-cadence records at each end.',
                'normal_lift_value_gradient_Hessian':np.max(np.abs(np.asarray(series['lift_jet_max'])[::stride][3:-3]),axis=0).tolist(),
                'pole_over_h_chi_P_Theta_alpha':np.max(np.abs(np.asarray(series['pole_over_h_max'])[::stride][3:-3]),axis=0).tolist(),
                'penalty_rate_max':float(np.max(np.abs(np.asarray(series['penalty_max'])[::stride][3:-3])))}
            for file in ['tensor-attribution.json','cadence-comparison.json']:
                if (case/file).exists():result['cases'][name][file]=load(case/file)
        if archives:
            t,news_values,labels=aligned(archives);errors=[float(np.max(abs(x))) for x in news_values]
            c={'N':ns,'Bondi_interval_M':[float(3*t[0]),float(3*t[-1])],'all_mode_error':errors,
               'adjacent_effective_orders':[float(np.log(errors[k]/errors[k+1])/np.log(ns[k+1]/ns[k])) for k in range(len(ns)-1)],'windows':{}}
            for label,lo,hi in [('short_Bondi_window',0,.7182463172),('intermediate_Bondi_window',.15,1.2),('late_Bondi_window',1.9,3)]:
                mask=(3*t>=lo)&(3*t<=hi)
                if np.any(mask):c['windows'][label]={'Bondi_interval_M':[float(3*t[mask][0]),float(3*t[mask][-1])],'all_mode_error':[float(np.max(abs(x[mask]))) for x in news_values]}
            start=max(v['time'][3] for v in native_series);end=min(v['time'][-4] for v in native_series)
            for v in native_series[1:]:
                assert v['time'].shape==native_series[0]['time'].shape
                assert np.max(abs(v['time']-native_series[0]['time']))<1e-12
            terr=[float(np.max(v['T_rhs'][3:-3])) for v in native_series]
            c['native_comparison']='Matched nominal native cuts at common physical cadence; N12 dense gauge series uses stride three; timestamp tolerance 1e-12.'
            c['common_native_interval_M']=[3*start,3*end];c['full_T_error']=terr
            c['full_T_adjacent_effective_orders']=[float(np.log(terr[k]/terr[k+1])/np.log(ns[k+1]/ns[k])) for k in range(len(ns)-1)]
            result['common_cuts'][family]=c
            modes=np.stack(news_values)
            assert modes.shape==(len(ns),len(t),len(labels)) and np.all(np.isfinite(modes))
            np.savez_compressed(out/f'{family}-common-cuts.npz',time_M=3*t,N=ns,ell_m=labels,news_modes=modes)
        for col,title in enumerate(['All-mode false news','Full projected-rate $T_{ab}$','Transported cut-metric defect']):
            axes[row,col].set_title(f'{family.capitalize()}: {title}',fontsize=10);axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('$u_B/M$' if col==0 else '$t/M$')
        axes[row,0].legend(fontsize=8)
    savefig(fig,out,'wide-errors')
    for rank,family in enumerate(('control','gauge')):
        for p in root.glob(f'runs/N16-*/wide-N12-attribution-rank-{rank}.json'):
            result['cases'][f'{family}-N12']['tensor-attribution.json']=load(p)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,family in zip(axes,('control','gauge')):
        available=[n for n in (12,16,20) if f'{family}-N{n}' in cases]
        if not available:continue
        n=max(available);name=f'{family}-N{n}';a=np.load(cases[name]/'news/news.npz');parts=a['native_cut_decomposition']
        result['cases'][name]['native_cut_piece_maxima']=np.max(abs(parts),axis=(0,2)).tolist()
        result['cases'][name]['native_cut_piece_sum_error']=float(np.max(abs(np.sum(parts,axis=1)-a['native_cut_news_coefficients'])))
        for k,label in enumerate(('Ricci','Hessian','Gradient')):ax.semilogy(3*a['native_times'],np.max(abs(parts[:,k]),axis=1),label=label)
        ax.semilogy(3*a['native_times'],np.max(abs(a['native_cut_news_coefficients']),axis=1),'k--',label='Sum')
        ax.set_title(f'{family.capitalize()}, N={n}: identical cuts/dyads',fontsize=10);ax.set_xlabel('$t/M$');ax.set_ylabel('All-mode magnitude');ax.grid(alpha=.2);ax.legend(fontsize=8)
    savefig(fig,out,'wide-decomposition')
    if 'gauge-N12' in cases:
        comparison_archives=[];comparison_native=[]
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for label,case in [('Narrow pulse',legacy/'analysis/native-N12-9093556/gauge-N12'),('Wide pulse',cases['gauge-N12']),('Control',cases['control-N12'])]:
            a=np.load(case/'news/news.npz');s=load(case/'series.json')
            if label!='Control':comparison_archives.append(a);comparison_native.append(s)
            axes[0].semilogy(a['bondi_time_over_M0'],np.max(abs(a['news_modes']),axis=1),label=label)
            axes[1].semilogy(3*np.array(s['time'])[3:-3],np.array(s['T_rhs'])[3:-3],label=label)
        for i,ax in enumerate(axes):
            ax.set_xlim(0,.7182463172);ax.set_xlabel('$u_B/M$' if i==0 else '$t/M$');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_title('N12: absolute all-mode false news');axes[1].set_title('N12: absolute full regularity error')
        savefig(fig,out,'narrow-wide-first-window')
        t,v,labels=aligned(comparison_archives)
        lo=max(vv['time'][3] for vv in comparison_native);hi=min(vv['time'][-4] for vv in comparison_native)
        result['narrow_vs_wide_N12']={'order':['narrow','wide'],'common_Bondi_interval_M':[float(3*t[0]),float(3*t[-1])],
            'all_mode_error':[float(np.max(abs(vv))) for vv in v],'common_native_interval_M':[3*lo,3*hi],
            'full_T_error':[float(np.max(np.asarray(vv['T_rhs'])[(np.asarray(vv['time'])>=lo-1e-12)&(np.asarray(vv['time'])<=hi+1e-12)])) for vv in comparison_native]}
    probes=list(root.glob('runs/N16-*/reference-analysis/control-N12/summary.json'))
    if probes:
        assert len(probes)==1;probe=probes[0].parent;baseline=cases['control-N12']
        a,b=[np.load(p/'news/news.npz') for p in (baseline,probe)];t,v,labels=aligned([a,b])
        result['reference']={'baseline_summary':load(baseline/'summary.json'),'corrected_summary':load(probe/'summary.json'),
             'common_Bondi_interval_M':[float(3*t[0]),float(3*t[-1])],
             'news_mode_difference_max':float(np.max(abs(v[0]-v[1])))}
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for label,path,archive in [('Original reference jets',baseline,a),('Analytic plateau jets',probe,b)]:
            s=load(path/'series.json');axes[0].semilogy(archive['bondi_time_over_M0'],np.max(abs(archive['news_modes']),axis=1),label=label)
            axes[1].semilogy(3*np.array(s['time'])[3:-3],np.array(s['T_rhs'])[3:-3],label=label)
        for i,ax in enumerate(axes):ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('$u_B/M$' if i==0 else '$t/M$')
        axes[0].set_title('N12 control: false news');axes[1].set_title('N12 control: full regularity error')
        savefig(fig,out,'reference-jet-comparison')
    result['step_control']={}
    for p in root.glob('runs/halfdt-*/*comparison.json'):
        result['step_control'][p.name]=load(p)
    for p in root.glob('runs/halfdt-*/analysis/frame-l8/news/news.npz'):
        a=np.load(cases['gauge-N12']/'cadence3/news.npz');b=np.load(p);t,v,labels=aligned([a,b])
        result['step_control']['lmax6_vs8_all_common_modes']={'common_ell_max':6,
            'Bondi_interval_M':[float(3*t[0]),float(3*t[-1])],'news_mode_difference_max':float(np.max(abs(v[0]-v[1])))}
    result['constraints']={}
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for row,family in enumerate(('control','gauge')):
        for n in (12,16,20):
            name=f'{family}-N{n}'
            if name not in cases:continue
            run=cases[name].parent.parent/name;color={12:'C0',16:'C1',20:'C2'}[n]
            full=np.loadtxt(next(run.glob('*.z4c_hyperboloidal.hst')))
            outer=np.loadtxt(next(run.glob('*.scri-shell-3.hst')))
            result['constraints'][name]={'final_time_M':float(3*full[-1,0]),'full_H_rms':float(full[-1,3]),'exterior_H_rms':float(full[-1,8]),'exterior_M_rms':float(full[-1,9]),'outer_band_H_rms':float(outer[-1,3]),'outer_band_M_rms':float(outer[-1,4])}
            for col,(a,index) in enumerate([(full,8),(full,9),(outer,3)]):axes[row,col].semilogy(3*a[:,0],a[:,index],color=color,label=f'N={n}')
        for col,title in enumerate(['Exterior Hamiltonian RMS','Exterior momentum RMS','Hamiltonian RMS, .97<=r<1']):
            axes[row,col].set_title(f'{family.capitalize()}: {title}',fontsize=10);axes[row,col].set_xlabel('$t/M$');axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
    savefig(fig,out,'constraint-errors')
    result['radial_constraint_onset']={}
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    bands=['.80<=r<.88','.88<=r<.94','.94<=r<.97','.97<=r<1']
    for row,family in enumerate(('control','gauge')):
        for col,n in enumerate((12,16,20)):
            ax=axes[row,col];name=f'{family}-N{n}'
            ax.set_title(f'{family.capitalize()}, N={n}',fontsize=10);ax.set_xlabel('$t/M$');ax.set_xlim(0,.75);ax.grid(alpha=.2)
            if name not in cases:continue
            run=cases[name].parent.parent/name;entry={}
            for k,band in enumerate(bands):
                a=np.loadtxt(next(run.glob(f'*.scri-shell-{k}.hst')))
                ax.semilogy(3*a[:,0],a[:,3],label=band)
                ids=[0,int(np.argmin(abs(3*a[:,0]-.15))),int(np.argmin(abs(3*a[:,0]-.3)))]
                entry[band]=[{'time_M':float(3*a[j,0]),'H_rms':float(a[j,3]),'M_rms':float(a[j,4])} for j in ids]
            result['radial_constraint_onset'][name]=entry
            if col==0:ax.set_ylabel('Hamiltonian RMS');ax.legend(fontsize=7)
    savefig(fig,out,'radial-constraint-onset')
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for family,marker in (('control','o'),('gauge','s')):
        c=result['common_cuts'][family];ns=np.asarray(c['N'])
        values=[c['all_mode_error'],c['full_T_error'],
                [result['constraints'][f'{family}-N{n}']['exterior_H_rms'] for n in ns]]
        for ax,v in zip(axes,values):ax.loglog(ns,v,marker+'-',label=family.capitalize())
    for ax,title in zip(axes,('All-mode news, common Bondi cuts','Full tensor, matched native cuts','Exterior Hamiltonian, final cut')):
        ax.set_title(title,fontsize=10);ax.set_xlabel('N');ax.set_ylabel('Absolute error')
        resolution_axis(ax);ax.grid(alpha=.2);ax.legend(fontsize=8)
    savefig(fig,out,'wide-convergence')
    (out/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('legacy',type=Path);p.add_argument('output',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args()
    report(a.root,a.legacy,a.output,a.partial)
