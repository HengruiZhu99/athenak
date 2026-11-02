"""Reproduce the fixed-cutoff, matching-operator comparison of admitted endpoints."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[4]
p=Path(__file__).resolve().parent/'comparison'
files={'A':('a-radial.json','native-shared-transition','n16-native-transition-v2-a-evolution-segment01'), 'A-wide':('wide-radial.json','wide-collar-a','n16-wide-collar-a-v1-a-evolution-segment01'), 'A-broad-smooth':('broad-radial.json','restored-wide-smooth','n16-restored-wide-smooth-v1-a-evolution-segment01')}
fig,axes=plt.subplots(1,2,figsize=(11,4));summary={}
for label,(filename,raw,campaign) in files.items():
    d=json.loads((p/filename).read_text());assert d['cutoff']==.12 and d['t_over_M']==10 and d['steps']==6667
    report=json.loads((root/f'output/{raw}/aurora/campaigns/{campaign}/output/puncture.report.json').read_text())
    rows=d['radial_cumulative'];r=np.array([s['cutoff'] for s in rows]);volume=-np.diff([s['volume'] for s in rows])
    exterior=next(s for s in rows if s['cutoff']==.12);regions={}
    for lo,hi in [(.16,.3),(.3,.6),(.6,.7),(.6,.9),(.7,.9),(.7,.775),(.775,.9),(.9,1.)]:
        below=next(s for s in rows if s['cutoff']==lo);above=next(s for s in rows if s['cutoff']==hi)
        regions[f'{lo}-{hi}']={k:{'integral':below[k]-above[k],'fraction_of_exterior':(below[k]-above[k])/exterior[k]} for k in ['H2','M2','Mr2']}
    # Native reports and reconstruction use the same full-domain measure/operator.
    checks={}
    for key in ['H_L2','M_L2','Mr_L2']:
        actual=d['norms']['total'][key];expected=report[key]
        checks[key]=abs(actual-expected)/max(abs(expected),1e-30)
        assert checks[key]<1e-9,(label,key,actual,expected)
    sensitivity={str(c):{k:np.sqrt(next(s for s in rows if s['cutoff']==c)[k]/next(s for s in rows if s['cutoff']==c)['volume']) for k in ['H2','M2','Mr2']} for c in [.1099,.12,.13]}
    summary[label]={'cutoff':.12,'norms':d['norms'],'regions':regions,'full_norm_report_relative_differences':checks,'fixed_cutoff_rms_sensitivity':sensitivity}
    for ax,k in zip(axes,['H2','M2']):
        sums=-np.diff([s[k] for s in rows]);assert sums.min()>-1e-12
        value=np.sqrt(np.maximum(sums,0)/np.maximum(volume,1e-30));keep=r[:-1]>=.12
        ax.stairs(value[keep],r[np.r_[keep,True]],label=label,color={'A':'#c75d19','A-wide':'#1769aa','A-broad-smooth':'#b02871'}[label],linestyle='--' if label=='A' else '-')
        ax.axvspan(.775,.9,color='gray',alpha=.12)
        ax.set(xlim=(.12,1),xlabel='compactified radius r',ylabel=k[:-1]+' regional RMS',yscale='log');ax.grid(alpha=.2)
changes={'exterior_rms_percent':{k:100*(summary['A-wide']['norms']['exterior'][k]/summary['A']['norms']['exterior'][k]-1) for k in ['H_L2','M_L2','Mr_L2']},'regional_integral_percent':{region:{k:100*(summary['A-wide']['regions'][region][k]['integral']/summary['A']['regions'][region][k]['integral']-1) for k in ['H2','M2','Mr2']} for region in summary['A']['regions']}}
changes_broad={'exterior_rms_percent':{k:100*(summary['A-broad-smooth']['norms']['exterior'][k]/summary['A']['norms']['exterior'][k]-1) for k in ['H_L2','M_L2','Mr_L2']},'regional_integral_percent':{region:{k:100*(summary['A-broad-smooth']['regions'][region][k]['integral']/summary['A']['regions'][region][k]['integral']-1) for k in ['H2','M2','Mr2']} for region in summary['A']['regions']}}
axes[0].legend();fig.suptitle('Restored A variants at 10M, N16; fixed cutoff r > 0.12');fig.tight_layout();fig.savefig(p/'radial.png',dpi=180);plt.close(fig)
(p/'radial-comparison.json').write_text(json.dumps({'scope':'Matching native operators and fixed coordinate-volume cutoffs; N16 combined-width/initial-slice comparison, not resolution convergence. .1099 sensitivity cut lies below the A-wide final mean horizon radius.','cases':summary,'wide_vs_A':changes,'broad_smooth_vs_A':changes_broad},indent=2)+'\n')
print(json.dumps(changes_broad,indent=2))
