import argparse
"""Reproduce the fixed-cutoff, matching-operator comparison of admitted endpoints."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[4]
parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent/'comparison');p=parser.parse_args().directory
files={'A':('a-radial.json','native-shared-transition','n16-native-transition-v2-a-evolution-segment01'), 'A-six-shells*':('six-radial.json','restored-six-shells','n16-restored-six-shells-v1-a-evolution-segment01'), 'A-broad-smooth':('broad-radial.json','restored-wide-smooth','n16-restored-wide-smooth-v1-a-evolution-segment01')}
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
        value=np.full_like(volume,np.nan);nonempty=volume>1e-14;value[nonempty]=np.sqrt(np.maximum(sums[nonempty],0)/volume[nonempty]);keep=r[:-1]>=.12
        ax.stairs(value[keep],r[np.r_[keep,True]],baseline=None,label=label,color={'A':'#c75d19','A-six-shells*':'#008060','A-broad-smooth':'#b02871'}[label],linestyle='--' if label=='A' else '-')
        ax.axvspan(.775,.9,color='gray',alpha=.12)
        ax.set(xlim=(.12,1),xlabel='compactified radius r',ylabel=k[:-1]+' regional RMS',yscale='log');ax.grid(alpha=.2)
changes={}
for baseline in ['A','A-broad-smooth']:
 changes[baseline]={'exterior_rms_percent':{k:100*(summary['A-six-shells*']['norms']['exterior'][k]/summary[baseline]['norms']['exterior'][k]-1) for k in ['H_L2','M_L2','Mr_L2']},'regional_integral_percent':{region:{k:100*(summary['A-six-shells*']['regions'][region][k]['integral']/summary[baseline]['regions'][region][k]['integral']-1) for k in ['H2','M2','Mr2']} for region in summary[baseline]['regions']}}
axes[0].legend(fontsize=8);fig.suptitle('N16 at 10M: original A, broad/smooth 3 shells, and 6 shells*')
fig.tight_layout(rect=(0,.10,1,.94));fig.text(.5,.025,'* Six-shell native evolution completed; frozen penalty-policy check failed (scale 0.651).\nMatching native derivative operators, fixed cutoff r > 0.12. Not a matched-parameter convergence test.',ha='center',fontsize=8)
fig.savefig(p/'radial.png',dpi=180);plt.close(fig)
(p/'radial-comparison.json').write_text(json.dumps({'scope':'Native-completed checkpoints and matching original derivative operators. Six-shell penalty-policy failure retained. Fixed coordinate-volume cutoff .12 lies outside all final horizon surfaces; anisotropic refinement with added interfaces and active boundary limiter is not a convergence proof.','cases':summary,'six_vs_baselines':changes},indent=2)+'\n')
print(json.dumps(changes,indent=2))
