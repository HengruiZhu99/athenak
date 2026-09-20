from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;p=json.loads((root/'principal-scan.json').read_text());n=json.loads((root/'nearest-cell-all-signs.json').read_text());v=json.loads((root/'symbol-validation.json').read_text());fig,ax=plt.subplots(1,3,figsize=(14.5,4.4),layout='constrained',facecolor='white')
for scheme,color,ls in [('standard','#2c76a0','-'),('compatible','#bc592c','--')]:
 q=[z for z in p['results']if z['point'].startswith('r')and z['scheme']==scheme and z['style']=='upwind_KO05_damping'and z['shift_driver']==2];r=[float(z['point'][1:])for z in q];ax[0].semilogx(r,[z['high_frequency_max_real_eigenvalue']for z in q],marker='o',ms=3,c=color,ls=ls,label=scheme)
ax[0].axhline(0,c='.4',lw=.8);ax[0].set(xlabel='Frozen coordinate radius / M',ylabel=r'Max high-frequency Re($\lambda$) [$M^{-1}$]',title='Principal + upwind, KO0.5, κ/η');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
for key,label,color,ls in [('max_real','All phases; ξ=0 maximum','#343434','-'),('high_frequency_max_real','High frequencies, KO0.5','#2c76a0','-')]:
 q=[z for z in n['results']if z.get('scheme')=='standard'and z.get('KO_diss')==.5];ax[1].semilogx([z['h']for z in q],[z[key]for z in q],marker='o',c=color,ls=ls,label=label)
q=[z for z in n['results']if z.get('scheme')=='standard'and z.get('KO_diss')==1];ax[1].semilogx([z['h']for z in q],[z['high_frequency_max_real']for z in q],marker='s',c='#bc592c',ls='--',label='High frequencies, KO1.0');ax[1].axhline(0,c='.4',lw=.8);ax[1].invert_xaxis();ax[1].set_xticks([.5,.25,.125,.0625],['0.5','0.25','0.125','0.0625']);ax[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax[1].set(xlabel='h / M; nearest point xᵢ=h/2',ylabel=r'Full frozen Re($\lambda$) [$M^{-1}$]',title='Lower-order local modes; not global growth');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
for scheme,color in [('standard','#2c76a0'),('compatible','#bc592c')]:
 q=[z for z in v['high_frequency_conditioning']if z['point']=='flat'and z['scheme']==scheme and z['direction']=='diagonal'and z['epsilon_to_Nyquist']>0];ax[2].loglog([z['epsilon_to_Nyquist']for z in q],[z['eigenvector_condition']for z in q],marker='o',label=scheme,c=color)
ax[2].invert_xaxis();ax[2].set(xlabel='ε; ξᵢ = π − ε',ylabel='Reduced eigenvector condition number',title='Undamped Minkowski G2 near Nyquist');ax[2].legend(fontsize=8);ax[2].grid(alpha=.2);fig.savefig(root/'symbol-summary.png',dpi=170);plt.close(fig)
