import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mode_operator import ROOT
base=json.loads((ROOT/'validated-modes.json').read_text());lin=json.loads((ROOT/'linear-validated-modes.json').read_text());scaled=json.loads(next(ROOT.glob('*scaledk03-results.json')).read_text())['modes']
vals=np.array([[x['gamma'] for x in seq[:2]] for seq in [base,lin,scaled]])
fig,ax=plt.subplots(figsize=(7.5,3.6),layout='constrained',facecolor='white')
x=np.arange(3);w=.34
for j,(name,col) in enumerate([('Faster mode','#285c93'),('Slower mode','#c97b33')]):
 b=ax.bar(x+(j-.5)*w,vals[:,j],w,color=col,label=name)
 ax.bar_label(b,fmt='%.5f',padding=4,fontsize=9)
ax.set_xticks(x,['Cubic ghosts\n'+r'$\alpha\kappa_1$, $\kappa_1=0.1$', 'Linear ghosts\n'+r'$\alpha\kappa_1$, $\kappa_1=0.1$', 'Cubic ghosts\n'+r'lapse-scaled $\kappa_1=0.3$'])
ax.set_ylabel(r'Growth rate $\gamma$ [$M^{-1}$]');ax.set_ylim(0,.055)
ax.legend(frameon=False,loc='upper right');ax.set_title(r'Vacuum full-step operator, $G=2$: growing modes persist')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(ROOT/'mode-configuration-comparison.png',dpi=180,bbox_inches='tight')
fig.savefig(ROOT/'mode-configuration-comparison.pdf',bbox_inches='tight')
