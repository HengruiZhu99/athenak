"""Plot the committed analytic spatial checks, not evolution convergence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
p=Path(__file__).resolve().parent
rows=json.loads((p/'geometry-results.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for o,label,color in [(0,'Previous V7','0.35'),(3,'Three evolved overlap layers','#1675b8')]:
    q=sorted([r for r in rows if r['overlap']==o],key=lambda r:r['n']);n=[r['n'] for r in q]
    ax[0].semilogy(n,[r['max_weight_l1'] for r in q],'o-',label=label,color=color)
    ax[1].loglog(n,[r['hessian_max'] for r in q],'o-',label=label,color=color)
for a in ax:
    a.set_xlabel('Nominal cells per block direction')
    a.set_xticks([12,16,24],['12','16','24']);a.xaxis.set_minor_locator(NullLocator());a.grid(alpha=.2)
ax[0].set_ylabel('Maximum interpolation weight absolute sum')
ax[1].set_ylabel('Maximum Hessian component error')
ax[0].set_title('Inner-interface halo plan')
ax[1].set_title(r'Transition shell: $f=\sin(x+2y-3z)$')
ax[1].legend(fontsize=8)
fig.suptitle('Spatial checks only — evolution stability is still being tested',fontsize=11)
fig.savefig(p/'spatial-comparison.png',dpi=180)
