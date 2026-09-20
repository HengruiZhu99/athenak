from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
j=json.loads((ROOT/'covariant_constant10-arnoldi32-results.json').read_text())
v=json.loads((ROOT/'sigma1-validated-modes.json').read_text())[0]
fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),layout='constrained',facecolor='white')
kk=[];gg=[];ee=[];rr=[];re=[]
for h in j['history']:
    if h['k']<20:continue
    c=[m for m in h['ritz'] if m['imag']>0]
    c=min(c,key=lambda m:abs(m['omega']-v['omega']))
    r=[m for m in h['ritz'] if abs(m['imag'])<1e-12]
    r=max(r,key=lambda m:m['gamma'])
    kk.append(h['k']);gg.append(c['gamma']);ee.append(c['relative_residual'])
    rr.append(r['gamma']);re.append(r['relative_residual'])
axes[0].plot(kk,gg,'o-',color='#1768ac',label='Oscillatory pair')
axes[0].plot(kk,rr,'s--',color='#b76a26',label='Real candidate: unresolved')
axes[0].axhline(0,color='.3',lw=.7)
axes[0].set_ylabel(r'Approximate growth rate $\gamma$ [$M^{-1}$]')
axes[0].set_ylim(-.001,.007)
axes[0].legend(frameon=False,fontsize=9)
axes[1].semilogy(kk,ee,'o-',color='#1768ac',label='Pair, Arnoldi')
axes[1].semilogy(kk,re,'s--',color='#b76a26',label='Real candidate, Arnoldi')
axes[1].scatter([32,32],[v['eigen_residual']['relative_l2'],v['eigen_residual']['active_relative_l2']],
                marker='x',s=70,color=['black','#65943f'],label='Pair, direct full / active')
axes[1].set_ylabel('Relative eigen-shape residual')
axes[1].legend(frameon=False,fontsize=8,loc='lower left')
for ax in axes:
    ax.set_xlabel('Krylov dimension')
    ax.grid(alpha=.2)
for fmt in ['png','pdf']:
    fig.savefig(ROOT/f'sigma1-mode-convergence.{fmt}',dpi=180)
