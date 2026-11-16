#!/usr/bin/env python3
"""Plot the dense arrival pilot; retain absolute errors for both geometries."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
for name,label,color in [('control-N12','Schwarzschild control','C0'),('gauge-N12','Pure-lapse seed','C1')]:
    root=a.analysis/name;points=np.load(root/'pointwise.npz');news=np.load(root/'news/news.npz')
    diagnostics=json.loads((root/'news/diagnostics.json').read_text());t=3*points['times'];tn=3*news['native_times']
    axes[0,0].semilogy(news['bondi_time_over_M0'],np.max(abs(news['news_modes']),axis=1),color=color,label=label)
    axes[0,1].semilogy(t,np.max(abs(points['T_rhs']),axis=(1,2,3)),color=color,label=label+' RHS')
    axes[0,1].semilogy(t,np.max(abs(points['T_fd']),axis=(1,2,3)),color=color,ls='--',alpha=.7)
    axes[0,2].semilogy(t,np.max(abs(points['lapse_terms'][:,:,0]),axis=1),color=color)
    axes[1,0].semilogy(tn,[d['round_cut_max'] for d in diagnostics],color=color)
    axes[1,2].semilogy(t[3:-3],points['metric_rate_difference'][3:-3],color=color)
    if name=='gauge-N12':
        for k,part in enumerate(['Ricci','Conformal Hessian','Conformal gradient']):
            axes[1,1].semilogy(tn,np.max(abs(news['native_cut_decomposition'][:,k]),axis=1),label=part)
        axes[1,1].semilogy(tn,np.max(abs(news['native_cut_news_coefficients']),axis=1),color='k',ls='--',label='Sum')
titles=['All-mode false news, $\\ell\\leq6$','Full $T_{ab}$: rates (solid), records (dashed)','Unprojected lapse compatibility','Transported cut-metric defect','Same-cut gauge-case contributions','Metric rate: RHS versus records']
ylabels=['$\\max_{\\ell m}|N_{\\ell m}|$','$\\max_{q,a,b}|T_{ab}|$','$\\max_q|\\mathcal{C}_\\alpha|$','Cut defect norm','Maximum mode magnitude','$\\max|\\dot g_{ab}^{RHS}-\\dot g_{ab}^{FD}|$']
for k,ax in enumerate(axes.flat):
    ax.set_title(titles[k],fontsize=10);ax.set_ylabel(ylabels[k]);ax.set_xlabel('$u_B/M$' if k==0 else '$t/M$');ax.grid(alpha=.2)
    if k!=0:ax.axvspan(.0541894,.0823968,color='grey',alpha=.12)
axes[0,0].legend(fontsize=8);axes[1,1].legend(fontsize=7)
fig.suptitle('N12 arrival pilot: unchanged A, initial lapse amplitude 0.001',fontsize=12)
for suffix in ('png','pdf'):fig.savefig(a.output/f'pilot-diagnostics.{suffix}',dpi=180)
