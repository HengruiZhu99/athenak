"""Plot the frozen uniform-grid curl test, including the factorized negative control."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);a=p.parse_args()
cpu=pd.read_csv(a.evidence/'curl-cpu.csv',comment='#');gpu=pd.read_csv(a.evidence/'curl-cuda.csv',comment='#')
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
x=np.arange(len(gpu));labels=[f'FD{int(r.order)}\n{int(r.dim)}D/{int(r.n)}' for r in gpu.itertuples()]
axes[0].semilogy(x,cpu.curl_scaled,'o',label='CPU serial',markersize=4)
axes[0].semilogy(x,gpu.curl_scaled,'x',label='CUDA A100',markersize=5)
axes[0].axhline(64,color='black',linestyle='--',label='Predeclared limit (64)')
axes[0].set(ylabel='Curl / (2 eps max|rho w| ||D_i||₁ ||D_j||₁)',title='Direct product gradient: scaled roundoff')
axes[1].semilogy(x,gpu.old_curl_max,'o-',label='Factorized target',color='#737373')
axes[1].semilogy(x,gpu.curl_max,'o-',label='Direct product gradient',color='#0072B2')
axes[1].set(ylabel='Maximum absolute discrete curl',title='Same fields, grid and finite differences')
for ax in axes:
    ax.set_xticks(x,labels,fontsize=7);ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.savefig(a.evidence/'correctness.png',dpi=180)
