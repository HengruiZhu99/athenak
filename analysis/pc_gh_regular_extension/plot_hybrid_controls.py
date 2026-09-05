"""Standalone scientific figures from measured hybrid controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('summary',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(a.summary.read_text())['pulses']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for row in rows:
        if row['family']!='p' or 'reductions_l2_over_amplitude' not in row:continue
        for ax,sector in zip(axes,['reductions','curls']):
            ax.loglog([256,512,1024],row[sector+'_l2_over_amplitude'],'o-',label=row['candidate'])
    for ax,title in zip(axes,['Reduction error','Curl error']):
        ax.set(title=title,xlabel='Uniform cells N',ylabel='L2 error / initial pulse amplitude',xticks=[256,512,1024])
        ax.set_xticklabels(['256','512','1024']);ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(alpha=.18,which='both')
    axes[0].legend(ncol=3,fontsize=9)
    fig.suptitle('Independent characteristic oracle: fixed p pulse, FD6/RK3, KO = 0.3')
    for ext in ['png','pdf']:fig.savefig(a.output/f'pulse-convergence.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
