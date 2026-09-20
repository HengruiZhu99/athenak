#!/usr/bin/env python3
"""Plot checkpoint samples of incoming C1, not physical constraint norms."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    cases=[('8842248','theta_primary',r'$A=10^{-6},\ d_\alpha=0.01$'),
           ('8842248','theta_lapse01',r'$A=10^{-6},\ d_\alpha=0.1$'),
           ('8842283','theta_amplitude',r'$A=10^{-7},\ d_\alpha=0.01$')]
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
    records={};summary={'scope':__doc__,'cases':{}}
    for job,case,label in cases:
        path=root/'incoming-gpu-final'/('face-trace-gpu-'+job+'-'+case+'.json')
        d=json.loads(path.read_text());rows=d['rows'];records[case]=rows
        times=np.array([r['time'] for r in rows]);c=np.array([r['C1_in_actual_basis'] for r in rows])
        a=rows[0]['seed']['amplitude'];scale=1e-6/a
        assert rows[0]['time']==0 and rows[-1]['time']==50000
        assert all(r['xyz']==[2016,32,32] for r in rows)
        axes[0].plot(times,c*scale,'o-',ms=3,label=label)
        axes[1].plot(times,(c-c[0])*scale**2,'o-',ms=3,label=label)
        summary['cases'][case]={'job':job,'time_final':float(times[-1]),'initial_C1':float(c[0]),
            'final_C1':float(c[-1]),'final_increment_C1':float(c[-1]-c[0]),
            'retained_initial_fraction_of_final':float(c[0]/c[-1]),
            'final_terms':rows[-1]['C1_terms'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    large=records['theta_primary'];small=records['theta_amplitude'];a=.1
    increment=lambda r:r[-1]['C1_in_actual_basis']-r[0]['C1_in_actual_basis']
    pred=a*large[0]['C1_in_actual_basis']+a*a*increment(large)
    summary['amplitude_comparison']={'amplitude_ratio':a,'final_time_both':50000,
       'measured_final_C1_ratio':small[-1]['C1_in_actual_basis']/large[-1]['C1_in_actual_basis'],
       'measured_acquired_increment_ratio':increment(small)/increment(large),
       'initial_plus_quadratic_prediction_for_small_C1':pred,
       'measured_small_C1':small[-1]['C1_in_actual_basis'],
       'relative_prediction_error':(small[-1]['C1_in_actual_basis']-pred)/pred,
       'scope':'Two amplitudes at one resolution; arithmetic, projection and higher-order contributions not separated; not proof of a pure quadratic law.'}
    axes[0].axhline(large[0]['C1_in_actual_basis'],ls='--',color='.5',lw=.8)
    axes[0].set_ylabel(r'$C_1(10^{-6}/A)$')
    axes[1].set_ylabel(r'$[C_1-C_1(0)](10^{-6}/A)^2$')
    for ax in axes:
        ax.set(xlabel=r'$t/M$',xlim=(0,50000));ax.grid(alpha=.2)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[0].legend(frameon=False,fontsize=9)
    fig.savefig(root/'incoming-c1-gpu.png',dpi=170);fig.savefig(root/'incoming-c1-gpu.pdf')
    plt.close(fig)
    (root/'incoming-c1-gpu-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['amplitude_comparison'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    main(p.parse_args().directory)
