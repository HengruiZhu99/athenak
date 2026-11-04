"""Initial continuum gauge attribution and derivative-step control."""
from pathlib import Path
import gzip,json,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
RAW=ROOT/'output/refined-convergence/initial-shear-killing-probe';OUT=HERE/'initial-gauge-shear';OUT.mkdir(exist_ok=True)
labels={0:'Current free shift/lapse',1:'Native outer shift; original lapse',2:'Native shift everywhere',3:'Stationary outer lapse and shift'}
report={};datasets={}
for case,end in [('central6',.65),('central20',.78)]:
 p=RAW/(case+'.csv');x=np.genfromtxt(p,delimiter=',',names=True);datasets[case]=x
 with gzip.open(OUT/(case+'.csv.gz'),'wb') as f:f.write(p.read_bytes())
 report[case]=dict(source_csv_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),outer_cut=end,modes={},
                  metric_equation_identity_max_error=float(np.max(abs(x['metric_rr_rate']-x['metric_rr_algebraic']))))
 for mode in range(4):
  m=x[(x['mode']==mode)&(x['r']>end)];fine=m[m['h']==.000025];i=np.argmax(abs(fine['metric_rr_rate']))
  step_checks=[]
  for step in [.0001,.00005,.000025]:
   z=m[m['h']==step];step_checks.append(dict(h=step,metric_rr_rate_max=float(max(abs(z['metric_rr_rate']))),chi_rate_max=float(max(abs(z['chi_rate'])))))
  report[case]['modes'][mode]=dict(label=labels[mode],max_metric_rate=float(abs(fine['metric_rr_rate'][i])),max_metric_rate_r=float(fine['r'][i]),
    max_chi_rate=float(max(abs(fine['chi_rate']))),max_shift_rate=float(max(abs(fine['shift_rate']))),max_lapse_rate=float(max(abs(fine['lapse_rate']))),
    max_restoring_source=float(max(abs(fine['restoring_source']))),derivative_step_controls=step_checks)
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
x=datasets['central20'];colors={0:'C0',1:'C1',3:'C2'}
for mode in [0,1,3]:
 a=x[(x['mode']==mode)&(x['h']==.000025)]
 for ax,key in [(axs[0,0],'alpha'),(axs[0,1],'beta_dev_r'),(axs[1,0],'metric_rr_rate'),(axs[1,1],'shift_rate')]:
  ax.plot(a['r'],a[key],color=colors[mode],label=labels[mode])
for ax,label in zip(axs.flat,['Initial conformal lapse','Initial (beta + x) radial','Initial d(metric rr)/dt','Original gauge d(beta radial)/dt']):
 ax.set(xlabel='Compact radius r',ylabel=label,xlim=(.6,.95));ax.grid(alpha=.2);ax.axvline(.78,color='gray',ls=':')
axs[0,0].legend(fontsize=7)
fig.suptitle('Large-center initial gauge mismatch: correcting shear alone is insufficient\n'
             'Same initial spatial geometry/K; pointwise equation audit, no evolution')
fig.savefig(OUT/'initial-gauge-shear.png',dpi=170);fig.savefig(OUT/'initial-gauge-shear.pdf');plt.close(fig)
report['qualification']='Continuum pointwise metric/chi rate attribution, using four-point centered first derivatives at three step sizes. No second-derivative evolution rates, no long evolution or stability claim. Geometry/initial constraints unchanged across gauge choices. Current production gauge sources are kept when evaluating all alternatives; none is yet an implemented production repair.'
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(RAW/'build.json',OUT/'build.json')
print(json.dumps(report,indent=2))
