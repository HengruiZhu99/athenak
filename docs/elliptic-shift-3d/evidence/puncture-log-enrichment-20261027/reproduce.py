#!/usr/bin/env python3
"""Compile/run the bounded host prototype and plot its complete local controls."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
HEADER=ROOT/'src/z4c/hyperboloidal/enriched_log_derivative.hpp'
TEST=ROOT/'tst/elliptic_shift_3d/test_enriched_log_derivative.cpp'
BUILD=ROOT/'build-shift-3d'
flags=[]
for line in (BUILD/'CMakeFiles/elliptic_shift_3d_puncture_position_tests.dir/flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
        flags.extend(shlex.split(line.split('=',1)[1]))
binary=BUILD/'enriched_log_derivative_prototype'
command=['/usr/bin/c++',*flags,str(TEST),'-o',str(binary),*map(str,(BUILD/'kokkos').glob('*/src/libkokkos*.a'))]
subprocess.run(command,cwd=ROOT,check=True)
usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
start=time.monotonic()
subprocess.run([str(binary),str(HERE/'results.json')],cwd=ROOT,check=True,timeout=30)
elapsed=time.monotonic()-start
usage1=resource.getrusage(resource.RUSAGE_CHILDREN)
data=json.loads((HERE/'results.json').read_text())
rows=data['cases']
summary=dict(data['summary'])
summary['rejections']=data['rejections']
summary['wall_seconds']=elapsed
summary['user_seconds']=usage1.ru_utime-usage0.ru_utime
summary['system_seconds']=usage1.ru_stime-usage0.ru_stime
summary['source_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
summary['prototype_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HEADER,TEST,Path(__file__))}
summary['compile_command']=shlex.join(command)
summary['run_command']=shlex.join([str(binary),str(HERE/'results.json')])
summary['scope']='Local Cartesian derivative tests only, no coupled evolution, no production integration; all64 accepted query locations retained'
summary['time_varying_power_control']={'law':'p(t)=2+0.4sin(t)','times':[.2,.8],'interpretation':'independent instantaneous spatial solves; no temporal regularity claim'}
summary['wormhole']=[]
for dx in [.04,.02,.01,.005]:
    selected=[r for r in rows if r['family']=='isotropic_wormhole_inner' and r['h']==dx]
    summary['wormhole'].append({'h':dx,'query_count':len(selected),'maximum_H_enriched_error':max(r['H_enriched_error'] for r in selected),'maximum_H_fd_error':max(r['H_fd_error'] for r in selected),'maximum_abs_H_exact':max(abs(r['H_exact']) for r in selected)})
summary['angular']=[]
for exponent in [.3,1.1,2.4]:
    selected=sorted((r for r in rows if r['family']=='angular_fractional' and r['s']==exponent),key=lambda r:-r['h'])
    orders=[math.log(a['H_enriched_error']/b['H_enriched_error'],2) for a,b in zip(selected,selected[1:])]
    summary['angular'].append({'s':exponent,'orders':orders,'fourth_order_gate_passed':all(p>=3.8 for p in orders),'cases':selected})
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for key,label,style in [('maximum_H_enriched_error','logρ enrichment','o-'),('maximum_H_fd_error','ordinary logFD','s--')]:
    axes[0].loglog([r['h'] for r in summary['wormhole']],[r[key] for r in summary['wormhole']],style,label=label)
axes[0].set_title('Isotropic inner wormhole: all 8 core queries')
axes[0].set_ylabel('maximum original physical |H − H_exact|')
for r in summary['angular']:
    axes[1].loglog([c['h'] for c in r['cases']],[c['H_enriched_error'] for c in r['cases']],'o-',label=f"angular power s={r['s']}")
axes[1].set_title('Fractional angular regularity remains limiting')
axes[1].set_ylabel('original physical |H − H_exact|')
for ax in axes:
    ax.set_xlabel('Cartesian spacing h');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(FixedLocator([.005,.01,.02,.04]))
    ax.xaxis.set_major_formatter(FixedFormatter(['.005','.01','.02','.04']))
    ax.xaxis.set_minor_formatter(NullFormatter())
fig.savefig(HERE/'hamiltonian-controls.png',dpi=160)
print(json.dumps({k:summary[k] for k in ['plans','polynomial_checks','maximum_scaled_derivative_error','maximum_condition','wall_seconds','user_seconds','wormhole']},indent=2))
