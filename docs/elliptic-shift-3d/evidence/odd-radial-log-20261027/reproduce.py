#!/usr/bin/env python3
"""Isolated, bounded odd-radial/log derivative falsifier; no pilot changes."""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BUILD=ROOT/'build-shift-3d'
TEST=ROOT/'tst/elliptic_shift_3d/test_odd_radial_log_derivative.cpp'
flags=[]
for line in (BUILD/'CMakeFiles/elliptic_shift_3d_puncture_position_tests.dir/flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
        flags.extend(shlex.split(line.split('=',1)[1]))
binary=BUILD/'odd_radial_log_derivative_prototype'
compile_command=['/usr/bin/c++',*flags,str(TEST),'-o',str(binary),*map(str,(BUILD/'kokkos').glob('*/src/libkokkos*.a'))]
subprocess.run(compile_command,cwd=ROOT,check=True)
start=time.monotonic();run_command=[str(binary),str(HERE/'results.json')]
result=subprocess.run(run_command,cwd=ROOT,capture_output=True,text=True,timeout=30)
wall=time.monotonic()-start
(HERE/'stderr.txt').write_text(result.stderr)
deps=[TEST,ROOT/'src/z4c/hyperboloidal/odd_radial_log_derivative.hpp',ROOT/'src/z4c/hyperboloidal/enriched_log_derivative.hpp',ROOT/'src/z4c/hyperboloidal/puncture_position.hpp',ROOT/'src/mesh/cubed_sphere/map.hpp',Path(__file__)]
manifest={'return_code':result.returncode,'wall_seconds':wall,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compile_command':shlex.join(compile_command),'run_command':shlex.join(run_command),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if result.returncode:
    print(result.stderr,end='');raise SystemExit(result.returncode)
data=json.loads((HERE/'results.json').read_text());summary=dict(data['summary'])
summary['rejections']=data['rejections'];summary['radius_conditioning']=data['radius_conditioning']
summary['wall_seconds']=wall
summary['wormhole']=[]
for dx in [.04,.02,.01,.005]:
    rows=[r for r in data['cases'] if r['family']=='isotropic_wormhole_inner' and r['h']==dx]
    summary['wormhole'].append({'h':dx,'query_count':len(rows),**{k:max(r[k] for r in rows) for k in ('H_enriched_error','H_log_only_error','H_fd_error')}})
summary['fractional']=[]
for family in ['angular_fractional','radial_fractional']:
    for exponent in [.3,1.1,2.4]:
        rows=sorted((r for r in data['cases'] if r['family']==family and r['s']==exponent),key=lambda r:-r['h'])
        orders=[math.log(a['H_enriched_error']/b['H_enriched_error'],2) for a,b in zip(rows,rows[1:])]
        summary['fractional'].append({'family':family,'s':exponent,'orders':orders,'fourth_order_gate_passed':all(p>=3.8 for p in orders),'cases':rows})
summary['qualification']='Isolated local host derivative prototype; not adopted into current pilot; actual wormhole improves but angular fractional errors worsen, no universal regularity/evolution claim'
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullFormatter
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for key,label,style in [('H_enriched_error','odd radial + log','o-'),('H_log_only_error','log only','s--'),('H_fd_error','ordinary logFD','^:')]:
    axes[0].loglog([r['h'] for r in summary['wormhole']],[r[key] for r in summary['wormhole']],style,label=label)
axes[0].set_title('Actual inner wormhole: all 8 core queries')
for column,family in [(1,'angular_fractional'),(2,'radial_fractional')]:
    for row in summary['fractional']:
        if row['family']!=family:continue
        line=axes[column].loglog([r['h'] for r in row['cases']],[r['H_enriched_error'] for r in row['cases']],'o-',label=f"odd basis s={row['s']}")[0]
        axes[column].loglog([r['h'] for r in row['cases']],[r['H_log_only_error'] for r in row['cases']],'--',color=line.get_color(),alpha=.6)
    axes[column].set_title(('Angular' if column==1 else 'Radial')+' fractional: dashed = log only')
for ax in axes:
    ax.set_xlabel('Cartesian spacing h');ax.set_ylabel('Original physical Hamiltonian error');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(FixedLocator([.005,.01,.02,.04]));ax.xaxis.set_major_formatter(FixedFormatter(['.005','.01','.02','.04']));ax.xaxis.set_minor_formatter(NullFormatter())
fig.savefig(HERE/'controls.png',dpi=160)
print(json.dumps({key:summary[key] for key in ['plans','conditioning_plans','exact_family_checks','maximum_scaled_derivative_error','maximum_condition','maximum_scaled_derivative_noise','baseline_scaled_derivative_noise','wall_seconds','wormhole']},indent=2))
