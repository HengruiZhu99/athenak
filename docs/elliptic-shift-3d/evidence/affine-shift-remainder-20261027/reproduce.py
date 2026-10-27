#!/usr/bin/env python3
"""Bounded local affine/remainder arithmetic experiment, no elliptic solve."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BUILD=ROOT/'build-shift-3d'
TEST=ROOT/'tst/elliptic_shift_3d/test_affine_shift_remainder.cpp'
flags=[]
for line in (BUILD/'CMakeFiles/elliptic_shift_3d_puncture_position_tests.dir/flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
        flags.extend(shlex.split(line.split('=',1)[1]))
binary=BUILD/'affine_shift_remainder_prototype'
compile_command=['/usr/bin/c++',*flags,str(TEST),'-o',str(binary),*map(str,(BUILD/'kokkos').glob('*/src/libkokkos*.a'))]
subprocess.run(compile_command,cwd=ROOT,check=True)
start=time.monotonic();run_command=[str(binary),str(HERE/'results.json')]
result=subprocess.run(run_command,cwd=ROOT,capture_output=True,text=True,timeout=30)
wall=time.monotonic()-start
(HERE/'stderr.txt').write_text(result.stderr)
deps=[TEST,ROOT/'tst/elliptic_shift_3d/manufactured.hpp',ROOT/'src/z4c/hyperboloidal/affine_shift_remainder.hpp',ROOT/'src/z4c/hyperboloidal/elliptic_shift.hpp',ROOT/'src/z4c/hyperboloidal/conformal_rhs.hpp',ROOT/'src/z4c/hyperboloidal/conformal_constraints.hpp',ROOT/'src/z4c/hyperboloidal/puncture_position.hpp',ROOT/'src/mesh/cubed_sphere/map.hpp',Path(__file__)]
manifest={'return_code':result.returncode,'wall_seconds':wall,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compile_command':shlex.join(compile_command),'run_command':shlex.join(run_command),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if result.returncode:
    print(result.stderr,end='');raise SystemExit(result.returncode)
data=json.loads((HERE/'results.json').read_text());summary=dict(data['summary']);summary['wall_seconds']=wall
summary['representative_cases']=[r for r in data['cases'] if r['translated'] and r['curved'] and r['family']>=2 and any(abs(r['h']/x-1)<1e-12 for x in [1e-2,1e-5,1e-7,1e-9])]
summary['qualification']='Isolated local representation/arithmetic experiment. Same full native vector operator for all controls. No augmented elliptic solve, full mesh, field migration, warmstart/restart or evolution validation.'
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for row,nu in enumerate([.5,1.09]):
    for column,near in enumerate([True,False]):
        cases=sorted((r for r in data['cases'] if r['translated'] and r['curved'] and bool(r['near'])==near and r['nu']==nu),key=lambda r:r['h'])
        ax=axes[row,column]
        for key,label,style in [('direct_original_residual','physical nodal β','o-'),('separate_original_residual','separate affine + u','s-'),('nodal_recovery_original_residual','decompose nodal β later','^--'),('truncation_coordinate_control','truncation / coordinate control',':')]:
            ax.loglog([r['h'] for r in cases],[max(r[key],1e-18) for r in cases],style,label=label)
        ax.set_title(f"ν={nu}, query={'near' if near else '8.5h offset'}, translated, nonflat")
        ax.set_xlabel('local spacing h');ax.set_ylabel('max component original vector residual')
        ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=7)
fig.savefig(HERE/'original-vector-residuals.png',dpi=160)
print(json.dumps({key:value for key,value in summary.items() if key!='representative_cases'},indent=2))
