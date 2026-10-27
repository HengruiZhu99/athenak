#!/usr/bin/env python3
"""Bounded actual native selected wormhole execution and independent dispatch controls.
Scientific acceptance is recorded separately and is expected to remain false.
No source or existing job is modified. Each newly owned process is capped180s.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from analyze_native_composite import native_bin
from analyze_native_wormhole import physical_adm_check

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--minkowski-baseline',type=Path,default=Path('docs/elliptic-shift-3d/evidence/native-composite-cfl-controls-20261027'));p.add_argument('--skip-minkowski',action='store_true');a=p.parse_args()
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 binary=root/'build-shift-3d/src/athena';coordinate_binary=root/'build-shift-3d/elliptic_shift_3d_coupled_pilot';input=root/'inputs/z4c/composite_wormhole.athinput'
 files=[f for f in (root/'src').rglob('*') if f.is_file() and f.suffix in ['.cpp','.hpp','.h']]
 files += [input,root/'tst/elliptic_shift_3d/native_wormhole_reference.cpp',root/'tst/elliptic_shift_3d/wormhole_initial.py',root/'tst/elliptic_shift_1d/wormhole_isotropic.py',Path(__file__).resolve()]
 source_hashes={str(f.relative_to(root)):sha(f) for f in files}
 m={'scope':'Actual Serial native Driver selected isotropic M1/3 wormhole; one native Cartesian block, host-owned shells, no AMR/MPI/GPU/tracking claim','source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':source_hashes,'binary_sha256':sha(binary),'input_sha256':sha(input),'coordinate_binary_sha256':sha(coordinate_binary),'cases':[],'scientific_acceptance':False}
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 def run(name,command,success=True,folder=None):
  folder=folder or out/name;folder.mkdir(parents=True,exist_ok=False)
  row={'name':name,'argv':list(map(str,command)),'maximum_seconds':180,'expected_success':success};m['cases'].append(row);save();start=time.monotonic()
  with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
   process=subprocess.Popen(list(map(str,command)),cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['owned_process_group']=process.pid;save()
   try:row['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    row['exit_code']=124
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('refusing unowned process group')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    row['cleanup']='only this newly owned process group terminated'
  row['seconds']=time.monotonic()-start;row['expectation_met']=(row['exit_code']==0)==success;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True)
  if not row['expectation_met']:raise RuntimeError('Unexpected execution result: '+name)
  return folder
 coordinates=out/'coordinates.csv';initial=out/'initial.dat';radii='.45,.55,.60,.65,.70,.75,1'
 run('coordinates',[coordinate_binary,'--coordinates','6',radii,'1','0',coordinates])
 run('independent-initializer',[sys.executable,root/'tst/elliptic_shift_3d/wormhole_initial.py',coordinates,initial])
 m['initializer']=json.loads(initial.with_suffix('.json').read_text());m['initializer']['file_sha256']=sha(initial);save()
 # Compile a local same-engine control; its RK arrays come from the actual
 # native Driver receipt rather than duplicated classical-RK coefficients.
 flags=[]
 for line in (root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_native_fields_tests.dir/flags.make').read_text().splitlines():
  if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags.extend(shlex.split(line.split('=',1)[1]))
 reference=out/'native_wormhole_reference';command=['/usr/bin/c++',*flags,str(root/'tst/elliptic_shift_3d/native_wormhole_reference.cpp'),'-o',str(reference),*map(str,(root/'build-shift-3d/kokkos').glob('*/src/libkokkos*.a'))]
 run('compile-matched-reference',command)
 full=out/'full';run('full',[binary,'-i',input,'z4c/composite_initial_file='+str(initial),'-d',full],folder=full)
 trace=[json.loads(line) for line in (full/'native_wormhole.jsonl').read_text().splitlines()]
 assert [r['cycle'] for r in trace]==[0,1,2];m['native_trace']=trace
 m['execution_qualification']={'actual_Driver_steps':2,'actual_RK_stages_per_cycle':4,'native_updates':trace[-1]['native_update_calls'],'time':trace[-1]['time']};save()
 # The paired sidecar is complete: hide the initializer before restarting.
 hidden=out/'initial.hidden-after-fresh-run.dat';initial.rename(hidden);m['initializer']['hidden_before_restart']=True;save()
 midpoint=full/'rst/native_wormhole.00001.rst';assert midpoint.exists();restart=out/'restart'
 run('restart-without-initializer',[binary,'-r',midpoint,'-d',restart],folder=restart)
 checks={}
 for suffix in ['.csv','.scri.csv']:
  x=full/('native_wormhole.00002'+suffix);y=restart/x.name;checks[suffix]={'bitwise_identical':x.read_bytes()==y.read_bytes(),'full_sha256':sha(x),'restart_sha256':sha(y)}
 m['accepted_restart']=checks;save();assert all(v['bitwise_identical'] for v in checks.values())
 run('reject-receipt-change',[binary,'-r',midpoint,'z4c/composite_initial_receipt=wrong-receipt','-d',out/'reject-receipt-change'],False)
 run('reject-taper-change',[binary,'-r',midpoint,'z4c/composite_telegraph_width=.08','-d',out/'reject-taper-change'],False)
 run('reject-chart-change',[binary,'-r',midpoint,'z4c/composite_positive_storage=1','-d',out/'reject-chart-change'],False)
 run('reject-minkowski-reinterpretation',[binary,'-r',midpoint,'problem/pgen_name=z4c_composite_minkowski','problem/mass=0','-d',out/'reject-minkowski-reinterpretation'],False)
 run('reject-minkowski-valid-geometry',[binary,'-r',midpoint,'problem/pgen_name=z4c_composite_minkowski','problem/mass=0','z4c/composite_radii=.65,1','-d',out/'reject-minkowski-valid-geometry'],False)
 dts=out/'actual-dt.dat';dts.write_text(''.join(format(trace[i+1]['time']-trace[i]['time'],'.17g')+'\n' for i in range(2)))
 run('matched-native-RK-reference',[reference,hidden,full/'native_wormhole.rk.dat',dts,out/'reference'])
 comparison={}
 for cycle in range(3):
  actual=np.genfromtxt(full/f'native_wormhole.{cycle:05d}.csv',delimiter=',',names=True)
  expected=np.genfromtxt(out/f'reference.{cycle:05d}.csv',delimiter=',',names=True)
  fields=[f'state_{f}' for f in range(22)]+[f'delta_beta_{i}' for i in range(3)]
  errors={field:float(np.max(abs(actual[field]-expected[field]))) for field in fields}
  comparison[str(cycle)]={'cells':len(actual),'maximum_coefficient_error':max(errors.values()),'field_errors':errors,'bitwise_all_fields':all(np.array_equal(actual[field],expected[field]) for field in fields)}
 m['matched_actual_RK_comparison']=comparison;save()
 # Physical native ADM is independent of the chart, including every core ghost.
 zbin=max((full/'bin').glob('*.z4c.*.bin'));number=zbin.name.rsplit('.',2)[1];q,xyz,_=native_bin(zbin);adm,_,_=native_bin(full/'bin'/('native_wormhole.adm.'+number+'.bin'))
 omega=(1-sum(x*x for x in xyz))/2;psi4=1/(omega**2*q['z4c_chi']);expected={'adm_psi4':psi4,'adm_alpha':q['z4c_alpha']/omega}
 for component in ['xx','xy','xz','yy','yz','zz']:
  expected['adm_g'+component]=psi4*q['z4c_g'+component]
  expected['adm_K'+component]=psi4*(omega*q['z4c_A'+component]+q['z4c_g'+component]*(q['z4c_Khat']+2*q['z4c_Theta'])/3)
 for c in ['x','y','z']:expected['adm_beta'+c]=q['z4c_beta'+c]
 errors={name:float(np.max(abs(adm[name]-value)/np.maximum(1,abs(value)))) for name,value in expected.items()}
 m['physical_ADM_all_native_core_and_ghosts']={'relative_errors':errors,'float32_output_reference_tolerance':3e-7,'passed':max(errors.values())<3e-7,'qualification':'Rounded-input proxy retained; fullprecision nativeRST is the authoritative independent conversion control.'};save()
 m['physical_ADM_fullprecision_RST_control']=physical_adm_check(full);save();assert m['physical_ADM_fullprecision_RST_control']['passed']
 m['scientific_gates']={'acceptance':False,'reason':'One coarse native execution/control cannot establish radial evolution/convergence/collapse milestone; retain raw physical residuals and unchanged boundary reference targets.','tensor_pole_reference_target':1e-7,'tensor_pole_passed':trace[-1]['scri_tensor_pole']<=1e-7,'clock_reference_target':1e-13,'null_reference_target':1e-12,'final_native_diagnostics':trace[-1],'minimum_sample_radius':m['initializer']['minimum_cell_radius'],'isotropic_throat_radius':m['initializer']['throat_radius'],'throat_unresolved':m['initializer']['minimum_cell_radius']>m['initializer']['throat_radius']};save()
 if not a.skip_minkowski:
  control=out/'minkowski-controls';run('minkowski-regression',[sys.executable,root/'tst/elliptic_shift_3d/run_native_composite_smoke.py','--output',control,'--extended-controls'],folder=out/'minkowski-runner')
  baseline=a.minkowski_baseline.resolve();identical={}
  for name in ['two-shell','one-shell','radial-pulse','angular-pulse','extended-minkowski']:
   for old in sorted((baseline/name).glob('*.csv')):
    current=control/name/old.name;identical[str(old.relative_to(baseline))]={'bitwise_identical':current.exists() and old.read_bytes()==current.read_bytes(),'baseline_sha256':sha(old),'current_sha256':sha(current) if current.exists() else None}
  m['default_Minkowski_bitwise_comparison']={'files':identical,'compared':len(identical),'all_identical':len(identical)==34 and all(v['bitwise_identical'] for v in identical.values())};save();assert m['default_Minkowski_bitwise_comparison']['all_identical']
 m['source_changes_during_controls']=[name for name,value in source_hashes.items() if sha(root/name)!=value]
 m['binary_unchanged']=sha(binary)==m['binary_sha256'];m['input_unchanged']=sha(input)==m['input_sha256'];save()
 print(json.dumps({'execution_qualification':m['execution_qualification'],'restart':checks,'same_engine_control':comparison,'scientific_acceptance':False,'binary_unchanged':m['binary_unchanged']}),flush=True)

if __name__=='__main__':main()
