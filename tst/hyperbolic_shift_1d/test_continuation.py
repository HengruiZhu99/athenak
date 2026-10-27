"""Bitwise split/uninterrupted evolution and corrupted-checkpoint rejection."""
import argparse,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
runner=Path(__file__).with_name('run.py');cont=runner.with_name('continue_puncture.py')
base=[sys.executable,str(runner),'--library',str(a.library),'--n','129','--cfl','.05','--restoring','4','--shift-source','lapse-inner','--puncture','--wall-seconds','30']
for name,t in [('half',.0015625),('full',.003125)]:
 result=subprocess.run(base+['--t',str(t),'--output',str(a.output/name)],capture_output=True,text=True,check=True)
 assert json.loads((a.output/name/'manifest.json').read_text())['status']=='completed',result.stdout
cmd=[sys.executable,str(cont),'--parent',str(a.output/'half'),'--output',str(a.output/'split'),'--t','.003125']
subprocess.run(cmd,check=True,capture_output=True,text=True)
x=np.load(a.output/'split/profiles.npz');y=np.load(a.output/'full/profiles.npz')
assert np.array_equal(x['last'],y['last']) and np.array_equal(x['compensation'],y['compensation'])
# Each rejection happens before an output directory is created.
bad=a.output/'corrupt';shutil.copytree(a.output/'half',bad);f=bad/'history.json';f.write_text(f.read_text()+' ')
rejected=subprocess.run([sys.executable,str(cont),'--parent',str(bad),'--output',str(a.output/'must-not-exist'),'--t','.003125'],capture_output=True,text=True)
assert rejected.returncode and 'checkpoint hash' in rejected.stderr and not (a.output/'must-not-exist').exists()
unaligned=subprocess.run([sys.executable,str(cont),'--parent',str(a.output/'half'),'--output',str(a.output/'unaligned'),'--t','.0032'],capture_output=True,text=True)
assert unaligned.returncode and 'time lattice' in unaligned.stderr and not (a.output/'unaligned').exists()
shutil.copy2(a.output/'half/history.json',bad/'history.json')
f=bad/'source/tst/hyperbolic_shift_1d/puncture.py';f.write_text(f.read_text()+'\n# changed source\n')
source_bad=subprocess.run([sys.executable,str(cont),'--parent',str(bad),'--output',str(a.output/'bad-source'),'--t','.003125'],capture_output=True,text=True)
assert source_bad.returncode and 'Source hash mismatch' in source_bad.stderr and not (a.output/'bad-source').exists()
m=json.loads((bad/'manifest.json').read_text());m['status']='failed';(bad/'manifest.json').write_text(json.dumps(m))
failed=subprocess.run([sys.executable,str(cont),'--parent',str(bad),'--output',str(a.output/'failed-parent'),'--t','.003125'],capture_output=True,text=True)
assert failed.returncode and 'completed puncture parent' in failed.stderr and not (a.output/'failed-parent').exists()
subprocess.run(base+['--t','.0046875','--output',str(a.output/'full12')],capture_output=True,text=True,check=True)
subprocess.run([sys.executable,str(cont),'--parent',str(a.output/'split'),'--output',str(a.output/'split12'),'--t','.0046875'],capture_output=True,text=True,check=True)
x2=np.load(a.output/'split12/profiles.npz');y2=np.load(a.output/'full12/profiles.npz')
assert np.array_equal(x2['last'],y2['last']) and np.array_equal(x2['compensation'],y2['compensation'])
result=dict(second_generation_bitwise_agreement=True,changed_source_rejected=True,failed_parent_rejected=True,bitwise_state_agreement=True,bitwise_compensation_agreement=True,corrupt_history_rejected=True,unaligned_target_rejected=True,
 scope='Eight- and twelve-step live-metric puncture RK oracle; does not establish long-duration restart accuracy or convergence.')
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
