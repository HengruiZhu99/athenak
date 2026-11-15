from pathlib import Path
import argparse,importlib.util,json,subprocess,os
ROOT=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111');CASE=ROOT/'full-overlap-30M-20261114/final-comparison'
spec=importlib.util.spec_from_file_location('previous_driver',ROOT/'hybrid_science_segment_v4.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);header=old.header;sha=old.sha
p=argparse.ArgumentParser();p.add_argument('--wrapper',required=True);p.add_argument('--binding',required=True);a=p.parse_args()
selected=[
 ('N12-full-final',ROOT/'science/N12-full-overlap-30M-20261114/segment-9089673/rst/headon.00030.rst'),
 ('N12-legacy-final',ROOT.parent/'headon-6M-20261109/runs/headon30-9056684/rst/headon_6M.00030.rst'),
 ('N16-full-final',ROOT/'science/N16-full-overlap-30M-20261114/segment-9091708/rst/headon.00031.rst'),
 ('N16-legacy-final',ROOT.parent/'headon-6M-N16-20261110/runs/headon-9074954/rst/headon_N16_6M.00035.rst')]
exe=ROOT/'checkpoint-history-full-N12-20261114/history_output';reducer=ROOT/'science-analysis/hybrid_checkpoint_audit.py'
assert sha(exe)=='e4dd9a4ffd5fe5bd3917027267f1672e3c5e6ef765f2673fb1ea31255b5dd71d'
run=CASE/('audit-'+os.environ['PBS_JOBID'].split('.')[0]);run.mkdir(exist_ok=False);results=[]
for name,rst in selected:
 folder=run/name;folder.mkdir();h=header(rst);assert 29.99<h['time_M0']<30.01
 before=sha(rst);cmd=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+a.binding,a.wrapper,str(exe),str(rst),'checkpoint','.25','.18','.195']
 provenance={'command':cmd,'checkpoint':str(rst),'checkpoint_header':h,'checkpoint_sha256':before,'executable_sha256':sha(exe),'driver_sha256':sha(Path(__file__)),'reducer_sha256':sha(reducer)}
 (folder/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 with (folder/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 r.check_returncode();assert 'PASS checkpoint diagnostics; zero evolution steps; accepted fields/records bitwise preserved' in (folder/'run.log').read_text();assert sha(rst)==before
 with (folder/'audit.log').open('w') as log:
  subprocess.run(['/usr/bin/python3',str(reducer),'--checkpoint',str(rst),'--history-dir',str(folder),'--output',str(folder/'independent')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=400)
 results.append({'name':name,'checkpoint':str(rst),'time_M0':h['time_M0'],'native_read_only_pass':True,'checkpoint_hash_unchanged':True,'independent_reducer_pass':True})
 (run/'results.json').write_text(json.dumps(results,indent=2)+'\n');print('PASS '+name,flush=True)
