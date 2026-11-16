#!/usr/bin/env python3
"""One inspected batch per allocation; no automatic successor submission."""
import hashlib,json,os,pathlib,re,shutil,subprocess,sys,time
root,run=map(pathlib.Path,sys.argv[1:3]);case=sys.argv[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
receipt=json.loads((root/'build/receipt.json').read_text())
for name,h in receipt['executables'].items():assert sha(root/'build'/name)==h
mpi=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+os.environ['NONRAD_BINDING'],os.environ['NONRAD_WRAPPER']]
assert sha(root/'seed-v2/nonradiative_seed')==json.loads((root/'seed-v2/receipt.json').read_text())['executable_sha256']
report={'runner_sha256':sha(pathlib.Path(__file__)),'seed_receipt_sha256':sha(root/'seed-v2/receipt.json'),'build_receipt_sha256':sha(root/'build/receipt.json'),'commands':[],'cases':[]}
start=time.monotonic()
shutil.copy2(__file__,run/'runner.py')
if case=='pilot': names=['control-N12','gauge-N12']
elif case.startswith('pair-N'):
 n=case.split('-')[1];names=['control-'+n,'gauge-'+n]
 if n=='N12':names+=['gauge-N12-halfdt']
else:names=[case]
for name in names:
 d=run/name;d.mkdir();text=(root/'inputs'/(name+'.athinput')).read_text()
 if case=='pilot':text=text.replace('dt=.0025','dcycle=1\ndt=-1')
 inp=d/'input.athinput';inp.write_text(text)
 seed=d/'seed';seed.mkdir()
 amp='.001' if name.startswith('gauge') else '0'
 command=mpi+[str(root/'seed-v2/nonradiative_seed'),str(inp),amp]
 with (seed/'run.log').open('w') as log:subprocess.run(command,cwd=seed,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 paths=list((seed/'rst').glob('*.rst'));assert len(paths)==1
 initial=paths[0];command2=mpi+[str(root/'build/athena'),'-r',str(initial),'-i',str(inp),'-d',str(d),'-t','00:48:00']
 if case=='pilot':command2+=['time/nlim=100']
 before=time.monotonic()
 with (d/'run.log').open('w') as log:subprocess.run(command2,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3000)
 final=sorted((d/'rst').glob('*.rst'))[-1]
 report['commands'] += [command,command2]
 cycles=re.findall(r'^time=([^ ]+) cycle=(\d+)$',(d/'run.log').read_text(),re.M)
 assert len(cycles)==1,cycles
 expected=100 if case=='pilot' else int(re.search(r'^nlim\s*=\s*(\d+)',text,re.M).group(1))
 report['cases'].append(dict(name=name,seconds=time.monotonic()-before,input_sha256=sha(inp),seed=str(initial),seed_sha256=sha(initial),final=str(final),final_sha256=sha(final),time=float(cycles[0][0]),cycles=int(cycles[0][1]),expected_cycles=expected))
 report['wall_seconds']=time.monotonic()-start
 (run/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report['cases'][-1]),flush=True)
 assert int(cycles[0][1])==expected,'run stopped before planned cycle'

if case=='pilot' or case.startswith('pair-N'):
 python=root.parent/'n12-full-scri-20261115/analysis-env/bin/python'
 initial=[next((run/n/'seed/rst').glob('*.rst')) for n in names[:2]]
 subprocess.run([str(python),str(root/'source/tst/z4c_hyperboloidal/nonradiative_seed_check.py'),*[str(p) for p in initial],str(run/'seed-check.json')],check=True)
