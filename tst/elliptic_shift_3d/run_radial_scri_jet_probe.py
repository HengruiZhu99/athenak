#!/usr/bin/env python3
"""Bounded frozen normal-jet diagnostic; no shared build/evolution mutation."""
import argparse,hashlib,json,shlex,subprocess,tarfile,time,os,signal,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--matrix',choices=['first','all'],default='first');a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));r=Path.cwd();b=r/'build-shift-3d';o=a.output.resolve();o.mkdir(parents=True,exist_ok=False);frozen=o/'source';frozen.mkdir();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();archive=r/'docs/elliptic-shift-3d/evidence/hermite-sampler-final-20261027/compiled-source.tar.gz'
with tarfile.open(archive) as t:
 for member in t.getmembers():
  target=(frozen/member.name).resolve()
  if not target.is_relative_to(frozen) or not member.isfile():
   if member.isdir():continue
   raise RuntimeError('Unexpected archive path/type')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(member).read())
files=[*r.glob('src/mesh/cubed_sphere/*.hpp'),*r.glob('src/z4c/hyperboloidal/*.hpp')]
files += [r/x for x in ['src/athena.hpp','build-shift-3d/config.hpp','LICENSE','tst/elliptic_shift_3d/diagnose_scri_pole.cpp','tst/elliptic_shift_3d/test_radial_scri_jet_probe.cpp','tst/elliptic_shift_3d/diagnose_radial_scri_jet.cpp','tst/elliptic_shift_3d/test_radial_scri_jet_contract.cpp']]
for source in files:
 target=frozen/source.relative_to(r);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
flags={};flag=b/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make'
for line in flag.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v)
flags['CXX_INCLUDES']=[x for x in flags['CXX_INCLUDES'] if x!='-I'+str(r/'src')];link=shlex.split((flag.parent/'link.txt').read_text());libs=[(b/x).resolve() for x in link if x.endswith('.a')];commands={}
for target,source in [('mms','test_radial_scri_jet_probe.cpp'),('pole','diagnose_radial_scri_jet.cpp'),('contract','test_radial_scri_jet_contract.cpp')]:
 commands[target+'-build']=[link[0],*flags['CXX_DEFINES'],'-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(frozen/'tst/elliptic_shift_3d'/source),'-o',str(o/target),*map(str,libs)]
 commands[target+'-deps']=[link[0],*flags['CXX_DEFINES'],'-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-M',str(frozen/'tst/elliptic_shift_3d'/source)]
commands['contract']=[str(o/'contract')]
cases=[(7,1,[6])] if a.matrix=='first' else [(1,1,[6,12,18]),(2,1,[6,10,14]),(7,1,[6,8,10]),(7,2,[6])]
for layers,sub,ns in cases:
 for n in ns:
  for radial in [6,7]:commands[f'mms-{layers}-{sub}-{n}-{radial}-log']=[str(o/'mms'),str(n),str(layers),str(sub),str(radial),str(radial),'2']
for nodes,basis in [(7,0),(7,1),(7,2),(8,0),(8,2),(9,0),(9,2),(10,0),(11,0)]:
 for state,checkpoint,tag in [('old',r/'docs/elliptic-shift-3d/evidence/amr-log-halo-coupled-20261027/coupled.bin','b79a95d2b4452b4ec64e32be753c54f78a4551b115f46ff4a133fb7bdbdb69f1'),('v16',r/'docs/elliptic-shift-3d/evidence/hermite-supported-coupled-20261027/coupled.checkpoint','ef2196506b829ccabb6d072909db40e8dd8891f394a4baa1429eced0858fc58b')]:
  commands[f'pole-{state}-{nodes}-{basis}']=[str(o/'pole'),str(checkpoint),tag,str(r/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat'),str(o/f'pole-{state}-{nodes}-{basis}.csv'),str(nodes),str(basis),'0']
for nodes in [9,13,15]:
 for basis in [0,2]:
  cmd=commands['pole-v16-7-0'].copy();cmd[4]=str(o/f'pole-v16-LS-{nodes}-{basis}.csv');cmd[5:]=[str(nodes),str(basis),'6'];commands[f'pole-v16-LS-{nodes}-{basis}']=cmd
for layers,sub,ns in [(2,1,[6,10,14]),(7,1,[6,8,10]),(7,2,[6])]:
 for n in ns:commands[f'mms-LS-{layers}-{sub}-{n}-13-log']=[str(o/'mms'),str(n),str(layers),str(sub),'7','13','2','6']
prior=json.loads((r/'docs/elliptic-shift-3d/evidence/hermite-sampler-final-20261027/manifest.json').read_text());m=dict(scope='Frozen host radial-jet alternatives, no evolution/solve or engine adoption',guard_limits=dict(condition=1e8,polynomial_reproduction=1e-10,derivative_amplification_ratio=4),inputs=[dict(path=str(f),sha256=sha(f)) for f in [archive,*files,Path(__file__).resolve()]],runtime_hashes_before={f:sha(Path(f)) for f in prior['runtime_hashes_before']},runs={})
for name,cmd in commands.items():
 start=time.monotonic()
 with (o/(name+'.stdout')).open('w') as so,(o/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   code=124
   if os.getpgid(child.pid)==child.pid:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
 m['runs'][name]=dict(command=cmd,exit_code=code,seconds=time.monotonic()-start);(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(name,code,m['runs'][name]['seconds'],flush=True)
 if code and name.endswith(('build','deps')):raise SystemExit(code)
m['inputs_unchanged']=all(sha(Path(x['path']))==x['sha256'] for x in m['inputs']);m['runtime_unchanged']=all(sha(Path(f))==h for f,h in m['runtime_hashes_before'].items());(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
with tarfile.open(o/'compiled-source.tar.gz','w:gz') as t:
 for f in sorted(frozen.rglob('*')):
  if f.is_file():t.add(f,arcname=str(f.relative_to(frozen)))
(o/'archive.json').write_text(json.dumps(dict(path='compiled-source.tar.gz',sha256=sha(o/'compiled-source.tar.gz')),indent=2)+'\n')
