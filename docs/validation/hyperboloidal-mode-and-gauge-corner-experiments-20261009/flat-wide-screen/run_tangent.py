"""Matched actual C_h J_h pure-gauge source, baseline versus wide flat height."""
from pathlib import Path
import hashlib,json,re,shlex,subprocess,time
root=Path(__file__).resolve().parents[3];w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(r['exit_status']==0 for r in json.loads((w/'local-build-receipt.json').read_text())['tests'])
assert json.loads((w/'oracle-wide.json').read_text())['checked_scalar_values']==1936
flags=(root/'build-layer-release/CMakeFiles/hyperboloidal_layer_constraint_tangent.dir/flags.make').read_text();makevar=lambda k:shlex.split(re.search(r'^'+k+r' = (.*)$',flags,re.M).group(1))
common=['/usr/bin/c++','-I'+str(w)]+makevar('CXX_DEFINES')+makevar('CXX_INCLUDES')+makevar('CXX_FLAGS')
libs=[str(root/f'build-layer-release/kokkos/{p}/src/libkokkos{p}.a') for p in ['containers','algorithms','core','simd']]
receipt={'scope':'Same actual Cartesian stencils/projector/diagnostics, native centered Jv at analytic reference, pure gauge pointwise lapse.1 shift.02 profile; no full matrix/global/native evolution',
 'parameters':{'S':1,'a':.5,'span':2.1,'r0':.05,'r1':.95,'degree':2,'symmetric':True,'kappa1':10,'kappa2':0,'KO':.1,'physicalP':True,'spatialnorm':True,'generator_eps':1e-4},'source_sha256':sha(w/'constraint_tangent.cpp'),'builds':[],'runs':[]}
for label in ['original','flat']:
 cmd=common.copy()
 if label=='flat':cmd.insert(1,'-I'+str(w/'overlay'))
 exe=w/f'tangent-{label}';source=w/'constraint_tangent.cpp';full=cmd+[str(source),'-o',str(exe)]+libs
 start=time.monotonic();p=subprocess.run(full,cwd=root,capture_output=True,text=True);(w/f'build-tangent-{label}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 build={'label':label,'command':full,'seconds':time.monotonic()-start,'exit_status':p.returncode,'executable_sha256':sha(exe),'link_archives':{k:sha(Path(k)) for k in libs}}
 dep=subprocess.run(cmd+[str(source),'-M','-MT','audit'],cwd=root,capture_output=True,text=True);assert dep.returncode==0
 (w/f'tangent-{label}.make').write_text(dep.stdout);paths=shlex.split(dep.stdout.replace('\\\n',' ').split(':',1)[1]);build['compiler_dependency_hashes']={str(Path(k).resolve()):sha(Path(k)) for k in paths};receipt['builds'].append(build)
 for n in [24,36]:
  for dt in [1e-6,1e-7]:
   command=[str(exe),'--native',str(n),'2.1','.5','1','2','1','.05','.95',str(dt)]
   start=time.monotonic();p=subprocess.run(command,cwd=root,capture_output=True,text=True);tag=f'{label}-N{n}-dt{dt}'
   (w/(tag+'.jsonl')).write_text(p.stdout);(w/(tag+'.stderr')).write_text(p.stderr);assert p.returncode==0,p.stderr
   rows=[json.loads(line) for line in p.stdout.splitlines()];r={'label':label,'n':n,'dt':dt,'command':command,'seconds':time.monotonic()-start,'exit_status':p.returncode,'rows':rows};receipt['runs'].append(r)
   (w/'tangent-results.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(label,n,dt,[(q['Hdot_rms'],q['Mdot_rms'],q['Zdot_rms']) for q in rows],flush=True)
by={(r['label'],r['n'],r['dt']):r for r in receipt['runs']};summary=[]
for n in [24,36]:
 for dt in [1e-6,1e-7]:
  avg=lambda label,key:sum(row[key] for row in by[label,n,dt]['rows'])/2
  keys=['Hdot_rms','Mdot_rms','Zdot_rms','Hdot_max','Mdot_max','Zdot_max']
  summary.append({'N':n,'dt':dt,'original':{k:avg('original',k) for k in keys},'flat':{k:avg('flat',k) for k in keys},'flat_over_original':{k:avg('flat',k)/avg('original',k) for k in keys}})
receipt['summary']=summary;receipt['sources_unchanged_after_runs']=sha(w/'constraint_tangent.cpp')==receipt['source_sha256'];assert receipt['sources_unchanged_after_runs']
(w/'tangent-results.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))
