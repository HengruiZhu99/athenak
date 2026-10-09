"""Freeze and compile actual Cartesian ghost-projection gate executables."""
from pathlib import Path
import hashlib,json,re,shlex,subprocess,time
root=Path(__file__).resolve().parents[3];work=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
flags=(root/'build-layer-release/CMakeFiles/hyperboloidal_layer_constraint_tangent.dir/flags.make').read_text()
part=lambda name:shlex.split(re.search(r'^'+name+r' = (.*)$',flags,re.M).group(1))
common=['/usr/bin/c++','-I'+str(work/'overlay')]+part('CXX_DEFINES')+part('CXX_INCLUDES')+part('CXX_FLAGS')
libs=[str(root/f'build-layer-release/kokkos/{p}/src/libkokkos{p}.a') for p in ['containers','algorithms','core','simd']]
receipt={'compiler_identity':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_sha256':{str(p.relative_to(root)):sha(p) for p in [root/'tst/hyperboloidal/test_layer_constraint_tangent.cpp',root/'src/utils/finite_diff.hpp']+sorted((root/'src/z4c/hyperboloidal').glob('*.hpp'))},'overlay_manifest':json.loads((work/'overlay-manifest.json').read_text()),'builds':[],'cases':[]}
for name,source in [('tangent-projected',root/'tst/hyperboloidal/test_layer_constraint_tangent.cpp'),('quantify-projected',work/'quantify.cpp'),('reference-checks',work/'reference_checks.cpp')]:
 command=common+[str(source),'-o',str(work/name)]+libs;start=time.monotonic();done=subprocess.run(command,cwd=root,capture_output=True,text=True)
 (work/f'build-{name}.log').write_text(done.stdout+done.stderr)
 if done.returncode:raise RuntimeError(done.stderr)
 receipt['builds'].append({'name':name,'command':command,'source_sha256':sha(source),'executable_sha256':sha(work/name),'wall_seconds':time.monotonic()-start})
 print('Built',name,flush=True)
command=[str(work/'reference-checks')];done=subprocess.run(command,capture_output=True,text=True);(work/'reference-checks.jsonl').write_text(done.stdout);(work/'reference-checks.stderr').write_text(done.stderr)
if done.returncode:raise RuntimeError(done.stderr)
receipt['cases'].append({'name':'reference-checks','command':command,'exit_status':done.returncode,'measurements':[json.loads(l) for l in done.stdout.splitlines()]})
for n in [24,36,48]:
 command=[str(work/'tangent-projected'),'--native',str(n),'2.1','.5','1','2','1','.05','.95','1e-6'];start=time.monotonic();done=subprocess.run(command,capture_output=True,text=True)
 (work/f'tangent-N{n}.jsonl').write_text(done.stdout);(work/f'tangent-N{n}.stderr').write_text(done.stderr)
 if done.returncode:raise RuntimeError(done.stderr)
 receipt['cases'].append({'name':f'tangent-N{n}','command':command,'exit_status':done.returncode,'wall_seconds':time.monotonic()-start,'measurements':[json.loads(l) for l in done.stdout.splitlines()]})
 print('Tangent',n,'PASS',flush=True)
projected=[]
for case in json.loads((work/'snapshot-extraction.json').read_text()):
 prefix=work/f'projected-degree{case["degree"]}-time{case["time"]}';command=[str(work/'quantify-projected'),str(case['degree']),case['output'],str(prefix)]
 done=subprocess.run(command,capture_output=True,text=True);Path(str(prefix)+'.jsonl').write_text(done.stdout);Path(str(prefix)+'.stderr').write_text(done.stderr)
 if done.returncode:raise RuntimeError(done.stderr)
 row=json.loads(done.stdout);row['time']=case['time'];projected.append(row)
receipt['projected_snapshot_measurements']=projected
(work/'audit-results.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt['cases'],indent=2),flush=True)
