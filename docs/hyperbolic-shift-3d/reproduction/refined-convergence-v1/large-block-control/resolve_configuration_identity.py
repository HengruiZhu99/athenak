"""Derive exact configuration identity with the actual production compiler flags.

No MPI/Kokkos initialization, device allocation, physics step or smoke test.
Accept only the already measured local/Aurora roundoff forms; all other drift
fails the build before science/launcher freeze. Runtime identity remains exact.
"""
from pathlib import Path
import argparse,json,shlex,subprocess,hashlib

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(root,build,source_tst,launcher,flags_make_fallback=False):
 root=Path(root).resolve();build=Path(build).resolve();source_tst=Path(source_tst).resolve();launch=Path(launcher).resolve()
 local=json.loads((launch/'qualified-local-identity.json').read_text());calibration=json.loads((launch/'known-aurora-calibration.json').read_text())
 expected=local['identity']
 for change in calibration['changes']:
  assert expected.count(change['local'])==1
  expected=expected.replace(change['local'],change['aurora'])
 target=build/'tst/hyperbolic_shift_3d'
 if (build/'compile_commands.json').exists():
  records=json.loads((build/'compile_commands.json').read_text())
  matches=[x for x in records if x['file'].endswith('/amr_evolution.cpp') and 'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/' in x.get('command',' '.join(x.get('arguments',[])))]
  assert len(matches)==1;record=matches[0];command=shlex.split(record['command']) if 'command' in record else record['arguments'][:]
  recipe_origin='production compile_commands.json'
 else:
  assert flags_make_fallback,'Production compile_commands.json required'
  flags={}
  for line in (target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/flags.make').read_text().splitlines():
   if ' = ' in line:
    key,value=line.split(' = ',1);flags[key]=shlex.split(value)
  compiler=shlex.split((target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/link.txt').read_text())[0]
  record=dict(file=str(source_tst/'amr_evolution.cpp'),directory=str(target))
  command=[compiler]+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-c',record['file'],'-o','unused.o']
  recipe_origin='explicit local-only flags.make fallback; production build requires compile database'
 src=launch/'configuration_identity.cpp';obj=launch/'configuration_identity.o';binary=launch/'configuration-identity'
 assert not obj.exists() and not binary.exists() and not (launch/'qualified-identity.json').exists()
 assert command.count(record['file'])==1;command[command.index(record['file'])]=str(src)
 command[command.index('-o')+1]=str(obj);command+=['-I'+str(source_tst)]
 target=build/'tst/hyperbolic_shift_3d';link=shlex.split((target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/link.txt').read_text())
 old='CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o';assert link.count(old)==1
 link[link.index(old)]=str(obj);link[link.index('-o')+1]=str(binary)
 with (launch/'configuration-identity-build.log').open('w') as log:
  subprocess.run(command,cwd=record['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(link,cwd=target,stdout=log,stderr=subprocess.STDOUT,check=True)
 output=subprocess.check_output([str(binary),str(root/'candidate.cfg')],text=True)
 (launch/'configuration-identity.jsonl').write_text(output)
 rows=[json.loads(x) for x in output.splitlines()];assert [x['n'] for x in rows]==[24]
 identities={x['identity'] for x in rows};assert len(identities)==1
 identity=identities.pop();assert identity in (local['identity'],expected),'Unexplained identity drift: inspect compiler/configuration before any submission'
 assert all(x['inner_gamma']==x['k']*x['k'] for x in rows)
 result=dict(local,identity=identity,identity_origin='Host configuration computed with production compile/link flags; no GPU or evolution execution',calibration_source_sha256=digest(launch/'known-aurora-calibration.json'),configuration_utility_sha256=digest(binary),matched_form='local' if identity==local['identity'] else 'previously_measured_Aurora_roundoff',config_sha256=digest(root/'candidate.cfg'))
 (launch/'qualified-identity.json').write_text(json.dumps(result,indent=2)+'\n')
 (launch/'configuration-identity-build.json').write_text(json.dumps(dict(compile_recipe_origin=recipe_origin,compile=command,compile_cwd=record['directory'],link=link,link_cwd=str(target),source_sha256=digest(src),binary_sha256=digest(binary),config_sha256=digest(root/'candidate.cfg'),production_flags_sha256=digest(target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/flags.make')),indent=2)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--build');p.add_argument('--source-tst');p.add_argument('--launcher');p.add_argument('--flags-make-fallback',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
 r=resolve(root,a.build or root/'build',a.source_tst or root/'source/tst/hyperbolic_shift_3d',a.launcher or root/'launcher-boosted-transition-v1',a.flags_make_fallback)
 print(json.dumps(dict(matched_form=r['matched_form'],configuration_utility_sha256=r['configuration_utility_sha256'])))
