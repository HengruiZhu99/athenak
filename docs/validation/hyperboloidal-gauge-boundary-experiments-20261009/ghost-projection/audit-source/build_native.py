"""Recompile every native TU depending on private CartesianPatch overlay; reuse frozen other objects."""
from pathlib import Path
import hashlib,json,re,shlex,subprocess,time,shutil,struct
root=Path(__file__).resolve().parents[3];work=Path(__file__).resolve().parent;build=root/'build-layer-release/src';target=work/'native-build';target.mkdir(exist_ok=True);(target/'baseline').mkdir(exist_ok=True);(target/'projected').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
flags=(build/'CMakeFiles/athena.dir/flags.make').read_text();part=lambda name:shlex.split(re.search(r'^'+name+r' = (.*)$',flags,re.M).group(1))
common=['/usr/bin/c++','-I'+str(work/'overlay')]+part('CXX_DEFINES')+part('CXX_INCLUDES')+part('CXX_FLAGS')
link=shlex.split((build/'CMakeFiles/athena.dir/link.txt').read_text());old_output=link[link.index('-o')+1]
manifest={'compiler_identity':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'compiler_sha256':sha(Path('/usr/bin/c++')),'compiler_flags':flags,'source_at_build':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'runtime_source_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2','overlay':json.loads((work/'overlay-manifest.json').read_text()),'original_executable_sha256':sha(build/'athena'),'object_inputs':{},'library_inputs':{},'builds':[]}
for t in link:
 p=(build/t).resolve()
 if t.endswith('.o'):manifest['object_inputs'][str(p)]=sha(p)
 if t.endswith('.a'):manifest['library_inputs'][str(p)]=sha(p)
(target/'link-original.txt').write_text((build/'CMakeFiles/athena.dir/link.txt').read_text());(target/'flags-original.make').write_text(flags)
# A private baseline relink proves exactly which original object set we inherit.
baseline=list(link);baseline[baseline.index('-o')+1]=str(target/'baseline/athena');subprocess.run(baseline,cwd=build,check=True,capture_output=True,text=True)
manifest['baseline_relink_command']=baseline;manifest['baseline_relinked_sha256']=sha(target/'baseline/athena');manifest['baseline_byte_identical']=manifest['baseline_relinked_sha256']==manifest['original_executable_sha256']
print('Baseline relink identical:',manifest['baseline_byte_identical'],flush=True)
def normalize_macho(contents):
 result=bytearray(contents);offset=32;ranges=[]
 for _ in range(struct.unpack_from('<I',contents,16)[0]):
  command,length=struct.unpack_from('<II',contents,offset)
  if command==0x1b:ranges.append({'field':'LC_UUID','offset':offset+8,'size':16})
  if command==0x1d:
   start,size=struct.unpack_from('<II',contents,offset+8);ranges.append({'field':'LC_CODE_SIGNATURE payload','offset':start,'size':size})
  offset+=length
 for entry in ranges:result[entry['offset']:entry['offset']+entry['size']]=b'\x00'*entry['size']
 return bytes(result),ranges
original=(build/'athena').read_bytes();relinked=(target/'baseline/athena').read_bytes();original_normal,original_ranges=normalize_macho(original);relinked_normal,relinked_ranges=normalize_macho(relinked)
proof={'scope':'Baseline relink uses exact original object/library bytes. The only binary differences are Mach-O LC_UUID and linker ad-hoc code signature; all remaining bytes compare exactly. No code/data/symbol differences.','full_byte_difference_count':sum(a!=b for a,b in zip(original,relinked)),'full_byte_difference_offsets':[i for i,(a,b) in enumerate(zip(original,relinked)) if a!=b],'original_ignored_ranges':original_ranges,'baseline_ignored_ranges':relinked_ranges,'equal_after_only_uuid_signature_normalization':original_normal==relinked_normal,'normalized_original_sha256':hashlib.sha256(original_normal).hexdigest(),'normalized_baseline_sha256':hashlib.sha256(relinked_normal).hexdigest()}
assert original_normal==relinked_normal and original_ranges==relinked_ranges
manifest['baseline_equivalence']=proof;(target/'baseline-relink-equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
replace={};sources=[]
for dep in sorted((build/'CMakeFiles/athena.dir').rglob('*.o.d')):
 if 'cartesian_patch.hpp' not in dep.read_text():continue
 rel=str(dep.relative_to(build/'CMakeFiles/athena.dir'))[:-4];source=root/'src'/rel;sources.append(source)
 obj=target/'objects'/Path(rel+'.o');obj.parent.mkdir(parents=True,exist_ok=True);depout=Path(str(obj)+'.d')
 saved=target/'source'/source.relative_to(root);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,saved)
 before=sha(source);runtime_bytes=subprocess.check_output(['git','show','27c19d20696ea6dd4704032c51dfd026218f64f2:'+str(source.relative_to(root))],cwd=root);matches=hashlib.sha256(runtime_bytes).hexdigest()==before
 if not matches:raise RuntimeError('Native TU differs from runtime source27: '+str(source))
 command=common+['-MMD','-MF',str(depout),'-c',str(source),'-o',str(obj)];start=time.monotonic();done=subprocess.run(command,cwd=build,capture_output=True,text=True);Path(str(obj)+'.log').write_text(done.stdout+done.stderr)
 if done.returncode:raise RuntimeError(done.stderr)
 if before!=sha(source):raise RuntimeError('Source changed during compilation')
 old_object=str(dep.relative_to(build))[:-2];replace[old_object]=str(obj)
 manifest['builds'].append({'source':str(source),'source_sha256':before,'matches_source27':matches,'saved_source':str(saved),'command':command,'object_sha256':sha(obj),'wall_seconds':time.monotonic()-start})
 print('Compiled',rel,flush=True)
# Save all project-local dependencies of the privately recompiled TUs.
manifest['recompiled_dependencies']={}
for dep in (target/'objects').rglob('*.o.d'):
 body=dep.read_text().replace('\\\n',' ');tokens=shlex.split(body.partition(':')[2])
 for token in tokens:
  path=Path(token).resolve()
  if path.is_file() and path.is_relative_to(root):
   name=str(path.relative_to(root));manifest['recompiled_dependencies'][name]=sha(path)
   saved=target/'dependency-source'/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,saved)
projected=[replace.get(t,t) for t in link];projected[projected.index('-o')+1]=str(target/'projected/athena');done=subprocess.run(projected,cwd=build,capture_output=True,text=True);(target/'link-projected.log').write_text(done.stdout+done.stderr)
if done.returncode:raise RuntimeError(done.stderr)
manifest['projected_link_command']=projected;manifest['projected_executable_sha256']=sha(target/'projected/athena');manifest['all_native_cartesian_dependents_recompiled']=len(replace)==6
if not manifest['all_native_cartesian_dependents_recompiled']:raise RuntimeError('Expected six native dependent TUs')
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Private native hash',manifest['projected_executable_sha256'],flush=True)
