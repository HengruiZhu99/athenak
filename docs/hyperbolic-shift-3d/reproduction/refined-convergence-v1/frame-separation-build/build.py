import pathlib,subprocess,hashlib,json,tarfile,os,time
from storage import capacity
root=pathlib.Path(__file__).resolve().parent
os.chdir(root)
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):p=root/p;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
state=dict(status='building',pid=os.getpid(),started=time.time())
with open('build-started.json','x') as f:json.dump(state,f)
try:
 write('storage-before-build.json',capacity(root,planned=2_000_000_000))
 manifest=json.load(open('source-manifest.json'))
 for f,v in manifest['archives'].items():assert digest(f)==v['sha256'] and pathlib.Path(f).stat().st_size==v['bytes']
 pathlib.Path('source').mkdir();pathlib.Path('source/kokkos').mkdir()
 for name,dest in [('source','source'),('kokkos','source/kokkos')]:
  with tarfile.open(name+'.tar.gz') as t:
   for m in t.getmembers():assert not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts and not m.issym() and not m.islnk()
   t.extractall(dest)
 for f,v in manifest['files'].items():assert digest(pathlib.Path('source')/f)==v['sha256']
 commands=[['cmake','-S','source','-B','build','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=icpx','-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int','-DAthena_ENABLE_MPI=ON','-DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON','-DKokkos_ENABLE_SYCL=ON','-DKokkos_ARCH_INTEL_PVC=ON','-DKokkos_ENABLE_SERIAL=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],['cmake','--build','build','-j64','--target','hyperbolic_shift_3d_refined_amr']]
 write('build-commands.json',commands)
 with open('build.log','w') as log:
  for cmd in commands:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   while p.poll() is None:
    try:capacity(root,planned=300_000_000)
    except Exception:p.terminate();p.wait();raise
    time.sleep(10)
   if p.returncode:raise RuntimeError('build failed '+str(cmd)+' return '+str(p.returncode))
 binary='build/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_refined_amr'
 science=dict(source_commit=manifest['source_commit'],source_manifest_sha256=digest('source-manifest.json'),binary=binary,binary_sha256=digest(binary),build_cache_sha256=digest('build/CMakeCache.txt'),flags_sha256=digest('build/tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/flags.make'))
 verifier_cmd=['c++','-O2','-std=c++17','launcher-v3/verify_checkpoint.cpp','-o','launcher-v3/verify-checkpoint']
 subprocess.run(verifier_cmd,check=True);os.chmod('launcher-v3/verify-checkpoint',0o755)
 assert os.access('launcher-v3/verify-checkpoint',os.X_OK)
 write('launcher-v3/verifier-build.json',dict(command=verifier_cmd,sha256=digest('launcher-v3/verify-checkpoint'),mode=oct(pathlib.Path('launcher-v3/verify-checkpoint').stat().st_mode&0o777)))
 launch={str(p.relative_to('launcher-v3')):dict(bytes=p.stat().st_size,sha256=digest(p),mode=oct(p.stat().st_mode&0o777)) for p in sorted(pathlib.Path('launcher-v3').iterdir()) if p.is_file() and p.name!='manifest.json'}
 write('launcher-v3/manifest.json',launch)
 write('science.json',science);state.update(status='completed',finished=time.time(),science=science);write('storage-after-build.json',capacity(root))
except Exception as e:state.update(status='failed',error=str(e),finished=time.time());raise
finally:write('build-result.json',state)
