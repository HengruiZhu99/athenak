import json,pathlib,subprocess,time,hashlib
root=pathlib.Path.cwd();out=root/'output/hyperbolic-core-shell/scri-donor-seven-duration-v1';binary=root/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_scri_radial_duration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[root/'tst/hyperbolic_shift_3d/scri_radial_duration.cpp',*sorted((root/'src/z4c/hyperboloidal').glob('*.hpp')),*sorted((root/'src/mesh/cubed_sphere').glob('*.hpp'))]
manifest={'binary_sha256':sha(binary),'sources':{str(p.relative_to(root)):sha(p) for p in sources},'script_sha256':sha(pathlib.Path(__file__)),'scope':'Bounded local n6 fresh Minkowski control, not Aurora or 100M validation'}
assert not (out/'source.json').exists();(out/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
for variant,name in [(1,'v5'),(2,'donor-seven')]:
 command=[str(binary),str(variant),'6','2',str(out/name)];start=time.monotonic()
 try:
  p=subprocess.run(command,capture_output=True,text=True,timeout=350);result={'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.monotonic()-start}
 except subprocess.TimeoutExpired as ex:result={'command':command,'status':'external_bounded','seconds':time.monotonic()-start,'stdout':str(ex.stdout),'stderr':str(ex.stderr)}
 result['artifacts']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.glob(name+'.*') if p.is_file()}
 (out/(name+'-execution.json')).write_text(json.dumps(result,indent=2)+'\n');print(name,result.get('returncode'),flush=True)
for p in sources:assert sha(p)==manifest['sources'][str(p.relative_to(root))],str(p)
assert sha(binary)==manifest['binary_sha256']
