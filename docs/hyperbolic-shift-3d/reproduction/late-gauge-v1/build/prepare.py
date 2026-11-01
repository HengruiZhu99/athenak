from pathlib import Path
import hashlib,json,shutil,sys
root=Path('/home/hzhu/hyperbolic-core-shell-20261028');src=root/'source-late-gauge-v1';launcher=root/'launcher-late-gauge-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
commit='e7c0b8ce2bdd639cfb9532b196fa5badb2b1c3b0'
binary='build-sycl-mpi-late-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_late_gauge'
assert (root/'build-logs/late-gauge-v1/binary.sha256').exists(),'build not complete'
assert sha(root/binary)==(root/'build-logs/late-gauge-v1/binary.sha256').read_text().split()[0]
expected=json.loads((root/'build-logs/late-gauge-v1/expected-source.json').read_text())
for relative,digest in expected.items():
 assert sha(src/relative)==digest,relative
old=json.loads((root/'source-inner-telegrapher-v1-manifest.json').read_text())
count=0
for name,digest in old.items():
 if name.startswith('source-inner-telegrapher-v1/kokkos/'):
  assert sha(root/name.replace('source-inner-telegrapher-v1','source-late-gauge-v1',1))==digest;count+=1
assert count>100
manifest={str(p.relative_to(root)):sha(p) for p in sorted(src.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
mp=root/'source-late-gauge-v1-manifest.json';assert not mp.exists();save(mp,manifest)
launcher.mkdir()
for name in ['control.py','worker.py','storage.py','run_debug.pbs','resolution_campaign.py','late_gauge_campaign.py']:
 shutil.copyfile(src/'tst/hyperbolic_shift_3d/aurora'/name,launcher/name)
science=dict(source_commit=commit,binary=binary,binary_sha256=sha(root/binary),source_manifest=mp.name,source_manifest_sha256=sha(mp))
save(launcher/'late-gauge-science.json',science)
save(launcher/'provenance.json',dict(commit=commit,files={p.name:sha(p) for p in sorted(launcher.iterdir()) if p.is_file()}))
sys.path.insert(0,str(launcher));import late_gauge_campaign as campaign
plans={}
for n in [16]:plans[str(n)]=campaign.make_plan(root,n,1,initial=f'inputs/puncture-continuous-n{n}-smooth-v1.txt')
save(root/'late-gauge-10m-state.json',dict(status='prepared',n24_gate='No N24 before reviewed N16 accuracy success',n=16,target_t_over_M=10,science=science,plans=plans))
print(json.dumps(dict(science=science,plans=plans,kokkos_files_verified=count),indent=2))
