"""Immutable new boosted namespace; no remote build or submission.

Reuse the proven v7 worker/controller/storage/verifier, adapting namespace only.
Its reviewed submit helper is the sole submit entry point. Preserve all old roots.
"""
import argparse,hashlib,io,json,pathlib,shutil,subprocess,tarfile,ast
p=argparse.ArgumentParser();p.add_argument('destination');p.add_argument('--source-commit',required=True);a=p.parse_args()
HERE=pathlib.Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
out=pathlib.Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=False)
commit=subprocess.check_output(['git','rev-parse',a.source_commit],cwd=ROOT,text=True).strip()
sha=lambda b:hashlib.sha256(b).hexdigest()
def record(p):return dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
qualified=json.loads((HERE/'sealed-topology-controls/verification.json').read_text())
assert qualified['passed'] and json.loads((HERE/'combined-smr-mpi/verification.json').read_text())['checkpoint_byte_equal']
names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines()
names=[n for n in names if n in ('CMakeLists.txt','config.hpp.in') or n.startswith(('src/','tst/','inputs/hyperbolic_shift_3d/refined_convergence/'))]
files={}
with tarfile.open(out/'source.tar.gz','w:gz') as archive:
 for name in names:
  b=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT);entry=tarfile.TarInfo(name);entry.size=len(b);entry.mode=0o644;archive.addfile(entry,io.BytesIO(b));files[name]=dict(bytes=len(b),sha256=sha(b))
# Source must include exactly the locally qualified evolution and changed headers.
localbuild=json.loads((HERE/'sealed-topology-controls/build.json').read_text())
for name,h in localbuild['headers'].items():assert files[name]['sha256']==h
assert files['tst/hyperbolic_shift_3d/amr_evolution.cpp']['sha256']==next(x['source_sha256'] for x in localbuild['targets'] if x['name']=='evolve')
config='inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial24_angular2_reference.cfg'
b=subprocess.check_output(['git','show',commit+':'+config],cwd=ROOT);assert sha(b)==qualified['config_sha256'];(out/'candidate.cfg').write_bytes(b)
old=ROOT/'output/refined-convergence/aurora-package-v1';previous=json.loads((old/'source-manifest.json').read_text())
shutil.copy2(old/'kokkos.tar.gz',out/'kokkos.tar.gz');assert record(out/'kokkos.tar.gz')==previous['archives']['kokkos.tar.gz']
files.update({n:v for n,v in previous['files'].items() if n.startswith('kokkos/')})
manifest=dict(source_commit=commit,kokkos_commit=previous['kokkos_commit'],files=files,archives={n:record(out/n) for n in ['source.tar.gz','kokkos.tar.gz']})
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
namespace='launcher-boosted-transition-v1';launch=out/namespace;launch.mkdir();adaptations={}
for name in ['control.py','worker.py','storage.py','submit_reviewed.py','verify_checkpoint.cpp']:
 original=(HERE/'launcher-v7'/name).read_text();adapted=original.replace('launcher-v7',namespace)
 (launch/name).write_text(adapted);adaptations[name]=dict(original_sha256=sha(original.encode()),packaged_sha256=sha(adapted.encode()),change='namespace string only' if original!=adapted else 'byte-identical')
shutil.copy2(HERE/'plan_boosted_transition.py',launch/'plan.py')
identity=dict(identity=qualified['full_smr_checkpoint']['identity'],config_sha256=qualified['config_sha256'],source_commit=commit,checkpoint_bytes_n12=qualified['full_smr_checkpoint']['bytes'],local_serial_checkpoint_sha256=qualified['full_smr_checkpoint']['sha256'],local_candidate_binary_sha256=qualified['candidate_binary_sha256'],qualification='Local Serial-MPI full-mesh two-step reference only. Actual SYCL binary hash is produced after the new build; no GPU smoke or long-time claim.')
(launch/'qualified-local-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
(launch/'namespace-provenance.json').write_text(json.dumps(adaptations,indent=2)+'\n')
for name in ['configuration_identity.cpp','resolve_configuration_identity.py']:shutil.copy2(HERE/name,launch/name)
shutil.copy2(HERE/'launcher-v7/identity-calibration.json',launch/'known-aurora-calibration.json')
shutil.copy2(launch/'storage.py',out/'storage.py');shutil.copy2(HERE/'aurora-build/build.sh',out/'build.sh')
s=(HERE/'stationary-reference-v2-build/build.py').read_text().replace('launcher-v6',namespace)
needle=" verifier_cmd=['c++'";assert s.count(needle)==1
s=s.replace(needle," subprocess.run(['/usr/bin/python3.11','"+namespace+"/resolve_configuration_identity.py',str(root)],check=True)\n"+needle)
(out/'build.py').write_text(s)
for f in out.rglob('*.py'):ast.parse(f.read_text(),filename=str(f))
records={str(f.relative_to(out)):record(f) for f in out.rglob('*') if f.is_file()};(out/'package-records.json').write_text(json.dumps(records,indent=2)+'\n')
# Exact manifest membership and contents; reject AppleDouble/unexpected files.
for name,prefix in [('source',''),('kokkos','kokkos/')]:
 expected={n[len(prefix):]:v for n,v in files.items() if (n.startswith(prefix) if prefix else not n.startswith('kokkos/'))}
 with tarfile.open(out/(name+'.tar.gz')) as archive:
  members=[x for x in archive.getmembers() if x.isfile()];assert len(members)==len(expected) and {x.name for x in members}==set(expected)
  for x in members:
   b=archive.extractfile(x).read();assert len(b)==expected[x.name]['bytes'] and sha(b)==expected[x.name]['sha256']
print(json.dumps(dict(destination=str(out),source_commit=commit,source_manifest_sha256=record(out/'source-manifest.json')['sha256'],files=len(files),namespace=namespace,remote_build_started=False,submitted=False,archive_membership_verified=True),indent=2))
