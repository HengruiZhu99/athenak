from pathlib import Path
import json,time,hashlib,subprocess,socket,sys
root=Path('/home/hzhu/hyperbolic-core-shell-20261028');sys.path.insert(0,str(root/'launcher-boosted-normalized-shift-v1'))
import control,native_transition_prepare as prep
prep.verify_package(root)
logdir=root/'build-logs/boosted-normalized-shift-v1';result=json.loads((logdir/'result.json').read_text())
assert result['commands'][-1]['returncode']==2 and not result['commands'][-1]['timed_out']
assert not (root/'launcher-boosted-normalized-shift-v1/provenance.json').exists()
q=root/'normalized-shift-transport-metadata-attempt01';q.mkdir()
manifest=json.loads((root/'source-boosted-normalized-shift-v1-manifest.json').read_text())
files=[]
for folder in ['source-boosted-normalized-shift-v1','launcher-boosted-normalized-shift-v1']:
 for p in (root/folder).rglob('._*'):
  rel=p.relative_to(root);assert str(rel) not in manifest and p.is_file() and not p.is_symlink()
  data=p.read_bytes();assert data[:4]==b'\x00\x05\x16\x07',str(rel)
  dest=q/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  files.append(dict(path=str(rel),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));p.rename(dest)
actual={str(p.relative_to(root)) for p in (root/'source-boosted-normalized-shift-v1').rglob('*') if p.is_file()}
assert actual==set(manifest),sorted(actual-set(manifest))
prep.verify_package(root)
(root/'build-sycl-mpi-boosted-normalized-shift-v1').rename(root/'build-sycl-mpi-boosted-normalized-shift-v1-failed-attempt01')
logdir.rename(root/'build-logs/boosted-normalized-shift-v1-failed-attempt01')
receipt=dict(reason='AppleDouble transport metadata were unmanifested extra files globbed by Kokkos. Quarantined without changing any manifest-bound source/launcher bytes. Failed build/logs preserved; no frozen science existed.',files=files,source_manifest_sha256=control.digest(root/'source-boosted-normalized-shift-v1-manifest.json'),time=time.time())
control.atomic(root/'normalized-shift-transport-repair.json',receipt)
state=root/'boosted-normalized-shift-state.json';s=json.loads(state.read_text());s.update(status='build_retry_starting',build_attempt=2,transport_repair='normalized-shift-transport-repair.json',updated=time.time());control.atomic(state,s)
newlog=root/'boosted-normalized-shift-build-attempt02.stdout.log'
with newlog.open('xb') as out:
 p=subprocess.Popen(['/bin/bash','-l',str(root/'launcher-boosted-normalized-shift-v1/build_native_transition.sh'),str(root)],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
s.update(status='build_started',build_pid=p.pid,build_host=socket.gethostname(),build_log=newlog.name,updated=time.time());control.atomic(state,s);print(json.dumps(s,indent=2))
