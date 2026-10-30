from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path('/home/hzhu/hyperbolic-core-shell-20261028');base=root/'campaigns/preflight-thin-probes-v1'
sys.path.insert(0,str(root/'launchers/thin-debug-scaling-v1'));import storage
terminal=json.loads((base/'collected.json').read_text());assert terminal['scheduler']['job_state']=='F' and terminal['scheduler']['Exit_status']==0
assert json.loads((base/'worker-status.json').read_text())['status']=='passed'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary=root/'build-sycl-mpi-thin-probes-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_thin_snapshot_profile'
build=json.loads((root/'build-logs/thin-probes-v1/build-status.json').read_text());assert build['status']=='passed'
assert sha(binary)==build['binaries'][str(binary.relative_to(root))]
assert storage.footprint(root)['bytes']+120000000<5000000000
out=base/'offline-audit';out.mkdir(exist_ok=False);records=[]
for name in ['new-enabled','minkowski-enabled']:
 checkpoint=base/'output'/(name+'.checkpoint');online=base/'output'/(name+'.profiles.csv');snapshot=out/(name+'.snapshot.csv')
 command=[str(binary),str(checkpoint),str(snapshot)];start=time.monotonic()
 result=subprocess.run(command,capture_output=True,text=True,timeout=120)
 (out/(name+'.log')).write_text(result.stdout+result.stderr);assert result.returncode==0
 lines=online.read_bytes().splitlines(keepends=True);expected=lines[0]+b''.join(lines[-1462:])
 records.append(dict(name=name,command=command,returncode=result.returncode,seconds=time.monotonic()-start,
  checkpoint_sha256=sha(checkpoint),online_sha256=sha(online),snapshot_sha256=sha(snapshot),
  online_final_sha256=hashlib.sha256(expected).hexdigest(),byte_equal=snapshot.read_bytes()==expected))
 (out/'results.json').write_text(json.dumps(dict(status='running',binary_sha256=sha(binary),records=records),indent=2)+'\n')
 assert records[-1]['byte_equal'],'Online/final-snapshot bytes differ'
result=dict(status='passed',binary_sha256=sha(binary),records=records,scope='Independent host reconstruction of final samples from completed GPU checkpoints. Exact CSV bytes, no evolution or repair performed.')
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
