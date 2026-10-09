"""Run one immutable native experiment of the ignored feedback overlay."""
from pathlib import Path
import subprocess
import sys
p=Path(__file__).resolve().parent
repo=p.parents[4]
phase=sys.argv[1]
if phase=='long':
    import json,hashlib
    gate_path=repo/'build-layer-research/detached-wormhole/spatialnorm-gate/receipt.json'
    gate=json.loads(gate_path.read_text());required={'test-kernel_norm-release','test-kernel_norm-debug','norm_leading','norm_second','norm_oracle'}
    results={r['label']:r['returncode'] for r in gate['results']}
    assert all(results.get(name)==0 for name in required),results
    assert gate['sources_unchanged'] and gate['prior_69_files_unchanged']
    for name,h in gate['source_sha256_after'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h,name
    print('Verified five-check independent initial BH gate',hashlib.sha256(gate_path.read_bytes()).hexdigest(),flush=True)
wide=True
basephase=phase
settings={"reference":("reference-long",.05),"short":("control",.02),
          "half":("control",.5),"long":("long",2.)}
suite,duration=settings[basephase]
command=[sys.executable,str(repo/"tst/hyperboloidal/run_layer_validation.py"),
         str(repo/"build-layer-spatial-norm-native/src/athena"),
         str(p/("native-"+phase)),"--suite",suite,
         "--overrides",str(p/"native.athinput"),
         "--duration",str(duration),"--output-cadence",str(min(duration,.025))]
print(" ".join(command),flush=True)
subprocess.run(command,cwd=repo,check=True)
