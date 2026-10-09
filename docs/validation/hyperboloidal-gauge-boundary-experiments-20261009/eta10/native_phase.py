"""Run one immutable native experiment of the ignored feedback overlay."""
from pathlib import Path
import subprocess
import sys
p=Path(__file__).resolve().parent
repo=p.parents[4]
phase=sys.argv[1]
wide=True
basephase=phase
settings={"reference":("reference-long",.05),"short":("control",.02),
          "half":("control",.5),"long":("long",2.)}
suite,duration=settings[basephase]
command=[sys.executable,str(repo/"tst/hyperboloidal/run_layer_validation.py"),
         str(repo/"build-layer-shift-native/src/athena"),
         str(p/("native-"+phase)),"--suite",suite,
         "--overrides",str(p.parent/"wide-kappa10.athinput"),
         "--duration",str(duration),"--output-cadence",str(min(duration,.025))]
print(" ".join(command),flush=True)
subprocess.run(command,cwd=repo,check=True)
