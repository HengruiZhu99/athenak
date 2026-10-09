"""Run one immutable native experiment of the ignored feedback overlay."""
from pathlib import Path
import subprocess
import sys
p=Path(__file__).resolve().parent
repo=p.parents[3]
phase=sys.argv[1]
wide=phase.startswith("wide-")
basephase=phase[5:] if wide else phase
settings={"reference":("reference-long",.05),"short":("control",.02),
          "half":("control",.5),"long":("long",2.)}
suite,duration=settings[basephase]
command=[sys.executable,str(repo/"tst/hyperboloidal/run_layer_validation.py"),
         str(repo/"build-layer-feedback-native/src/athena"),
         str(p/("native-"+phase)),"--suite",suite,
         "--overrides",str(p/("wide-kappa10.athinput" if wide else "physical-pulse.athinput")),
         "--duration",str(duration),"--output-cadence",str(min(duration,.025))]
print(" ".join(command),flush=True)
subprocess.run(command,cwd=repo,check=True)
