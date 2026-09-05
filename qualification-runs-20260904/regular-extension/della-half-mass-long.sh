set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 - <<'REMOTE'
import subprocess
from pathlib import Path
source=Path('../runs/half-mass-controls/single/pcgh-m05-l2-smr-r16')
assert (source/'completed.json').exists()
parent=max((source/'rst').glob('*.rst'),key=lambda p:p.stat().st_mtime_ns)
subprocess.run(['python3','analysis/pc_gh_regular_extension/cuda_driver.py','run','--build','build-regular-cuda','--input','../inputs/half-mass-long/pcgh-m05-l2-smr-r16-t30.athinput','--output','../runs/half-mass-long/pcgh-m05-l2-smr-r16-t30','--restart-from',str(parent),'--wall-segment','00:15:00'],check=True)
REMOTE
