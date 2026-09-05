set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 - <<'REMOTE'
from pathlib import Path
import subprocess
for n,point in [(128,[-.12109375,.10546875,.00390625]),(256,[-.001953125,-.041015625,-.060546875])]:
  run=Path('../runs/rate16-qualification')/f'pcgh-l16-o6-rk3-core{n}-R8'
  checkpoint=max((run/'rst').glob('*.rst'),key=lambda p:p.stat().st_mtime_ns)
  subprocess.run(['python3','analysis/pc_gh_regular_extension/sample_restart_sources.py',str(checkpoint),'--output',str(run/'source-interface-samples'),'--point',*map(str,point),'--extrema'],check=True)
REMOTE
