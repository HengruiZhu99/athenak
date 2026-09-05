set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 - <<'REMOTE'
import subprocess
from pathlib import Path
for rate in [0,1]:
 run=Path('../runs/binary-order2')/f'headon-o2-rk4-l{rate}-k1-t100'
 rst=max((run/'rst').glob('*.rst'),key=lambda p:p.stat().st_mtime_ns)
 subprocess.run(['python3','analysis/pc_gh_regular_extension/sample_restart_sources.py',str(rst),'--output',str(run/'source-near-failure'),'--extrema','--point','2.21875','.03125','.03125','--point','-2.53125','-.03125','-.03125'],check=True)
REMOTE
