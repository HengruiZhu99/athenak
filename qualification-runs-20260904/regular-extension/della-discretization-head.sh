set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 - <<'REMOTE'
import json,subprocess,time
from pathlib import Path
for path in sorted(Path('../inputs/single-discretization').glob('*-smr-*.athinput')):
    name=path.stem;out=Path('../runs/single-discretization')/name
    argv=['python3','analysis/pc_gh_regular_extension/cuda_driver.py','run','--build','build-regular-cuda','--input',str(path),'--output',str(out)]
    print('START',name,flush=True);start=time.monotonic();result=subprocess.run(argv)
    with Path('../runs/single-discretization/head-collection.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(run=name,argv=argv,exit_code=result.returncode,elapsed=time.monotonic()-start))+'\n')
    # Controls are independent; preserve failed runs and continue the next.
    print('END',name,result.returncode,flush=True)
REMOTE
