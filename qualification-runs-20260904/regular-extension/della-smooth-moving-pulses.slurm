#!/bin/bash
#SBATCH --job-name=pcgh_smooth_moving
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/smooth-moving-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source-smooth
python3 - <<'REMOTE'
import subprocess,json,time
from pathlib import Path
assert Path('build-smooth-cuda/binary.sha256').exists(), 'Build must complete before dispatch'
started=time.monotonic()
for path in sorted(Path('../inputs/smooth-controls/moving-pulses').glob('*.athinput')):
 out=Path('../runs/smooth-controls/moving-pulses')/path.stem
 if (out/'completed.json').exists():
  print('ALREADY COMPLETE',out,flush=True)
  continue
 if time.monotonic()-started>2400:
  print('COLLECTION PAUSED before next experiment',path,flush=True)
  raise SystemExit(0)
 argv=['python3','analysis/pc_gh_regular_extension/cuda_driver.py','run',
       '--build','build-smooth-cuda','--input',str(path),'--output',str(out),
       '--wall-segment','00:10:00','--max-segments','1']
 if out.exists(): argv+=['--resume']
 print('START',path,flush=True)
 result=subprocess.run(argv)
 with Path('../runs/smooth-controls/collection-moving.jsonl').open('a') as f:
  f.write(json.dumps(dict(input=str(path),argv=argv,exit_code=result.returncode))+'\n')
 if result.returncode or not (out/'completed.json').exists():
  raise SystemExit(result.returncode or 1)
 subprocess.run(['python3','analysis/pc_gh_regular_extension/verify_smooth_pulses.py',str(out)],check=True)
 print('COMPLETE',path,flush=True)
REMOTE
