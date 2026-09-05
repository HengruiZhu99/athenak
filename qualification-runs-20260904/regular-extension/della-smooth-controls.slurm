#!/bin/bash
#SBATCH --job-name=pcgh_smooth_controls
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/smooth-controls-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source-smooth
python3 - <<'REMOTE'
import subprocess,json,time
from pathlib import Path
assert Path('build-smooth-cuda/binary.sha256').exists(), 'Build must complete before dispatch'
started=time.monotonic()
for group in ['oracle','flat','constant-regression','fixed-pulses','waves','moving-waves']:
 for path in sorted(Path('../inputs/smooth-controls',group).glob('*.athinput')):
  out=Path('../runs/smooth-controls',group,path.stem)
  if (out/'completed.json').exists():
   print('ALREADY COMPLETE',out,flush=True)
   continue
  if time.monotonic()-started>3000:
   print('COLLECTION PAUSED before next experiment',path,flush=True)
   raise SystemExit(0)
  argv=['python3','analysis/pc_gh_regular_extension/cuda_driver.py','run',
        '--build','build-smooth-cuda','--input',str(path),'--output',str(out),
        '--wall-segment','00:05:00','--max-segments','1']
  if out.exists(): argv+=['--resume']
  print('START',path,flush=True)
  result=subprocess.run(argv)
  with Path('../runs/smooth-controls/collection.jsonl').open('a') as f:
   f.write(json.dumps(dict(input=str(path),argv=argv,exit_code=result.returncode))+'\n')
  if result.returncode or not (out/'completed.json').exists():
   raise SystemExit(result.returncode or 1)
  verifier={'oracle':'verify_production_symbol.py','constant-regression':'verify_pulses.py',
            'fixed-pulses':'verify_smooth_pulses.py','moving-waves':'verify_moving_wave.py'}.get(group)
  if verifier:
   subprocess.run(['python3','analysis/pc_gh_regular_extension/'+verifier,str(out)],check=True)
  print('COMPLETE',path,flush=True)
REMOTE
