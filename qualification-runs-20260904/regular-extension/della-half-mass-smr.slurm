#!/bin/bash
#SBATCH --job-name=pcgh_m05_smr
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/half-mass-smr-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 - <<'REMOTE'
import subprocess,json,time
from pathlib import Path
paths=[Path('../inputs/half-mass-controls/single')/x for x in ['pcgh-m05-l2-smr-r20.athinput','pcgh-m05-l2-smr-r24.athinput']]
for path in paths:
 out=Path('../runs/half-mass-controls')/path.parent.name/path.stem
 argv=['python3','analysis/pc_gh_regular_extension/cuda_driver.py','run','--build','build-regular-cuda','--input',str(path),'--output',str(out),'--wall-segment','00:15:00']
 print('START',path,flush=True);start=time.monotonic();result=subprocess.run(argv)
 with Path('../runs/half-mass-controls/collection-smr.jsonl').open('a') as f:
  f.write(json.dumps(dict(path=str(path),argv=argv,exit_code=result.returncode,elapsed=time.monotonic()-start))+'\n')
 if result.returncode:
  print('FAILED',path,'preserved; continuing independent controls',flush=True)
  continue
 if path.parent.name in ['pulses','amr']:
  subprocess.run(['python3','analysis/pc_gh_regular_extension/verify_pulses.py',str(out)],check=True)
 print('COMPLETE',path,flush=True)
REMOTE
