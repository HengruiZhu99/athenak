#!/bin/bash
#SBATCH --job-name=pcgh_ko03_p
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/order6-amr-ko03-p-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 - <<'REMOTE'
import subprocess,json,time
from pathlib import Path
for group in ['fast-shift','moving-shift']:
  for path in sorted(Path('../inputs/order6-amr-ko03',group).glob('p-*.athinput')):
    out=Path('../runs/order6-amr-ko03',group,path.stem)
    argv=['python3','analysis/pc_gh_regular_extension/cuda_driver.py','run','--build','build-regular-cuda','--input',str(path),'--output',str(out),'--wall-segment','00:55:00']
    print('START',group,path.stem,flush=True);start=time.monotonic();result=subprocess.run(argv)
    with Path('../runs/order6-amr-ko03/collection-p.jsonl').open('a') as f:
      f.write(json.dumps(dict(group=group,name=path.stem,argv=argv,exit_code=result.returncode,elapsed=time.monotonic()-start))+'\n')
    if result.returncode: raise SystemExit(result.returncode)
    subprocess.run(['python3','analysis/pc_gh_regular_extension/verify_pulses.py',str(out)],check=True)
    print('COMPLETE',group,path.stem,flush=True)
REMOTE
