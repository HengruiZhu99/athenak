#!/bin/bash
#SBATCH --job-name=pcgh_o6amr_L
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/order6-amr-L-%j.log
set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 - <<'REMOTE'
import subprocess,json,time
from pathlib import Path
for path in sorted(Path('../inputs/order6-controls/amr').glob('L-*.athinput')):
    name=path.stem;out=Path('../runs/order6-amr')/name
    argv=['python3','analysis/pc_gh_regular_extension/cuda_driver.py','run','--build','build-regular-cuda','--input',str(path),'--output',str(out),'--wall-segment','00:55:00']
    print('START',name,flush=True);start=time.monotonic();result=subprocess.run(argv)
    with Path('../runs/order6-amr/L-collection.jsonl').open('a') as f:
        f.write(json.dumps(dict(name=name,argv=argv,exit_code=result.returncode,elapsed=time.monotonic()-start))+'\n')
    if result.returncode: raise SystemExit(result.returncode)
    subprocess.run(['python3','analysis/pc_gh_regular_extension/verify_pulses.py',str(out)],check=True)
    print('COMPLETE',name,flush=True)
REMOTE
