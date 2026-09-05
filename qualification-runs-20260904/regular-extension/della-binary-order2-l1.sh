set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
python3 analysis/pc_gh_regular_extension/cuda_driver.py run --build build-binary-cuda --input ../inputs/binary-order2/headon-o2-rk4-l1-k1-t100.athinput --output ../runs/binary-order2/headon-o2-rk4-l1-k1-t100 --wall-segment 00:15:00
