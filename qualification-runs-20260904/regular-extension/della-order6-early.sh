set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source
for group in flat waves pulses; do
    python3 analysis/pc_gh_regular_extension/run_collection.py --inputs ../inputs/order6-controls/$group --build build-regular-cuda --output ../runs/order6-$group
done
python3 analysis/pc_gh_regular_extension/verify_pulses.py ../runs/order6-pulses/*/
