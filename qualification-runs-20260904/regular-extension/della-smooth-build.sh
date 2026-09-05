set -e
source /home/hz0693/athenak_env
cd /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source-smooth
# Include the new production header in each experiment's source-diff snapshot.
git add -N src/pc_gh/reduction_profile.hpp
python3 analysis/pc_gh_regular_extension/cuda_driver.py build --kind oracle --build build-smooth-cuda --jobs 4
python3 analysis/pc_gh_regular_extension/cuda_driver.py build --kind binary --build build-smooth-binary-cuda --jobs 4
