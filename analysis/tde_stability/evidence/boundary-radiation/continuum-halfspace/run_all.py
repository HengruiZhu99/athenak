"""Reproduce the bounded CPU continuum analysis (requires NumPy and SciPy)."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for script in ['determinant.py','schur_check.py','validate_root.py','physical_and_gauge.py','dirichlet_profile.py','boundary_variants.py','weyl_profile.py']:
 with (root/(Path(script).stem+'.reproduce.log')).open('w') as f:
  subprocess.run([sys.executable,str(root/script)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('Reproduced continuum roots and bounded boundary alternatives.')
