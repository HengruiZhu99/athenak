#!/usr/bin/env python3
"""Same-grid coupled control for the declared affine fine discretization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,choices=(6,8,10),default=6);p.add_argument('--affine-shift',type=int,choices=(0,1),default=1);a=p.parse_args()
root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
binary=out/'pilot';shutil.copy2(root/'build-shift-3d/elliptic_shift_3d_coupled_pilot',binary);(out/'.gitignore').write_text('/pilot\n')
initial=root/f'docs/elliptic-shift-3d/evidence/related-scri-family-20261027/n{a.n}/initial.dat'
oldcoords=initial.with_name('coordinates.csv');coordinates=out/'coordinates.csv';radii='.45,.55,.60,.65,.70,.75,1'
subprocess.run([str(binary),'--coordinates',str(a.n),radii,'2','0',str(coordinates)],timeout=30,check=True)
def points(p):
 q=np.genfromtxt(p,delimiter=',',names=True);return np.column_stack([q[x]for x in ('x','y','z','r')])
assert np.array_equal(points(coordinates),points(oldcoords))
data=np.loadtxt(initial,skiprows=1);assert np.array_equal(data[:,0],points(coordinates)[:,3])and np.all(data[:,1:3]>0)
shutil.copy2(initial,out/'initial.dat');initial=out/'initial.dat'
dep=root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/tst/elliptic_shift_3d/coupled.cpp.o.d'
files={Path(x).resolve()for x in dep.read_text().replace('\\\n',' ').split(':',1)[1].split()}|{root/'CMakeLists.txt',Path(__file__).resolve()}
sources={str(f.relative_to(root)):sha(f)for f in files if f.is_file()and root in f.parents}
with tarfile.open(out/'compiled-source.tar.gz','w:gz')as archive:
 for name in sorted(sources):archive.add(root/name,arcname=name)
tag=sha(binary);command=[str(x)for x in [binary,a.n,0,'.0001',2,0,7,'1e-10',out/'coupled.csv',out/'coupled.checkpoint',tag,4,1,4,8,1,initial,2,0,radii,7,1,2,'.24',1,1,1,0,0,'.24',1,1,a.affine_shift]]
report=dict(command=command,workers=1,maximum_seconds=180,active_cell_budget=50000,actual_cells=50*a.n**3,source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources_sha256=sources,binary_sha256=tag,initial_sha256=sha(initial),all_coordinate_tuples_bitwise_equal=True,qualification='Genuine full3D research RK4, optional DECLARED fine discretization; oldFD residual is a separate control')
def save():(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
save();start=time.monotonic()
with (out/'pilot.stdout').open('w')as stdout,(out/'pilot.stderr').open('w')as stderr:
 process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True);report['owned_process_group']=process.pid;save()
 try:report['exit_code']=process.wait(timeout=180)
 except subprocess.TimeoutExpired:
  report['exit_code']=124
  assert os.getpgid(process.pid)==process.pid
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
  report['cleanup']='only owned process group terminated'
report['seconds']=time.monotonic()-start;report['binary_unchanged']=sha(binary)==tag;report['initial_unchanged']=sha(initial)==report['initial_sha256'];report['sources_unchanged']=all(sha(root/n)==v for n,v in sources.items())
rows=[json.loads(x)for x in (out/'pilot.stdout').read_text().splitlines()if x.startswith('{')]
if rows:report['result']=rows[-1]
save();print(json.dumps({k:v for k,v in report.items()if k not in ('sources_sha256','command')},indent=2))
