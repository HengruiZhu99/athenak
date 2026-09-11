"""Run native VC Cartoon gauge-pulse timestep refinement on an unchanged grid."""
from pathlib import Path
import argparse,json,subprocess,sys,re
import numpy as np
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'vis/python'))
parser=argparse.ArgumentParser();parser.add_argument('exe',type=Path);parser.add_argument('output',type=Path)
a=parser.parse_args();a.exe=a.exe.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
template='''<job>
basename = pulse
<mesh>
nghost = 4
nx1 = 16
x1min = 0
x1max = 4
ix1_bc = axis
ox1_bc = outflow
nx2 = 32
x2min = -4
x2max = 4
ix2_bc = outflow
ox2_bc = outflow
nx3 = 1
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 16
nx2 = 16
nx3 = 1
<mesh_refinement>
refinement = none
<time>
evolution = dynamic
integrator = rk4_classical
cfl_number = {cfl}
nlim = -1
tlim = 0.5
ndiag = 100
<z4c>
grid_centering = vertex
symmetry = cartoon_so2
coordinate_map = half_rho_z_suppressed_y_v2
symmetry_schema = 2
spatial_order = 4
diss = 0.02
floor_chi = false
damp_kappa1 = 0
damp_kappa2 = 0
lapse_oplog = 2
lapse_harmonic = 0
lapse_advect = 0
shift_Gamma = 0
shift_eta = 0
shift_advect = 0
<problem>
pgen_name = z4c_vc_minkowski
lapse_gaussian_amplitude = 0.1
lapse_gaussian_width = 1
<output1>
file_type = tab
variable = z4c
data_format = %24.16e
slice_x2 = 0.5
id = state
dt = 0.5
<output2>
file_type = hst
dcycle = 1
data_format = %24.16e
'''
fields=[];results=[]
for cfl in [.4,.2,.1,.05]:
 case=a.output/str(cfl);case.mkdir();(case/'input').write_text(template.format(cfl=cfl))
 with (case/'stdout').open('w') as out,(case/'stderr').open('w') as err:
  subprocess.run([str(a.exe),'-i','input'],cwd=case,stdout=out,stderr=err,check=True)
 assert 'Terminating on time limit' in (case/'stdout').read_text()
 final=sorted(case.rglob('*.tab'))[-1]
 text=final.read_text()
 time=float(re.search(r'time=(\S+)',text).group(1))
 cycle=int(re.search(r'cycle=(\d+)',text).group(1))
 assert abs(time-.5)<1e-12
 # gid plus three pairs of index/coordinate columns; compare all evolved fields.
 table=np.loadtxt(final)
 header=next(line for line in text.splitlines() if line.startswith("# gid"))
 nfields=sum(name.startswith("z4c_") for name in header.split())
 values=table[:,-nfields:]
 assert np.isfinite(values).all()
 fields.append(values);results.append(dict(cfl=cfl,time=time,cycle=cycle,file=str(final)))
differences=[float(np.sqrt(np.mean((x-y)**2))) for x,y in zip(fields,fields[1:])]
ratios=[x/y for x,y in zip(differences,differences[1:])]
report=dict(runs=results,rms_successive_differences=differences,ratios=ratios,passed=all(12<q<20 for q in ratios),scope='Synchronous native VC Cartoon smooth-pulse radial-slice temporal self-convergence; no AMR or production-gauge qualification')
(a.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit('Temporal convergence gate failed')
