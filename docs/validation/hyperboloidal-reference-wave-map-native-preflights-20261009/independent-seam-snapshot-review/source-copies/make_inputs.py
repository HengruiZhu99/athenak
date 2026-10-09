"""Write fixed, held native inputs only; never run AthenaK."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE='''<job>
basename={name}

<mesh>
nx1={n}
nx2={n}
nx3={n}
nghost=3
x1min=-1.1
x1max=1.1
x2min=-1.1
x2max=1.1
x3min=-1.1
x3max=1.1
ix1_bc=outflow
ox1_bc=outflow
ix2_bc=outflow
ox2_bc=outflow
ix3_bc=outflow
ox3_bc=outflow

<meshblock>
nx1={n}
nx2={n}
nx3={n}

<time>
evolution=dynamic
integrator=rk3
cfl_number=.1
nlim=-1
tlim={t}
ndiag=1

<z4c>
hyperboloidal=true
hyperboloidal_layer=true
hyperboloidal_curvature_radius=.5
hyperboloidal_layer_r0=.05
hyperboloidal_layer_r1=.95
hyperboloidal_gauge_r0=.45
hyperboloidal_gauge_r1=.85
hyperboloidal_gauge_q0=.5
hyperboloidal_physical_trace_lapse=true
hyperboloidal_preferred_source=false
hyperboloidal_scri_lapse_damping=2
hyperboloidal_layer_lapse_inner=0
hyperboloidal_layer_lapse_outer=1.5
hyperboloidal_layer_shift_inner=0
hyperboloidal_layer_shift_outer=1
hyperboloidal_slicing=2
hyperboloidal_shift_driver=.1
hyperboloidal_lapse_damping=1.5
hyperboloidal_shift_damping=1
hyperboloidal_kappa1=10
hyperboloidal_dissipation=.1
hyperboloidal_pole_cfl=.03
hyperboloidal_ghost_degree=2
hyperboloidal_symmetric_ghosts=true
hyperboloidal_mass_diagnostics=false
hyperboloidal_mass_nmu=32
floor_chi=false
nrad_wave_extraction=0
spatial_order=4

<problem>
pgen_name=z4c_hyperboloidal
mass=0
lapse_pulse={alpha}
shift_pulse={beta}
pulse_width=.35
pulse_angular=true

<output1>
file_type=bin
variable=z4c
id=z4c
dt={cadence}
ghost_zones=true

<output2>
file_type=bin
variable=adm
id=adm
dt={cadence}
ghost_zones=true

<output3>
file_type=hst
dt={cadence}
data_format=%24.16e

<output4>
file_type=bin
variable=con
id=con
dt={cadence}
ghost_zones=true

<output5>
file_type=rst
dt={cadence}
'''

def main():
    paths=[]; rows=[]
    for n in [16,24,32]:
        rows += [('wave-map',n,'reference',.05,0,0,.005),
                 ('wave-map',n,'large-short',.02,.2,.1,.005),
                 ('wave-map',n,'large',2,.2,.1,.025),
                 ('c0',n,'large',2,.2,.1,.025)]
    rows += [('wave-map',24,'small-short',.02,.02,.01,.005),
             ('wave-map',24,'small',2,.02,.01,.025),
             ('wave-map',24,'reference-long',2,0,0,.025),
             ('wave-map-half',24,'large',2,.2,.1,.025),
             ('c0',24,'reference',.05,0,0,.005)]
    for mode,n,case,t,alpha,beta,cadence in rows:
        name=f'{mode}-N{n}-{case}-t{t:g}'
        p=HERE/'inputs'/(name+'.athinput')
        assert not p.exists(),p
        p.write_text(BASE.format(name=name,n=n,t=t,alpha=alpha,beta=beta,cadence=cadence))
        paths.append({'path':str(p.relative_to(HERE)),
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'mode':mode,'N':n,'case':case,'tlim':t,'lapse_pulse':alpha,
            'shift_pulse':beta,'width':.35,'output_interval':cadence,
            'native_run_authorized':False})
    (HERE/'inputs/index.json').write_text(json.dumps({'scope':'Fixed held inputs only; no native process run',
         'inputs':paths},indent=2,allow_nan=False)+'\n')
    print('PASS held input generation',len(paths))

if __name__=='__main__': main()
