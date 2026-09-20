#!/usr/bin/env python3
"""Independent analytic fixture for rank7 trace parsing and field coefficients.

Generates a temporary supported binary checkpoint, avoiding archived raw data.
It is a parser/initial-data test, not an evolution or characteristic-rate test.
"""
import json
import math
from pathlib import Path
import struct
import tempfile
import numpy as np
from checkpoint_face_trace import evaluate

PARAMS = '''<problem>
bh_mass=0
outer_sponge_test_theta_pulse_amplitude={amplitude}
outer_sponge_test_theta_pulse_width=384
<mesh_refinement>
refinement=none
<mesh>
nx1={global_n}
nx2={global_n}
nx3={global_n}
x1min=-2048
x2min=-2048
x3min=-2048
x1max=2048
x2max=2048
x3max=2048
<meshblock>
nx1={block_n}
nx2={block_n}
nx3={block_n}
<mhd>
nscalars=0
<z4c>
residual_gauge=background_adapted
shift_Gamma=1
residual_lapse_f=1
sss_damping_amp=0
lapse_oplog=2
lapse_harmonicf=1
lapse_harmonic=0
<par_end>
'''


def make(path, amplitude, nx):
    n=nx+8
    cells=n**3
    coords=(np.arange(n)-4+.5)*(2048/nx)
    z,y,x=np.meshgrid(coords,coords,coords,indexing='ij')
    state=np.zeros((25,n,n,n),dtype='<f8')
    state[17]=amplitude*np.exp(-(x*x+y*y+z*z)/(2*384**2))
    # Metadata and field layout are independently constructed from restart.cpp:
    # global block count/root level, RegionSize, root/block RegionIndcs,
    # time/dt/cycle, locations/costs, output times, bytes-per-block, payload.
    indcs=[4,nx,nx,nx]+[0]*15
    root_indcs=[4,2*nx,2*nx,2*nx]+[0]*15
    meta=struct.pack('<ii',8,0)+bytes(72)
    meta+=struct.pack('<19i',*root_indcs)+struct.pack('<19i',*indcs)
    meta+=struct.pack('<ddi',0.,3.2,0)
    for rank in range(8):
        meta+=struct.pack('<4i',rank%2,(rank//2)%2,rank//4,0)
    meta+=struct.pack('<8f',*([1.]*8))+bytes(16)
    nmhd=5; faces=3*(n+1)*n*n
    offset=8*(nmhd*cells+faces)
    meta+=struct.pack('<Q',offset+25*cells*8)
    path.write_bytes(PARAMS.format(amplitude=amplitude,global_n=2*nx,block_n=nx).encode()+meta+bytes(offset)+state.tobytes())


with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'analytic.rst'
    for nx,amplitude in ((32,1e-6),(32,1e-7),(16,1e-6),(16,1e-7)):
        make(path,amplitude,nx)
        row=evaluate(path)
        # Hard selected analytic value is not obtained from the implementation.
        expected=(1.027692618952470e-12*(amplitude/1e-6) if nx==32 else
                  amplitude*math.exp(-3944448/(2*384**2)))
        xyz=[2016,32,32] if nx==32 else [1984,64,64]
        direct=amplitude*math.exp(-sum(x*x for x in xyz)/(2*384**2))
        assert math.isclose(direct,expected,rel_tol=2e-15)
        assert row['xyz']==xyz and row['rank']==7
        assert row['global_nx']==2*nx and row['block_nx']==nx
        assert row['seed']['amplitude']==amplitude and row['seed']['width']==384
        assert math.isclose(row['seed']['initial_analytic_Theta_at_point'],expected,rel_tol=2e-15)
        assert math.isclose(row['Theta'],expected,rel_tol=2e-15)
        assert math.isclose(row['C1_in_actual_basis'],expected,rel_tol=2e-15)
        assert math.isclose(row['C2_in_actual_basis'],2*expected/3,rel_tol=2e-15)
        assert math.isclose(row['longitudinal_gauge_in_actual_basis'],4*expected/9,rel_tol=2e-15)
        assert row['C1_terms']['halfchi_Gamma_n']==0
        assert row['C1_terms']['D_n_chi']==0
        assert row['C1_actual_minus_flat']==row['C2_actual_minus_flat']==0
        if nx==32:
            name='face-trace-local-gate.json' if amplitude==1e-6 else 'face-trace-local-amplitude1e7.json'
            archived=json.loads((Path(__file__).parent/name).read_text())['rows'][0]
            assert math.isclose(archived['Theta'],row['Theta'],rel_tol=2e-15)
    path.write_bytes(path.read_bytes().replace(b'bh_mass=0',b'bh_mass=1',1))
    try:
        evaluate(path)
    except AssertionError:
        pass
    else:
        raise AssertionError('Unsupported curved background was accepted')
print('PASS: synthetic rank7 global32/64 Gaussian A=1e-6/1e-7; analytic Theta, C1/C2/gauge coefficients; archived initial data; rejects M!=0')
