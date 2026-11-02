"""Verify saved local geometry/wiring and rank-count control, without evolving."""
from pathlib import Path
import json, hashlib, struct, time
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'output/restored-six-shells'
DOC=Path(__file__).parent

def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(prefix,steps):
 r=json.loads(prefix.with_suffix('.report.json').read_text())
 assert r['status']=='completed' and r['n']==16 and r['cells']==232928
 assert r['checkpoint_schema']=='HYP3D_MPI_N61' and r['mesh_blocks']==44 and r['shell_layers']==6
 assert r['shell_outer_radii']==[.305,.45,.6,.75,.875,1]
 assert r['broad_smooth_a'] and not r['zero_shift_restoration'] and not r['scri_first6']
 assert r['native_initialization_profile']=='polynomial_c6' and r['native_initialization_start_M']==4 and r['native_initialization_end_M']==12
 assert r['restoring_coefficient']==4 and r['restoring_collar_begin']==.6 and r['restoring_collar_end']==.9
 assert r['bulk_dissipation']=='block_energy_ko8'
 cp=prefix.with_suffix('.checkpoint');b=cp.read_bytes();h=struct.unpack('<16s4i3Q2d',b[:72])
 assert h[:7]==(b'HYP3D_MPI_N61\0\0\0',16,31,2,1,232928,47616)
 assert h[7]==steps and h[8]==steps*.0005 and h[9]==.0005
 assert len(b)==116294312 and struct.unpack('<4i',b[104:120])==(4,2,6,3)
 a=np.frombuffer(b,dtype='<f8',offset=168);assert np.isfinite(a).all()
 return r,b,dict(checkpoint=ident(cp),report=ident(prefix.with_suffix('.report.json')))
initial,b,ir=read(OUT/'initial-check/state',0)
one,b1,r1=read(OUT/'serial-one/state',2)
twelve,b12,r12=read(OUT/'serial-twelve/state',2)
assert one['mpi_size']==1 and twelve['mpi_size']==12 and b1==b12
old=ROOT/'output/restored-wide-smooth/initial-check/state.checkpoint'
v=np.frombuffer(b,dtype='<f8',offset=168,count=8*16**3*31)
vold=np.frombuffer(old.read_bytes(),dtype='<f8',offset=168,count=v.size)
assert np.array_equal(v,vold)
report=dict(verified=True,checked_at=time.time(),initial=ir,one_rank=r1,twelve_ranks=r12,
 exact_one_twelve_checkpoint_parity=True,cartesian_primary_t0_fields_identical_to_NB1=True,
 original_initial_checkpoint=ident(old),initial_cfl=initial['max_cfl'],two_step_cfl=twelve['max_cfl'],
 initial_surface_null=initial['max_null_surface'],initial_ghost_null=initial['max_null_ghost'],
 scope='Local SERIAL backend; not GPU evidence. Fresh six-layer topology and unchanged Cartesian/physical wiring; two-step puncture MPI parity, not long-time stability or convergence.')
(DOC/'checks/local-verification.json').write_text(json.dumps(report,indent=2)+'\n')
for name,r in [('initial',initial),('serial-one',one),('serial-twelve',twelve)]:
 (DOC/'checks'/f'{name}-report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(report,indent=2))
