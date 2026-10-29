#!/usr/bin/env python3
"""Bounded V11 continuous-core positive-variable MPI/restart test; no long-time stability claim."""
import argparse, hashlib, json, math, struct, subprocess, time
from array import array
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--control',type=Path,required=True)
p.add_argument('--frozen-control-checkpoint',type=Path,required=True)
p.add_argument('--mpiexec',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
assert not any(out.iterdir()), 'Use fresh evidence directory'
records=[];deadline=time.monotonic()+410

def run(name,ranks=1,end=.0005,restart='-',kind='pulse',candidate=True,success=True,error=None):
    prefix=out/name;n=12 if candidate else 6
    cmd=[str(a.mpiexec.resolve()),'-n',str(ranks),str((a.binary if candidate else a.control).resolve()),str(n),str(prefix),kind,str(end),'.00025','-',str(restart),'120','.0005','.001','6','12','2']
    started=time.monotonic();left=deadline-started
    assert left>0,'Test deadline exceeded'
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=min(150,left))
    (out/(name+'.log')).write_text(r.stdout)
    records.append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.monotonic()-started))
    assert (r.returncode==0)==success,(name,r.stdout)
    if error:assert error in r.stdout,(name,r.stdout)
    if not success:return
    report=json.loads(prefix.with_suffix('.report.json').read_text())
    assert report['status']=='completed' and report['t']==end and report['diagnostics_current']
    assert report['checkpoint_admitted'] and report['mpi_size']==ranks
    assert report['max_null_surface']<5e-13 and report['max_null_ghost']<5e-13
    cp=prefix.with_suffix('.checkpoint');data=cp.read_bytes();header=struct.unpack_from('<16s4i3Q2d',data)
    offset=152 if candidate else 104
    assert header[0].rstrip(b'\0')==(b'HYP3D_MPI_V11' if candidate else b'HYP3D_MPI_V7')
    assert struct.unpack_from('<dii',data,72)==(2.,7,0)
    assert struct.unpack_from('<4i',data,88)==(3,7,4,3)
    if candidate:
        assert header[5]==12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2
        assert struct.unpack_from('<4i',data,104)==(4,2,3,3)
        assert struct.unpack_from('<2d',data,120)==(.18,.16)
        assert struct.unpack_from('<dii',data,136)==(1e-10,1,0)
        assert report['lapse_floor_node_interventions']==0
        assert report['positive_variables']=='log_chi_log_lapse_rk_and_ko'
        assert report['shell_geometry']=='spherical_equiangular'
        assert report['nominal_cells']==26*n**3 and report['evolved_overlap']==3
    assert len(data)==offset+16*(31*header[5]+header[6])
    values=array('d');values.frombytes(data[offset:]);assert all(map(math.isfinite,values))
    if kind=='minkowski':assert report['max_state_deviation']<2e-10
    return cp

one=run('one')
two=run('two',ranks=2)
part=run('part',end=.00025)
restart=run('restart',ranks=2,restart=part)
assert one.read_bytes()==two.read_bytes()==restart.read_bytes(), 'MPI/restart bits differ'
run('minkowski',kind='minkowski',end=.00025)
old=run('v7-control',candidate=False)
assert old.read_bytes()==a.frozen_control_checkpoint.read_bytes(),'V7 changed from frozen checkpoint'
run('v7-rejected',restart=old,success=False,error='checkpoint schema/configuration/admission mismatch')
run('v11-rejected',restart=part,candidate=False,success=False,error='checkpoint schema/configuration/admission mismatch')
for index in range(4):
    data=bytearray(part.read_bytes());struct.pack_into('<i',data,104+4*index,-1)
    bad=out/f'bad-mesh-{index}.checkpoint';bad.write_bytes(data)
    run(f'reject-mesh-{index}',restart=bad,success=False,error='checkpoint fixed-core topology mismatch')
for index in range(2):
    data=bytearray(part.read_bytes());struct.pack_into('<d',data,120+8*index,.17)
    bad=out/f'bad-geometry-{index}.checkpoint';bad.write_bytes(data)
    run(f'reject-geometry-{index}',restart=bad,success=False,error='checkpoint spherical geometry mismatch')
for index,fmt,value in [(136,'<d',0.),(144,'<i',-1),(148,'<i',1)]:
    data=bytearray(part.read_bytes());struct.pack_into(fmt,data,index,value)
    bad=out/f'bad-positive-{index}.checkpoint';bad.write_bytes(data)
    run(f'reject-positive-{index}',restart=bad,success=False,error='checkpoint positive-variable configuration mismatch')
summary=dict(positive_rejection_checks=3,geometry_rejection_checks=2,status='passed',scope='two-step overlap ownership and restart checks only',mpi_partition_bitwise=True,restart_bitwise=True,frozen_v7_bitwise=True,mesh_rejection_checks=4,cross_schema_rejections=2,commands=records)
summary['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['artifacts','commands']}))
