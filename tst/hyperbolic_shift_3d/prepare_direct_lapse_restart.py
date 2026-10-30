#!/usr/bin/env python3
"""Explicitly migrate the verified 3M DT1 state to the AL1 direct-lapse policy."""
from pathlib import Path
import argparse,hashlib,json,struct
SOURCE_SHA='628c2109bdf7ca6624afff58b65e8a28baca724e9e73de17b517454b26fe6a31'
def migrate(source,output):
    data=source.read_bytes();sha=hashlib.sha256(data).hexdigest()
    if sha!=SOURCE_SHA:raise ValueError('Requires the exact verified, admitted 3M checkpoint')
    header=struct.unpack('<16s4i3Q2d',data[:72])
    if header[:5]!=(b'HYP3D_MPI_DT1'.ljust(16,b'\0'),16,31,2,1) or header[-3:]!=(2001,1.,.0005):
        raise ValueError('Unexpected 3M header')
    if len(data)!=152+16*(31*header[5]+header[6]):raise ValueError('Unexpected checkpoint length')
    if output.exists():raise ValueError('Output already exists')
    config=struct.pack('<2d',1e-8,.03)
    migrated=b'HYP3D_MPI_AL1'.ljust(16,b'\0')+data[16:152]+config+data[152:]
    # The physical fields are already stored in physical variables. Only the
    # lapse compensation belongs to the old logarithmic RK variable; reset it
    # explicitly rather than silently treating it as a linear residual.
    migrated=bytearray(migrated)
    struct.pack_into('<i',migrated,144,2)  # PositiveConfig.scheme
    count=31*header[5]
    for cell in range(header[5]):struct.pack_into('<d',migrated,168+8*(count+31*cell+18),0.)
    assert migrated[168:168+8*count]==data[152:152+8*count]
    migrated=bytes(migrated)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:f.write(migrated)
    result=dict(source_sha256=sha,output_sha256=hashlib.sha256(migrated).hexdigest(),chi_floor=1e-8,chi_floor_radius=.03,header_bytes=168,physical_state_bitwise_unchanged=True,lapse_compensation_reset=True,positive_scheme=2,source_schema='HYP3D_MPI_DT1',output_schema='HYP3D_MPI_AL1',t_over_M=3.)
    output.with_suffix(output.suffix+'.migration.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(migrate(a.source,a.output),indent=2))
