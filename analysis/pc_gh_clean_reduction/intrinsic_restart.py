"""Audited reader for double-precision uniform intrinsic-only restart fixtures.

Matches restart.cpp's native POD stream: RegionSize=9 doubles,
RegionIndcs=19 int32, LogicalLocation=4 int32, IOWrapperSizeT=uint64.
This rejects other physics and per-rank restart files. Refinement decoding
requires an explicit opt-in; global-array and periodic-wrap helpers stay uniform.
"""
from pathlib import Path
import struct
import numpy as np


def read_restart(path, *, allow_refinement=False):
    raw=Path(path).read_bytes()
    marker=b'<par_end>\n';pos=raw.index(marker)+len(marker)
    header=raw[:pos].decode();blocks={};section=None
    for line in header.splitlines():
        line=line.split('#')[0].strip()
        if line.startswith('<'):
            section=line.strip('<>');blocks.setdefault(section,{})
        elif '=' in line:
            key,value=line.split('=',1);blocks[section][key.strip()]=value.strip()
    pc=blocks['pc_gh']
    assert (pc['formulation'],pc['restart_layout'],int(pc['restart_layout_version']),
            int(pc['restart_layout_fields']))==('intrinsic_clean','intrinsic_pcgh50',1,50)
    assert not any(key in blocks for key in ['hydro','mhd','z4c','radiation','turb_driving'])
    nmb,level=struct.unpack_from('=2i',raw,pos);pos+=8
    domain=np.frombuffer(raw,dtype='=f8',count=9,offset=pos).copy();pos+=72
    mesh=np.frombuffer(raw,dtype='=i4',count=19,offset=pos).copy();pos+=76
    mb=np.frombuffer(raw,dtype='=i4',count=19,offset=pos).copy();pos+=76
    time,dt,cycle=struct.unpack_from('=ddi',raw,pos);pos+=20
    locations=np.frombuffer(raw,dtype='=i4',count=4*nmb,offset=pos).reshape(nmb,4).copy();pos+=16*nmb
    assert np.all(locations[:,3]>=level),'invalid leaf level'
    if not allow_refinement:
        assert np.all(locations[:,3]==level),'not a uniform mesh'
    pos+=4*nmb  # costs
    block_bytes,=struct.unpack_from('=Q',raw,pos);pos+=8
    ng,nx,ny,nz=mb[:4];shape=(nmb,50,nz+2*ng if nz>1 else 1,ny+2*ng if ny>1 else 1,nx+2*ng)
    assert block_bytes==np.prod(shape[1:])*8
    assert len(raw)==pos+nmb*block_bytes,'unsupported extra state, ABI or per-rank layout'
    state=np.frombuffer(raw,dtype='=f8',offset=pos).reshape(shape).copy()
    assert np.isfinite(state).all()
    if not allow_refinement:
        assert nmb==np.prod(mesh[1:4]//mb[1:4])
    for axis in range(1,4):
        if mesh[axis]>1:
            assert all(blocks['mesh'][f'{side}x{axis}_bc']=='periodic' for side in ['i','o']), 'not periodic'
    return dict(root_level=level,header=blocks,locations=locations,state=state,mesh=mesh,mb=mb,domain=domain,time=time,dt=dt,cycle=cycle)


def global_state(data):
    assert np.all(data['locations'][:,3]==data['root_level']), 'global array requires uniform mesh'
    nx,ny,nz=data['mesh'][1:4];mb=data['mb'];ng,bx,by,bz=mb[:4]
    result=np.empty((50,nz,ny,nx));seen=np.zeros((nz,ny,nx),dtype=np.int8)
    for m,(x,y,z,_) in enumerate(data['locations']):
        region=(slice(z*bz,(z+1)*bz),slice(y*by,(y+1)*by),slice(x*bx,(x+1)*bx))
        ks=slice(ng,ng+bz) if bz>1 else slice(0,1)
        js=slice(ng,ng+by) if by>1 else slice(0,1)
        result[(slice(None),*region)]=data['state'][m,:,ks,js,ng:ng+bx]
        seen[region]+=1
    assert np.all(seen==1),'each global cell must occur exactly once'
    return result


def ghost_error(data,reference):
    """Every stored cell, including repeated faces/edges/corners, against global wrap."""
    assert np.all(data['locations'][:,3]==data['root_level']), 'ghost wrap requires uniform mesh'
    nx,ny,nz=data['mesh'][1:4];ng,bx,by,bz=data['mb'][:4];maximum=0.
    for m,(x,y,z,_) in enumerate(data['locations']):
        iz=(z*bz+np.arange(data['state'].shape[2])-(ng if bz>1 else 0))%nz
        iy=(y*by+np.arange(data['state'].shape[3])-(ng if by>1 else 0))%ny
        ix=(x*bx+np.arange(data['state'].shape[4])-ng)%nx
        expected=reference[:,iz[:,None,None],iy[None,:,None],ix[None,None,:]]
        maximum=max(maximum,float(np.max(abs(data['state'][m]-expected)/(1+abs(expected)))))
    return maximum
