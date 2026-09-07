"""Read and assemble explicit intrinsic float64 stage payloads."""
import json
import numpy as np


def assemble_ranks(parts,ranks):
    """Join a single synchronized operation, rejecting missing/mixed rank data."""
    assert len(parts)==ranks and {p['header']['rank'] for p in parts}==set(range(ranks))
    first=parts[0]['header']
    for p in parts:
        h=p['header']
        assert set(h)==set(first)
        for key in first:
            if key not in ['rank','blocks','shape']:assert h[key]==first[key],key
        assert h['shape'][1:]==first['shape'][1:]
    parts=sorted(parts,key=lambda p:p['header']['rank'])
    blocks=[b for p in parts for b in p['header']['blocks']]
    gids=[b['gid'] for b in blocks];assert len(set(gids))==len(gids)
    order=np.argsort(gids)
    header=dict(first,rank=-1,blocks=[blocks[i] for i in order],shape=[len(gids),*first['shape'][1:]])
    result=dict(header=header)
    for key in parts[0]:
        if key!='header':result[key]=np.concatenate([p[key] for p in parts],axis=0)[order]
    return result

def read_dump(path):
    with path.open('rb') as f:
        header=json.loads(f.readline());raw=np.frombuffer(f.read(),dtype=np.float64)
    assert header['version']==1 and header['dtype']=='native_float64'
    shape=header['shape'];ks,ke,js,je,iss,ie=header['active_kji']
    active=(shape[0],50,ke-ks+1,je-js+1,ie-iss+1)
    count=int(np.prod(shape));acount=int(np.prod(active))
    assert len(raw)==count+(2*acount if header['rhs_active_only'] else 0)
    state=raw[:count].reshape(shape)
    result=dict(header=header,state=state,active=state[:,:,ks:ke+1,js:je+1,iss:ie+1])
    if header['rhs_active_only']:
        result.update(rhs=raw[count:count+acount].reshape(active),register=raw[count+acount:].reshape(active))
    assert np.isfinite(raw).all()
    return result

def global_fields(data,values):
    blocks=data['header']['blocks'];spacing=np.array(blocks[0]['spacing'])
    origins=np.array([b['origin'] for b in blocks]);minimum=origins.min(axis=0)
    assert all(np.array_equal(b['spacing'],spacing) for b in blocks)
    offsets=np.rint((origins-minimum)/spacing).astype(int)
    local=np.array(values.shape[2:][::-1]);shape=offsets.max(axis=0)+local
    result=np.empty((50,*shape[::-1]));owners=np.zeros(tuple(shape[::-1]),dtype=int)
    for m,(i,j,k) in enumerate(offsets):
        nx,ny,nz=local;result[:,k:k+nz,j:j+ny,i:i+nx]=values[m]
        owners[k:k+nz,j:j+ny,i:i+nx]+=1
    assert np.all(owners==1)
    return result,spacing,offsets

def ghost_error(data):
    g,_,offsets=global_fields(data,data['active']);nz,ny,nx=g.shape[1:]
    ks,ke,js,je,iss,ie=data['header']['active_kji'];error=0.
    for m,(i,j,k) in enumerate(offsets):
        z=(k+np.arange(data['state'].shape[2])-ks)%nz
        y=(j+np.arange(data['state'].shape[3])-js)%ny
        x=(i+np.arange(data['state'].shape[4])-iss)%nx
        expected=g[:,z[:,None,None],y[None,:,None],x[None,None,:]]
        error=max(error,float(np.max(abs(data['state'][m]-expected))))
    return error
