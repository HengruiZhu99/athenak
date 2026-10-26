#!/usr/bin/env python3
"""Direct active-field differences between matched-grid t=2 pulse checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def read(path, receipt):
    payload=json.loads(receipt.read_text())['checkpoint_payload']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==payload['sha256']
    assert path.stat().st_size==payload['bytes']
    with path.open('rb') as f:
        assert f.read(16)==b'ATHENA_CSP_CK01!'
        tag=f.read(64).decode();signature=struct.unpack('<17d',f.read(136))
        fingerprint,time,cycle=struct.unpack('<Qdq',f.read(24));packs=[]
        for _ in range(2):
            shape=struct.unpack('<5Q',f.read(40));offset=f.tell()
            packs.append(np.memmap(path,mode='r',dtype='<f8',offset=offset,shape=shape))
            f.seek(int(np.prod(shape))*8,1)
        assert f.tell()+8==path.stat().st_size
    assert time==2 and signature[6]==0 and signature[-1]==8
    return tag,signature,packs,payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints',type=Path);p.add_argument('receipts',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--first',default='degree2');p.add_argument('--second',default='degree4')
    p.add_argument('--second-receipts',type=Path)
    p.add_argument('--variation',choices=['closure','timestep'],default='closure')
    a=p.parse_args();assert a.first!=a.second
    data={a.first:read(a.checkpoints/(a.first+'.bin'),a.receipts/a.first/'receipt.json'),
          a.second:read(a.checkpoints/(a.second+'.bin'),(a.second_receipts or a.receipts)/a.second/'receipt.json')}
    tag,sig,packs,payload=data[a.first];other=data[a.second]
    assert tag==other[0]
    varied={12} if a.variation=='closure' else {13,14}
    assert all(x==y for i,(x,y) in enumerate(zip(sig,other[1])) if i not in varied)
    if a.variation=='closure':assert sig[12]!=other[1][12]
    else:assert all(other[1][i]==.5*sig[i] for i in varied)
    L,R,stretch=sig[:3];results={}
    for region,(first,second) in enumerate(zip(packs,other[2])):
        assert first.shape==second.shape
        u=np.array(first[:,:,4:-4,4:-4,4:-4]);v=np.array(second[:,:,4:-4,4:-4,4:-4])
        assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
        masks={'all':np.ones(u[:,0].shape,dtype=bool)}
        if region:
            na,nr=u.shape[2],u.shape[4]
            vv,uu,ss=np.meshgrid(-1+2*(np.arange(na)+.5)/na,-1+2*(np.arange(na)+.5)/na,
                                (np.arange(nr)+.5)/nr,indexing='ij')
            t=ss if stretch==0 else -np.expm1(-stretch*ss)/(-np.expm1(-stretch))
            radius=(1-t)*L*np.sqrt(1+uu*uu+vv*vv)+t*R
            masks['r<0.5']=np.broadcast_to(radius<.5,u[:,0].shape)
            masks['r>0.9']=np.broadcast_to(radius>.9,u[:,0].shape)
        for label,mask in masks.items():
            values={}
            for field in range(25):
                delta=np.abs(u[:,field]-v[:,field]);location=np.unravel_index(np.argmax(np.where(mask,delta,-1)),delta.shape)
                values[str(field)]={'maximum_absolute_difference':float(delta[location]),
                                   'active_patch_kji':list(map(int,location)),
                                   'first_value':float(u[(location[0],field,*location[1:])]),
                                   'second_value':float(v[(location[0],field,*location[1:])])}
            results[('shell/' if region else 'bulk/')+label]=values
    a.output.write_text(json.dumps({'binary_sha256':tag,'payloads':{n:d[3] for n,d in data.items()},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'variation':a.variation,'first':a.first,'second':a.second,
        'scope':'Active-cell componentwise absolute differences, no ghost cells, no normalization. Same grid and exact t=2; only the declared closure or half-CFL signature entries may differ. This is sensitivity, not spatial convergence.',
        'regions':results},indent=2)+'\n')
    print({r:max(d['maximum_absolute_difference'] for d in fields.values()) for r,fields in results.items()})


if __name__=='__main__':main()
